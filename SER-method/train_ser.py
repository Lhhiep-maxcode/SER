"""Speculative Environment Rollouts (SER) training entrypoint.

This is a research implementation aligned with the SER paper:

- separate math/code environments,
- vLLM/OpenAI-compatible trajectory critic for early accept/reject,
- environment-aware mixed-batch allocation via utility/cost ratios.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import signal
import sys
import time
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
import torch
import yaml
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_utils import load_processed_dataset, render_prompt  # noqa: E402
from reward_utils import RewardStats, compute_reward  # noqa: E402
from budget_allocator import BudgetAllocator  # noqa: E402
from critic_client import CriticClient  # noqa: E402


DEFAULT_CONFIG: dict[str, Any] = {
    "model_dir": "",
    "output_dir": "SER-method/outputs/ser_qwen3_8b",
    "log_file": "SER-method/outputs/ser_qwen3_8b/train.jsonl",
    "resume_from_checkpoint": "",
    "tensorboard_log_dir": "SER-method/outputs/ser_qwen3_8b/tensorboard",
    "use_tensorboard": True,
    "env_data_paths": {
        "math": "SER-method/processed/dapo_taco/math",
        "code": "SER-method/processed/dapo_taco/code",
    },
    "batch_size": None,
    "env_batch_size": {"math": 1, "code": 1},
    "ensure_all_envs_per_step": True,
    "max_env_samples": {"math": None, "code": None},
    "shuffle_dataset": True,
    "num_workers": 2,
    "num_epochs": 1,
    "max_steps": None,
    "accumulation_steps": 8,
    "target_lr": 1e-6,
    "max_grad_norm": 1.0,
    "repeated_generate_nums": 8,
    "grpo_iteration_num": 1,
    "temperature": 1.0,
    "top_p": 0.95,
    "max_length": 2048,
    "max_prompt_length": 1024,
    "max_training_token": 3072,
    "max_training_padding_gap": 256,
    "epsilon": 0.1,
    "beta": 0.01,
    "rollout_chunk_tokens": 256,
    "rollout_generation_batch_size": 8,
    "enable_thinking": True,
    "allow_code_execution": False,
    "code_timeout_seconds": 5.0,
    "use_cache": True,
    "gradient_checkpointing": False,
    "lora_r": 64,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "critic": {
        "enabled": True,
        "base_url": "http://127.0.0.1:8000",
        "model": "Qwen/Qwen3-235B-A22B",
        "api_key": "",
        "timeout_seconds": 30.0,
        "max_retries": 2,
        "max_prompt_chars": 6000,
        "temperature": 0.0,
        "concurrency": 8,
    },
    "thresholds": {
        "math": {"accept": 0.9, "reject": 0.1, "min_tokens": 128, "check_every_tokens": 256},
        "code": {"accept": 0.98, "reject": 0.05, "min_tokens": 256, "check_every_tokens": 256},
    },
    "budget": {
        "ema_alpha": 0.2,
        "utility_floor": 0.01,
        "cost_floor": 1e-3,
        "min_probability": 0.1,
        "utility_mode": "reward",
    },
    "save_steps": 500,
    "seed": 42,
}


@dataclass
class RolloutItem:
    env_name: str
    row_index: int
    repeat_index: int
    row: dict[str, Any]
    prompt_text: str
    prompt_ids: list[int]
    token_ids: list[int]
    generated_tokens: int = 0
    reward: float | None = None
    decision: str = "active"
    critic_score: float | None = None
    critic_calls: int = 0
    verifier_called: bool = False

    @property
    def completion_ids(self) -> list[int]:
        return self.token_ids[len(self.prompt_ids) :]


class CyclingLoader:
    def __init__(self, dataloader: DataLoader) -> None:
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.buffer: list[dict[str, Any]] = []

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)

    def next_rows(self, count: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        while len(rows) < count:
            if not self.buffer:
                self.buffer.extend(self.next()["rows"])
            needed = count - len(rows)
            rows.extend(self.buffer[:needed])
            del self.buffer[:needed]
        return rows


class TrainingState:
    def __init__(self) -> None:
        self.optimizer_steps = 0
        self.accumulated_batches = 0
        self.env_updates: dict[str, int] = {}
        self.start_time = time.time()
        self.last_saved_step = 0


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    setup_output_dirs(args)
    print_config(args)

    writer = build_tensorboard_writer(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = build_model(args)
    print_trainable_parameters(model)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.target_lr, betas=(0.9, 0.95), weight_decay=0.05)
    state = TrainingState()
    state.optimizer_steps = maybe_load_optimizer(args, optimizer)
    reward_stats = RewardStats()

    loaders = build_environment_loaders(args, tokenizer)
    allocator = BudgetAllocator(
        list(loaders),
        ema_alpha=float(args.budget["ema_alpha"]),
        utility_floor=float(args.budget["utility_floor"]),
        cost_floor=float(args.budget["cost_floor"]),
        min_probability=float(args.budget["min_probability"]),
        utility_mode=str(args.budget.get("utility_mode", "reward")),
        seed=args.seed,
    )
    critic = build_critic(args)
    optimizer.zero_grad(set_to_none=True)
    install_signal_handler()

    try:
        with Path(args.log_file).open("a", encoding="utf-8") as log_handle:
            total_iterations = (
                args.max_steps * args.accumulation_steps
                if args.max_steps is not None
                else args.num_epochs * estimate_epoch_iterations(loaders, args.batch_size)
            )
            progress = tqdm(range(total_iterations), desc="SER training")
            for iteration in progress:

                # === Environment-aware budget allocation ===
                allocation = allocator.allocation(
                    args.batch_size,
                    ensure_each=bool(args.ensure_all_envs_per_step),
                )

                rows_by_env = {
                    env_name: loaders[env_name].next_rows(sample_count)
                    for env_name, sample_count in allocation.items()
                    if sample_count > 0
                }   # {'code': [{...}, {...}, ...], 'math': [...]}
                # ===========================================

                print("Starting to generate and score trajectories ...")
                rollout_start = time.time()
                env_rollout_batches = collect_mixed_ser_rollouts(
                    model,
                    tokenizer,
                    rows_by_env,
                    args,
                    critic,
                    reward_stats,
                )
                # env_rollout_batches = {
                #   'math': {'messages': [...], 'rewards': [...], 'advantages': [...], ...}, 
                #   'code': {...},
                #   ...
                # }

                rollout_seconds = time.time() - rollout_start
                print("===> Time passed by:", round(rollout_seconds, 2), "seconds")

                env_seconds_by_name = attribute_env_seconds(env_rollout_batches, rollout_seconds, equal=True)

                for env_name, rollout_batch_for_env in env_rollout_batches.items():
                    env_seconds = env_seconds_by_name.get(env_name, 0.0)
                    rollout_rewards = rollout_batch_for_env["rewards"]
                    env_reward = mean(rollout_rewards) if rollout_rewards else 0.0
                    allocator.update(env_name, reward=float(env_reward), cost_seconds=env_seconds)
                    state.env_updates[env_name] = state.env_updates.get(env_name, 0) + 1

                rollout_batch = merge_rollout_batches(env_rollout_batches)
                did_backward = bool(rollout_batch["messages"])
                if did_backward:
                    loss_logs = train_on_batch(
                        model,
                        tokenizer,
                        rollout_batch,
                        optimizer,
                        args,
                        state,
                        env_weight=1.0,
                    )
                else:
                    loss_logs = {"loss": 0.0, "kl": 0.0, "num_train_sequences": 0.0}
                if did_backward and state.accumulated_batches % args.accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    state.optimizer_steps += 1

                log_record = build_log_record(
                    args,
                    state,
                    env_name="mixed",
                    env_weight=1.0,
                    rollout_batch=rollout_batch,
                    loss_logs=loss_logs,
                    env_seconds=sum(env_seconds_by_name.values()),
                    reward_stats=reward_stats,
                    allocator=allocator,
                    iteration=iteration,
                    allocation=allocation,
                    env_rollout_batches=env_rollout_batches,
                    env_seconds_by_name=env_seconds_by_name,
                )
                log_handle.write(json.dumps(log_record) + "\n")
                log_handle.flush()
                write_tensorboard_scalars(writer, log_record, max(1, state.accumulated_batches))
                progress.set_postfix(
                    alloc=",".join(f"{name}:{count}" for name, count in allocation.items()),
                    step=state.optimizer_steps,
                    reward=round(float(log_record["mean_reward"]), 3),
                    accept=rollout_batch["early_accepts"],
                    reject=rollout_batch["early_rejects"],
                )

                if (
                    args.save_steps > 0
                    and state.optimizer_steps > 0
                    and state.optimizer_steps % args.save_steps == 0
                    and state.optimizer_steps != state.last_saved_step
                ):
                    save_checkpoint(model, tokenizer, optimizer, args, state.optimizer_steps)
                    state.last_saved_step = state.optimizer_steps
                if args.max_steps is not None and state.optimizer_steps >= args.max_steps:
                    save_checkpoint(model, tokenizer, optimizer, args, state.optimizer_steps)
                    return

        if state.accumulated_batches % args.accumulation_steps != 0:
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            state.optimizer_steps += 1
        save_checkpoint(model, tokenizer, optimizer, args, state.optimizer_steps)
    finally:
        if writer is not None:
            writer.close()


def build_environment_loaders(args, tokenizer) -> dict[str, CyclingLoader]:
    loaders = {}
    for env_name, path in args.env_data_paths.items():
        dataset = load_processed_dataset(path, max_samples=args.max_env_samples.get(env_name))
        dataloader = DataLoader(
            dataset,
            batch_size=int(args.env_batch_size.get(env_name, 1)),
            shuffle=args.shuffle_dataset,
            num_workers=args.num_workers,
            collate_fn=collate_rows,
            drop_last=False,
        )
        loaders[env_name] = CyclingLoader(dataloader)
    return loaders


def estimate_epoch_iterations(loaders: dict[str, CyclingLoader], batch_size: int) -> int:
    total_rows = sum(len(loader.dataloader.dataset) for loader in loaders.values())
    return max(1, math.ceil(total_rows / max(1, int(batch_size))))


def rollout_generation_batch_size(args) -> int:
    configured = args.rollout_generation_batch_size
    return max(1, int(configured))


def critic_concurrency(args) -> int:
    return max(1, int(args.critic.get("concurrency", 1)))


def collect_mixed_ser_rollouts(
    model,
    tokenizer,
    rows_by_env: dict[str, list[dict[str, Any]]],
    args,
    critic: CriticClient,
    reward_stats: RewardStats,
) -> dict[str, dict[str, Any]]:
    items: list[RolloutItem] = []
    for env_name, rows in rows_by_env.items():
        # Init group rollout for each sample in all env
        items.extend(initialize_rollout_items(tokenizer, rows, args, env_name=env_name))
        # items = [RolloutItem(...), RolloutItem(...), ...] for all envs

    # active stores indices of items need more generation.
    active = list(range(len(items)))
    generated_token_count_by_env = {env_name: 0 for env_name in rows_by_env}
    critic_errors_by_env = {env_name: 0 for env_name in rows_by_env}
    generation_batch_size = rollout_generation_batch_size(args)
    critic_max_concurrency = critic_concurrency(args)

    while active:
        # Rollouts that are not finished in this pass are carried into the
        # next chunk-generation round.
        next_active = []
        pending_critic: list[dict[str, Any]] = []
        was_training = model.training
        model.eval()
        try:
            for start in range(0, len(active), generation_batch_size):
                batch_indices = active[start : start + generation_batch_size]

                # Drop trajectories that already reached the hard token limit.
                remaining = [args.max_length - len(items[idx].token_ids) for idx in batch_indices]
                batch_indices = [idx for idx, rem in zip(batch_indices, remaining) if rem > 0]
                if not batch_indices:
                    continue

                max_new_tokens = min(
                    args.rollout_chunk_tokens,
                    min(args.max_length - len(items[idx].token_ids) for idx in batch_indices),
                )

                # Consider using EAGLE speculative here
                generated = generate_token_chunk(
                    model,
                    tokenizer,
                    [items[idx].token_ids for idx in batch_indices],
                    max_new_tokens,
                    args,
                )

                for idx, token_ids in zip(batch_indices, generated):
                    item = items[idx]
                    previous_len = len(item.token_ids)
                    item.token_ids = token_ids  # assign new token_ids with newly generated tokens
                    item.generated_tokens = max(0, len(item.token_ids) - len(item.prompt_ids))
                    generated_token_count_by_env[item.env_name] = (
                        generated_token_count_by_env.get(item.env_name, 0)
                        + max(0, len(item.token_ids) - previous_len)
                    )

                    completion = tokenizer.decode(item.completion_ids, skip_special_tokens=True)
                    finished = has_eos(item.completion_ids, tokenizer.eos_token_id) or len(item.token_ids) >= args.max_length
                    if finished:
                        # Full rollout path: once generation naturally stops or
                        # hits the limit, call the real environment verifier.
                        verify_rollout(item, completion, args, reward_stats)
                        continue

                    thresholds = args.thresholds[item.env_name]
                    if should_query_critic(item, thresholds):   # check if the trajectory have enough token and divisible by check_every_tokens
                        # Speculative path: ask the critic whether this partial
                        # trajectory is already clearly good or clearly bad.
                        item.critic_calls += 1
                        pending_critic.append(
                            {
                                "idx": idx,
                                "thresholds": thresholds,
                                "request": {
                                    "task": str(item.row.get("task") or item.env_name),
                                    "prompt_text": item.prompt_text,
                                    "partial_completion": completion,
                                    "env_name": item.env_name,
                                },
                            }
                        )
                        continue

                    # If not ready to query the critic, keep this trajectory active for the next chunk.
                    next_active.append(idx)
        finally:
            if was_training:
                model.train()

        if pending_critic:
            results = critic.score(
                [entry["request"] for entry in pending_critic],
                max_concurrency=critic_max_concurrency,
            )
            for entry, result in zip(pending_critic, results):
                idx = int(entry["idx"])
                item = items[idx]
                thresholds = entry["thresholds"]
                item.critic_score = result.score
                critic_errors_by_env[item.env_name] = (
                    critic_errors_by_env.get(item.env_name, 0) + int(bool(result.error))
                )
                if result.score >= float(thresholds["accept"]):
                    # Early accept saves the remaining rollout and verifier
                    # cost, but gives reward 1 immediately.
                    item.reward = 1.0
                    item.decision = "early_accept"
                    continue
                elif result.score <= float(thresholds["reject"]):
                    # Early reject also stops generation immediately, assigning
                    # reward 0 without full verification.
                    item.reward = 0.0
                    item.decision = "early_reject"
                    continue
                else:
                    next_active.append(idx)

        active = next_active

    env_batches: dict[str, dict[str, Any]] = {}
    for env_name in rows_by_env:
        # env_items = [RolloutItem(env_name,...), RolloutItem(env_name, ...), ...]
        env_items = [item for item in items if item.env_name == env_name]

        # Convert completed rollout items into trainable GRPO messages, rewards,
        # group-normalized advantages, and logging statistics.
        env_batches[env_name] = build_training_batch_from_rollouts(
            env_items,
            tokenizer,
            env_name,
            generated_token_count_by_env.get(env_name, 0),
            critic_errors_by_env.get(env_name, 0),
            max_length=args.max_length,
        )
    # {
    #   'math': {'messages': [...], 'rewards': [...], 'advantages': [...], ...}, 
    #   'code': {...}
    # }
    return env_batches


def initialize_rollout_items(tokenizer, rows: list[dict[str, Any]], args, *, env_name: str) -> list[RolloutItem]:
    items = []
    for row_index, row in enumerate(rows):
        # apply chat template
        prompt_text = render_prompt(tokenizer, row["prompt"], enable_thinking=args.enable_thinking)
        # encode to ids
        prompt_ids = tokenizer.encode(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_prompt_length,
        )
        for repeat_index in range(args.repeated_generate_nums):
            items.append(
                RolloutItem(
                    env_name=env_name,
                    row_index=row_index,
                    repeat_index=repeat_index,
                    row=row,
                    prompt_text=prompt_text,
                    prompt_ids=prompt_ids,
                    token_ids=list(prompt_ids),
                )
            )
    return items


def generate_token_chunk(model, tokenizer, token_lists: list[list[int]], max_new_tokens: int, args) -> list[list[int]]:
    pad_id = tokenizer.pad_token_id
    max_len = max(len(tokens) for tokens in token_lists)
    input_ids = []
    attention_mask = []
    pad_lengths = []
    for tokens in token_lists:
        pad_len = max_len - len(tokens)
        pad_lengths.append(pad_len)
        input_ids.append([pad_id] * pad_len + tokens)
        attention_mask.append([0] * pad_len + [1] * len(tokens))

    input_tensor = torch.tensor(input_ids, device=model_device(model), dtype=torch.long)
    mask_tensor = torch.tensor(attention_mask, device=model_device(model), dtype=torch.long)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_tensor,
            attention_mask=mask_tensor,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    result = []
    for output, pad_len in zip(outputs, pad_lengths):
        ids = [int(token) for token in output.tolist()[pad_len:]]
        result.append(ids)
    del input_tensor, mask_tensor, outputs
    return result


def should_query_critic(item: RolloutItem, thresholds: dict[str, Any]) -> bool:
    min_tokens = int(thresholds.get("min_tokens", 0))
    check_every = max(1, int(thresholds.get("check_every_tokens", 1)))
    if item.generated_tokens < min_tokens:
        return False
    return item.generated_tokens % check_every == 0


def verify_rollout(item: RolloutItem, completion: str, args, reward_stats: RewardStats) -> None:
    item.reward = compute_reward(
        completion,
        item.row,
        allow_code_execution=args.allow_code_execution,
        code_timeout_seconds=args.code_timeout_seconds,
        stats=reward_stats,
    )
    item.verifier_called = True
    item.decision = "verified"


def build_training_batch_from_rollouts(
    items: list[RolloutItem],
    tokenizer,
    env_name: str,
    generated_token_count: int,
    critic_errors: int,
    max_length: int,
) -> dict[str, Any]:
    by_row: dict[int, list[RolloutItem]] = {}
    for item in items:
        by_row.setdefault(item.row_index, []).append(item)
    # by_row = {0: [RolloutItem(row_0, repeat_0), RolloutItem(row_0, repeat_1), ...], 1: [RolloutItem(row_1, repeat_0), ...], ...}

    messages = []
    raw_rewards = []
    advantages = []
    skipped_zero_std = 0
    skipped_correct = 0
    skipped_incorrect = 0

    for row_items in by_row.values():
        rewards = np.array([float(item.reward or 0.0) for item in row_items], dtype=np.float32)
        if float(rewards.std()) == 0.0:
            skipped_zero_std += 1
            if float(rewards[0]) >= 1.0:
                skipped_correct += 1
            else:
                skipped_incorrect += 1
            continue
        group_advantages = ((rewards - rewards.mean()) / rewards.std()).tolist()
        for item, reward, advantage in zip(row_items, rewards.tolist(), group_advantages):
            completion = tokenizer.decode(item.completion_ids, skip_special_tokens=True)
            message = deepcopy(item.row["prompt"])
            message.append({"role": "assistant", "content": completion})
            messages.append(message)
            raw_rewards.append(float(reward))
            advantages.append(float(advantage))
    
    # After this loop, we have:
    # messages = [{'role': ..., 'content': ...}, {'role': ..., 'content': ...}, ...]
    # rewards = [0.0, 1.0, 0.0, ...]
    # advantages = [-0.5, 0.5, -0.5, ...]

    lengths = [item.generated_tokens for item in items]
    early_accepts = sum(1 for item in items if item.decision == "early_accept")
    early_rejects = sum(1 for item in items if item.decision == "early_reject")
    verified = sum(1 for item in items if item.verifier_called)
    total = max(1, len(items))
    max_possible_new_tokens = [
        max(1, max_length - len(item.prompt_ids))
        for item in items
    ]
    rollout_fractions = [
        item.generated_tokens / max_tokens
        for item, max_tokens in zip(items, max_possible_new_tokens)
    ]
    return {
        "messages": messages,
        "rewards": raw_rewards,
        "advantages": advantages,
        "generated_lengths": lengths,
        "env_name": env_name,
        "generated_tokens": generated_token_count,
        "early_accepts": early_accepts,
        "early_rejects": early_rejects,
        "verified": verified,
        "verification_fraction": verified / total,
        "rollout_fraction": float(mean(rollout_fractions)) if rollout_fractions else 0.0,
        "critic_calls": sum(item.critic_calls for item in items),
        "critic_errors": critic_errors,
        "skipped_zero_std": skipped_zero_std,
        "skipped_correct": skipped_correct,
        "skipped_incorrect": skipped_incorrect,
    }


def merge_rollout_batches(env_rollout_batches: dict[str, dict[str, Any]]) -> dict[str, Any]:
    merged = {
        "messages": [],
        "rewards": [],
        "advantages": [],
        "generated_lengths": [],
        "env_name": "mixed",
        "generated_tokens": 0,
        "early_accepts": 0,
        "early_rejects": 0,
        "verified": 0,
        "verification_fraction": 0.0,
        "rollout_fraction": 0.0,    # average generated tokens / max_possible_tokens across all rollouts
        "critic_calls": 0,
        "critic_errors": 0,
        "skipped_zero_std": 0,
        "skipped_correct": 0,
        "skipped_incorrect": 0,
    }
    total_rollouts = 0
    rollout_fraction_sum = 0.0

    for batch in env_rollout_batches.values():
        merged["messages"].extend(batch["messages"])
        merged["rewards"].extend(batch["rewards"])
        merged["advantages"].extend(batch["advantages"])
        merged["generated_lengths"].extend(batch["generated_lengths"])
        for key in (
            "generated_tokens",
            "early_accepts",
            "early_rejects",
            "verified",
            "critic_calls",
            "critic_errors",
            "skipped_zero_std",
            "skipped_correct",
            "skipped_incorrect",
        ):
            merged[key] += batch[key]

        env_rollouts = len(batch["generated_lengths"])
        total_rollouts += env_rollouts
        rollout_fraction_sum += float(batch["rollout_fraction"]) * env_rollouts

    if total_rollouts > 0:
        merged["verification_fraction"] = float(merged["verified"]) / total_rollouts
        merged["rollout_fraction"] = rollout_fraction_sum / total_rollouts
    return merged


def attribute_env_seconds(env_rollout_batches: dict[str, dict[str, Any]], total_seconds: float, equal: bool) -> dict[str, float]:
    if not env_rollout_batches:
        return {}

    if not equal:
        token_counts = {
            env_name: float(max(0, batch.get("generated_tokens", 0)))
            for env_name, batch in env_rollout_batches.items()
        }
        total_tokens = sum(token_counts.values())
        if total_tokens > 0:
            return {
                env_name: float(total_seconds) * token_count / total_tokens
                for env_name, token_count in token_counts.items()
            }

    equal_share = float(total_seconds) / len(env_rollout_batches)
    return {env_name: equal_share for env_name in env_rollout_batches}


def train_on_batch(model, tokenizer, train_batch: dict[str, Any], optimizer, args, state: TrainingState, *, env_weight: float):
    input_ids, attention_mask, loss_mask = encode_messages_for_loss(tokenizer, train_batch["messages"], args.enable_thinking)
    sorted_items = sorted(
        zip(input_ids, attention_mask, loss_mask, train_batch["advantages"]),
        key=lambda item: len(item[0]),
    )
    chunks = make_training_chunks(sorted_items, args.max_training_token, args.max_training_padding_gap)
    # chunks = [chunk_1, chunk_2, ...], where each chunk is a list of tuples (input_ids, attention_mask, loss_mask, advantage)

    total_loss = 0.0
    total_kl = 0.0
    old_logps_by_chunk = [None for _ in chunks]
    ref_logps_by_chunk = [None for _ in chunks]
    for _ in range(args.grpo_iteration_num):
        for chunk_idx, chunk in enumerate(chunks):
            tensors = pad_chunk(chunk, tokenizer.pad_token_id, model_device(model))
            labels = tensors["input_ids"]
            mask = tensors["loss_mask"]
            reward = tensors["advantages"].unsqueeze(-1)

            if args.beta > 0:
                ref_logits = ref_logps_by_chunk[chunk_idx]
                if ref_logits is None:
                    with adapters_disabled(model), torch.no_grad():
                        ref_logits = model(input_ids=tensors["input_ids"], attention_mask=tensors["attention_mask"]).logits
                    ref_logps_by_chunk[chunk_idx] = ref_logits
                ref_logps = gather_token_logps(ref_logits, labels).detach()
                del ref_logits
            else:
                ref_logps = None

            outputs = model(input_ids=tensors["input_ids"], attention_mask=tensors["attention_mask"])
            logps = gather_token_logps(outputs.logits, labels)
            old_logps = old_logps_by_chunk[chunk_idx]
            if old_logps is None:
                old_logps = logps.detach()
            
            old_logps_by_chunk[chunk_idx] = logps.detach()

            loss, kl = compute_grpo_loss(
                logps=logps,
                old_logps=old_logps,
                ref_logps=ref_logps,
                mask=mask[:, :-1],
                reward=reward,
                epsilon=args.epsilon,
                beta=args.beta,
            )   # loss as sum of sequence in one chunk (group or anything)
            scaled_loss = (
                loss
                * float(env_weight)
                / max(1, len(train_batch["messages"]))
                / max(1, args.accumulation_steps)
            )
            scaled_loss.backward()
            total_loss += float(loss.detach().item())
            total_kl += float(kl.detach().item())
            del tensors, labels, mask, reward, ref_logps, outputs, logps, old_logps, loss, kl, scaled_loss

    state.accumulated_batches += 1
    denom = max(1, len(chunks) * args.grpo_iteration_num)
    return {"loss": total_loss / denom, "kl": total_kl / denom, "num_train_sequences": float(len(train_batch["messages"]))}

def as_token_ids(value):
    if isinstance(value, dict):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return list(value)

def encode_messages_for_loss(tokenizer, messages: list[list[dict[str, str]]], enable_thinking: bool):
    input_ids = [
        as_token_ids(
            tokenizer.apply_chat_template(
                message,
                tokenize=True,
                add_generation_prompt=False,
            )
        )
        for message in messages
    ]
    attention_mask = [[1] * len(ids) for ids in input_ids]
    loss_mask = []
    prompt_ids = []
    for message in messages:
        try:
            ids = as_token_ids(
                tokenizer.apply_chat_template(
                    message[:-1],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            )
        except TypeError:
            ids = as_token_ids(
                tokenizer.apply_chat_template(
                    message[:-1],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
        prompt_ids.append(ids)

    for prompt_id, full_ids in zip(prompt_ids, input_ids):
        prompt_len = len(prompt_id)
        cur_mask = [0] * max(0, prompt_len - 1) + [1] * max(0, len(full_ids) - prompt_len + 1)
        cur_mask = cur_mask[: len(full_ids)]
        if len(cur_mask) < len(full_ids):
            cur_mask += [0] * (len(full_ids) - len(cur_mask))
        loss_mask.append(cur_mask)
    return input_ids, attention_mask, loss_mask


def make_training_chunks(items, max_training_token: int, max_padding_gap: int):
    chunks = []
    current = []
    current_max_len = 0
    current_token_count = 0
    for item in items:
        seq_len = len(item[0])
        current_token_count += seq_len
        current_max_len = max(current_max_len, seq_len)
        can_add = not current or (
            current_max_len * (len(current) + 1) <= max_training_token and
            (current_max_len * (len(current) + 1) - current_token_count) <= max_padding_gap
        )
        if not can_add:
            chunks.append(current)
            current = []
            current_max_len = seq_len
            current_token_count = seq_len
        current.append(item)
    if current:
        chunks.append(current)
    return chunks


def pad_chunk(chunk, pad_token_id: int, device):
    max_len = max(len(item[0]) for item in chunk)
    input_ids, attention_mask, loss_mask, advantages = [], [], [], []
    for ids, mask, lmask, advantage in chunk:
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad_len)
        attention_mask.append(mask + [0] * pad_len)
        loss_mask.append(lmask + [0] * pad_len)
        advantages.append(float(advantage))
    return {
        "input_ids": torch.tensor(input_ids, device=device, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, device=device, dtype=torch.long),
        "loss_mask": torch.tensor(loss_mask, device=device, dtype=torch.float32),
        "advantages": torch.tensor(advantages, device=device, dtype=torch.float32),
    }


def gather_token_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = logits[:, :-1, :].float()  # shape: (batch_size, seq_len - 1, vocab_size)
    labels = labels[:, 1:].to(logits.device)    # shape: (batch_size, seq_len - 1)
    return torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(-1)).squeeze(-1)


def compute_grpo_loss(*, logps, old_logps, ref_logps, mask, reward, epsilon: float, beta: float):
    ratio = torch.exp(logps - old_logps)
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    policy_term = torch.minimum(ratio * reward, clipped_ratio * reward)
    if beta > 0 and ref_logps is not None:
        diff = ref_logps - logps
        kl = torch.exp(diff) - diff - 1.0
    else:
        kl = torch.zeros_like(policy_term)
    token_loss = -(policy_term - beta * kl) * mask
    denom = mask.sum(dim=-1).clamp_min(1.0)
    sequence_loss = token_loss.sum(dim=-1) / denom
    sequence_kl = (kl * mask).sum(dim=-1) / denom
    return sequence_loss.sum(), sequence_kl.mean()


def build_model(args):
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype="auto",
        trust_remote_code=True,
    ).cuda()
    base_model.config.use_cache = bool(args.use_cache)
    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
    else:
        base_model.gradient_checkpointing_disable()

    if args.resume_from_checkpoint:
        model = PeftModel.from_pretrained(base_model, args.resume_from_checkpoint, is_trainable=True)
    else:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=parse_list_arg(args.lora_target_modules),
        )
        model = get_peft_model(base_model, lora_config)
    return model.cuda()


def build_critic(args) -> CriticClient:
    cfg = args.critic
    return CriticClient(
        base_url=str(cfg.get("base_url", "")),
        model=str(cfg.get("model", "")),
        api_key=str(cfg.get("api_key", "")),
        timeout_seconds=float(cfg.get("timeout_seconds", 30.0)),
        max_retries=int(cfg.get("max_retries", 2)),
        max_prompt_chars=int(cfg.get("max_prompt_chars", 6000)),
        max_new_tokens=int(cfg.get("max_new_tokens", 512)),
        temperature=float(cfg.get("temperature", 0.0)),
        enabled=bool(cfg.get("enabled", True)),
    )


def build_log_record(
    args,
    state,
    *,
    env_name,
    env_weight,
    rollout_batch,
    loss_logs,
    env_seconds,
    reward_stats,
    allocator,
    iteration,
    allocation: dict[str, int] | None = None,
    env_rollout_batches: dict[str, dict[str, Any]] | None = None,
    env_seconds_by_name: dict[str, float] | None = None,
):
    lengths = rollout_batch["completion_lengths"]
    record = {
        "iteration": iteration,
        "optimizer_step": state.optimizer_steps,
        "accumulated_batches": state.accumulated_batches,
        "env": env_name,
        "env_weight": float(env_weight),
        "env_seconds": float(env_seconds),
        "loss": loss_logs["loss"],
        "kl": loss_logs["kl"],
        "num_train_sequences": loss_logs["num_train_sequences"],
        "mean_reward": float(mean(rollout_batch["rewards"])) if rollout_batch["rewards"] else 0.0,
        "generated_tokens": float(rollout_batch["generated_tokens"]),
        "mean_completion_length": float(mean(lengths)) if lengths else 0.0,
        "max_completion_length": float(max(lengths)) if lengths else 0.0,
        "length_stdev": float(stdev(lengths)) if len(lengths) > 1 else 0.0,
        "early_accepts": float(rollout_batch["early_accepts"]),
        "early_rejects": float(rollout_batch["early_rejects"]),
        "verified": float(rollout_batch["verified"]),
        "verification_fraction": float(rollout_batch["verification_fraction"]),
        "rollout_fraction": float(rollout_batch["rollout_fraction"]),
        "critic_calls": float(rollout_batch["critic_calls"]),
        "critic_errors": float(rollout_batch["critic_errors"]),
        "skipped_zero_std": float(rollout_batch["skipped_zero_std"]),
        "used_time_minutes": round((time.time() - state.start_time) / 60, 4),
    }
    if allocation:
        record.update({f"allocation/{key}": float(value) for key, value in allocation.items()})
    if env_seconds_by_name:
        record.update({f"env_seconds/{key}": float(value) for key, value in env_seconds_by_name.items()})
    if env_rollout_batches:
        for key, batch in env_rollout_batches.items():
            env_lengths = batch["completion_lengths"]
            record[f"env_reward/{key}"] = float(mean(batch["rewards"])) if batch["rewards"] else 0.0
            record[f"env_generated_tokens/{key}"] = float(batch["generated_tokens"])
            record[f"env_mean_completion_length/{key}"] = float(mean(env_lengths)) if env_lengths else 0.0
            record[f"env_early_accepts/{key}"] = float(batch["early_accepts"])
            record[f"env_early_rejects/{key}"] = float(batch["early_rejects"])
            record[f"env_verified/{key}"] = float(batch["verified"])
            record[f"env_critic_calls/{key}"] = float(batch["critic_calls"])
            record[f"env_skipped_zero_std/{key}"] = float(batch["skipped_zero_std"])
    record.update({f"env_updates/{key}": float(value) for key, value in state.env_updates.items()})
    record.update(reward_stats.as_dict())
    record.update(allocator.as_dict())
    return record


def save_checkpoint(model, tokenizer, optimizer, args, step: int) -> None:
    output = Path(args.output_dir) / f"step{step}"
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    torch.save({"optimizer": optimizer.state_dict(), "step": step}, output / "optimizer.pt")
    print(f"Saved checkpoint to {output}")


def maybe_load_optimizer(args, optimizer) -> int:
    if not args.resume_from_checkpoint:
        return 0
    optimizer_path = Path(args.resume_from_checkpoint) / "optimizer.pt"
    if optimizer_path.exists():
        state = torch.load(optimizer_path, map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        print(f"Loaded optimizer state from {optimizer_path}")
        return int(state.get("step", 0))
    return 0


def adapters_disabled(model):
    disable_adapter = getattr(model, "disable_adapter", None)
    if callable(disable_adapter):
        return disable_adapter()
    if hasattr(model, "disable_adapter_layers") and hasattr(model, "enable_adapter_layers"):
        class AdapterContext:
            def __enter__(self):
                model.disable_adapter_layers()

            def __exit__(self, exc_type, exc, tb):
                model.enable_adapter_layers()
                return False

        return AdapterContext()
    return nullcontext()


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or ""),
        "source": str(row.get("source") or ""),
        "task": str(row.get("task") or ""),
        "prompt": row.get("prompt") or [],
        "answer": str(row.get("answer") or ""),
        "tests": row.get("tests") or [],
        "entry_point": str(row.get("entry_point") or ""),
        "test_type": str(row.get("test_type") or ""),
        "difficulty": str(row.get("difficulty") or ""),
    }


def collate_rows(examples: list[dict[str, Any]]) -> dict[str, Any]:
    return {"rows": [normalize_row(example) for example in examples]}


def has_eos(completion_ids: list[int], eos_token_id: int | None) -> bool:
    return eos_token_id is not None and eos_token_id in completion_ids


def model_device(model):
    return next(model.parameters()).device


def print_trainable_parameters(model) -> None:
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
        return
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100 * trainable / total:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--allow_code_execution", action="store_true")
    parser.add_argument("--max_steps", type=int)
    cli = parser.parse_args()
    config = dict(DEFAULT_CONFIG)
    config.update(load_yaml_mapping(cli.config))
    if cli.allow_code_execution:
        config["allow_code_execution"] = True
    if cli.max_steps is not None:
        config["max_steps"] = cli.max_steps
    args = argparse.Namespace(**config)
    if not args.model_dir:
        parser.error("model_dir is required in config.")
    if args.batch_size is None:
        args.batch_size = sum(int(value or 0) for value in args.env_batch_size.values())
        if args.batch_size <= 0:
            args.batch_size = len(args.env_data_paths)
    args.batch_size = int(args.batch_size)
    if bool(args.ensure_all_envs_per_step) and args.batch_size < len(args.env_data_paths):
        parser.error("batch_size must be >= number of environments when ensure_all_envs_per_step is true.")
    args.rollout_generation_batch_size = max(1, int(args.rollout_generation_batch_size))
    args.critic["concurrency"] = max(1, int(args.critic.get("concurrency", 1)))
    if not args.allow_code_execution:
        print("WARNING: code execution is disabled. Code environment rewards will fail unless early terminated.")
    return args


def load_yaml_mapping(path: str) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return data


def parse_list_arg(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def setup_output_dirs(args) -> None:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    if args.use_tensorboard:
        Path(args.tensorboard_log_dir).mkdir(parents=True, exist_ok=True)
    if not args.resume_from_checkpoint:
        Path(args.log_file).write_text("", encoding="utf-8")


def build_tensorboard_writer(args):
    if not args.use_tensorboard:
        return None
    if SummaryWriter is None:
        raise RuntimeError("TensorBoard is enabled but unavailable.")
    writer = SummaryWriter(log_dir=args.tensorboard_log_dir)
    writer.add_text("config/resolved", json.dumps(vars(args), indent=2), 0)
    return writer


def write_tensorboard_scalars(writer, record: dict[str, Any], step: int) -> None:
    if writer is None:
        return
    for key, value in record.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isfinite(float(value)):
            writer.add_scalar(f"ser/{key}", float(value), step)
    writer.flush()


def print_config(args) -> None:
    print("=" * 60)
    print("SER Configuration")
    print("=" * 60)
    for key, value in sorted(vars(args).items()):
        print(f"{key}: {value}")
    print("=" * 60)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def install_signal_handler() -> None:
    def handle_signal(signum, frame):
        print("Received signal, cleaning up...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)


if __name__ == "__main__":
    main()
