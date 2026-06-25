"""From-scratch GRPO baseline without speculative decoding or TRL.

This intentionally mirrors the FastGRPO script style: LoRA target training,
group-normalized rewards, zero-variance group skipping, and the clipped GRPO
objective with a KL term against the base model obtained by disabling adapters.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import yaml
except Exception:  # pragma: no cover - optional unless --config is used.
    yaml = None

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - tensorboard may be absent on minimal installs.
    SummaryWriter = None

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Baseline_GRPO.data_utils import (  # noqa: E402
    ProcessedDataCollator,
    load_processed_dataset,
    render_full_message,
)
from Baseline_GRPO.reward_utils import RewardStats, compute_reward  # noqa: E402


DEFAULT_CONFIG: dict[str, Any] = {
    "model_dir": "",
    "dataset_path": "Baseline/processed/dapo_taco/train",
    "output_dir": "Baseline_GRPO/outputs/grpo_dapo_taco",
    "log_file": "Baseline_GRPO/outputs/grpo_dapo_taco/train.jsonl",
    "resume_from_checkpoint": "",
    "batch_size": 1,
    "num_epochs": 1,
    "max_steps": None,
    "max_train_samples": None,
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
    "gradient_checkpointing": True,
    "use_cache": False,
    "lora_r": 64,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "lora_target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    "enable_thinking": False,
    "allow_code_execution": False,
    "code_timeout_seconds": 5.0,
    "save_steps": 500,
    "num_workers": 4,
    "seed": 42,
    "use_tensorboard": True,
    "tensorboard_log_dir": "Baseline_GRPO/outputs/grpo_dapo_taco/tensorboard",
}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    setup_output_dirs(args)

    print_config(args)
    writer = build_tensorboard_writer(args)
    dataset = load_processed_dataset(args.dataset_path, max_samples=args.max_train_samples)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = build_model(args)
    model.print_trainable_parameters()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.target_lr)
    resume_step = maybe_load_optimizer(args, optimizer)

    dataloader = DataLoader(
        dataset,
        collate_fn=ProcessedDataCollator(
            tokenizer,
            max_prompt_length=args.max_prompt_length,
            enable_thinking=args.enable_thinking,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    state = TrainingState()
    state.optimizer_steps = resume_step
    stats = RewardStats()
    optimizer.zero_grad(set_to_none=True)
    install_signal_handler()

    try:
        with Path(args.log_file).open("a", encoding="utf-8") as log_handle:
            for epoch in range(args.num_epochs):
                epoch_iter = tqdm(dataloader, desc=f"epoch {epoch + 1}/{args.num_epochs}")
                for batch_index, batch in enumerate(epoch_iter):
                    if batch["input_ids"].shape[-1] >= args.max_length:
                        continue
                    batch_start = time.time()
                    train_batch = generate_and_score_batch(model, tokenizer, batch, args, stats)
                    if not train_batch["messages"]:
                        continue

                    train_start = time.time()
                    loss_logs = train_on_batch(model, tokenizer, train_batch, optimizer, args, state)
                    train_seconds = time.time() - train_start
                    state.used_items += train_batch["used_items"]
                    state.generated_groups += len(batch["rows"])
                    state.skipped_zero_std += train_batch["skipped_zero_std"]
                    state.skipped_correct += train_batch["skipped_correct"]
                    state.skipped_incorrect += train_batch["skipped_incorrect"]

                    if state.accumulated_batches % args.accumulation_steps == 0:
                        if args.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        state.optimizer_steps += 1

                    log_data = build_log_record(
                        args,
                        state,
                        stats,
                        epoch=epoch,
                        batch_index=batch_index,
                        train_batch=train_batch,
                        loss_logs=loss_logs,
                        batch_seconds=time.time() - batch_start,
                        train_seconds=train_seconds,
                    )
                    log_handle.write(json.dumps(log_data) + "\n")
                    log_handle.flush()
                    write_tensorboard_scalars(writer, log_data, state.accumulated_batches)
                    epoch_iter.set_postfix(
                        step=state.optimizer_steps,
                        reward=round(log_data["mean_reward"], 4),
                        used=state.used_items,
                    )

                    if args.save_steps > 0 and state.optimizer_steps > 0 and state.optimizer_steps % args.save_steps == 0:
                        save_checkpoint(model, tokenizer, optimizer, args, state.optimizer_steps)

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


class TrainingState:
    def __init__(self) -> None:
        self.used_items = 0
        self.generated_groups = 0
        self.skipped_zero_std = 0
        self.skipped_correct = 0
        self.skipped_incorrect = 0
        self.accumulated_batches = 0
        self.optimizer_steps = 0
        self.start_time = time.time()


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="")
    config_args, _ = config_parser.parse_known_args()

    defaults = dict(DEFAULT_CONFIG)
    if config_args.config:
        defaults.update(load_yaml_config(config_args.config))
    defaults["config"] = config_args.config

    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    parser.add_argument("--model_dir", default=defaults["model_dir"])
    parser.add_argument("--dataset_path", default=defaults["dataset_path"])
    parser.add_argument("--output_dir", default=defaults["output_dir"])
    parser.add_argument("--log_file", default=defaults["log_file"])
    parser.add_argument("--resume_from_checkpoint", default=defaults["resume_from_checkpoint"])

    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"])
    parser.add_argument("--num_epochs", type=int, default=defaults["num_epochs"])
    parser.add_argument("--max_steps", type=optional_int, default=defaults["max_steps"])
    parser.add_argument("--max_train_samples", type=optional_int, default=defaults["max_train_samples"])
    parser.add_argument("--accumulation_steps", type=int, default=defaults["accumulation_steps"])
    parser.add_argument("--target_lr", type=float, default=defaults["target_lr"])
    parser.add_argument("--max_grad_norm", type=float, default=defaults["max_grad_norm"])

    parser.add_argument("--repeated_generate_nums", type=int, default=defaults["repeated_generate_nums"])
    parser.add_argument("--grpo_iteration_num", type=int, default=defaults["grpo_iteration_num"])
    parser.add_argument("--temperature", type=float, default=defaults["temperature"])
    parser.add_argument("--top_p", type=float, default=defaults["top_p"])
    parser.add_argument("--max_length", type=int, default=defaults["max_length"])
    parser.add_argument("--max_prompt_length", type=int, default=defaults["max_prompt_length"])
    parser.add_argument("--max_training_token", type=int, default=defaults["max_training_token"])
    parser.add_argument("--max_training_padding_gap", type=int, default=defaults["max_training_padding_gap"])
    parser.add_argument("--epsilon", type=float, default=defaults["epsilon"])
    parser.add_argument("--beta", type=float, default=defaults["beta"])

    parser.add_argument(
        "--gradient_checkpointing",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=defaults["gradient_checkpointing"],
    )
    parser.add_argument(
        "--use_cache",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=defaults["use_cache"],
    )
    parser.add_argument("--lora_r", type=int, default=defaults["lora_r"])
    parser.add_argument("--lora_alpha", type=int, default=defaults["lora_alpha"])
    parser.add_argument("--lora_dropout", type=float, default=defaults["lora_dropout"])
    parser.add_argument(
        "--lora_target_modules",
        default=defaults["lora_target_modules"],
    )

    parser.add_argument("--enable_thinking", type=str_to_bool, nargs="?", const=True, default=defaults["enable_thinking"])
    parser.add_argument("--allow_code_execution", type=str_to_bool, nargs="?", const=True, default=defaults["allow_code_execution"])
    parser.add_argument("--code_timeout_seconds", type=float, default=defaults["code_timeout_seconds"])
    parser.add_argument("--save_steps", type=int, default=defaults["save_steps"])
    parser.add_argument("--num_workers", type=int, default=defaults["num_workers"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--use_tensorboard", type=str_to_bool, nargs="?", const=True, default=defaults["use_tensorboard"])
    parser.add_argument("--tensorboard_log_dir", default=defaults["tensorboard_log_dir"])

    args = parser.parse_args()
    args.enable_thinking = str_to_bool(args.enable_thinking)
    args.allow_code_execution = str_to_bool(args.allow_code_execution)
    args.use_tensorboard = str_to_bool(args.use_tensorboard)
    args.gradient_checkpointing = str_to_bool(args.gradient_checkpointing)
    args.use_cache = str_to_bool(args.use_cache)
    if not args.model_dir:
        parser.error("model_dir is required. Set it in --config YAML or pass --model_dir.")
    return args


def build_model(args: argparse.Namespace):
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


def generate_and_score_batch(model, tokenizer, batch: dict[str, Any], args, stats: RewardStats) -> dict[str, Any]:
    input_ids = batch["input_ids"].cuda()
    attention_mask = batch["attention_mask"].cuda()
    rows = batch["rows"]
    repeated = args.repeated_generate_nums

    generation_input_ids = input_ids.repeat_interleave(repeated, dim=0)
    generation_attention_mask = attention_mask.repeat_interleave(repeated, dim=0)

    torch.cuda.synchronize()
    # Generation
    generate_start = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=generation_input_ids,
            attention_mask=generation_attention_mask,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_length=args.max_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    generate_seconds = time.time() - generate_start

    prompt_width = input_ids.shape[-1]
    completion_ids = generated_ids[:, prompt_width:]
    completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    completion_lengths = [int((ids != tokenizer.pad_token_id).sum().item()) for ids in completion_ids]

    messages: list[list[dict[str, str]]] = []
    raw_rewards: list[float] = []
    advantages: list[float] = []
    used_items = 0
    skipped_zero_std = 0
    skipped_correct = 0
    skipped_incorrect = 0

    for row_index, row in enumerate(rows):
        start = row_index * repeated
        end = start + repeated
        group_completions = completions[start:end]
        rewards = np.array(
            [
                compute_reward(
                    completion,
                    row,
                    allow_code_execution=args.allow_code_execution,
                    code_timeout_seconds=args.code_timeout_seconds,
                    stats=stats,
                )
                for completion in group_completions
            ],
            dtype=np.float32,
        )

        if float(rewards.std()) == 0.0:
            skipped_zero_std += 1
            if float(rewards[0]) >= 1.0:
                skipped_correct += 1
            else:
                skipped_incorrect += 1
            continue

        group_advantages = ((rewards - rewards.mean()) / rewards.std()).tolist()
        for completion, reward, advantage in zip(group_completions, rewards.tolist(), group_advantages):
            message = deepcopy(row["prompt"])
            message.append({"role": "assistant", "content": completion})
            messages.append(message)
            raw_rewards.append(float(reward))
            advantages.append(float(advantage))
        used_items += 1

    return {
        "messages": messages,
        "rewards": raw_rewards,
        "advantages": advantages,
        "completion_lengths": completion_lengths,
        "generate_seconds": generate_seconds,
        "used_items": used_items,
        "skipped_zero_std": skipped_zero_std,
        "skipped_correct": skipped_correct,
        "skipped_incorrect": skipped_incorrect,
    }


def train_on_batch(model, tokenizer, train_batch: dict[str, Any], optimizer, args, state: TrainingState) -> dict[str, float]:
    input_ids, attention_mask, loss_mask = encode_messages_for_loss(
        tokenizer,
        train_batch["messages"],
        args.enable_thinking,
    )
    advantages = train_batch["advantages"]
    sorted_items = sorted(
        zip(input_ids, attention_mask, loss_mask, advantages),
        key=lambda item: len(item[0]),
    )
    chunks = make_training_chunks(sorted_items, args.max_training_token, args.max_training_padding_gap)

    total_loss = 0.0
    total_kl = 0.0
    old_logps_by_chunk = [None for _ in chunks]
    ref_logps_by_chunk = [None for _ in chunks]

    for grpo_iteration in range(args.grpo_iteration_num):
        for chunk_index, chunk in enumerate(chunks):
            tensors = pad_chunk(chunk, tokenizer.pad_token_id, model_device(model))
            labels = tensors["input_ids"]
            mask = tensors["loss_mask"]
            reward = tensors["advantages"].unsqueeze(-1)

            ref_logps = ref_logps_by_chunk[chunk_index]
            if grpo_iteration == 0:
                if args.beta > 0:
                    with adapters_disabled(model), torch.no_grad():
                        ref_logits = model(
                            input_ids=tensors["input_ids"],
                            attention_mask=tensors["attention_mask"],
                        ).logits
                    ref_logps = gather_token_logps(ref_logits, labels).detach()
                else:
                    ref_logps = None

            outputs = model(
                input_ids=tensors["input_ids"],
                attention_mask=tensors["attention_mask"],
            )
            logps = gather_token_logps(outputs.logits, labels)
            old_logps = old_logps_by_chunk[chunk_index]
            if grpo_iteration == 0:
                old_logps = logps.detach()

            loss, kl = compute_grpo_loss(
                logps=logps,
                old_logps=old_logps,
                ref_logps=ref_logps,
                mask=mask[:, :-1],
                reward=reward,
                epsilon=args.epsilon,
                beta=args.beta,
            )
            scaled_loss = loss / max(1, len(train_batch["messages"])) / max(1, args.accumulation_steps)
            scaled_loss.backward()

            total_loss += float(loss.detach().item())
            total_kl += float(kl.detach().item())
            if grpo_iteration == 0:
                old_logps_by_chunk[chunk_index] = old_logps
                ref_logps_by_chunk[chunk_index] = ref_logps

    state.accumulated_batches += 1
    denom = max(1, len(chunks) * args.grpo_iteration_num)
    return {
        "loss": total_loss / denom,
        "kl": total_kl / denom,
        "num_train_sequences": float(len(train_batch["messages"])),
    }


def encode_messages_for_loss(tokenizer, messages: list[list[dict[str, str]]], enable_thinking: bool):
    full_texts = [render_full_message(tokenizer, message) for message in messages]
    tokenized = tokenizer(full_texts, padding=False, add_special_tokens=False)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    loss_mask = []
    for message, full_ids in zip(messages, input_ids):
        try:
            prompt_text = tokenizer.apply_chat_template(
                message[:-1],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                message[:-1],
                tokenize=False,
                add_generation_prompt=True,
            )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)
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
    for item in items:
        seq_len = len(item[0])
        can_add = (
            (max(current_max_len, seq_len) * (len(current) + 1) <= max_training_token)
            and ((seq_len - current_max_len) * len(current) <= max_padding_gap)
        ) or not current
        if not can_add:
            chunks.append(current)
            current = []
            current_max_len = 0
        current.append(item)
        current_max_len = max(current_max_len, seq_len)
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
    logits = logits[:, :-1, :].float()
    labels = labels[:, 1:].to(logits.device)
    return torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(-1)).squeeze(-1)


def compute_grpo_loss(
    *,
    logps: torch.Tensor,                # [num_samples, num_tokens]
    old_logps: torch.Tensor,            # [num_samples, num_tokens]
    ref_logps: torch.Tensor | None,     # [num_samples, num_tokens] or None 
    mask: torch.Tensor,             # [num_samples, num_tokens]
    reward: torch.Tensor,       # [num_samples, 1]
    epsilon: float,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ratio = torch.exp(logps - old_logps)    # token level
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)    # token level
    policy_term = torch.minimum(ratio * reward, clipped_ratio * reward) # token level

    if beta > 0 and ref_logps is not None:
        diff = ref_logps - logps    
        kl = torch.exp(diff) - diff - 1.0   # token level
    else:
        kl = torch.zeros_like(policy_term)

    token_loss = -(policy_term - beta * kl) * mask
    denom = mask.sum(dim=-1).clamp_min(1.0)
    sequence_loss = token_loss.sum(dim=-1) / denom      # sequence level
    sequence_kl = (kl * mask).sum(dim=-1) / denom      # sequence level
    return sequence_loss.sum(), sequence_kl.mean()


def adapters_disabled(model):
    if hasattr(model, "disable_adapter_layers") and hasattr(model, "enable_adapter_layers"):
        class _AdapterContext:
            def __enter__(self_inner):
                model.disable_adapter_layers()

            def __exit__(self_inner, exc_type, exc, tb):
                model.enable_adapter_layers()
                return False

        return _AdapterContext()
    return nullcontext()


def build_log_record(args, state, stats, *, epoch, batch_index, train_batch, loss_logs, batch_seconds, train_seconds):
    lengths = train_batch["completion_lengths"]
    mean_reward = mean(train_batch["rewards"]) if train_batch["rewards"] else 0.0
    return {
        "epoch": epoch + 1,
        "batch_index": batch_index,
        "optimizer_step": state.optimizer_steps,
        "used_items": state.used_items,
        "generated_groups": state.generated_groups,
        "skipped_zero_std_total": state.skipped_zero_std,
        "skipped_correct_total": state.skipped_correct,
        "skipped_incorrect_total": state.skipped_incorrect,
        "batch_skipped_zero_std": train_batch["skipped_zero_std"],
        "batch_used_items": train_batch["used_items"],
        "mean_reward": float(mean_reward),
        "loss": loss_logs["loss"],
        "kl": loss_logs["kl"],
        "num_train_sequences": loss_logs["num_train_sequences"],
        "generate_time_cost": train_batch["generate_seconds"],
        "train_time_cost": train_seconds,
        "batch_time_cost": batch_seconds,
        "mean_completion_length": float(mean(lengths)) if lengths else 0.0,
        "max_completion_length": float(max(lengths)) if lengths else 0.0,
        "length_stdev": float(stdev(lengths)) if len(lengths) > 1 else 0.0,
        "used_time_minutes": round((time.time() - state.start_time) / 60, 4),
        **stats.as_dict(),
    }


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


def load_yaml_config(path: str) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required for --config. Install it with `pip install pyyaml`.")
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    unknown_keys = sorted(set(data) - set(DEFAULT_CONFIG))
    if unknown_keys:
        raise ValueError(f"Unknown config keys in {config_path}: {', '.join(unknown_keys)}")
    return data


def str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def optional_int(value) -> int | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    return int(value)


def parse_list_arg(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def model_device(model):
    return next(model.parameters()).device


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
        raise RuntimeError("TensorBoard logging is enabled, but tensorboard is not installed.")
    writer = SummaryWriter(log_dir=args.tensorboard_log_dir)
    writer.add_text("config/resolved", json.dumps(serializable_config(args), indent=2), 0)
    return writer


def write_tensorboard_scalars(writer, log_data: dict[str, Any], step: int) -> None:
    if writer is None:
        return
    for key, value in log_data.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            writer.add_scalar(f"train/{key}", numeric, step)
    writer.flush()


def serializable_config(args) -> dict[str, Any]:
    return {key: value for key, value in sorted(vars(args).items())}


def print_config(args) -> None:
    print("=" * 60)
    print("Baseline GRPO Configuration")
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
