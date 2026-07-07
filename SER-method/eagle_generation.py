"""EAGLE speculative generation adapter for SER.

This module intentionally keeps the training loop clean.  The public engine
accepts the same token-list batch used by ``generate_token_chunk`` and returns
the same full-token-id lists, while hiding the EAGLE prefill, dynamic draft
tree, target verification, accepted-path extraction, and online draft training.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from eagle_draft import EagleDraftWrapper
from FastGRPO.helper.specualtive_generate import speculative_generate


@dataclass
class EagleStats:
    calls: int = 0
    fallback_calls: int = 0
    generated_tokens: int = 0
    accepted_tokens: float = 0.0
    decoded_positions: float = 0.0
    total_seconds: float = 0.0
    target_seconds: float = 0.0
    draft_seconds: float = 0.0
    check_seconds: float = 0.0
    draft_updates: int = 0
    draft_loss_feature_sum: float = 0.0
    draft_loss_logit_sum: float = 0.0

    def update_from_output(self, output: dict[str, Any], generated_tokens: int, seconds: float) -> None:
        self.calls += 1
        self.generated_tokens += int(generated_tokens)
        self.accepted_tokens += float(output.get("total_acc_length", 0.0) or 0.0)
        self.decoded_positions += float(output.get("total_decoded_token_num", 0.0) or 0.0)
        self.total_seconds += float(seconds)
        self.target_seconds += float(output.get("target_time_cost", 0.0) or 0.0)
        self.draft_seconds += float(output.get("draft_time_cost", 0.0) or 0.0)
        self.check_seconds += float(output.get("check_time_cost", 0.0) or 0.0)

    def update_draft_loss(self, feature_loss: float, logit_loss: float, did_step: bool) -> None:
        self.draft_loss_feature_sum += float(feature_loss)
        self.draft_loss_logit_sum += float(logit_loss)
        self.draft_updates += int(bool(did_step))

    def as_dict(self, prefix: str = "speculative") -> dict[str, float]:
        avg_accept = self.accepted_tokens / max(1.0, self.decoded_positions)
        return {
            f"{prefix}/calls": float(self.calls),
            f"{prefix}/fallback_calls": float(self.fallback_calls),
            f"{prefix}/generated_tokens": float(self.generated_tokens),
            f"{prefix}/avg_accepted_tokens": float(avg_accept),
            f"{prefix}/seconds": float(self.total_seconds),
            f"{prefix}/target_seconds": float(self.target_seconds),
            f"{prefix}/draft_seconds": float(self.draft_seconds),
            f"{prefix}/check_seconds": float(self.check_seconds),
            f"{prefix}/draft_updates": float(self.draft_updates),
            f"{prefix}/draft_loss_feature": self.draft_loss_feature_sum / max(1, self.calls),
            f"{prefix}/draft_loss_logit": self.draft_loss_logit_sum / max(1, self.calls),
        }

    def reset_window(self) -> None:
        self.calls = 0
        self.fallback_calls = 0
        self.generated_tokens = 0
        self.accepted_tokens = 0.0
        self.decoded_positions = 0.0
        self.total_seconds = 0.0
        self.target_seconds = 0.0
        self.draft_seconds = 0.0
        self.check_seconds = 0.0
        self.draft_updates = 0
        self.draft_loss_feature_sum = 0.0
        self.draft_loss_logit_sum = 0.0


@dataclass
class EagleSpeculativeEngine:
    target_model: Any
    tokenizer: Any
    cfg: dict[str, Any]
    max_training_token: int
    max_training_padding_gap: int
    wrapper: EagleDraftWrapper
    optimizer: torch.optim.Optimizer | None = None
    stats: EagleStats = field(default_factory=EagleStats)
    draft_accumulated_batches: int = 0
    draft_optimizer_steps: int = 0
    loaded_from_checkpoint: bool = False

    @classmethod
    def build(cls, target_model, tokenizer, args) -> "EagleSpeculativeEngine | None":
        cfg = dict(getattr(args, "speculative", {}) or {})
        if not bool(cfg.get("enabled", False)):
            return None

        adapter_path = str(cfg.get("draft_adapter_path") or "").strip()
        allow_scratch = bool(cfg.get("allow_scratch_draft", True))
        if not adapter_path and not allow_scratch:
            raise ValueError("speculative.draft_adapter_path is required when allow_scratch_draft is false.")
        if not adapter_path:
            print("WARNING: EAGLE speculative enabled with scratch draft initialization.")

        wrapper = EagleDraftWrapper(        # EAGLE draft model wrapper
            target_model,
            draft_layers=int(cfg.get("draft_num_layers", 1)),
            adapter_path=adapter_path or None,
        ).to(next(target_model.parameters()).device)
        for param in wrapper.draft_model.parameters():
            param.requires_grad_(True)

        optimizer = None
        if bool(cfg.get("train_draft", True)):
            optimizer = torch.optim.AdamW(
                wrapper.draft_model.parameters(),
                lr=float(cfg.get("draft_lr", 1e-4)),
                betas=(0.9, 0.95),
                weight_decay=float(cfg.get("draft_weight_decay", 0.0)),
            )
            optimizer.zero_grad(set_to_none=True)

        engine = cls(
            target_model=target_model,
            tokenizer=tokenizer,
            cfg=cfg,
            max_training_token=int(args.max_training_token),
            max_training_padding_gap=int(args.max_training_padding_gap),
            wrapper=wrapper,
            optimizer=optimizer,
        )
        engine.maybe_load_checkpoint(getattr(args, "resume_from_checkpoint", ""))
        return engine

    def enabled(self) -> bool:
        return bool(self.cfg.get("enabled", False))

    def should_run_warmup(self) -> bool:
        """Return whether scratch-draft alignment should run before SER.

        Online draft learning happens during rollout, but a scratch draft is so
        unaligned that speculative acceptance can be poor at the beginning.  The
        warmup stage mirrors FastGRPO's `train_draft.py`: freeze the target,
        compute target hidden states, and train only the EAGLE draft to predict
        the next target feature/logit distribution.
        """

        if self.optimizer is None or not bool(self.cfg.get("train_draft", True)):
            return False
        if self.loaded_from_checkpoint and not bool(self.cfg.get("draft_warmup_always", False)):
            return False
        if str(self.cfg.get("draft_adapter_path") or "").strip() and not bool(self.cfg.get("draft_warmup_always", False)):
            return False
        return int(self.cfg.get("draft_warmup_steps", 0) or 0) > 0

    def should_fallback(self, batch_size: int) -> bool:
        if not torch.cuda.is_available():
            return True
        fallback_batch_size = self.cfg.get("fallback_batch_size")
        if fallback_batch_size is not None and int(fallback_batch_size) > 0 and batch_size >= int(fallback_batch_size):
            return True
        verification_num = min(
            math.floor(float(self.cfg.get("verification_capacity", 160)) / max(1, batch_size)),
            int(self.cfg.get("max_verification_num", 160)),
        )
        return verification_num <= 1

    def generate(
        self,
        token_lists: list[list[int]],
        max_new_tokens: int,
        *,
        temperature: float,
        top_p: float,
        pad_token_id: int,
        eos_token_id: int | None,
    ) -> list[list[int]]:
        if not token_lists:
            return []
        if self.should_fallback(len(token_lists)):
            self.stats.fallback_calls += 1
            results = normal_generate(
                self.target_model,
                self.tokenizer,
                token_lists,
                max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            self.stats.generated_tokens += sum(
                max(0, len(result) - len(tokens))
                for result, tokens in zip(results, token_lists)
            )
            return results

        input_tensor, attention_mask, _ = pad_left(token_lists, pad_token_id, self.wrapper.device)
        max_total_length = input_tensor.shape[-1] + int(max_new_tokens)
        start_lengths = [len(tokens) for tokens in token_lists]

        # Phase 1-4 live inside speculative_generate:
        #   1. target prefill obtains target features for the prefix,
        #   2. the draft recursively expands a confidence-ranked tree,
        #   3. the target verifies the tree in parallel,
        #   4. the longest accepted path is stitched back into each sequence.
        was_training = self.wrapper.draft_model.training
        self.wrapper.draft_model.eval()
        start_time = time.time()
        with torch.no_grad():
            output = speculative_generate(
                model=self.wrapper,
                input_ids=input_tensor,
                attention_mask=attention_mask,
                tokenizer=self.tokenizer,
                do_sample=True,
                repeated_generate_nums=None,
                temperature=float(temperature),
                top_p=float(top_p),
                verification_capacity=int(self.cfg.get("verification_capacity", 160)),
                max_draft_token_length=int(self.cfg.get("max_draft_token_length", 5)),
                max_draft_k=int(self.cfg.get("max_draft_k", 8)),
                max_verification_num=int(self.cfg.get("max_verification_num", 160)),
                min_draft_token_length=int(self.cfg.get("min_draft_token_length", 3)),
                draft_token_length_c=float(self.cfg.get("draft_token_length_c", 0.75)),
                statistical_time=True,
                return_all_draft_input=bool(self.cfg.get("train_draft", True)),
                max_length=max_total_length,
            )
        if was_training:
            self.wrapper.draft_model.train()

        generated_ids = output.get("generated_token_ids") or [[] for _ in token_lists]
        results: list[list[int]] = []
        generated_count = 0
        for tokens, generated in zip(token_lists, generated_ids):
            new_tokens = [int(token) for token in list(generated)[:max_new_tokens]]
            if eos_token_id is not None and eos_token_id in new_tokens:
                new_tokens = new_tokens[: new_tokens.index(eos_token_id) + 1]
            generated_count += len(new_tokens)
            results.append(list(tokens) + new_tokens)

        self.stats.update_from_output(output, generated_count, time.time() - start_time)
        if self.optimizer is not None and bool(self.cfg.get("train_draft", True)):
            if bool(self.cfg.get("draft_train_from_target_hidden", True)):
                # Online draft learning is anchored to target hidden states.
                # Recompute the target features on the accepted prompt+rollout
                # sequences, then train the draft against those teacher states.
                self.train_draft_from_token_lists(results, start_lengths)
            else:
                self.train_draft_from_trace(output, start_lengths)
        del input_tensor, attention_mask
        return results

    def generate_teacher_sequences(
        self,
        prompt_token_lists: list[list[int]],
        max_new_tokens: int,
        *,
        temperature: float,
        top_p: float,
        pad_token_id: int,
        eos_token_id: int | None,
        do_sample: bool,
    ) -> list[list[int]]:
        """Generate target-model assistant responses for draft warmup.

        Some RLVR rows, especially code rows, have tests but no supervised
        answer text.  For those rows, we let the target produce a real
        assistant continuation first.  The warmup step later computes target
        hidden states on this full prompt+response sequence and trains only the
        draft to imitate those target features.
        """

        if not prompt_token_lists:
            return []
        was_training = self.target_model.training
        self.target_model.eval()
        try:
            return normal_generate(
                self.target_model,
                self.tokenizer,
                prompt_token_lists,
                max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                do_sample=do_sample,
            )
        finally:
            if was_training:
                self.target_model.train()

    def train_draft_from_trace(self, output: dict[str, Any], start_lengths: list[int]) -> None:
        states = output.get("all_draft_input_states")
        ids = output.get("all_draft_input_ids")
        if not states or not ids:
            return
        examples = [
            (input_ids.detach(), hidden_states.detach(), int(prefix_len))
            for input_ids, hidden_states, prefix_len in zip(ids, states, start_lengths)
            if len(input_ids) > 1
        ]
        self.train_draft_examples(examples)

    def train_draft_from_token_lists(self, token_lists: list[list[int]], prefix_lengths: list[int]) -> dict[str, float]:
        """Train the draft from target hidden states on full sequences.

        `token_lists` are complete prompt+completion sequences.  The target
        forward pass is no-grad and produces the teacher feature stream; the
        draft then learns the EAGLE transition:

            target h_t + embedding(token_{t+1}) -> predict target h_{t+1}
        """

        if self.optimizer is None or not token_lists:
            return {"feature_loss": 0.0, "logit_loss": 0.0, "did_step": 0.0}
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
        input_ids, attention_mask = pad_right(token_lists, pad_token_id=int(pad_id), device=self.wrapper.device)
        examples = self.build_target_hidden_examples(input_ids, attention_mask, prefix_lengths)
        del input_ids, attention_mask
        return self.train_draft_examples(examples)

    def pretrain_from_token_batches(
        self,
        batches: list[tuple[torch.Tensor, torch.Tensor, list[int]]],
        *,
        max_steps: int,
        progress_bar=None,
    ) -> dict[str, float]:
        """Align the draft model with the current target before SER rollouts.

        Each batch contains full sequence tokens, attention masks, and prefix
        lengths.  We run the target once with `output_hidden_states=True`, then
        shift ids/features exactly like FastGRPO draft pretraining:

            target hidden h_t + token embedding x_{t+1} -> predict h_{t+1}

        The target forward pass is no-grad; only `self.wrapper.draft_model`
        receives gradients.
        """

        if self.optimizer is None or max_steps <= 0:
            return {"draft_warmup_steps": 0.0}

        was_training = self.target_model.training
        self.target_model.eval()
        self.wrapper.draft_model.train()
        steps = 0
        total_feature = 0.0
        total_logit = 0.0
        total_examples = 0
        total_train_tokens = 0
        optimizer_steps_before = self.draft_optimizer_steps
        start_time = time.time()
        for input_ids, attention_mask, prefix_lengths in batches:
            if steps >= max_steps:
                break
            input_ids = input_ids.to(self.wrapper.device)
            attention_mask = attention_mask.to(self.wrapper.device)
            examples = self.build_target_hidden_examples(input_ids, attention_mask, prefix_lengths)

            if examples:
                logs = self.train_draft_examples(
                    examples,
                    accumulation_steps=int(self.cfg.get("draft_warmup_accumulation_steps", 16)),
                )
                total_feature += float(logs.get("feature_loss", 0.0))
                total_logit += float(logs.get("logit_loss", 0.0))
                total_examples += len(examples)
                total_train_tokens += sum(max(0, len(ids) - max(0, int(prefix_len)) - 1) for ids, _, prefix_len in examples)
                steps += 1
                if progress_bar is not None:
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        examples=total_examples,
                        tokens=total_train_tokens,
                        opt_steps=self.draft_optimizer_steps - optimizer_steps_before,
                    )

        if was_training:
            self.target_model.train()
        return {
            "draft_warmup_steps": float(steps),
            "draft_warmup_seconds": float(time.time() - start_time),
            "draft_warmup_feature_loss": total_feature / max(1, steps),
            "draft_warmup_logit_loss": total_logit / max(1, steps),
            "draft_warmup_examples": float(total_examples),
            "draft_warmup_train_tokens": float(total_train_tokens),
            "draft_warmup_optimizer_steps": float(self.draft_optimizer_steps - optimizer_steps_before),
        }

    def build_target_hidden_examples(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix_lengths: list[int],
    ) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
        """Build shifted EAGLE examples from target hidden states.

        We freeze the target forward with `torch.no_grad()`.  For a sequence
        token_0..token_n, the draft receives token_1..token_n together with
        target hidden_0..hidden_{n-1}; its feature prediction is supervised by
        target hidden_1..hidden_n inside `_draft_loss_for_chunk`.
        """

        was_training = self.target_model.training
        self.target_model.eval()
        try:
            with torch.no_grad():
                outputs = self.target_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                hidden = outputs.hidden_states[-1]
        finally:
            if was_training:
                self.target_model.train()

        examples: list[tuple[torch.Tensor, torch.Tensor, int]] = []
        lengths = attention_mask.sum(dim=-1).tolist()
        for row_idx, seq_len in enumerate(lengths):
            seq_len = int(seq_len)
            if seq_len < 4:
                continue
            ids = input_ids[row_idx, 1:seq_len].detach()
            states = hidden[row_idx, : seq_len - 1, :].detach()
            prefix_len = max(0, min(int(prefix_lengths[row_idx]) - 1, len(ids)))
            examples.append((ids, states, prefix_len))
        return examples

    def train_draft_examples(
        self,
        examples: list[tuple[torch.Tensor, torch.Tensor, int]],
        *,
        accumulation_steps: int | None = None,
    ) -> dict[str, float]:
        if not examples:
            return {"feature_loss": 0.0, "logit_loss": 0.0, "did_step": 0.0}
        accumulation_steps = max(1, int(accumulation_steps or self.cfg.get("draft_online_accumulation_steps", 1)))
        examples.sort(key=lambda item: int(item[0].shape[-1]))
        total_feature = 0.0
        total_logit = 0.0
        total_examples = len(examples)
        self.wrapper.draft_model.train()
        for chunk in make_draft_training_chunks(
            examples,
            max_training_token=max(1, self.max_training_token * 2),
            max_padding_gap=max(1, self.max_training_padding_gap),
        ):
            feature_loss, logit_loss = self._draft_loss_for_chunk(chunk)
            loss = (
                float(self.cfg.get("feature_loss_weight", 2.0)) * feature_loss
                + float(self.cfg.get("logit_loss_weight", 0.1)) * logit_loss
            )
            if not torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                continue
            scaled = loss / max(1, total_examples) / accumulation_steps
            scaled.backward()
            total_feature += float(feature_loss.detach().item())
            total_logit += float(logit_loss.detach().item())
            del feature_loss, logit_loss, loss, scaled

        self.draft_accumulated_batches += 1
        did_step = False
        if self.draft_accumulated_batches % accumulation_steps == 0:
            max_grad_norm = float(self.cfg.get("draft_max_grad_norm", 1.0))
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.wrapper.draft_model.parameters(), max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.draft_optimizer_steps += 1
            did_step = True
        self.stats.update_draft_loss(total_feature / max(1, len(examples)), total_logit / max(1, len(examples)), did_step)
        return {
            "feature_loss": total_feature / max(1, len(examples)),
            "logit_loss": total_logit / max(1, len(examples)),
            "did_step": float(did_step),
        }

    def _draft_loss_for_chunk(self, chunk: list[tuple[torch.Tensor, torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.wrapper.device
        max_len = max(int(item[0].shape[-1]) for item in chunk)
        hidden_size = int(chunk[0][1].shape[-1])
        input_ids, hidden_states, attention_mask, loss_mask = [], [], [], []
        for ids, states, prefix_len in chunk:
            cur_len = int(ids.shape[-1])
            pad_len = max_len - cur_len
            prefix_len = min(max(0, int(prefix_len)), cur_len)
            input_ids.append(F.pad(ids.to(device), (0, pad_len), value=0))
            hidden_states.append(
                torch.cat(
                    [
                        states.to(device),
                        torch.zeros((pad_len, hidden_size), dtype=states.dtype, device=device),
                    ],
                    dim=0,
                )
            )
            attention_mask.append([1] * cur_len + [0] * pad_len)
            loss_mask.append([0] * prefix_len + [1] * (cur_len - prefix_len) + [0] * pad_len)

        input_ids_t = torch.stack(input_ids, dim=0).long()
        hidden_states_t = torch.stack(hidden_states, dim=0)
        attention_mask_t = torch.tensor(attention_mask, device=device, dtype=torch.long)
        loss_mask_t = torch.tensor(loss_mask, device=device, dtype=torch.float32)

        # Draft training phase: recompute the draft with gradients.  The target
        # feature tensor is a fixed teacher signal; only draft parameters update.
        draft_outputs = self.wrapper(
            hidden_states=hidden_states_t,
            input_ids=input_ids_t,
            attention_mask=attention_mask_t,
            use_cache=False,
        )
        next_feature_states = draft_outputs["next_feature_states"]
        draft_hidden_states = draft_outputs["hidden_states"].to(self.wrapper.dtype)
        draft_logits = self.wrapper.lm_head(draft_hidden_states)

        shifted_mask = loss_mask_t[:, :-1]
        denom = shifted_mask.sum(dim=-1).clamp_min(1.0)
        feature_loss = F.smooth_l1_loss(
            next_feature_states[:, :-1, :].float(),
            hidden_states_t[:, 1:, :].float(),
            reduction="none",
        ).mean(dim=-1)
        feature_loss = ((feature_loss * shifted_mask).sum(dim=-1) / denom).sum()

        with torch.no_grad():
            target_logits = self.wrapper.target_lm_head(hidden_states_t.to(self.wrapper.dtype))
            target_probs = target_logits[:, 1:, :].float().softmax(dim=-1)
        log_probs = draft_logits[:, :-1, :].float().log_softmax(dim=-1)
        logit_loss = -(target_probs * log_probs).sum(dim=-1)
        logit_loss = ((logit_loss * shifted_mask).sum(dim=-1) / denom).sum()
        return feature_loss, logit_loss

    def save_checkpoint(self, output_dir: str | Path) -> None:
        path = Path(output_dir) / "speculative.pt"
        torch.save(
            {
                "draft_model": self.wrapper.draft_model.state_dict(),
                "draft_optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
                "draft_accumulated_batches": int(self.draft_accumulated_batches),
                "draft_optimizer_steps": int(self.draft_optimizer_steps),
            },
            path,
        )

    def maybe_load_checkpoint(self, checkpoint_dir: str | Path) -> None:
        if not checkpoint_dir:
            return
        path = Path(checkpoint_dir) / "speculative.pt"
        if not path.exists():
            return
        state = torch.load(path, map_location="cpu")
        self.wrapper.draft_model.load_state_dict(state["draft_model"])
        if self.optimizer is not None and state.get("draft_optimizer") is not None:
            self.optimizer.load_state_dict(state["draft_optimizer"])
        self.draft_accumulated_batches = int(state.get("draft_accumulated_batches", 0))
        self.draft_optimizer_steps = int(state.get("draft_optimizer_steps", 0))
        self.loaded_from_checkpoint = True
        print(f"Loaded EAGLE speculative state from {path}")

    def pop_log_stats(self) -> dict[str, float]:
        stats = self.stats.as_dict()
        stats["speculative/draft_optimizer_steps"] = float(self.draft_optimizer_steps)
        self.stats.reset_window()
        return stats


def build_eagle_engine(args, model, tokenizer) -> EagleSpeculativeEngine | None:
    return EagleSpeculativeEngine.build(model, tokenizer, args)


def pad_left(token_lists: list[list[int]], pad_token_id: int, device: torch.device):
    max_len = max(len(tokens) for tokens in token_lists)
    input_ids, attention_mask, pad_lengths = [], [], []
    for tokens in token_lists:
        pad_len = max_len - len(tokens)
        pad_lengths.append(pad_len)
        input_ids.append([pad_token_id] * pad_len + list(tokens))
        attention_mask.append([0] * pad_len + [1] * len(tokens))
    return (
        torch.tensor(input_ids, device=device, dtype=torch.long),
        torch.tensor(attention_mask, device=device, dtype=torch.long),
        pad_lengths,
    )


def pad_right(token_lists: list[list[int]], pad_token_id: int, device: torch.device):
    max_len = max(len(tokens) for tokens in token_lists)
    input_ids, attention_mask = [], []
    for tokens in token_lists:
        pad_len = max_len - len(tokens)
        input_ids.append(list(tokens) + [pad_token_id] * pad_len)
        attention_mask.append([1] * len(tokens) + [0] * pad_len)
    return (
        torch.tensor(input_ids, device=device, dtype=torch.long),
        torch.tensor(attention_mask, device=device, dtype=torch.long),
    )


def normal_generate(
    model,
    tokenizer,
    token_lists: list[list[int]],
    max_new_tokens: int,
    *,
    temperature: float,
    top_p: float,
    pad_token_id: int,
    eos_token_id: int | None,
    do_sample: bool = True,
) -> list[list[int]]:
    input_tensor, mask_tensor, pad_lengths = pad_left(token_lists, pad_token_id, next(model.parameters()).device)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_tensor,
            attention_mask=mask_tensor,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
    result = []
    for output, pad_len in zip(outputs, pad_lengths):
        result.append([int(token) for token in output.tolist()[pad_len:]])
    return result


def make_draft_training_chunks(
    examples: list[tuple[torch.Tensor, torch.Tensor, int]],
    *,
    max_training_token: int,
    max_padding_gap: int,
) -> list[list[tuple[torch.Tensor, torch.Tensor, int]]]:
    chunks = []
    current = []
    current_max_len = 0
    current_token_count = 0
    for example in examples:
        seq_len = int(example[0].shape[-1])
        proposed_max = max(current_max_len, seq_len)
        proposed_count = current_token_count + seq_len
        can_add = not current or (
            proposed_max * (len(current) + 1) <= max_training_token
            and proposed_max * (len(current) + 1) - proposed_count <= max_padding_gap
        )
        if not can_add:
            chunks.append(current)
            current = []
            current_max_len = 0
            current_token_count = 0
        current.append(example)
        current_max_len = max(current_max_len, seq_len)
        current_token_count += seq_len
    if current:
        chunks.append(current)
    return chunks
