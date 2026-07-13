"""Thin adapter from SER rollouts to the bundled FastGRPO EAGLE generator.

The heavy speculative decoding logic lives in
``FastGRPO.helper.specualtive_generate``.  This module only adapts the SER
target model to the small interface that FastGRPO expects:

    wrapper.target_model  -> current SER policy
    wrapper.draft_model   -> pretrained EAGLE draft model
    wrapper(...)          -> draft forward pass

The draft model can optionally be trained online from traces returned by
``speculative_generate(..., return_all_draft_input=True)``.  Keep that path
small because rollout speed matters more than exhaustive draft optimization.
"""

from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from FastGRPO.helper.modeling_draft import DraftModel
from FastGRPO.helper.specualtive_generate import speculative_generate


def resolve_attr(model, names: tuple[str, ...]):
    current = model
    for name in names:
        if not hasattr(current, name):
            return None
        current = getattr(current, name)
    return current


def resolve_embed_tokens(target_model) -> nn.Module:
    for path in (
        ("model", "embed_tokens"),
        ("base_model", "model", "model", "embed_tokens"),
        ("base_model", "model", "embed_tokens"),
    ):
        module = resolve_attr(target_model, path)
        if module is not None:
            return module
    raise AttributeError("Could not locate target embedding layer for FastGRPO draft.")


def resolve_lm_head(target_model) -> nn.Module:
    for path in (
        ("lm_head",),
        ("base_model", "model", "lm_head"),
    ):
        module = resolve_attr(target_model, path)
        if module is not None:
            return module
    raise AttributeError("Could not locate target LM head for FastGRPO draft.")


def frozen_lm_head(lm_head: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Run the target LM head while keeping its weights out of draft training."""

    if isinstance(lm_head, nn.Linear):
        bias = lm_head.bias.detach() if lm_head.bias is not None else None
        return F.linear(hidden_states, lm_head.weight.detach(), bias)

    old_flags = [param.requires_grad for param in lm_head.parameters()]
    try:
        for param in lm_head.parameters():
            param.requires_grad_(False)
        return lm_head(hidden_states)
    finally:
        for param, flag in zip(lm_head.parameters(), old_flags):
            param.requires_grad_(flag)


class TargetModelProxy(nn.Module):
    """Expose device/dtype helpers while delegating to the SER target policy."""

    def __init__(self, target_model):
        super().__init__()
        self.target = target_model

    @property
    def device(self) -> torch.device:
        return next(self.target.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.target.parameters()).dtype

    @property
    def lm_head(self) -> nn.Module:
        return resolve_lm_head(self.target)

    def forward(self, *args, **kwargs):
        return self.target(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.target, name)


class FastGRPODraftWrapper(nn.Module):
    """FastGRPO-compatible wrapper around the SER target and pretrained draft."""

    def __init__(self, target_model, draft_model_path: str | Path, draft_num_layers: int = 1):
        super().__init__()
        config = deepcopy(target_model.config)
        dtype = getattr(config, "dtype", None) or getattr(config, "dtype", None)
        if dtype is None or isinstance(dtype, str):
            dtype = next(target_model.parameters()).dtype
        config.dtype = dtype
        config.dtype = dtype
        config.rope_scaling = None
        config.num_hidden_layers = int(draft_num_layers)

        self.dtype = dtype
        self.target_model = TargetModelProxy(target_model)
        self.draft_model = DraftModel(config)
        self.embed_tokens = resolve_embed_tokens(target_model)
        self.lm_head = resolve_lm_head(target_model)
        self.norm = resolve_attr(target_model, ("model", "norm")) or resolve_attr(
            target_model, ("base_model", "model", "model", "norm")
        )
        self.load_model(draft_model_path)

    @property
    def device(self) -> torch.device:
        return self.target_model.device

    def load_model(self, load_path: str | Path) -> None:
        state = torch.load(str(load_path), map_location="cpu")
        if isinstance(state, dict) and "draft_model" in state:
            state = state["draft_model"]
        self.draft_model.load_state_dict(state)

    def forward(
        self,
        hidden_states,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
    ):
        # Use target embeddings as fixed inputs, exactly like FastGRPO's Model.
        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids)
        return self.draft_model(
            hidden_states,
            inputs_embeds,
            attention_mask,
            position_ids,
            past_key_values,
            use_cache,
        )


@dataclass
class FastGRPOSpeculativeStats:
    calls: int = 0
    fallback_calls: int = 0
    generated_tokens: int = 0
    accepted_tokens: float = 0.0
    decoded_tokens: float = 0.0
    total_seconds: float = 0.0
    target_seconds: float = 0.0
    draft_seconds: float = 0.0
    check_seconds: float = 0.0
    draft_train_seconds: float = 0.0
    draft_train_calls: int = 0
    draft_train_sequences: int = 0
    draft_optimizer_steps: int = 0
    draft_feature_loss_sum: float = 0.0
    draft_logit_loss_sum: float = 0.0

    def as_dict(self) -> dict[str, float]:
        accept_rate = self.accepted_tokens / max(1.0, self.decoded_tokens)
        train_calls = max(1, self.draft_train_calls)
        return {
            "speculative/calls": float(self.calls),
            "speculative/fallback_calls": float(self.fallback_calls),
            "speculative/generated_tokens": float(self.generated_tokens),
            "speculative/accept_rate": float(accept_rate),
            "speculative/seconds": float(self.total_seconds),
            "speculative/target_seconds": float(self.target_seconds),
            "speculative/draft_seconds": float(self.draft_seconds),
            "speculative/check_seconds": float(self.check_seconds),
            "speculative/draft_train_seconds": float(self.draft_train_seconds),
            "speculative/draft_train_calls": float(self.draft_train_calls),
            "speculative/draft_train_sequences": float(self.draft_train_sequences),
            "speculative/draft_optimizer_steps": float(self.draft_optimizer_steps),
            "speculative/draft_feature_loss": float(self.draft_feature_loss_sum / train_calls),
            "speculative/draft_logit_loss": float(self.draft_logit_loss_sum / train_calls),
        }

    def reset(self) -> None:
        self.calls = 0
        self.fallback_calls = 0
        self.generated_tokens = 0
        self.accepted_tokens = 0.0
        self.decoded_tokens = 0.0
        self.total_seconds = 0.0
        self.target_seconds = 0.0
        self.draft_seconds = 0.0
        self.check_seconds = 0.0
        self.draft_train_seconds = 0.0
        self.draft_train_calls = 0
        self.draft_train_sequences = 0
        self.draft_optimizer_steps = 0
        self.draft_feature_loss_sum = 0.0
        self.draft_logit_loss_sum = 0.0


class FastGRPOSpeculativeEngine:
    """Runtime EAGLE speculative generator used by SER rollout collection."""

    def __init__(self, target_model, tokenizer, cfg: dict[str, Any]):
        self.cfg = dict(cfg or {})
        draft_path = str(self.cfg.get("draft_model_path") or "").strip()
        if not draft_path:
            raise ValueError("speculative.draft_model_path is required when speculative.enabled=true.")
        self.tokenizer = tokenizer
        self.wrapper = FastGRPODraftWrapper(
            target_model,
            draft_model_path=draft_path,
            draft_num_layers=int(self.cfg.get("draft_num_layers", 1)),
        ).to(next(target_model.parameters()).device)
        self.wrapper.draft_model.eval()
        self.train_draft = bool(self.cfg.get("train_draft", False))
        for param in self.wrapper.draft_model.parameters():
            param.requires_grad_(self.train_draft)
        self.optimizer = None
        self.draft_accumulated_batches = 0
        self.draft_optimizer_steps = 0
        if self.train_draft:
            self.optimizer = torch.optim.AdamW(
                self.wrapper.draft_model.parameters(),
                lr=float(self.cfg.get("draft_lr", 1.0e-4)),
                betas=(0.9, 0.95),
                weight_decay=float(self.cfg.get("draft_weight_decay", 0.0)),
            )
        self.stats = FastGRPOSpeculativeStats()

    def save_checkpoint(self, output_dir: str | Path) -> None:
        if not self.train_draft:
            return
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

    def load_checkpoint(self, checkpoint_dir: str | Path) -> None:
        path = Path(checkpoint_dir) / "speculative.pt"
        if not path.exists():
            return
        state = torch.load(path, map_location="cpu")
        draft_state = state.get("draft_model")
        if draft_state is not None:
            self.wrapper.draft_model.load_state_dict(draft_state)
        if self.optimizer is not None and state.get("draft_optimizer") is not None:
            self.optimizer.load_state_dict(state["draft_optimizer"])
        self.draft_accumulated_batches = int(state.get("draft_accumulated_batches", 0))
        self.draft_optimizer_steps = int(state.get("draft_optimizer_steps", 0))
        print(f"Loaded speculative draft state from {path}")

    def should_fallback(self, batch_size: int) -> bool:
        if not torch.cuda.is_available():
            return True
        
        fallback_batch_size = int(self.cfg.get("fallback_batch_size", 0) or 0)
        if fallback_batch_size > 0 and batch_size >= fallback_batch_size:
            return True
        
        verification_num = min(
            math.floor(float(self.cfg.get("verification_capacity", 160)) / max(1, batch_size)),
            int(self.cfg.get("max_verification_num", 160)),
        )

        if verification_num <= 1:
            return True
        
        draft_token_length_c = float(self.cfg.get("draft_token_length_c", 0.75))
        if draft_token_length_c <= 0:
            return True

        max_draft_token_length = int(self.cfg.get("max_draft_token_length", 5))
        min_draft_token_length = int(self.cfg.get("min_draft_token_length", 3))

        draft_token_length = min(math.floor(math.log2(verification_num/draft_token_length_c)), max_draft_token_length)
        
        return draft_token_length < min_draft_token_length

    def _draft_training_examples(
        self,
        output: dict[str, Any],
        prefix_lengths: list[int],
    ) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
        input_ids = output.get("all_draft_input_ids")
        hidden_states = output.get("all_draft_input_states")
        if not input_ids or not hidden_states:
            return []

        examples: list[tuple[torch.Tensor, torch.Tensor, int]] = []
        for ids, states, prefix_len in zip(input_ids, hidden_states, prefix_lengths):
            if ids is None or states is None:
                continue
            ids = ids.detach().long()
            states = states.detach().to(self.wrapper.dtype)
            seq_len = int(ids.shape[-1])
            if seq_len < 3 or int(states.shape[-2]) != seq_len:
                continue
            prefix_len = min(max(0, int(prefix_len)), seq_len - 1)
            if seq_len - prefix_len <= 1:
                continue
            examples.append((ids, states, prefix_len))

        examples.sort(key=lambda item: int(item[0].shape[-1]))
        max_sequences = int(self.cfg.get("draft_train_max_sequences", 8) or 0)
        if max_sequences > 0:
            examples = examples[:max_sequences]
        return examples

    def _draft_training_chunks(
        self,
        examples: list[tuple[torch.Tensor, torch.Tensor, int]],
    ) -> list[list[tuple[torch.Tensor, torch.Tensor, int]]]:
        max_tokens = int(self.cfg.get("draft_train_max_tokens", 4096))
        max_padding_gap = int(self.cfg.get("draft_train_max_padding_gap", 1024))
        chunks: list[list[tuple[torch.Tensor, torch.Tensor, int]]] = []
        current: list[tuple[torch.Tensor, torch.Tensor, int]] = []
        current_max_len = 0
        current_token_count = 0
        for item in examples:
            seq_len = int(item[0].shape[-1])
            candidate_max_len = max(current_max_len, seq_len)
            candidate_count = current_token_count + seq_len
            can_add = not current or (
                candidate_max_len * (len(current) + 1) <= max_tokens
                and candidate_max_len * (len(current) + 1) - candidate_count <= max_padding_gap
            )
            if not can_add:
                chunks.append(current)
                current = []
                current_max_len = 0
                current_token_count = 0
            current.append(item)
            current_max_len = max(current_max_len, seq_len)
            current_token_count += seq_len
        if current:
            chunks.append(current)
        return chunks

    def _draft_loss_for_chunk(
        self,
        chunk: list[tuple[torch.Tensor, torch.Tensor, int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.wrapper.device
        max_len = max(int(item[0].shape[-1]) for item in chunk)
        hidden_size = int(chunk[0][1].shape[-1])
        input_ids, hidden_states, attention_mask, loss_mask = [], [], [], []
        for ids, states, prefix_len in chunk:
            ids = ids.to(device)
            states = states.to(device)
            cur_len = int(ids.shape[-1])
            pad_len = max_len - cur_len
            input_ids.append(F.pad(ids, (0, pad_len), value=0))
            hidden_states.append(
                torch.cat(
                    [
                        states,
                        torch.zeros((pad_len, hidden_size), dtype=states.dtype, device=device),
                    ],
                    dim=0,
                )
            )
            attention_mask.append([1] * cur_len + [0] * pad_len)
            loss_mask.append([0] * int(prefix_len) + [1] * (cur_len - int(prefix_len)) + [0] * pad_len)

        input_ids_t = torch.stack(input_ids, dim=0).long()
        hidden_states_t = torch.stack(hidden_states, dim=0)
        attention_mask_t = torch.tensor(attention_mask, device=device, dtype=torch.long)
        loss_mask_t = torch.tensor(loss_mask, device=device, dtype=torch.float32)

        draft_outputs = self.wrapper(
            hidden_states=hidden_states_t,
            input_ids=input_ids_t,
            attention_mask=attention_mask_t,
            use_cache=False,
        )
        next_feature_states = draft_outputs["next_feature_states"]
        shifted_mask = loss_mask_t[:, :-1]
        denom = shifted_mask.sum(dim=-1).clamp_min(1.0)
        feature_loss = F.smooth_l1_loss(
            next_feature_states[:, :-1, :].float(),
            hidden_states_t[:, 1:, :].float(),
            reduction="none",
        ).mean(dim=-1)
        feature_loss = ((feature_loss * shifted_mask).sum(dim=-1) / denom).sum()

        logit_weight = float(self.cfg.get("logit_loss_weight", 0.0))
        if logit_weight <= 0:
            return feature_loss, feature_loss.new_zeros(())

        draft_hidden_states = draft_outputs["hidden_states"].to(self.wrapper.target_model.dtype)
        draft_logits = frozen_lm_head(self.wrapper.lm_head, draft_hidden_states[:, :-1, :])
        with torch.no_grad():
            target_logits = frozen_lm_head(
                self.wrapper.lm_head,
                hidden_states_t[:, 1:, :].to(self.wrapper.target_model.dtype),
            )
            target_probs = target_logits.float().softmax(dim=-1)
        log_probs = draft_logits.float().log_softmax(dim=-1)
        logit_loss = -(target_probs * log_probs).sum(dim=-1)
        logit_loss = ((logit_loss * shifted_mask).sum(dim=-1) / denom).sum()
        return feature_loss, logit_loss

    def train_draft_from_output(self, output: dict[str, Any], prefix_lengths: list[int]) -> None:
        if not self.train_draft or self.optimizer is None:
            return
        examples = self._draft_training_examples(output, prefix_lengths)
        if not examples:
            return

        start_time = time.time()
        was_training = self.wrapper.draft_model.training
        self.wrapper.draft_model.train()
        feature_weight = float(self.cfg.get("feature_loss_weight", 2.0))
        logit_weight = float(self.cfg.get("logit_loss_weight", 0.0))
        accumulation_steps = max(1, int(self.cfg.get("draft_accumulation_steps", 1)))
        total_feature = 0.0
        total_logit = 0.0
        did_backward = False

        for chunk in self._draft_training_chunks(examples):
            feature_loss, logit_loss = self._draft_loss_for_chunk(chunk)
            loss = feature_weight * feature_loss + logit_weight * logit_loss
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                self.optimizer.zero_grad(set_to_none=True)
                continue
            total_feature += float(feature_loss.detach().item())
            total_logit += float(logit_loss.detach().item())
            loss = loss / max(1, len(examples)) / accumulation_steps
            loss.backward()
            did_backward = True

        if did_backward:
            self.draft_accumulated_batches += 1
            if self.draft_accumulated_batches % accumulation_steps == 0:
                max_grad_norm = float(self.cfg.get("draft_max_grad_norm", 1.0))
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.wrapper.draft_model.parameters(), max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.draft_optimizer_steps += 1
                self.stats.draft_optimizer_steps += 1

        if not was_training:
            self.wrapper.draft_model.eval()
        self.stats.draft_train_seconds += float(time.time() - start_time)
        self.stats.draft_train_calls += 1
        self.stats.draft_train_sequences += len(examples)
        self.stats.draft_feature_loss_sum += total_feature / max(1, len(examples))
        self.stats.draft_logit_loss_sum += total_logit / max(1, len(examples))

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
        self.stats.calls += 1
        if self.should_fallback(len(token_lists)):
            self.stats.fallback_calls += 1
            outputs = normal_generate(
                self.wrapper.target_model,
                token_lists,
                max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            self.stats.generated_tokens += sum(max(0, len(output) - len(tokens)) for output, tokens in zip(outputs, token_lists))
            return outputs

        input_tensor, attention_mask, pad_lengths = pad_left(token_lists, pad_token_id, self.wrapper.device)
        max_total_length = input_tensor.shape[-1] + int(max_new_tokens)
        prefix_lengths = [len(tokens) for tokens in token_lists]
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
                statistical_time=bool(self.cfg.get("profile_timing", False)),
                transfer_workers=int(self.cfg.get("transfer_workers", 1)),
                return_all_draft_input=self.train_draft,
                max_length=max_total_length,
            )

        generated_ids = output.get("generated_token_ids") or [[] for _ in token_lists]
        results: list[list[int]] = []
        generated_count = 0
        for tokens, generated in zip(token_lists, generated_ids):
            new_tokens = [int(token) for token in list(generated)[:max_new_tokens]]
            if eos_token_id is not None and eos_token_id in new_tokens:
                new_tokens = new_tokens[: new_tokens.index(eos_token_id) + 1]
            generated_count += len(new_tokens)
            results.append(list(tokens) + new_tokens)

        self.stats.generated_tokens += generated_count
        self.stats.accepted_tokens += float(output.get("total_acc_length", 0.0) or 0.0)
        self.stats.decoded_tokens += float(output.get("total_decoded_token_num", 0.0) or 0.0)
        self.stats.total_seconds += float(time.time() - start_time)
        self.stats.target_seconds += float(output.get("target_time_cost", 0.0) or 0.0)
        self.stats.draft_seconds += float(output.get("draft_time_cost", 0.0) or 0.0)
        self.stats.check_seconds += float(output.get("check_time_cost", 0.0) or 0.0)
        self.train_draft_from_output(output, prefix_lengths)
        del input_tensor, attention_mask, pad_lengths
        return results

    def pop_log_stats(self) -> dict[str, float]:
        stats = self.stats.as_dict()
        self.stats.reset()
        return stats


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


def normal_generate(
    model,
    token_lists: list[list[int]],
    max_new_tokens: int,
    *,
    temperature: float,
    top_p: float,
    pad_token_id: int,
    eos_token_id: int | None,
) -> list[list[int]]:
    input_tensor, attention_mask, pad_lengths = pad_left(token_lists, pad_token_id, model.device)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_tensor,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
    results = []
    for output, pad_len in zip(outputs, pad_lengths):
        results.append([int(token) for token in output.tolist()[pad_len:]])
    del input_tensor, attention_mask, outputs
    return results


def build_fastgrpo_speculative_engine(args, model, tokenizer) -> FastGRPOSpeculativeEngine | None:
    cfg = dict(getattr(args, "speculative", {}) or {})
    if not bool(cfg.get("enabled", False)):
        return None
    engine = FastGRPOSpeculativeEngine(model, tokenizer, cfg)
    resume_from = str(getattr(args, "resume_from_checkpoint", "") or "").strip()
    if resume_from:
        engine.load_checkpoint(resume_from)
    return engine
