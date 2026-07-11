"""Thin adapter from SER rollouts to the bundled FastGRPO EAGLE generator.

The heavy speculative decoding logic lives in
``FastGRPO.helper.specualtive_generate``.  This module only adapts the SER
target model to the small interface that FastGRPO expects:

    wrapper.target_model  -> current SER policy
    wrapper.draft_model   -> pretrained EAGLE draft model
    wrapper(...)          -> draft forward pass

The adapter intentionally does not train the draft model.  It assumes the draft
was pretrained by ``SER-method/FastGRPO/train_draft.py`` and loaded from a
``.pth`` file containing ``{"draft_model": state_dict}``.
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
        dtype = getattr(config, "torch_dtype", None) or getattr(config, "dtype", None)
        if dtype is None or isinstance(dtype, str):
            dtype = next(target_model.parameters()).dtype
        config.torch_dtype = dtype
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

    def as_dict(self) -> dict[str, float]:
        accept_rate = self.accepted_tokens / max(1.0, self.decoded_tokens)
        return {
            "speculative/calls": float(self.calls),
            "speculative/fallback_calls": float(self.fallback_calls),
            "speculative/generated_tokens": float(self.generated_tokens),
            "speculative/accept_rate": float(accept_rate),
            "speculative/seconds": float(self.total_seconds),
            "speculative/target_seconds": float(self.target_seconds),
            "speculative/draft_seconds": float(self.draft_seconds),
            "speculative/check_seconds": float(self.check_seconds),
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
        for param in self.wrapper.draft_model.parameters():
            param.requires_grad_(False)
        self.stats = FastGRPOSpeculativeStats()

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
                return_all_draft_input=False,
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
    return FastGRPOSpeculativeEngine(model, tokenizer, cfg)
