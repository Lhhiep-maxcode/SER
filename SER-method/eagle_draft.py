"""EAGLE draft model used by SER speculative rollouts.

The draft model follows the EAGLE architecture idea:

    feature h_t + target embedding(token_t) -> predicted feature h'_{t+1}

The target embedding table and LM head are reused so the draft model predicts
in the same representation space as the target policy.  During tree expansion,
the next draft step is fed the *predicted* feature from the previous step; target
features are used again only after target verification anchors an accepted path.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
) -> torch.Tensor:
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, dtype=dtype, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    if past_key_values_length > 0:
        prefix = torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device)
        mask = torch.cat([prefix, mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None) -> torch.Tensor:
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids):
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class DraftRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class DraftAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)

        dtype = config.torch_dtype
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False, dtype=dtype)
        self.rotary_emb = DraftRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        present = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output), present


class DraftMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        dtype = config.torch_dtype
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False, dtype=dtype)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class EagleFeatureFusion(nn.Module):
    """Fuse target feature and token embedding before draft self-attention.

    This is the EAGLE feature step.  The feature side carries h_t, while the
    embedding side carries token_t in the target model's own embedding space.
    """

    def __init__(self, config):
        super().__init__()
        dtype = config.torch_dtype
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False, dtype=dtype)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(token_embeds)) * self.up_proj(hidden_states))


class DraftRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class DraftDecoderLayer(nn.Module):
    def __init__(self, config, last: bool):
        super().__init__()
        self.last = last
        self.input_layernorm = DraftRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = DraftAttention(config)
        if not last:
            self.post_attention_layernorm = DraftRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.mlp = DraftMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        if not self.last:
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = residual + self.mlp(hidden_states)
        return hidden_states, present


class EagleDraftModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.fs = EagleFeatureFusion(config)
        self.layers = nn.ModuleList(
            [DraftDecoderLayer(config, layer_idx == config.num_hidden_layers - 1) for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_norm = DraftRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.states_last_norm = DraftRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.logits_last_norm = DraftRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.states_mlp = DraftMLP(config)
        self.logits_mlp = DraftMLP(config)

    def _prepare_decoder_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_shape: tuple[int, int],
        hidden_states: torch.Tensor,
        past_key_values_length: int,
    ) -> Optional[torch.Tensor]:
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                hidden_states.dtype,
                device=hidden_states.device,
                past_key_values_length=past_key_values_length,
            )
        if attention_mask is not None:
            expanded = _expand_mask(attention_mask, hidden_states.dtype, tgt_len=input_shape[-1]).to(hidden_states.device)
            combined_attention_mask = expanded if combined_attention_mask is None else expanded + combined_attention_mask
        return combined_attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
    ) -> dict[str, Any]:
        bsz, seq_len, _ = hidden_states.shape
        past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        seq_len_with_past = seq_len + past_len
        if position_ids is None:
            position_ids = torch.arange(past_len, seq_len + past_len, dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_len)
        else:
            position_ids = position_ids.view(-1, seq_len).long()
        if attention_mask is None:
            attention_mask = torch.ones((bsz, seq_len_with_past), dtype=torch.bool, device=hidden_states.device)
        if attention_mask.dim() != 4:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask,
                (bsz, seq_len),
                hidden_states,
                past_len,
            )

        hidden_states = hidden_states.to(self.dtype)
        token_embeds = token_embeds.to(self.dtype)

        # EAGLE fusion: combine h_t and embedding(token_t) before the draft
        # transformer.  For deeper draft-tree nodes, h_t is the draft-predicted
        # feature from the previous step, not a target feature.
        residual = hidden_states
        hidden_states = residual + self.fs(hidden_states, token_embeds)

        cache = []
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            hidden_states, present = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=bool(use_cache),
            )
            if use_cache:
                cache.append(present)

        residual = hidden_states
        hidden_states = self.post_norm(hidden_states)
        next_feature_states = self.states_mlp(hidden_states) + residual
        draft_hidden_states = self.logits_mlp(hidden_states) + residual
        next_feature_states = self.states_last_norm(next_feature_states)
        draft_hidden_states = self.logits_last_norm(draft_hidden_states)
        return {
            "hidden_states": draft_hidden_states,
            "past_key_values": cache,
            "next_feature_states": next_feature_states,
        }


class DetachedLMHead(nn.Module):
    """Reuse target LM head while preventing draft loss from updating it."""

    def __init__(self, lm_head: nn.Module):
        super().__init__()
        self.lm_head = lm_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if isinstance(self.lm_head, nn.Linear):
            bias = self.lm_head.bias.detach() if self.lm_head.bias is not None else None
            return F.linear(hidden_states, self.lm_head.weight.detach(), bias)

        old_flags = [param.requires_grad for param in self.lm_head.parameters()]
        try:
            for param in self.lm_head.parameters():
                param.requires_grad_(False)
            return self.lm_head(hidden_states)
        finally:
            for param, flag in zip(self.lm_head.parameters(), old_flags):
                param.requires_grad_(flag)


class TargetModelView(nn.Module):
    """Delegate to the SER target policy while exposing device/dtype properties."""

    def __init__(self, target_model):
        super().__init__()
        self.target = target_model

    @property
    def device(self) -> torch.device:
        return next(self.target.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.target.parameters()).dtype

    def forward(self, *args, **kwargs):
        return self.target(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.target, name)


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
        ("transformer", "wte"),
    ):
        module = resolve_attr(target_model, path)
        if module is not None:
            return module
    raise AttributeError("Could not locate target embedding layer for EAGLE draft model.")


def resolve_lm_head(target_model) -> nn.Module:
    for path in (
        ("lm_head",),
        ("base_model", "model", "lm_head"),
    ):
        module = resolve_attr(target_model, path)
        if module is not None:
            return module
    raise AttributeError("Could not locate target LM head for EAGLE draft model.")


class EagleDraftWrapper(nn.Module):
    """Wrap the draft model with references to target embedding and LM head."""

    def __init__(self, target_model, draft_layers: int = 1, adapter_path: str | None = None):
        super().__init__()
        config = deepcopy(target_model.config)
        dtype = getattr(config, "torch_dtype", None)
        if dtype is None or isinstance(dtype, str):
            dtype = next(target_model.parameters()).dtype
        config.torch_dtype = dtype
        config.num_hidden_layers = int(draft_layers)
        config.rope_scaling = None
        self.dtype = dtype
        self.target_model = TargetModelView(target_model)
        self.draft_model = EagleDraftModel(config)
        self.embed_tokens = resolve_embed_tokens(target_model)
        self.target_lm_head = resolve_lm_head(target_model)
        self.lm_head = DetachedLMHead(self.target_lm_head)
        if adapter_path:
            self.load_model(adapter_path)

    @property
    def device(self) -> torch.device:
        return next(self.target_model.parameters()).device

    def load_model(self, load_path: str | Path) -> None:
        state = torch.load(str(load_path), map_location="cpu")
        if isinstance(state, dict) and "draft_model" in state:
            state = state["draft_model"]
        self.draft_model.load_state_dict(state)

    def save_model(self, save_path: str | Path) -> None:
        torch.save({"draft_model": self.draft_model.state_dict()}, str(save_path))

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
    ) -> dict[str, torch.Tensor]:
        # Token embeddings are reused from the target model, but they are inputs
        # to the draft model rather than trainable draft parameters.
        with torch.no_grad():
            token_embeds = self.embed_tokens(input_ids)
        return self.draft_model(
            hidden_states=hidden_states,
            token_embeds=token_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
