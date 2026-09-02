# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""B200-only MiniMax-H3 SubBlock attention backed by FlashInfer blk64 BSA."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import regex as re
import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    PackedPaddingMetadata,
)
from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl
from vllm_omni.diffusion.attention.backends.subblock import (
    SubBlockRouter,
    run_bsa_attn_blk64,
    validate_b200_bsa_available,
)
from vllm_omni.diffusion.attention.backends.trtllm_attn import HAS_FLASHINFER, TrtllmAttentionImpl
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available

logger = init_logger(__name__)

_HEAD_DIM = 128
_DIT_LAYER_PREFIX = re.compile(r"(?:^|\.)blocks\.(\d+)\.")
_PACKED_KEYS = ("cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k")


def _dit_layer_index(prefix: str, role: str) -> int | None:
    if role != "self":
        return None
    match = _DIT_LAYER_PREFIX.search(prefix)
    return int(match.group(1)) if match else None


def _integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        comparator = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"SUBBLOCK_ATTN {name} must be a {comparator} integer, got {value!r}")
    return value


@dataclass(frozen=True)
class SubBlockSchedule:
    sparsity: float = 0.75
    skip_first_steps: int = 10
    skip_first_layers: int = 0
    n_q: int = 4
    n_k: int = 4
    min_seq_len: int = 24576

    @classmethod
    def from_backend_kwargs(cls, backend_kwargs: dict[str, Any] | None) -> SubBlockSchedule:
        values = backend_kwargs or {}
        raw_sparsity = values.get("sparsity", 0.75)
        if isinstance(raw_sparsity, bool):
            raise ValueError(f"SUBBLOCK_ATTN sparsity must be numeric, not boolean; got {raw_sparsity!r}")
        try:
            sparsity = float(raw_sparsity)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"SUBBLOCK_ATTN sparsity must be numeric, got {raw_sparsity!r}") from exc
        if not math.isfinite(sparsity) or not 0.0 <= sparsity < 1.0:
            raise ValueError(f"SUBBLOCK_ATTN sparsity must be finite and in [0, 1), got {sparsity!r}")
        schedule = cls(
            sparsity=sparsity,
            skip_first_steps=_integer(values.get("skip_first_steps", 10), "skip_first_steps", minimum=0),
            skip_first_layers=_integer(values.get("skip_first_layers", 0), "skip_first_layers", minimum=0),
            n_q=_integer(values.get("n_q", 4), "n_q", minimum=1),
            n_k=_integer(values.get("n_k", 4), "n_k", minimum=1),
            min_seq_len=_integer(values.get("min_seq_len", 24576), "min_seq_len", minimum=1),
        )
        if schedule.n_q not in (1, 2, 4, 8) or schedule.n_k not in (1, 2, 4, 8):
            raise ValueError(
                f"SUBBLOCK_ATTN n_q/n_k must be one of (1, 2, 4, 8), got n_q={schedule.n_q}, n_k={schedule.n_k}"
            )
        return schedule


class SubBlockAttentionBackend(AttentionBackend):
    """Explicit backend; never selected as a platform default."""

    supported_platforms = ("cuda",)

    @classmethod
    def supports_packed_mask_free(cls) -> bool:
        return True

    @classmethod
    def supports_multi_doc_packed_varlen(cls) -> bool:
        # V1 intentionally handles only H3's one-real-document plus padding
        # contract. Batched documents need one independent BSA plan per doc.
        return False

    @classmethod
    def validate_available(cls) -> None:
        validate_b200_bsa_available()
        if not HAS_FLASHINFER:
            raise ImportError(
                "SUBBLOCK_ATTN uses TRTLLM_ATTN for early-step/short-sequence dense fallback, but this "
                "FlashInfer build lacks flashinfer.prefill.trtllm_ragged_attention_deepseek."
            )

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [_HEAD_DIM]

    @staticmethod
    def get_name() -> str:
        return "SUBBLOCK_ATTN"

    @staticmethod
    def get_impl_cls() -> type[SubBlockAttentionImpl]:
        return SubBlockAttentionImpl


class SubBlockAttentionImpl(AttentionImpl):
    """Route eligible H3 DiT self-attention calls to FlashInfer BSA."""

    subblock_configured = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        qkv_layout: str | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        role: str = "self",
        **extra_impl_args: Any,
    ) -> None:
        del extra_impl_args
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.prefix = prefix
        self.role = role
        self.qkv_layout = qkv_layout

        if head_size != _HEAD_DIM:
            raise ValueError(f"SUBBLOCK_ATTN requires head_dim={_HEAD_DIM}, got {head_size}")
        if self.num_heads != self.num_kv_heads:
            raise ValueError(
                f"SUBBLOCK_ATTN V1 supports MHA only; got num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}"
            )
        if causal:
            raise ValueError("SUBBLOCK_ATTN does not support causal attention")
        if qkv_layout != "BSND":
            raise ValueError(f"SUBBLOCK_ATTN requires qkv_layout='BSND', got {qkv_layout!r}")

        self.schedule = SubBlockSchedule.from_backend_kwargs(backend_kwargs)
        self.layer_idx = _dit_layer_index(prefix, role)
        self.layer_enabled = (
            self.layer_idx is not None
            and self.layer_idx >= self.schedule.skip_first_layers
            and self.schedule.sparsity > 0.0
        )
        self.router = SubBlockRouter(n_q=self.schedule.n_q, n_k=self.schedule.n_k)
        self._dense_impl = TrtllmAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=self.num_kv_heads,
            prefix=f"{prefix}.subblock_dense",
            qkv_layout=qkv_layout,
            role=role,
        )
        self._masked_dense_impl = SDPAImpl(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=self.num_kv_heads,
            prefix=f"{prefix}.subblock_masked_dense",
            qkv_layout=qkv_layout,
        )
        self._sparse_calls = 0
        self._dense_calls: Counter[str] = Counter()
        self._last_plan_density: float | None = None

        if self.layer_enabled:
            logger.info_once(
                "SUBBLOCK_ATTN configured: sparsity=%.3f, n_q=%d, n_k=%d, first %d denoise steps dense, "
                "first %d DiT layers dense, min_seq_len=%d",
                self.schedule.sparsity,
                self.schedule.n_q,
                self.schedule.n_k,
                self.schedule.skip_first_steps,
                self.schedule.skip_first_layers,
                self.schedule.min_seq_len,
            )

    def runtime_stats(self) -> dict[str, Any]:
        """Counters used to prove that an integration benchmark ran BSA."""
        return {
            "actual_sparse_calls": self._sparse_calls,
            "dense_calls_by_reason": dict(self._dense_calls),
            "last_plan_density": self._last_plan_density,
        }

    def _step_index(self) -> int | None:
        if not is_forward_context_available():
            return None
        return getattr(get_forward_context(), "denoise_step_idx", None)

    def _dense_reason(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor | None = None) -> str | None:
        if not self.layer_enabled:
            if self.schedule.sparsity == 0.0:
                return "zero_sparsity"
            if self.layer_idx is None:
                return "non_dit_role"
            return "skipped_layer"
        step_idx = self._step_index()
        if step_idx is None:
            return "missing_denoise_step"
        if step_idx < self.schedule.skip_first_steps:
            return "warmup_step"
        if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or (v is not None and v.dtype != torch.bfloat16):
            return "non_bf16"
        if q.shape[0] != 1 or k.shape[0] != 1:
            return "multi_batch"
        if k.shape[1] < self.schedule.min_seq_len:
            return "short_sequence"
        return None

    @staticmethod
    def _validate_packed_padding(
        query: torch.Tensor,
        key: torch.Tensor,
        packed_padding: PackedPaddingMetadata,
        extra: dict[str, Any],
    ) -> tuple[int, int]:
        lengths = (
            packed_padding.q_length,
            packed_padding.kv_length,
            extra["max_seqlen_q"],
            extra["max_seqlen_k"],
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in lengths):
            raise ValueError("Mask-free packed SUBBLOCK_ATTN lengths must be Python integers")
        q_len, kv_len, max_q_len, max_kv_len = lengths
        if query.shape[0] != 1 or key.shape[0] != 1:
            raise ValueError("Mask-free packed SUBBLOCK_ATTN requires a single physical batch")
        if not 0 < q_len <= query.shape[1] or not 0 < kv_len <= key.shape[1]:
            raise ValueError("PackedPaddingMetadata lengths must be within the physical Q/K sequences")
        if q_len != kv_len:
            raise ValueError("SUBBLOCK_ATTN V1 requires equal packed Q and KV lengths")
        if max_q_len != q_len or max_kv_len != kv_len:
            raise ValueError("Mask-free packed SUBBLOCK_ATTN lengths must match max_seqlen_q/k")
        published_kv_len = extra.get("valid_kv_length")
        if published_kv_len is not None and published_kv_len != kv_len:
            raise ValueError("PackedPaddingMetadata.kv_length must match valid_kv_length")
        cu_q = packed_padding.cu_seqlens_q
        cu_k = packed_padding.cu_seqlens_k
        if cu_q.dtype != torch.int32 or cu_k.dtype != torch.int32:
            raise ValueError("Mask-free packed SUBBLOCK_ATTN requires int32 cu_seqlens")
        if cu_q.device != query.device or cu_k.device != key.device:
            raise ValueError("Mask-free packed SUBBLOCK_ATTN metadata must be on the Q/K device")
        if cu_q.shape != (2,) or cu_k.shape != (2,):
            raise ValueError("Mask-free packed SUBBLOCK_ATTN requires canonical two-element cu_seqlens")
        return q_len, kv_len

    def _dense_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
        reason: str,
    ) -> torch.Tensor:
        self._dense_calls[reason] += 1
        if attn_metadata is not None and attn_metadata.attn_mask is not None:
            return self._masked_dense_impl.forward_cuda(query, key, value, attn_metadata)
        return self._dense_impl.forward_cuda(query, key, value, attn_metadata)

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        extra = getattr(attn_metadata, "extra", {}) if attn_metadata is not None else {}
        present_packed_keys = [name for name in _PACKED_KEYS if name in extra]
        if present_packed_keys and len(present_packed_keys) != len(_PACKED_KEYS):
            missing = sorted(set(_PACKED_KEYS) - set(present_packed_keys))
            raise ValueError(f"Incomplete packed SUBBLOCK_ATTN metadata; missing {missing}")
        has_packed_metadata = len(present_packed_keys) == len(_PACKED_KEYS)
        packed_padding = getattr(attn_metadata, "packed_padding", None) if attn_metadata is not None else None
        if packed_padding is not None and not isinstance(packed_padding, PackedPaddingMetadata):
            raise ValueError("packed_padding must be PackedPaddingMetadata")
        if packed_padding is not None and not has_packed_metadata:
            raise ValueError("PackedPaddingMetadata requires complete packed SUBBLOCK_ATTN metadata")

        if attn_metadata is not None and attn_metadata.attn_mask is not None:
            return self._dense_forward(query, key, value, attn_metadata, "attention_mask")
        if has_packed_metadata and packed_padding is None:
            return self._dense_forward(query, key, value, attn_metadata, "multi_doc_packed")

        active_q = query
        active_k = key
        active_v = value
        active_q_len = query.shape[1]
        if packed_padding is not None:
            active_q_len, active_kv_len = self._validate_packed_padding(query, key, packed_padding, extra)
            active_q = query[:, :active_q_len]
            active_k = key[:, :active_kv_len]
            active_v = value[:, :active_kv_len]

        reason = self._dense_reason(active_q, active_k, active_v)
        if reason is not None:
            return self._dense_forward(query, key, value, attn_metadata, reason)

        plan = self.router.route(
            active_q,
            active_k,
            sparsity=self.schedule.sparsity,
            softmax_scale=self.softmax_scale,
        )
        sparse_out = run_bsa_attn_blk64(
            active_q,
            active_k,
            active_v,
            plan.index,
            plan.topk,
            self.softmax_scale,
        )
        self._sparse_calls += 1
        self._last_plan_density = plan.density
        logger.info_once(
            "SUBBLOCK_ATTN BSA active: Sq=%d, Sk=%d, heads=%d, kept_blocks=%d/%d, realized_sparsity=%.4f",
            active_q.shape[1],
            active_k.shape[1],
            active_q.shape[2],
            plan.topk,
            plan.num_blocks,
            1.0 - plan.density,
        )
        if active_q_len == query.shape[1]:
            return sparse_out
        out = torch.zeros_like(query)
        out[:, :active_q_len] = sparse_out
        return out


def collect_subblock_runtime_stats(module: torch.nn.Module) -> dict[str, Any]:
    """Aggregate rank-local counters from SubBlock impls below ``module``.

    ``AttentionImpl`` is intentionally not an ``nn.Module`` in vLLM-Omni, so
    walking ``module.modules()`` alone cannot see it. MiniMax-H3 wraps each
    impl in ``MiniMaxH3Attention -> Attention -> AttentionImpl``; follow that
    short chain and de-duplicate impls reached from both wrapper levels.
    """
    impls: list[SubBlockAttentionImpl] = []
    seen: set[int] = set()
    for child in module.modules():
        candidate: Any = child
        for _ in range(3):
            if isinstance(candidate, SubBlockAttentionImpl):
                if id(candidate) not in seen:
                    seen.add(id(candidate))
                    impls.append(candidate)
                break
            candidate = getattr(candidate, "attention", None)
            if candidate is None:
                break

    dense_calls: Counter[str] = Counter()
    sparse_calls = 0
    densities: list[float] = []
    for impl in impls:
        stats = impl.runtime_stats()
        sparse_calls += int(stats["actual_sparse_calls"])
        dense_calls.update(stats["dense_calls_by_reason"])
        if stats["last_plan_density"] is not None:
            densities.append(float(stats["last_plan_density"]))
    return {
        "configured_layers": len(impls),
        "sparse_eligible_layers": sum(impl.layer_enabled for impl in impls),
        "actual_sparse_calls": sparse_calls,
        "dense_calls_by_reason": dict(sorted(dense_calls.items())),
        "last_plan_density_min": min(densities) if densities else None,
        "last_plan_density_max": max(densities) if densities else None,
    }


def subblock_runtime_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Subtract two aggregate snapshots taken around one model execution."""
    reasons = set(before["dense_calls_by_reason"]) | set(after["dense_calls_by_reason"])
    dense_delta = {
        reason: max(
            0,
            int(after["dense_calls_by_reason"].get(reason, 0)) - int(before["dense_calls_by_reason"].get(reason, 0)),
        )
        for reason in sorted(reasons)
    }
    dense_delta = {reason: count for reason, count in dense_delta.items() if count}
    sparse_delta = max(0, int(after["actual_sparse_calls"]) - int(before["actual_sparse_calls"]))
    return {
        "configured_layers": int(after["configured_layers"]),
        "sparse_eligible_layers": int(after["sparse_eligible_layers"]),
        "actual_sparse_calls": sparse_delta,
        "dense_calls_by_reason": dense_delta,
        "last_plan_density_min": after["last_plan_density_min"] if sparse_delta else None,
        "last_plan_density_max": after["last_plan_density_max"] if sparse_delta else None,
    }


__all__ = [
    "SubBlockAttentionBackend",
    "SubBlockAttentionImpl",
    "SubBlockSchedule",
    "collect_subblock_runtime_stats",
    "subblock_runtime_delta",
]
