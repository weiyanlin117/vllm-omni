# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Training-free SubBlock router for FlashInfer's 64-token BSA kernel.

The score and budget contract follows SGLang PR #34148: split each 64-token
query/key block into sub-blocks, combine their mean-Q/mean-K logits with a
log-sum-exp, and retain a per-head top-k block plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as torch_functional

LOG2E = 1.4426950408889634
BLOCK_SIZE = 64
BUDGET_GRANULARITY = 8
VALID_SUBBLOCK_COUNTS = (1, 2, 4, 8)


def snap_topk_to_kernel_budget(topk: int, num_blocks: int) -> int:
    """Round up to the eight-block budget granularity charged by blk64 BSA."""
    return min(num_blocks, max(1, math.ceil(topk / BUDGET_GRANULARITY)) * BUDGET_GRANULARITY)


@dataclass(frozen=True)
class RoutingPlan:
    index: torch.Tensor
    topk: int
    num_blocks: int

    @property
    def density(self) -> float:
        return self.topk / self.num_blocks


class SubBlockRouter:
    def __init__(self, n_k: int = 4, n_q: int = 4) -> None:
        if n_k not in VALID_SUBBLOCK_COUNTS or n_q not in VALID_SUBBLOCK_COUNTS:
            raise ValueError(f"n_q/n_k must be one of {VALID_SUBBLOCK_COUNTS}, got n_q={n_q}, n_k={n_k}")
        self.n_k = n_k
        self.n_q = n_q

    @staticmethod
    def _reference_pool(x: torch.Tensor, n_cells: int, subblock_size: int, scale: float) -> torch.Tensor:
        """Portable reference pooling used by CPU numerical tests."""
        batch, seq_len, heads, head_dim = x.shape
        padded_len = n_cells * subblock_size
        if padded_len != seq_len:
            x = torch_functional.pad(x, (0, 0, 0, 0, 0, padded_len - seq_len))
        x = x.reshape(batch, n_cells, subblock_size, heads, head_dim).float()
        starts = torch.arange(n_cells, device=x.device) * subblock_size
        counts = (seq_len - starts).clamp(1, subblock_size).view(1, n_cells, 1, 1)
        pooled = x.sum(dim=2) / counts
        return (pooled * scale).permute(0, 2, 1, 3).reshape(batch * heads, n_cells, head_dim)

    @torch.no_grad()
    def reference_scores(self, q: torch.Tensor, k: torch.Tensor, softmax_scale: float) -> torch.Tensor:
        """Pure PyTorch score oracle for unit tests and non-CUDA diagnostics."""
        batch, q_len, heads, head_dim = q.shape
        k_len = k.shape[1]
        q_blocks = math.ceil(q_len / BLOCK_SIZE)
        k_blocks = math.ceil(k_len / BLOCK_SIZE)
        q_subblock = BLOCK_SIZE // self.n_q
        k_subblock = BLOCK_SIZE // self.n_k
        pooled_q = self._reference_pool(q, q_blocks * self.n_q, q_subblock, softmax_scale)
        pooled_k = self._reference_pool(k, k_blocks * self.n_k, k_subblock, 1.0)
        logits = torch.bmm(pooled_q, pooled_k.transpose(1, 2))
        logits = logits.view(batch * heads, q_blocks, self.n_q, k_blocks, self.n_k)
        q_valid = torch.arange(q_blocks * self.n_q, device=q.device) < math.ceil(q_len / q_subblock)
        k_valid = torch.arange(k_blocks * self.n_k, device=k.device) < math.ceil(k_len / k_subblock)
        valid = q_valid.view(1, q_blocks, self.n_q, 1, 1) & k_valid.view(1, 1, 1, k_blocks, self.n_k)
        logits = logits.masked_fill(~valid, -torch.inf)
        scores = torch.logsumexp(logits, dim=(2, 4))
        return scores.view(batch, heads, q_blocks, k_blocks)

    @torch.no_grad()
    def scores(self, q: torch.Tensor, k: torch.Tensor, softmax_scale: float) -> torch.Tensor:
        """Return FP32 ``[B, H, Q-blocks, K-blocks]`` routing scores."""
        if q.ndim != 4 or k.ndim != 4:
            raise ValueError("SubBlockRouter expects Q/K in [B, S, H, D] layout")
        if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
            raise ValueError("SubBlockRouter requires matching Q/K batch, head, and head-dim shapes")
        if q.shape[1] == 0 or k.shape[1] == 0:
            raise ValueError("SubBlockRouter requires non-empty Q/K sequences")
        if q.device != k.device:
            raise ValueError("SubBlockRouter requires Q/K on the same device")
        if not q.is_cuda or not k.is_cuda:
            return self.reference_scores(q, k, softmax_scale)
        if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
            raise ValueError("The fused SubBlock router requires BF16 Q/K tensors")

        from .kernels import fused_pool, fused_scores

        batch, q_len, heads, head_dim = q.shape
        k_len = k.shape[1]
        q_blocks = math.ceil(q_len / BLOCK_SIZE)
        k_blocks = math.ceil(k_len / BLOCK_SIZE)
        q_subblock = BLOCK_SIZE // self.n_q
        k_subblock = BLOCK_SIZE // self.n_k
        pooled_q = torch.empty(
            batch * heads,
            q_blocks * self.n_q,
            head_dim,
            dtype=torch.bfloat16,
            device=q.device,
        )
        pooled_k = torch.empty(
            batch * heads,
            k_blocks * self.n_k,
            head_dim,
            dtype=torch.bfloat16,
            device=k.device,
        )
        fused_pool(q, q_blocks * self.n_q, q_subblock, pooled_q, scale=softmax_scale * LOG2E)
        fused_pool(k, k_blocks * self.n_k, k_subblock, pooled_k)
        out = torch.empty(batch * heads, q_blocks, k_blocks, dtype=torch.float32, device=q.device)
        fused_scores(
            pooled_q,
            pooled_k,
            out,
            n_k=self.n_k,
            n_valid=math.ceil(k_len / k_subblock),
            n_q=self.n_q,
            m_valid=math.ceil(q_len / q_subblock),
        )
        return out.view(batch, heads, q_blocks, k_blocks)

    @torch.no_grad()
    def route(self, q: torch.Tensor, k: torch.Tensor, sparsity: float, softmax_scale: float) -> RoutingPlan:
        if not 0.0 <= sparsity < 1.0:
            raise ValueError(f"sparsity must be in [0, 1), got {sparsity!r}")
        batch, _, heads, _ = q.shape
        num_blocks = math.ceil(k.shape[1] / BLOCK_SIZE)
        scores = self.scores(q, k, softmax_scale)
        q_blocks = scores.shape[2]
        topk = snap_topk_to_kernel_budget(math.ceil((1.0 - sparsity) * num_blocks), num_blocks)
        scores_2d = scores.reshape(-1, num_blocks).contiguous()
        if scores.is_cuda:
            from .kernels import fused_topk

            index = fused_topk(scores_2d, topk)
        else:
            index = torch.topk(scores_2d, topk, dim=-1, sorted=False).indices.to(torch.int32)
        return RoutingPlan(index=index.view(batch, heads, q_blocks, topk), topk=topk, num_blocks=num_blocks)

    @staticmethod
    def block_sizes(seq_len: int, device: torch.device) -> torch.Tensor:
        num_blocks = math.ceil(seq_len / BLOCK_SIZE)
        starts = torch.arange(num_blocks, device=device, dtype=torch.int32) * BLOCK_SIZE
        return (seq_len - starts).clamp(0, BLOCK_SIZE).to(torch.int32)


__all__ = [
    "BLOCK_SIZE",
    "BUDGET_GRANULARITY",
    "RoutingPlan",
    "SubBlockRouter",
    "snap_topk_to_kernel_budget",
]
