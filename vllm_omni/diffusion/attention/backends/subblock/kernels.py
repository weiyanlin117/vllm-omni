# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused router kernels for FlashInfer 64-token BSA plans.

Adapted from SGLang's Apache-2.0 SubBlock implementation introduced by
sglang#34148 and subsequently optimized in ``subblock_sparse/kernels.py``.
The numerical contract is intentionally kept aligned with that implementation:
BF16 pooled Q/K, FP32 block scores, and an unsorted int32 top-k plan.
"""

import math

import torch
from vllm.triton_utils import tl, triton

_NEG = tl.constexpr(-1.0e30)
_LN2 = tl.constexpr(0.6931471805599453)


@triton.jit
def _score_kernel(
    q_ptr,
    k_ptr,
    out_ptr,
    stride_qm,
    stride_ql,
    stride_kn,
    stride_kl,
    stride_om,
    stride_on,
    stride_ol,
    m,
    m_valid,
    n_valid,
    n_out,
    m_out,
    blk_m: tl.constexpr,
    blk_n: tl.constexpr,
    n_k: tl.constexpr,
    n_q: tl.constexpr,
    head_dim: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_l = tl.program_id(2)
    offs_m = pid_m * blk_m + tl.arange(0, blk_m)
    offs_n = pid_n * blk_n + tl.arange(0, blk_n)
    offs_d = tl.arange(0, head_dim)
    q = tl.load(
        q_ptr + pid_l * stride_ql + offs_m[:, None] * stride_qm + offs_d[None, :],
        mask=offs_m[:, None] < m,
        other=0.0,
    )
    k = tl.load(
        k_ptr + pid_l * stride_kl + offs_n[:, None] * stride_kn + offs_d[None, :],
        mask=offs_n[:, None] < n_valid,
        other=0.0,
    )
    acc = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
    acc = tl.where(offs_n[None, :] < n_valid, acc, _NEG)
    acc = tl.where(offs_m[:, None] < m_valid, acc, _NEG)
    acc = tl.reshape(acc, (blk_m, blk_n // n_k, n_k))
    max_k = tl.max(acc, axis=2)
    sum_k = tl.sum(tl.exp2(acc - max_k[:, :, None]), axis=2)
    lse = max_k + tl.log2(sum_k)
    lse = tl.where(max_k > _NEG / 2, lse, _NEG)
    if n_q > 1:
        lse = tl.reshape(lse, (blk_m // n_q, n_q, blk_n // n_k))
        max_q = tl.max(lse, axis=1)
        sum_q = tl.sum(tl.exp2(lse - max_q[:, None, :]), axis=1)
        lse = tl.where(max_q > _NEG / 2, max_q + tl.log2(sum_q), _NEG)
    out = (lse * _LN2).to(out_ptr.dtype.element_ty)
    offs_o = pid_n * (blk_n // n_k) + tl.arange(0, blk_n // n_k)
    offs_q = pid_m * (blk_m // n_q) + tl.arange(0, blk_m // n_q)
    tl.store(
        out_ptr + pid_l * stride_ol + offs_q[:, None] * stride_om + offs_o[None, :] * stride_on,
        out,
        mask=(offs_q[:, None] < m_out) & (offs_o[None, :] < n_out),
    )


_SCORE_TILE_M = 128
_SCORE_TILE_N = 128


def fused_scores(
    pooled_q: torch.Tensor,
    pooled_k: torch.Tensor,
    out: torch.Tensor,
    *,
    n_k: int,
    n_valid: int,
    n_q: int,
    m_valid: int,
) -> torch.Tensor:
    """Reduce sub-cell QK logits directly into FP32 64x64 block scores."""
    layers, m, head_dim = pooled_q.shape
    n = pooled_k.shape[1]
    m_out, n_out = out.shape[1], out.shape[2]
    grid = (triton.cdiv(m, _SCORE_TILE_M), triton.cdiv(n, _SCORE_TILE_N), layers)
    _score_kernel[grid](
        pooled_q,
        pooled_k,
        out,
        pooled_q.stride(1),
        pooled_q.stride(0),
        pooled_k.stride(1),
        pooled_k.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(0),
        m,
        m_valid,
        n_valid,
        n_out,
        m_out,
        blk_m=_SCORE_TILE_M,
        blk_n=_SCORE_TILE_N,
        n_k=n_k,
        n_q=n_q,
        head_dim=head_dim,
        num_warps=4,
        num_stages=3,
    )
    return out


@triton.jit
def _pool_kernel(
    x_ptr,
    out_ptr,
    stride_xb,
    stride_xt,
    stride_xh,
    stride_yl,
    stride_yn,
    seq_len,
    heads,
    subblock_size: tl.constexpr,
    head_dim: tl.constexpr,
    scale,
):
    cell = tl.program_id(0)
    layer = tl.program_id(1)
    batch = layer // heads
    head = layer % heads
    offs_t = cell * subblock_size + tl.arange(0, subblock_size)
    offs_d = tl.arange(0, head_dim)
    mask = offs_t < seq_len
    x = tl.load(
        x_ptr + batch * stride_xb + offs_t[:, None] * stride_xt + head * stride_xh + offs_d[None, :],
        mask=mask[:, None],
        other=0.0,
    ).to(tl.float32)
    count = tl.sum(mask.to(tl.float32), axis=0)
    pooled = tl.sum(x, axis=0) / tl.maximum(count, 1.0) * scale
    tl.store(out_ptr + layer * stride_yl + cell * stride_yn + offs_d, pooled.to(tl.bfloat16))


def fused_pool(
    x: torch.Tensor,
    n_cells: int,
    subblock_size: int,
    out: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Pool ``[B, S, H, D]`` into BF16 sub-cell means without a full copy."""
    batch, seq_len, heads, head_dim = x.shape
    _pool_kernel[(n_cells, batch * heads)](
        x,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        seq_len,
        heads,
        subblock_size=subblock_size,
        head_dim=head_dim,
        scale=scale,
        num_warps=1,
    )
    return out


@triton.jit
def _topk_kernel(
    scores_ptr,
    out_ptr,
    num_blocks,
    topk,
    block_size: tl.constexpr,
    num_iters: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    valid = offsets < num_blocks
    scores = tl.load(scores_ptr + row * num_blocks + offsets, mask=valid, other=-float("inf")).to(tl.float32)
    lo = tl.min(tl.where(valid, scores, float("inf")))
    hi = tl.max(tl.where(valid, scores, -float("inf"))) + 1.0
    count_lo = tl.sum(valid.to(tl.int32), axis=0).to(tl.float32)
    count_hi = 0.0
    for _ in tl.static_range(num_iters):
        denominator = count_lo - count_hi
        fraction = (count_lo - topk) / tl.where(denominator > 0.5, denominator, 1.0)
        fraction = tl.minimum(tl.maximum(fraction, 0.05), 0.95)
        midpoint = lo + (hi - lo) * fraction
        count = tl.sum(((scores >= midpoint) & valid).to(tl.int32), axis=0).to(tl.float32)
        take = count >= topk
        lo = tl.where(take, midpoint, lo)
        count_lo = tl.where(take, count, count_lo)
        hi = tl.where(take, hi, midpoint)
        count_hi = tl.where(take, count_hi, count)
    selected = (scores >= lo) & valid
    position = tl.cumsum(selected.to(tl.int32), axis=0) - 1
    tl.store(out_ptr + row * topk + position, offsets.to(tl.int32), mask=selected & (position < topk))


def _topk_iters(num_blocks: int, topk: int) -> int:
    tail_distance = math.log2(max(num_blocks, 1) / max(topk, 1))
    if tail_distance <= 2.5:
        return 16
    if tail_distance <= 3.5:
        return 24
    return 32


def fused_topk(scores: torch.Tensor, topk: int) -> torch.Tensor:
    """Select unsorted block indices from contiguous ``[rows, blocks]`` scores."""
    rows, num_blocks = scores.shape
    out = torch.empty(rows, topk, dtype=torch.int32, device=scores.device)
    _topk_kernel[(rows,)](
        scores,
        out,
        num_blocks,
        topk,
        block_size=triton.next_power_of_2(num_blocks),
        num_iters=_topk_iters(num_blocks, topk),
        num_warps=4 if num_blocks >= 1024 else 2,
    )
    return out
