# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""B200 microbenchmark for MiniMax-H3 SUBBLOCK_ATTN.

This benchmark is intentionally strict about warmup and pairing. It warms the
complete dense and sparse arms independently, then alternates their measured
order on the same tensors/device. A CUDA profiler pass separates the stock
FlashInfer wrapper's layout/packing work from ``fused_attn_device`` itself.

Examples:
    python benchmarks/diffusion/bench_subblock_attn.py --preset smoke
    python benchmarks/diffusion/bench_subblock_attn.py --preset h3-long --output-json result.json

The synthetic long-sequence shape is a kernel/integration gate, not a real-model H3
quality or end-to-end benchmark. Any real-model claim must name its task and
report DiT, pipeline, video/audio diagnostics, and actual sparse-call counters.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

from vllm_omni.diffusion.attention.backends.subblock import SubBlockRouter, run_bsa_attn_blk64
from vllm_omni.diffusion.attention.backends.subblock_attn import SubBlockAttentionImpl
from vllm_omni.diffusion.attention.backends.trtllm_attn import TrtllmAttentionImpl
from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context

_PRESETS = {
    "smoke": {"seq_len": 4096, "heads": 8},
    # Representative long H3 self-attention shape used for the initial
    # performance gate. Replace with the observed packed length in final runs.
    "h3-long": {"seq_len": 37760, "heads": 56},
}


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _percentile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _bootstrap_median_ci(values: list[float], samples: int = 2000) -> tuple[float, float]:
    rng = random.Random(0)
    bootstrapped = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        bootstrapped.append(statistics.median(draw))
    bootstrapped.sort()
    return _percentile(bootstrapped, 0.025), _percentile(bootstrapped, 0.975)


def _cuda_time_ms(fn: Callable[[], torch.Tensor]) -> tuple[float, torch.Tensor]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end), out


def _warmup(name: str, fn: Callable[[], torch.Tensor], count: int) -> None:
    print(f"warming {name}: {count} complete calls", flush=True)
    for _ in range(count):
        fn()
    torch.accelerator.synchronize()


def _measure_component(fn: Callable[[], torch.Tensor], iters: int) -> dict[str, Any]:
    times = [_cuda_time_ms(fn)[0] for _ in range(iters)]
    return {
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "samples_ms": times,
    }


def _peak_memory_mib(fn: Callable[[], torch.Tensor], device: torch.device) -> dict[str, float]:
    torch.accelerator.reset_peak_memory_stats(device)
    baseline_allocated = torch.accelerator.memory_allocated(device)
    baseline_reserved = torch.accelerator.memory_reserved(device)
    out = fn()
    torch.accelerator.synchronize(device)
    peak_allocated = torch.accelerator.max_memory_allocated(device)
    peak_reserved = torch.accelerator.max_memory_reserved(device)
    del out
    mib = 1024**2
    return {
        "baseline_allocated": baseline_allocated / mib,
        "baseline_reserved": baseline_reserved / mib,
        "peak_allocated": peak_allocated / mib,
        "peak_reserved": peak_reserved / mib,
        "incremental_allocated": (peak_allocated - baseline_allocated) / mib,
    }


def _layout_view_us(qkv: torch.Tensor, head_dim: int, iters: int) -> float:
    samples = []
    for _ in range(iters):
        start = time.perf_counter_ns()
        q = qkv[..., :head_dim]
        k = qkv[..., head_dim : 2 * head_dim]
        v = qkv[..., 2 * head_dim :]
        samples.append((time.perf_counter_ns() - start) / 1000.0)
        if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
            raise RuntimeError("Q/K/V head-dim axis must remain contiguous")
    return statistics.median(samples)


def _profile_bsa_cuda(
    fn: Callable[[], torch.Tensor],
    iters: int,
    trace_path: Path | None = None,
) -> dict[str, Any]:
    """Split serialized CUDA work inside the stock blk64 wrapper.

    FlashInfer 0.6.16.post3 copies strided Q/K/V to contiguous BSHD in its
    Python wrapper, then performs additional B/H/S and block packing inside
    the extension. The only actual attention event is
    ``flash::fused_attn_device``; all other CUDA events are reported together
    as layout/packing/other wrapper work instead of being mislabeled as the
    attention kernel.
    """
    torch.accelerator.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        for _ in range(iters):
            fn()
    torch.accelerator.synchronize()
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_path))

    cuda_events = [event for event in prof.events() if str(event.device_type).endswith(".CUDA")]
    attention_events = [event for event in cuda_events if "flash::fused_attn_device" in event.name]
    if len(attention_events) != iters:
        names = sorted({event.name for event in cuda_events})
        raise RuntimeError(
            f"Expected {iters} flash::fused_attn_device events, found {len(attention_events)}; CUDA events were {names}"
        )
    total_us = sum(event.device_time_total for event in cuda_events)
    attention_us = sum(event.device_time_total for event in attention_events)
    return {
        "profiled_calls": iters,
        "attention_kernel_ms_per_call": attention_us / (iters * 1000.0),
        "layout_pack_and_other_cuda_ms_per_call": (total_us - attention_us) / (iters * 1000.0),
        "total_cuda_ms_per_call": total_us / (iters * 1000.0),
        "cuda_event_count_per_call": len(cuda_events) / iters,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("SUBBLOCK_ATTN benchmark requires CUDA")
    device = torch.device(args.device)
    capability = torch.cuda.get_device_capability(device)
    if capability != (10, 0):
        raise RuntimeError(f"This benchmark is B200-only; {device} reports compute capability {capability}")
    if args.warmup < 1:
        raise ValueError("--warmup must be at least 1")
    if args.iters < 5:
        raise ValueError("--iters must be at least 5")
    if args.profile_iters < 1:
        raise ValueError("--profile-iters must be at least 1")

    preset = _PRESETS[args.preset]
    seq_len = args.seq_len or preset["seq_len"]
    heads = args.heads or preset["heads"]
    head_dim = 128
    scale = head_dim**-0.5
    torch.manual_seed(args.seed)
    qkv = torch.randn(1, seq_len, heads, 3 * head_dim, device=device, dtype=torch.bfloat16)
    q_view = qkv[..., :head_dim]
    k_view = qkv[..., head_dim : 2 * head_dim]
    v_view = qkv[..., 2 * head_dim :]
    if args.input_layout == "fused-qkv-strided":
        q, k, v = q_view, k_view, v_view
    elif args.input_layout == "h3-no-sp":
        # H3's fused norm/RoPE writes contiguous Q/K, while V remains a view
        # of the fused QKV projection when sequence parallelism is disabled.
        q, k, v = q_view.contiguous(), k_view.contiguous(), v_view
    else:
        # Ulysses all-to-all materializes contiguous BSND outputs for Q/K/V.
        q, k, v = q_view.contiguous(), k_view.contiguous(), v_view.contiguous()
    if any(tensor.stride(-1) != 1 for tensor in (q, k, v)):
        raise RuntimeError("Q/K/V must have a contiguous head-dim axis")

    router = SubBlockRouter(n_q=args.n_q, n_k=args.n_k)
    dense_impl = TrtllmAttentionImpl(
        num_heads=heads,
        head_size=head_dim,
        softmax_scale=scale,
        causal=False,
        num_kv_heads=heads,
        qkv_layout="BSND",
        backend_kwargs={},
        role="self",
    )
    production_impl = SubBlockAttentionImpl(
        num_heads=heads,
        head_size=head_dim,
        softmax_scale=scale,
        causal=False,
        num_kv_heads=heads,
        prefix="transformer.blocks.0.attn",
        qkv_layout="BSND",
        backend_kwargs={
            "sparsity": args.sparsity,
            "skip_first_steps": args.skip_first_steps,
            "skip_first_layers": 0,
            "n_q": args.n_q,
            "n_k": args.n_k,
            "min_seq_len": args.min_seq_len,
        },
        role="self",
    )

    def dense() -> torch.Tensor:
        return dense_impl.forward_cuda(q, k, v)

    def production() -> torch.Tensor:
        return production_impl.forward_cuda(q, k, v)

    def route():
        return router.route(q, k, sparsity=args.sparsity, softmax_scale=scale)

    plan = route()
    bsa_call_count = 0

    def execute_bsa(current_plan, current_q=q, current_k=k, current_v=v) -> torch.Tensor:
        nonlocal bsa_call_count
        bsa_call_count += 1
        return run_bsa_attn_blk64(
            current_q,
            current_k,
            current_v,
            current_plan.index,
            current_plan.topk,
            scale,
        )

    def bsa_kernel() -> torch.Tensor:
        return execute_bsa(plan)

    def candidate_sparse() -> torch.Tensor:
        current_plan = route()
        return execute_bsa(current_plan)

    # First calls may resolve artifacts/JIT compile. They are excluded before
    # each arm receives its own complete-call warmup.
    dense()
    candidate_sparse()
    torch.accelerator.synchronize(device)
    forward_context = ForwardContext(denoise_step_idx=args.denoise_step)
    with override_forward_context(forward_context):
        _warmup("dense", dense, args.warmup)
        _warmup("production SUBBLOCK dispatcher", production, args.warmup)

        production_before = production_impl.runtime_stats()
        dense_ms: list[float] = []
        selected_ms: list[float] = []
        dense_out = selected_out = None
        for iteration in range(args.iters):
            arms = (("dense", dense), ("selected", production))
            if iteration % 2:
                arms = tuple(reversed(arms))
            for name, fn in arms:
                elapsed, out = _cuda_time_ms(fn)
                if name == "dense":
                    dense_ms.append(elapsed)
                    dense_out = out
                else:
                    selected_ms.append(elapsed)
                    selected_out = out
        production_after = production_impl.runtime_stats()
        peak_vram = {
            "dense": _peak_memory_mib(dense, device),
            "production_subblock": _peak_memory_mib(production, device),
        }

    _warmup("router", lambda: route().index, args.warmup)
    _warmup("stock BSA wrapper", bsa_kernel, args.warmup)

    assert dense_out is not None and selected_out is not None
    measured_sparse_calls = production_after["actual_sparse_calls"] - production_before["actual_sparse_calls"]
    dense_reasons = set(production_before["dense_calls_by_reason"]) | set(production_after["dense_calls_by_reason"])
    measured_dense_reasons = {
        reason: production_after["dense_calls_by_reason"].get(reason, 0)
        - production_before["dense_calls_by_reason"].get(reason, 0)
        for reason in sorted(dense_reasons)
    }
    measured_dense_reasons = {reason: count for reason, count in measured_dense_reasons.items() if count}
    paired_speedups = [dense_time / selected_time for dense_time, selected_time in zip(dense_ms, selected_ms)]
    speedup_ci = _bootstrap_median_ci(paired_speedups)
    sparse_expected = args.sparsity > 0.0 and args.denoise_step >= args.skip_first_steps and seq_len >= args.min_seq_len
    if sparse_expected:
        gate_passed = measured_sparse_calls == args.iters and speedup_ci[0] > 1.0
        gate_reason = "BSA selected and paired speedup 95% CI lower bound exceeds 1.0"
    else:
        gate_passed = measured_sparse_calls == 0 and statistics.median(paired_speedups) >= 0.97
        gate_reason = "dense fallback selected and median dispatcher throughput is within 3% of dense"
    dense_float = dense_out.float()
    selected_float = selected_out.float()
    diff = selected_float - dense_float
    relative_l2 = float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(dense_float))
    cosine = float(torch.nn.functional.cosine_similarity(dense_float.flatten(), selected_float.flatten(), dim=0))
    if not torch.isfinite(selected_out).all():
        raise RuntimeError("Production SUBBLOCK output contains non-finite values")
    if args.sparsity == 0.0 and relative_l2 >= 0.02:
        raise RuntimeError(f"All-block BSA relative L2 {relative_l2:.6f} exceeds 0.02")

    router_stats = _measure_component(lambda: route().index, args.iters)
    wrapper_stats = _measure_component(bsa_kernel, args.iters)
    production_input_profile = _profile_bsa_cuda(bsa_kernel, args.profile_iters, args.profile_trace)

    # Keep a second input arm only for profiling the wrapper's explicit
    # ``q.contiguous()/k.contiguous()/v.contiguous()`` cost. Allocate it after
    # the real-arm peak-VRAM measurements so benchmark-only tensors do not
    # inflate those values.
    prepared_q = q.contiguous()
    prepared_k = k.contiguous()
    prepared_v = v.contiguous()

    def bsa_precontiguous() -> torch.Tensor:
        return execute_bsa(plan, prepared_q, prepared_k, prepared_v)

    _warmup("stock BSA wrapper (precontiguous input)", bsa_precontiguous, args.warmup)
    precontiguous_profile = _profile_bsa_cuda(bsa_precontiguous, args.profile_iters)
    python_contiguous_estimate = max(
        0.0,
        production_input_profile["layout_pack_and_other_cuda_ms_per_call"]
        - precontiguous_profile["layout_pack_and_other_cuda_ms_per_call"],
    )
    result = {
        "environment": {
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(capability),
            "torch": torch.__version__,
            "flashinfer_python": _package_version("flashinfer-python"),
            "vllm": _package_version("vllm"),
        },
        "config": {
            "preset": args.preset,
            "seq_len": seq_len,
            "heads": heads,
            "head_dim": head_dim,
            "sparsity_requested": args.sparsity,
            "n_q": args.n_q,
            "n_k": args.n_k,
            "min_seq_len": args.min_seq_len,
            "skip_first_steps": args.skip_first_steps,
            "denoise_step": args.denoise_step,
            "warmup_complete_calls_per_arm": args.warmup,
            "measured_interleaved_pairs": args.iters,
            "profiled_bsa_calls_per_input_layout": args.profile_iters,
            "input_layout": args.input_layout,
            "input_contiguous": {"q": q.is_contiguous(), "k": k.is_contiguous(), "v": v.is_contiguous()},
            "input_strides": {"q": list(q.stride()), "k": list(k.stride()), "v": list(v.stride())},
            "seed": args.seed,
        },
        "routing_plan": {
            "kept_blocks": plan.topk,
            "available_blocks": plan.num_blocks,
            "density": plan.density,
            "realized_sparsity": 1.0 - plan.density,
        },
        "production_dispatch": {
            "expected_path": "sparse" if sparse_expected else "dense",
            "actual_sparse_calls": measured_sparse_calls,
            "dense_calls_by_reason": measured_dense_reasons,
        },
        "timing": {
            "layout_view_median_us": _layout_view_us(qkv, head_dim, max(args.iters, 100)),
            "router": router_stats,
            "layout_copy_and_pack": {
                "production_input_ms_per_call": production_input_profile["layout_pack_and_other_cuda_ms_per_call"],
                "precontiguous_input_ms_per_call": precontiguous_profile["layout_pack_and_other_cuda_ms_per_call"],
                "python_contiguous_copy_estimate_ms_per_call": python_contiguous_estimate,
                "note": (
                    "Includes stock-wrapper Q/K/V normalization plus extension-side B/H/S and block packing; "
                    "the estimate is the selected production input minus precontiguous profiler difference."
                ),
            },
            "attention_kernel": {
                "flash_fused_attn_device_ms_per_call": production_input_profile["attention_kernel_ms_per_call"],
                "profiled_calls": production_input_profile["profiled_calls"],
            },
            "stock_bsa_wrapper": wrapper_stats,
            "stock_bsa_profiler": {
                "production_input": production_input_profile,
                "precontiguous_input": precontiguous_profile,
            },
            "dense_e2e": {
                "median_ms": statistics.median(dense_ms),
                "samples_ms": dense_ms,
            },
            "production_subblock_e2e": {
                "median_ms": statistics.median(selected_ms),
                "samples_ms": selected_ms,
            },
            "paired_speedup": {
                "median": statistics.median(paired_speedups),
                "bootstrap_95pct_ci": list(speedup_ci),
                "samples": paired_speedups,
            },
        },
        "peak_vram_allocator_mib": peak_vram,
        "numerics": {
            "relative_l2_vs_dense": relative_l2,
            "cosine_vs_dense": cosine,
            "finite": True,
        },
        "acceptance": {
            "passed": gate_passed,
            "criterion": gate_reason,
        },
        "actual_sparse_calls": measured_sparse_calls,
        "stock_bsa_calls_executed": bsa_call_count,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(_PRESETS), default="smoke")
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--sparsity", type=float, default=0.75)
    parser.add_argument("--n-q", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--n-k", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--min-seq-len", type=int, default=24576)
    parser.add_argument("--skip-first-steps", type=int, default=10)
    parser.add_argument("--denoise-step", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--profile-iters", type=int, default=3)
    parser.add_argument("--profile-trace", type=Path)
    parser.add_argument(
        "--input-layout",
        choices=("fused-qkv-strided", "h3-no-sp", "contiguous"),
        default="fused-qkv-strided",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    with torch.inference_mode():
        result = run(args)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    if not result["acceptance"]["passed"]:
        raise SystemExit(f"SUBBLOCK_ATTN acceptance gate failed: {result['acceptance']['criterion']}")


if __name__ == "__main__":
    main()
