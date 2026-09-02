# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Narrow adapter around FlashInfer's stock SM100 blk64 BSA entry point."""

from __future__ import annotations

import functools
import math
import os
import shutil
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

from .router import SubBlockRouter


def _flashinfer_version() -> str:
    try:
        return version("flashinfer-python")
    except PackageNotFoundError:
        return "not installed"


@functools.lru_cache(maxsize=1)
def load_bsa_attn_blk64_fwd() -> Callable:
    """Probe the exact low-level symbol needed by SUBBLOCK_ATTN."""
    try:
        from flashinfer.cute_dsl.sparse import bsa_attn_blk64_fwd
    except Exception as exc:
        raise ImportError(
            "SUBBLOCK_ATTN requires FlashInfer's "
            "flashinfer.cute_dsl.sparse.bsa_attn_blk64_fwd symbol; installed "
            f"flashinfer-python version is {_flashinfer_version()!r}. Install a version/build "
            "that contains the SM100 blk64 BSA kernel."
        ) from exc
    return bsa_attn_blk64_fwd


def _compiler_include_paths() -> tuple[Path, ...]:
    paths: list[Path] = []
    for variable in ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"):
        paths.extend(Path(entry) for entry in os.environ.get(variable, "").split(os.pathsep) if entry)
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        if root := os.environ.get(variable):
            paths.append(Path(root) / "include")
    paths.extend((Path("/usr/local/cuda/include"), Path("/usr/include")))
    return tuple(dict.fromkeys(paths))


def _find_compiler_header(relative_path: str) -> Path | None:
    for include_path in _compiler_include_paths():
        candidate = include_path / relative_path
        if candidate.is_file():
            return candidate
    return None


@functools.lru_cache(maxsize=1)
def _validate_blk64_jit_assets() -> None:
    """Probe assets imported lazily by FlashInfer's current blk64 loader.

    The 0.6.16.post3 wheel exposes ``bsa_attn_blk64_fwd`` even when its first
    call cannot compile: the loader also needs the FlashInfer source checkout,
    the CUTLASS submodule, NVTX v3 headers, NVCC, and Ninja. Keep this check
    conditional on that JIT loader so a future AOT implementation is not tied
    to today's source layout.
    """
    try:
        from flashinfer.cute_dsl.sparse.blk64 import loader
    except ModuleNotFoundError:
        return

    get_cutlass_root = getattr(loader, "_get_cutlass_root", None)
    if get_cutlass_root is None:
        return
    try:
        cutlass_root = Path(get_cutlass_root())
    except RuntimeError as exc:
        raise ImportError(
            "FlashInfer exposes bsa_attn_blk64_fwd, but its blk64 JIT sources cannot locate "
            "3rdparty/cutlass. Install FlashInfer from a recursive source checkout (including "
            "the CUTLASS submodule); the standalone flashinfer-python wheel is not sufficient "
            f"for this kernel in version {_flashinfer_version()!r}."
        ) from exc
    if not (cutlass_root / "include" / "cutlass" / "cutlass.h").is_file():
        raise ImportError(f"FlashInfer blk64 found CUTLASS at {cutlass_root}, but its headers are incomplete.")

    launch_header = Path(loader.__file__).resolve().with_name("flash_fwd_launch_template.h")
    if launch_header.is_file() and "nvtx3/nvToolsExt.h" in launch_header.read_text(encoding="utf-8", errors="ignore"):
        if _find_compiler_header("nvtx3/nvToolsExt.h") is None:
            raise ImportError(
                "FlashInfer blk64 JIT requires the NVTX v3 header nvtx3/nvToolsExt.h. "
                "Install NVIDIA/NVTX and expose its include directory through CPATH."
            )
    if shutil.which("ninja") is None:
        raise ImportError("FlashInfer blk64 JIT requires Ninja on PATH.")
    nvcc = shutil.which("nvcc")
    cuda_roots = [os.environ.get(name) for name in ("CUDA_HOME", "CUDA_PATH")]
    if nvcc is None and not any(root and (Path(root) / "bin" / "nvcc").is_file() for root in cuda_roots):
        if not Path("/usr/local/cuda/bin/nvcc").is_file():
            raise ImportError("FlashInfer blk64 JIT requires an NVCC toolchain, not a runtime-only CUDA image.")


@functools.cache
def _validate_b200_device(device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError(f"SUBBLOCK_ATTN requires a CUDA B200 device, got {device}.")
    capability = torch.cuda.get_device_capability(device)
    if capability != (10, 0):
        raise ValueError(
            "SUBBLOCK_ATTN is intentionally limited to B200/GB200 (SM100, compute capability 10.0); "
            f"the selected device reports {capability[0]}.{capability[1]}."
        )
    device_name = torch.cuda.get_device_name(device)
    if "B200" not in device_name.upper():
        raise ValueError(
            "SUBBLOCK_ATTN has only been validated on B200/GB200, not every SM100 product; "
            f"the selected device reports {device_name!r}."
        )


def validate_b200_bsa_available() -> None:
    """Fail during backend resolution if hardware or the exact symbol is absent."""
    if not torch.cuda.is_available():
        raise ValueError("SUBBLOCK_ATTN requires an available CUDA B200 GPU.")
    _validate_b200_device(torch.device("cuda", torch.accelerator.current_device_index()))
    load_bsa_attn_blk64_fwd()
    _validate_blk64_jit_assets()


@functools.lru_cache(maxsize=16)
def _cached_block_sizes(seq_len: int, device: torch.device) -> torch.Tensor:
    return SubBlockRouter.block_sizes(seq_len, device)


def run_bsa_attn_blk64(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    topk: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Execute stock FlashInfer BSA while preserving its exact public contract.

    The low-level blk64 API does not accept a caller-owned workspace. Its only
    persistent auxiliary input here is the cached tail ``block_sizes`` tensor.
    FlashInfer 0.6.16.post3 itself normalizes Q/K/V to contiguous BSHD tensors
    and repacks them again inside its extension; benchmark those copies as part
    of the stock wrapper instead of treating this adapter as a zero-copy path.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("FlashInfer blk64 BSA expects Q/K/V in [B, S, H, D] layout")
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1:
        raise ValueError("FlashInfer blk64 BSA integration supports one contiguous sequence per call")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError("FlashInfer blk64 BSA requires BF16 Q/K/V")
    if k.shape != v.shape:
        raise ValueError(f"FlashInfer blk64 BSA requires matching K/V shapes, got {k.shape} and {v.shape}")
    if q.shape[-1] != 128 or k.shape[-1] != 128 or v.shape[-1] != 128:
        raise ValueError("FlashInfer blk64 BSA requires head_dim=128")
    if q.shape[2] != k.shape[2] or q.shape[2] != v.shape[2]:
        raise ValueError("SUBBLOCK_ATTN currently supports MHA only (equal Q/K/V head counts)")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("FlashInfer blk64 BSA requires a contiguous head-dim axis")
    if q.device != k.device or q.device != v.device or q2k_block_index.device != q.device:
        raise ValueError("Q/K/V and q2k_block_index must be on the same CUDA device")
    if q2k_block_index.dtype != torch.int32:
        raise ValueError("q2k_block_index must use int32")
    if isinstance(topk, bool) or not isinstance(topk, int) or not 0 < topk <= math.ceil(k.shape[1] / 64):
        raise ValueError(f"Invalid blk64 BSA topk={topk!r} for K length {k.shape[1]}")
    expected_index_shape = (1, q.shape[2], math.ceil(q.shape[1] / 64), topk)
    if q2k_block_index.shape != expected_index_shape or not q2k_block_index.is_contiguous():
        raise ValueError(
            f"q2k_block_index must be contiguous with shape {expected_index_shape}, got {tuple(q2k_block_index.shape)}"
        )
    _validate_b200_device(q.device)
    _validate_blk64_jit_assets()

    try:
        result = load_bsa_attn_blk64_fwd()(
            q,
            k,
            v,
            q2k_block_index,
            topk,
            block_sizes=_cached_block_sizes(k.shape[1], k.device),
            q2k_block_nums=None,
            softmax_scale=softmax_scale,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "FlashInfer blk64 BSA failed during JIT build or launch. Verify the recursive FlashInfer "
            "source checkout, CUTLASS/NVTX/NVCC/Ninja assets, SM100 driver compatibility, and clear "
            "FLASHINFER_WORKSPACE_BASE only if its cached extension is stale."
        ) from exc
    return result[0] if isinstance(result, tuple) else result


__all__ = ["load_bsa_attn_blk64_fwd", "run_bsa_attn_blk64", "validate_b200_bsa_available"]
