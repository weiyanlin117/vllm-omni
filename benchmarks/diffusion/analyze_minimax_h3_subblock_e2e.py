# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Analyze paired MiniMax H3 dense/SubBlock end-to-end benchmark results.

The input is a per-duration paired benchmark summary. Warmup requests are
excluded, and every measured pair must have the same seed, prompt, reference
image, and container identity before a speedup is reported.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

_DENSE_BACKEND = "TRTLLM_ATTN"
_SPARSE_BACKEND = "SUBBLOCK_ATTN"
_MIN_MEASURED_PAIRS = 5


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty sample")
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _exact_bootstrap_median_ci(values: list[float]) -> tuple[float, float]:
    """Return the exact n-out-of-n percentile-bootstrap interval.

    Five benchmark pairs require only 5**5 = 3,125 resamples. Refuse larger
    inputs whose exact enumeration could accidentally become expensive rather
    than silently changing to a random approximation.
    """

    sample_count = len(values)
    if sample_count == 0:
        raise ValueError("cannot bootstrap an empty sample")
    resample_count = sample_count**sample_count
    if resample_count > 1_000_000:
        raise ValueError(
            f"exact bootstrap would require {resample_count:,} resamples; use no more than six paired runs"
        )
    medians = sorted(
        statistics.median(values[index] for index in indices)
        for indices in itertools.product(range(sample_count), repeat=sample_count)
    )
    return _percentile(medians, 0.025), _percentile(medians, 0.975)


def _measured_runs(arm: dict[str, Any]) -> list[dict[str, Any]]:
    runs = [run for run in arm["runs"] if re.fullmatch(r"measured_\d+", str(run.get("label", "")))]
    return sorted(runs, key=lambda run: int(run["label"].split("_")[-1]))


def _nested_float(payload: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = payload
    for key in path:
        value = value[key]
    return float(value)


def _paired_metric(
    dense_runs: list[dict[str, Any]],
    sparse_runs: list[dict[str, Any]],
    getter: Callable[[dict[str, Any]], float],
) -> dict[str, Any]:
    dense_values = [getter(run) for run in dense_runs]
    sparse_values = [getter(run) for run in sparse_runs]
    paired_speedups = [
        dense_value / sparse_value for dense_value, sparse_value in zip(dense_values, sparse_values, strict=True)
    ]
    ci_low, ci_high = _exact_bootstrap_median_ci(paired_speedups)
    return {
        "dense_s": dense_values,
        "subblock_s": sparse_values,
        "dense_median_s": statistics.median(dense_values),
        "subblock_median_s": statistics.median(sparse_values),
        "ratio_of_medians": statistics.median(dense_values) / statistics.median(sparse_values),
        "paired_speedups": paired_speedups,
        "paired_speedup_median": statistics.median(paired_speedups),
        "paired_speedup_exact_bootstrap_95pct_ci": [ci_low, ci_high],
    }


def _validate_pairing(
    summary: dict[str, Any],
    dense: dict[str, Any],
    sparse: dict[str, Any],
    dense_runs: list[dict[str, Any]],
    sparse_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    if summary.get("status") != "passed":
        raise ValueError(f"paired summary did not pass: {summary.get('status')!r}")
    if not summary.get("same_container_proven"):
        raise ValueError("paired summary does not prove same-container execution")
    if summary.get("arm_order") is None or sorted(summary["arm_order"]) != sorted((_DENSE_BACKEND, _SPARSE_BACKEND)):
        raise ValueError(f"paired summary has invalid arm order: {summary.get('arm_order')!r}")
    identity = summary["container_identity"]
    gpu_inventory = identity.get("gpu_inventory")
    if not isinstance(gpu_inventory, list) or len(gpu_inventory) != 1 or "B200" not in gpu_inventory[0].upper():
        raise ValueError(f"paired summary must prove execution on exactly one B200/GB200; got {gpu_inventory!r}")
    for backend, arm in ((_DENSE_BACKEND, dense), (_SPARSE_BACKEND, sparse)):
        if arm.get("status") != "passed":
            raise ValueError(f"{backend} arm did not pass: {arm.get('status')!r}")
        if arm.get("benchmark_backend") != backend:
            raise ValueError(f"{backend} arm reports benchmark_backend={arm.get('benchmark_backend')!r}")
        if arm.get("container_identity") != summary["container_identity"]:
            raise ValueError(f"{backend} arm container identity differs from the paired invocation")
        if arm.get("gpu_inventory") != summary["container_identity"].get("gpu_inventory"):
            raise ValueError(f"{backend} arm GPU inventory differs from the paired invocation")
        warmups = [run for run in arm.get("runs", []) if run.get("label") == "warmup"]
        if len(warmups) != 1:
            raise ValueError(f"{backend} arm must contain exactly one warmup request; found {len(warmups)}")
        if int(arm.get("measured_runs", -1)) != len(_measured_runs(arm)):
            raise ValueError(f"{backend} arm measured_runs disagrees with its request records")

    if dense.get("task") != sparse.get("task"):
        raise ValueError(f"arm tasks differ: dense={dense.get('task')!r}, subblock={sparse.get('task')!r}")
    if dense.get("shape") != sparse.get("shape"):
        raise ValueError("arm request shapes differ")
    if float(summary.get("duration_s", math.nan)) != float(dense["shape"]["duration_s_requested"]):
        raise ValueError("paired-summary duration differs from the arm request shape")
    if not dense.get("warmup_excluded_from_measurement") or not sparse.get("warmup_excluded_from_measurement"):
        raise ValueError("both arms must exclude a complete warmup request")
    if not dense_runs or len(dense_runs) != len(sparse_runs):
        raise ValueError(f"measured arm sizes differ: dense={len(dense_runs)}, subblock={len(sparse_runs)}")
    if len(dense_runs) < _MIN_MEASURED_PAIRS:
        raise ValueError(f"at least {_MIN_MEASURED_PAIRS} measured pairs are required; found {len(dense_runs)}")
    expected_labels = [f"measured_{index}" for index in range(1, len(dense_runs) + 1)]
    if [run["label"] for run in dense_runs] != expected_labels or [
        run["label"] for run in sparse_runs
    ] != expected_labels:
        raise ValueError("measured request labels must be contiguous and start at measured_1")

    checked_pairs = []
    for dense_run, sparse_run in zip(dense_runs, sparse_runs, strict=True):
        checks = {
            "label": dense_run["label"] == sparse_run["label"],
            "seed": dense_run["seed"] == sparse_run["seed"],
            "prompt": dense_run["prompt"] == sparse_run["prompt"],
            "reference_sha256": dense_run["reference_image"]["sha256"] == sparse_run["reference_image"]["sha256"],
        }
        if not all(checks.values()):
            raise ValueError(f"unmatched benchmark pair {dense_run.get('label')}: {checks}")
        for backend, run in ((_DENSE_BACKEND, dense_run), (_SPARSE_BACKEND, sparse_run)):
            media = run.get("media", {})
            if not media.get("full_decode_passed"):
                raise ValueError(f"{backend} {run['label']} did not pass a full media decode")
            if not media.get("sha256"):
                raise ValueError(f"{backend} {run['label']} has no media SHA-256")
        checked_pairs.append(
            {
                "label": dense_run["label"],
                "seed": dense_run["seed"],
                "reference_sha256": dense_run["reference_image"]["sha256"],
                "checks": checks,
            }
        )

    return {
        "passed": True,
        "same_container_proven": True,
        "pair_invocation_id": identity["pair_invocation_id"],
        "container_boot_nonce": identity["container_boot_nonce"],
        "gpu_inventory": identity["gpu_inventory"],
        "warmup_requests_excluded_per_arm": 1,
        "measured_pair_count": len(checked_pairs),
        "all_measured_media_fully_decoded": True,
        "measured_pairs": checked_pairs,
    }


def _subblock_execution(sparse: dict[str, Any]) -> dict[str, Any]:
    stats_lines = sparse.get("subblock_stats_lines", [])
    densities: list[float] = []
    configured_layers: list[int] = []
    eligible_layers: list[int] = []
    sparse_calls: list[int] = []
    early_dense_calls: list[int] = []
    for line in stats_lines:
        density_match = re.search(r"last_plan_density=\[([0-9.]+), ([0-9.]+)\]", line)
        if density_match:
            densities.extend(float(value) for value in density_match.groups())
        configured_match = re.search(r"configured_layers=(\d+)", line)
        if configured_match:
            configured_layers.append(int(configured_match.group(1)))
        eligible_match = re.search(r"sparse_eligible_layers=(\d+)", line)
        if eligible_match:
            eligible_layers.append(int(eligible_match.group(1)))
        sparse_match = re.search(r"actual_sparse_calls=(\d+)", line)
        if sparse_match:
            sparse_calls.append(int(sparse_match.group(1)))
        early_dense_match = re.search(r"'warmup_step': (\d+)", line)
        if early_dense_match:
            early_dense_calls.append(int(early_dense_match.group(1)))

    if not densities or not configured_layers or not eligible_layers or not sparse_calls:
        raise ValueError("SubBlock runtime evidence is incomplete")
    expected_request_records = len(sparse.get("runs", []))
    if len(stats_lines) < expected_request_records:
        raise ValueError(
            "SubBlock runtime evidence has fewer request summaries than request records: "
            f"stats={len(stats_lines)}, requests={expected_request_records}"
        )
    layers = configured_layers[0]
    if any(value != layers for value in configured_layers):
        raise ValueError(f"configured layer count changed across requests: {configured_layers}")
    eligible = eligible_layers[0]
    if any(value != eligible for value in eligible_layers):
        raise ValueError(f"eligible layer count changed across requests: {eligible_layers}")
    if not 0 < eligible <= layers:
        raise ValueError(f"invalid configured/eligible layer counts: {layers}/{eligible}")
    if any(value <= 0 for value in sparse_calls):
        raise ValueError(f"a request did not execute sparse attention: {sparse_calls}")

    schedule = sparse["attention"]["default"]["subblock"]
    requested_sparsity = float(schedule["sparsity"])
    denoise_transitions = int(sparse["shape"]["steps"]) - 1
    early_dense_transitions = min(int(schedule["skip_first_steps"]), denoise_transitions)
    expected_early_dense_calls = eligible * early_dense_transitions
    expected_sparse_calls = eligible * (denoise_transitions - early_dense_transitions)
    if any(value != expected_sparse_calls for value in sparse_calls):
        raise ValueError(f"unexpected sparse call count; expected {expected_sparse_calls}, observed {sparse_calls}")
    if any(value != expected_early_dense_calls for value in early_dense_calls):
        raise ValueError(
            f"unexpected early dense call count; expected {expected_early_dense_calls}, observed {early_dense_calls}"
        )
    median_density = statistics.median(densities)
    if not math.isfinite(median_density) or not 0.0 < median_density <= 1.0:
        raise ValueError(f"invalid observed plan density: {median_density!r}")
    return {
        "requested_sparsity": requested_sparsity,
        "observed_plan_density_min": min(densities),
        "observed_plan_density_max": max(densities),
        "observed_plan_density_median": median_density,
        "realized_plan_sparsity": 1.0 - median_density,
        "configured_layers": layers,
        "sparse_eligible_layers": eligible,
        "denoise_transitions_per_request": denoise_transitions,
        "actual_sparse_calls_per_request": sorted(set(sparse_calls)),
        "sparse_denoise_transitions_per_request": sorted(set(value / layers for value in sparse_calls)),
        "early_dense_calls_per_request": sorted(set(early_dense_calls)),
        "early_dense_transitions_per_request": sorted(set(value / layers for value in early_dense_calls)),
    }


def analyze(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    arms = summary["arms"]
    dense = arms[_DENSE_BACKEND]
    sparse = arms[_SPARSE_BACKEND]
    dense_runs = _measured_runs(dense)
    sparse_runs = _measured_runs(sparse)
    pairing = _validate_pairing(summary, dense, sparse, dense_runs, sparse_runs)

    def stage(name: str) -> Callable[[dict[str, Any]], float]:
        def get_stage(run: dict[str, Any]) -> float:
            return _nested_float(run, ("stage_durations", f"MiniMaxH3Pipeline.{name}"))

        return get_stage

    dense_peak = max(dense["sampled_peak_memory_mib_per_gpu_whole_run"])
    sparse_peak = max(sparse["sampled_peak_memory_mib_per_gpu_whole_run"])
    return {
        "source": str(path),
        "duration_s": float(summary["duration_s"]),
        "task": dense["task"],
        "shape": dense["shape"],
        "arm_order": summary["arm_order"],
        "pairing_validation": pairing,
        "metrics": {
            "client_wall": _paired_metric(dense_runs, sparse_runs, lambda run: float(run["wall_time_s"])),
            "server_inference": _paired_metric(
                dense_runs,
                sparse_runs,
                lambda run: float(run["server_inference_time_s"]),
            ),
            "dit_diffuse": _paired_metric(dense_runs, sparse_runs, stage("diffuse")),
            "encode_prompt": _paired_metric(dense_runs, sparse_runs, stage("encode_prompt")),
            "encode_visual_conditions": _paired_metric(dense_runs, sparse_runs, stage("_encode_visual_conditions")),
            "decode": _paired_metric(dense_runs, sparse_runs, stage("decode")),
        },
        "memory": {
            "dense_peak_mib": dense_peak,
            "subblock_peak_mib": sparse_peak,
            "delta_mib": sparse_peak - dense_peak,
        },
        "subblock_execution": _subblock_execution(sparse),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = {"durations": [analyze(path) for path in args.summaries]}
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
