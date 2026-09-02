# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import copy
import json

import pytest

from benchmarks.diffusion.analyze_minimax_h3_subblock_e2e import analyze

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _run(label: str, index: int, duration_scale: float) -> dict:
    seed = 1701 if label == "warmup" else 1701 + index
    return {
        "label": label,
        "seed": seed,
        "prompt": f"prompt-{index % 3}",
        "reference_image": {"sha256": f"reference-{index % 3}"},
        "wall_time_s": 20.0 * duration_scale,
        "server_inference_time_s": 19.0 * duration_scale,
        "stage_durations": {
            "MiniMaxH3Pipeline.diffuse": 17.0 * duration_scale,
            "MiniMaxH3Pipeline.encode_prompt": 1.0 * duration_scale,
            "MiniMaxH3Pipeline._encode_visual_conditions": 0.5 * duration_scale,
            "MiniMaxH3Pipeline.decode": 1.5 * duration_scale,
        },
        "media": {
            "full_decode_passed": True,
            "sha256": f"media-{duration_scale}-{index}",
        },
    }


def _arm(backend: str, identity: dict, duration_scale: float) -> dict:
    shape = {
        "width": 1344,
        "height": 768,
        "fps": 24,
        "duration_s_requested": 5.0,
        "steps": 50,
    }
    runs = [_run("warmup", 0, duration_scale)] + [
        _run(f"measured_{index}", index, duration_scale) for index in range(1, 6)
    ]
    arm = {
        "status": "passed",
        "benchmark_backend": backend,
        "container_identity": identity,
        "gpu_inventory": identity["gpu_inventory"],
        "task": "fl2va",
        "shape": shape,
        "warmup_excluded_from_measurement": True,
        "measured_runs": 5,
        "runs": runs,
        "sampled_peak_memory_mib_per_gpu_whole_run": [70000],
    }
    if backend == "SUBBLOCK_ATTN":
        arm.update(
            {
                "attention": {
                    "default": {
                        "backend": "SUBBLOCK_ATTN",
                        "subblock": {
                            "sparsity": 0.75,
                            "skip_first_steps": 10,
                        },
                    }
                },
                "sampled_peak_memory_mib_per_gpu_whole_run": [72000],
                "subblock_stats_lines": [
                    "configured_layers=50, sparse_eligible_layers=50, "
                    "actual_sparse_calls=1950, dense_calls_by_reason={'warmup_step': 500}, "
                    "last_plan_density=[0.25, 0.25]"
                ]
                * 6,
            }
        )
    return arm


def _summary() -> dict:
    identity = {
        "pair_invocation_id": "pair",
        "container_boot_nonce": "boot",
        "gpu_inventory": ["GPU-test, NVIDIA B200"],
    }
    return {
        "status": "passed",
        "same_container_proven": True,
        "container_identity": identity,
        "duration_s": 5.0,
        "arm_order": ["SUBBLOCK_ATTN", "TRTLLM_ATTN"],
        "arms": {
            "TRTLLM_ATTN": _arm("TRTLLM_ATTN", identity, 1.0),
            "SUBBLOCK_ATTN": _arm("SUBBLOCK_ATTN", identity, 0.5),
        },
    }


def _analyze(tmp_path, payload: dict) -> dict:
    path = tmp_path / "paired.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return analyze(path)


def test_analyzer_accepts_complete_five_pair_same_container_summary(tmp_path):
    result = _analyze(tmp_path, _summary())

    assert result["pairing_validation"]["measured_pair_count"] == 5
    assert result["pairing_validation"]["all_measured_media_fully_decoded"] is True
    assert result["metrics"]["client_wall"]["paired_speedup_median"] == 2.0
    assert result["metrics"]["client_wall"]["paired_speedup_exact_bootstrap_95pct_ci"] == [2.0, 2.0]
    assert result["subblock_execution"]["denoise_transitions_per_request"] == 49
    assert result["subblock_execution"]["actual_sparse_calls_per_request"] == [1950]


def test_analyzer_rejects_fewer_than_five_pairs(tmp_path):
    payload = _summary()
    for arm in payload["arms"].values():
        arm["runs"].pop()
        arm["measured_runs"] = 4

    with pytest.raises(ValueError, match="at least 5 measured pairs"):
        _analyze(tmp_path, payload)


def test_analyzer_rejects_container_change(tmp_path):
    payload = _summary()
    payload["arms"]["SUBBLOCK_ATTN"]["container_identity"] = copy.deepcopy(payload["container_identity"])
    payload["arms"]["SUBBLOCK_ATTN"]["container_identity"]["container_boot_nonce"] = "other"

    with pytest.raises(ValueError, match="container identity differs"):
        _analyze(tmp_path, payload)


def test_analyzer_rejects_multiple_gpus(tmp_path):
    payload = _summary()
    payload["container_identity"]["gpu_inventory"].append("GPU-test-2, NVIDIA B200")
    for arm in payload["arms"].values():
        arm["gpu_inventory"] = payload["container_identity"]["gpu_inventory"]

    with pytest.raises(ValueError, match="exactly one B200/GB200"):
        _analyze(tmp_path, payload)


def test_analyzer_rejects_unmatched_prompt(tmp_path):
    payload = _summary()
    payload["arms"]["SUBBLOCK_ATTN"]["runs"][1]["prompt"] = "different"

    with pytest.raises(ValueError, match="unmatched benchmark pair"):
        _analyze(tmp_path, payload)


def test_analyzer_rejects_media_without_full_decode(tmp_path):
    payload = _summary()
    payload["arms"]["TRTLLM_ATTN"]["runs"][1]["media"]["full_decode_passed"] = False

    with pytest.raises(ValueError, match="did not pass a full media decode"):
        _analyze(tmp_path, payload)


def test_analyzer_rejects_impossible_sparse_call_count(tmp_path):
    payload = _summary()
    payload["arms"]["SUBBLOCK_ATTN"]["subblock_stats_lines"] = [
        line.replace("actual_sparse_calls=1950", "actual_sparse_calls=1900")
        for line in payload["arms"]["SUBBLOCK_ATTN"]["subblock_stats_lines"]
    ]

    with pytest.raises(ValueError, match="unexpected sparse call count"):
        _analyze(tmp_path, payload)
