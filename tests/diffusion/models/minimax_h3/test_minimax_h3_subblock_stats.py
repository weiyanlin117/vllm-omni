# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-mode MiniMax-H3 SubBlock accounting tests."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_dense_transformer_runtime_snapshot_is_empty_without_backend_probe():
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod

    assert mod._minimax_h3_subblock_runtime_snapshot(torch.nn.Linear(2, 2)) == {
        "configured_layers": 0,
        "sparse_eligible_layers": 0,
        "actual_sparse_calls": 0,
        "dense_calls_by_reason": {},
        "last_plan_density_min": None,
        "last_plan_density_max": None,
    }


def test_diffuse_reports_rank_local_subblock_counter_delta(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod

    pipeline = object.__new__(mod.MiniMaxH3Pipeline)
    transformer = object()
    branch = object()
    pipeline.transformer = transformer
    pipeline.device = torch.device("cpu")
    pipeline._transformer_for_task = lambda task: transformer
    pipeline._resident_dit_layers_on_device = lambda *, enabled: nullcontext()
    pipeline.progress_bar = lambda *, total: nullcontext(SimpleNamespace(update=lambda: None))
    pipeline._build_denoise_inputs = lambda **kwargs: {
        "branch": branch,
        "video_rows": torch.zeros(1),
        "audio_rows": torch.zeros(1),
        "cond_anchor": None,
        "audio_anchor": None,
        "sigmas_video": [1.0, 0.0],
        "sigmas_audio": [1.0, 0.0],
    }
    pipeline._unpack_denoised_rows = lambda *args, **kwargs: ("video", "audio")

    before = {"actual_sparse_calls": 10}
    after = {"actual_sparse_calls": 60}
    snapshots = iter((before, after))
    logged: list[dict] = []
    monkeypatch.setattr(mod, "_minimax_h3_subblock_runtime_snapshot", lambda _: next(snapshots))
    monkeypatch.setattr(mod, "_log_minimax_h3_subblock_runtime", lambda **kwargs: logged.append(kwargs))
    monkeypatch.setattr(
        mod,
        "minimax_h3_denoise_loop",
        lambda **kwargs: (torch.ones(1), torch.ones(1)),
    )

    output = pipeline.diffuse(
        task="t2va",
        text_embeddings=torch.zeros(1),
        text_tags=torch.zeros(1, dtype=torch.long),
        seed=0,
        latent_t=1,
        latent_h=2,
        latent_w=2,
        audio_t=1,
        num_frames=1,
        num_steps=1,
        video_shift=1.0,
        audio_shift=1.0,
        base_schedule=None,
        visual_condition=None,
        visual_condition_shape=None,
        audio_condition=None,
        ref_audio_t=None,
    )

    assert output == ("video", "audio")
    assert logged == [{"task": "t2va", "before": before, "after": after}]
