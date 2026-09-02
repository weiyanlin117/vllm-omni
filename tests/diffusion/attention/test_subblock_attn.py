# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as torch_functional

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, PackedPaddingMetadata
from vllm_omni.diffusion.attention.backends.subblock.flashinfer_bsa import (
    load_bsa_attn_blk64_fwd,
    run_bsa_attn_blk64,
    validate_b200_bsa_available,
)
from vllm_omni.diffusion.attention.backends.subblock.router import (
    BLOCK_SIZE,
    RoutingPlan,
    SubBlockRouter,
    snap_topk_to_kernel_budget,
)
from vllm_omni.diffusion.attention.backends.subblock_attn import (
    SubBlockAttentionBackend,
    SubBlockAttentionImpl,
    SubBlockSchedule,
    collect_subblock_runtime_stats,
    subblock_runtime_delta,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _impl(**backend_kwargs):
    return SubBlockAttentionImpl(
        num_heads=2,
        head_size=128,
        softmax_scale=1.0 / math.sqrt(128),
        causal=False,
        num_kv_heads=2,
        prefix="transformer.blocks.3.attn",
        qkv_layout="BSND",
        backend_kwargs=backend_kwargs,
        role="self",
    )


def _manual_scores(q, k, *, n_q, n_k, scale):
    batch, q_len, heads, _ = q.shape
    k_len = k.shape[1]
    q_blocks = math.ceil(q_len / BLOCK_SIZE)
    k_blocks = math.ceil(k_len / BLOCK_SIZE)
    q_sub = BLOCK_SIZE // n_q
    k_sub = BLOCK_SIZE // n_k
    out = torch.empty(batch, heads, q_blocks, k_blocks, dtype=torch.float32)
    for b in range(batch):
        for h in range(heads):
            for qb in range(q_blocks):
                for kb in range(k_blocks):
                    logits = []
                    for qi in range(n_q):
                        q_start = qb * BLOCK_SIZE + qi * q_sub
                        q_stop = min(q_start + q_sub, q_len)
                        if q_start >= q_stop:
                            continue
                        q_mean = q[b, q_start:q_stop, h].float().mean(0)
                        for ki in range(n_k):
                            k_start = kb * BLOCK_SIZE + ki * k_sub
                            k_stop = min(k_start + k_sub, k_len)
                            if k_start >= k_stop:
                                continue
                            k_mean = k[b, k_start:k_stop, h].float().mean(0)
                            logits.append(torch.dot(q_mean, k_mean) * scale)
                    out[b, h, qb, kb] = torch.logsumexp(torch.stack(logits), dim=0)
    return out


def test_reference_scores_match_definition_with_ragged_tail():
    torch.manual_seed(0)
    q = torch.randn(1, 70, 2, 8)
    k = torch.randn(1, 95, 2, 8)
    router = SubBlockRouter(n_q=4, n_k=4)
    scale = 8**-0.5

    actual = router.reference_scores(q, k, scale)
    expected = _manual_scores(q, k, n_q=4, n_k=4, scale=scale)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "requested,num_blocks,expected",
    [(1, 590, 8), (148, 590, 152), (590, 590, 590), (3, 3, 3)],
)
def test_topk_budget_snaps_to_eight(requested, num_blocks, expected):
    assert snap_topk_to_kernel_budget(requested, num_blocks) == expected


def test_cpu_route_contract():
    torch.manual_seed(1)
    q = torch.randn(1, 130, 3, 16)
    k = torch.randn(1, 190, 3, 16)
    plan = SubBlockRouter().route(q, k, sparsity=0.75, softmax_scale=16**-0.5)

    assert plan.index.dtype == torch.int32
    assert plan.index.shape == (1, 3, 3, 3)
    assert plan.topk == 3 and plan.num_blocks == 3
    assert torch.all((0 <= plan.index) & (plan.index < plan.num_blocks))


def test_schedule_validation_and_defaults():
    assert SubBlockSchedule.from_backend_kwargs(None) == SubBlockSchedule()
    assert SubBlockSchedule.from_backend_kwargs({"sparsity": 0.8, "skip_first_steps": 5}).sparsity == 0.8
    with pytest.raises(ValueError, match="sparsity"):
        SubBlockSchedule.from_backend_kwargs({"sparsity": 1.0})
    with pytest.raises(ValueError, match="sparsity"):
        SubBlockSchedule.from_backend_kwargs({"sparsity": True})
    with pytest.raises(ValueError, match="n_q/n_k"):
        SubBlockSchedule.from_backend_kwargs({"n_q": 3})
    with pytest.raises(ValueError, match="min_seq_len"):
        SubBlockSchedule.from_backend_kwargs({"min_seq_len": 0})


def test_registry_exposes_explicit_subblock_backend():
    from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum

    assert DiffusionAttentionBackendEnum.SUBBLOCK_ATTN.get_path().endswith("subblock_attn.SubBlockAttentionBackend")


def test_backend_only_supports_single_document_mask_free_packing():
    assert SubBlockAttentionBackend.supports_packed_mask_free() is True
    assert SubBlockAttentionBackend.supports_multi_doc_packed_varlen() is False


@pytest.mark.parametrize(
    ("allgather_degree", "ring_degree", "message"),
    [(2, 1, "AllGather-KV"), (1, 2, "ring sequence parallelism")],
)
def test_attention_init_rejects_incompatible_sequence_parallelism(
    monkeypatch,
    allgather_degree,
    ring_degree,
    message,
):
    from vllm_omni.diffusion.attention import layer as layer_mod

    class FakeBackend:
        @staticmethod
        def get_name():
            return "SUBBLOCK_ATTN"

    config = SimpleNamespace(
        model_class_name="MiniMaxH3Pipeline",
        diffusion_attention_config=None,
        parallel_config=SimpleNamespace(
            allgather_degree=allgather_degree,
            ring_degree=ring_degree,
        ),
    )
    monkeypatch.setattr(layer_mod, "get_current_diffusion_config_or_none", lambda: config)
    monkeypatch.setattr(
        layer_mod,
        "get_attn_backend_for_role",
        lambda **kwargs: (FakeBackend, SimpleNamespace(backend="SUBBLOCK_ATTN")),
    )

    with pytest.raises(ValueError, match=message):
        layer_mod.Attention(
            num_heads=2,
            head_size=128,
            causal=False,
            softmax_scale=128**-0.5,
            qkv_layout="BSND",
        )


def test_b200_contract_rejects_b300_even_though_both_are_blackwell(monkeypatch):
    import vllm_omni.diffusion.attention.backends.subblock.flashinfer_bsa as mod

    mod._validate_b200_device.cache_clear()
    monkeypatch.setattr(mod.torch.cuda, "get_device_capability", lambda device: (10, 3))
    with pytest.raises(ValueError, match=r"B200/GB200.*10\.0.*10\.3"):
        mod._validate_b200_device(torch.device("cuda"))


def test_b200_contract_rejects_unvalidated_b100_on_same_sm100(monkeypatch):
    import vllm_omni.diffusion.attention.backends.subblock.flashinfer_bsa as mod

    mod._validate_b200_device.cache_clear()
    monkeypatch.setattr(mod.torch.cuda, "get_device_capability", lambda device: (10, 0))
    monkeypatch.setattr(mod.torch.cuda, "get_device_name", lambda device: "NVIDIA B100")
    with pytest.raises(ValueError, match="only been validated on B200/GB200.*B100"):
        mod._validate_b200_device(torch.device("cuda"))


def test_impl_rejects_non_h3_kernel_contract():
    kwargs = dict(
        num_heads=2,
        head_size=128,
        softmax_scale=128**-0.5,
        causal=False,
        num_kv_heads=2,
        prefix="blocks.0.attn",
        qkv_layout="BSND",
    )
    with pytest.raises(ValueError, match="head_dim=128"):
        SubBlockAttentionImpl(**{**kwargs, "head_size": 64})
    with pytest.raises(ValueError, match="MHA only"):
        SubBlockAttentionImpl(**{**kwargs, "num_kv_heads": 1})
    with pytest.raises(ValueError, match="causal"):
        SubBlockAttentionImpl(**{**kwargs, "causal": True})
    with pytest.raises(ValueError, match="qkv_layout='BSND'"):
        SubBlockAttentionImpl(**{**kwargs, "qkv_layout": "BNSD"})


def test_token_refiner_is_dense_even_when_long():
    impl = SubBlockAttentionImpl(
        num_heads=2,
        head_size=128,
        softmax_scale=128**-0.5,
        causal=False,
        num_kv_heads=2,
        prefix="token_refiner.blocks.0.attn",
        qkv_layout="BSND",
        backend_kwargs={"skip_first_steps": 0, "min_seq_len": 1},
        role="minimax_h3.token_refiner",
    )
    q = k = torch.empty(1, 4096, 2, 128, dtype=torch.bfloat16)
    assert impl._dense_reason(q, k) == "non_dit_role"


@pytest.mark.parametrize(
    ("q_shape", "dtype", "expected"),
    [
        ((1, 4095, 2, 128), torch.bfloat16, "short_sequence"),
        ((2, 4096, 2, 128), torch.bfloat16, "multi_batch"),
        ((1, 4096, 2, 128), torch.float16, "non_bf16"),
    ],
)
def test_ineligible_runtime_shapes_have_explicit_dense_reason(monkeypatch, q_shape, dtype, expected):
    import vllm_omni.diffusion.attention.backends.subblock_attn as mod

    impl = _impl(skip_first_steps=0, min_seq_len=4096)
    monkeypatch.setattr(mod, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(mod, "get_forward_context", lambda: SimpleNamespace(denoise_step_idx=0))
    q = k = v = torch.empty(q_shape, dtype=dtype)
    assert impl._dense_reason(q, k, v) == expected


def test_packed_padding_sparse_path_slices_real_document_and_zeroes_tail(monkeypatch):
    import vllm_omni.diffusion.attention.backends.subblock_attn as mod

    impl = _impl(skip_first_steps=0, min_seq_len=1, sparsity=0.75)
    monkeypatch.setattr(mod, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(mod, "get_forward_context", lambda: SimpleNamespace(denoise_step_idx=0))

    captured = {}

    def fake_route(q, k, sparsity, softmax_scale):
        captured.update(q_shape=q.shape, k_shape=k.shape, sparsity=sparsity, scale=softmax_scale)
        q_blocks = math.ceil(q.shape[1] / BLOCK_SIZE)
        k_blocks = math.ceil(k.shape[1] / BLOCK_SIZE)
        index = torch.zeros(1, q.shape[2], q_blocks, 1, dtype=torch.int32)
        return RoutingPlan(index=index, topk=1, num_blocks=k_blocks)

    def fake_bsa(q, k, v, index, topk, scale):
        captured.update(index_shape=index.shape, topk=topk, bsa_scale=scale)
        return q + 1

    monkeypatch.setattr(impl.router, "route", fake_route)
    monkeypatch.setattr(mod, "run_bsa_attn_blk64", fake_bsa)

    total, used = 192, 130
    q, k, v = (torch.randn(1, total, 2, 128, dtype=torch.bfloat16) for _ in range(3))
    cu_seqlens = torch.tensor([0, used, total], dtype=torch.int32)
    metadata = AttentionMetadata(
        packed_padding=PackedPaddingMetadata(
            q_length=used,
            kv_length=used,
            cu_seqlens_q=cu_seqlens[:2],
            cu_seqlens_k=cu_seqlens[:2],
        ),
        extra={
            "cu_seqlens_q": cu_seqlens,
            "cu_seqlens_k": cu_seqlens,
            "max_seqlen_q": used,
            "max_seqlen_k": used,
            "valid_kv_length": used,
        },
    )

    out = impl.forward_cuda(q, k, v, metadata)

    assert captured["q_shape"] == (1, used, 2, 128)
    assert captured["k_shape"] == (1, used, 2, 128)
    torch.testing.assert_close(out[:, :used], q[:, :used] + 1)
    assert torch.count_nonzero(out[:, used:]) == 0
    assert impl.runtime_stats() == {
        "actual_sparse_calls": 1,
        "dense_calls_by_reason": {},
        "last_plan_density": 1 / 3,
    }


def test_warmup_step_uses_dense_fallback_and_counts_reason(monkeypatch):
    import vllm_omni.diffusion.attention.backends.subblock_attn as mod

    impl = _impl(skip_first_steps=10, min_seq_len=1)
    monkeypatch.setattr(mod, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(mod, "get_forward_context", lambda: SimpleNamespace(denoise_step_idx=9))
    impl._dense_impl = SimpleNamespace(forward_cuda=lambda q, k, v, metadata: q + 2)
    monkeypatch.setattr(impl.router, "route", lambda *args, **kwargs: pytest.fail("router must not run in warmup"))
    q, k, v = (torch.randn(1, 128, 2, 128, dtype=torch.bfloat16) for _ in range(3))

    out = impl.forward_cuda(q, k, v)

    torch.testing.assert_close(out, q + 2)
    assert impl.runtime_stats()["dense_calls_by_reason"] == {"warmup_step": 1}


def test_rank_local_runtime_stats_aggregate_and_delta_across_wrappers():
    class ImplHolder(torch.nn.Module):
        def __init__(self, impl):
            super().__init__()
            self.attention = impl

    class ModelAttention(torch.nn.Module):
        def __init__(self, impl):
            super().__init__()
            self.attention = ImplHolder(impl)

    first = _impl(skip_first_steps=0, min_seq_len=1)
    second = _impl(skip_first_steps=0, min_seq_len=1)
    model = torch.nn.ModuleList([ModelAttention(first), ModelAttention(second)])
    before = collect_subblock_runtime_stats(model)

    first._sparse_calls = 3
    first._dense_calls.update({"warmup_step": 2})
    first._last_plan_density = 0.25
    second._sparse_calls = 4
    second._dense_calls.update({"warmup_step": 1, "short_sequence": 2})
    second._last_plan_density = 0.5
    after = collect_subblock_runtime_stats(model)

    assert before == {
        "configured_layers": 2,
        "sparse_eligible_layers": 2,
        "actual_sparse_calls": 0,
        "dense_calls_by_reason": {},
        "last_plan_density_min": None,
        "last_plan_density_max": None,
    }
    assert after["actual_sparse_calls"] == 7
    assert after["dense_calls_by_reason"] == {"short_sequence": 2, "warmup_step": 3}
    assert after["last_plan_density_min"] == 0.25
    assert after["last_plan_density_max"] == 0.5
    assert subblock_runtime_delta(before, after) == after


def test_ring_path_rejects_subblock_before_bypass():
    from vllm_omni.diffusion.attention.layer import Attention

    fake = SimpleNamespace(attention=SimpleNamespace(subblock_configured=True), ring_runner=None)
    with pytest.raises(NotImplementedError, match="SUBBLOCK_ATTN.*ring sequence parallelism"):
        Attention._run_ring_attention(fake, None, None, None, None)


def _has_b200_bsa() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        return False
    try:
        load_bsa_attn_blk64_fwd()
    except Exception:
        return False
    return True


requires_b200_bsa = pytest.mark.skipif(
    not _has_b200_bsa(),
    reason="requires B200/GB200 (SM100) and FlashInfer bsa_attn_blk64_fwd",
)


@hardware_test(res={"cuda": "B200"})
@requires_b200_bsa
def test_b200_preflight_checks_real_blk64_jit_assets():
    validate_b200_bsa_available()


@hardware_test(res={"cuda": "B200"})
@requires_b200_bsa
def test_b200_fused_router_matches_fp32_reference_with_tail():
    torch.manual_seed(2)
    q = torch.randn(1, 130, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 190, 2, 128, device="cuda", dtype=torch.bfloat16)
    router = SubBlockRouter(n_q=4, n_k=4)

    actual = router.scores(q, k, 128**-0.5)
    expected = router.reference_scores(q, k, 128**-0.5)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.1)


@hardware_test(res={"cuda": "B200"})
@requires_b200_bsa
def test_b200_blk64_all_blocks_matches_sdpa_for_strided_qkv_and_tail():
    torch.manual_seed(3)
    batch, seq_len, heads, head_dim = 1, 130, 2, 128
    qkv = torch.randn(batch, seq_len, heads, 3 * head_dim, device="cuda", dtype=torch.bfloat16)
    q = qkv[..., :head_dim]
    k = qkv[..., head_dim : 2 * head_dim]
    v = qkv[..., 2 * head_dim :]
    assert not q.is_contiguous() and not k.is_contiguous() and not v.is_contiguous()

    num_blocks = math.ceil(seq_len / BLOCK_SIZE)
    block_index = (
        torch.arange(num_blocks, device="cuda", dtype=torch.int32)
        .view(1, 1, 1, num_blocks)
        .expand(batch, heads, num_blocks, num_blocks)
        .contiguous()
    )
    out = run_bsa_attn_blk64(q, k, v, block_index, num_blocks, head_dim**-0.5).float()
    ref = torch_functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        scale=head_dim**-0.5,
    ).transpose(1, 2)

    relative_error = (out - ref.float()).abs().mean() / ref.float().abs().mean()
    assert relative_error < 0.02, f"all-block BSA relative error {relative_error:.4f} is too high"


@hardware_test(res={"cuda": "B200"})
@requires_b200_bsa
def test_b200_full_backend_runs_router_bsa_and_packed_tail(monkeypatch):
    import vllm_omni.diffusion.attention.backends.subblock_attn as mod

    impl = _impl(skip_first_steps=0, min_seq_len=1, sparsity=0.01)
    monkeypatch.setattr(mod, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(mod, "get_forward_context", lambda: SimpleNamespace(denoise_step_idx=0))

    torch.manual_seed(4)
    total, used, heads, head_dim = 192, 130, 2, 128
    qkv = torch.randn(1, total, heads, 3 * head_dim, device="cuda", dtype=torch.bfloat16)
    q = qkv[..., :head_dim]
    k = qkv[..., head_dim : 2 * head_dim]
    v = qkv[..., 2 * head_dim :]
    cu_seqlens = torch.tensor([0, used, total], device="cuda", dtype=torch.int32)
    metadata = AttentionMetadata(
        packed_padding=PackedPaddingMetadata(
            q_length=used,
            kv_length=used,
            cu_seqlens_q=cu_seqlens[:2],
            cu_seqlens_k=cu_seqlens[:2],
        ),
        extra={
            "cu_seqlens_q": cu_seqlens,
            "cu_seqlens_k": cu_seqlens,
            "max_seqlen_q": used,
            "max_seqlen_k": used,
            "valid_kv_length": used,
        },
    )

    out = impl.forward_cuda(q, k, v, metadata)
    ref = torch_functional.scaled_dot_product_attention(
        q[:, :used].transpose(1, 2),
        k[:, :used].transpose(1, 2),
        v[:, :used].transpose(1, 2),
        scale=head_dim**-0.5,
    ).transpose(1, 2)

    relative_error = (out[:, :used].float() - ref.float()).abs().mean() / ref.float().abs().mean()
    assert relative_error < 0.02
    assert torch.count_nonzero(out[:, used:]) == 0
    assert impl.runtime_stats() == {
        "actual_sparse_calls": 1,
        "dense_calls_by_reason": {},
        "last_plan_density": 1.0,
    }
