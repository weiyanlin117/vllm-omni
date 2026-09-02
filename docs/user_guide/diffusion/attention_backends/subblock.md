# SubBlock Attention

`SUBBLOCK_ATTN` is an explicit, B200-only backend for MiniMax H3 main-DiT
self-attention. It ports the training-free 64-token SubBlock router introduced
by SGLang PR [#34148](https://github.com/sgl-project/sglang/pull/34148) and calls
FlashInfer's stock `bsa_attn_blk64_fwd` kernel. It is not selected
automatically.

## Supported contract

The initial integration is deliberately narrow:

| Property | Supported value |
| --- | --- |
| GPU | NVIDIA B200/GB200, exactly SM100 (compute capability 10.0) |
| Q/K/V | BF16, BSND, non-causal MHA, `head_dim=128` |
| Sparse block | 64 tokens; key-block budget rounded up to a multiple of 8 |
| Packed input | One real document followed by structural suffix padding |
| Sequence parallelism | Ulysses; Ring and AllGather-KV are rejected |
| H3 role | Main `blocks.*` self-attention; token refiner stays dense |
| Dense fallback | Early denoise steps/layers, short sequences, masks, and unsupported runtime shapes |

Multi-request packed documents are not supported by this first version. Use
`--max-num-seqs 1`; otherwise H3 must issue one forward per request. B300,
H100/H200, FP16, GQA, causal attention, and Ring are intentionally rejected
instead of silently running an unvalidated path.

## FlashInfer runtime assets

The tested dependency is FlashInfer `v0.6.16.post3` at commit
`9dc1b2495b40314dec8a8cde8cd7faf5c5206702`. Its Python wheel exposes the BSA
symbol but does not contain all files needed by the kernel's first-call JIT.
Keep a recursive source checkout available at runtime:

```bash
git clone --depth 1 --branch v0.6.16.post3 \
  --recurse-submodules --shallow-submodules \
  https://github.com/flashinfer-ai/flashinfer.git /opt/flashinfer-src
python -m pip install --no-build-isolation -e /opt/flashinfer-src
```

This tag's blk64 source also includes `nvtx3/nvToolsExt.h`. If the CUDA image
does not ship the NVTX v3 headers, install NVIDIA's lightweight C/C++ header
branch and publish its include directory:

```bash
git clone --depth 1 --branch v3.6.0-c-cpp \
  https://github.com/NVIDIA/NVTX.git /opt/nvtx
export CPATH="/opt/nvtx/include${CPATH:+:${CPATH}}"
```

NVCC and Ninja must be available. Set `FLASHINFER_WORKSPACE_BASE` to persistent
storage so each worker image does not rebuild the extension after restart.
`SUBBLOCK_ATTN` probes the exact symbol, SM100 device, CUTLASS checkout, NVTX
header, NVCC, and Ninja during backend resolution. This catches the otherwise
misleading state in which import succeeds but the first request cannot compile.

Do not bump FlashInfer solely because a newer wheel imports. Treat the tag (and
its CUTLASS submodule) as one tested unit, then rerun the B200 numerical and
performance gates before changing it.

## Configuration

The following keeps the short token-refiner role explicitly dense and enables
SubBlock only for eligible main-DiT layers:

```bash
--diffusion-attention-config '{
  "default": {
    "backend": "SUBBLOCK_ATTN",
    "subblock": {
      "sparsity": 0.75,
      "skip_first_steps": 10,
      "skip_first_layers": 0,
      "n_q": 4,
      "n_k": 4,
      "min_seq_len": 24576
    }
  },
  "per_role": {
    "minimax_h3.token_refiner": {
      "backend": "TRTLLM_ATTN"
    }
  }
}'
```

`sparsity` is the requested fraction of key blocks dropped per query block.
The realized value can be lower because the stock kernel charges blocks in
groups of eight. The router records FP32 scores while its pooled Q/K tensors
are BF16, matching the upstream algorithm.

With the default `skip_first_steps=10`, a request containing ten or fewer
denoise steps intentionally records zero BSA calls. Do not lower that guard for
a distilled/Turbo schedule until a separate end-to-end quality test validates
the change.

The default `min_seq_len=24576` is a measured B200 guardrail, not an upstream
algorithmic constant. In same-device paired runs, 8K regressed and 16K did not
produce a speedup confidence interval wholly above 1.0, while 24K and longer
did. Re-register the crossover before lowering this value for another H3
shape, head count, FlashInfer revision, or parallel topology.

The initial synthetic crossover used one B200, BF16, 75% requested sparsity,
five complete warmups per arm, ten interleaved pairs, and seed 0. The 8K–32K
rows forced router+BSA to locate the crossover; production selection keeps the
first two dense. The final rows benchmark the real `SubBlockAttentionImpl`
dispatcher. Its 37,760-token inputs use H3's actual layouts: contiguous Q/K/V
after USP4, and contiguous Q/K plus strided V without sequence parallelism.

| Sequence | Heads/rank | Measured path | Median dense/selected speedup | Bootstrap 95% CI | Gate |
| ---: | ---: | --- | ---: | ---: | --- |
| 4,096 | 14 | production dense fallback | 0.985x | 0.960–1.014 | pass; 0 BSA calls |
| 8,192 | 14 | forced BSA crossover probe | 0.823x | 0.787–0.841 | reject BSA |
| 16,384 | 14 | forced BSA crossover probe | 1.148x | 0.742–1.151 | reject BSA; unstable |
| 24,576 | 14 | forced BSA crossover probe | 1.362x | 1.344–1.367 | pass |
| 32,768 | 14 | forced BSA crossover probe | 1.492x | 1.451–1.502 | pass |
| 37,760 | 14 | production BSA, USP4 rank | 1.517x | 1.507–1.532 | pass; 10 BSA calls |
| 37,760 | 56 | production BSA, single rank | 1.552x | 1.419–1.572 | pass; 10 BSA calls |

At 4,096 tokens and 14 heads, the dense baseline and production dispatcher
medians were 0.168 ms and 0.172 ms; the dispatcher recorded ten
`short_sequence` fallbacks, zero BSA calls, identical output, and the same
14.0 MiB incremental allocator peak.

At 37,760 tokens and 14 contiguous heads, dense and production SubBlock
medians were 7.243 ms and 4.787 ms. The sparse breakdown was 0.283 ms for
routing, 2.601 ms for `flash::fused_attn_device`, and 1.410 ms for
layout/packing and other wrapper CUDA work. The production sparse arm used
about 1,307 MiB of incremental allocator memory versus 130 MiB for dense
attention. Preserve VRAM headroom in the real model run; the current
FlashInfer wrapper trades extra packing storage for the long-sequence speedup.

## Correctness and performance gates

Run the B200 unit and numerical tests first:

```bash
pytest -q tests/diffusion/attention/test_subblock_attn.py \
  tests/diffusion/attention/test_attention_config.py \
  tests/diffusion/models/minimax_h3/test_minimax_h3_subblock_stats.py
```

Then prewarm both complete arms and run the paired microbenchmark on the same
B200:

```bash
python benchmarks/diffusion/bench_subblock_attn.py \
  --preset smoke --warmup 5 --iters 10 \
  --profile-trace subblock-smoke-trace.json \
  --output-json subblock-smoke.json

python benchmarks/diffusion/bench_subblock_attn.py \
  --preset h3-5s --heads 14 --input-layout contiguous \
  --warmup 5 --iters 10 \
  --output-json subblock-h3-5s.json
```

The second command models one USP4 rank (`56 / 4 = 14` heads); use
`--heads 56 --input-layout h3-no-sp` for a single-rank H3 attention shape.

The benchmark excludes first-call JIT, warms dense and the production
`SubBlockAttentionImpl` arm independently, and alternates their measured
order. It reports the selected path and fallback reason, router time, stock
BSA wrapper time, the actual `flash::fused_attn_device` CUDA event,
layout/packing CUDA work, paired bootstrap speedup confidence intervals, peak
VRAM, numerics, and the number of real BSA calls. It exits nonzero unless a
short sequence selects dense with less than 3% median regression, or a long
sequence selects BSA and the speedup CI lower bound exceeds 1.0.

FlashInfer `v0.6.16.post3` is not a zero-copy wrapper: it first normalizes
Q/K/V to contiguous BSHD and then performs additional B/H/S and 64-token block
packing inside the extension. Compare the complete sparse arm against the
complete dense arm; a fast isolated `fused_attn_device` event does not by
itself establish an end-to-end speedup.

The synthetic benchmark is a kernel gate, not proof that MiniMax H3 is
integrated. Request-mode H3 logs one rank-local summary after denoising,
including configured and eligible layer counts, actual BSA calls, dense
fallback reasons, and the observed plan-density range.

## Single-B200 real FL2VA gate

The completed real-model gate ran on 2026-09-01 against MiniMax H3 snapshot
`42ed227ee7df40d41602854ae760620d6eb651fe`. It used one B200, BF16, CPU model
offload, no sequence parallelism, one request at a time, one real FL2VA
reference image, 1344x768 output, 24 FPS, five requested seconds, and 50
inference steps. Every output contained 124 frames and 5.175 seconds of H.264
video plus stereo 32 kHz AAC audio.

Dense `TRTLLM_ATTN` and `SUBBLOCK_ATTN` ran serially inside one decorated Modal
function invocation. Both arms recorded the same container boot nonce, process
identity, host boot ID, task ID, and physical GPU UUID
`GPU-0f8c9915-869f-9734-e9a5-f42cea824260`. Each arm received one complete
warmup request followed by five measured requests over three prompts and seeds
1702 through 1706. Server initialization, model loading, first-call JIT, and
warmup were excluded.

| Metric | Dense median | SubBlock median | Ratio of medians | Median paired speedup (exact bootstrap 95% CI) |
| --- | ---: | ---: | ---: | ---: |
| Client wall time | 161.053 s | 132.999 s | 1.211x | 1.211x (1.210-1.214) |
| Server inference time | 161.033 s | 132.931 s | 1.211x | 1.212x (1.211-1.213) |
| DiT `diffuse` time | 154.236 s | 125.958 s | 1.225x | 1.224x (1.223-1.226) |
| Prompt encoding | 1.196 s | 1.208 s | 0.990x | 0.947x (0.778-1.054) |
| Visual-condition encoding | 0.494 s | 0.509 s | 0.971x | 1.015x (0.917-1.251) |
| VAE/audio decode | 4.617 s | 4.620 s | 0.999x | 0.999x (0.956-1.027) |
| Sampled whole-run peak HBM | 73,824 MiB | 76,550 MiB | - | +2,726 MiB for SubBlock |

The confidence interval is the exact five-out-of-five percentile bootstrap of
the five matched seed/prompt/reference-image speedups. All five client-wall
pairs were faster. The server-scoped arms were grouped rather than alternated
per request, so the interval does not cover a systematic arm-order or
long-timescale machine drift effect. The dispatcher microbenchmark above is
the interleaved kernel evidence. Reciprocal arm-order end-to-end evidence is
needed before extending this experimental result into a broader production
claim.

The benchmark used the SGLang `#34148` schedule: 75% requested sparsity,
`n_q=n_k=4`, the first ten denoise transitions dense, and no skipped DiT
layers. The five-second sequence had 39,774 active tokens (622 key blocks) and
retained 160 blocks per query block, for plan density 0.257235 and 74.2765%
realized sparsity. A 50-step
request produced 49 denoise transitions: every measured request logged 500
early dense calls and 1,950 real BSA calls, or 50 main-DiT layers times 10
dense plus 39 sparse transitions. Token-refiner attention stayed dense. The
run intentionally set `min_seq_len=4096` to reproduce the upstream policy;
the observed sequence was also above the safer 24,576-token production
guardrail.

All ten measured MP4s passed full server-side ffmpeg decode and an independent
local full-frame decode. Same-prompt, same-seed diagnostics over all five pairs
produced:

| Diagnostic | Median | Range |
| --- | ---: | ---: |
| Native 1344x768 PSNR, all 124 frames | 25.960 dB | 18.969-29.205 dB |
| Native 1344x768 SSIM, all 124 frames | 0.8838 | 0.6927-0.9034 |
| SqueezeNet LPIPS mean at 672x384 | 0.0407 | 0.0339-0.1306 |
| Audio log-mel MAE | 3.205 dB | 2.510-4.256 dB |
| Audio waveform correlation | 0.907 | 0.880-0.933 |
| Reference-to-first-frame normalized MAE | dense 0.01054 | SubBlock 0.01064 |
| Audio/video stream-duration skew | 8.333 ms for both arms | arm delta 0 ms |

A non-blind 12-frame contact-sheet inspection per pair found no obvious
SubBlock-only object duplication, identity break, motion discontinuity, or
gross corruption. This is diagnostic evidence, not a quality-equivalence
pass: a dense-versus-dense repeat distribution and threshold were not
preregistered, LPIPS used the SqueezeNet rather than AlexNet/VGG backbone, and
stream-duration skew is not content-level AV synchronization. Semantic video
scoring and human blind review remain open.

The recorded end-to-end evidence is intentionally scoped to five-second
FL2VA. No 10/15-second result or extrapolation is claimed here. Longer-duration
coverage remains follow-up work and must use the same complete warmup, five
matched measurements, and single-container identity checks before it is
reported.

The Modal source overlay also carried a stale generated
`vllm_omni/_version.py` value (`0.22.0rc2.dev14`) on top of vLLM 0.28.0 even
though the checkout itself was `v0.28.0rc1-76-g6f16f954`; rebuild the checkout
as a clean package for future benchmark expansion. NCU was absent from the
tested image, so no end-to-end kernel-level NCU claim is made. Treat the
single-B200 five-second integration and performance result as experimental;
registered quality equivalence, reciprocal arm order, and longer-duration
generalization are not claimed by this integration.
