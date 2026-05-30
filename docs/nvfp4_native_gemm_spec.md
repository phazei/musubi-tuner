# Spec: Native Blackwell FP4 GEMM for LTX-2 NVFP4 Models

## Status (read this first)

The native FP4 forward GEMM is **implemented, correct, and merged** into
`nvfp4_utils.py`. It is **not yet faster than fp8 end-to-end**, and the reason is
now precisely understood: the training step is **CPU-kernel-launch-bound** across
the *whole model*, not GEMM-bound and **not quant-bound**. The previously-proposed
fused Triton W4A4 quant kernel has been **ruled out** by direct measurement (see
"QuTLASS evaluation" and the corrected launch-count profile below): the NVFP4
quant is already a tiny fraction of the ~48k launches/step, so fusing it cannot
move the needle. This document is the complete handoff: what was built, what was
measured, what was ruled out, and why the remaining gap is a whole-model eager
dispatch problem, not an NVFP4-specific one.

| Config | s/step (skintest, 512x768x1, 12 steps, eager+sage+gradckpt) |
|---|---|
| fp8 (baseline to beat) | **~2.3** |
| NVFP4 original (dequant→bf16, W4A16) | ~4.4 |
| NVFP4 native FP4 (this work), eager helpers | ~4.45 |
| NVFP4 native FP4 (this work), **compiled quant+dequant** | **~3.15–3.46** |
| NVFP4 native FP4 + QuTLASS fused quant (measured) | **~3.20 (NOT faster)** |
| NVFP4 native FP4 + fused Triton quant kernel | RULED OUT (would shave ≤2.6k of ~48k launches) |

Environment is unchanged and verified:
- GPU: RTX 5090, sm_120 (Blackwell), compute cap **(12, 0)**.
- torch **2.12.0+cu130**, CUDA 13.0.
- venv python (use THIS): `D:\AITools\LTXTraining\musubi-tuner\venv\Scripts\python.exe`
  (global python is CPU-only torch 2.8 — do not use it).

---

## What this task is (unchanged classification)

This is **accelerated QLoRA**, not fully-quantized training (FQT). The base is
frozen; we only promote the frozen base's forward GEMM from "dequant-to-bf16 then
bf16 matmul" (W4A16) to a true fp4 tensor-core GEMM (W4A4).

Three regimes, do not conflate:
1. **Quantize-for-storage (QLoRA-style)** — base 4-bit, dequant to bf16 before
   matmul. Near-lossless but no compute speedup. This was the original slow path.
2. **Fully-quantized training (FQT)** — forward AND backward (incl. weight-grad)
   in fp4. The hard regime. **NOT this task.**
3. **PTQ inference** — 4-bit inference only.

**This task is regime 1 with the forward GEMM promoted to native fp4 (W4A4).**
The base is frozen for LoRA/slider training, so the **weight-gradient GEMM does
not exist** — the single most fragile FQT operation is simply absent.

### Why the quality risk is low (unchanged, validated by design)
The slider loss is **self-distillation against the same frozen base**: the 3-pass
target (`pred_pos`, `pred_neu`, `pred_neg`) is produced under `torch.no_grad()`
with LoRA disabled (`set_multiplier(0.0)`), `.detach()`-ed; the trained passes
(`±1`) produce base+LoRA and the loss is `MSE(base+LoRA, detached_target)`. The
base's quantization error appears on **both sides of the MSE and largely cancels**.
`_norm_like_tensor` strips magnitude; `direction = pred_pos - pred_neg` is a
difference through identical weights. The W4A4 activation-quant error (see below)
is structurally suppressed by this loss. The one residual risk is the
**train/deploy precision mismatch** (trained on fp4 base, deployed on fp8/fp16 in
ComfyUI), which must be **validated in a real training+deploy run**, not assumed.

---

## Precision Contract (honored by the implementation)

| Component | Forward | Backward | Notes |
|---|---|---|---|
| **fp4-marked base Linears** (`_nvfp4_quantized=True`, **528** for this 22B ckpt) | native fp4 `scaled_mm`, **W4A4** | **grad_input only, bf16** (dequant weight) | no grad_weight (frozen) |
| **Non-fp4 base layers** (adaln, connectors, timestep embedders, early blocks) | **bf16** (kept as stored) | bf16 | decision: keep bf16; checkpoint stored them bf16 for dynamic range / sensitivity |
| **LoRA / slider adapters** | bf16/fp32 | bf16/fp32 | never quantized |

Rules honored:
- Never add a layer to the fp4 set beyond the checkpoint's own mark.
- **No `--fp8_base --fp8_scaled` with NVFP4.** NVFP4 is auto-detected only when
  those flags are absent. Run with `NO_FP8=1` in the harness (or simply omit the
  flags). The non-fp4 layers run bf16 internally — no user flags, no `NO_FP8`
  workaround needed at the model level (the harness still gates the flags).

---

## How the native path works (implemented)

### The `scaled_mm` FP4 call (proven on this exact torch build)
```python
import torch.nn.functional as F
from torch.nn.functional import ScalingType, SwizzleType
out = F.scaled_mm(
    a_fp4, b_fp4.t(),                                  # both torch.float4_e2m1fn_x2
    scale_a=[a_block_scale_swizzled, a_tensor_scale],  # 2-level: block + per-tensor
    scale_recipe_a=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
    swizzle_a=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
    scale_b=[b_block_scale_swizzled, b_tensor_scale],
    scale_recipe_b=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
    swizzle_b=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
    output_dtype=torch.bfloat16,
)
```
- `a` = activation `[M, K]` packed to `[M, K//2]`; `b` = weight `[N, K]` packed to
  `[N, K//2]`, passed as `b_fp4.t()`. Output `[M, N]`.
- The two-level scale (block + per-tensor) is passed **as a 2-element list** — the
  kernel applies both. Maps directly to the checkpoint's `weight_scale` (block,
  e4m3) and `weight_scale_2` (per-tensor, f32).
- **All 528 fp4 layers have K%32==0 and N%16==0** — no alignment edge cases. M
  (sequence length) can be arbitrary; the kernel/scale-swizzle pad it internally.

### The scale swizzle (critical, fully solved)
The checkpoint stores block scales in **NATURAL (un-swizzled) layout, row-padded
to a multiple of 128**, shape `[pad128(N), K//16]`, dtype e4m3. The Blackwell
kernel requires the `SWIZZLE_32_4_4` (128×4 tile / 32×4×4 internal) layout.

**Verified empirically** (large layer N=4096): passing the stored scale as-is
("already swizzled") gives **0.56 rel-err** (garbage); trimming to N rows and
applying `to_blocked()` gives **0.003** (correct). So:
1. Trim the stored block scale to the real `out_features` rows.
2. Apply `to_blocked()` (the `torch.testing._internal.common_quantized` helper,
   round-trip-verified exact). This is done **once at load** for weights.

`swizzle_block_scale()` in `nvfp4_utils.py` is the vendored `to_blocked`.

### Activation quantization (W4A4 is forced — no native bf16×fp4 path)
Activations are quantized to fp4 **every forward**, deterministic round-to-nearest
(RTN), per 1×16 block, two-level scaling matching the weight recipe. RTN
determinism is **required** so `torch.utils.checkpoint` recompute reproduces the
forward exactly (verified: recompute max|diff| == 0.0).

Implementation note: RTN is done with **`torch.bucketize`** against the 7 E2M1
midpoints `[0.25,0.75,1.25,1.75,2.5,3.5,5.0]` — exact match to LUT-argmin and ~13x
faster. The sign bit is the high nibble bit. The **activation-quant error floor is
~9.5% rel** for Gaussian activations (the oracle per-block fp4 floor is 9.42%; our
impl is 9.51% — i.e. optimal). This is the inherent fp4 activation error the
self-distillation loss absorbs; it is NOT a bug and cannot be reduced without
changing the format.

### Backward (grad_input only, bf16)
`grad_input = grad_output @ W_decode`, where `W_decode` is the **exact bf16 decode
of the frozen fp4 weights** (`dequantize_nvfp4_weight`). Straight-through for the
activation quant. No grad_weight (frozen). Verified: grad_input vs analytic `g@W`
= 1.0e-4 (bf16 noise).

A **second fp4 GEMM in the backward was investigated and REJECTED** — see
"Ruled out" below.

---

## Code map (current implementation)

`src/musubi_tuner/modules/nvfp4_utils.py`:
- `swizzle_block_scale(scale)` — vendored `to_blocked` (SWIZZLE_32_4_4). Used at
  load time for weights.
- `quantize_activation_to_nvfp4(x_2d)` — RTN fp4 quant; **returns the block scale
  already swizzled** (swizzle folded into the compiled quant graph). Wraps
  `_quantize_activation_impl` (+ `_swizzle_block_scale_compilable`), compiled via
  `torch.compile(dynamic=True, fullgraph=True)` unless `NVFP4_NO_COMPILE=1`.
- `dequantize_nvfp4_weight(...)` — bf16 decode (backward + non-native fallback).
  Wraps `_dequantize_nvfp4_weight_impl`, also compiled.
- `_nvfp4_scaled_mm(...)` — the `F.scaled_mm` FP4 call.
- `NVFP4LinearFunction(autograd.Function)` — forward = quant→`scaled_mm` (+bias);
  backward = bf16 `grad_output @ W_decode`. Composes with `torch.utils.checkpoint`.
- `nvfp4_linear_forward_patch(self, x)` — calls `NVFP4LinearFunction.apply`, with a
  legacy dequant fallback if `scaled_mm`/swizzle types are unavailable.
- `apply_nvfp4_monkey_patch(model, state_dict)` — registers buffers `weight`
  (uint8), `nvfp4_block_scale` (natural, for backward decode),
  `nvfp4_block_scale_swizzled` (flat swizzled, for forward), `nvfp4_tensor_scale`;
  sets `_nvfp4_quantized=True`; swaps `forward`.
- `load_nvfp4_state_dict(...)` — keeps weights packed; **trims block-scale row
  padding to N and pre-swizzles once**, storing both natural and swizzled scales.

`src/musubi_tuner/ltx2_train_network.py`:
- `_ensure_nvfp4_buffers_on_device(model)` — co-locates all 4 NVFP4 buffers
  (mirrors `_ensure_nf4_buffers_on_device`; respects block-swap / model-parallel).
- Chained into `_ensure_fp8_buffers_on_device` (covers the slider's 3 unconditional
  call sites) and added as the `else` branch at the 3 `if fp8 / elif nf4` sites.

`src/musubi_tuner/ltx2_model_loading.py`:
- NVFP4 auto-detect gate (~571): `if not nf4_base and not fp8_scaled`. Loads via
  `load_nvfp4_state_dict` + `apply_nvfp4_monkey_patch`, then
  `load_state_dict(assign=True)` + `base_model.to(device)` (moves all buffers).

`src/musubi_tuner/ltx2_train_slider.py`:
- `_text_slider_step` (~527): 3-pass no-grad target (mult 0.0) + two grad passes
  (±1). Confirms the autograd topology the backward relies on.
- Debug hooks (env-gated, off by default, keep for the Triton work):
  `NVFP4_STEPTIME=1` prints per-step wall time; `NVFP4_TORCHPROF=1` dumps a
  torch.profiler CPU/CUDA op table at step 4.

Other env toggles:
- `NVFP4_NO_COMPILE=1` — eager quant/dequant (debugging / A/B).
- `NVFP4_NO_EMPTY_CACHE=1` (in `utils/device_utils.py`) — skip the per-pass
  `torch.cuda.empty_cache()`; saves ~0.2s/step. Consider making this the default.
- `NVFP4_USE_QUTLASS=1` — route the activation quant through QuTLASS's fused
  `fusedQuantizeNv` kernel instead of the compiled RTN path (opt-in; off by
  default; auto-falls-back if `import qutlass` fails). Correct + deterministic but
  **~2% slower end-to-end** — see "QuTLASS evaluation". Implemented in
  `nvfp4_utils._quantize_activation_qutlass`.

---

## Validation (what's proven)

1. **Kernel correctness (exact, identical-fp4-operands).** Native `scaled_mm`
   vs a bf16 dequant reference of the same fp4 operands, across all real
   checkpoint layer shapes: **worst 0.0036 rel-err** (hardware bf16-MMA precision).
   Catches swizzle/scale-layout/packed-dim bugs. Test:
   `C:\Users\HomeStar\AppData\Local\Temp\opencode\compiletest\kernel_correctness.py`.
2. **Module forward/backward/checkpoint.** Real-weight single-Linear test:
   kernel 0.0032, backward grad_input 1.0e-4, checkpoint recompute determinism
   exactly 0.0. Test: `...\compiletest\module_smoke.py`.
3. Still TODO (yours): **training-equivalence** (loss tracks the old path) and
   **deploy-precision** (adapter trained on fp4 base transfers to fp8/fp16 base in
   ComfyUI).

Helper scripts in `C:\Users\HomeStar\AppData\Local\Temp\opencode\compiletest\`:
`kernel_correctness.py`, `module_smoke.py`, `bench_layer.py`, `fp4_probe.py`,
`scale_layout_probe.py`, the run harness `run2.ps1` / `run_vc.cmd`
(`<label> <attn> <compile> <gradckpt> [ckpt]`, `NO_FP8=1` for NVFP4), and the
QuTLASS-evaluation scripts `qutlass_probe.py` (recipe/correctness),
`qutlass_bench.py` (per-layer latency), `qutlass_launchcount.py` (per-op launch
count).

---

## THE PERFORMANCE PROBLEM (diagnosed and closed — read before optimizing)

### Per-layer, NVFP4 is already faster than fp8
Microbench (real W[8192,4096] layer, M=4096), compiled:
| Op | NVFP4 | fp8-equivalent |
|---|---|---|
| forward (quant + fp4 gemm) | **0.357 ms** | 0.715 ms (dequant + bf16 linear) |
| backward dequant | 0.197 ms | — |
| backward bf16 matmul `g@W` | 0.769 ms | 0.769 ms (**identical, irreducible floor**) |

So the GEMM is not the problem. The native fp4 forward is ~2x faster than a bf16
linear, and the backward matmul is the same as fp8's.

### The real bottleneck: CPU kernel-launch overhead (now broken down by op)
`torch.profiler` over one real training step (NVFP4, eager+sage+gradckpt):
- **Self CPU total = 3297 ms vs Self CUDA total = 1549 ms** — the GPU is idle
  ~53% of the time waiting on the CPU.
- **`cudaLaunchKernel` = 481 ms across ~47,900 launches/step** (~10 µs each).

**Per-op launch breakdown (the important correction):** the launches are spread
across the *whole eager model*, NOT concentrated in the NVFP4 quant:

| aten op | # calls/step | what it is |
|---|---|---|
| `aten::_scaled_mm_v2` | **2,640** | the NVFP4 fp4 GEMM — exactly 528 layers × 5 passes |
| `aten::mm` | 6,238 | LoRA + attention projections (not quant) |
| `aten::mul` | 11,999 | elementwise (norms, scales, LoRA, attention) |
| `aten::copy_` | 17,815 | dtype casts + block-swap / offload movement |
| `aten::_to_copy` | 14,550 | dtype casts |
| `aten::add` | 8,378 | residuals, bias, LoRA add |
| `aten::view` | 29,726 | (free, no launch) |

The model has 528 fp4 Linears, each executed ~5×/step (3-pass target is one
batched call; two grad passes; gradient-checkpointing recomputes each grad-pass
forward in the backward). **The NVFP4 quant + GEMM is already efficient on the CPU
axis: the entire forward GEMM is one `_scaled_mm_v2` launch per layer-pass (2,640
total), and the compiled quant collapses to ~1 `cudaLaunchKernel` per call.** The
~48k launches come overwhelmingly from `mul`/`copy_`/`to_copy`/`add`/`mm` — the
generic eager transformer, dtype casts, LoRA, and block-swap movement.

**This reframes the problem:** it is a *whole-model eager dispatch* problem, not an
NVFP4-quant problem. fp8 is faster because its layers are 1 fused `addmm` AND
because it pays for far fewer of these same generic eager ops along the way — not
because NVFP4's quant is slow.

**This is why end-to-end is ~3.15–3.46s while per-layer NVFP4 work is faster than
fp8: the gap lives in the model-wide launch storm, which no NVFP4-local kernel can
fix.**

### What was tried and its effect
- LUT-argmin → `torch.bucketize` quant: large per-op win (kept).
- `torch.compile` the quant + dequant leaf helpers (NOT the model graph — distinct
  from the no-whole-model-compile non-goal): **4.45s → 3.46s** (kept).
- Skip per-pass `empty_cache()`: ~0.2s (kept, env-gated).
- Folding the scale-swizzle into the compiled quant graph: **no end-to-end gain**
  (kernel-shaving can't fix a 58k-launch problem). Kept anyway (one fewer launch,
  harmless).
- `torch.compile(mode="reduce-overhead")` / CUDA graphs on the quant alone: no
  better than default compile in isolation; risky with dynamic shapes + autograd.

**Conclusion: incremental NVFP4 kernel-shaving is exhausted AND cannot help — the
NVFP4 quant is not the launch bottleneck (it is 2,640 of ~48k launches). The
remaining gap is whole-model eager dispatch, addressable only by model-wide means
(CUDA graphs / a compile strategy that survives the slider's grad-mode flips),
which is out of scope for the NVFP4 work and separately shown unprofitable for the
slider — see `ltx2_slider_why_not_compile.md`.**

---

## QuTLASS evaluation (done — fused quant kernel does NOT help; keep as opt-in)

A community library, **QuTLASS** (IST-DASLab, CUTLASS-backed FP4 BLAS for
Blackwell), provides exactly the "fused NVFP4 activation quant" kernel that the
old T1 plan proposed to hand-write. It was fully evaluated on this machine.

**Build (Windows/MSVC + CUDA 13.0 + torch 2.12, sm_120):** QuTLASS does not build
out of the box on Windows. Two classes of fix were required (documented here so a
future rebuild is fast):
1. **`asm volatile(...)` PTX inside `CUTLASS_HOST_DEVICE` functions** — MSVC's host
   compiler tries to parse the GNU inline-asm and fails (`expected a "("`). Guard
   each such body with `#ifdef __CUDA_ARCH__` and a trivial host fallback. Files:
   `qutlass/csrc/include/cutlass_extensions/epilogue/threadblock/epilogue_quant.h`
   (5 fns) and `.../epilogue/fusion/sm100_visitor_store_tma_warpspecialized.hpp`
   (3 fns). The sm120 mma asm in `arch/mma_mx_sm120.h` is already guarded by
   `CUTLASS_ARCH_MMA_SM120_ENABLED` — leave it.
2. **`self.template op_NN(...)` dependent-member calls** — MSVC rejects the
   `.template` disambiguator without explicit `<...>`. Strip `.template` (7 sites
   in `epilogue_quant.h`). Build `sm_120a` only (drop `sm_100a` from `setup.py`
   `get_cuda_arch_flags`) to halve compile time. Install via a `.pth` pointing at
   the repo (a normal `pip install -e` re-runs the long cmake every import attempt).

**API used:** `qutlass.fusedQuantizeNv(x, h, global_scale, method="abs_max")` →
`(packed_e2m1 [M,K//2] uint8, e4m3 block scale padded to [pad128(M), K//16],
NATURAL layout)`. Pass an **identity** `h` (16×16 bf16) for "no rotation". Its
scale convention: returned e4m3 = `block_amax`, dequant divides by `alpha=6`, so it
maps to our two-level `scaled_mm` recipe as `tensor_scale = 1/6`, block scale used
as-is, then `swizzle_block_scale` (trim to M first). QuTLASS's own
`matmul_nvf4_bf16_tn` **fails on sm_120** ("Unsupported CUDA arch") — its CUTLASS
GEMM is not enabled for the 5090; only the fused quant kernel runs. We do not need
it: our `F.scaled_mm` is already optimal.

**Results (all measured, not assumed):**
- *Correctness:* QuTLASS abs_max activation quant matches our RTN quality exactly —
  W4A16 rel-err **0.0949** (ours: 0.0948), the inherent fp4 activation floor. The
  full GEMM via our `scaled_mm` is correct; backward grad_input 1.05e-4; **checkpoint
  recompute determinism 0.0** (bit-identical — gradient-checkpointing safe).
  (`module_smoke.py` reports its W4A4 "fail" only because its *reference* uses our
  RTN scale math, not QuTLASS abs_max — a reference-scheme mismatch, not a kernel
  bug; the W4A16 number, which is scheme-independent, is identical.)
- *Per-layer microbench:* QuTLASS fused quant + our `scaled_mm` is **1.2–1.7×
  faster than our compiled quant + `scaled_mm`** on raw GPU time.
- *End-to-end (the only test that matters):* **baseline ~3.15s/step vs QuTLASS
  ~3.20s/step — QuTLASS is ~2% SLOWER.** Reason: our compiled quant already
  collapses to `cudaLaunchKernel=1`/call; QuTLASS in eager issues ~4 CPU launches
  /call (fused-quant + swizzle ops + scaled_mm). The per-layer GPU win is irrelevant
  because the step is CPU-launch-bound, and QuTLASS *adds* CPU launches.

**Decision:** keep the integration as an **opt-in, off by default** (`NVFP4_USE_QUTLASS=1`,
implemented in `nvfp4_utils._quantize_activation_qutlass`, with automatic fallback
to the compiled path if `import qutlass` fails). It is correct and deterministic
and may help on a future driver/torch where the eager-launch overhead changes or
where QuTLASS's GEMM gets sm_120 support — but **it is not the default and does not
beat the baseline today.** The qutlass package itself is not a project dependency;
the patched build lives at `C:\Users\HomeStar\AppData\Local\Temp\opencode\qutlass`.

---

## Ruled out (do not re-attempt without new information)

0. **Fused Triton W4A4 quant kernel (the former "T1 next step").** RULED OUT by the
   per-op launch profile: the NVFP4 quant+GEMM is only **2,640 of ~47,900
   launches/step**, and the compiled quant is already ~1 CPU launch/call. Fusing it
   into a single Triton kernel would remove a rounding error's worth of the launch
   storm. The QuTLASS experiment (above) is the empirical proof: a *real* fused
   quant kernel, faster per-layer, made the step ~2% **slower** because the bottleneck
   is whole-model eager dispatch (`mul`/`copy_`/`to_copy`/`add` across attention,
   norms, LoRA, block-swap), not the quant. Do not write the Triton quant kernel.
1. **fp4 backward GEMM (W4A4 backward).** `grad_input = grad_output @ W` contracts
   over **N**, but the checkpoint packs/scale-blocks W over **K**. A native fp4
   backward needs W packed over N. Options and why they fail:
   - Pre-compute an N-packed transposed fp4 weight at load: **+9.6 GB VRAM**
     (the packed weight is 8.5 GB + 1.07 GB scales), which **erases NVFP4's entire
     reason to exist** (it exists to use *less* VRAM than fp8).
   - Re-quantize `W.t()` per backward: requires dequantizing W first — no win.
   - Also, the backward bf16 matmul is the **irreducible floor shared with fp8**;
     the only NVFP4-specific backward cost is the 0.2 ms dequant. Upside is tiny.
   Verified feasible numerically (rel 0.0036) but not viable on VRAM. **Dead end.**
2. **Whole-model `torch.compile`.** Separately determined to give no benefit on
   this model (graph breaks). Do not combine with this work. (Compiling the
   isolated quant/dequant *leaf helpers* is fine and is what we do.)
3. **Adding layers to the fp4 set / raising non-fp4 layers to fp4.** Honor the
   checkpoint's mixed-precision map exactly.

---

## What's left, and what is now out of scope

The NVFP4 forward GEMM goal is **functionally complete**: correct, deterministic,
checkpoint-safe, and per-layer faster than fp8. The end-to-end-faster-than-fp8 goal
is **not reachable by any NVFP4-local change** — the profile (above) and the
QuTLASS experiment together prove the bottleneck is whole-model eager dispatch.

Things that have now been measured and abandoned (so nobody re-tries them):
- **Fused quant kernel (Triton OR QuTLASS):** the quant is 2,640 / ~48k launches;
  a faster fused quant made the step ~2% slower (QuTLASS). Dead.
- **Out-writing `F.scaled_mm`:** it is already optimal (~0.116 ms for a 4096³ GEMM);
  QuTLASS's own GEMM doesn't even compile-enable on sm_120.

The only thing that *could* close the gap is a **model-wide** launch reducer:
- **CUDA-graph capture** of the per-block forward/backward (keyed on (M,K,N)).
  Real risk: dynamic sequence length (M varies) and the gradient-checkpointing
  recompute boundary and the slider's 3-pass grad-mode flips. This is a
  *whole-model* effort, not NVFP4 work, and overlaps the (negative) findings in
  `ltx2_slider_why_not_compile.md`. If anyone pursues faster slider training, that
  is the doc to start from — not this one.

Practical recommendation: **ship NVFP4 as-is for its VRAM benefit** (it exists to
fit the 22B model in less memory than fp8), accept ~3.15s/step, and treat the
"beat fp8 on speed" target as closed-unreachable for the NVFP4 layer in isolation.

### How to benchmark / iterate
- Per-step truth: `NVFP4_STEPTIME=1` (ignore step 1 = compile warmup; read the
  steady-state delta). Cumulative `s/it` in the progress bar is misleading.
- Op breakdown: `NVFP4_TORCHPROF=1` (dumps CPU vs CUDA totals + top ops + the
  `cudaLaunchKernel` count at step 4). The target metric is **Self CPU total ≈ or
  below Self CUDA total** and a large drop in launch count.
- Do NOT wrap per-op `torch.cuda.synchronize()` for timing inside the forward — it
  serializes async kernels and produces wildly inflated, misleading numbers (this
  caused an early misdiagnosis that the quant cost ~1.1s; it does not).
- Models: NVFP4 `D:\AITools\StabilityMatrixData\Models\DiffusionModels\ltx\ltx-2.3-22b-dev-nvfp4.safetensors`;
  fp8 baseline `...\ltx-2.3-22b-dev.safetensors`.
- Recommended config: eager + sage + gradient_checkpointing, skintest, 512x768x1,
  ~8-12 steps. Run with `NO_FP8=1` for NVFP4.

---

## Definition of done (updated)
1. ✅ NVFP4 fp4-layer forward uses native `scaled_mm` FP4 GEMM (W4A4), scales
   pre-swizzled at load.
2. ✅ Kernel-correctness test passes (worst 0.0036).
3. ✅ Backward grad_input in bf16 from the frozen dequant weight; works under
   autograd AND `torch.utils.checkpoint` (deterministic RTN). 
4. 🚫 **End-to-end NVFP4 s/it <= fp8 (~2.3).** ~3.15s, CLOSED AS UNREACHABLE by any
   NVFP4-local change: the step is whole-model eager-launch-bound (NVFP4 quant is
   only 2.6k of ~48k launches; a measured faster fused quant kernel — QuTLASS — made
   it ~2% slower). Closing this needs a model-wide CUDA-graph/compile effort that is
   out of scope here and separately unprofitable for the slider. NVFP4's value is
   VRAM, not speed.
5. ⏳ Deploy-precision validation (adapter trained on fp4 base → fp8/fp16 base in
   ComfyUI). Owner: user.
6. ✅ Non-fp4 layers run bf16 internally, no `--fp8` flags.
7. ✅ fp8-flag-suppresses-detection gotcha resolved (auto-detect when flags absent;
   non-fp4 layers handled internally as bf16).

## Non-goals
- Whole-model torch.compile.
- Changing the standalone fp8 model path.
- Quantizing beyond the checkpoint's own fp4 mark; never raise a layer's precision.
- FQT (fp4 weight-grad). Backward is intentionally bf16.
- fp4 backward GEMM (ruled out — VRAM, see above).
