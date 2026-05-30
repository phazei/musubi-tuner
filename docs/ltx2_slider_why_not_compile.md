# LTX-2 Slider Training: Why `torch.compile` Does Not Help

## TL;DR

For LTX-2 slider training (`ltx2_train_slider.py`), **do not use `--compile`.**

The recommended (and fastest) configuration is plain eager mode with:

- SageAttention (`--sage_attn`)
- gradient checkpointing (`--gradient_checkpointing`)
- fp8 (`--fp8_base --fp8_scaled`) for the 22B model

`torch.compile` at best **matches** this configuration and at worst **regresses it by ~2x**. Several of the reasons are counterintuitive, so they are documented here to save the next person from re-deriving them.

> Scope note: these conclusions were measured on the LTX-2 22B model, fp8, RTX 5090, Windows, text-mode slider, 512×768×1 latent, BS1. The qualitative conclusions (especially the role of checkpointing and the grad-mode behavior) are architectural and should generalize; the absolute numbers will not.

---

## Benchmark summary

12-step runs, text-mode slider, identical config except the listed variable:

| Configuration | s/it | Notes |
|---|---|---|
| eager + sage + **checkpointing** | **~2.0** | **Recommended. Fastest.** |
| eager + sdpa + checkpointing | ~2.1 | sage slightly faster than sdpa |
| eager + sage + **no checkpointing** | ~12.3 | ~6x slower than with checkpointing |
| eager + sdpa + no checkpointing | ~23.7 | |
| compile + sage + checkpointing | ~2.0–2.1 | No win over eager; 8 graph breaks |
| compile + sdpa + no checkpointing | ~22–24 | "Clean" compile, still no faster than eager |
| compile + sage + checkpointing, *with the `fields()` break removed* | ~3.8–3.9 | **A "fix" that made it ~2x slower** |
| compile + grad-on guidance pass (no checkpointing) | ~49 | Worst case |

The single most important number: **eager + sage + checkpointing is ~2.0 s/it, and nothing involving `torch.compile` beat it.**

---

## The non-intuitive findings

### 1. Gradient checkpointing makes training ~6x *faster* here, not slower

Conventional wisdom: checkpointing trades compute for memory, so it is ~20–30% *slower* but uses less VRAM.

On the LTX-2 22B fp8 model the opposite is true: **disabling checkpointing is ~6x slower** (2.0 → 12.3 s/it with the same attention backend). Without checkpointing the model runs in a much worse memory regime — at this size the activation/weight memory pressure forces the offloading/device-movement machinery into a slow path that dominates step time and dwarfs any compute saved.

Consequence: the entire strategy of "drop checkpointing so the graph is cleaner / so compile can work" is a net loss before `torch.compile` is even considered. You would be trading a 6x slowdown for, at best, a few percent of compile speedup that never materializes anyway.

### 2. `torch.compile` provides no net speedup for this model

Even in the best case (sage + checkpointing + all graph breaks fixed that *can* safely be fixed), the compiled run lands at ~2.0–2.1 s/it — statistically identical to eager. There is no regime in which compile is meaningfully faster.

Why: the dominant costs are already on optimized paths. SageAttention provides the fast attention kernel, fp8 handles the matmuls, and the per-block gradient-checkpointing boundary means Dynamo only ever sees a fragment of each block between graph breaks. Inductor has very little left to fuse or optimize, and what it does produce is not faster than the hand-optimized eager kernels.

### 3. Removing a graph break made compile *slower*

The block forward reconstructs a `TransformerArgs` dataclass around the checkpoint boundary using `dataclasses.fields(...)`, which Dynamo cannot trace and which therefore causes a graph break in every block (`transformer.py`, `_reconstruct_transformer_args`).

Replacing `fields()` with a precomputed constant tuple of field names *did* eliminate that break — and made the compiled run **~2x slower** (2.0 → 3.9 s/it) with a large jump in recompiles.

Reason: the `fields()` graph break was *protecting* performance. It kept the grad-sensitive portion of the block in eager. Once the break was removed, Dynamo traced more of the block into the compiled graph, including code that is sensitive to PyTorch's global grad-mode state. That graph then recompiled on every grad-mode flip (see finding 4).

The takeaway: **not every graph break is a problem, and removing one can be actively harmful.** Always measure end-to-end step time, never optimize the graph-break count in isolation. This `fields()` call is intentionally left as-is, with a code comment explaining why.

### 4. The slider's 3-pass design forces grad-mode recompiles that cannot be suppressed by wrappers

Each text-slider step runs three forwards through the transformer:

1. a guidance forward under `torch.no_grad()` (LoRA multiplier 0, used only to build *detached* targets), then
2. two training forwards with grad enabled (LoRA multipliers +1 and −1).

When the blocks are compiled, Dynamo guards on the global grad-mode (`torch.is_grad_enabled()`). Alternating `no_grad → grad → grad` every step invalidates that guard, producing the `GLOBAL_STATE changed: grad_mode` recompiles — observed as **hundreds of recompiles** and ~22–24 s/it in the no-checkpointing path.

Things that were tried and **do not** fix this:

- **`@torch.compiler.disable` / `@torch._dynamo.disable(recursive=True)` on a wrapper around the guidance forward.** This only stops Dynamo from tracing the *wrapper frame*. The transformer blocks were already compiled as independent `OptimizedModule`s; calling them from a disabled frame still enters their own compiled dispatch, which still guards on grad-mode and still recompiles. (Verified: ~816 grad-mode recompiles with the wrapper in place.)
- **`torch._dynamo.disable()` as a context manager.** It raises `RuntimeError` — it is not a context manager in this PyTorch version despite exposing `__enter__`.
- **Running the guidance pass with grad enabled and `.detach()`ing the result** (so grad-mode never flips). This removes the grad-mode recompiles but builds a full autograd graph for the guidance forward every step, which is *worse*: ~49 s/it.

The grad-mode alternation is intrinsic to the slider algorithm. With per-block compilation there is no clean wrapper-level escape hatch.

Note: in the **recommended** (checkpointing) configuration this problem is naturally small — the `fields()` graph break (finding 3) keeps the grad-sensitive code in eager, so only ~2 grad-mode recompiles occur and step time stays at ~2.0 s/it. This is another reason to leave that break alone.

### 5. SageAttention is per-call, not all-or-nothing

The one-time warning `SageAttention does not support attention masks; falling back to PyTorch SDPA` is misleading. The fallback is decided **per attention call by whether a mask is present**, and the warning is emitted only once (it is gated by a flag).

- `attn1` (self-attention, the dominant cost) usually has no mask → runs on SageAttention.
- `attn2` (text cross-attention) has a mask → falls back to SDPA.

So seeing the warning does **not** mean all attention ran on SDPA. In these benchmarks sage was consistently a bit faster than forcing sdpa everywhere (2.0 vs 2.1 with checkpointing; 12.3 vs 23.7 without).

### 6. None of this is slider-specific at the architecture level

The graph breaks in findings 3–4 originate in the shared LTX-2 transformer (`ltx_2/model/transformer/transformer.py`), not in the slider trainer. The standard LTX-2 network trainer would hit the same breaks. What is slider-specific is the **3-pass grad-mode alternation**, which is what turns a benign break into a recompile storm when checkpointing is disabled.

---

## What was changed in the code, and what was not

Behavior-preserving improvements that were **kept** (they help only if `--compile` is ever used elsewhere, and are harmless otherwise):

- `_ATTN_RETRY_FP32` resolved once at import instead of reading `os.getenv` on every forward, so the `torch.isfinite(...).all()` retry branch is a compile-time constant and can be pruned by Dynamo.
- A fast-path early return in `_run_attn_with_optional_fp32_retry` that skips the data-dependent finite-check branch and the attribute mutation when the retry/pytorch-override flags are off (the normal case).

Changes that were tried and **reverted** because they hurt or added complexity with no benefit:

- Replacing `dataclasses.fields()` with a constant (finding 3) — reverted; left with an explanatory comment.
- A `torch._dynamo.disable`-wrapped guidance helper and a grad-on guidance variant (finding 4) — reverted; the guidance pass is back to a plain `torch.no_grad()` forward.

---

## Recommendation

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  src/musubi_tuner/ltx2_train_slider.py \
  (... other args ...) \
  --sage_attn \
  --gradient_checkpointing \
  --fp8_base --fp8_scaled
  # do NOT add --compile
```

If you want to re-evaluate `torch.compile` on future hardware or PyTorch versions, the only configuration worth testing is **compile + sage + checkpointing**, and the bar to clear is the eager baseline of ~2.0 s/it. Measure end-to-end step time after warmup; do not rely on graph-break or recompile counts as a proxy for speed.
