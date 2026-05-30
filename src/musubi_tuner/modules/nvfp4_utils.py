"""NVIDIA FP4 (E2M1) checkpoint loading utilities for pre-quantized LTX-2 models.

Handles the Lightricks nvfp4 checkpoint format where transformer block weights
(typically blocks 10-47) are stored as packed uint8 with two-level scaling:
  - ``weight``: uint8 packed (two FP4 E2M1 nibbles per byte), shape ``[out, in/2]``
  - ``weight_scale``: float8_e4m3fn per-block scales
  - ``weight_scale_2``: float32 per-tensor scalar scale

Early blocks (0-9), VAE, vocoder, connectors, biases, and norms are stored in bf16
and passed through unchanged.

The on-the-fly forward patch keeps packed weights in GPU memory (~10-11 GB for the
transformer) and dequantizes to bf16 only during each ``F.linear`` call, mirroring
the approach used by ``nf4_optimization_utils.py``.
"""

import json
import os
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

from tqdm import tqdm

from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen, TensorWeightAdapter, WeightTransformHooks

logger = logging.getLogger(__name__)

# NVFP4 block size (elements per block scale) and compile opt-out.
# Set NVFP4_NO_COMPILE=1 to run the quant/dequant helpers in eager mode.
_NVFP4_BLOCK = 16
_NVFP4_NO_COMPILE = os.getenv("NVFP4_NO_COMPILE", "0") == "1"
# Use the QuTLASS fused NVFP4 activation-quant kernel (one CUDA kernel does
# transform + quant + per-block e4m3 scale) instead of the compiled-RTN path.
# Requires the `qutlass` package built for this GPU (sm_120). Falls back to the
# compiled path automatically if import fails. See docs/nvfp4_native_gemm_spec.md.
_NVFP4_USE_QUTLASS = os.getenv("NVFP4_USE_QUTLASS", "0") == "1"

# ---------------------------------------------------------------------------
# FP4 E2M1 lookup table
# ---------------------------------------------------------------------------
# Nibble values 0x0-0x7 are positive, 0x8-0xF are negative mirrors.
FP4_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.bfloat16,
)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_nvfp4_checkpoint(model_path: str) -> bool:
    """Check whether a safetensors file is in Lightricks nvfp4 format.

    Detection is based on the ``_quantization_metadata`` key in the file
    metadata containing ``"nvfp4"`` format entries, or the presence of
    ``weight_scale_2`` keys alongside ``weight_scale`` keys.
    """
    try:
        from safetensors import safe_open
        check_path = model_path if isinstance(model_path, str) else model_path[0]
        with safe_open(check_path, framework="pt") as f:
            meta = f.metadata()
            if meta is not None:
                qm = meta.get("_quantization_metadata", "")
                if '"nvfp4"' in qm:
                    return True
            # Fallback: check for weight_scale_2 keys
            keys = f.keys()
            has_ws2 = any(k.endswith(".weight_scale_2") for k in keys)
            has_ws = any(k.endswith(".weight_scale") for k in keys)
            has_uint8 = False
            if has_ws2 and has_ws:
                # Confirm at least one uint8 weight
                for k in keys:
                    if k.endswith(".weight"):
                        t = f.get_tensor(k)
                        if t.dtype == torch.uint8:
                            has_uint8 = True
                            break
            return has_ws2 and has_ws and has_uint8
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Dequantize (for on-the-fly forward or one-shot conversion)
# ---------------------------------------------------------------------------

def dequantize_nvfp4_weight(
    w_uint8: torch.Tensor,
    block_scale: torch.Tensor,
    tensor_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an NVFP4-packed weight tensor.

    Args:
        w_uint8: Packed weight ``[out_features, in_features // 2]`` as uint8.
        block_scale: Per-block scales (float8_e4m3fn or similar), flattened or shaped.
        tensor_scale: Per-tensor scalar scale (float32).
        dtype: Output dtype.

    Returns:
        ``[out_features, in_features]`` tensor in *dtype*.
    """
    return _dequantize_nvfp4_weight_fn(w_uint8, block_scale, tensor_scale, dtype)


def _dequantize_nvfp4_weight_impl(
    w_uint8: torch.Tensor,
    block_scale: torch.Tensor,
    tensor_scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    lut = FP4_E2M1_LUT.to(device=w_uint8.device, dtype=torch.bfloat16)
    R, Ch = w_uint8.shape
    C = Ch * 2
    block = _NVFP4_BLOCK

    # Unpack two 4-bit values per byte directly into a [R, C] layout (no flatten).
    lo = (w_uint8 & 0x0F).long()
    hi = (w_uint8 >> 4).long()
    codes = torch.empty(R, C, dtype=torch.long, device=w_uint8.device)
    codes[:, 0::2] = lo
    codes[:, 1::2] = hi
    values = lut[codes]                                            # bf16 [R, C]

    # Per-block scale broadcast over the block dim; fold in the per-tensor scale.
    bs = block_scale.to(torch.bfloat16).reshape(R, C // block, 1)
    scale = bs * tensor_scale.to(torch.bfloat16)
    result = (values.reshape(R, C // block, block) * scale).reshape(R, C)
    return result.to(dtype)


_dequantize_compiled = None


def _dequantize_nvfp4_weight_fn(w_uint8, block_scale, tensor_scale, dtype):
    global _dequantize_compiled
    if _NVFP4_NO_COMPILE:
        return _dequantize_nvfp4_weight_impl(w_uint8, block_scale, tensor_scale, dtype)
    if _dequantize_compiled is None:
        try:
            _dequantize_compiled = torch.compile(
                _dequantize_nvfp4_weight_impl, dynamic=True, fullgraph=True
            )
        except Exception:
            _dequantize_compiled = _dequantize_nvfp4_weight_impl
    return _dequantize_compiled(w_uint8, block_scale, tensor_scale, dtype)


# ---------------------------------------------------------------------------
# Native Blackwell FP4 GEMM path (scaled_mm, W4A4)
# ---------------------------------------------------------------------------
#
# This is the fast path: the frozen base's forward GEMM runs on Blackwell FP4
# tensor cores via ``F.scaled_mm`` with both operands in ``float4_e2m1fn_x2``
# and two-level block (1x16) + per-tensor scaling, matching the NVFP4 recipe.
#
# Design (see docs/nvfp4_native_gemm_spec.md):
#   * Forward is W4A4: the weight is already fp4; the *activation* is quantized
#     to fp4 per 1x16 block with deterministic round-to-nearest (RTN) every
#     forward (no native bf16xfp4 path exists on this HW).
#   * Backward computes ``grad_input`` only, in bf16, from the exact bf16 decode
#     of the frozen fp4 weights (straight-through activation quant). There is no
#     grad_weight (the base is frozen). This is correct, not an approximation,
#     and composes with ``torch.utils.checkpoint`` because the activation quant
#     is deterministic.
#
# The block scales stored in the checkpoint are PLAIN (natural) layout; the
# Blackwell kernel requires them physically rearranged into the SWIZZLE_32_4_4
# (128x4 tile / 32x4x4 internal) layout. ``swizzle_block_scale`` performs this;
# weight scales are swizzled ONCE at load time, activation scales every forward.

# FP4 E2M1 positive magnitudes (codes 0..7); the sign bit is the high nibble bit.
_FP4_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
# Midpoints between consecutive magnitudes — used for branchless round-to-nearest
# encoding via torch.bucketize (exact match to LUT-argmin, ~13x faster).
_FP4_E2M1_MIDPOINTS = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
_FP4_MAX = 6.0
_F8E4M3_MAX = 448.0

# Cache of threshold tensors per device to avoid reallocation each forward.
_MID_CACHE: dict = {}


def _get_fp4_midpoints(device: torch.device) -> torch.Tensor:
    t = _MID_CACHE.get(device)
    if t is None:
        t = torch.tensor(_FP4_E2M1_MIDPOINTS, device=device, dtype=torch.float32)
        _MID_CACHE[device] = t
    return t


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def swizzle_block_scale(scale: torch.Tensor) -> torch.Tensor:
    """Rearrange a plain [rows, cols] block-scale matrix into the Blackwell
    SWIZZLE_32_4_4 layout expected by ``scaled_mm`` (flattened).

    Mirrors ``torch.testing._internal.common_quantized.to_blocked`` (the
    reference used by PyTorch's own scaled_mm tests). The matrix is padded to a
    128x4 tile grid, then each tile is rearranged into the 32x4x4 internal
    order. Output is a 1-D tensor; ``scaled_mm`` accepts the flat swizzled scale.
    """
    rows, cols = scale.shape
    n_row_blocks = _ceil_div(rows, 128)
    n_col_blocks = _ceil_div(cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros((padded_rows, padded_cols), device=scale.device, dtype=scale.dtype)
        padded[:rows, :cols] = scale
    else:
        padded = scale
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten().contiguous()


def _swizzle_block_scale_compilable(scale: torch.Tensor) -> torch.Tensor:
    """to_blocked swizzle, written for torch.compile (avoids data-dependent
    branching so it fuses into the quant graph)."""
    rows, cols = scale.shape
    n_row_blocks = (rows + 127) // 128
    n_col_blocks = (cols + 3) // 4
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4
    padded = scale.new_zeros((padded_rows, padded_cols))
    padded[:rows, :cols] = scale
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


def _quantize_activation_impl(x_2d: torch.Tensor, mids: torch.Tensor):
    """Core RTN activation quant + scale swizzle (compilable, single graph).

    Returns ``(packed_uint8 [M, K//2], block_scale_swizzled_flat_e4m3,
    tensor_scale_f32 [1])`` — the block scale is already in SWIZZLE_32_4_4 layout
    so the forward needs no separate swizzle launches.
    """
    M, K = x_2d.shape
    block = _NVFP4_BLOCK
    xf = x_2d.float()
    blocks = xf.reshape(M, K // block, block)
    block_amax = blocks.abs().amax(dim=-1)                      # [M, K//block]
    global_amax = xf.abs().amax().clamp(min=1e-8)
    # Per-tensor scale chosen so block scales stay representable in e4m3.
    tensor_scale = (global_amax / (_FP4_MAX * _F8E4M3_MAX)).clamp(min=1e-8)
    block_scale = (block_amax / _FP4_MAX / tensor_scale).to(torch.float8_e4m3fn)
    effective = block_scale.float() * tensor_scale                # real per-block scale
    normed = blocks / effective.unsqueeze(-1).clamp(min=1e-12)     # [M, K//block, block]
    # Round-to-nearest E2M1 (deterministic) via branchless magnitude bucketize.
    # Sign bit is the high bit of the nibble; magnitude code 0..7 from thresholds.
    mag = normed.abs()
    codes_mag = torch.bucketize(mag, mids).to(torch.uint8)         # 0..7
    sign = (normed < 0).to(torch.uint8)
    codes = (codes_mag | (sign << 3)).reshape(M, K)                # 0..15 nibble
    lo = codes[:, 0::2]
    hi = codes[:, 1::2]
    packed = (lo | (hi << 4)).contiguous()
    block_scale_swz = _swizzle_block_scale_compilable(block_scale)
    return packed, block_scale_swz, tensor_scale.reshape(1)


# The activation quant runs once per Linear, every forward — it is the dominant
# per-step cost in eager mode (many small elementwise kernels). Compiling JUST
# this leaf function (not the model graph) fuses them for ~3-4x. This is distinct
# from the project's "no torch.compile on the model" non-goal: it is a localized,
# graph-break-free helper compile with a clean eager fallback.
_quantize_activation_compiled = None


def _get_quantize_fn():
    global _quantize_activation_compiled
    if _NVFP4_NO_COMPILE:
        return _quantize_activation_impl
    if _quantize_activation_compiled is None:
        try:
            _quantize_activation_compiled = torch.compile(
                _quantize_activation_impl, dynamic=True, fullgraph=True
            )
        except Exception:
            _quantize_activation_compiled = _quantize_activation_impl
    return _quantize_activation_compiled


def quantize_activation_to_nvfp4(x_2d: torch.Tensor):
    """Quantize a 2-D bf16/fp16/fp32 activation [M, K] to NVFP4 (W4A4 forward).

    Deterministic round-to-nearest, per 1x16 block, two-level scaling matching
    the weight recipe. K must be divisible by 16. The block scale is returned
    ALREADY SWIZZLED (SWIZZLE_32_4_4, flat) so the forward needs no separate
    swizzle op.

    Returns ``(packed_uint8 [M, K//2], block_scale_swizzled_flat_e4m3,
    tensor_scale_f32 [1])``.
    """
    if _NVFP4_USE_QUTLASS:
        out = _quantize_activation_qutlass(x_2d)
        if out is not None:
            return out
    mids = _get_fp4_midpoints(x_2d.device)
    return _get_quantize_fn()(x_2d, mids)


# ---------------------------------------------------------------------------
# QuTLASS fused NVFP4 activation quant (optional fast path)
# ---------------------------------------------------------------------------
# QuTLASS' ``fusedQuantizeNv`` does transform + abs-max quant + e4m3 block-scale
# compute in ONE CUDA kernel. Its scale convention (verified empirically against
# our reference recipe, rel-err == fp4 floor): the returned e4m3 block scale is
# ``block_amax`` and dequant divides by ``alpha=6`` -> effective per-block scale
# is ``block_amax / 6``. We therefore map it to our two-level ``scaled_mm`` recipe
# as ``tensor_scale = 1/6`` and ``block_scale = the returned e4m3`` (used as-is).
# The returned scale is row-padded to a 128-multiple in NATURAL layout, so we
# trim to M and swizzle to SWIZZLE_32_4_4 (same as the compiled path).
_qutlass_mod = None              # the imported qutlass module (or False if unavailable)
_qutlass_hadamard: dict = {}     # device -> identity rotation matrix [16,16] bf16
_qutlass_gscale: dict = {}       # device -> global_scale tensor [6.0]
_qutlass_ats: dict = {}          # device -> per-tensor scale tensor [1/6]


def _get_qutlass():
    global _qutlass_mod
    if _qutlass_mod is None:
        try:
            import qutlass  # noqa: F401
            _qutlass_mod = qutlass
        except Exception as e:  # pragma: no cover - environment dependent
            logger.warning(f"NVFP4_USE_QUTLASS=1 but qutlass import failed ({e!r}); "
                           "falling back to the compiled RTN quant path.")
            _qutlass_mod = False
    return _qutlass_mod or None


def _quantize_activation_qutlass(x_2d: torch.Tensor):
    """Fused NVFP4 activation quant via QuTLASS. Returns the same tuple as
    ``_quantize_activation_impl`` (packed uint8 [M, K//2], swizzled flat e4m3
    block scale, per-tensor scale [1]) or ``None`` if QuTLASS is unavailable."""
    qt = _get_qutlass()
    if qt is None:
        return None
    dev = x_2d.device
    M, K = x_2d.shape
    block = _NVFP4_BLOCK
    h = _qutlass_hadamard.get(dev)
    if h is None:
        # Identity rotation = no transform (we only want the fused quant+scale).
        h = torch.eye(block, dtype=torch.bfloat16, device=dev)
        _qutlass_hadamard[dev] = h
    gscale = _qutlass_gscale.get(dev)
    if gscale is None:
        gscale = torch.tensor([_FP4_MAX], device=dev)
        _qutlass_gscale[dev] = gscale
    ats = _qutlass_ats.get(dev)
    if ats is None:
        ats = torch.tensor([1.0 / _FP4_MAX], device=dev)
        _qutlass_ats[dev] = ats

    x_bf16 = x_2d if x_2d.dtype == torch.bfloat16 else x_2d.to(torch.bfloat16)
    # fusedQuantizeNv returns (packed_e2m1 [M, K//2] uint8, e4m3 block scale
    # padded to [pad128(M), K//16] NATURAL layout).
    a_e2m1, a_e4m3 = qt.fusedQuantizeNv(x_bf16, h, gscale, method="abs_max")
    m_pad = ((M + 127) // 128) * 128
    a_view = a_e4m3.reshape(m_pad, K // block)[:M, :].contiguous()
    a_bs_swz = swizzle_block_scale(a_view)
    return a_e2m1, a_bs_swz, ats


# scaled_mm imports (resolved once; this build exposes them on F).
_ScalingType = getattr(F, "ScalingType", None)
_SwizzleType = getattr(F, "SwizzleType", None)


def _nvfp4_scaled_mm(
    a_packed: torch.Tensor,        # uint8 [M, K//2]
    a_block_scale_swz: torch.Tensor,  # flat swizzled e4m3
    a_tensor_scale: torch.Tensor,  # f32 [1]
    w_packed: torch.Tensor,        # uint8 [N, K//2]
    w_block_scale_swz: torch.Tensor,  # flat swizzled e4m3
    w_tensor_scale: torch.Tensor,  # f32 [1]
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run the native FP4 GEMM ``a @ w.t()`` -> [M, N] via scaled_mm."""
    a_fp4 = a_packed.view(torch.float4_e2m1fn_x2)
    b_fp4 = w_packed.view(torch.float4_e2m1fn_x2)
    return F.scaled_mm(
        a_fp4,
        b_fp4.t(),
        scale_a=[a_block_scale_swz, a_tensor_scale.float()],
        scale_recipe_a=[_ScalingType.BlockWise1x16, _ScalingType.TensorWise],
        swizzle_a=[_SwizzleType.SWIZZLE_32_4_4, _SwizzleType.NO_SWIZZLE],
        scale_b=[w_block_scale_swz, w_tensor_scale.float()],
        scale_recipe_b=[_ScalingType.BlockWise1x16, _ScalingType.TensorWise],
        swizzle_b=[_SwizzleType.SWIZZLE_32_4_4, _SwizzleType.NO_SWIZZLE],
        bias=None,
        output_dtype=out_dtype,
    )


class NVFP4LinearFunction(torch.autograd.Function):
    """W4A4 native FP4 forward; bf16 grad_input backward (frozen base).

    forward: quantize activation -> fp4 (RTN, 1x16), native fp4 scaled_mm against
             the pre-swizzled fp4 weight, add bias in compute dtype.
    backward: grad_input = grad_output @ W_decode (bf16), where W_decode is the
              exact bf16 decode of the frozen fp4 weight (straight-through for the
              activation quant). No grad_weight (frozen). bias grad not needed
              (frozen) but returned as None.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,                 # [*, K] any float dtype
        w_packed: torch.Tensor,          # uint8 [N, K//2]
        w_block_scale: torch.Tensor,     # e4m3 [N, K//16] (natural, for backward decode)
        w_block_scale_swz: torch.Tensor, # flat swizzled e4m3 (for forward GEMM)
        w_tensor_scale: torch.Tensor,    # f32 [1]
        bias: Optional[torch.Tensor],    # bf16 [N] or None
        out_features: int,
    ) -> torch.Tensor:
        compute_dtype = x.dtype if x.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
        orig_shape = x.shape
        K = orig_shape[-1]
        x_2d = x.reshape(-1, K).to(torch.bfloat16)

        # Quant returns the activation block scale ALREADY swizzled (fused in the
        # compiled quant graph) — no separate swizzle launch.
        a_packed, a_bs_swz, a_ts = quantize_activation_to_nvfp4(x_2d)
        out = _nvfp4_scaled_mm(
            a_packed, a_bs_swz, a_ts,
            w_packed, w_block_scale_swz, w_tensor_scale,
            out_dtype=torch.bfloat16,
        )  # [M, N]

        if bias is not None:
            out = out + bias.to(out.dtype)

        # Save for backward: only the frozen weight buffers + shapes. The
        # activation is NOT stashed (under checkpointing the forward is recomputed
        # deterministically; the frozen weight is always resident).
        ctx.save_for_backward(w_packed, w_block_scale, w_tensor_scale)
        ctx.orig_shape = orig_shape
        ctx.out_features = out_features
        ctx.compute_dtype = compute_dtype

        out = out.reshape(*orig_shape[:-1], out_features)
        return out.to(compute_dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        w_packed, w_block_scale, w_tensor_scale = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            # Exact bf16 decode of the frozen fp4 weight = the straight-through
            # gradient weight. grad_input = grad_output @ W  (W is [N, K]).
            w_bf16 = dequantize_nvfp4_weight(
                w_packed, w_block_scale, w_tensor_scale, dtype=torch.bfloat16
            )
            go = grad_output.reshape(-1, ctx.out_features).to(torch.bfloat16)
            gi = go @ w_bf16                       # [M, K]
            grad_input = gi.reshape(*ctx.orig_shape).to(grad_output.dtype)
        # grads for (w_packed, w_block_scale, w_block_scale_swz, w_tensor_scale,
        #            bias, out_features)
        return grad_input, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# Monkey-patched forward
# ---------------------------------------------------------------------------

def nvfp4_linear_forward_patch(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Native Blackwell FP4 forward for NVFP4-quantized Linear layers (W4A4).

    Quantizes the activation to fp4 and runs the native fp4 ``scaled_mm`` against
    the pre-swizzled fp4 weight via :class:`NVFP4LinearFunction` (autograd-aware,
    checkpoint-safe). Falls back to the bf16 dequant path only if the native
    scaled_mm machinery is unavailable.
    """
    if _ScalingType is None or _SwizzleType is None or not hasattr(self, "nvfp4_block_scale_swizzled"):
        # Fallback: legacy dequant-to-bf16 path (W4A16). Should not happen on
        # Blackwell + torch>=2.12; kept for safety.
        w = dequantize_nvfp4_weight(
            self.weight, self.nvfp4_block_scale, self.nvfp4_tensor_scale, dtype=torch.bfloat16
        )
        if w.dtype != x.dtype:
            w = w.to(x.dtype)
        return F.linear(x, w, self.bias)

    return NVFP4LinearFunction.apply(
        x,
        self.weight,
        self.nvfp4_block_scale,
        self.nvfp4_block_scale_swizzled,
        self.nvfp4_tensor_scale,
        self.bias,
        self.nvfp4_out_features,
    )


# ---------------------------------------------------------------------------
# Monkey-patch application (mirrors apply_nf4_monkey_patch)
# ---------------------------------------------------------------------------

def apply_nvfp4_monkey_patch(
    model: nn.Module,
    state_dict: dict,
) -> nn.Module:
    """Register NVFP4 buffers and replace forward on quantized Linears.

    Identifies quantized layers by looking for ``weight_scale`` and
    ``weight_scale_2`` keys in the state dict alongside uint8 ``.weight``
    tensors.  Non-quantized layers (bf16 weights) are left untouched.
    """
    # Collect quantized module paths from state dict.
    # Keys have been renamed by load_nvfp4_state_dict:
    #   .weight_scale   -> .nvfp4_block_scale
    #   .weight_scale_2 -> .nvfp4_tensor_scale
    nvfp4_modules: dict = {}  # module_path -> (packed_shape, real_out, real_in)
    for key in state_dict:
        if key.endswith(".nvfp4_tensor_scale"):
            module_path = key.rsplit(".nvfp4_tensor_scale", 1)[0]
            weight_key = module_path + ".weight"
            bs_key = module_path + ".nvfp4_block_scale"
            if weight_key in state_dict and bs_key in state_dict:
                wt = state_dict[weight_key]
                if wt.dtype == torch.uint8:
                    out_f = wt.shape[0]
                    in_f = wt.shape[1] * 2  # real in_features (2 values per packed byte)
                    nvfp4_modules[module_path] = (wt.shape, out_f, in_f)

    patched_count = 0
    for name, module in model.named_modules():
        if name not in nvfp4_modules:
            continue
        if not isinstance(module, nn.Linear):
            continue

        packed_shape, out_f, in_f = nvfp4_modules[name]
        bs_key = name + ".nvfp4_block_scale"
        ts_key = name + ".nvfp4_tensor_scale"
        swz_key = name + ".nvfp4_block_scale_swizzled"

        # Replace the weight parameter with a buffer of packed shape
        # so load_state_dict(assign=True) doesn't complain about shape mismatch.
        del module.weight
        module.register_buffer("weight", torch.zeros(packed_shape, dtype=torch.uint8))

        # Register scale buffers (natural block scale kept for the bf16 backward
        # decode; the pre-swizzled flat scale feeds the forward scaled_mm).
        bs_tensor = state_dict[bs_key]
        ts_tensor = state_dict[ts_key]
        module.register_buffer("nvfp4_block_scale", torch.zeros_like(bs_tensor))
        module.register_buffer("nvfp4_tensor_scale", torch.zeros_like(ts_tensor))
        if swz_key in state_dict:
            module.register_buffer(
                "nvfp4_block_scale_swizzled", torch.zeros_like(state_dict[swz_key])
            )

        # Store metadata
        module.nvfp4_out_features = out_f
        module.nvfp4_in_features = in_f
        module._nvfp4_quantized = True

        # Replace forward
        def new_forward(self, x):
            return nvfp4_linear_forward_patch(self, x)
        module.forward = new_forward.__get__(module, type(module))
        patched_count += 1

    logger.info(f"Number of NVFP4 monkey-patched Linear layers (native FP4 GEMM): {patched_count}")
    return model


# ---------------------------------------------------------------------------
# State dict loading for nvfp4 checkpoints
# ---------------------------------------------------------------------------

def load_nvfp4_state_dict(
    model_files: Union[str, List[str]],
    state_dict_key_filter: Optional[Callable[[str], bool]] = None,
    move_to_device: bool = False,
    target_device: Optional[Union[str, torch.device]] = None,
) -> dict:
    """Load an nvfp4 checkpoint into a state dict, preserving packed format.

    Quantized layers keep their uint8 packed weights, weight_scale (fp8), and
    weight_scale_2 (fp32) as separate state dict entries.  Non-quantized layers
    (bf16) are loaded as-is.

    The ``weight_scale`` / ``weight_scale_2`` keys are renamed to
    ``nvfp4_block_scale`` / ``nvfp4_tensor_scale`` in the output so they
    match the buffer names registered by ``apply_nvfp4_monkey_patch``.
    """
    if isinstance(model_files, str):
        model_files = [model_files]

    state_dict = {}
    for model_file in model_files:
        with MemoryEfficientSafeOpen(model_file) as f:
            all_keys = list(f.keys())

            # Identify quantized layers
            ws2_keys = set(k for k in all_keys if k.endswith(".weight_scale_2"))
            ws_keys = set(k for k in all_keys if k.endswith(".weight_scale"))
            skip_suffixes = (".weight_scale", ".weight_scale_2", ".pre_quant_scale", ".input_scale", ".comfy_quant")

            for key in tqdm(all_keys, desc=f"Loading {os.path.basename(model_file)}", unit="key"):
                if state_dict_key_filter is not None and not state_dict_key_filter(key):
                    continue

                # Skip auxiliary scale keys — they are loaded alongside their .weight
                if key.endswith(skip_suffixes):
                    continue

                value = f.get_tensor(key)

                # For uint8 .weight tensors with nvfp4 scales, also load the scales
                if value.dtype == torch.uint8 and key.endswith(".weight"):
                    prefix = key[:-len(".weight")]
                    ws_key = prefix + ".weight_scale"
                    ws2_key = prefix + ".weight_scale_2"
                    if ws_key in ws_keys and ws2_key in ws2_keys:
                        # Load scales and store with buffer-compatible names
                        block_scale = f.get_tensor(ws_key)
                        tensor_scale = f.get_tensor(ws2_key)

                        # The checkpoint stores block scales in NATURAL (un-swizzled)
                        # layout, row-padded to a multiple of 128. Trim the row
                        # padding to the real out_features so both the bf16 backward
                        # decode and the swizzle are correct. Real out = packed rows.
                        out_f = value.shape[0]
                        if block_scale.dim() == 2 and block_scale.shape[0] != out_f:
                            block_scale = block_scale[:out_f, :].contiguous()

                        # Pre-swizzle ONCE at load time into the SWIZZLE_32_4_4
                        # layout the Blackwell scaled_mm kernel requires for the
                        # forward GEMM. Stored flat (1-D) e4m3.
                        block_scale_swz = swizzle_block_scale(block_scale)

                        if move_to_device and target_device is not None:
                            value = value.to(target_device)
                            block_scale = block_scale.to(target_device)
                            block_scale_swz = block_scale_swz.to(target_device)
                            tensor_scale = tensor_scale.to(target_device)

                        state_dict[key] = value
                        state_dict[prefix + ".nvfp4_block_scale"] = block_scale
                        state_dict[prefix + ".nvfp4_block_scale_swizzled"] = block_scale_swz
                        state_dict[prefix + ".nvfp4_tensor_scale"] = tensor_scale
                        continue

                # Non-quantized tensor — load normally
                if move_to_device and target_device is not None:
                    value = value.to(target_device)
                state_dict[key] = value

    return state_dict


# ---------------------------------------------------------------------------
# Detection helper
# ---------------------------------------------------------------------------

def is_nvfp4_module(module: nn.Module) -> bool:
    """Check whether *module* has been NVFP4-quantized."""
    return getattr(module, "_nvfp4_quantized", False)
