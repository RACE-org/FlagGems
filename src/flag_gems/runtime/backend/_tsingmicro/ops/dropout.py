import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.random_utils import philox_backend_seed_offset, uint_to_uniform_float

logger = logging.getLogger(__name__)

# ===========================================================================
# Tx81 dropout — block-level Philox + fp32 SIMD hash expansion.
#
# Upstream calls tl.philox per 4 elements (int32/uint32 scalar chain) to
# generate per-element uniform random masks.  Tx81 has no vector int ops so
# Philox falls back to ~31k RISC-V scalar iterations per CTA.
#
# Optimisation (same pattern as exponential_):
#   1. Philox once per CTA (block seed) — O(num_ctas) instead of O(N).
#   2. Expand within block via pure-fp32 SIMD hash (FMA + floor).
#   3. Compare hash output with p to produce bool mask.
#   4. Main + tail split — main path zero mask overhead.
#
# Backward is unchanged (load mask + dy, compute, store) — no RNG needed.
# ===========================================================================


# ---------------------------------------------------------------------------
# fp32 SIMD hash — expands 2 fp32 seeds to BLOCK uniform samples via
# 3 rounds of FMA + floor.  All fp32, zero int ops.
# ---------------------------------------------------------------------------

@triton.jit
def _hash_block(s0, s1, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK).to(tl.float32)
    x = s0 * 0.1031 + s1 * 0.11369 + lane * 0.75487766
    x = x - tl.floor(x)
    x = x * (x + 33.33)
    x = x - tl.floor(x)
    x = x * (x + x + 19.19)
    return x - tl.floor(x)


# ---------------------------------------------------------------------------
# Main kernel — zero mask
# ---------------------------------------------------------------------------


@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N", "p", "scale"])
def _dropout_forward_main_kernel(
    X: tl.tensor,
    Y: tl.tensor,
    mask: tl.tensor,
    N: int,
    p: float,
    scale: float,
    philox_seed: int,
    philox_offset: int,
    BLOCK: tl.constexpr,
):
    """Main kernel — no mask, N is exact multiple of BLOCK."""
    pid = tle.program_id(0)

    # One Philox call per CTA → 4 uint64 values.
    off64 = philox_offset.to(tl.int64) + pid.to(tl.int64)
    lo = (off64 & 0xFFFFFFFF).to(tl.uint32)
    hi = ((off64 >> 32) & 0xFFFFFFFF).to(tl.uint32)
    z = lo * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, lo, hi, z, z)

    s0 = uint_to_uniform_float(r0)
    s1 = uint_to_uniform_float(r1)

    u = _hash_block(s0, s1, BLOCK)

    # Bernoulli mask: keep with probability 1-p.
    keep = u > p

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = tl.where(keep, x * scale, 0.0)
    tl.store(Y + offs, y)
    tl.store(mask + offs, keep)


# ---------------------------------------------------------------------------
# Tail kernel — single CTA with mask
# ---------------------------------------------------------------------------


@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N", "p", "scale"])
def _dropout_forward_tail_kernel(
    X: tl.tensor,
    Y: tl.tensor,
    mask: tl.tensor,
    N: int,
    N_start: int,
    p: float,
    scale: float,
    philox_seed: int,
    philox_offset: int,
    N_CTA: int,
    BLOCK: tl.constexpr,
):
    """Tail kernel — masked store for the last N - N_start elements."""
    off64 = philox_offset.to(tl.int64) + N_CTA.to(tl.int64)
    lo = (off64 & 0xFFFFFFFF).to(tl.uint32)
    hi = ((off64 >> 32) & 0xFFFFFFFF).to(tl.uint32)
    z = lo * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, lo, hi, z, z)

    s0 = uint_to_uniform_float(r0)
    s1 = uint_to_uniform_float(r1)

    u = _hash_block(s0, s1, BLOCK)
    keep = u > p

    offs = N_start + tl.arange(0, BLOCK)
    m = offs < N
    x = tl.load(X + offs, mask=m, other=0.0)
    y = tl.where(keep, x * scale, 0.0)
    tl.store(Y + offs, y, mask=m)
    tl.store(mask + offs, keep, mask=m)


# ---------------------------------------------------------------------------
# Backward — no RNG, just load + compute + store
# ---------------------------------------------------------------------------


@libentry()
@triton.jit
def dropout_backward_kernel(
    DY: tl.tensor,
    DX: tl.tensor,
    dropout_mask: tl.tensor,
    N: int,
    scale: float,
    BLOCK: tl.constexpr,
):
    pid = tle.program_id(0)
    ctas = tle.num_programs(0)
    for j in range(tl.cdiv(tl.cdiv(N, BLOCK), ctas)):
        block_id = pid + j * ctas
        offset = block_id * BLOCK + tl.arange(0, BLOCK)
        m = offset < N
        mask_val = tl.load(dropout_mask + offset, mask=m, other=False)
        dy = tl.load(DY + offset, mask=m, other=0.0)
        dx = dy * mask_val * scale
        tl.store(DX + offset, dx, mask=m)


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------

_FULL_TILE_BLOCK = 65536


def _pick_block_size(N):
    """Target ~16 CTAs for good tile utilisation, capped for SPM."""
    per_cta = triton.cdiv(N, 16)
    block = 1
    while block * 2 <= per_cta and block * 2 <= _FULL_TILE_BLOCK:
        block *= 2
    return max(block, 64)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def dropout(input, p, train=True):
    logger.debug("GEMS TSINGMICRO DROPOUT FORWARD")
    if not train or p == 0:
        out = input.clone()
        mask = torch.ones_like(input, dtype=torch.bool)
        return out, mask
    if p == 1:
        out = torch.zeros_like(input)
        mask = torch.zeros_like(input, dtype=torch.bool)
        return out, mask
    assert p > 0.0 and p < 1.0, "p must be in (0, 1)"

    device = input.device
    input = input.contiguous()
    N = input.numel()
    scale = 1.0 / (1.0 - p)

    out = torch.empty_like(input)
    mask = torch.empty_like(input, dtype=torch.bool)

    BLOCK = _pick_block_size(N)
    num_ctas = triton.cdiv(N, BLOCK)
    # Philox state: each CTA consumes 1 call.
    increment = num_ctas + 1
    philox_seed, philox_offset = philox_backend_seed_offset(increment)

    with torch_device_fn.device(device):
        if N % BLOCK == 0:
            _dropout_forward_main_kernel[(num_ctas,)](
                input, out, mask, N, p, scale,
                philox_seed, philox_offset,
                BLOCK=BLOCK,
            )
        else:
            N_aligned = (N // BLOCK) * BLOCK
            main_ctas = max(N_aligned // BLOCK, 0)
            if main_ctas > 0:
                _dropout_forward_main_kernel[(main_ctas,)](
                    input, out, mask, N_aligned, p, scale,
                    philox_seed, philox_offset,
                    BLOCK=BLOCK,
                )
            _dropout_forward_tail_kernel[(1,)](
                input, out, mask, N, N_aligned, p, scale,
                philox_seed, philox_offset,
                main_ctas,
                BLOCK=BLOCK,
            )

    return out, mask


def dropout_backward(grad_output, mask, scale):
    logger.debug("GEMS TSINGMICRO DROPOUT BACKWARD")
    grad_output = grad_output.contiguous()
    grad_input = torch.empty_like(grad_output)
    N = grad_output.numel()
    BLOCK = _pick_block_size(N)
    grid = (min(16, triton.cdiv(N, BLOCK)),)

    with torch_device_fn.device(grad_output.device):
        dropout_backward_kernel[grid](
            grad_output, grad_input, mask, N, scale, BLOCK=BLOCK,
        )
    return grad_input
