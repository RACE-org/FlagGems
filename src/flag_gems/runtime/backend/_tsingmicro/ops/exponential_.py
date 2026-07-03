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
# Tx81 exponential_ — block-level Philox + fp32 SIMD hash expansion.
#
# Upstream calls tl.philox per 4 elements → the full int32/uint32 chain
# (mului_extended, xori, addi, bitcast, shift, and) falls back to RISC-V
# scalar on Tx81.  ~31k scalar integer ops per CTA for a 1024-element
# BLOCK is the dominant cost.
#
# Optimisation (from doc/ops/exponential_.md):
#   1. Philox once per CTA (block seed) — O(num_ctas) instead of O(N).
#   2. Expand within block via pure-fp32 SIMD hash (FMA + floor + fract).
#   3. tl.math.log(u) — vector log on Tx81 CT unit.
#   4. Main + tail split → main path has zero mask overhead.
#
# fp64 falls back to upstream (uncommon, double-precision philox is worse).
# ===========================================================================


@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N", "lambd", "eps"])
def _exponential_main_kernel(
    out_ptr: tl.tensor,
    N: int,  # aligned count (N_aligned = N - N_tail)
    lambd: float,
    eps: float,
    philox_seed: int,
    philox_offset: int,
    BLOCK: tl.constexpr,
):
    """Main kernel — no mask, handles multiples of BLOCK exactly.

    One Philox call per CTA seeds a 3-round fp32 hash that expands to
    BLOCK uniform samples.  -log(u)/lambd → store.
    """
    pid = tle.program_id(0)

    # ---- Block-level Philox (once per CTA) ----
    seed64 = philox_seed
    off64 = philox_offset.to(tl.int64) + pid.to(tl.int64)
    lo = (off64 & 0xFFFFFFFF).to(tl.uint32)
    hi = ((off64 >> 32) & 0xFFFFFFFF).to(tl.uint32)
    z = lo * 0
    r0, r1, r2, r3 = tl.philox(seed64, lo, hi, z, z)

    # Convert Philox uint64 → fp32 uniform [0,1) as hash seeds.
    s0 = uint_to_uniform_float(r0)
    s1 = uint_to_uniform_float(r1)

    # ---- fp32 SIMD hash: s0,s1 + lane → BLOCK uniform samples ----
    lane = tl.arange(0, BLOCK).to(tl.float32)

    # Round 1
    x = s0 * 0.1031 + s1 * 0.11369 + lane * 0.75487766
    x = x - tl.floor(x)

    # Round 2
    x = x * (x + 33.33)
    x = x - tl.floor(x)

    # Round 3
    x = x * (x + x + 19.19)
    u = x - tl.floor(x)

    # Clamp away from 0 to avoid log(0).
    # min_u = 0.5 * eps matches upstream transform_exponential safeguard.
    min_u = 0.5 * eps
    u = tl.maximum(u, min_u)

    # Exponential transform: y = -log(u) / lambd.
    # tl.math.log → vector log on Tx81 CT unit.
    y = -tl.math.log(u) / lambd

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, y)


@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N", "lambd", "eps"])
def _exponential_tail_kernel(
    out_ptr: tl.tensor,
    N: int,
    N_start: int,  # offset where tail begins
    lambd: float,
    eps: float,
    philox_seed: int,
    philox_offset: int,
    N_CTA: int,  # total CTA count from main (for offset consistency)
    BLOCK: tl.constexpr,
):
    """Tail kernel — single CTA, masked store for the last N - N_start elements.

    Uses a fresh Philox call with offset = philox_offset + N_CTA to avoid
    overlapping with the main kernel's Philox counter space.
    """
    seed64 = philox_seed
    off64 = philox_offset.to(tl.int64) + N_CTA.to(tl.int64)
    lo = (off64 & 0xFFFFFFFF).to(tl.uint32)
    hi = ((off64 >> 32) & 0xFFFFFFFF).to(tl.uint32)
    z = lo * 0
    r0, r1, r2, r3 = tl.philox(seed64, lo, hi, z, z)

    s0 = uint_to_uniform_float(r0)
    s1 = uint_to_uniform_float(r1)

    lane = tl.arange(0, BLOCK).to(tl.float32)

    x = s0 * 0.1031 + s1 * 0.11369 + lane * 0.75487766
    x = x - tl.floor(x)

    x = x * (x + 33.33)
    x = x - tl.floor(x)

    x = x * (x + x + 19.19)
    u = x - tl.floor(x)

    min_u = 0.5 * eps
    u = tl.maximum(u, min_u)

    y = -tl.math.log(u) / lambd

    offs = N_start + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, y, mask=offs < N)


# ===========================================================================
# Host helper
# ===========================================================================


def _pick_block_size(N):
    """Target ~16 CTAs for good tile utilisation, capped at 65536 for SPM."""
    per_cta = triton.cdiv(N, 16)
    return min(triton.next_power_of_2(per_cta), 65536)


# ===========================================================================
# Entry point
# ===========================================================================


def exponential_(x, lambd: float = 1.0, *, gen=None):
    logger.debug("GEMS TSINGMICRO EXPONENTIAL_")
    dtype = x.dtype
    device = x.device
    assert dtype in (torch.float16, torch.bfloat16, torch.float32), (
        f"Tx81 exponential_ supports fp16/bf16/fp32, got {dtype}"
    )

    N = x.numel()
    eps = torch.finfo(dtype).eps
    inplace = x.is_contiguous()
    x_out = x if inplace else torch.empty(x.shape, dtype=dtype, device=device)

    # Philox state: each CTA consumes one Philox call (block seed).
    BLOCK = _pick_block_size(N)
    num_ctas = triton.cdiv(N, BLOCK)
    increment = num_ctas + 1  # +1 for potential tail kernel
    philox_seed, philox_offset = philox_backend_seed_offset(increment, generator=gen)

    with torch_device_fn.device(device):
        if N % BLOCK == 0:
            # Exact multiple — main kernel only, zero tail.
            grid = (num_ctas,)
            _exponential_main_kernel[grid](
                x_out, N, lambd, eps, philox_seed, philox_offset,
                BLOCK=BLOCK,
            )
        else:
            # Main + tail split.
            N_aligned = (N // BLOCK) * BLOCK
            main_ctas = max(N_aligned // BLOCK, 0)

            if main_ctas > 0:
                _exponential_main_kernel[(main_ctas,)](
                    x_out, N_aligned, lambd, eps,
                    philox_seed, philox_offset,
                    BLOCK=BLOCK,
                )

            # Tail: single CTA, offset = philox_offset + main_ctas.
            _exponential_tail_kernel[(1,)](
                x_out, N, N_aligned, lambd, eps,
                philox_seed, philox_offset,
                main_ctas,
                BLOCK=BLOCK,
            )

    if not inplace:
        x.copy_(x_out)
    return x
