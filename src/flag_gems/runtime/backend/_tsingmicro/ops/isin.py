import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.all import reduce_all
from flag_gems.ops.any import reduce_any
from flag_gems.ops.unique import _unique2
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.libentry import libentry

logger = logging.getLogger(__name__)

# ===========================================================================
# Tx81 optimisation summary
#
# 1. Scalar_Tensor / Tensor_Scalar: bypass generic isin entirely — use
#    elementwise == (eq kernel, already Tx81-optimised) + reduce_any /
#    logical_not.  Zero binary search, zero integer division.
#
# 2. Value-domain SIMD linear scan (NEW kernel, replaces binary search):
#    When in1 is small after sort/unique (≤ 256 values), loop over in1
#    values with fp32 vector compare + bool or.  Pure SIMD — no integer
#    division, no gather, no per-element branch divergence.
#
#    Complexity O(in0 × in1_size) vs binary search O(in0 × log in1_size),
#    but every operation in the linear scan maps to a Tx81 vector intrinsic
#    (tx.boolequalvv → tx.boolorvv) while binary search hits the integer-
#    division corrective chain (recip → mul → trunc → cmpf → mask_move)
#    on every iteration.  The vector path wins for in1_size ≤ 256.
#
# 3. broadcast-comparison path: kept identical to upstream kernel.
#    BLOCK_M=1 avoids 2D cartesian broadcast (which triggers expensive
#    log-scaled tx.gatherscatter chains).  Grid capped at 16 CTAs.
#
# 4. Binary-search path: kept as fallback for large in1.  Grid capped at 16.
# ===========================================================================

# When in1 has ≤ this many unique values, use the SIMD values-loop kernel
# instead of binary search.
_MAX_VALUES_FOR_SIMD_SCAN = 256


# ===========================================================================
# Path 0: Tensor_Scalar / Scalar_Tensor — elementwise compare, no kernel.
# ===========================================================================


def _isin_tensor_scalar(in0: torch.Tensor, scalar, invert: bool) -> torch.Tensor:
    """in0 is a tensor, in1 is a scalar: just elementwise == then invert."""
    out = torch.eq(in0, scalar)
    if invert:
        out = torch.logical_not(out)
    return out


def _isin_scalar_tensor(scalar, in1: torch.Tensor, invert: bool) -> torch.Tensor:
    """in0 is a scalar, in1 is a tensor: reduce_any(in1 == scalar)."""
    out = torch.any(in1 == scalar)
    if invert:
        out = torch.logical_not(out)
    return out


# ===========================================================================
# Path 1: Value-domain SIMD linear scan.
# When in1 has ≤ _MAX_VALUES_FOR_SIMD_SCAN unique values, loop over them
# with fp32 vector compare — no integer division, no gather, pure SIMD.
# ===========================================================================


@libentry()
@triton.jit
def isin_values_loop_kernel(
    in0_ptr: tl.tensor,
    in1_ptr: tl.tensor,
    out_ptr: tl.tensor,
    M: int,
    N: int,
    BLOCK_M: tl.constexpr,
    invert: tl.constexpr,
):
    """Linear scan over in1 values using fp32 vector compare.

    For each in0 chunk, iterate over ALL in1 values, doing:
        out |= (in0_f == in1_val_f)

    Every op in the inner loop is a fp32 vector intrinsic on Tx81:
      in0_f == in1_val_f  →  tx.boolequalvv (fp-SIMD compare)
      out | ...           →  tx.boolorvv     (fp-SIMD boolean OR)

    N is guaranteed ≤ _MAX_VALUES_FOR_SIMD_SCAN (=256) by the caller.
    """
    pid = tle.program_id(0)
    ctas = tle.num_programs(0)
    for j in range(tl.cdiv(M, BLOCK_M * ctas)):
        global_pid = pid + j * ctas
        rows = global_pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = rows < M

        in0_val = tl.load(in0_ptr + rows, mask=mask).to(tl.float32)

        out = tl.zeros([BLOCK_M], dtype=tl.int1)
        for v in range(N):
            in1_val = tl.load(in1_ptr + v).to(tl.float32)  # scalar load
            out = out | (in0_val == in1_val)

        tl.store(out_ptr + rows, out ^ invert, mask=mask)


def _isin_values_loop(
    in0_ravel: torch.Tensor,
    in1_ravel: torch.Tensor,
    invert: bool,
):
    """Host-side wrapper for isin_values_loop_kernel.

    Returns a 1D bool tensor of shape (in0_ravel.numel(),).
    Caller is responsible for reshuffling (gather) and reshaping (view_as).
    """
    M = in0_ravel.numel()
    N = in1_ravel.numel()
    assert N <= _MAX_VALUES_FOR_SIMD_SCAN, f"N={N} exceeds SIMD scan limit"

    BLOCK_M = 1024 if M >= 1024 else triton.next_power_of_2(M)
    tiles_total = triton.cdiv(M, BLOCK_M)
    ctas_num = min(16, tiles_total)
    grid = (ctas_num,)

    out = torch.empty_like(in0_ravel, dtype=torch.bool)
    with torch_device_fn.device(in0_ravel.device.index):
        isin_values_loop_kernel[grid](
            in0_ravel, in1_ravel, out, M, N,
            BLOCK_M=BLOCK_M, invert=invert,
        )
    return out


# ===========================================================================
# Path 2: broadcast comparison — every in0 element vs every in1 element.
# Used when both inputs are small (≤ 12288) but in1 > _MAX_VALUES_FOR_SIMD_SCAN.
#
# Identical to upstream kernel.  Tx81 adaptation in host-side grid sizing.
# ===========================================================================


@triton.jit
def isin_by_comparation_impl(
    global_pid,
    in0_ravel_ptr: tl.tensor,
    in1_ravel_ptr: tl.tensor,
    out_ptr: tl.tensor,
    M: int,
    N: int,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    invert: tl.constexpr,
):
    row_off = global_pid * BLOCK_M
    rows = row_off + tl.arange(0, BLOCK_M)[:, None]
    row_mask = rows < M
    out_ptr += rows
    in0_ravel_ptr += rows + tl.zeros([BLOCK_N], dtype=tl.int32)
    in1_ravel_ptr += tl.zeros([BLOCK_M], dtype=tl.int32)[:, None]

    block = tl.full([BLOCK_M, BLOCK_N], value=(1 if invert else 0), dtype=tl.int1)
    in0 = tl.load(in0_ravel_ptr, row_mask, other=0)
    for col_off in range(0, N, BLOCK_N):
        cols = col_off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask
        in1 = tl.load(in1_ravel_ptr + cols, mask, other=0)
        block = tl.where(
            mask,
            tl.where(invert, block and (in0 != in1), block or (in0 == in1)),
            invert,
        )
    out = tl.reduce(block, axis=1, combine_fn=(reduce_all if invert else reduce_any))
    tl.store(out_ptr, out[:, None], row_mask)


@libentry()
@triton.jit
def isin_by_comparation_kernel(
    in0_ravel_ptr: tl.tensor,
    in1_ravel_ptr: tl.tensor,
    out_ptr: tl.tensor,
    M: int,
    N: int,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    tiles_per_cta: int,
    invert: tl.constexpr,
):
    pid = tle.program_id(0)
    ctas_num = tle.num_programs(0)
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        isin_by_comparation_impl(
            global_pid, in0_ravel_ptr, in1_ravel_ptr, out_ptr,
            M, N, BLOCK_M, BLOCK_N, invert,
        )


def isin_by_comparation(in0: torch.Tensor, in1: torch.Tensor, invert: bool):
    in0_ravel = in0.contiguous().ravel()
    in1_ravel = in1.contiguous().ravel()
    M = in0.numel()
    N = in1.numel()

    # Upstream heuristic for BLOCK_M/BLOCK_N — keep the tiny SPM footprint.
    if M <= 1024:
        BLOCK_M, BLOCK_N, num_warps = 1, min(256, triton.next_power_of_2(N)), 4
    elif M <= 3072:
        BLOCK_M, BLOCK_N, num_warps = 2, min(256, triton.next_power_of_2(N)), 4
    elif M <= 6144:
        BLOCK_M, BLOCK_N, num_warps = 4, min(128, triton.next_power_of_2(N)), 4
    elif M <= 9216:
        BLOCK_M, BLOCK_N, num_warps = 4, min(256, triton.next_power_of_2(N)), 8
    else:
        BLOCK_M, BLOCK_N, num_warps = 4, min(128, triton.next_power_of_2(N)), 4

    tiles_total = triton.cdiv(M, BLOCK_M)
    ctas_num = min(16, tiles_total)
    tiles_per_cta = triton.cdiv(tiles_total, ctas_num)
    grid = (ctas_num,)

    out = torch.empty_like(in0_ravel, dtype=torch.bool)
    with torch_device_fn.device(in0_ravel.device.index):
        isin_by_comparation_kernel[grid](
            in0_ravel, in1_ravel, out, M, N,
            BLOCK_M, BLOCK_N,
            tiles_per_cta=tiles_per_cta,
            invert=invert,
            num_warps=num_warps,
        )
    return out.view_as(in0)


# ===========================================================================
# Path 3: binary search (fallback) — sort in1, binary-search each in0.
# Used when in1 is too large for the values-loop kernel.
#
# Identical to upstream kernel.  Tx81 adaptation in host-side grid sizing.
# ===========================================================================


@triton.jit
def isin_by_search_impl(
    global_pid,
    in0_ravel_ptr: tl.tensor,
    in1_sorted_ptr: tl.tensor,
    out_ptr: tl.tensor,
    M: int,
    N: int,
    log_n: tl.constexpr,
    BLOCK_M: tl.constexpr,
    invert: tl.constexpr,
):
    r = tl.arange(0, BLOCK_M)
    i0 = global_pid * BLOCK_M + r
    mask = i0 < M

    in0_ravel = tl.load(in0_ravel_ptr + i0, mask=mask)

    out = tl.zeros_like(r).to(tl.int1)
    start = tl.zeros_like(r)
    end = start + N
    while_mask = start < end
    for i in range(log_n):
        mid = tl.where(while_mask, start + (end - start) // 2, 0)
        mid_val = tl.load(in1_sorted_ptr + mid, mask=while_mask)
        out = tl.where(while_mask, out or (mid_val == in0_ravel), out)
        start = tl.where(while_mask and (mid_val < in0_ravel), mid + 1, start)
        end = tl.where(while_mask and (mid_val > in0_ravel), mid, end)
        while_mask = start < end

    tl.store(out_ptr + i0, not out if invert else out, mask=mask)


@libentry()
@triton.jit
def isin_by_search_kernel(
    in0_ravel_ptr: tl.tensor,
    in1_sorted_ptr: tl.tensor,
    out_ptr: tl.tensor,
    M: int,
    N: int,
    log_n: tl.constexpr,
    BLOCK_M: tl.constexpr,
    tiles_per_cta: int,
    invert: tl.constexpr,
):
    pid = tle.program_id(0)
    ctas_num = tle.num_programs(0)
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        isin_by_search_impl(
            global_pid, in0_ravel_ptr, in1_sorted_ptr, out_ptr,
            M, N, log_n, BLOCK_M, invert,
        )


def isin_by_search(
    in0: torch.Tensor,
    in1: torch.Tensor,
    invert: bool,
    unique_in0: bool,
    unique_in1: bool,
):
    # ---- preprocess: unique / sort ----
    if unique_in0:
        in0_ravel, unique_order, _ = _unique2(
            in0, sorted=True, return_inverse=True, return_counts=False
        )
    else:
        in0_ravel = in0.contiguous().ravel()
        unique_order = None

    if unique_in1:
        in1_ravel, _, _ = _unique2(
            in1, sorted=True, return_inverse=False, return_counts=False
        )
    else:
        in1_ravel, _ = torch.sort(in1.ravel())

    M = in0_ravel.numel()
    N = in1_ravel.numel()

    # ---- NEW: if in1 is small after unique/sort, use SIMD linear scan ----
    if N <= _MAX_VALUES_FOR_SIMD_SCAN:
        out = _isin_values_loop(in0_ravel, in1_ravel, invert)
        if unique_order is not None:
            out = torch.gather(out, 0, unique_order.ravel().to(torch.int64))
        return out.view_as(in0)

    # ---- fallback: binary search ----
    if M <= 1048576:
        BLOCK_M, num_warps = 512, 4
    elif M <= 4194304:
        BLOCK_M, num_warps = 1024, 4
    elif M <= 8388608:
        BLOCK_M, num_warps = 2048, 4
    elif M <= 268435456:
        BLOCK_M, num_warps = 4096, 8
    else:
        BLOCK_M, num_warps = 2048, 4

    log_n = int(math.log2(N)) + 1
    tiles_total = triton.cdiv(M, BLOCK_M)
    ctas_num = min(16, tiles_total)
    tiles_per_cta = triton.cdiv(tiles_total, ctas_num)
    grid = (ctas_num,)

    out = torch.empty_like(in0_ravel, dtype=torch.bool)
    with torch_device_fn.device(in0_ravel.device.index):
        isin_by_search_kernel[grid](
            in0_ravel, in1_ravel, out, M, N,
            log_n, BLOCK_M,
            tiles_per_cta=tiles_per_cta,
            invert=invert,
            num_warps=num_warps,
        )
    if unique_order is not None:
        out = torch.gather(out, 0, unique_order.ravel().to(torch.int64))
    return out.view_as(in0)


# ===========================================================================
# Main entry point.
# ===========================================================================


def isin(
    in0,
    in1,
    *,
    assume_unique: bool = False,
    invert: bool = False,
) -> torch.Tensor:
    logger.debug("GEMS TSINGMICRO ISIN")

    # ---- Scalar_Tensor: any(in1 == scalar) ^ invert ----
    if not torch.is_tensor(in0):
        assert torch.is_tensor(in1), "Scalar_Tensor: in1 must be a tensor"
        return _isin_scalar_tensor(in0, in1, invert)

    # ---- Tensor_Scalar: (in0 == scalar) ^ invert ----
    if not torch.is_tensor(in1):
        assert torch.is_tensor(in0), "Tensor_Scalar: in0 must be a tensor"
        return _isin_tensor_scalar(in0, in1, invert)

    # ---- empty input ----
    if in0.numel() == 0 or in1.numel() == 0:
        return torch.zeros_like(in0, dtype=torch.bool)

    # ---- small both → comparation path ----
    if in0.numel() <= 12288 and in1.numel() <= 12288:
        # If in1 is tiny, SIMD linear scan is better than block-broadcast
        if in1.numel() <= _MAX_VALUES_FOR_SIMD_SCAN:
            in0_ravel = in0.contiguous().ravel()
            in1_ravel = in1.contiguous().ravel()
            out = _isin_values_loop(in0_ravel, in1_ravel, invert)
            return out.view_as(in0)
        return isin_by_comparation(in0, in1, invert)

    # ---- large inputs → search path (with SIMD scan when possible) ----
    if assume_unique or in1.numel() <= 4194304:
        return isin_by_search(in0, in1, invert, unique_in0=False, unique_in1=False)
    else:
        return isin_by_search(in0, in1, invert, unique_in0=False, unique_in1=True)
