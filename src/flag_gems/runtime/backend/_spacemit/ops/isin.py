"""
SPACEMIT-specific isin implementation with fixed logical operators.
"""
import math

import torch
import triton
import triton.language as tl

from flag_gems.utils.libentry import libentry
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as tle


def launch_arg(BLOCK_M, BLOCK_N, N, num_warps):
    return BLOCK_M, min(BLOCK_N, triton.next_power_of_2(N)), num_warps


# Import reduce functions from base ops
from flag_gems.ops.all import reduce_all
from flag_gems.ops.any import reduce_any


@triton.jit
def isin_by_comparation_impl(
    global_pid,
    in0_ravel_ptr: tl.tensor,
    in1_ravel_ptr: tl.tensor,  # in
    out_ptr: tl.tensor,  # out
    M: int,  # num_tasks
    N: int,  # num_tasks_1
    BLOCK_M: tl.constexpr,  # tile_size
    BLOCK_N: tl.constexpr,  # tile_size_1
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
        # FIX: Use & instead of 'and' for tensor operations
        mask = row_mask & col_mask
        in1 = tl.load(in1_ravel_ptr + cols, mask, other=0)
        # FIX: Use & and | instead of 'and' and 'or' for tensor operations
        block = tl.where(
            mask,
            tl.where(invert, block & (in0 != in1), block | (in0 == in1)),
            invert,
        )
    out = tl.reduce(block, axis=1, combine_fn=(reduce_all if invert else reduce_any))
    tl.store(out_ptr, out[:, None], row_mask)


@libentry()
@triton.jit
def isin_by_comparation_kernel(
    in0_ravel_ptr: tl.tensor,
    in1_ravel_ptr: tl.tensor,  # in
    out_ptr: tl.tensor,  # out
    M: int,  # num_tasks
    N: int,  # num_tasks_1
    BLOCK_M: tl.constexpr,  # tile_size
    BLOCK_N: tl.constexpr,  # tile_size_1
    tiles_per_cta: int,
    invert: tl.constexpr,
):
    pid = tle.program_id(0)
    ctas_num = tle.num_programs(0)
    # grid-stride-loop style kernel
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        isin_by_comparation_impl(
            global_pid,
            in0_ravel_ptr,
            in1_ravel_ptr,  # in
            out_ptr,  # out
            M,
            N,
            BLOCK_M,
            BLOCK_N,
            invert,
        )


def isin_by_comparation(
    in0: torch.tensor,
    in1: torch.tensor,
    invert: bool,
):
    in0_ravel = in0.contiguous().ravel()
    in1_ravel = in1.contiguous().ravel()
    M = in0.numel()
    N = in1.numel()
    if M <= 1024:
        BLOCK_M, BLOCK_N, num_warps = launch_arg(1, 256, N, 4)
    elif M <= 3072:
        BLOCK_M, BLOCK_N, num_warps = launch_arg(2, 256, N, 4)
    elif M <= 6144:
        BLOCK_M, BLOCK_N, num_warps = launch_arg(4, 128, N, 4)
    else:
        BLOCK_M, BLOCK_N, num_warps = launch_arg(8, 128, N, 4)

    out = torch.empty(in0.shape, dtype=torch.bool, device=in0.device)

    tiles_per_cta = 1
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) // tiles_per_cta,)

    with torch_device_fn.device(in0.device):
        isin_by_comparation_kernel[grid](
            in0_ravel,
            in1_ravel,
            out,
            M,
            N,
            BLOCK_M,
            BLOCK_N,
            tiles_per_cta,
            invert,
            num_warps=num_warps,
        )
    return out


def isin(elements, test_elements, assume_unique=False, invert=False):
    """
    SPACEMIT-specific isin with fixed logical operators.
    """
    return isin_by_comparation(elements, test_elements, invert)
