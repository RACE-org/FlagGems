import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_min

TOTAL_CORE_NUM = 16

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def argmax_kernel_1(
    inp,
    mid_value,
    mid_index,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    # Tiny-M path: a single CTA handles the whole reduce.
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M
    min_value = get_dtype_min(inp.type.element_ty)
    inp_val = tl.load(inp + offset, mask=mask, other=min_value)
    max_val, max_index = tl.max(inp_val, axis=0, return_indices=True)
    max_index = max_index + pid * BLOCK_SIZE
    tl.store(mid_value + pid, max_val)
    tl.store(mid_index + pid, max_index)


@libentry()
@triton.jit
def argmax_kernel_full_tile(
    inp,
    mid_value,
    mid_index,
    M,
    CHUNK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # One CTA per tile (grid = TOTAL_CORE_NUM). Each CTA chews through a
    # contiguous CHUNK-sized slice via an inner BLOCK_SIZE loop. Tracks
    # (running_max, running_idx) as scalars across iterations. Strict `>`
    # update preserves first-occurrence tie-break (matches torch.argmax).
    pid = tle.program_id(0)
    start = pid * CHUNK
    min_value = get_dtype_min(inp.type.element_ty)
    running_max = min_value
    running_idx = 0
    for off in range(0, CHUNK, BLOCK_SIZE):
        offset = start + off + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        v = tl.load(inp + offset, mask=mask, other=min_value)
        local_max, local_pos = tl.max(v, axis=0, return_indices=True)
        local_idx_global = (start + off + local_pos).to(tl.int32)
        update = local_max > running_max
        running_max = tl.where(update, local_max, running_max)
        running_idx = tl.where(update, local_idx_global, running_idx)
    tl.store(mid_value + pid, running_max)
    tl.store(mid_index + pid, running_idx)


@libentry()
@triton.jit
def argmax_kernel_2(mid_value, mid_index, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid_value + offset
    mask = offset < mid_size
    min_value = get_dtype_min(mid_value.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=min_value)
    index_val = tl.argmax(mid_val, axis=0)
    mid_index_ptrs = mid_index + index_val
    out_val = tl.load(mid_index_ptrs)
    tl.store(out, out_val)


def _argmax_block_m(args):
    """Pick BLOCK_M so grid ≈ TOTAL_CORE_NUM (instead of upstream's BLOCK_M=8
    which produces 32+ rounds/tile on large M). Capped at 256 so per-CTA
    SPM stays well under 3MB even with BLOCK_N=4096."""
    return min(
        max(triton.next_power_of_2(triton.cdiv(args["M"], TOTAL_CORE_NUM)), 8),
        256,
    )

def _argmax_block_n(args):
    """Cover N in as few inner iterations as possible, capped at 4096 for
    SPM safety when BLOCK_M is at its cap."""
    return min(triton.next_power_of_2(args["N"]), 4096)


@libentry()
@triton.heuristics(
    {
        "BLOCK_M": _argmax_block_m,
        "BLOCK_N": _argmax_block_n,
    }
)
@triton.jit
def argmax_kernel(
    inp,
    out_index,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Dim-reduce path: same kernel body as upstream, but with a Tx81-tuned
    # heuristic that picks a larger BLOCK_M to keep grid ≈ TOTAL_CORE_NUM.
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    dtype = inp.type.element_ty
    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    min_value = get_dtype_min(dtype)
    max_values = tl.full([BLOCK_M], dtype=acc_type, value=min_value)
    argmax_values = tl.full([BLOCK_M], dtype=tl.int64, value=0)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value)
        local_max, local_argmax = tl.max(
            inp_vals, 1, return_indices=True, return_indices_tie_break_left=True
        )
        update = local_max > max_values
        max_values = tl.where(update, local_max, max_values)
        argmax_values = tl.where(update, start_n.to(tl.int64) + local_argmax.to(tl.int64), argmax_values)

    offset_index = m_offset * K + pid_k
    out_index_ptrs = out_index + offset_index
    mask1 = m_offset < M
    tl.store(out_index_ptrs, argmax_values, mask=mask1)


# Inner-loop BLOCK cap (per-CTA SPM-resident vector width).
_FULL_TILE_BLOCK = 65536

# Below this per-tile workload, 16-CTA split wastes launch overhead — use
# a single CTA instead.
_MIN_PER_TILE = 64


def _pick_block_size(per_tile):
    # Largest power of 2 ≤ per_tile, capped at _FULL_TILE_BLOCK, floored at
    # _MIN_PER_TILE. Keeps the inner-loop vector width proportional to
    # actual work.
    block = 1
    while block * 2 <= per_tile and block * 2 <= _FULL_TILE_BLOCK:
        block *= 2
    return max(block, _MIN_PER_TILE)


def argmax(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_TSINGMICRO ARGMAX")
    if dim is None:
        device = inp.device
        if dtype is None:
            dtype = inp.dtype
        M = inp.numel()

        if keepdim:
            shape = [1] * inp.dim()
            out = torch.empty(shape, dtype=torch.int64, device=device)
        else:
            out = torch.empty([], dtype=torch.int64, device=device)

        with torch_device_fn.device(device):
            per_tile = triton.cdiv(M, TOTAL_CORE_NUM)
            if per_tile >= _MIN_PER_TILE:
                # Spread work across 16 tiles; one launch.
                block_size = _pick_block_size(per_tile)
                chunk = triton.cdiv(per_tile, block_size) * block_size
                mid_value = torch.empty(
                    (TOTAL_CORE_NUM,), dtype=dtype, device=device
                )
                mid_index = torch.empty(
                    (TOTAL_CORE_NUM,), dtype=torch.int64, device=device
                )
                argmax_kernel_full_tile[(TOTAL_CORE_NUM,)](
                    inp, mid_value, mid_index, M, chunk, block_size
                )
                mid_size = TOTAL_CORE_NUM
            else:
                # Tiny M: one CTA, no point splitting across tiles.
                block_size = max(triton.next_power_of_2(M), 1)
                mid_value = torch.empty((1,), dtype=dtype, device=device)
                mid_index = torch.empty((1,), dtype=torch.int64, device=device)
                argmax_kernel_1[(1,)](
                    inp, mid_value, mid_index, M, block_size
                )
                mid_size = 1

            block_mid = triton.next_power_of_2(mid_size)
            argmax_kernel_2[(1,)](
                mid_value, mid_index, out, mid_size, block_mid
            )
        return out
    else:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        shape = inp.shape
        dim = dim % inp.ndim
        N = shape[dim]
        M = math.prod(shape[:dim])
        K = inp.numel() // M // N

        inp = inp.contiguous()

        shape_list = list(shape)
        shape_list[dim] = 1
        out_index = torch.empty(
            shape_list, dtype=torch.int64, device=inp.device
        )
        if not keepdim:
            out_index = torch.squeeze(out_index, dim)

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            K,
        )
        with torch_device_fn.device(inp.device):
            argmax_kernel[grid](
                inp,
                out_index,
                M,
                N,
                K,
            )

        return out_index
