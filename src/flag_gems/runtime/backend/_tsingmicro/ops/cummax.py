import logging
import math
from typing import List, Tuple, Union

import torch
import triton
import triton.language as tl

from flag_gems.ops.cummax import cummax as _generic_cummax
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_min

Tensor = torch.Tensor

logger = logging.getLogger(__name__)

_FP32_EXACT_INDEX_LIMIT = 1 << 24


@triton.jit
def _tl_cummax_fidx(input, index, axis=0):
    return tl.associative_scan(
        (input, index), axis, tle.maximum_with_index_tie_break_right
    )


@triton.jit
def _tl_max_fidx_tie_break_right(input, index, axis=None, keep_dims=False):
    return tl.reduce(
        (input, index),
        axis,
        tle.maximum_with_index_tie_break_right,
        keep_dims=keep_dims,
    )


def _pick_block_size(n):
    if n < 1024 * 4:
        return triton.next_power_of_2(n)
    return 1024


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def _store_indices_i64_kernel(
    idx_f32,
    out_indices,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    idx = tl.load(idx_f32 + offs, mask=mask, other=0.0)
    tl.store(out_indices + offs, idx.to(tl.int64), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def _scan_part_max_fidx_kernel(
    inp,
    out,
    out_indices_f32,
    partial_max,
    partial_max_indices_f32,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NEED_PARTIAL: tl.constexpr,
    USE_OUT_INDICES: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    min_value = get_dtype_min(inp.type.element_ty)
    vals = tl.load(inp + offs, mask=mask, other=min_value).to(tl.float32)
    if tl.constexpr(USE_OUT_INDICES):
        idxs = tl.load(out_indices_f32 + offs, mask=mask, other=0.0)
    else:
        # Keep scan indices in fp32. It is exact while the reduction dimension is
        # <= 2^24 and avoids carrying int64 through the associative scan chain.
        idxs = offs.to(tl.float32)

    result, cummax_indices = _tl_cummax_fidx(vals, idxs, axis=0)

    if tl.constexpr(NEED_PARTIAL):
        part_max, part_max_indices = _tl_max_fidx_tie_break_right(
            vals, idxs, axis=0
        )

    tl.store(out + offs, result, mask=mask)
    tl.store(out_indices_f32 + offs, cummax_indices, mask=mask)

    if tl.constexpr(NEED_PARTIAL):
        tl.store(partial_max + pid, part_max)
        tl.store(partial_max_indices_f32 + pid, part_max_indices)


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def _add_base_max_fidx_kernel(
    out,
    out_indices_f32,
    partial_max,
    partial_max_indices_f32,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    out_vals = tl.load(out + offs, mask=mask)
    out_indices = tl.load(out_indices_f32 + offs, mask=mask)

    if pid > 0:
        base_val = tl.load(partial_max + pid - 1)
        base_idx = tl.load(partial_max_indices_f32 + pid - 1)

        final_vals = tl.maximum(out_vals, base_val)
        # Current block has larger indices than all previous blocks, so equality
        # keeps the current index and preserves cummax's rightmost tie-break.
        final_indices = tl.where(out_vals >= base_val, out_indices, base_idx)
        tl.store(out + offs, final_vals.to(out_vals.dtype), mask=mask)
        tl.store(out_indices_f32 + offs, final_indices, mask=mask)


def _scan_then_fan_col_fidx(inp, out, out_indices_f32, n_ele, dtype, use_out_indices=False):
    block_size = _pick_block_size(n_ele)
    part_num = math.ceil(n_ele / block_size)
    need_partial = part_num >= 2
    if need_partial:
        partial_max = torch.empty(part_num, dtype=dtype, device=inp.device)
        partial_max_indices = torch.empty(
            part_num, dtype=torch.float32, device=inp.device
        )
    else:
        partial_max = None
        partial_max_indices = None

    grid = (part_num,)
    with torch_device_fn.device(inp.device):
        _scan_part_max_fidx_kernel[grid](
            inp,
            out,
            out_indices_f32,
            partial_max,
            partial_max_indices,
            n_ele,
            block_size,
            need_partial,
            use_out_indices,
        )

    if need_partial:
        _scan_then_fan_col_fidx(
            partial_max,
            partial_max,
            partial_max_indices,
            part_num,
            dtype,
            use_out_indices=True,
        )
        with torch_device_fn.device(inp.device):
            _add_base_max_fidx_kernel[grid](
                out,
                out_indices_f32,
                partial_max,
                partial_max_indices,
                n_ele,
                block_size,
            )


@libentry()
@triton.jit(do_not_specialize=["part_num"])
def _scan_part_max_abc_fidx_kernel(
    inp,
    out,
    out_indices_f32,
    partial_max,
    partial_max_indices_f32,
    B,
    C,
    part_num,
    BLOCK_SIZE: tl.constexpr,
    NEED_PARTIAL: tl.constexpr,
    USE_OUT_INDICES: tl.constexpr,
):
    pid_a = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_c = tl.program_id(2)

    b_idx = pid_b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset = pid_a * B * C + b_idx * C + pid_c
    part_offset = pid_a * part_num * C + pid_b * C + pid_c
    mask = b_idx < B

    min_value = get_dtype_min(inp.type.element_ty)
    vals = tl.load(inp + offset, mask=mask, other=min_value).to(tl.float32)
    if tl.constexpr(USE_OUT_INDICES):
        idxs = tl.load(out_indices_f32 + offset, mask=mask, other=0.0)
    else:
        idxs = b_idx.to(tl.float32)

    result, cummax_indices = _tl_cummax_fidx(vals, idxs, axis=0)

    if tl.constexpr(NEED_PARTIAL):
        part_max, part_max_indices = _tl_max_fidx_tie_break_right(
            vals, idxs, axis=0
        )

    tl.store(out + offset, result, mask=mask)
    tl.store(out_indices_f32 + offset, cummax_indices, mask=mask)

    if tl.constexpr(NEED_PARTIAL):
        tl.store(partial_max + part_offset, part_max)
        tl.store(partial_max_indices_f32 + part_offset, part_max_indices)


@libentry()
@triton.jit(do_not_specialize=["part_num"])
def _add_base_max_abc_fidx_kernel(
    out,
    out_indices_f32,
    partial_max,
    partial_max_indices_f32,
    B,
    C,
    part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid_a = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_c = tl.program_id(2)

    b_idx = pid_b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset = pid_a * B * C + b_idx * C + pid_c
    last_part_offset = pid_a * part_num * C + (pid_b - 1) * C + pid_c
    mask = b_idx < B

    out_vals = tl.load(out + offset, mask=mask)
    out_indices = tl.load(out_indices_f32 + offset, mask=mask)

    if pid_b > 0:
        base_val = tl.load(partial_max + last_part_offset)
        base_idx = tl.load(partial_max_indices_f32 + last_part_offset)

        final_vals = tl.maximum(out_vals, base_val)
        final_indices = tl.where(out_vals >= base_val, out_indices, base_idx)
        tl.store(out + offset, final_vals.to(out_vals.dtype), mask=mask)
        tl.store(out_indices_f32 + offset, final_indices, mask=mask)


def _scan_then_fan_fidx(
    inp,
    out,
    out_indices_f32,
    A,
    B,
    C,
    dtype,
    use_out_indices=False,
):
    block_size = _pick_block_size(B)
    part_num = math.ceil(B / block_size)
    need_partial = part_num >= 2
    if need_partial:
        partial_max = torch.empty(A, part_num, C, dtype=dtype, device=inp.device)
        partial_max_indices = torch.empty(
            A, part_num, C, dtype=torch.float32, device=inp.device
        )
    else:
        partial_max = None
        partial_max_indices = None

    grid = (A, part_num, C)
    with torch_device_fn.device(inp.device):
        _scan_part_max_abc_fidx_kernel[grid](
            inp,
            out,
            out_indices_f32,
            partial_max,
            partial_max_indices,
            B,
            C,
            part_num,
            block_size,
            need_partial,
            use_out_indices,
        )

    if need_partial:
        _scan_then_fan_fidx(
            partial_max,
            partial_max,
            partial_max_indices,
            A,
            part_num,
            C,
            dtype,
            use_out_indices=True,
        )
        with torch_device_fn.device(inp.device):
            _add_base_max_abc_fidx_kernel[grid](
                out,
                out_indices_f32,
                partial_max,
                partial_max_indices,
                B,
                C,
                part_num,
                block_size,
            )


@libentry()
@triton.jit
def _scan_part_max_abc_loop_fidx_kernel(
    inp,
    out,
    out_indices_f32,
    B,
    C,
    loop_num: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_a = tl.program_id(0)
    pid_c = tl.program_id(1)

    t_idx = tl.arange(0, BLOCK_SIZE)
    ac_offset = pid_a * B * C + pid_c

    min_value = get_dtype_min(inp.type.element_ty)
    prev_max_val = tl.full([], min_value, dtype=tl.float32)
    prev_max_idx = tl.full([], 0.0, dtype=tl.float32)
    last_mask = t_idx == (BLOCK_SIZE - 1)

    for l_idx in tl.range(loop_num):
        b_idx = l_idx * BLOCK_SIZE + t_idx
        mask = b_idx < B
        offset = ac_offset + b_idx * C

        vals = tl.load(inp + offset, mask=mask, other=min_value).to(tl.float32)
        idxs = b_idx.to(tl.float32)

        result, cummax_indices = _tl_cummax_fidx(vals, idxs, axis=0)

        prev_max_val_b = tl.broadcast_to(prev_max_val, (BLOCK_SIZE,))
        prev_max_idx_b = tl.broadcast_to(prev_max_idx, (BLOCK_SIZE,))

        cummax_indices = tl.where(
            result >= prev_max_val_b, cummax_indices, prev_max_idx_b
        )
        result = tl.maximum(result, prev_max_val_b)

        prev_max_val = tl.sum(tl.where(last_mask, result, 0.0), axis=0)
        prev_max_idx = tl.sum(tl.where(last_mask, cummax_indices, 0.0), axis=0)

        tl.store(out + offset, result, mask=mask)
        tl.store(out_indices_f32 + offset, cummax_indices, mask=mask)


def _scan_then_fan_loop_fidx(inp, out, out_indices_f32, A, B, C):
    block_size = _pick_block_size(B)
    loop_num = math.ceil(B / block_size)
    grid = (A, C)
    with torch_device_fn.device(inp.device):
        _scan_part_max_abc_loop_fidx_kernel[grid](
            inp,
            out,
            out_indices_f32,
            B,
            C,
            loop_num,
            block_size,
        )


def _finish_indices(out_indices_f32, out_indices):
    n_elements = out_indices.numel()
    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    with torch_device_fn.device(out_indices.device):
        _store_indices_i64_kernel[grid](
            out_indices_f32,
            out_indices,
            n_elements,
            block_size,
        )


def cummax(
    input: Tensor,
    dim: int,
    *,
    out: Union[Tensor, Tuple[Tensor, ...], List[Tensor], None] = None,
) -> torch.return_types.cummax:
    logger.debug("GEMS TSINGMICRO CUMMAX")
    assert dim >= -input.ndim and dim < input.ndim, "Invalid dim"

    shape = input.shape
    dim = dim % input.ndim
    N = shape[dim]

    # fp32 indices are exact only up to 2^24. Other dtypes keep the generic path
    # because value precision or integer semantics may be user-visible.
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return _generic_cummax(input, dim, out=out)
    if N > _FP32_EXACT_INDEX_LIMIT:
        return _generic_cummax(input, dim, out=out)

    M = 1
    for i in range(dim):
        M *= shape[i]
    input = input.contiguous()
    K = input.numel() // M // N

    if out is None:
        out_values = torch.empty_like(input)
        out_indices = torch.empty_like(input, dtype=torch.int64)
    else:
        out_values, out_indices = out

    idx_work = torch.empty_like(input, dtype=torch.float32)

    compute_dtype = torch.float32
    if M == 1 and K == 1:
        _scan_then_fan_col_fidx(input, out_values, idx_work, N, compute_dtype)
    elif M * K <= 16:
        _scan_then_fan_fidx(input, out_values, idx_work, M, N, K, compute_dtype)
    else:
        _scan_then_fan_loop_fidx(input, out_values, idx_work, M, N, K)

    _finish_indices(idx_work, out_indices)
    return out_values, out_indices
