import logging
from numbers import Number

import torch
import triton
import triton.language as tl

from flag_gems.ops.lerp import (
    lerp_scalar as _generic_lerp_scalar,
    lerp_scalar_ as _generic_lerp_scalar_,
    lerp_tensor as _generic_lerp_tensor,
    lerp_tensor_ as _generic_lerp_tensor_,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_TILE_GRID = 16
_BLOCK_SIZE = 4096
_SMALL_NUMEL_THRESHOLD = 1024


@libentry()
@triton.jit(do_not_specialize=["N", "WEIGHT"])
def _lerp_scalar_head_kernel(
    input_ptr,
    end_ptr,
    out_ptr,
    N: int,
    WEIGHT: float,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    lane = tl.arange(0, BLOCK_SIZE)

    for base in tl.range(pid * BLOCK_SIZE, N, tl.num_programs(0) * BLOCK_SIZE):
        offsets = base + lane
        mask = offsets < N
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(end_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, x + WEIGHT * (y - x), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N", "WEIGHT"])
def _lerp_scalar_tail_kernel(
    input_ptr,
    end_ptr,
    out_ptr,
    N: int,
    WEIGHT: float,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    lane = tl.arange(0, BLOCK_SIZE)

    for base in tl.range(pid * BLOCK_SIZE, N, tl.num_programs(0) * BLOCK_SIZE):
        offsets = base + lane
        mask = offsets < N
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(end_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, y - (y - x) * (1.0 - WEIGHT), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _lerp_tensor_kernel(
    input_ptr,
    end_ptr,
    weight_ptr,
    out_ptr,
    N: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    lane = tl.arange(0, BLOCK_SIZE)

    for base in tl.range(pid * BLOCK_SIZE, N, tl.num_programs(0) * BLOCK_SIZE):
        offsets = base + lane
        mask = offsets < N
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(end_ptr + offsets, mask=mask, other=0.0)
        w = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        head = x + w * (y - x)
        tail = y - (y - x) * (1.0 - w)
        tl.store(out_ptr + offsets, tl.where(tl.abs(w) < 0.5, head, tail), mask=mask)


def _same_shape_contiguous(input, end):
    return input.shape == end.shape and input.is_contiguous() and end.is_contiguous()


def _can_use_fast_binary(input, end):
    return (
        input.is_floating_point()
        and end.is_floating_point()
        and input.dtype == end.dtype
        and _same_shape_contiguous(input, end)
        and input.numel() > _SMALL_NUMEL_THRESHOLD
    )


def _can_use_fast_tensor(input, end, weight):
    return (
        _can_use_fast_binary(input, end)
        and weight.is_floating_point()
        and weight.dtype == input.dtype
        and weight.shape == input.shape
        and weight.is_contiguous()
    )


def _launch_scalar(input, end, weight, out):
    n_elements = input.numel()
    grid = (min(_TILE_GRID, triton.cdiv(n_elements, _BLOCK_SIZE)),)
    with torch_device_fn.device(input.device):
        if weight < 0.5:
            _lerp_scalar_head_kernel[grid](
                input,
                end,
                out,
                n_elements,
                float(weight),
                BLOCK_SIZE=_BLOCK_SIZE,
                num_warps=8,
            )
        else:
            _lerp_scalar_tail_kernel[grid](
                input,
                end,
                out,
                n_elements,
                float(weight),
                BLOCK_SIZE=_BLOCK_SIZE,
                num_warps=8,
            )
    return out


def _launch_tensor(input, end, weight, out):
    n_elements = input.numel()
    grid = (min(_TILE_GRID, triton.cdiv(n_elements, _BLOCK_SIZE)),)
    with torch_device_fn.device(input.device):
        _lerp_tensor_kernel[grid](
            input,
            end,
            weight,
            out,
            n_elements,
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=8,
        )
    return out


def lerp_scalar(input, end, weight):
    logger.debug("GEMS TSINGMICRO LERP SCALAR")
    if not isinstance(weight, Number) or not _can_use_fast_binary(input, end):
        return _generic_lerp_scalar(input, end, weight)

    out = torch.empty_like(input)
    return _launch_scalar(input, end, weight, out)


def lerp_scalar_(input, end, weight):
    logger.debug("GEMS TSINGMICRO LERP SCALAR_")
    if not isinstance(weight, Number) or not _can_use_fast_binary(input, end):
        return _generic_lerp_scalar_(input, end, weight)

    return _launch_scalar(input, end, weight, input)


def lerp_tensor(input, end, weight):
    logger.debug("GEMS TSINGMICRO LERP TENSOR")
    if not _can_use_fast_tensor(input, end, weight):
        return _generic_lerp_tensor(input, end, weight)

    out = torch.empty_like(input)
    return _launch_tensor(input, end, weight, out)


def lerp_tensor_(input, end, weight):
    logger.debug("GEMS TSINGMICRO LERP TENSOR_")
    if not _can_use_fast_tensor(input, end, weight) or weight.data_ptr() == input.data_ptr():
        return _generic_lerp_tensor_(input, end, weight)

    return _launch_tensor(input, end, weight, input)
