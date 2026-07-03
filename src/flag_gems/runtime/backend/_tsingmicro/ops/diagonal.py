import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.diagonal import diagonal_backward as _generic_diagonal_backward
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_FILL_BLOCK = 4096
_DIAG_BLOCK = 1024
_TILE_GRID = 16
_SMALL_RANK2_NUMEL_THRESHOLD = 256 * 256


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def _zero_fill_kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, 0.0, mask=mask)


@libentry()
@triton.jit(
    do_not_specialize=["DIAG_LEN", "N_BLOCKS", "GRAD_STEP", "OUT_BASE", "OUT_STEP"]
)
def _diagonal_backward_rank2_kernel(
    grad_ptr,
    out_ptr,
    DIAG_LEN: int,
    N_BLOCKS: int,
    GRAD_STEP: int,
    OUT_BASE: int,
    OUT_STEP: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    lane = tl.arange(0, BLOCK_SIZE)

    for block_id in tl.range(pid, N_BLOCKS, tl.num_programs(0)):
        diag_idx = block_id * BLOCK_SIZE + lane
        mask = diag_idx < DIAG_LEN

        vals = tl.load(grad_ptr + diag_idx * GRAD_STEP, mask=mask, other=0.0)
        # grad_input is contiguous, so every diagonal position is a linear
        # base + i * (stride(dim1) + stride(dim2)) scatter.
        tl.store(out_ptr + OUT_BASE + diag_idx * OUT_STEP, vals, mask=mask)


@libentry()
@triton.jit(
    do_not_specialize=[
        "PREFIX_SIZE",
        "DIAG_LEN",
        "N_BLOCKS",
        "TOTAL_JOBS",
        "GRAD_PREFIX_STRIDE",
        "GRAD_DIAG_STRIDE",
        "PREFIX_STRIDE",
        "OUT_BASE",
        "OUT_STEP",
    ]
)
def _diagonal_backward_rank3_kernel(
    grad_ptr,
    out_ptr,
    PREFIX_SIZE: int,
    DIAG_LEN: int,
    N_BLOCKS: int,
    TOTAL_JOBS: int,
    GRAD_PREFIX_STRIDE: int,
    GRAD_DIAG_STRIDE: int,
    PREFIX_STRIDE: int,
    OUT_BASE: int,
    OUT_STEP: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    lane = tl.arange(0, BLOCK_SIZE)

    for job in tl.range(pid, TOTAL_JOBS, tl.num_programs(0)):
        prefix_idx = job // N_BLOCKS
        block_idx = job - prefix_idx * N_BLOCKS
        diag_idx = block_idx * BLOCK_SIZE + lane
        mask = (prefix_idx < PREFIX_SIZE) & (diag_idx < DIAG_LEN)

        vals = tl.load(
            grad_ptr + prefix_idx * GRAD_PREFIX_STRIDE + diag_idx * GRAD_DIAG_STRIDE,
            mask=mask,
            other=0.0,
        )
        out_offsets = prefix_idx * PREFIX_STRIDE + OUT_BASE + diag_idx * OUT_STEP
        tl.store(out_ptr + out_offsets, vals, mask=mask)


@libentry()
@triton.jit(
    do_not_specialize=[
        "PREFIX0_SIZE",
        "PREFIX1_SIZE",
        "DIAG_LEN",
        "N_BLOCKS",
        "TOTAL_JOBS",
        "GRAD_PREFIX0_STRIDE",
        "GRAD_PREFIX1_STRIDE",
        "GRAD_DIAG_STRIDE",
        "PREFIX0_STRIDE",
        "PREFIX1_STRIDE",
        "OUT_BASE",
        "OUT_STEP",
    ]
)
def _diagonal_backward_rank4_kernel(
    grad_ptr,
    out_ptr,
    PREFIX0_SIZE: int,
    PREFIX1_SIZE: int,
    DIAG_LEN: int,
    N_BLOCKS: int,
    TOTAL_JOBS: int,
    GRAD_PREFIX0_STRIDE: int,
    GRAD_PREFIX1_STRIDE: int,
    GRAD_DIAG_STRIDE: int,
    PREFIX0_STRIDE: int,
    PREFIX1_STRIDE: int,
    OUT_BASE: int,
    OUT_STEP: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    lane = tl.arange(0, BLOCK_SIZE)

    for job in tl.range(pid, TOTAL_JOBS, tl.num_programs(0)):
        prefix_pair = job // N_BLOCKS
        block_idx = job - prefix_pair * N_BLOCKS
        prefix0_idx = prefix_pair // PREFIX1_SIZE
        prefix1_idx = prefix_pair - prefix0_idx * PREFIX1_SIZE
        diag_idx = block_idx * BLOCK_SIZE + lane
        mask = (
            (prefix0_idx < PREFIX0_SIZE)
            & (prefix1_idx < PREFIX1_SIZE)
            & (diag_idx < DIAG_LEN)
        )

        grad_base = (
            prefix0_idx * GRAD_PREFIX0_STRIDE
            + prefix1_idx * GRAD_PREFIX1_STRIDE
        )
        vals = tl.load(
            grad_ptr + grad_base + diag_idx * GRAD_DIAG_STRIDE,
            mask=mask,
            other=0.0,
        )
        out_offsets = (
            prefix0_idx * PREFIX0_STRIDE
            + prefix1_idx * PREFIX1_STRIDE
            + OUT_BASE
            + diag_idx * OUT_STEP
        )
        tl.store(out_ptr + out_offsets, vals, mask=mask)


def _zero_fill(output):
    n_elements = output.numel()
    if n_elements == 0:
        return
    grid = (triton.cdiv(n_elements, _FILL_BLOCK),)
    with torch_device_fn.device(output.device):
        _zero_fill_kernel[grid](output, n_elements, BLOCK_SIZE=_FILL_BLOCK)


def _contiguous_strides(shape):
    strides = [1] * len(shape)
    running = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = running
        running *= shape[i]
    return strides


def _shape_numel(shape):
    n_elements = 1
    for size in shape:
        n_elements *= size
    return n_elements


def _diag_layout(input_sizes, offset, dim1, dim2):
    rank = len(input_sizes)
    dim1 = dim1 % rank
    dim2 = dim2 % rank
    if dim1 == dim2:
        raise RuntimeError("diagonal dimensions cannot be identical")

    strides = _contiguous_strides(input_sizes)
    size1 = input_sizes[dim1]
    size2 = input_sizes[dim2]

    if offset >= 0:
        diag_len = max(0, min(size1, size2 - offset))
        out_base = offset * strides[dim2]
    else:
        diag_len = max(0, min(size1 + offset, size2))
        out_base = (-offset) * strides[dim1]

    out_step = strides[dim1] + strides[dim2]
    prefix_dims = [i for i in range(rank) if i not in (dim1, dim2)]
    return diag_len, out_base, out_step, prefix_dims, strides


def _diagonal_backward_rank2(grad_output, grad_input, diag_len, out_base, out_step):
    if diag_len == 0:
        return grad_input

    n_blocks = triton.cdiv(diag_len, _DIAG_BLOCK)
    grid = (min(_TILE_GRID, n_blocks),)
    with torch_device_fn.device(grad_output.device):
        _diagonal_backward_rank2_kernel[grid](
            grad_output,
            grad_input,
            diag_len,
            n_blocks,
            grad_output.stride(0),
            out_base,
            out_step,
            BLOCK_SIZE=_DIAG_BLOCK,
        )
    return grad_input


def _diagonal_backward_rank3(
    grad_output, grad_input, input_sizes, diag_len, out_base, out_step, prefix_dims, strides
):
    if diag_len == 0:
        return grad_input

    prefix_dim = prefix_dims[0]
    prefix_size = input_sizes[prefix_dim]
    n_blocks = triton.cdiv(diag_len, _DIAG_BLOCK)
    total_jobs = prefix_size * n_blocks
    grid = (min(_TILE_GRID, total_jobs),)
    with torch_device_fn.device(grad_output.device):
        _diagonal_backward_rank3_kernel[grid](
            grad_output,
            grad_input,
            prefix_size,
            diag_len,
            n_blocks,
            total_jobs,
            grad_output.stride(0),
            grad_output.stride(1),
            strides[prefix_dim],
            out_base,
            out_step,
            BLOCK_SIZE=_DIAG_BLOCK,
        )
    return grad_input


def _diagonal_backward_rank4(
    grad_output, grad_input, input_sizes, diag_len, out_base, out_step, prefix_dims, strides
):
    if diag_len == 0:
        return grad_input

    prefix0_dim, prefix1_dim = prefix_dims
    prefix0_size = input_sizes[prefix0_dim]
    prefix1_size = input_sizes[prefix1_dim]
    n_blocks = triton.cdiv(diag_len, _DIAG_BLOCK)
    total_jobs = prefix0_size * prefix1_size * n_blocks
    grid = (min(_TILE_GRID, total_jobs),)
    with torch_device_fn.device(grad_output.device):
        _diagonal_backward_rank4_kernel[grid](
            grad_output,
            grad_input,
            prefix0_size,
            prefix1_size,
            diag_len,
            n_blocks,
            total_jobs,
            grad_output.stride(0),
            grad_output.stride(1),
            grad_output.stride(2),
            strides[prefix0_dim],
            strides[prefix1_dim],
            out_base,
            out_step,
            BLOCK_SIZE=_DIAG_BLOCK,
        )
    return grad_input


def diagonal_backward(grad_output, input_sizes, offset, dim1, dim2):
    logger.debug("GEMS TSINGMICRO DIAGONAL_BACKWARD")

    input_sizes = tuple(int(s) for s in input_sizes)
    rank = len(input_sizes)
    if (
        rank not in (2, 3, 4)
        or grad_output.is_complex()
        or grad_output.dtype is torch.float64
        or dim1 < -rank
        or dim1 >= rank
        or dim2 < -rank
        or dim2 >= rank
    ):
        return _generic_diagonal_backward(grad_output, input_sizes, offset, dim1, dim2)

    if rank == 2 and _shape_numel(input_sizes) <= _SMALL_RANK2_NUMEL_THRESHOLD:
        # For small matrices such as 256x256, this op is launch-bound: the
        # custom path runs zero-fill + scatter, while the generic diagonal-view
        # copy is slightly cheaper on Tx81.
        return _generic_diagonal_backward(grad_output, input_sizes, offset, dim1, dim2)

    diag_len, out_base, out_step, prefix_dims, strides = _diag_layout(
        input_sizes, offset, dim1, dim2
    )
    grad_input = torch.empty(input_sizes, dtype=grad_output.dtype, device=grad_output.device)
    _zero_fill(grad_input)

    if rank == 2:
        return _diagonal_backward_rank2(
            grad_output, grad_input, diag_len, out_base, out_step
        )
    if rank == 3:
        return _diagonal_backward_rank3(
            grad_output,
            grad_input,
            input_sizes,
            diag_len,
            out_base,
            out_step,
            prefix_dims,
            strides,
        )
    return _diagonal_backward_rank4(
        grad_output,
        grad_input,
        input_sizes,
        diag_len,
        out_base,
        out_step,
        prefix_dims,
        strides,
    )
