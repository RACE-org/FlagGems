import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.diag_embed import diag_embed as _generic_diag_embed
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_FILL_BLOCK = 4096
_DIAG_BLOCK = 1024
_SMALL_RANK1_THRESHOLD = 512
_SMALL_RANK2_LAST_DIM_THRESHOLD = 512
_SMALL_RANK2_BATCH_THRESHOLD = 16


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def _zero_fill_kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, 0.0, mask=mask)


@libentry()
@triton.jit
def _diag_embed_rank1_kernel(
    input_ptr,
    output_ptr,
    N: tl.constexpr,
    OUT_BASE: tl.constexpr,
    OUT_STEP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N

    vals = tl.load(input_ptr + idx, mask=mask, other=0.0)
    tl.store(output_ptr + OUT_BASE + idx * OUT_STEP, vals, mask=mask)


@libentry()
@triton.jit
def _diag_embed_rank2_kernel(
    input_ptr,
    output_ptr,
    N: tl.constexpr,
    BATCH_STRIDE: tl.constexpr,
    OUT_BASE: tl.constexpr,
    OUT_STEP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    diag_idx = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = diag_idx < N

    vals = tl.load(input_ptr + batch_idx * N + diag_idx, mask=mask, other=0.0)
    out_offsets = batch_idx * BATCH_STRIDE + OUT_BASE + diag_idx * OUT_STEP
    tl.store(output_ptr + out_offsets, vals, mask=mask)


def _zero_fill(output):
    n_elements = output.numel()
    if n_elements == 0:
        return
    grid = (triton.cdiv(n_elements, _FILL_BLOCK),)
    with torch_device_fn.device(output.device):
        _zero_fill_kernel[grid](output, n_elements, BLOCK_SIZE=_FILL_BLOCK)


def _normalize_dims(x, offset, dim1, dim2):
    rank = x.ndim + 1

    assert dim1 >= -rank and dim1 < rank, f"Invalid dim1: {dim1}"
    assert dim2 >= -rank and dim2 < rank, f"Invalid dim2: {dim2}"

    dim1 = dim1 % rank
    dim2 = dim2 % rank
    assert dim1 != dim2, "diagonal dimensions cannot be identical"

    # PyTorch documents dim exchange as equivalent to offset sign exchange.
    if dim1 > dim2:
        offset = -offset
        dim1, dim2 = dim2, dim1

    return offset, dim1, dim2


def _make_output(x, offset, dim1, dim2):
    last_dim = x.size(-1) + abs(offset)
    y_shape = list(x.shape)
    y_shape.pop()
    y_shape.insert(dim1, last_dim)
    y_shape.insert(dim2, last_dim)
    return torch.empty(y_shape, dtype=x.dtype, device=x.device)


def _diag_base_and_step(output, offset, dim1, dim2):
    dim1_stride = output.stride(dim1)
    dim2_stride = output.stride(dim2)
    if offset >= 0:
        out_base = offset * dim2_stride
    else:
        out_base = (-offset) * dim1_stride
    return out_base, dim1_stride + dim2_stride


def _diag_embed_rank1(x, output, offset, dim1, dim2):
    n = x.size(-1)
    if n == 0:
        return output

    out_base, out_step = _diag_base_and_step(output, offset, dim1, dim2)
    grid = (triton.cdiv(n, _DIAG_BLOCK),)
    with torch_device_fn.device(x.device):
        _diag_embed_rank1_kernel[grid](
            x,
            output,
            n,
            out_base,
            out_step,
            BLOCK_SIZE=_DIAG_BLOCK,
        )
    return output


def _diag_embed_rank2(x, output, offset, dim1, dim2):
    b, n = x.shape
    total = x.numel()
    if total == 0:
        return output

    prefix_pos = next(i for i in range(output.ndim) if i not in (dim1, dim2))
    batch_stride = output.stride(prefix_pos)
    out_base, out_step = _diag_base_and_step(output, offset, dim1, dim2)

    grid = (b, triton.cdiv(n, _DIAG_BLOCK))
    with torch_device_fn.device(x.device):
        _diag_embed_rank2_kernel[grid](
            x,
            output,
            n,
            batch_stride,
            out_base,
            out_step,
            BLOCK_SIZE=_DIAG_BLOCK,
        )
    return output


def diag_embed(x, offset=0, dim1=-2, dim2=-1):
    logger.debug("GEMS TSINGMICRO DIAG_EMBED")

    if x.ndim not in (1, 2):
        return _generic_diag_embed(x, offset, dim1, dim2)

    if x.ndim == 1 and x.size(-1) <= _SMALL_RANK1_THRESHOLD:
        # Small 1D diag_embed is launch-bound, similar to diag(256,).
        return _generic_diag_embed(x, offset, dim1, dim2)

    if (
        x.ndim == 2
        and x.size(-1) <= _SMALL_RANK2_LAST_DIM_THRESHOLD
        and x.size(0) <= _SMALL_RANK2_BATCH_THRESHOLD
    ):
        # For tiny [B, N], the generic diagonal-view copy can be cheaper than
        # separate zero-fill and scatter kernels.
        return _generic_diag_embed(x, offset, dim1, dim2)

    offset, dim1, dim2 = _normalize_dims(x, offset, dim1, dim2)
    x = x.contiguous()
    output = _make_output(x, offset, dim1, dim2)
    _zero_fill(output)

    if x.ndim == 1:
        return _diag_embed_rank1(x, output, offset, dim1, dim2)
    return _diag_embed_rank2(x, output, offset, dim1, dim2)
