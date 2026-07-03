import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.diag import diag as _generic_diag
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_FILL_BLOCK = 4096
_DIAG_BLOCK = 1024
_SMALL_1D_TO_2D_THRESHOLD = 512


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def _zero_fill_kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, 0.0, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N", "IN_STRIDE", "OUT_BASE", "OUT_STEP"])
def _diag_1d_to_2d_kernel(
    input_ptr,
    output_ptr,
    N,
    IN_STRIDE,
    OUT_BASE,
    OUT_STEP,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N

    vals = tl.load(input_ptr + idx * IN_STRIDE, mask=mask, other=0.0)
    # The target diagonal is linear: base + i * (M + 1). This avoids per-lane
    # row/column decomposition, which is expensive on Tx81 integer paths.
    tl.store(output_ptr + OUT_BASE + idx * OUT_STEP, vals, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N", "IN_BASE", "IN_STEP"])
def _diag_2d_to_1d_kernel(
    input_ptr,
    output_ptr,
    N,
    IN_BASE,
    IN_STEP,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N

    # Same linear diagonal address: base + i * (stride0 + stride1).
    vals = tl.load(input_ptr + IN_BASE + idx * IN_STEP, mask=mask, other=0.0)
    tl.store(output_ptr + idx, vals, mask=mask)


def _zero_fill(output):
    n_elements = output.numel()
    if n_elements == 0:
        return
    grid = (triton.cdiv(n_elements, _FILL_BLOCK),)
    with torch_device_fn.device(output.device):
        _zero_fill_kernel[grid](output, n_elements, BLOCK_SIZE=_FILL_BLOCK)


def diag_1d_to_2d(x, diagonal=0):
    n = x.shape[0]
    if n <= _SMALL_1D_TO_2D_THRESHOLD:
        # For small vectors the two custom kernels are launch-bound. The
        # original path's smaller diagonal block is faster on Tx81 for N=256.
        return _generic_diag(x, diagonal)

    m = n + abs(diagonal)
    output = torch.empty((m, m), dtype=x.dtype, device=x.device)
    _zero_fill(output)

    if n == 0:
        return output

    if diagonal >= 0:
        out_base = diagonal
    else:
        out_base = (-diagonal) * m
    out_step = m + 1

    grid = (triton.cdiv(n, _DIAG_BLOCK),)
    with torch_device_fn.device(x.device):
        _diag_1d_to_2d_kernel[grid](
            x,
            output,
            n,
            x.stride(0),
            out_base,
            out_step,
            BLOCK_SIZE=_DIAG_BLOCK,
        )
    return output


def diag_2d_to_1d(x, diagonal=0):
    rows, cols = x.shape
    if diagonal >= 0:
        diag_len = min(rows, cols - diagonal)
        in_base = diagonal * x.stride(1)
    else:
        diag_len = min(rows + diagonal, cols)
        in_base = (-diagonal) * x.stride(0)

    if diag_len <= 0:
        return torch.empty(0, dtype=x.dtype, device=x.device)

    output = torch.empty(diag_len, dtype=x.dtype, device=x.device)
    in_step = x.stride(0) + x.stride(1)

    grid = (triton.cdiv(diag_len, _DIAG_BLOCK),)
    with torch_device_fn.device(x.device):
        _diag_2d_to_1d_kernel[grid](
            x,
            output,
            diag_len,
            in_base,
            in_step,
            BLOCK_SIZE=_DIAG_BLOCK,
        )
    return output


def diag(x, diagonal=0):
    logger.debug("GEMS TSINGMICRO DIAG")
    if x.dim() == 1:
        return diag_1d_to_2d(x, diagonal)
    if x.dim() == 2:
        return diag_2d_to_1d(x, diagonal)
    raise ValueError("Input must be a 1D or 2D tensor.")
