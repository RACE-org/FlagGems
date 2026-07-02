import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_MAX_RANK = 5


@libentry()
@triton.jit
def _contiguous_row_copy_kernel(
    inp_ptr,
    out_ptr,
    M: int,
    N: int,
    RANK: tl.constexpr,
    ps0: int,
    ps1: int,
    ps2: int,
    ps3: int,
    pst0: int,
    pst1: int,
    pst2: int,
    pst3: int,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Copy rows as contiguous blocks when input stride[-1] is 1."""
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = rows < M
    cm = cols < N

    # Decompose the flattened prefix row id into input prefix coordinates.
    if RANK == 1:
        inp_row_start = tl.zeros((BLOCK_M,), dtype=tl.int32)
    elif RANK == 2:
        inp_row_start = rows * pst0
    elif RANK == 3:
        r = rows
        i1 = r % ps1
        i0 = r // ps1
        inp_row_start = i0 * pst0 + i1 * pst1
    elif RANK == 4:
        r = rows
        i2 = r % ps2
        r = r // ps2
        i1 = r % ps1
        i0 = r // ps1
        inp_row_start = i0 * pst0 + i1 * pst1 + i2 * pst2
    else:
        r = rows
        i3 = r % ps3
        r = r // ps3
        i2 = r % ps2
        r = r // ps2
        i1 = r % ps1
        i0 = r // ps1
        inp_row_start = i0 * pst0 + i1 * pst1 + i2 * pst2 + i3 * pst3

    val = tl.load(
        inp_ptr + inp_row_start[:, None] + cols[None, :],
        mask=rm[:, None] & cm[None, :],
        other=0.0,
    )

    out_off = rows[:, None] * N + cols[None, :]
    tl.store(out_ptr + out_off, val, mask=rm[:, None] & cm[None, :])


@libentry()
@triton.jit
def _contiguous_leading_slice_kernel(
    inp_ptr,
    out_ptr,
    outer: int,
    tail: int,
    inp_s0: int,
    BLOCK_OUTER: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
):
    """Copy tensors where only dim0 is strided and dims 1..end are contiguous."""
    pid_outer = tle.program_id(0)
    pid_tail = tle.program_id(1)

    outer_idx = pid_outer * BLOCK_OUTER + tl.arange(0, BLOCK_OUTER)
    tail_idx = pid_tail * BLOCK_TAIL + tl.arange(0, BLOCK_TAIL)
    om = outer_idx < outer
    tm = tail_idx < tail

    val = tl.load(
        inp_ptr + outer_idx[:, None] * inp_s0 + tail_idx[None, :],
        mask=om[:, None] & tm[None, :],
        other=0.0,
    )
    tl.store(
        out_ptr + outer_idx[:, None] * tail + tail_idx[None, :],
        val,
        mask=om[:, None] & tm[None, :],
    )


@libentry()
@triton.jit
def _contiguous_transpose_2d_kernel(
    inp_ptr,
    out_ptr,
    M: int,
    N: int,
    inp_s0: int,
    inp_s1: int,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Contiguous copy for a 2D transposed view with stride[0] == 1."""
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = rows < M
    cm = cols < N

    val = tl.load(
        inp_ptr + rows[:, None] * inp_s0 + cols[None, :] * inp_s1,
        mask=rm[:, None] & cm[None, :],
        other=0.0,
    )

    # `contiguous()` preserves the view's logical [M, N] order.  For a
    # transposed view, storing cols * M + rows would transpose a second time.
    tl.store(
        out_ptr + rows[:, None] * N + cols[None, :],
        val,
        mask=rm[:, None] & cm[None, :],
    )


def _is_row_copy_candidate(inp):
    return inp.ndim <= 2 and inp.stride(-1) == 1


def _is_leading_slice_candidate(inp):
    if inp.ndim <= 2 or inp.ndim > _MAX_RANK or inp.stride(-1) != 1:
        return False
    expected = 1
    for dim in range(inp.ndim - 1, 0, -1):
        if inp.stride(dim) != expected:
            return False
        expected *= inp.shape[dim]
    return inp.stride(0) >= expected


def _is_2d_transpose(inp):
    return inp.ndim == 2 and inp.stride(0) == 1 and inp.stride(1) != 1


def _pad_tuple(tup, max_rank, pad_val=0):
    if len(tup) > max_rank:
        return tuple(tup)
    return tuple(tup) + (pad_val,) * (max_rank - len(tup))


def contiguous(inp, memory_format=torch.contiguous_format):
    assert memory_format == torch.contiguous_format
    logger.debug("GEMS_TSINGMICRO CONTIGUOUS")

    if inp.is_contiguous(memory_format=memory_format):
        return inp

    out = torch.empty_like(inp, memory_format=memory_format)
    if inp.numel() == 0:
        return out

    if _is_row_copy_candidate(inp):
        rank = inp.ndim
        N = inp.shape[-1]
        M = inp.numel() // N

        prefix_shape = inp.shape[:-1]
        prefix_stride = inp.stride()[:-1]
        ps = _pad_tuple(prefix_shape, _MAX_RANK - 1, pad_val=1)
        pst = _pad_tuple(prefix_stride, _MAX_RANK - 1, pad_val=0)

        BLOCK_M = min(triton.next_power_of_2(M), 128)
        BLOCK_N = min(triton.next_power_of_2(N), 512)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        with torch_device_fn.device(inp.device):
            _contiguous_row_copy_kernel[grid](
                inp,
                out,
                M,
                N,
                rank,
                int(ps[0]),
                int(ps[1]),
                int(ps[2]),
                int(ps[3]),
                int(pst[0]),
                int(pst[1]),
                int(pst[2]),
                int(pst[3]),
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
        return out

    if _is_leading_slice_candidate(inp):
        outer = inp.shape[0]
        tail = inp.numel() // outer
        inp_s0 = inp.stride(0)

        BLOCK_OUTER = 1
        BLOCK_TAIL = min(triton.next_power_of_2(tail), 4096)
        grid = (triton.cdiv(outer, BLOCK_OUTER), triton.cdiv(tail, BLOCK_TAIL))

        with torch_device_fn.device(inp.device):
            _contiguous_leading_slice_kernel[grid](
                inp,
                out,
                outer,
                tail,
                inp_s0,
                BLOCK_OUTER=BLOCK_OUTER,
                BLOCK_TAIL=BLOCK_TAIL,
            )
        return out

    if _is_2d_transpose(inp):
        M, N = inp.shape
        inp_s0, inp_s1 = inp.stride()

        BLOCK_M = min(triton.next_power_of_2(M), 64)
        BLOCK_N = min(triton.next_power_of_2(N), 64)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        with torch_device_fn.device(inp.device):
            _contiguous_transpose_2d_kernel[grid](
                inp,
                out,
                M,
                N,
                inp_s0,
                inp_s1,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
        return out

    from flag_gems.ops.copy import copy

    return copy(inp, out0=out)
