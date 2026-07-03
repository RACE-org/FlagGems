import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# ===========================================================================
# Tx81 tile optimisation
#
# Upstream codegen kernel does per-output-element flat-index decomposition
# (RANK × % + RANK × //) plus input wrapping (RANK × %).  On Tx81 each int
# % and // triggers the corrective chain.
#
# 1. Last-dim-only repeat → 2D block copy.  Load input row once, store R
#    times contiguously.  Zero % or //.
#
# 2. Rank-2 general → 2D grid over output.  program_id gives coords
#    directly, only % for input wrapping (2 % vs upstream 4 % + 2 //).
#
# 3. Rank 3-5 → flat-index decomposition + % wrapping, RANK: constexpr,
#    int32 throughout.  Same algorithm as upstream but no codegen.
#
# All arithmetic is int32.  int64 is never introduced.
# ===========================================================================

_MAX_RANK = 5


# ===========================================================================
# Kernel 1 — Last-dim-only repeat: block copy, zero %//
# ===========================================================================


@libentry()
@triton.jit
def _tile_lastdim_kernel(
    inp_ptr: tl.tensor,
    out_ptr: tl.tensor,
    M: int,
    N: int,
    R: int,
    inp_row_stride: int,
    out_row_stride: int,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Input [M, N], output [M, R*N].  Block copy: load once, store R times.

    All prefix dims flattened into M.  N is the last input dim.
    R is the last repeat factor.  Zero % or //.
    """
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = rows < M
    cm = cols < N

    val = tl.load(
        inp_ptr + rows[:, None] * inp_row_stride + cols[None, :],
        mask=rm[:, None] & cm[None, :],
        other=0.0,
    )

    # Store the block R times, each at column offset r*N.
    for r in range(R):
        out_cols = r * N + cols[None, :]
        tl.store(
            out_ptr + rows[:, None] * out_row_stride + out_cols,
            val,
            mask=rm[:, None] & cm[None, :],
        )


# ===========================================================================
# Kernel 2 — Rank-2 general: 2D grid, % wrapping only, no flat-index decomp
# ===========================================================================


@libentry()
@triton.jit
def _tile_2d_kernel(
    inp_ptr: tl.tensor,
    out_ptr: tl.tensor,
    M: int,  # = R0 * A  (output rows)
    N: int,  # = R1 * B  (output cols)
    A: int,  # input rows
    B: int,  # input cols
    inp_stride0: int,
    inp_stride1: int,
    out_stride0: int,
    out_stride1: int,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Rank-2 tile with 2D grid over output.  program_id gives output coords
    directly — zero flat-index decomposition.  Only 2 × % for input wrapping."""
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = rows < M
    cm = cols < N

    inp_rows = rows % A
    inp_cols = cols % B

    val = tl.load(
        inp_ptr + inp_rows[:, None] * inp_stride0 + inp_cols[None, :] * inp_stride1,
        mask=rm[:, None] & cm[None, :],
        other=0.0,
    )
    tl.store(
        out_ptr + rows[:, None] * out_stride0 + cols[None, :] * out_stride1,
        val,
        mask=rm[:, None] & cm[None, :],
    )


# ===========================================================================
# Kernel 3 — Rank 3-5 general: flat-index decomposition + % wrapping
# ===========================================================================


@libentry()
@triton.jit
def _tile_nd_kernel(
    inp_ptr: tl.tensor,
    out_ptr: tl.tensor,
    num_tasks: int,
    RANK: tl.constexpr,
    # output shapes
    os0: int, os1: int, os2: int, os3: int, os4: int,
    # input shapes  (for % wrapping)
    is0: int, is1: int, is2: int, is3: int, is4: int,
    # input strides
    irst0: int, irst1: int, irst2: int, irst3: int, irst4: int,
    # output strides
    oust0: int, oust1: int, oust2: int, oust3: int, oust4: int,
    BLOCK_SIZE: tl.constexpr,
):
    """Rank 3-5 tile.  Flat-index decomposition + input wrapping via %.
    RANK: constexpr → only matching branch compiles, exact op count per rank."""

    pid = tle.program_id(0)
    ctas = tle.num_programs(0)

    for j in range(tl.cdiv(tl.cdiv(num_tasks, BLOCK_SIZE), ctas)):
        block_id = pid + j * ctas
        off = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = off < num_tasks

        if RANK == 3:
            cur = off
            d2 = cur % os2; cur = cur // os2
            d1 = cur % os1; cur = cur // os1
            d0 = cur
            inp_off = (
                (d0 % is0) * irst0
                + (d1 % is1) * irst1
                + (d2 % is2) * irst2
            )
            out_off = d0 * oust0 + d1 * oust1 + d2 * oust2

        elif RANK == 4:
            cur = off
            d3 = cur % os3; cur = cur // os3
            d2 = cur % os2; cur = cur // os2
            d1 = cur % os1; cur = cur // os1
            d0 = cur
            inp_off = (
                (d0 % is0) * irst0
                + (d1 % is1) * irst1
                + (d2 % is2) * irst2
                + (d3 % is3) * irst3
            )
            out_off = d0 * oust0 + d1 * oust1 + d2 * oust2 + d3 * oust3

        elif RANK == 5:
            cur = off
            d4 = cur % os4; cur = cur // os4
            d3 = cur % os3; cur = cur // os3
            d2 = cur % os2; cur = cur // os2
            d1 = cur % os1; cur = cur // os1
            d0 = cur
            inp_off = (
                (d0 % is0) * irst0
                + (d1 % is1) * irst1
                + (d2 % is2) * irst2
                + (d3 % is3) * irst3
                + (d4 % is4) * irst4
            )
            out_off = (
                d0 * oust0 + d1 * oust1 + d2 * oust2 + d3 * oust3 + d4 * oust4
            )

        val = tl.load(inp_ptr + inp_off, mask=mask, other=0.0)
        tl.store(out_ptr + out_off, val, mask=mask)


# ===========================================================================
# Host helpers
# ===========================================================================


def _pad_tuple(tup, max_rank, pad_val=0):
    """Trailing-pad to max_rank.  pad_val=1 for shapes, 0 for strides."""
    n = len(tup)
    if n >= max_rank:
        return tuple(tup)
    return tuple(tup) + (pad_val,) * (max_rank - n)


def _only_lastdim_greater_1(dims):
    """True when only the last dim has repeat > 1."""
    for d in dims[:-1]:
        if d != 1:
            return False
    return len(dims) > 0 and dims[-1] > 1


def _all_ones(dims):
    return all(d == 1 for d in dims)


def _adaptive_2d_blocks(M, N, max_block_m=256, max_block_n=512):
    """Pick BLOCK_M, BLOCK_N so the 2D grid has at least ~16 CTAs for small
    problems (DMA pipelining across tiles), but stays within max_block limits
    for large problems (avoid oversubscription)."""
    total = M * N
    # Target ~16 CTAs; floor at 1024 elems/CTA to keep per-CTA work non-trivial.
    target_per_cta = max(1024, total // 16)
    target_per_dim = max(16, int(target_per_cta ** 0.5))
    bm = min(triton.next_power_of_2(target_per_dim), max_block_m)
    bn = min(triton.next_power_of_2(target_per_dim), max_block_n)
    return max(16, bm), max(16, bn)


# ===========================================================================
# Entry point
# ===========================================================================


def tile(inp: torch.Tensor, dims) -> torch.Tensor:
    logger.debug("GEMS TSINGMICRO TILE")

    inp_ndim = inp.ndim
    dims_ndim = len(dims)

    # ---- align ranks (mirrors upstream) ----
    if dims_ndim < inp_ndim:
        dims = (1,) * (inp_ndim - dims_ndim) + tuple(dims)
    elif dims_ndim > inp_ndim:
        inp = inp.reshape((1,) * (dims_ndim - inp_ndim) + inp.shape)

    rank = len(dims)
    assert all(d >= 0 for d in dims), (
        f"repeat dims must be >= 0, got {dims}"
    )

    # ---- any dim 0 → empty output ----
    if any(d == 0 for d in dims):
        out_shape = [inp.shape[i] * dims[i] for i in range(rank)]
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # ---- all 1 → clone ----
    if _all_ones(dims):
        return inp.clone()

    # ---- rank too high → not supported ----
    assert rank <= _MAX_RANK, f"Tx81 tile supports rank ≤ {_MAX_RANK}, got {rank}"

    with torch_device_fn.device(inp.device):
        out_shape = [inp.shape[i] * dims[i] for i in range(rank)]
        out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

        # =============================================================
        # Fast path: only last dim repeats → block copy, zero %//
        # =============================================================
        if _only_lastdim_greater_1(dims) and inp.is_contiguous():
            N = inp.shape[-1]
            R = dims[-1]
            M = inp.numel() // N
            inp_row_stride = N
            out_row_stride = out.shape[-1]

            BLOCK_M, BLOCK_N = _adaptive_2d_blocks(M, N)
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

            _tile_lastdim_kernel[grid](
                inp.reshape(M, N), out.reshape(M, out_row_stride),
                M, N, R, inp_row_stride, out_row_stride,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            )
            return out

        # =============================================================
        # Rank 2 → 2D grid, % wrapping only, no flat-index decomp
        # =============================================================
        if rank == 2:
            A, B = inp.shape
            M, N = out.shape
            inp_stride0, inp_stride1 = inp.stride()
            out_stride0, out_stride1 = out.stride()

            BLOCK_M, BLOCK_N = _adaptive_2d_blocks(M, N)
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

            _tile_2d_kernel[grid](
                inp, out,
                M, N, A, B,
                inp_stride0, inp_stride1,
                out_stride0, out_stride1,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            )
            return out

        # =============================================================
        # Rank 3-5 → flat-index decomposition + % wrapping
        # =============================================================
        num_tasks = out.numel()

        osh = _pad_tuple(out.shape, _MAX_RANK, pad_val=1)
        ish = _pad_tuple(inp.shape, _MAX_RANK, pad_val=1)
        ist = _pad_tuple(inp.stride(), _MAX_RANK, pad_val=0)
        ost = _pad_tuple(out.stride(), _MAX_RANK, pad_val=0)

        BLOCK_SIZE = min(512, triton.next_power_of_2(num_tasks))
        grid = (min(16, triton.cdiv(num_tasks, BLOCK_SIZE)),)

        _tile_nd_kernel[grid](
            inp, out, num_tasks, rank,
            int(osh[0]), int(osh[1]), int(osh[2]), int(osh[3]), int(osh[4]),
            int(ish[0]), int(ish[1]), int(ish[2]), int(ish[3]), int(ish[4]),
            int(ist[0]), int(ist[1]), int(ist[2]), int(ist[3]), int(ist[4]),
            int(ost[0]), int(ost[1]), int(ost[2]), int(ost[3]), int(ost[4]),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out
