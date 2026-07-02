import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.count_nonzero import count_nonzero as _generic_count_nonzero
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry

logger = logging.getLogger(__name__)

TILE_GRID = 16
BLOCK_SIZE = 4096
MAX_BLOCK_ROWS = 64
FP32_EXACT_INT = 1 << 24


@libentry()
@triton.jit
def _count_tiles_kernel(
    x_ptr,
    partials_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    acc = tl.full((), 0.0, tl.float32)

    # One CTA per tile.  Keep all hot-path count values in fp32: Tx81 handles
    # fp32 SIMD reductions well, while vector i64 materialization lowers to
    # scalar RISC-V loops.  The Python dispatcher only uses this kernel when
    # each tile's count is <= 2**24, so the fp32 partial is exact.
    for base in tl.range(pid * BLOCK, n_elements, tl.num_programs(0) * BLOCK):
        offsets = base + lane
        mask = offsets < n_elements
        vals = tl.load(x_ptr + offsets, mask=mask, other=0)
        acc += tl.sum((vals != 0).to(tl.float32), axis=0)

    tl.store(partials_ptr + pid, acc)


@libentry()
@triton.jit
def _count_row_blocks_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    reduce_n,
    n_row_blocks,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row_lane = tl.arange(0, BLOCK_M)
    col_lane = tl.arange(0, BLOCK_N)

    # Stride row blocks across the 16-tile grid.  A CTA reduces a small
    # row/column tile, which avoids a scalar row loop when the reduced dim is
    # short.  Counts stay fp32 until the final API-mandated int64 output store.
    for row_block in tl.range(pid, n_row_blocks, tl.num_programs(0)):
        rows = row_block * BLOCK_M + row_lane
        acc = tl.full((BLOCK_M,), 0.0, tl.float32)
        for start in tl.range(0, reduce_n, BLOCK_N):
            cols = start + col_lane
            offsets = rows[:, None] * reduce_n + cols[None, :]
            mask = (rows[:, None] < n_rows) & (cols[None, :] < reduce_n)
            vals = tl.load(x_ptr + offsets, mask=mask, other=0)
            acc += tl.sum((vals != 0).to(tl.float32), axis=1)
        tl.store(out_ptr + rows, acc.to(tl.int64), mask=rows < n_rows)


@libentry()
@triton.jit
def _count_blocks_kernel(
    x_ptr,
    counts_ptr,
    n_elements,
    n_blocks,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)

    # Fallback for very large tensors: store exact fp32 block counts.  Each
    # block count is <= BLOCK, so host-side integer accumulation stays exact
    # without creating device-side i64 vectors.
    for block_id in tl.range(pid, n_blocks, tl.num_programs(0)):
        offsets = block_id * BLOCK + lane
        mask = offsets < n_elements
        vals = tl.load(x_ptr + offsets, mask=mask, other=0)
        count_f32 = tl.sum((vals != 0).to(tl.float32), axis=0)
        tl.store(counts_ptr + block_id, count_f32)


def _count_nonzero_flat(x):
    x = x.contiguous().flatten()
    n_elements = x.numel()
    if n_elements == 0:
        return torch.tensor(0, dtype=torch.int64, device=x.device)

    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    grid = min(TILE_GRID, n_blocks)

    if n_elements <= TILE_GRID * FP32_EXACT_INT:
        partials = torch.empty((grid,), dtype=torch.float32, device=x.device)
        with torch_device_fn.device(x.device):
            _count_tiles_kernel[(grid,)](
                x,
                partials,
                n_elements,
                BLOCK=BLOCK_SIZE,
                num_warps=8,
            )
        total = sum(int(v) for v in partials.cpu().tolist())
    else:
        counts = torch.empty((n_blocks,), dtype=torch.float32, device=x.device)
        with torch_device_fn.device(x.device):
            _count_blocks_kernel[(grid,)](
                x,
                counts,
                n_elements,
                n_blocks,
                BLOCK=BLOCK_SIZE,
                num_warps=8,
            )
        total = sum(int(v) for v in counts.cpu().tolist())

    # PyTorch's public result dtype is int64, but Tx81 only sees fp32 partials.
    # The single scalar int64 is created after the reduction, outside the SIMD
    # hot path that used to materialize memref<...xi64> buffers.
    return torch.tensor(total, dtype=torch.int64, device=x.device)


def _count_nonzero_dim(x, dim):
    if not isinstance(dim, int) or x.ndim == 0:
        return _generic_count_nonzero(x, dim)

    dim = dim % x.ndim
    reduce_n = x.shape[dim]
    out_shape = list(x.shape)
    del out_shape[dim]

    if reduce_n == 0:
        return torch.zeros(out_shape, dtype=torch.int64, device=x.device)
    if reduce_n > FP32_EXACT_INT:
        return _generic_count_nonzero(x, dim)

    x = dim_compress(x, dim).flatten()
    n_rows = x.numel() // reduce_n
    out = torch.empty(out_shape, dtype=torch.int64, device=x.device)
    if n_rows == 0:
        return out

    block_n = min(BLOCK_SIZE, triton.next_power_of_2(reduce_n))
    block_m = min(MAX_BLOCK_ROWS, max(1, BLOCK_SIZE // block_n))
    n_row_blocks = triton.cdiv(n_rows, block_m)

    with torch_device_fn.device(x.device):
        _count_row_blocks_kernel[(min(TILE_GRID, n_row_blocks),)](
            x,
            out,
            n_rows,
            reduce_n,
            n_row_blocks,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=8,
        )
    return out


def count_nonzero(x, dim=None):
    logger.debug("GEMS_TSINGMICRO COUNT_NONZERO")
    if x.is_complex():
        return _generic_count_nonzero(x, dim)
    if dim is None:
        return _count_nonzero_flat(x)
    return _count_nonzero_dim(x, dim)
