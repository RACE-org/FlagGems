import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

TOTAL_CORE_NUM = 16

logger = logging.getLogger(__name__)


@triton.jit
def reduce_any(a, b):
    return a or b


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def any_kernel_dim(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    _any = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int1)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=0.0)
        _any = _any or (a != 0)
    any_val = tl.reduce(_any, axis=1, combine_fn=reduce_any)
    tl.store(out, any_val[:, None], row_mask)


@libentry()
@triton.jit
def any_kernel_full_tile(
    inp,
    mid,
    M,
    CHUNK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # One CTA per tile (grid = TOTAL_CORE_NUM). Each CTA chews through a
    # contiguous CHUNK-sized slice via an inner BLOCK_SIZE loop. Single
    # launch amortises DMA setup / kernel launch overhead.
    pid = tle.program_id(0)
    start = pid * CHUNK
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.int1)
    for off in range(0, CHUNK, BLOCK_SIZE):
        offset = start + off + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        v = tl.load(inp + offset, mask=mask, other=0)
        acc = acc | (v != 0)
    any_val = tl.reduce(acc, axis=0, combine_fn=reduce_any)
    tl.store(mid + pid, any_val)


@libentry()
@triton.jit
def any_kernel_2(
    mid,
    out,
    MID_SIZE,
    BLOCK_MID: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_SIZE
    mid_val = tl.load(mid + offset, mask=mask, other=0).to(tl.int1)
    any_val = tl.reduce(mid_val, axis=0, combine_fn=reduce_any)
    tl.store(out, any_val)


# Inner-loop BLOCK cap. 64K i1 = 64KB per tile, well under 3MB SPM.
_FULL_TILE_BLOCK = 65536

# Smallest per-tile work that justifies a 16-CTA full-tile launch. Below
# this, launch overhead dominates and one CTA is faster.
_MIN_PER_TILE = 64


def _pick_block_size(per_tile):
    # Largest power of 2 ≤ per_tile, capped at _FULL_TILE_BLOCK. Keeps the
    # per-CTA SPM working set proportional to actual work and avoids
    # over-allocating a fixed 64K acc tile when M is small.
    block = 1
    while block * 2 <= per_tile and block * 2 <= _FULL_TILE_BLOCK:
        block *= 2
    return max(block, _MIN_PER_TILE)


def any(inp):
    logger.debug("GEMS_TSINGMICRO ANY")
    device = inp.device
    M = inp.numel()

    with torch_device_fn.device(device):
        per_tile = triton.cdiv(M, TOTAL_CORE_NUM)
        if per_tile >= _MIN_PER_TILE:
            # Always spread across 16 tiles when each tile has enough work
            # to justify the launch. BLOCK adapts to M so we don't waste
            # SPM on a fixed 64K accumulator when only a few KB is needed.
            block_size = _pick_block_size(per_tile)
            chunk = triton.cdiv(per_tile, block_size) * block_size
            mid = torch.empty((TOTAL_CORE_NUM,), dtype=torch.bool, device=device)
            any_kernel_full_tile[(TOTAL_CORE_NUM,)](
                inp, mid, M, chunk, block_size
            )
            curr = mid
            curr_size = TOTAL_CORE_NUM
        else:
            # Tiny M: full-tile split would launch 16 CTAs with almost no
            # work each. Skip stage-1 entirely.
            curr = inp
            curr_size = M

        out = torch.empty([], dtype=torch.bool, device=device)
        block_mid = triton.next_power_of_2(curr_size)
        any_kernel_2[(1,)](curr, out, curr_size, block_mid)

    return out


def any_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_TSINGMICRO ANY DIM")
    shape = list(inp.shape)
    if dim is None:
        out = any(inp)
        if keepdim:
            out = torch.reshape(out, [1] * inp.ndim)
    else:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        dim = dim % inp.ndim
        inp = dim_compress(inp, dim)
        N = shape[dim]
        shape[dim] = 1
        M = inp.numel() // N

        out = torch.empty(shape, dtype=torch.bool, device=inp.device)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch_device_fn.device(inp.device):
            any_kernel_dim[grid](inp, out, M, N)
        if not keepdim:
            out = out.squeeze(dim=dim)
    return out


def any_dims(inp, dim=None, keepdim=False):
    logger.debug("GEMS_TSINGMICRO ANY DIMS")

    if dim is None or isinstance(dim, int):
        return any_dim(inp, dim=dim, keepdim=keepdim)
    assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    inp = dim_compress(inp, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    out = torch.empty(shape, dtype=torch.bool, device=inp.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    with torch_device_fn.device(inp.device):
        any_kernel_dim[grid](inp, out, M, N)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
