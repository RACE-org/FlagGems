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
def reduce_all(a, b):
    return a and b


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def all_kernel_dim(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of inp it should compute.
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    _all = tl.full([BLOCK_M, BLOCK_N], value=1, dtype=tl.int1)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=1.0)
        _all = _all and (a != 0)
    all = tl.reduce(_all, axis=1, combine_fn=reduce_all)
    tl.store(out, all[:, None], row_mask)

@libentry()
@triton.jit
def all_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    # Single-pass per-CTA reduction: one BLOCK_SIZE chunk per program.
    # Stays in int1 (no int32 widening) — widening to int32 would inflate
    # the in-SPM working set by 32x and the subsequent reduce reads all of
    # it, which is what made the previous version slower than upstream.
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M
    inp_val = tl.load(inp + offset, mask=mask, other=1)
    all_val = tl.reduce(inp_val != 0, axis=0, combine_fn=reduce_all)
    tl.store(mid + pid, all_val)


@libentry()
@triton.jit
def all_kernel_full_tile(
    inp,
    mid,
    M,
    CHUNK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # One CTA per tile (grid = TOTAL_CORE_NUM). Each CTA owns a contiguous
    # CHUNK-sized slice of inp and reduces it in SPM via an inner BLOCK_SIZE
    # loop. This minimises DMA setup overhead (one tile, one big stream)
    # while keeping the per-iteration working set comfortably inside SPM.
    # Accumulator stays in int1 — no int32 widening.
    pid = tle.program_id(0)
    start = pid * CHUNK
    acc = tl.full([BLOCK_SIZE], 1, dtype=tl.int1)
    for off in range(0, CHUNK, BLOCK_SIZE):
        offset = start + off + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        v = tl.load(inp + offset, mask=mask, other=1)
        acc = acc & (v != 0)
    all_val = tl.reduce(acc, axis=0, combine_fn=reduce_all)
    tl.store(mid + pid, all_val)


@libentry()
@triton.jit
def all_kernel_2(
    mid,
    out,
    MID_SIZE,
    BLOCK_MID: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_SIZE
    mid_val = tl.load(mid + offset, mask=mask, other=1).to(tl.int1)
    all_val = tl.reduce(mid_val, axis=0, combine_fn=reduce_all)
    tl.store(out, all_val)


def _select_all_block_size(curr_size):
    # Aim for grid ≈ 16 × small_factor on a 16-tile NPU: one or a few CTAs
    # per tile maximises DMA utilisation while keeping per-CTA working set
    # comfortably inside the 3MB SPM.
    # block ≈ sqrt(M) is the upstream heuristic; cap at 32768 to stay well
    # under SPM, floor at 1024 for small M so we don't launch too few CTAs.
    if curr_size >= 1048576:
        return 32768
    if curr_size >= 262144:
        return 8192
    if curr_size >= 16384:
        return 2048
    return 1024


# BLOCK_SIZE used inside the full-tile inner loop. 64K×i16 = 128KB per tile,
# still well under 3MB SPM, but halves the inner-loop trip count vs 32K.
_FULL_TILE_BLOCK = 65536

# When M ≥ this many elements, switch to the 16-CTA full-tile kernel.
# Threshold: each tile needs at least 1 full inner iteration of work to
# amortise the kernel launch / DMA setup cost (≈ TOTAL_CORE_NUM * BLOCK).
_FULL_TILE_THRESHOLD = TOTAL_CORE_NUM * _FULL_TILE_BLOCK  # ≈ 1M elements


def all(inp):
    logger.debug("GEMS_TSINGMICRO ALL")
    device = inp.device
    M = inp.numel()

    with torch_device_fn.device(device):
        if M >= _FULL_TILE_THRESHOLD:
            # Large-M path: 16 CTAs, each chews through M/16 elements in SPM,
            # producing 16 partials. Single stage-1 launch keeps DMA setup
            # cost amortised over the whole input.
            chunk = triton.cdiv(M, TOTAL_CORE_NUM)
            # Round chunk up to a multiple of BLOCK_SIZE so the inner loop
            # doesn't end on a partial block.
            chunk = triton.cdiv(chunk, _FULL_TILE_BLOCK) * _FULL_TILE_BLOCK
            mid = torch.empty((TOTAL_CORE_NUM,), dtype=torch.bool, device=device)
            all_kernel_full_tile[(TOTAL_CORE_NUM,)](
                inp, mid, M, chunk, _FULL_TILE_BLOCK
            )
            curr = mid
            curr_size = TOTAL_CORE_NUM
        else:
            # Small/medium-M path: classic sum-style multi-stage. Keeps the
            # per-CTA working set small so latency stays low when M is too
            # small to feed 16 fully-loaded tiles.
            curr = inp
            curr_size = M
            while curr_size > 4096:
                block_size = _select_all_block_size(curr_size)
                mid_size = triton.cdiv(curr_size, block_size)
                mid = torch.empty((mid_size,), dtype=torch.bool, device=device)
                all_kernel_1[(mid_size,)](curr, mid, curr_size, block_size)
                curr = mid
                curr_size = mid_size

        out = torch.empty([], dtype=torch.bool, device=device)
        block_mid = triton.next_power_of_2(curr_size)
        all_kernel_2[(1,)](curr, out, curr_size, block_mid)

    return out


def all_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_TSINGMICRO ALL DIM")
    shape = list(inp.shape)
    if dim is None:
        out = all(inp)
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
            all_kernel_dim[grid](inp, out, M, N)
        if not keepdim:
            out = out.squeeze(dim=dim)
    return out


def all_dims(inp, dim=None, keepdim=False):
    logger.debug("GEMS_TSINGMICRO ALL DIMS")

    if dim is None or isinstance(dim, int):
        return all_dim(inp, dim=dim, keepdim=keepdim)
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
        all_kernel_dim[grid](inp, out, M, N)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
