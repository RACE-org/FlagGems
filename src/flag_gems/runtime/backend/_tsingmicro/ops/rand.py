import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.runtime.backend._tsingmicro.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger(__name__)
device_ = device

# Tx81 has exactly 16 tiles. Pin grid=(16,) so one CTA maps to one tile.
NUM_TILES = 16
UNROLL = 4


# BLOCK sized so 16 * UNROLL * BLOCK ≈ N — i.e. one launch wave spreads work
# across all 16 tiles rather than letting tail tiles mask out completely.
# Min BLOCK = 128 keeps each tx.wdma at least 512 bytes (fp32). Cap at 8192
# (16384 produced corrupted output in earlier testing).
def _rand_heur_block(args):
    n = args["N"]
    if n <= 64 * 128:
        return 128
    if n <= 64 * 256:
        return 256
    if n <= 64 * 512:
        return 512
    if n <= 64 * 1024:
        return 1024
    if n <= 64 * 4096:
        return 4096
    return 8192


# SINGLE_ITER == True ↔ one launch wave covers all of N (iters would be 1).
# Used as a compile-time switch to bypass the scf.for + i32 overflow guards
# that Triton emits for the runtime-loop path. Holds for any N ≤ 64 * 8192.
def _rand_heur_single_iter(args):
    block = _rand_heur_block(args)
    return triton.cdiv(args["N"], block * UNROLL * NUM_TILES) == 1


def _rand_heur_num_warps(args):
    n = args["N"]
    if n <= 4096:
        return 4
    if n <= 32768:
        return 8
    return 16


@triton.jit
def _rand_one_iter(out_ptr, N, philox_seed, c0_base, c1, launch_idx, BLOCK: tl.constexpr):
    i4 = launch_idx * BLOCK + tl.arange(0, BLOCK)
    c0 = c0_base + i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    r0 = uint_to_uniform_float(r0)
    r1 = uint_to_uniform_float(r1)
    r2 = uint_to_uniform_float(r2)
    r3 = uint_to_uniform_float(r3)
    off_0 = launch_idx * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    tl.store(out_ptr + off_0, r0, mask=off_0 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_1, r1, mask=off_1 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_2, r2, mask=off_2 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_3, r3, mask=off_3 < N, eviction_policy="evict_first")


@triton.heuristics({
    "BLOCK": _rand_heur_block,
    "SINGLE_ITER": _rand_heur_single_iter,
    "num_warps": _rand_heur_num_warps,
})
@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def rand_kernel(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
    SINGLE_ITER: tl.constexpr,
):
    NUM_TILES: tl.constexpr = 16
    UNROLL: tl.constexpr = 4

    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0_base = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    pid = tl.program_id(0)

    if SINGLE_ITER:
        # Compile-time-eliminated branch when iters>1; pure inline path
        # when iters==1: no scf.for, no pid*iters arithmetic, no i32
        # overflow guards. launch_idx == pid.
        _rand_one_iter(out_ptr, N, philox_seed, c0_base, c1, pid, BLOCK)
    else:
        iters = tl.cdiv(N, BLOCK * UNROLL * NUM_TILES)
        for j in range(iters):
            launch_idx = pid * iters + j
            _rand_one_iter(out_ptr, N, philox_seed, c0_base, c1, launch_idx, BLOCK)


def rand(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS RAND")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)

    out = torch.empty(size, device=device, dtype=dtype)
    N = volume(size)
    # (TODO) Using Triton autotuner makes kernel parameters opaque to the caller,
    # hence we cannot obtain the per thread offset as in Pytorch.
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(increment)
    grid = (NUM_TILES,)
    with torch_device_fn.device(device):
        rand_kernel[grid](out, N, philox_seed, philox_offset)
    return out
