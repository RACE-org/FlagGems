"""
SPACEMIT-specific dropout implementation with improved precision.
"""
import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)
from flag_gems.runtime import torch_device_fn


def heur_block(args):
    if args["N"] <= 512:
        return 512
    else:
        return 1024


def heur_num_warps(args):
    if args["N"] <= 512:
        return 4
    elif args["N"] <= 1024:
        return 8
    else:
        return 16


@triton.heuristics(
    {
        "BLOCK": heur_block,
        "num_warps": heur_num_warps,
    }
)
@triton.jit(do_not_specialize=["p", "philox_seed", "philox_offset"])
def dropout_forward_kernel(
    X,
    Y,
    N,
    p,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    UNROLL: tl.constexpr = 4  # philox generate 128 random bits at a time
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    r0 = uint_to_uniform_float(r0)
    r1 = uint_to_uniform_float(r1)
    r2 = uint_to_uniform_float(r2)
    r3 = uint_to_uniform_float(r3)

    mask0 = r0 > p
    mask1 = r1 > p
    mask2 = r2 > p
    mask3 = r3 > p

    # SPACEMIT FIX: Compute scale in float32 for better precision
    scale = (1.0 / (1.0 - p)).to(tl.float32)

    off_0 = tl.program_id(0) * BLOCK * UNROLL + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK

    x0 = tl.load(X + off_0, mask=off_0 < N, other=0.0, eviction_policy="evict_first")
    x1 = tl.load(X + off_1, mask=off_1 < N, other=0.0, eviction_policy="evict_first")
    x2 = tl.load(X + off_2, mask=off_2 < N, other=0.0, eviction_policy="evict_first")
    x3 = tl.load(X + off_3, mask=off_3 < N, other=0.0, eviction_policy="evict_first")

    # Convert to float32 for computation, then back to original dtype
    y0 = (x0.to(tl.float32) * scale * mask0).to(x0.dtype)
    y1 = (x1.to(tl.float32) * scale * mask1).to(x1.dtype)
    y2 = (x2.to(tl.float32) * scale * mask2).to(x2.dtype)
    y3 = (x3.to(tl.float32) * scale * mask3).to(x3.dtype)

    tl.store(Y + off_0, y0, mask=off_0 < N, eviction_policy="evict_first")
    tl.store(Y + off_1, y1, mask=off_1 < N, eviction_policy="evict_first")
    tl.store(Y + off_2, y2, mask=off_2 < N, eviction_policy="evict_first")
    tl.store(Y + off_3, y3, mask=off_3 < N, eviction_policy="evict_first")


@triton.heuristics(
    {
        "BLOCK": heur_block,
        "num_warps": heur_num_warps,
    }
)
@triton.jit(do_not_specialize=["p", "philox_seed", "philox_offset"])
def dropout_backward_kernel(
    DY,
    DX,
    N,
    p,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    UNROLL: tl.constexpr = 4
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    r0 = uint_to_uniform_float(r0)
    r1 = uint_to_uniform_float(r1)
    r2 = uint_to_uniform_float(r2)
    r3 = uint_to_uniform_float(r3)

    mask0 = r0 > p
    mask1 = r1 > p
    mask2 = r2 > p
    mask3 = r3 > p

    # SPACEMIT FIX: Compute scale in float32 for better precision
    scale = (1.0 / (1.0 - p)).to(tl.float32)

    off_0 = tl.program_id(0) * BLOCK * UNROLL + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK

    dy0 = tl.load(DY + off_0, mask=off_0 < N, other=0.0, eviction_policy="evict_first")
    dy1 = tl.load(DY + off_1, mask=off_1 < N, other=0.0, eviction_policy="evict_first")
    dy2 = tl.load(DY + off_2, mask=off_2 < N, other=0.0, eviction_policy="evict_first")
    dy3 = tl.load(DY + off_3, mask=off_3 < N, other=0.0, eviction_policy="evict_first")

    # Convert to float32 for computation, then back to original dtype
    dx0 = (dy0.to(tl.float32) * scale * mask0).to(dy0.dtype)
    dx1 = (dy1.to(tl.float32) * scale * mask1).to(dy1.dtype)
    dx2 = (dy2.to(tl.float32) * scale * mask2).to(dy2.dtype)
    dx3 = (dy3.to(tl.float32) * scale * mask3).to(dy3.dtype)

    tl.store(DX + off_0, dx0, mask=off_0 < N, eviction_policy="evict_first")
    tl.store(DX + off_1, dx1, mask=off_1 < N, eviction_policy="evict_first")
    tl.store(DX + off_2, dx2, mask=off_2 < N, eviction_policy="evict_first")
    tl.store(DX + off_3, dx3, mask=off_3 < N, eviction_policy="evict_first")


class Dropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, training):
        if not training or p == 0:
            return x
        if p == 1:
            return torch.zeros_like(x)

        N = x.numel()
        philox_seed, philox_offset = philox_backend_seed_offset(N)
        y = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK"] * 4),)
        with torch_device_fn.device(x.device):
            dropout_forward_kernel[grid](x, y, N, p, philox_seed, philox_offset)
        ctx.p = p
        ctx.philox_seed = philox_seed
        ctx.philox_offset = philox_offset
        return y

    @staticmethod
    def backward(ctx, dy):
        N = dy.numel()
        dx = torch.empty_like(dy)
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK"] * 4),)
        with torch_device_fn.device(dy.device):
            dropout_backward_kernel[grid](
                dy, dx, N, ctx.p, ctx.philox_seed, ctx.philox_offset
            )
        return dx, None, None


def dropout(x, p=0.5, training=True, inplace=False):
    """
    SPACEMIT-specific dropout with improved float16 precision.
    """
    if inplace:
        logging.warning("SPACEMIT backend: dropout inplace not supported, using out-of-place")
    return Dropout.apply(x, p, training)
