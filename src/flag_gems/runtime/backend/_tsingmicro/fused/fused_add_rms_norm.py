import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.fused.fused_add_rms_norm import (
    fused_add_rms_norm as _generic_fused_add_rms_norm,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_TILE_GRID = 16
_SMALL_N_FALLBACK = 16
_SINGLE_PASS_MAX_N = 8192
_GENERIC_WIDE_N_MAX = 65536
_MIN_TILE_BLOCK = 1024
_MAX_TILE_BLOCK = 65536
_EXACT_N_FASTPATHS = (16384, 32768, 65536)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "eps"])
def _fused_add_rms_norm_single_pass_kernel(
    x_ptr,
    r_ptr,
    w_ptr,
    M: int,
    N: int,
    eps,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N

    for row in tl.range(pid, M, tl.num_programs(0)):
        row_base = row * N
        x = tl.load(x_ptr + row_base + cols, mask=col_mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + row_base + cols, mask=col_mask, other=0.0).to(tl.float32)

        xr = x + r
        tl.store(r_ptr + row_base + cols, xr, mask=col_mask)

        var = tl.sum(xr * xr * (1.0 / N), axis=0)
        rrms = 1.0 / tl.sqrt(var + eps)

        w = tl.load(w_ptr + cols, mask=col_mask, other=0.0)
        y = (xr * rrms).to(x_ptr.dtype.element_ty) * w
        tl.store(x_ptr + row_base + cols, y, mask=col_mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "eps"])
def _fused_add_rms_norm_tiled_kernel(
    x_ptr,
    r_ptr,
    w_ptr,
    M: int,
    N: int,
    eps,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    lane = tl.arange(0, BLOCK_N)

    for row in tl.range(pid, M, tl.num_programs(0)):
        row_base = row * N
        var = tl.full((), 0.0, tl.float32)
        inv_n = 1.0 / N

        # Pass 1: add residual, write it back once, and accumulate rms sum.
        for col_start in tl.range(0, N, BLOCK_N):
            cols = col_start + lane
            mask = cols < N
            x = tl.load(x_ptr + row_base + cols, mask=mask, other=0.0).to(tl.float32)
            r = tl.load(r_ptr + row_base + cols, mask=mask, other=0.0).to(tl.float32)
            xr = x + r
            tl.store(r_ptr + row_base + cols, xr, mask=mask)
            var += tl.sum(xr * xr * inv_n, axis=0)

        rrms = 1.0 / tl.sqrt(var + eps)

        # Pass 2: residual already stores x + r, so avoid reloading x and
        # redoing the addition for the normalization writeback.
        for col_start in tl.range(0, N, BLOCK_N):
            cols = col_start + lane
            mask = cols < N
            xr = tl.load(r_ptr + row_base + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + cols, mask=mask, other=0.0)
            y = (xr * rrms).to(x_ptr.dtype.element_ty) * w
            tl.store(x_ptr + row_base + cols, y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def _fused_add_rms_norm_exact_n_kernel(
    x_ptr,
    r_ptr,
    w_ptr,
    eps,
    BLOCK_N: tl.constexpr,
):
    """Original-like wide-row path for exact power-of-two hidden sizes.

    Tx81 handles long contiguous rows well when each CTA owns one full row.
    For N == BLOCK_N, all lanes are valid, so avoid masked memory ops while
    keeping the original single-pass reduction shape.
    """
    row = tle.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    row_base = row * BLOCK_N

    x = tl.load(x_ptr + row_base + cols).to(tl.float32)
    r = tl.load(r_ptr + row_base + cols).to(tl.float32)
    xr = x + r
    tl.store(r_ptr + row_base + cols, xr)

    var = tl.sum(xr * xr * (1.0 / BLOCK_N), axis=0)
    rrms = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + cols)
    y = (xr * rrms).to(x_ptr.dtype.element_ty) * w
    tl.store(x_ptr + row_base + cols, y)


def _pick_single_pass_block(n_cols):
    return min(_SINGLE_PASS_MAX_N, triton.next_power_of_2(n_cols))


def _pick_tiled_block(n_cols):
    # This path is only for rows wider than the original single-CTA block can
    # comfortably cover; keep each chunk as large as Tx81 SPM allows.
    if n_cols >= _MAX_TILE_BLOCK:
        return _MAX_TILE_BLOCK
    return min(_MAX_TILE_BLOCK, max(_MIN_TILE_BLOCK, triton.next_power_of_2(n_cols)))


def _num_warps(block_n):
    if block_n >= 4096:
        return 8
    if block_n >= 1024:
        return 4
    return 1


def fused_add_rms_norm(x, residual, normalized_shape, weight, eps=1e-5):
    logger.debug("GEMS TSINGMICRO FUSED_ADD_RMS_NORM FORWARD")

    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    if (
        N <= _SMALL_N_FALLBACK
        or x.is_complex()
        or residual.is_complex()
        or weight.is_complex()
        or weight.numel() != N
        or (
            _SINGLE_PASS_MAX_N < N <= _GENERIC_WIDE_N_MAX
            and N not in _EXACT_N_FASTPATHS
        )
    ):
        return _generic_fused_add_rms_norm(x, residual, normalized_shape, weight, eps)

    x = x.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()

    grid = (min(_TILE_GRID, M),)
    with torch_device_fn.device(x.device):
        if N <= _SINGLE_PASS_MAX_N:
            block_n = _pick_single_pass_block(N)
            _fused_add_rms_norm_single_pass_kernel[grid](
                x,
                residual,
                weight,
                M,
                N,
                eps,
                BLOCK_N=block_n,
                num_warps=_num_warps(block_n),
            )
        elif N in _EXACT_N_FASTPATHS:
            _fused_add_rms_norm_exact_n_kernel[(M,)](
                x,
                residual,
                weight,
                eps,
                BLOCK_N=N,
                num_warps=_num_warps(N),
            )
        else:
            block_n = _pick_tiled_block(N)
            _fused_add_rms_norm_tiled_kernel[grid](
                x,
                residual,
                weight,
                M,
                N,
                eps,
                BLOCK_N=block_n,
                num_warps=_num_warps(block_n),
            )
    return x, residual
