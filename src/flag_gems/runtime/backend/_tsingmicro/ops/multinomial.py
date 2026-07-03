import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

from .cumsum import normed_cumsum

logger = logging.getLogger(__name__)

_TILE_K = 1024


@triton.jit
def _hash_uniform(s_a, s_b, BLOCK: tl.constexpr):
    """Expand two fp32 seeds into BLOCK uniform samples via a fp32 hash."""
    lane = tl.arange(0, BLOCK).to(tl.float32)
    x = s_a * 0.1031 + s_b * 0.11369 + lane * 0.75487766
    x = x - tl.floor(x)
    x = x * (x + 33.33)
    x = x - tl.floor(x)
    x = x * (x + x + 19.19)
    return x - tl.floor(x)


@libentry()
@triton.jit(do_not_specialize=["K", "philox_seed", "philox_offset"])
def _multinomial_categorical_kernel(
    cum_prob_ptr: tl.tensor,
    out_ptr: tl.tensor,
    K: int,
    philox_seed: int,
    philox_offset: int,
    TILE_K: tl.constexpr,
):
    """One CTA per distribution row: one uniform and a tiled CDF scan."""
    pid = tl.program_id(0)

    # One Philox counter per row. Keep counters as uint32 before tl.philox; this
    # avoids the 32->64 bitcast failure in Triton/Tx81 Philox lowering.
    off64 = philox_offset.to(tl.int64) + pid.to(tl.int64)
    lo = (off64 & 0xFFFFFFFF).to(tl.uint32)
    hi = ((off64 >> 32) & 0xFFFFFFFF).to(tl.uint32)
    z = lo * 0
    r0, _, _, _ = tl.philox(philox_seed.to(tl.int64), lo, hi, z, z)
    u = uint_to_uniform_float(r0)
    u = tl.minimum(tl.maximum(u, 0.0001), 0.9999)

    cum_prob_ptr += pid * K
    idx = tl.zeros((1,), dtype=tl.float32)
    active = tl.full((1,), True, dtype=tl.int1)
    num_tiles = tl.cdiv(K, TILE_K)

    for t in range(num_tiles):
        t_off = t * TILE_K + tl.arange(0, TILE_K)
        t_mask = t_off < K
        cdf = tl.load(cum_prob_ptr + t_off, mask=t_mask, other=1.0)
        tile_count = tl.sum((cdf < u).to(tl.float32), axis=0)
        idx += tl.where(active, tile_count, 0.0)

        # Triton on Tx81 does not support tensor negative indexing or break.
        # Keep a scalar active flag instead of early-exiting the loop.
        last_off = tl.minimum((t + 1) * TILE_K - 1, K - 1)
        last_cdf = tl.load(cum_prob_ptr + last_off)
        active = active & (last_cdf < u)

    idx = tl.minimum(idx, (K - 1) * 1.0)
    tl.store(out_ptr + pid, idx.to(tl.int32))


@libentry()
@triton.jit(do_not_specialize=["K", "n_samples", "philox_seed", "philox_offset"])
def _multinomial_with_replacement_kernel(
    cum_prob_ptr: tl.tensor,
    out_ptr: tl.tensor,
    K: int,
    n_samples: int,
    philox_seed: int,
    philox_offset: int,
    NBLOCK: tl.constexpr,
    TILE_K: tl.constexpr,
):
    """With-replacement sampling via block-level Philox and tiled compare-sum."""
    batch_id = tl.program_id(0)
    dist_id = tl.program_id(1)

    # One Philox call per CTA, then expand to NBLOCK random samples with fp32
    # hash. This replaces per-sample Philox and avoids integer-heavy search.
    cta_idx = dist_id.to(tl.int64) * tl.num_programs(0).to(tl.int64) + batch_id.to(
        tl.int64
    )
    off64 = philox_offset.to(tl.int64) + cta_idx
    lo = (off64 & 0xFFFFFFFF).to(tl.uint32)
    hi = ((off64 >> 32) & 0xFFFFFFFF).to(tl.uint32)
    z = lo * 0
    r0, r1, _, _ = tl.philox(philox_seed.to(tl.int64), lo, hi, z, z)

    s0 = uint_to_uniform_float(r0)
    s1 = uint_to_uniform_float(r1)
    rvs = _hash_uniform(s0, s1, NBLOCK)
    rvs = tl.minimum(tl.maximum(rvs, 0.0001), 0.9999)

    cum_prob_ptr += dist_id * K
    n = batch_id * NBLOCK + tl.arange(0, NBLOCK)
    n_mask = n < n_samples

    idx = tl.zeros((NBLOCK,), dtype=tl.float32)
    num_tiles = tl.cdiv(K, TILE_K)
    for t in range(num_tiles):
        t_off = t * TILE_K + tl.arange(0, TILE_K)
        t_mask = t_off < K
        cdf_tile = tl.load(cum_prob_ptr + t_off, mask=t_mask, other=1.0)
        cmp = cdf_tile[None, :] < rvs[:, None]
        idx += tl.sum(cmp.to(tl.float32), axis=1)

    idx = tl.minimum(idx, (K - 1) * 1.0)
    out_base = out_ptr + dist_id * n_samples
    tl.store(out_base + n, idx.to(tl.int32), mask=n_mask)


def _pick_tile_k(k):
    if k <= _TILE_K:
        return triton.next_power_of_2(k)
    return _TILE_K


def _pick_nblock(n_samples):
    return min(triton.next_power_of_2(n_samples), 128)


def _zero_output(prob, n_samples):
    if prob.dim() == 1:
        return torch.zeros((n_samples,), device=prob.device, dtype=torch.int64)
    return torch.zeros((prob.size(0), n_samples), device=prob.device, dtype=torch.int64)


def multinomial(prob, n_samples, with_replacement=False, *, gen=None):
    logger.debug("GEMS_TSINGMICRO MULTINOMIAL")
    assert prob.dtype in (torch.float16, torch.float32, torch.bfloat16, torch.float64)
    assert 0 < prob.dim() <= 2, "prob_dist must be 1 or 2 dim"
    n_categories = prob.size(-1)
    assert n_categories <= (1 << 24), "number of categories cannot exceed 2^24"
    assert (
        with_replacement or n_samples <= n_categories
    ), "cannot sample n_samples > prob.size(-1) samples without replacement."

    if prob.dtype == torch.float64:
        from flag_gems.ops.multinomial import multinomial as _generic_multinomial

        return _generic_multinomial(prob, n_samples, with_replacement, gen=gen)

    prob = prob.contiguous()

    if n_categories == 1:
        return _zero_output(prob, n_samples)

    # For one sample, with- and without-replacement are equivalent. This path is
    # the critical token-sampling case and avoids exponential_ + topk.
    if n_samples == 1:
        cum_prob = normed_cumsum(prob, dim=-1)
        if cum_prob.dim() == 1:
            n_dist = 1
            out = torch.empty((1,), device=prob.device, dtype=torch.int32)
        else:
            n_dist = cum_prob.size(0)
            out = torch.empty((n_dist, 1), device=prob.device, dtype=torch.int32)

        tile_k = _pick_tile_k(n_categories)
        increment = n_dist
        philox_seed, philox_offset = philox_backend_seed_offset(
            increment, generator=gen
        )
        with torch_device_fn.device(prob.device):
            _multinomial_categorical_kernel[(n_dist,)](
                cum_prob,
                out,
                n_categories,
                philox_seed,
                philox_offset,
                TILE_K=tile_k,
                num_warps=4,
            )
        return out.to(torch.int64)

    # For multiple samples without replacement, keep the standard exponential
    # race + topk path. It already maps the heavy work to existing kernels.
    if not with_replacement:
        q = torch.empty_like(prob).exponential_(1.0)
        s = torch.div(prob, q, out=q)
        _, indices = torch.topk(s, n_samples, dim=-1)
        return indices.to(torch.int64)

    cum_prob = normed_cumsum(prob, dim=-1)
    if cum_prob.dim() == 1:
        n_dist = 1
        out = torch.empty((n_samples,), device=prob.device, dtype=torch.int32)
    else:
        n_dist = cum_prob.size(0)
        out = torch.empty((n_dist, n_samples), device=prob.device, dtype=torch.int32)

    nblock = _pick_nblock(n_samples)
    tile_k = _pick_tile_k(n_categories)
    grid = (triton.cdiv(n_samples, nblock), n_dist)
    # One Philox counter is consumed per CTA.
    increment = grid[0] * grid[1]
    philox_seed, philox_offset = philox_backend_seed_offset(increment, generator=gen)

    with torch_device_fn.device(prob.device):
        _multinomial_with_replacement_kernel[grid](
            cum_prob,
            out,
            n_categories,
            n_samples,
            philox_seed,
            philox_offset,
            NBLOCK=nblock,
            TILE_K=tile_k,
            num_warps=4,
        )

    return out.to(torch.int64)
