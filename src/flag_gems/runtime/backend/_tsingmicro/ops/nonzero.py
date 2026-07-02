import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from .cumsum import cumsum_wrapper

logger = logging.getLogger(__name__)

# ===========================================================================
# Block-based nonzero — avoids per-element // % for all n_elements.
#
# Current upstream kernel does:
#   1. cumsum over ALL n_elements (O(N) scalar prefix-sum)
#   2. for EVERY element: // and % to compute multi-dim coords (O(N*d) scalar
#      int division — Tx81 has no int vector div, kills perf)
#
# New approach (5 stages):
#   Stage 1: block counting — vector reduce per block, O(1) per block
#   Stage 2: cumsum on block counts — small tensor, GEMM on NE
#   Stage 3: block-local cumsum + compact flat indices — per-block GEMM
#   Stage 4: flat index → multi-dim coords — only for nnz elements
# ===========================================================================

_BLOCK_SIZE = 1024


@libentry()
@triton.jit
def nonzero_block_count_kernel(
    inp,
    block_counts,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Stage 1: count nonzeros per block via hardware reduce."""
    pid = tle.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    vals = tl.load(inp + offs, mask=mask, other=0).to(tl.int1)
    count = tl.sum(vals.to(tl.int32))
    tl.store(block_counts + pid, count)


@libentry()
@triton.jit
def nonzero_compact_kernel(
    inp,
    block_base,
    flat_out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Stage 3: block-local cumsum + scatter flat index for nonzero elements.

    BLOCK_SIZE ≤ 1024 keeps tl.cumsum scalar iterations per CTA bounded.
    """
    pid = tle.program_id(0)
    blk_start = pid * BLOCK_SIZE
    offs = blk_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    vals_int = tl.load(inp + offs, mask=mask, other=0).to(tl.int32)
    local_cumsum = tl.cumsum(vals_int, axis=0)

    base = tl.load(block_base + pid)

    out_pos = base + local_cumsum - 1
    # Only store positions where the original value is nonzero.
    # local_cumsum > 0 would also match zero elements that follow a
    # nonzero (cumsum plateaus), causing them to overwrite the
    # preceding nonzero's flat-index entry.
    store_mask = mask & (vals_int != 0)
    tl.store(flat_out + out_pos, offs.to(tl.int32), mask=store_mask)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("elementwise_generic"))
@triton.jit
def nonzero_flat_to_coord_kernel(
    flat_in,
    out,
    shape,
    nnz,
    ndim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Stage 4: convert flat indices to multi-dimensional coordinates.

    Only processes nnz elements (not all n_elements), reducing scalar
    int // % operations by ~10× (assuming 10% nonzeros).
    """
    pid = tle.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < nnz

    idx_flat = tl.load(flat_in + offs, mask=mask, other=0)
    for dim in range(ndim - 1, -1, -1):
        dim_size = tl.load(shape + dim)
        remainder = idx_flat % dim_size
        idx_flat //= dim_size
        tl.store(out + offs * ndim + dim, remainder.to(tl.int64), mask=mask)


def nonzero(inp, *, as_tuple=False):
    logger.debug("GEMS TSINGMICRO NONZERO")

    inp_ndim = inp.ndim
    inp = inp.contiguous()
    n_elements = inp.numel()
    inp_view = inp.view(n_elements)

    shape = torch.tensor(inp.shape, dtype=torch.int32, device=inp.device)

    inp_bool = inp_view.to(torch.int8)
    if inp_view.dtype != torch.bool:
        inp_bool = (inp_view != 0).to(torch.int8)

    # Stage 1: block counting
    num_blocks = triton.cdiv(n_elements, _BLOCK_SIZE)
    block_counts = torch.empty(num_blocks, dtype=torch.int32, device=inp.device)

    grid1 = (num_blocks,)
    with torch_device_fn.device(inp.device):
        nonzero_block_count_kernel[grid1](
            inp_bool, block_counts, n_elements, _BLOCK_SIZE,
        )

    # Stage 2: cumsum on block counts → block_base (exclusive prefix).
    # block_counts is int32 with small values (≤1024 per block); cumsum
    # via scan_then_fan_col gives exact scalar addi (1024 iterations).
    # Avoids the GEMM path — NE GEMM uses tf32 (~10-bit mantissa) which
    # would introduce ULP≈1024 in the final cumulative sum, shifting the
    # exclusive-prefix offsets and producing the wrong output shape.
    with torch_device_fn.device(inp.device):
        cumsum_1d = cumsum_wrapper(block_counts, dim=0)

    # Exclusive prefix: block_base[0] = 0, block_base[i] = cumsum_1d[i-1]
    block_base = torch.zeros_like(cumsum_1d)
    block_base[1:] = cumsum_1d[:-1]

    # Stage 3: block-local cumsum + compact flat indices
    total_nonzeros = cumsum_1d[-1].item()

    if total_nonzeros == 0:
        out = torch.empty(0, inp_ndim, dtype=torch.int64, device=inp.device)
        if as_tuple:
            return tuple(out[:, i] for i in range(inp_ndim))
        return out

    flat_out = torch.empty(total_nonzeros, dtype=torch.int32, device=inp.device)

    grid3 = (num_blocks,)
    with torch_device_fn.device(inp.device):
        nonzero_compact_kernel[grid3](
            inp_bool, block_base, flat_out, n_elements, _BLOCK_SIZE,
        )

    # Stage 4: flat index → multi-dim coords
    out = torch.empty(total_nonzeros, inp_ndim, dtype=torch.int64, device=inp.device)

    grid4 = lambda meta: (triton.cdiv(total_nonzeros, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(inp.device):
        nonzero_flat_to_coord_kernel[grid4](
            flat_out, out, shape, total_nonzeros, inp_ndim,
        )

    if as_tuple:
        return tuple(out[:, i] for i in range(inp_ndim))
    else:
        return out
