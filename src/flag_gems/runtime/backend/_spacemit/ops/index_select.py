import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle


def heur_block_m(args):
    # Cover all M rows in a single tile to avoid multi-CTA and scf.for issues
    # on SpaceMIT (only CTA(0,0) executes and loop induction vars are broken).
    return min(128, triton.next_power_of_2(args["M"]))


def heur_block_n(args):
    m = min(triton.next_power_of_2(triton.cdiv(args["N"], 16)), 512)
    return max(m, 16)


@libentry()
@triton.heuristics({"BLOCK_M": heur_block_m, "BLOCK_N": heur_block_n})
@triton.jit
def index_select_kernel(
    inp, out, M, N, index, index_len, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_y = tle.program_id(axis=1)
    rows_offsets = tl.arange(0, BLOCK_M)[:, None]
    rows_mask = rows_offsets < M
    cols_offsets = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols_offsets < index_len

    indices = tl.load(index + cols_offsets, mask=col_mask, other=0)
    inp_off = rows_offsets * N + indices[None, :]
    out_off = rows_offsets * index_len + cols_offsets[None, :]

    selected = tl.load(inp + inp_off, mask=rows_mask, other=0.0)
    tl.store(out + out_off, selected, mask=rows_mask & col_mask[None, :])


def index_select(inp, dim, index):
    logging.debug("GEMS_SPACEMIT INDEX_SELECT")
    assert dim >= -inp.ndim and dim < inp.ndim
    assert index.ndim <= 1
    if index.ndim == 0:
        index = index.unsqueeze(0)
    dim = dim % inp.ndim
    inp_shape = list(inp.shape)
    index_len = index.numel()

    inp = dim_compress(inp, dim)
    N = inp_shape[dim]
    M = inp.numel() // N
    out_shape = list(inp.shape)
    out_shape[inp.ndim - 1] = index_len
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    grid = lambda meta: (1, triton.cdiv(index_len, meta["BLOCK_N"]))
    index_select_kernel[grid](inp, out, M, N, index.to(torch.int32), index_len)

    if dim != out.ndim - 1:
        order = list(range(out.ndim - 1))
        order.insert(dim, out.ndim - 1)
        return out.permute(order)
    return out
