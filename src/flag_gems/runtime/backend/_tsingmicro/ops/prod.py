import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

TOTAL_CORE_NUM = 16
_FULL_TILE_BLOCK = 65536
_MIN_PER_TILE = 64


def _pick_block_size(per_tile):
    block = 1
    while block * 2 <= per_tile and block * 2 <= _FULL_TILE_BLOCK:
        block *= 2
    return max(block, _MIN_PER_TILE)


@triton.jit
def reduce_mul(a, b):
    return a * b


@libentry()
@triton.jit
def prod_kernel_mid(inp, mid, M, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0)
    per_tile = tl.cdiv(M, TOTAL_CORE_NUM)
    start = pid * per_tile

    acc = tl.full([1], 1.0, dtype=tl.float32)
    for off in range(0, per_tile, BLOCK_SIZE):
        idx = start + off + tl.arange(0, BLOCK_SIZE)
        mask = idx < M
        inp_val = tl.load(inp + idx, mask=mask, other=1.0).to(tl.float32)
        acc *= tl.reduce(inp_val, axis=0, combine_fn=reduce_mul)

    tl.store(mid + pid, tl.sum(acc))


@libentry()
@triton.jit
def prod_kernel_mid_single(inp, mid, M, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M
    inp_val = tl.load(inp + offset, mask=mask, other=1.0).to(tl.float32)
    prod_val = tl.reduce(inp_val, axis=0, combine_fn=reduce_mul)
    tl.store(mid + pid, prod_val)


@libentry()
@triton.jit
def prod_kernel_result(mid, out, MID_SIZE: tl.constexpr):
    idx = tl.arange(0, MID_SIZE)
    mid_val = tl.load(mid + idx).to(tl.float32)
    prod_val = tl.reduce(mid_val, axis=0, combine_fn=reduce_mul)
    tl.store(out, prod_val.to(out.dtype.element_ty))


def prod(inp, *, dtype=None):
    logger.debug("GEMS TSINGMICRO PROD")
    if dtype is None:
        dtype = inp.dtype

    M = inp.numel()
    per_tile = triton.cdiv(M, TOTAL_CORE_NUM)

    if per_tile >= _MIN_PER_TILE:
        block_size = _pick_block_size(per_tile)
        mid = torch.empty((TOTAL_CORE_NUM,), dtype=torch.float32, device=inp.device)
        out = torch.empty([], dtype=dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            prod_kernel_mid[(TOTAL_CORE_NUM, 1, 1)](inp, mid, M, block_size)
            prod_kernel_result[(1, 1, 1)](mid, out, TOTAL_CORE_NUM)
    else:
        block_size = max(triton.next_power_of_2(M), 1)
        mid = torch.empty((1,), dtype=torch.float32, device=inp.device)
        out = torch.empty([], dtype=dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            prod_kernel_mid_single[(1, 1, 1)](inp, mid, M, block_size)
            prod_kernel_result[(1, 1, 1)](mid, out, 1)
    return out


def heur_block_n(args):
    return triton.next_power_of_2(args["N"])


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def prod_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tle.program_id(0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    acc = tl.full((BLOCK_M, BLOCK_N), value=1.0, dtype=tl.float32)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N + n_offset[None, :]

        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=1.0).to(tl.float32)
        acc *= inp_vals
    result_index = tl.reduce(acc, axis=1, combine_fn=reduce_mul)

    offset_index = m_offset
    out_ptrs = out + offset_index
    mask1 = m_offset < M
    tl.store(out_ptrs, result_index, mask=mask1)


def prod_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS TSINGMICRO PROD DIM")

    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = list(inp.shape)
    dim = dim % inp.ndim
    inp = dim_compress(inp, dim)
    N = shape[dim]
    shape[dim] = 1
    M = inp.numel() // N

    if dtype is None:
        dtype = inp.dtype
    out = torch.empty(shape, dtype=dtype, device=inp.device)
    if not keepdim:
        out = torch.squeeze(out, dim)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    with torch_device_fn.device(inp.device):
        prod_kernel[grid](inp, out, M, N)

    return out
