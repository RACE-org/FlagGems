import logging
import math

import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

try:
    from triton.backends.spine_triton.env import alloc_mbarrier, release_mbarrier
except ImportError:
    alloc_mbarrier = None
    release_mbarrier = None

logger = logging.getLogger(__name__)

NUM_CTAS = 8


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
    row_start = tl.program_id(0) * BLOCK_M
    row_end = min(row_start + BLOCK_M, M)
    for mi in range(row_start, row_end, 1):
        _all = tl.full([1, BLOCK_N], value=1, dtype=tl.int1)
        row_inp = inp + mi * N
        row_out = out + mi
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            mask = cols < N
            a = tl.load(row_inp + cols, mask, other=1.0)
            _all = _all and (a != 0)
        all_val = tl.reduce(_all, axis=1, combine_fn=reduce_all)
        tl.store(row_out, all_val)


@libentry()
@triton.jit
def all_kernel_1(
    inp,
    mid,
    n_elements,
    NUM_BLOCKS,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_INNER: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    sub_num = tl.cdiv(max(NUM_BLOCKS - pid, 0), num_ctas)
    _all = True

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        n_start = task_idx * BLOCK_SIZE
        n_end = min(n_start + BLOCK_SIZE, n_elements)

        for ni in range(n_start, n_end, BLOCK_INNER):
            offset = ni + tl.arange(0, BLOCK_INNER)
            mask = offset < n_elements
            inp_val = tl.load(inp + offset, mask=mask, other=1.0)
            all_val = tl.reduce(inp_val != 0, axis=0, combine_fn=reduce_all)
            _all = _all and all_val

    tl.store(mid + pid, _all)



@libentry()
@triton.jit
def all_kernel_2(mid, out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_SIZE
    mid_val = tl.load(mid + offset, mask=mask, other=1).to(tl.int1)
    all_val = tl.reduce(mid_val, axis=0, combine_fn=reduce_all)
    tl.store(out, all_val)


@libentry()
@triton.jit
def all_kernel_barrier(
    inp,
    mid,
    out,
    bar,
    n_elements,
    NUM_BLOCKS,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_INNER: tl.constexpr,
    BLOCK_MID: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    sub_num = tl.cdiv(max(NUM_BLOCKS - pid, 0), num_ctas)
    _all = True

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        n_start = task_idx * BLOCK_SIZE
        n_end = min(n_start + BLOCK_SIZE, n_elements)

        for ni in range(n_start, n_end, BLOCK_INNER):
            offset = ni + tl.arange(0, BLOCK_INNER)
            mask = offset < n_elements
            inp_val = tl.load(inp + offset, mask=mask, other=1.0)
            all_val = tl.reduce(inp_val != 0, axis=0, combine_fn=reduce_all)
            _all = _all and all_val

    tl.store(mid + pid, _all)
    smt.barrier_arrive(bar)

    if pid == tl.num_programs(0) - 1:
        smt.barrier_wait(bar, flag=1)
        offset = tl.arange(0, BLOCK_MID)
        mask = offset < tl.num_programs(0)
        mid_val = tl.load(mid + offset, mask=mask, other=1).to(tl.int1)
        all_val = tl.reduce(mid_val, axis=0, combine_fn=reduce_all)
        tl.store(out, all_val)


def all(inp):
    logger.debug("GEMS_SPACEMIT ALL")
    inp = inp.contiguous()
    n_elements = inp.numel()

    # Larger block_size than original sqrt-based (1024->4096) to reduce programs
    block_size = min(4096, triton.next_power_of_2(n_elements))
    block_inner = 256
    num_blocks = triton.cdiv(n_elements, block_size)
    mid_size = min(NUM_CTAS, num_blocks)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=torch.bool, device=inp.device)
    out = torch.empty([], dtype=torch.bool, device=inp.device)

    with torch_device_fn.device(inp.device):
        if alloc_mbarrier is not None and release_mbarrier is not None and mid_size <= 32767:
            bar = alloc_mbarrier(mid_size)
            try:
                all_kernel_barrier[(mid_size,)](
                    inp, mid, out, bar, n_elements, num_blocks, block_size, block_inner, block_mid
                )
            finally:
                release_mbarrier(bar)
        else:
            all_kernel_1[(mid_size,)](inp, mid, n_elements, num_blocks, block_size, block_inner)
            all_kernel_2[(1, 1)](mid, out, mid_size, block_mid)
    return out


def all_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_SPACEMIT ALL_DIM")
    shape = list(inp.shape)
    if dim is None:
        out = all(inp)
        if keepdim:
            out = torch.reshape(out, [1] * inp.ndim)
    else:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        dim = dim % inp.ndim
        inp = dim_compress(inp, dim)
        inp = inp.contiguous()
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
    logger.debug("GEMS_SPACEMIT ALL_DIMS")
    if dim is None or isinstance(dim, int):
        return all_dim(inp, dim=dim, keepdim=keepdim)
    assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"
    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    inp = dim_compress(inp, dim)
    inp = inp.contiguous()
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
