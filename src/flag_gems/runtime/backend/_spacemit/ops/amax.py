import logging
import builtins

import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_min

try:
    from triton.backends.spine_triton.env import alloc_mbarrier, release_mbarrier
except ImportError:
    alloc_mbarrier = None
    release_mbarrier = None

logger = logging.getLogger(__name__)

NUM_CTAS = 8


@libentry()
@triton.jit
def amax_kernel_1(
    inp,
    mid,
    M,
    NUM_BLOCKS,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_INNER: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    sub_num = tl.cdiv(tl.maximum(NUM_BLOCKS - pid, 0), num_ctas)
    dtype = inp.type.element_ty
    min_value = get_dtype_min(dtype)
    amax_val = tl.full((), value=min_value, dtype=dtype)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        n_start = task_idx * BLOCK_SIZE
        n_end = tl.minimum(n_start + BLOCK_SIZE, M)

        for ni in range(n_start, n_end, BLOCK_INNER):
            offset = ni + tl.arange(0, BLOCK_INNER)
            mask = offset < M
            inp_val = tl.load(inp + offset, mask=mask, other=min_value)
            amax_val = tl.maximum(amax_val, tl.max(inp_val).to(dtype))

    tl.store(mid + pid, amax_val)


@libentry()
@triton.jit
def amax_kernel_2(mid, out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_SIZE
    min_value = get_dtype_min(mid.type.element_ty)
    mid_val = tl.load(mid + offset, mask=mask, other=min_value)
    amax_val = tl.max(mid_val)
    tl.store(out, amax_val)


@libentry()
@triton.jit
def amax_kernel_barrier(
    inp,
    mid,
    out,
    bar,
    M,
    NUM_BLOCKS,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_INNER: tl.constexpr,
    BLOCK_MID: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    sub_num = tl.cdiv(tl.maximum(NUM_BLOCKS - pid, 0), num_ctas)
    dtype = inp.type.element_ty
    min_value = get_dtype_min(dtype)
    amax_val = tl.full((), value=min_value, dtype=dtype)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        n_start = task_idx * BLOCK_SIZE
        n_end = tl.minimum(n_start + BLOCK_SIZE, M)

        for ni in range(n_start, n_end, BLOCK_INNER):
            offset = ni + tl.arange(0, BLOCK_INNER)
            mask = offset < M
            inp_val = tl.load(inp + offset, mask=mask, other=min_value)
            amax_val = tl.maximum(amax_val, tl.max(inp_val).to(dtype))

    tl.store(mid + pid, amax_val)
    smt.barrier_arrive(bar)

    if pid == tl.num_programs(0) - 1:
        smt.barrier_wait(bar, flag=1)
        offset = tl.arange(0, BLOCK_MID)
        mask = offset < tl.num_programs(0)
        mid_val = tl.load(mid + offset, mask=mask, other=min_value)
        final_amax_val = tl.max(mid_val).to(dtype)
        tl.store(out, final_amax_val)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def amax_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    dtype = inp.type.element_ty
    min_value = get_dtype_min(dtype)

    pid = tle.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    _all = tl.full([BLOCK_M, BLOCK_N], value=min_value, dtype=acc_type)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask
        a = tl.load(inp + cols, mask, other=min_value)
        _all = tl.maximum(_all, a)
    all = tl.max(_all, axis=1)[:, None]
    tl.store(out, all, row_mask)


def amax(inp, dim=None, keepdim=False):
    logger.debug("GEMS_SPACEMIT AMAX")
    if dim is None or len(dim) == 0:
        inp = inp.contiguous()
        M = inp.numel()
        block_size = builtins.min(4096, triton.next_power_of_2(M))
        block_inner = 256
        num_blocks = triton.cdiv(M, block_size)
        mid_size = min(NUM_CTAS, num_blocks)
        block_mid = triton.next_power_of_2(mid_size)
        dtype = inp.dtype
        mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        if not keepdim:
            out = torch.empty([], dtype=dtype, device=inp.device)
        else:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape, dtype=dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            if alloc_mbarrier is not None and release_mbarrier is not None and mid_size <= 32767:
                bar = alloc_mbarrier(mid_size)
                try:
                    amax_kernel_barrier[(mid_size,)](
                        inp, mid, out, bar, M, num_blocks, block_size, block_inner, block_mid
                    )
                finally:
                    release_mbarrier(bar)
            else:
                amax_kernel_1[(mid_size,)](inp, mid, M, num_blocks, block_size, block_inner)
                amax_kernel_2[(1, 1)](mid, out, mid_size, block_mid)
        return out
    else:
        if isinstance(dim, int):
            dim = [dim]
        assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"
        dtype = inp.dtype

        shape = list(inp.shape)
        dim = [d % inp.ndim for d in dim]
        inp = dim_compress(inp, dim)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = inp.numel() // N

        out = torch.empty(shape, dtype=dtype, device=inp.device)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch_device_fn.device(inp.device):
            amax_kernel[grid](inp, out, M, N)
        if not keepdim:
            out = out.squeeze(dim=dim)
        return out