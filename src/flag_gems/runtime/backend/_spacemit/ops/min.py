import logging
import builtins
from collections import namedtuple

import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_max

try:
    from triton.backends.spine_triton.env import alloc_mbarrier, release_mbarrier
except ImportError:
    alloc_mbarrier = None
    release_mbarrier = None

logger = logging.getLogger(__name__)

NUM_CTAS = 8


@libentry()
@triton.jit
def min_kernel_1(
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
    max_value = get_dtype_max(dtype)
    min_val = tl.full((), value=max_value, dtype=dtype)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        n_start = task_idx * BLOCK_SIZE
        n_end = tl.minimum(n_start + BLOCK_SIZE, M)

        for ni in range(n_start, n_end, BLOCK_INNER):
            offset = ni + tl.arange(0, BLOCK_INNER)
            mask = offset < M
            inp_val = tl.load(inp + offset, mask=mask, other=max_value)
            local_min = tl.min(inp_val).to(dtype)
            min_val = tl.where(local_min < min_val, local_min, min_val)

    tl.store(mid + pid, min_val)


@libentry()
@triton.jit
def min_kernel_2(mid, out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_SIZE
    max_value = get_dtype_max(mid.type.element_ty)
    mid_val = tl.load(mid + offset, mask=mask, other=max_value)
    min_val = tl.min(mid_val)
    tl.store(out, min_val)


@libentry()
@triton.jit
def min_kernel_barrier(
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
    max_value = get_dtype_max(dtype)
    min_val = tl.full((), value=max_value, dtype=dtype)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        n_start = task_idx * BLOCK_SIZE
        n_end = tl.minimum(n_start + BLOCK_SIZE, M)

        for ni in range(n_start, n_end, BLOCK_INNER):
            offset = ni + tl.arange(0, BLOCK_INNER)
            mask = offset < M
            inp_val = tl.load(inp + offset, mask=mask, other=max_value)
            local_min = tl.min(inp_val).to(dtype)
            min_val = tl.where(local_min < min_val, local_min, min_val)

    tl.store(mid + pid, min_val)
    smt.barrier_arrive(bar)

    if pid == tl.num_programs(0) - 1:
        smt.barrier_wait(bar, flag=1)
        offset = tl.arange(0, BLOCK_MID)
        mask = offset < tl.num_programs(0)
        mid_val = tl.load(mid + offset, mask=mask, other=max_value)
        final_min_val = tl.min(mid_val).to(dtype)
        tl.store(out, final_min_val)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def min_kernel(
    inp,
    out_value,
    out_index,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tle.program_id(0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    dtype = inp.type.element_ty
    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    max_value = get_dtype_max(dtype)
    min_values = tl.full([BLOCK_M], dtype=acc_type, value=max_value)
    argmin_values = tl.full([BLOCK_M], dtype=tl.int64, value=0)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N + n_offset[None, :]
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_vals = tl.load(inp + offset, mask=mask, other=max_value)
        local_min, local_argmin = tl.min(inp_vals, 1, return_indices=True)
        update = local_min < min_values
        min_values = tl.where(update, local_min, min_values)
        argmin_values = tl.where(update, start_n + local_argmin, argmin_values)

    mask1 = m_offset < M
    tl.store(out_value + m_offset, min_values, mask=mask1)
    tl.store(out_index + m_offset, argmin_values, mask=mask1)


def min(inp):
    logger.debug("GEMS_SPACEMIT MIN")
    inp = inp.contiguous()
    M = inp.numel()
    block_size = builtins.min(4096, triton.next_power_of_2(M))
    block_inner = 256
    num_blocks = triton.cdiv(M, block_size)
    mid_size = builtins.min(NUM_CTAS, num_blocks)
    block_mid = triton.next_power_of_2(mid_size)

    dtype = inp.dtype
    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        if alloc_mbarrier is not None and release_mbarrier is not None and mid_size <= 32767:
            bar = alloc_mbarrier(mid_size)
            try:
                min_kernel_barrier[(mid_size,)](
                    inp, mid, out, bar, M, num_blocks, block_size, block_inner, block_mid
                )
            finally:
                release_mbarrier(bar)
        else:
            min_kernel_1[(mid_size,)](inp, mid, M, num_blocks, block_size, block_inner)
            min_kernel_2[(1, 1)](mid, out, mid_size, block_mid)
    return out


def min_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_SPACEMIT MIN DIM")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = list(inp.shape)
    dim = dim % inp.ndim
    inp = dim_compress(inp, dim)
    N = shape[dim]
    shape[dim] = 1
    M = inp.numel() // N

    out_value = torch.empty(shape, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(shape, dtype=torch.int64, device=inp.device)

    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    with torch_device_fn.device(inp.device):
        min_kernel[grid](inp, out_value, out_index, M, N)
    Min_out = namedtuple("min", ["values", "indices"])
    out = Min_out(values=out_value, indices=out_index)
    return out