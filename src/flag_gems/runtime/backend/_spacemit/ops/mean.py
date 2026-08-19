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

import os

try:
    from triton.backends.spine_triton.env import alloc_mbarrier, release_mbarrier
except ImportError:
    alloc_mbarrier = None
    release_mbarrier = None

if os.environ.get("SPINE_TRITON_RPC_HOST"):
    alloc_mbarrier = None
    release_mbarrier = None

logger = logging.getLogger(__name__)

NUM_CTAS = 8


@libentry()
@triton.jit
def mean_kernel_1(
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
    sum_val = tl.zeros((), dtype=tl.float32)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        n_start = task_idx * BLOCK_SIZE
        n_end = tl.minimum(n_start + BLOCK_SIZE, M)

        for ni in range(n_start, n_end, BLOCK_INNER):
            offset = ni + tl.arange(0, BLOCK_INNER)
            mask = offset < M
            inp_val = tl.load(inp + offset, mask=mask, other=0.0).to(tl.float32)
            sum_val += tl.sum(inp_val)

    tl.store(mid + pid, sum_val)


@libentry()
@triton.jit
def mean_kernel_2(mid, out, M, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_SIZE
    mid_val = tl.load(mid + offset, mask=mask, other=0.0).to(tl.float32)
    sum_val = tl.sum(mid_val, axis=0) / M
    tl.store(out, sum_val.to(out.dtype.element_ty))


@libentry()
@triton.jit
def mean_kernel_barrier(
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
    sum_val = tl.zeros((), dtype=tl.float32)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        n_start = task_idx * BLOCK_SIZE
        n_end = tl.minimum(n_start + BLOCK_SIZE, M)

        for ni in range(n_start, n_end, BLOCK_INNER):
            offset = ni + tl.arange(0, BLOCK_INNER)
            mask = offset < M
            inp_val = tl.load(inp + offset, mask=mask, other=0.0).to(tl.float32)
            sum_val += tl.sum(inp_val)

    tl.store(mid + pid, sum_val)
    smt.barrier_arrive(bar)

    if pid == tl.num_programs(0) - 1:
        smt.barrier_wait(bar, flag=1)
        offset = tl.arange(0, BLOCK_MID)
        mask = offset < tl.num_programs(0)
        mid_val = tl.load(mid + offset, mask=mask, other=0.0).to(tl.float32)
        final_sum = tl.sum(mid_val, axis=0) / M
        tl.store(out, final_sum.to(out.dtype.element_ty))


def mean(inp, *, dtype=None):
    logger.debug("GEMS_SPACEMIT MEAN")
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype

    block_size = builtins.min(4096, triton.next_power_of_2(M))
    block_inner = 256
    num_blocks = triton.cdiv(M, block_size)
    mid_size = min(NUM_CTAS, num_blocks)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=torch.float32, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        if alloc_mbarrier is not None and release_mbarrier is not None and mid_size <= 32767:
            bar = alloc_mbarrier(mid_size)
            try:
                mean_kernel_barrier[(mid_size,)](
                    inp, mid, out, bar, M, num_blocks, block_size, block_inner, block_mid
                )
            finally:
                release_mbarrier(bar)
        else:
            mean_kernel_1[(mid_size,)](inp, mid, M, num_blocks, block_size, block_inner)
            mean_kernel_2[(1, 1)](mid, out, M, mid_size, block_mid)
    return out


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mean_spacemit_v1"),
    key=["M", "N"],
)
@triton.jit
def mean_dim_kernel(X, Mean, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tle.program_id(0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = m_offset < M

    _mean_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N + cols[None, :]
        mask = row_mask[:, None] & (cols[None, :] < N)
        a = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
        _mean_acc += a

    mean = tl.sum(_mean_acc, axis=1) / N
    tl.store(Mean + m_offset, mean.to(Mean.dtype.element_ty), mask=row_mask)


def mean_dim(x, dim, keepdim=False, *, dtype=None):
    logger.debug("GEMS_SPACEMIT MEAN_DIM")

    if dtype is None:
        dtype = x.dtype
    if dim is None:
        out = mean(x, dtype=dtype)
        if not keepdim:
            out = out.reshape([1] * x.ndim)
        return out

    shape = list(x.shape)
    dim = [d % x.ndim for d in dim]
    x = dim_compress(x, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = x.numel() // N
    out = torch.empty(shape, dtype=dtype, device=x.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
    with torch_device_fn.device(x.device):
        mean_dim_kernel[grid](x, out, M, N)
    if not keepdim:
        out = out.squeeze(dim)
    return out


def avg_pool2d(x, kernel_size=None, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    return mean_dim(x, dim=[2, 3], keepdim=True)
