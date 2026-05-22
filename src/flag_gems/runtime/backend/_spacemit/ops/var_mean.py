import logging
import builtins

import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

try:
    from triton.backends.spine_triton.env import alloc_mbarrier, release_mbarrier
except ImportError:
    alloc_mbarrier = None
    release_mbarrier = None

logger = logging.getLogger(__name__)

NUM_CTAS = 8


@triton.jit
def welford_func(mean_x, count_x, M_x, mean_y, count_y, M_y):
    count = count_x + count_y
    one = tl.full(count.shape, 1, count.dtype)
    safe_count = tl.maximum(count, one)
    mc_x = mean_x * count_x
    mc_y = mean_y * count_y
    mean = (mc_x + mc_y) / safe_count
    M = M_x + mc_x * mean_x + M_y + mc_y * mean_y - count * mean * mean
    return mean, count, M


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("var_mean"), key=["M", "N"])
@triton.jit(do_not_specialize=["correction"])
def var_mean_welford_kernel(
    X,
    Var,
    Mean,
    M,
    N,
    correction,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = rows < M

    X_block_ptr = tl.make_block_ptr(
        base=X,
        shape=(M, N),
        strides=(N, 1),
        offsets=(pid * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    var_block_ptr = tl.make_block_ptr(
        base=Var,
        shape=(M, 1),
        strides=(1, 1),
        offsets=(pid * BLOCK_M, 0),
        block_shape=(BLOCK_M, 1),
        order=(1, 0),
    )
    mean_block_ptr = tl.make_block_ptr(
        base=Mean,
        shape=(M, 1),
        strides=(1, 1),
        offsets=(pid * BLOCK_M, 0),
        block_shape=(BLOCK_M, 1),
        order=(1, 0),
    )

    dtype = X.dtype.element_ty

    _mean = tl.zeros((BLOCK_M, BLOCK_N), dtype=dtype)
    _acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dtype)
    _count = tl.zeros((BLOCK_M, BLOCK_N), dtype=dtype)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask
        mask_t = mask.to(dtype)

        x = tl.load(X_block_ptr, boundary_check=(0, 1)).to(dtype)
        X_block_ptr = tl.advance(X_block_ptr, (0, BLOCK_N))

        x = x * mask_t

        count = _count + mask_t
        one = tl.full(count.shape, 1, dtype)
        cnt = tl.maximum(count, one)
        cur_mean = (_mean * _count + x) / cnt
        _acc += (x - cur_mean) * (x - _mean) * mask_t
        _mean = cur_mean
        _count = count

    mean, _, acc = tl.reduce((_mean, _count, _acc), axis=1, combine_fn=welford_func)
    var = acc / (N - correction)

    mean = mean[:, None]
    var = var[:, None]

    tl.store(mean_block_ptr, mean, boundary_check=(0, 1))
    tl.store(var_block_ptr, var, boundary_check=(0, 1))


@libentry()
@triton.jit
def var_mean_kernel_1(
    X,
    Acc,
    Average,
    Count,
    N,
    NUM_BLOCKS,
    BLOCK_N: tl.constexpr,
    BLOCK_INNER: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    sub_num = tl.cdiv(tl.maximum(NUM_BLOCKS - pid, 0), num_ctas)
    dtype = X.dtype.element_ty
    count = tl.zeros((), dtype=dtype)
    sum_val = tl.zeros((), dtype=dtype)
    sum_square = tl.zeros((), dtype=dtype)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        n_start = task_idx * BLOCK_N
        n_end = tl.minimum(n_start + BLOCK_N, N)

        for ni in range(n_start, n_end, BLOCK_INNER):
            offset = ni + tl.arange(0, BLOCK_INNER)
            mask = offset < N
            x = tl.load(X + offset, mask=mask, other=0.0).to(dtype)
            sum_val += tl.sum(x)
            sum_square += tl.sum(x * x)
            count += tl.sum(mask.to(dtype))

    safe_count = tl.maximum(count, tl.full((), 1, dtype))
    average = sum_val / safe_count
    acc = sum_square - count * average * average

    tl.store(Average + pid, average)
    tl.store(Acc + pid, acc)
    tl.store(Count + pid, count)


@libentry()
@triton.jit(do_not_specialize=["correction"])
def var_mean_kernel_barrier(
    X,
    Acc,
    Average,
    Count,
    Var,
    Mean,
    bar,
    N,
    correction,
    NUM_BLOCKS,
    BLOCK_N: tl.constexpr,
    BLOCK_INNER: tl.constexpr,
    BLOCK_MID: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    sub_num = tl.cdiv(tl.maximum(NUM_BLOCKS - pid, 0), num_ctas)
    dtype = X.dtype.element_ty
    count = tl.zeros((), dtype=dtype)
    sum_val = tl.zeros((), dtype=dtype)
    sum_square = tl.zeros((), dtype=dtype)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        n_start = task_idx * BLOCK_N
        n_end = tl.minimum(n_start + BLOCK_N, N)

        for ni in range(n_start, n_end, BLOCK_INNER):
            offset = ni + tl.arange(0, BLOCK_INNER)
            mask = offset < N
            x = tl.load(X + offset, mask=mask, other=0.0).to(dtype)
            sum_val += tl.sum(x)
            sum_square += tl.sum(x * x)
            count += tl.sum(mask.to(dtype))

    safe_count = tl.maximum(count, tl.full((), 1, dtype))
    average = sum_val / safe_count
    acc = sum_square - count * average * average
    tl.store(Average + pid, average)
    tl.store(Acc + pid, acc)
    tl.store(Count + pid, count)
    smt.barrier_arrive(bar)

    if pid == tl.num_programs(0) - 1:
        smt.barrier_wait(bar, flag=1)
        offset = tl.arange(0, BLOCK_MID)
        mask = offset < tl.num_programs(0)
        zero = tl.full(offset.shape, 0, dtype)
        acc = tl.load(Acc + offset, mask=mask, other=zero).to(dtype)
        average = tl.load(Average + offset, mask=mask, other=zero).to(dtype)
        count = tl.load(Count + offset, mask=mask, other=zero).to(dtype)

        mean, _, nvar = tl.reduce((average, count, acc), axis=0, combine_fn=welford_func)
        var = nvar / (N - correction)
        tl.store(Mean, mean)
        tl.store(Var, var)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("var_mean"))
@triton.jit(do_not_specialize=["correction"])
def var_mean_kernel_2(
    Acc,
    Average,
    Count,
    Var,
    Mean,
    N,
    correction,
    BLOCK_NUM,
    BLOCK_N: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_N)
    mask = offset < BLOCK_NUM

    Acc = Acc + offset
    Average = Average + offset
    Count = Count + offset

    dtype = Acc.dtype.element_ty
    zero = tl.full(offset.shape, 0, dtype)

    acc = tl.load(Acc, mask=mask, other=zero).to(dtype)
    average = tl.load(Average, mask=mask, other=zero).to(dtype)
    count = tl.load(Count, mask=mask, other=zero).to(dtype)

    mean, _, nvar = tl.reduce((average, count, acc), axis=0, combine_fn=welford_func)

    var = nvar / (N - correction)
    tl.store(Mean, mean)
    tl.store(Var, var)


def var_mean(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_SPACEMIT VAR_MEAN")
    if correction is None:
        correction = 1.0

    if dim is None or len(dim) == x.ndim:
        dim = list(range(x.ndim))
        shape = [1] * x.ndim
        N = x.numel()

        var = torch.empty(shape, dtype=x.dtype, device=x.device)
        mean = torch.empty(shape, dtype=x.dtype, device=x.device)

        BLOCK_N = builtins.min(4096, triton.next_power_of_2(N))
        BLOCK_INNER = 256
        NUM_BLOCKS = triton.cdiv(N, BLOCK_N)
        BLOCK_NUM = min(NUM_CTAS, NUM_BLOCKS)
        BLOCK_MID = triton.next_power_of_2(BLOCK_NUM)

        acc = torch.empty((BLOCK_NUM,), dtype=x.dtype, device=x.device)
        average = torch.empty((BLOCK_NUM,), dtype=x.dtype, device=x.device)
        count = torch.empty((BLOCK_NUM,), dtype=x.dtype, device=x.device)

        with torch_device_fn.device(x.device):
            if alloc_mbarrier is not None and release_mbarrier is not None and BLOCK_NUM <= 32767:
                bar = alloc_mbarrier(BLOCK_NUM)
                try:
                    var_mean_kernel_barrier[(BLOCK_NUM,)](
                        x,
                        acc,
                        average,
                        count,
                        var,
                        mean,
                        bar,
                        N,
                        correction,
                        NUM_BLOCKS,
                        BLOCK_N,
                        BLOCK_INNER,
                        BLOCK_MID,
                    )
                finally:
                    release_mbarrier(bar)
            else:
                var_mean_kernel_1[(BLOCK_NUM,)](
                    x, acc, average, count, N, NUM_BLOCKS, BLOCK_N, BLOCK_INNER
                )
                var_mean_kernel_2[(1,)](
                    acc, average, count, var, mean, N, correction, BLOCK_NUM, BLOCK_MID
                )
    else:
        shape = list(x.shape)
        dim = [d % x.ndim for d in dim]
        x = dim_compress(x, dim)

        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1

        M = x.numel() // N
        var = torch.empty(shape, dtype=x.dtype, device=x.device)
        mean = torch.empty(shape, dtype=x.dtype, device=x.device)

        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
        with torch_device_fn.device(x.device):
            var_mean_welford_kernel[grid](x, var, mean, M, N, correction)

    if not keepdim:
        var = var.squeeze(dim=dim)
        mean = mean.squeeze(dim=dim)

    return var, mean
