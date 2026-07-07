import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as tle

try:
    import torch_npu  # noqa: F401
except:  # noqa: E722
    pow = tl_extra_shim.pow
logger = logging.getLogger(__name__)


# ---- L2 norm (dim reduction) ----
@libentry()
@triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.jit
def l2_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    num_tasks = tl.cdiv(M, BLOCK_M)
    sub_num = tl.cdiv(num_tasks - pid, num_ctas)

    for block_idx in range(sub_num):
        task_idx = pid + num_ctas * block_idx

        X_block_ptr = tl.make_block_ptr(
            base=X,
            shape=[M, N],
            strides=[N, 1],
            offsets=[task_idx * BLOCK_M, 0],
            block_shape=[BLOCK_M, BLOCK_N],
            order=[1, 0],
        )
        Out_block_ptr = tl.make_block_ptr(
            base=Out,
            shape=(M, 1),
            strides=(1, 1),
            offsets=(task_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, 1),
            order=(1, 0),
        )

        x_dtype = X_block_ptr.dtype.element_ty
        _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            a = tl.load(X_block_ptr, boundary_check=(0, 1))
            a = a.to(tl.float32)
            X_block_ptr = tl.advance(X_block_ptr, (0, BLOCK_N))
            _sum += a * a
        sum = tl.sum(_sum, axis=1)
        out = tl.sqrt(sum)[:, None]
        tl.store(Out_block_ptr, out.to(x_dtype), boundary_check=(0, 1))


# ---- L2 norm (global, two-pass) ----
@libentry()
@triton.jit
def l2_norm_kernel_1(X, Mid, M, num_tasks, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0).to(tl.int64)
    num_ctas = tl.num_programs(0)
    sub_num = tl.cdiv(num_tasks - pid, num_ctas)

    for block_idx in range(sub_num):
        task_idx = pid + num_ctas * block_idx

        X_block_ptr = tl.make_block_ptr(
            base=X,
            shape=(M,),
            strides=(1,),
            offsets=(task_idx * BLOCK_SIZE,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        offset = task_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < M

        x = tl.load(X_block_ptr, boundary_check=(0,)).to(tl.float32)
        x = tl.where(mask, x, 0.0)
        mid = tl.sum(x * x)
        tl.store(Mid + task_idx, mid)


@libentry()
@triton.jit
def l2_norm_kernel_2(Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    Mid = Mid + offset
    mask = offset < MID_SIZE
    mid = tl.load(Mid, mask=mask, other=0.0).to(tl.float32)
    out = tl.sqrt(tl.sum(mid))
    tl.store(Out, out)


# ---- Max norm (dim reduction) ----
@libentry()
@triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.jit
def max_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_raw = tle.program_id(0).to(tl.int64)
    num_ctas = tl.num_programs(0)
    num_tasks = tl.cdiv(M, BLOCK_M)
    sub_num = tl.cdiv(num_tasks - pid_raw, num_ctas)

    for block_idx in range(sub_num):
        task_idx = pid_raw + num_ctas * block_idx
        pid = task_idx * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        X_ptr = X + pid * N
        Out_ptr = Out + pid
        row_mask = pid < M

        _max = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = (row_mask) & (col_mask)
            a = tl.load(X_ptr + cols, mask, other=0.0).to(tl.float32)
            _max = tl.maximum(tl.abs(a), _max)
        max = tl.max(_max, axis=1)
        out = max[:, None]
        tl.store(Out_ptr, out, row_mask)


@libentry()
@triton.jit
def max_norm_kernel_1(X, Mid, M, num_tasks, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0).to(tl.int64)
    num_ctas = tl.num_programs(0)
    sub_num = tl.cdiv(num_tasks - pid, num_ctas)

    for block_idx in range(sub_num):
        task_idx = pid + num_ctas * block_idx
        offset = task_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        X_ptr = X + offset
        mask = offset < M
        x = tl.load(X_ptr, mask=mask, other=0.0).to(tl.float32)
        mid = tl.max(tl.abs(x))
        tl.store(Mid + task_idx, mid)


@libentry()
@triton.jit
def max_norm_kernel_2(Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    Mid = Mid + offset
    mask = offset < MID_SIZE
    mid = tl.load(Mid, mask=mask, other=0.0).to(tl.float32)
    out = tl.max(mid)
    tl.store(Out, out)


# ---- Min norm (dim reduction) ----
@libentry()
@triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.jit
def min_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_raw = tle.program_id(0).to(tl.int64)
    num_ctas = tl.num_programs(0)
    num_tasks = tl.cdiv(M, BLOCK_M)
    sub_num = tl.cdiv(num_tasks - pid_raw, num_ctas)

    for block_idx in range(sub_num):
        task_idx = pid_raw + num_ctas * block_idx
        pid = task_idx * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        X_ptr = X + pid * N
        Out_ptr = Out + pid
        row_mask = pid < M

        _min = tl.full([BLOCK_M, BLOCK_N], value=float("inf"), dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = (row_mask) & (col_mask)
            a = tl.load(X_ptr + cols, mask, other=float("inf")).to(tl.float32)
            _min = tl.minimum(tl.abs(a), _min)
        min = tl.min(_min, axis=1)
        out = min[:, None]
        tl.store(Out_ptr, out, row_mask)


@libentry()
@triton.jit
def min_norm_kernel_1(X, Mid, M, num_tasks, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0).to(tl.int64)
    num_ctas = tl.num_programs(0)
    sub_num = tl.cdiv(num_tasks - pid, num_ctas)

    for block_idx in range(sub_num):
        task_idx = pid + num_ctas * block_idx
        offset = task_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        X_ptr = X + offset
        mask = offset < M
        x = tl.load(X_ptr, mask=mask, other=float("inf")).to(tl.float32)
        mid = tl.min(tl.abs(x))
        tl.store(Mid + task_idx, mid)


@libentry()
@triton.jit
def min_norm_kernel_2(Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    Mid = Mid + offset
    mask = offset < MID_SIZE
    mid = tl.load(Mid, mask=mask, other=float("inf")).to(tl.float32)
    out = tl.min(mid)
    tl.store(Out, out)


# ---- L0 norm (dim reduction) ----
@libentry()
@triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.jit
def l0_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_raw = tle.program_id(0).to(tl.int64)
    num_ctas = tl.num_programs(0)
    num_tasks = tl.cdiv(M, BLOCK_M)
    sub_num = tl.cdiv(num_tasks - pid_raw, num_ctas)

    for block_idx in range(sub_num):
        task_idx = pid_raw + num_ctas * block_idx
        pid = task_idx * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        X_ptr = X + pid * N
        Out_ptr = Out + pid
        row_mask = pid < M

        _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = (row_mask) & (col_mask)
            a = tl.load(X_ptr + cols, mask, other=0.0).to(tl.float32)
            _sum += (a != 0).to(tl.float32)
        sum = tl.sum(_sum, axis=1)
        out = sum[:, None]
        tl.store(Out_ptr, out, row_mask)


@libentry()
@triton.jit
def l0_norm_kernel_1(X, Mid, M, num_tasks, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0).to(tl.int64)
    num_ctas = tl.num_programs(0)
    sub_num = tl.cdiv(num_tasks - pid, num_ctas)

    for block_idx in range(sub_num):
        task_idx = pid + num_ctas * block_idx
        offset = task_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        X_ptr = X + offset
        mask = offset < M
        x = tl.load(X_ptr, mask=mask, other=0.0).to(tl.float32)
        cnt = (x != 0).to(tl.float32)
        mid = tl.sum(cnt)
        tl.store(Mid + task_idx, mid)


@libentry()
@triton.jit
def l0_norm_kernel_2(Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    Mid = Mid + offset
    mask = offset < MID_SIZE
    mid = tl.load(Mid, mask=mask, other=0.0).to(tl.float32)
    out = tl.sum(mid)
    tl.store(Out, out)


# ---- General Lp norm (dim reduction) ----
@libentry()
@triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.jit(do_not_specialize=["ord"])
def v_norm_kernel(X, Out, M, N, ord, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_raw = tle.program_id(0).to(tl.int64)
    num_ctas = tl.num_programs(0)
    num_tasks = tl.cdiv(M, BLOCK_M)
    sub_num = tl.cdiv(num_tasks - pid_raw, num_ctas)

    for block_idx in range(sub_num):
        task_idx = pid_raw + num_ctas * block_idx
        pid = task_idx * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        X_ptr = X + pid * N
        Out_ptr = Out + pid
        row_mask = pid < M

        _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = (row_mask) & (col_mask)
            a = tl.load(X_ptr + cols, mask, other=0.0).to(tl.float32)
            _sum += pow(tl.abs(a), ord)
        sum = tl.sum(_sum, axis=1)
        out = pow(sum, 1 / ord)[:, None]
        tl.store(Out_ptr, out, row_mask)


@libentry()
@triton.jit(do_not_specialize=["ord"])
def l1_norm_kernel_1(X, Mid, ord, M, num_tasks, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0).to(tl.int64)
    num_ctas = tl.num_programs(0)
    sub_num = tl.cdiv(num_tasks - pid, num_ctas)

    for block_idx in range(sub_num):
        task_idx = pid + num_ctas * block_idx
        offset = task_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        X_ptr = X + offset
        mask = offset < M
        x = tl.load(X_ptr, mask=mask, other=0.0).to(tl.float32)
        mid = tl.sum(pow(tl.abs(x), ord))
        tl.store(Mid + task_idx, mid)


@libentry()
@triton.jit(do_not_specialize=["ord"])
def l1_norm_kernel_2(Mid, Out, ord, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    Mid = Mid + offset
    mask = offset < MID_SIZE
    mid = tl.load(Mid, mask=mask, other=0.0).to(tl.float32)
    out = pow(tl.sum(mid), 1 / ord)
    tl.store(Out, out)


def vector_norm(x, ord=2, dim=None, keepdim=False, dtype=None):
    logger.debug("GEMS_SPACEMIT VECTOR_NORM")
    if dtype is not None:
        dtype = torch.dtype(dtype)
    else:
        dtype = x.dtype
    if dtype not in [torch.float16, torch.float32, torch.bfloat16]:
        raise NotImplementedError(f"vector_norm not implemented for {dtype}")

    with torch_device_fn.device(x.device):
        if (not dim) or len(dim) == x.ndim:
            dim = list(range(x.ndim))
            shape = [1] * x.ndim
            x = dim_compress(x, dim)
            M = x.numel()
            BLOCK_SIZE = triton.next_power_of_2(math.ceil(math.sqrt(M)))
            MID_SIZE = triton.cdiv(M, BLOCK_SIZE)
            BLOCK_MID = triton.next_power_of_2(MID_SIZE)

            num_ctas = min(16, max(1, MID_SIZE))
            mid = torch.empty([MID_SIZE], dtype=dtype, device=x.device)
            out = torch.empty(shape, dtype=dtype, device=x.device)
            if ord == 2:
                l2_norm_kernel_1[(num_ctas,)](x, mid, M, MID_SIZE, BLOCK_SIZE)
                l2_norm_kernel_2[(1,)](mid, out, MID_SIZE, BLOCK_MID)
            elif ord == float("inf"):
                max_norm_kernel_1[(num_ctas,)](x, mid, M, MID_SIZE, BLOCK_SIZE)
                max_norm_kernel_2[(1,)](mid, out, MID_SIZE, BLOCK_MID)
            elif ord == -float("inf"):
                min_norm_kernel_1[(num_ctas,)](x, mid, M, MID_SIZE, BLOCK_SIZE)
                min_norm_kernel_2[(1,)](mid, out, MID_SIZE, BLOCK_MID)
            elif ord == 0:
                l0_norm_kernel_1[(num_ctas,)](x, mid, M, MID_SIZE, BLOCK_SIZE)
                l0_norm_kernel_2[(1,)](mid, out, MID_SIZE, BLOCK_MID)
            else:
                l1_norm_kernel_1[(num_ctas,)](x, mid, ord, M, MID_SIZE, BLOCK_SIZE)
                l1_norm_kernel_2[(1,)](mid, out, ord, MID_SIZE, BLOCK_MID)
        else:
            shape = list(x.shape)
            dim = [d % x.ndim for d in dim]
            x = dim_compress(x, dim)
            N = 1
            for i in dim:
                N *= shape[i]
                shape[i] = 1
            M = x.numel() // N
            out = torch.empty(shape, dtype=dtype, device=x.device)
            grid = lambda META: (min(16, max(1, triton.cdiv(M, META["BLOCK_M"]))),)
            if ord == 2:
                l2_norm_kernel[grid](x, out, M, N)
            elif ord == float("inf"):
                max_norm_kernel[grid](x, out, M, N)
            elif ord == -float("inf"):
                min_norm_kernel[grid](x, out, M, N)
            elif ord == 0:
                l0_norm_kernel[grid](x, out, M, N)
            else:
                v_norm_kernel[grid](x, out, M, N, ord)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
