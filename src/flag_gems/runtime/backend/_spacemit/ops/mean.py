import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle


@libentry()
@triton.jit
def mean_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    # Vectorized load with block_ptr for better spine-triton codegen
    pid = tle.program_id(0)
    offset_start = pid * BLOCK_SIZE

    inp_ptr = tl.make_block_ptr(
        base=inp,
        shape=[M],
        strides=[1],
        offsets=[offset_start],
        block_shape=[BLOCK_SIZE],
        order=[0],
    )

    v = tl.load(inp_ptr, boundary_check=[0], padding_option="zero").to(tl.float32)
    sum_val = tl.sum(v, axis=0)
    tl.store(mid + pid, sum_val)


@libentry()
@triton.jit
def mean_kernel_2(mid, out, M, MID_SIZE, BLOCK_MID: tl.constexpr):
    mid_ptr = tl.make_block_ptr(
        base=mid,
        shape=[MID_SIZE],
        strides=[1],
        offsets=[0],
        block_shape=[BLOCK_MID],
        order=[0],
    )
    mid_val = tl.load(mid_ptr, boundary_check=[0], padding_option="zero").to(tl.float32)
    sum_val = tl.sum(mid_val, axis=0) / M
    tl.store(out, sum_val.to(out.dtype.element_ty))


def mean(inp, *, dtype=None):
    logging.debug("GEMS_SPACEMIT MEAN")
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype

    # Use gems' standard two-stage design: many programs * BLOCK_SIZE each
    # Gems baseline strategy: block_size = sqrt(M), mid_size programs
    import math
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=torch.float32, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        # Stage 1: parallel reduction across mid_size programs
        mean_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
        # Stage 2: final reduction in single program (fully on device)
        mean_kernel_2[(1, 1, 1)](mid, out, M, mid_size, block_mid)
    return out


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("mean"),
    key=["M", "N"],
)
@triton.jit
def mean_dim_kernel(X, Mean, M, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    X = X + row * N
    Mean = Mean + row
    _mean = 0.0

    num_pid_n = tl.cdiv(N, BLOCK_N)

    x_ptr_desc = tl.make_block_ptr(
        base=X,
        shape=[N],
        strides=[1],
        offsets=[0],
        block_shape=[BLOCK_N],
        order=[0],
    )

    for off_n in range(0, num_pid_n):
        a = tl.load(x_ptr_desc, boundary_check=[0])
        _mean += tl.sum(a)
        x_ptr_desc = tl.advance(x_ptr_desc, [BLOCK_N])

    mean = _mean / N
    tl.store(Mean, mean.to(Mean.dtype.element_ty))


def mean_dim(x, dim, keepdim=False, *, dtype=None):
    logging.debug("GEMS_SPACEMIT MEAN_DIM")

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
    grid = (M,)
    with torch_device_fn.device(x.device):
        mean_dim_kernel[grid](x, out, M, N)
    if not keepdim:
        out = out.squeeze(dim)
    return out


def avg_pool2d(x, kernel_size=None, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    return mean_dim(x, dim=[2, 3], keepdim=True)
