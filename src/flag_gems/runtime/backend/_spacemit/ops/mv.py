import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def mv_kernel(
    A,
    B,
    C,
    N,
    M,
    stride_an,
    stride_am,
    stride_bm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    sub_num = tl.cdiv(max(num_blocks_n - pid, 0), num_ctas)

    offs_n = tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start_n = task_idx * BLOCK_N
        rows = block_start_n + offs_n
        n_mask = rows < N
        acc_dtype = tl.float32
        acc = tl.zeros((BLOCK_N,), dtype=acc_dtype)

        for m in range(0, M, BLOCK_M):
            cols = m + offs_m
            m_mask = cols < M
            a_ptrs = A + rows[:, None] * stride_an + cols[None, :] * stride_am
            b_ptrs = B + cols * stride_bm
            a = tl.load(a_ptrs, mask=n_mask[:, None] & m_mask[None, :], other=0.0).to(
                acc_dtype
            )
            b = tl.load(b_ptrs, mask=m_mask, other=0.0).to(acc_dtype)
            partial = tl.sum(a * b[None, :], axis=1)
            acc += partial

        c_ptrs = C + rows * stride_cn
        tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=n_mask)


def mv(inp, vec, block_n=32, block_m=32, num_ctas=16):
    logger.debug("GEMS_SPACEMIT MV")
    if inp.stride(0) > 1 and inp.stride(1) > 1:
        inp = inp.contiguous()
    assert inp.shape[1] == vec.shape[0], "incompatible dimensions"
    N, M = inp.shape
    out = torch.empty((N,), device=inp.device, dtype=inp.dtype)
    with torch_device_fn.device(inp.device):
        mv_kernel[(num_ctas,)](
            inp,
            vec,
            out,
            N,
            M,
            inp.stride(0),
            inp.stride(1),
            vec.stride(0),
            out.stride(0),
            BLOCK_N=block_n,
            BLOCK_M=block_m,
            num_ctas=num_ctas,
        )
    return out
