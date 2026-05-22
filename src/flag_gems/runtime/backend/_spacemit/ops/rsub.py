import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)

NUM_CTAS = 8


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("rsub_tensor_tensor"),
    key=["n_elements"],
)
@triton.jit
def rsub_kernel_tt(
    A_ptr,
    B_ptr,
    Out_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """rsub: B * alpha - A (reverse subtract)."""
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)

    total_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(total_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        a_blk = tl.make_block_ptr(
            base=A_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        b_blk = tl.make_block_ptr(
            base=B_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        out_blk = tl.make_block_ptr(
            base=Out_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )

        a = tl.load(a_blk, boundary_check=(0,))
        b = tl.load(b_blk, boundary_check=(0,))
        out = b * alpha - a
        tl.store(out_blk, out.to(b.dtype), boundary_check=(0,))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("rsub_tensor_scalar"),
    key=["n_elements"],
)
@triton.jit
def rsub_kernel_ts(
    A_ptr,
    Out_ptr,
    scalar,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """rsub: scalar * alpha - tensor."""
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)

    total_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(total_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        a_blk = tl.make_block_ptr(
            base=A_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        out_blk = tl.make_block_ptr(
            base=Out_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )

        a = tl.load(a_blk, boundary_check=(0,))
        out = scalar * alpha - a
        tl.store(out_blk, out.to(a.dtype), boundary_check=(0,))


def rsub(A, B, *, alpha=1):
    """rsub(A, B, alpha) = B * alpha - A"""
    logger.debug("GEMS_SPACEMIT RSUB")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        A, B = A.contiguous(), B.contiguous()
        out = torch.empty_like(A)
        n = A.numel()
        rsub_kernel_tt[(NUM_CTAS,)](A, B, out, alpha, n)
        return out
    elif isinstance(A, torch.Tensor):
        # rsub(tensor, scalar, alpha) = scalar * alpha - tensor
        A = A.contiguous()
        out = torch.empty_like(A)
        n = A.numel()
        rsub_kernel_ts[(NUM_CTAS,)](A, out, B, alpha, n)
        return out
    else:
        return torch.tensor(B * alpha - A)
