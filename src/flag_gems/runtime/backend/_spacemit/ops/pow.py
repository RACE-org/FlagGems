import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry
from flag_gems.utils import libtuner
from flag_gems.utils import tl_extra_shim

_pow = tl_extra_shim.pow
logger = logging.getLogger(__name__)
NUM_CTAS = 8


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("pow_tensor_tensor"),
    key=["n_elements"],
)
@triton.jit
def pow_kernel_tt(
    A_ptr,
    B_ptr,
    Out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
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

        a = tl.load(a_blk, boundary_check=(0,)).to(tl.float32)
        b = tl.load(b_blk, boundary_check=(0,)).to(tl.float32)
        out = _pow(a, b)
        tl.store(out_blk, out.to(Out_ptr.type.element_ty), boundary_check=(0,))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("pow_tensor_scalar"),
    key=["n_elements"],
)
@triton.jit
def pow_kernel_ts(
    A_ptr,
    Out_ptr,
    exponent,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
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

        a = tl.load(a_blk, boundary_check=(0,)).to(tl.float32)
        out = _pow(a, exponent.to(tl.float32))
        tl.store(out_blk, out.to(Out_ptr.type.element_ty), boundary_check=(0,))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("pow_scalar_tensor"),
    key=["n_elements"],
)
@triton.jit
def pow_kernel_st(
    B_ptr,
    Out_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)

    total_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(total_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

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

        b = tl.load(b_blk, boundary_check=(0,)).to(tl.float32)
        out = _pow(scalar.to(tl.float32), b)
        tl.store(out_blk, out.to(Out_ptr.type.element_ty), boundary_check=(0,))


def _normalize_tensor_operand(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.bool:
        return x.to(torch.int64)
    return x


def _normalize_scalar_operand(x):
    if isinstance(x, bool):
        return int(x)
    return x


def _pow_out_dtype(x: torch.Tensor, exponent=None) -> torch.dtype:
    if x.dtype.is_floating_point:
        return x.dtype
    if exponent is not None and isinstance(exponent, torch.Tensor) and exponent.dtype.is_floating_point:
        return exponent.dtype
    return torch.float32


def pow_tensor_tensor(A, exponent):
    logger.debug("GEMS_SPACEMIT POW_TENSOR_TENSOR")
    A = _normalize_tensor_operand(A).contiguous()
    exponent = _normalize_tensor_operand(exponent).contiguous()
    out = torch.empty(A.shape, dtype=_pow_out_dtype(A, exponent), device=A.device)
    n = A.numel()
    pow_kernel_tt[(NUM_CTAS,)](A, exponent, out, n)
    return out


def pow_tensor_tensor_(A, exponent):
    logger.debug("GEMS_SPACEMIT POW_TENSOR_TENSOR_")
    result = pow_tensor_tensor(A, exponent)
    A.copy_(result.to(A.dtype))
    return A


def pow_tensor_scalar(A, exponent):
    logger.debug("GEMS_SPACEMIT POW_TENSOR_SCALAR")
    A = _normalize_tensor_operand(A).contiguous()
    exponent = _normalize_scalar_operand(exponent)
    out = torch.empty(A.shape, dtype=_pow_out_dtype(A), device=A.device)
    n = A.numel()
    pow_kernel_ts[(NUM_CTAS,)](A, out, float(exponent), n)
    return out


def pow_tensor_scalar_(A, exponent):
    logger.debug("GEMS_SPACEMIT POW_TENSOR_SCALAR_")
    result = pow_tensor_scalar(A, exponent)
    A.copy_(result.to(A.dtype))
    return A


def pow_scalar(A, exponent):
    logger.debug("GEMS_SPACEMIT POW_SCALAR")
    A = _normalize_scalar_operand(A)
    exponent = _normalize_tensor_operand(exponent).contiguous()
    out = torch.empty(exponent.shape, dtype=_pow_out_dtype(exponent), device=exponent.device)
    n = exponent.numel()
    pow_kernel_st[(NUM_CTAS,)](exponent, out, float(A), n)
    return out
