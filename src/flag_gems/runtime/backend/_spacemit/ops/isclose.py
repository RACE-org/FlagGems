import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(__name__)

NUM_CTAS = 8

_isfinited = tl_extra_shim.isfinited
_finitef = tl_extra_shim.finitef


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("isclose"),
    key=["n_elements"],
)
@triton.jit
def isclose_kernel(
    A_ptr,
    B_ptr,
    Out_ptr,
    n_elements,
    rtol,
    atol,
    equal_nan: tl.constexpr,
    zero_tol: tl.constexpr,
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
            base=A_ptr, shape=(n_elements,), strides=(1,),
            offsets=(block_start,), block_shape=(BLOCK_SIZE,), order=(0,),
        )
        b_blk = tl.make_block_ptr(
            base=B_ptr, shape=(n_elements,), strides=(1,),
            offsets=(block_start,), block_shape=(BLOCK_SIZE,), order=(0,),
        )
        out_blk = tl.make_block_ptr(
            base=Out_ptr, shape=(n_elements,), strides=(1,),
            offsets=(block_start,), block_shape=(BLOCK_SIZE,), order=(0,),
        )

        x = tl.load(a_blk, boundary_check=(0,))
        y = tl.load(b_blk, boundary_check=(0,))

        cast_x = x if x.dtype.is_fp64() else x.to(tl.float32)
        cast_y = y if x.dtype.is_fp64() else y.to(tl.float32)
        if x.dtype.is_bf16():
            close = cast_x == cast_y
        else:
            close = x == y
        if equal_nan:
            close |= (cast_x != cast_x) & (cast_y != cast_y)
        if not zero_tol:
            allowed = atol + tl.abs(rtol * cast_y)
            actual = tl.abs(cast_x - cast_y)
            actual_finite = _isfinited(actual) if x.dtype.is_fp64() else _finitef(actual)
            close |= actual_finite.to(tl.int1) & (actual <= allowed)

        tl.store(out_blk, close.to(Out_ptr.type.element_ty), boundary_check=(0,))


def isclose(
    A,
    B,
    rtol=1e-05,
    atol=1e-08,
    equal_nan=False,
):
    logger.debug("GEMS_SPACEMIT ISCLOSE")
    if A.dtype == torch.bool:
        return A == B
    if A.dtype != B.dtype:
        raise RuntimeError(f"{A.dtype} did not match {B.dtype}")
    if A.is_quantized or B.is_quantized:
        raise RuntimeError("isclose is not supported for quantized inputs.")
    if rtol < 0:
        raise RuntimeError(f"rtol must be >= 0, got {rtol}")
    if atol < 0:
        raise RuntimeError(f"atol must be >= 0, got {atol}")
    zero_tol = (rtol == 0) and (atol == 0)

    A, B = A.contiguous(), B.contiguous()
    out = torch.empty(A.shape, dtype=torch.uint8, device=A.device)
    n = A.numel()
    isclose_kernel[(NUM_CTAS,)](A, B, out, n, float(rtol), float(atol), equal_nan, zero_tol)
    return out.view(torch.bool)


def allclose(
    A,
    B,
    rtol=1e-05,
    atol=1e-08,
    equal_nan=False,
):
    logger.debug("GEMS_SPACEMIT ALLCLOSE")
    from .all import all
    return all(isclose(A, B, rtol, atol, equal_nan)).item()
