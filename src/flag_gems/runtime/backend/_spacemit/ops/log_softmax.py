import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.ops.log_softmax import log_softmax as common_log_softmax
from flag_gems.utils import libentry
from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(__name__)
exp = tl_extra_shim.exp
log = tl_extra_shim.log


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("log_softmax_spacemit"), key=["n_rows", "n_cols"]
)
@triton.heuristics({"ONE_TILE_PER_ROW": lambda args: args["COL_SIZE"] >= args["n_cols"]})
@triton.jit
def log_softmax_kernel_spacemit(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    ROW_SIZE: tl.constexpr,
    COL_SIZE: tl.constexpr,
    ONE_TILE_PER_ROW: tl.constexpr,
):
    element_ty = output_ptr.type.element_ty
    row_start = tl.program_id(0) * ROW_SIZE
    for row_idx in range(row_start, row_start + ROW_SIZE):
        if row_idx < n_rows:
            if ONE_TILE_PER_ROW:
                # Fast path for short rows: keep the reduction 1-D, but batch
                # ROW_SIZE rows per program to reduce program dispatch overhead.
                input_block_ptr = tl.make_block_ptr(
                    base=input_ptr + row_idx * input_row_stride,
                    shape=(n_cols,),
                    strides=(1,),
                    offsets=(0,),
                    block_shape=(COL_SIZE,),
                    order=(0,),
                )
                output_block_ptr = tl.make_block_ptr(
                    base=output_ptr + row_idx * output_row_stride,
                    shape=(n_cols,),
                    strides=(1,),
                    offsets=(0,),
                    block_shape=(COL_SIZE,),
                    order=(0,),
                )
                row = tl.load(
                    input_block_ptr, boundary_check=(0,), padding_option="neg_inf"
                ).to(tl.float32)
                row_max = tl.max(row, axis=0)
                numerator = exp(row - row_max)
                denominator = tl.sum(numerator, axis=0)
                log_denom = log(denominator)
                result = (row - row_max - log_denom).to(element_ty)
                tl.store(output_block_ptr, result, boundary_check=(0,))
            else:
                # Two-pass online log_softmax for large n_cols.
                row_max_total = tl.full((1,), value=-float("inf"), dtype=tl.float32)
                slow_denominator = tl.zeros((1,), dtype=tl.float32)

                for col_idx in range(0, n_cols, COL_SIZE):
                    input_block_ptr = tl.make_block_ptr(
                        base=input_ptr + row_idx * input_row_stride,
                        shape=(n_cols,),
                        strides=(1,),
                        offsets=(col_idx,),
                        block_shape=(COL_SIZE,),
                        order=(0,),
                    )
                    row = tl.load(
                        input_block_ptr, boundary_check=(0,), padding_option="neg_inf"
                    ).to(tl.float32)
                    block_max = tl.max(row, axis=0)
                    new_max = tl.maximum(row_max_total, block_max)
                    slow_denominator = slow_denominator * exp(row_max_total - new_max)
                    slow_denominator += tl.sum(exp(row - new_max), axis=0)
                    row_max_total = new_max

                slow_log_denom = log(slow_denominator)

                for col_idx in range(0, n_cols, COL_SIZE):
                    input_block_ptr = tl.make_block_ptr(
                        base=input_ptr + row_idx * input_row_stride,
                        shape=(n_cols,),
                        strides=(1,),
                        offsets=(col_idx,),
                        block_shape=(COL_SIZE,),
                        order=(0,),
                    )
                    output_block_ptr = tl.make_block_ptr(
                        base=output_ptr + row_idx * output_row_stride,
                        shape=(n_cols,),
                        strides=(1,),
                        offsets=(col_idx,),
                        block_shape=(COL_SIZE,),
                        order=(0,),
                    )
                    row = tl.load(
                        input_block_ptr, boundary_check=(0,), padding_option="neg_inf"
                    ).to(tl.float32)
                    result = (row - row_max_total - slow_log_denom).to(element_ty)
                    tl.store(output_block_ptr, result, boundary_check=(0,))


def _spacemit_log_softmax_lastdim(inp, out):
    n_rows, n_cols = inp.shape
    grid = lambda meta: (triton.cdiv(n_rows, meta["ROW_SIZE"]),)
    with torch_device_fn.device(inp.device):
        log_softmax_kernel_spacemit[grid](
            out,
            inp,
            inp.stride(0),
            out.stride(0),
            n_rows,
            n_cols,
        )


def log_softmax(self, dim, half_to_float=False):
    logger.debug("GEMS_SPACEMIT LOG_SOFTMAX")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim

    if dim != self.ndim - 1:
        return common_log_softmax(self, dim, half_to_float)

    if half_to_float:
        dtype = torch.float32
    else:
        dtype = self.dtype

    inp = self.contiguous()

    n_cols = inp.shape[-1]
    n_rows = inp.numel() // n_cols
    inp_2d = inp.view(n_rows, n_cols)
    out_2d = torch.empty_like(inp_2d, dtype=dtype)
    _spacemit_log_softmax_lastdim(inp_2d, out_2d)
    return out_2d.view_as(inp)
