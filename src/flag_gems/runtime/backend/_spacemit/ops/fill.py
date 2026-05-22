import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)

NUM_CTAS = 8

_TORCH_TO_TRITON_DTYPE = {
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
    torch.int8: tl.int8,
    torch.int16: tl.int16,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
    torch.bool: tl.int8,
    torch.bfloat16: tl.bfloat16,
}


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("fill"),
    key=["n_elements"],
)
@triton.jit
def fill_kernel(
    Out_ptr,
    value,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    total_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(total_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        out_blk = tl.make_block_ptr(
            base=Out_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )

        val = tl.full([BLOCK_SIZE], value=value, dtype=DTYPE)
        tl.store(out_blk, val, boundary_check=(0,))


def fill_scalar(input, value):
    logger.debug("GEMS_SPACEMIT FILL_SCALAR")
    out = torch.empty_like(input)
    n = out.numel()
    dtype = _TORCH_TO_TRITON_DTYPE.get(input.dtype, tl.float32)
    with torch_device_fn.device(input.device):
        fill_kernel[(NUM_CTAS,)](out, value, n, DTYPE=dtype)
    return out


def fill_tensor(input, value):
    if not value.is_cuda:
        return fill_scalar(input, value.item())
    logger.debug("GEMS_SPACEMIT FILL_TENSOR")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    return fill_scalar(input, value.item())


def fill_scalar_(self, value):
    logger.debug("GEMS_SPACEMIT FILL_SCALAR_")
    n = self.numel()
    self_contig = self.contiguous()
    dtype = _TORCH_TO_TRITON_DTYPE.get(self.dtype, tl.float32)
    with torch_device_fn.device(self.device):
        fill_kernel[(NUM_CTAS,)](self_contig, value, n, DTYPE=dtype)
    if not self.is_contiguous():
        self.copy_(self_contig)
    return self


def fill_tensor_(self, value):
    if not value.is_cuda:
        return fill_scalar_(self, value.item())
    logger.debug("GEMS_SPACEMIT FILL_TENSOR_")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    return fill_scalar_(self, value.item())
