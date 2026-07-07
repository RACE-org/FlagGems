import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libtuner

logger = logging.getLogger(__name__)

NUM_CTAS = 8

# A100: L1=32KB — 1024 elements is a good fit
@libtuner(
    configs=runtime.get_tuned_config("to"),
    key=["n_elements"],
)
@triton.jit
def to_dtype_kernel(
    In_ptr,
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

        in_blk = tl.make_block_ptr(
            base=In_ptr,
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

        x = tl.load(in_blk, boundary_check=(0,))
        x = x.to(Out_ptr.type.element_ty)
        tl.store(out_blk, x, boundary_check=(0,))


def to_dtype(x, dtype, non_blocking=False, copy=False, memory_format=None):
    logger.debug("GEMS_SPACEMIT TO_DTYPE")
    if not torch.is_tensor(x):
        return x
    if not copy and x.dtype == dtype:
        return x
    x = x.contiguous()
    out = torch.empty_like(x, dtype=dtype, memory_format=memory_format)
    n = x.numel()
    with torch_device_fn.device(x.device):
        to_dtype_kernel[(NUM_CTAS,)](x, out, n)
    return out
