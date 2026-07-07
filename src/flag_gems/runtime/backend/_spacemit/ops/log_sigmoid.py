import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)

NUM_CTAS = 8


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("log_sigmoid"),
    key=["n_elements"],
)
@triton.jit
def log_sigmoid_kernel(
    X_ptr,
    Out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """log_sigmoid(x) = -log(1 + exp(-x)) = x - softplus(x)
    For numerical stability: log_sigmoid(x) = min(0, x) - log(1 + exp(-|x|))
    """
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    total_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(total_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        x_blk = tl.make_block_ptr(
            base=X_ptr,
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

        x = tl.load(x_blk, boundary_check=(0,))
        x_dtype = x.dtype
        x = x.to(tl.float32)
        min_val = tl.minimum(x, 0.0)
        abs_x = tl.abs(x)
        # Use native tl.log and tl.exp to avoid isfinite issues
        out = min_val - tl.log(1.0 + tl.exp(-abs_x))
        tl.store(out_blk, out.to(x_dtype), boundary_check=(0,))


def log_sigmoid(input):
    logger.debug("GEMS_SPACEMIT LOG_SIGMOID")
    input = input.contiguous()
    out = torch.empty_like(input)
    buffer = torch.empty_like(input)
    n = input.numel()
    with torch_device_fn.device(input.device):
        log_sigmoid_kernel[(NUM_CTAS,)](input, out, n)
    return out, buffer
