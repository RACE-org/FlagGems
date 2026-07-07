import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.shape_utils import volume

device_ = device
logger = logging.getLogger(__name__)

NUM_CTAS = 8


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ones"),
    key=["n_elements"],
)
@triton.jit
def ones_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_CTAS_K: tl.constexpr = NUM_CTAS,
):
    pid = tl.program_id(axis=0)
    total_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(total_blocks - pid, 0), NUM_CTAS_K)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + NUM_CTAS_K * block_idx
        block_start = task_idx * BLOCK_SIZE

        output_block_ptr = tl.make_block_ptr(
            base=output_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        fill_dtype = output_ptr.dtype.element_ty
        if fill_dtype == tl.int1:
            fill_dtype = tl.int8
        value = tl.full((BLOCK_SIZE,), 1, dtype=fill_dtype)

        tl.store(
            output_block_ptr,
            value,
            boundary_check=(0,),
        )


def ones(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS_SPACEMIT ONES")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)

    out = torch.empty(size, device=device, dtype=dtype)
    N = volume(size)
    with torch_device_fn.device(device):
        ones_kernel[(NUM_CTAS,)](out, N)
    return out
