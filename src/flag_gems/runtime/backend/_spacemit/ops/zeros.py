import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.shape_utils import volume

device_ = device
logger = logging.getLogger(__name__)

BLOCK_SIZE = 4096
SUB_BLOCK_SIZE = 128


@libentry()
@triton.jit
def zeros_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    SUB_BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    fill_dtype = output_ptr.dtype.element_ty
    if fill_dtype == tl.int1:
        fill_dtype = tl.int8
    value = tl.full((SUB_BLOCK_SIZE,), 0, dtype=fill_dtype)

    for sub_offset in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
        offsets = block_start + sub_offset + tl.arange(0, SUB_BLOCK_SIZE)
        mask = offsets < n_elements
        tl.store(output_ptr + offsets, value, mask=mask)


def zeros(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS_SPACEMIT ZEROS")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)

    out = torch.empty(size, device=device, dtype=dtype)
    N = volume(size)
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    with torch_device_fn.device(device):
        zeros_kernel[grid](out, N, BLOCK_SIZE, SUB_BLOCK_SIZE)
    return out
