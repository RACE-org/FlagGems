import logging

import torch
import triton

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._tsingmicro.ops.rand import (
    NUM_TILES,
    UNROLL,
    rand_kernel,
)
from flag_gems.runtime.backend._tsingmicro.utils.random_utils import (
    philox_backend_seed_offset,
)

logger = logging.getLogger(__name__)


def rand_like(
    x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    # Thin wrapper — reuses the locally-optimized rand_kernel (grid=(16,),
    # inner runtime loop, BLOCK heuristic). Only difference vs rand() is
    # that output shape/device/dtype default to the input tensor.
    logger.debug("GEMS RAND_LIKE")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(increment)
    grid = (NUM_TILES,)
    with torch_device_fn.device(x.device):
        rand_kernel[grid](out, N, philox_seed, philox_offset)
    return out
