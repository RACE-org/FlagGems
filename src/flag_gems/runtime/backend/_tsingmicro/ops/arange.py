import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

TOTAL_CORE_NUM = 16
_FULL_TILE_BLOCK = 65536

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def arange_func(y_ptr, start, end, step, size, BLOCK: tl.constexpr):
    pid = tle.program_id(0)
    y_ptr += pid * BLOCK
    step_offset = pid * BLOCK * step

    cols = tl.arange(0, BLOCK)
    arange_val = cols * step + step_offset + start
    mask = cols + pid * BLOCK
    tl.store(y_ptr + cols, arange_val, mask=mask < size)


def _pick_block(size):
    # Per-CTA element count aimed at TOTAL_CORE_NUM CTAs (≈ one per tile),
    # capped at _FULL_TILE_BLOCK so vector width stays inside SPM budget.
    per_tile = max(triton.cdiv(size, TOTAL_CORE_NUM), 1)
    block = 1
    while block * 2 <= per_tile and block * 2 <= _FULL_TILE_BLOCK:
        block *= 2
    return block


def arange_start(
    start, end, step=1, *, dtype=None, layout=None, device=None, pin_memory=None
):
    logger.debug("GEMS_TSINGMICRO ARANGE")
    if dtype is torch.int64:
        sgn = (step > 0) - (step < 0)
        size = (end - start + step - sgn) // step
    else:
        size = math.ceil((end - start) / step)

    if dtype is None:
        dtype = torch.int64

    if pin_memory is None:
        pin_memory = False

    if device is None:
        device = runtime.device.name

    result = torch.empty((size,), device=device, dtype=dtype, pin_memory=pin_memory)
    BLOCK = _pick_block(size)
    grid = (triton.cdiv(size, BLOCK),)
    arange_func[grid](result, start, end, step, size, BLOCK)
    return result


def arange(end, *, dtype=None, layout=None, device=None, pin_memory=None):
    return arange_start(
        0, end, 1, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )
