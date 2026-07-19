import logging
from enum import Enum

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, pointwise_dynamic
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

TOTAL_CORE_NUM = 16
_FULL_TILE_BLOCK = 65536


def _pick_block_size(per_tile):
    block = 1
    while block * 2 <= per_tile and block * 2 <= _FULL_TILE_BLOCK:
        block *= 2
    return max(block, 64)


@libentry()
@triton.jit
def mse_sum_kernel(inp, target, mid, M, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0)
    per_tile = tl.cdiv(M, TOTAL_CORE_NUM)
    start = pid * per_tile

    acc = tl.zeros([1], dtype=tl.float32)
    for off in range(0, per_tile, BLOCK_SIZE):
        idx = start + off + tl.arange(0, BLOCK_SIZE)
        # Mask must bound BOTH the global tensor end (M) AND the CTA's
        # assigned range (start + per_tile).  Without the per_tile bound,
        # the last inner iteration of each CTA reads past its partition
        # into the next CTA's region, double-counting up to
        # (BLOCK_SIZE - per_tile % BLOCK_SIZE) elements per CTA pair.
        mask = idx < tl.minimum(M, start + per_tile)
        inp_val = tl.load(inp + idx, mask=mask, other=0.0).to(tl.float32)
        target_val = tl.load(target + idx, mask=mask, other=0.0).to(tl.float32)
        diff = inp_val - target_val
        acc += tl.sum(diff * diff)

    tl.store(mid + pid, tl.sum(acc))


@libentry()
@triton.jit
def final_reduce_kernel(mid, out, M, reduction: tl.constexpr):
    idx = tl.arange(0, TOTAL_CORE_NUM)
    mid_val = tl.load(mid + idx).to(tl.float32)
    total = tl.sum(mid_val)
    if reduction == 1:  # MEAN
        total = total / M
    tl.store(out, total.to(out.dtype.element_ty))


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def func(x, y):
    return (x - y) * (x - y)


class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2


def mse_loss(inp, target, reduction=Reduction.MEAN.value):
    logger.debug("GEMS TSINGMICRO MSE LOSS")
    if reduction == Reduction.NONE.value:
        return func(inp, target)

    inp = inp.contiguous()
    target = target.contiguous()
    M = inp.numel()
    dtype = inp.dtype

    per_tile = triton.cdiv(M, TOTAL_CORE_NUM)
    block_size = _pick_block_size(per_tile)

    mid = torch.empty((TOTAL_CORE_NUM,), dtype=torch.float32, device=inp.device)
    out = torch.empty([], dtype=torch.float32, device=inp.device)

    with torch_device_fn.device(inp.device):
        mse_sum_kernel[(TOTAL_CORE_NUM, 1, 1)](inp, target, mid, M, block_size)
        final_reduce_kernel[(1, 1, 1)](mid, out, M, reduction)
    return out.to(dtype)
