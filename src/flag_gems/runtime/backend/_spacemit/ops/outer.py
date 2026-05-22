import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.ops import mul
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _outer_flat_kernel(
    lhs,
    rhs,
    out,
    total,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    lhs_offsets = offsets // N
    rhs_offsets = offsets - lhs_offsets * N

    lhs_vals = tl.load(lhs + lhs_offsets, mask=mask)
    rhs_vals = tl.load(rhs + rhs_offsets, mask=mask)
    out_vals = lhs_vals * rhs_vals
    tl.store(out + offsets, out_vals, mask=mask)


def _spacemit_outer_flat(lhs, rhs):
    m = lhs.shape[0]
    n = rhs.shape[0]
    out = torch.empty((m, n), dtype=lhs.dtype, device=lhs.device)
    total = m * n

    grid = lambda META: (triton.cdiv(total, META["BLOCK_SIZE"]),)
    _outer_flat_kernel[grid](
        lhs,
        rhs,
        out,
        total,
        n,
        BLOCK_SIZE=1024,
    )
    return out


def _spacemit_outer(lhs, rhs):
    """Dedicated flat-store outer kernel for SpacemiT A100.

    Use a 1D output-flattened schedule instead of a 2D block-pointer schedule,
    so stores are contiguous and the runtime sees many independent tasks.
    """
    if lhs.is_contiguous() and rhs.is_contiguous():
        return _spacemit_outer_flat(lhs, rhs)

    lhs2 = lhs[:, None].contiguous()
    rhs2 = rhs[None, :].contiguous()
    with torch_device_fn.device(lhs.device):
        return mul(lhs2, rhs2)


class Outer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, weight):
        print("[DEBUG] ⭐ SPACEMIT OUTER CALLED!")
        logger.debug("GEMS_SPACEMIT OUTER")
        assert inp.ndim == 1 and weight.ndim == 1, "Invalid input"
        out = _spacemit_outer(inp, weight)
        ctx.save_for_backward(inp, weight)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("GEMS_SPACEMIT OUTER VJP")
        assert out_grad.ndim == 2, "invalid out_grad shape"
        from flag_gems.runtime.backend._spacemit.ops.mv import mv

        inp, weight = ctx.saved_tensors
        inp_grad = mv(out_grad, weight)
        weight_grad = mv(out_grad.t().contiguous(), inp)
        return inp_grad, weight_grad


def outer(inp, weight):
    return Outer.apply(inp, weight)
