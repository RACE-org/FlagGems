import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry, libtuner

logger = logging.getLogger(__name__)

NUM_CTAS = 8


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("masked_fill"),
    key=["n_elements"],
)
@triton.jit
def masked_fill_kernel(
    inp_ptr,
    mask_ptr,
    out_ptr,
    value,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Masked fill using make_block_ptr + boundary_check."""
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)

    total_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(total_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        inp_blk = tl.make_block_ptr(
            base=inp_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        mask_blk = tl.make_block_ptr(
            base=mask_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        out_blk = tl.make_block_ptr(
            base=out_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )

        inp = tl.load(inp_blk, boundary_check=(0,))
        mask = tl.load(mask_blk, boundary_check=(0,))
        mask_bool = mask != 0
        result = tl.where(mask_bool, value, inp)
        tl.store(out_blk, result, boundary_check=(0,))


def masked_fill(inp, mask, value):
    logger.debug("GEMS_SPACEMIT MASKED_FILL")
    assert (
        (torch.is_tensor(value) and value.ndim == 0)
        or isinstance(value, int)
        or isinstance(value, float)
    ), "masked_fill_ only supports a 0-dimensional value tensor"
    if torch.is_tensor(value):
        value = value.item()
    assert broadcastable_to(
        mask.shape, inp.shape
    ), "The shape of mask must be broadcastable with the shape of the underlying tensor"

    if inp.ndim == 0:
        return (
            torch.tensor(value, dtype=inp.dtype, device=inp.device)
            if mask.item()
            else inp.clone()
        )

    expand_mask = mask.expand(inp.shape).to(torch.uint8).contiguous()
    inp = inp.contiguous()
    out = torch.empty_like(inp)
    n = inp.numel()
    with torch_device_fn.device(inp.device):
        masked_fill_kernel[(NUM_CTAS,)](inp, expand_mask, out, value, n)
    return out


def masked_fill_(inp, mask, value):
    logger.debug("GEMS_SPACEMIT MASKED_FILL_")
    assert (
        (torch.is_tensor(value) and value.ndim == 0)
        or isinstance(value, int)
        or isinstance(value, float)
    ), "masked_fill_ only supports a 0-dimensional value tensor"
    if torch.is_tensor(value):
        value = value.item()
    assert broadcastable_to(
        mask.shape, inp.shape
    ), "The shape of mask must be broadcastable with the shape of the underlying tensor"

    if inp.ndim == 0:
        if mask.item():
            inp[()] = value
        return inp

    expand_mask = mask.expand(inp.shape).to(torch.uint8).contiguous()
    inp_contig = inp.contiguous()
    n = inp_contig.numel()
    with torch_device_fn.device(inp.device):
        masked_fill_kernel[(NUM_CTAS,)](
            inp_contig, expand_mask, inp_contig, value, n
        )
    if not inp.is_contiguous():
        inp.copy_(inp_contig)
    return inp
