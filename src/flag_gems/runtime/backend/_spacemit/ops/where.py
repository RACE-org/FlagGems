import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry
from flag_gems.utils import libtuner

logger = logging.getLogger(__name__)

NUM_CTAS = 8


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("where"),
    key=["n_elements"],
)
@triton.jit
def where_kernel(
    Cond_ptr,
    A_ptr,
    B_ptr,
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

        cond_blk = tl.make_block_ptr(
            base=Cond_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        a_blk = tl.make_block_ptr(
            base=A_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        b_blk = tl.make_block_ptr(
            base=B_ptr,
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

        cond = tl.load(cond_blk, boundary_check=(0,))
        a = tl.load(a_blk, boundary_check=(0,))
        b = tl.load(b_blk, boundary_check=(0,))
        out = tl.where(cond, a, b)
        tl.store(out_blk, out, boundary_check=(0,))


def where_self_out(condition, self, other, out=None):
    logger.debug("GEMS_SPACEMIT WHERE_SELF_OUT")
    result_type = torch.result_type(self, other)
    if out is not None:
        assert (
            out.dtype == result_type
        ), f"Expected out type to be {result_type}, but got {out.dtype}."

    c, a, b = list(
        map(
            lambda x: x if isinstance(x, torch.Tensor) else torch.tensor(x),
            (condition, self, other),
        )
    )

    if a.dtype != result_type:
        a = a.to(result_type)
    if b.dtype != result_type:
        b = b.to(result_type)

    devices = [x.device for x in (c, a, b)]

    assert all(device.type == "cpu" for device in devices), (
        "CPU only. Expected all tensors to be on CPU, " f"but found devices {devices}"
    )

    device = devices[0]
    if c.device != device and c.ndim == 0:
        c = c.to(device)
    if a.device != device and a.ndim == 0:
        a = a.to(device)
    if b.device != device and b.ndim == 0:
        b = b.to(device)

    assert (
        len(set(devices)) == 1
    ), f"Expected all tensors to be on the same device, but found at least two devices, {devices}"
    assert (
        c.dtype == torch.bool
    ), f"where expected condition to be a boolean tensor, but got a tensor with dtype {condition.dtype}"

    out_shape = torch.broadcast_shapes(c.shape, a.shape, b.shape)
    # Broadcast all inputs to the same shape and make contiguous
    c = c.broadcast_to(out_shape).contiguous()
    a = a.broadcast_to(out_shape).contiguous()
    b = b.broadcast_to(out_shape).contiguous()

    if out is None:
        out = torch.empty(out_shape, dtype=result_type, device=device)

    n_elements = out.numel()
    where_kernel[(NUM_CTAS,)](c, a, b, out, n_elements)
    return out


def where_self(condition, self, other):
    logger.debug("GEMS_SPACEMIT WHERE_SELF")
    return where_self_out(condition, self, other)


def where_scalar_self(condition, self, other):
    logger.debug("GEMS_SPACEMIT WHERE_SCALAR_SELF")
    return where_self_out(condition, self, other)


def where_scalar_other(condition, self, other):
    logger.debug("GEMS_SPACEMIT WHERE_SCALAR_OTHER")
    return where_self_out(condition, self, other)
