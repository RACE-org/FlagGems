import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device
from flag_gems.utils import broadcastable_to, pointwise_dynamic
from flag_gems.utils.codegen_config_utils import CodeGenConfig

my_config = CodeGenConfig(
    max_tile_size=65536,
    max_grid_size=(16, 16, 16),
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=False,
)

logger = logging.getLogger(__name__)
device = device.name


@pointwise_dynamic(
    is_tensor=[True, True, False],
    promotion_methods=[(0, "NO_OPMATH")],
    config=my_config,
)
@triton.jit
def masked_fill_kernel(inp, expand_mask, value):
    # expand_mask is guaranteed torch.bool by the host wrapper → tl.int1 in
    # triton.  tl.where with a tl.int1 condition maps directly to tx.wherevv
    # on Tx81 (single fp vector select, no arithmetic).  The upstream
    # `expand_mask == 1` comparison would generate an int comparison that
    # falls back to RISC-V scalar on Tx81 (no int vector ops), so we
    # eliminate it by ensuring the mask is already boolean.
    return tl.where(expand_mask, value, inp)


def masked_fill(inp, mask, value):
    logger.debug("GEMS TSINGMICRO MASKED FILL")
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

    # Ensure boolean mask so the kernel can skip the `== 1` comparison.
    # Tx81 has no int vector ops — int comparison falls back to RISC-V
    # scalar.  Converting to bool on the host (cheap) lets the kernel do
    # tl.where(bool_mask, …) → tx.wherevv directly.
    if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)

    expand_mask = mask.expand(inp.shape)
    return masked_fill_kernel(inp, expand_mask, value)


def masked_fill_(inp, mask, value):
    logger.debug("GEMS TSINGMICRO MASKED FILL_")
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

    # Same bool conversion as masked_fill — see comment there.
    if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)

    expand_mask = mask.expand(inp.shape)
    return masked_fill_kernel(inp, expand_mask, value, out0=inp)
