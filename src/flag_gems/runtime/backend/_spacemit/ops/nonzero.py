import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

NUM_CTAS = 8


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("nonzero"),
    key=["n_elements"],
)
@triton.jit
def nonzero_kernel(
    inp,
    prefix_sum,
    out,
    n_elements,
    shape,
    ndim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)

    total_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(total_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        offset = task_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < n_elements

        inp_vals = tl.load(inp + offset, mask=mask)
        out_offset = tl.load(prefix_sum + offset, mask=mask) - 1

        nonzero_mask = mask & (inp_vals != 0)  # noqa

        idx_flat = offset
        for i in tl.static_range(0, ndim):
            dim = ndim - 1 - i
            dim_size = tl.load(shape + dim)
            remainder = idx_flat % dim_size
            idx_flat //= dim_size
            tl.store(out + out_offset * ndim + dim, remainder, mask=nonzero_mask)


def nonzero(inp, *, as_tuple=False):
    logger.debug("GEMS_SPACEMIT NONZERO")

    inp_ndim = inp.ndim

    inp = inp.contiguous()
    n_elements = inp.numel()
    inp_view = inp.view(n_elements)

    shape = torch.tensor(inp.shape, dtype=torch.int32, device=inp.device)

    inp_bool = inp_view
    if inp_view.dtype != torch.bool:
        inp_bool = inp_view != 0

    prefix_sum = inp_bool.cumsum(axis=0)

    num_nonzeros = n_elements
    out = torch.empty(num_nonzeros, inp_ndim, dtype=torch.int64, device=inp.device)

    with torch_device_fn.device(inp.device):
        nonzero_kernel[(NUM_CTAS,)](
            inp_bool.to(torch.int8),
            prefix_sum,
            out,
            n_elements,
            shape,
            inp_ndim,
        )

    num_nonzeros = prefix_sum[n_elements - 1].item()
    out = out[0:num_nonzeros]

    if as_tuple:
        return torch.unbind(out, dim=1)
    else:
        return out
