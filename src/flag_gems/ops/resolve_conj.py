import logging

import torch
import triton
import triton.language as tl

import flag_gems

logger = logging.getLogger(__name__)


@triton.jit
def resolve_conj_kernel_1d(
    x_real_ptr,
    x_img_ptr,
    output_ptr,
    n_elements_total,
    is_conj: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements_total

    real = tl.load(x_real_ptr + offsets, mask=mask)
    imag = tl.load(x_img_ptr + offsets, mask=mask)

    output_real_offsets = 2 * offsets
    output_img_offsets = 2 * offsets + 1

    tl.store(output_ptr + output_real_offsets, real, mask=mask)
    tl.store(output_ptr + output_img_offsets, tl.where(is_conj, -imag, imag), mask=mask)


def resolve_conj(A: torch.Tensor):
    logger.debug("GEMS RESOLVE_CONJ")
    if not A.is_complex():
        return A

    if A.numel() == 0:
        return A.clone()

    out = torch.empty_like(A)
    n_elements = A.numel()
    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)

    with flag_gems.runtime.torch_device_fn.device(A.device):
        resolve_conj_kernel_1d[grid](
            A.real,
            A.imag,
            out.view(torch.float32),
            n_elements,
            A.is_conj(),
            BLOCK_SIZE=block_size,
        )

    return out
