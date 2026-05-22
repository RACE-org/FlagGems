import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _unfold_kernel(
    in_ptr,
    out_ptr,
    N: int,
    C: int,
    H: int,
    W: int,
    kh: int,
    kw: int,
    stride: int,
    padding: int,
    dilation: int,
    H_out: int,
    W_out: int,
    L: int,
    BLOCK_L: tl.constexpr,
):
    pid_n = tl.program_id(2)
    pid_f = tl.program_id(1)
    pid_l = tl.program_id(0) * BLOCK_L + tl.arange(0, BLOCK_L)
    l_mask = pid_l < L

    oh = pid_l // W_out
    ow = pid_l % W_out

    ih = oh * stride - padding
    iw = ow * stride - padding

    c_idx = pid_f // (kh * kw)
    k_idx = pid_f % (kh * kw)
    kh_idx = k_idx // kw
    kw_idx = k_idx % kw

    in_h = ih + kh_idx * dilation
    in_w = iw + kw_idx * dilation

    in_addr = pid_n * (C * H * W) + c_idx * (H * W) + in_h * W + in_w
    valid_mask = l_mask & (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W)
    val = tl.where(valid_mask, tl.load(in_ptr + in_addr), 0.0)

    out_addr = pid_n * (C * kh * kw * L) + pid_f * L + pid_l
    tl.store(out_ptr + out_addr, val, mask=l_mask)


def unfold(
    input: torch.Tensor,
    kernel_size: tuple,
    stride: int = 1,
    dilation: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    logger.debug("GEMS_SPACEMIT UNFOLD")
    assert len(input.shape) == 4, "input must be 4D tensor(N, C, H, W)"
    assert len(kernel_size) == 2, "kernel_size should be (height, width)"
    kh, kw = kernel_size

    N, C, H, W = input.shape

    H_out = (H + 2 * padding - dilation * (kh - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (kw - 1) - 1) // stride + 1
    L = H_out * W_out

    output = torch.empty((N, C * kh * kw, L), device=input.device, dtype=input.dtype)

    grid = lambda meta: (triton.cdiv(L, meta["BLOCK_L"]), C * kh * kw, N)

    with torch_device_fn.device(input.device):
        _unfold_kernel[grid](
            input,
            output,
            N,
            C,
            H,
            W,
            kh,
            kw,
            stride,
            padding,
            dilation,
            H_out,
            W_out,
            L,
            BLOCK_L=triton.next_power_of_2(L),
        )
    return output
