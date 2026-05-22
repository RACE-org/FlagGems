import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)


@triton.jit
def maxpool2d_kernel(
    input_ptr,
    output_ptr,
    C,
    IH,
    IW,
    OH,
    OW,
    KH: tl.constexpr,
    KW: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    input_batch_stride,
    input_channel_stride,
    input_height_stride,
    input_width_stride,
    output_batch_stride,
    output_channel_stride,
    output_height_stride,
    output_width_stride,
    num_blocks_c,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid = tl.program_id(0)
    block_c_index = pid % num_blocks_c
    combined_n_oh_ow = pid // num_blocks_c

    total_spatial = OH * OW
    n = combined_n_oh_ow // total_spatial
    oh_ow = combined_n_oh_ow % total_spatial
    oh = oh_ow // OW
    ow = oh_ow % OW

    c_base = block_c_index * BLOCK_SIZE_C
    c_global = c_base + tl.arange(0, BLOCK_SIZE_C)
    channel_mask = c_global < C

    window_h = oh * stride_h - pad_h
    window_w = ow * stride_w - pad_w
    min_value = get_dtype_min(input_ptr.type.element_ty)
    max_vals = tl.full((BLOCK_SIZE_C,), min_value, dtype=tl.float32)

    total_iters = KH * KW
    for k in range(total_iters):
        kh = k // KW
        kw = k % KW
        h = window_h + kh * dilation_h
        w = window_w + kw * dilation_w

        valid_h = (h >= 0) & (h < IH)
        valid_w = (w >= 0) & (w < IW)
        valid = valid_h & valid_w
        if valid:
            input_offset = (
                n * input_batch_stride
                + c_global * input_channel_stride
                + h * input_height_stride
                + w * input_width_stride
            )
            input_ptrs = input_ptr + input_offset
            total_mask = valid & channel_mask

            current = tl.load(input_ptrs, mask=total_mask, other=min_value)
            max_vals = tl.maximum(max_vals, current)
    max_vals = tl.where(max_vals == min_value, 0.0, max_vals)

    output_offset = (
        n * output_batch_stride
        + c_global * output_channel_stride
        + oh * output_height_stride
        + ow * output_width_stride
    )
    output_ptrs = output_ptr + output_offset
    tl.store(output_ptrs, max_vals, mask=channel_mask)


def maxpool2d(
    input: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
    dilation: int = 1,
) -> torch.Tensor:
    logger.debug("GEMS_SPACEMIT MAXPOOL2D")
    KH, KW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    stride_h, stride_w = (stride, stride) if isinstance(stride, int) else stride
    pad_h, pad_w = (padding, padding) if isinstance(padding, int) else padding
    dil_h, dil_w = (dilation, dilation) if isinstance(dilation, int) else dilation

    N, C, IH, IW = input.shape
    OH = (IH + 2 * pad_h - dil_h * (KH - 1) - 1) // stride_h + 1
    OW = (IW + 2 * pad_w - dil_w * (KW - 1) - 1) // stride_w + 1
    output = torch.empty((N, C, OH, OW), dtype=input.dtype, device=input.device)

    (
        input_batch_stride,
        input_channel_stride,
        input_height_stride,
        input_width_stride,
    ) = input.stride()
    (
        output_batch_stride,
        output_channel_stride,
        output_height_stride,
        output_width_stride,
    ) = output.stride()

    # A100: BLOCK_SIZE_C=64 fits well in L1 (64*4B=256B per spatial point)
    BLOCK_SIZE_C = 64
    num_blocks_c = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    total_programs = N * OH * OW * num_blocks_c
    grid = (total_programs,)

    with torch_device_fn.device(input.device):
        maxpool2d_kernel[grid](
            input,
            output,
            C,
            IH,
            IW,
            OH,
            OW,
            KH,
            KW,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dil_h,
            dil_w,
            input_batch_stride,
            input_channel_stride,
            input_height_stride,
            input_width_stride,
            output_batch_stride,
            output_channel_stride,
            output_height_stride,
            output_width_stride,
            num_blocks_c,
            BLOCK_SIZE_C,
        )
    return output
