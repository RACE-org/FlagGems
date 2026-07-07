import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

from .mm import mm

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def im2col_kernel(
    input_ptr,
    cols_ptr,
    N, C, H, W, KH, KW,
    stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
    OH, OW, K, M,
    in_stride_n, in_stride_c, in_stride_h, in_stride_w,
    col_stride_n, col_stride_k, col_stride_m,
    BLOCK_M: tl.constexpr,
):
    pid_nk = tl.program_id(0)
    pid_m  = tl.program_id(1)

    n  = pid_nk // K
    k  = pid_nk % K
    c  = k // (KH * KW)
    rem = k % (KH * KW)
    kh = rem // KW
    kw = rem % KW

    m      = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m < M
    oh = m // OW
    ow = m % OW

    ih = oh * stride_h - pad_h + kh * dil_h
    iw = ow * stride_w - pad_w + kw * dil_w
    valid = mask_m & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

    in_offs = (
        n  * in_stride_n
        + c  * in_stride_c
        + ih * in_stride_h
        + iw * in_stride_w
    )
    vals = tl.load(input_ptr + in_offs, mask=valid, other=0.0)

    col_offs = n * col_stride_n + k * col_stride_k + m * col_stride_m
    tl.store(cols_ptr + col_offs, vals, mask=mask_m)


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    logger.debug("GEMS_SPACEMIT CONV2D")

    input = input.contiguous()
    N, C, H, W = input.shape
    OC, C_per_group, KH, KW = weight.shape

    str_h, str_w = (stride, stride) if isinstance(stride, int) else tuple(stride)
    pad_h, pad_w = (padding, padding) if isinstance(padding, int) else tuple(padding)
    dil_h, dil_w = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)

    OH = (H + 2 * pad_h - dil_h * (KH - 1) - 1) // str_h + 1
    OW = (W + 2 * pad_w - dil_w * (KW - 1) - 1) // str_w + 1
    M = OH * OW
    K_total = C * KH * KW
    K_per_group = C_per_group * KH * KW
    OC_per_group = OC // groups

    cols = torch.empty((N, K_total, M), dtype=input.dtype, device=input.device)
    BLOCK_M = 128
    grid = (N * K_total, triton.cdiv(M, BLOCK_M))
    im2col_kernel[grid](
        input, cols,
        N, C, H, W, KH, KW,
        str_h, str_w, pad_h, pad_w, dil_h, dil_w,
        OH, OW, K_total, M,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        cols.stride(0), cols.stride(1), cols.stride(2),
        BLOCK_M=BLOCK_M,
    )

    weight_flat = weight.reshape(OC, K_per_group).contiguous()

    out_groups = []
    for g in range(groups):
        w_g = weight_flat[g * OC_per_group : (g + 1) * OC_per_group, :]
        a_g = (
            cols[:, g * K_per_group : (g + 1) * K_per_group, :]
            .permute(1, 0, 2)
            .reshape(K_per_group, N * M)
        )
        r_g = mm(w_g, a_g)
        out_groups.append(r_g.reshape(OC_per_group, N, M))

    output = torch.cat(out_groups, dim=0).permute(1, 0, 2).contiguous()

    if bias is not None:
        output = output + bias.reshape(1, OC, 1)

    return output.reshape(N, OC, OH, OW)
