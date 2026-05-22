import logging

import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

try:
    from triton.backends.spine_triton.env import alloc_mbarrier, release_mbarrier
except ImportError:
    alloc_mbarrier = None
    release_mbarrier = None

logger = logging.getLogger(__name__)



@libentry()
@libtuner(
    configs=runtime.get_tuned_config("im2col"),
    key=["C", "GEMM_K"],
)
@triton.jit
def im2col_kernel(
    input_ptr,
    im2col_buf_ptr,
    N,
    C,
    IH,
    IW,
    KH,
    KW,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    OH,
    OW,
    GEMM_M,
    GEMM_K,
    KK,
    input_stride_n,
    input_stride_h,
    input_stride_w,
    input_stride_c,
    im2col_stride_n,
    im2col_stride_m,
    im2col_stride_k,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid = tl.program_id(0)
    n_im2col = pid // (OH * OW)
    ohow = pid % (OH * OW)
    oh = ohow // OW
    ow = ohow % OW
    window_h = oh * stride_h - pad_h
    window_w = ow * stride_w - pad_w

    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(N, IH, IW, C),
        strides=(input_stride_n, input_stride_h, input_stride_w, input_stride_c),
        offsets=(n_im2col, 0, 0, 0),
        block_shape=(1, 1, 1, BLOCK_SIZE_C),
        order=(3, 2, 1, 0),
    )
    output_col_base_ptr = tl.make_block_ptr(
        base=im2col_buf_ptr,
        shape=(N, GEMM_M, GEMM_K),
        strides=(im2col_stride_n, im2col_stride_m, im2col_stride_k),
        offsets=(n_im2col, ohow, 0),
        block_shape=(1, 1, BLOCK_SIZE_C),
        order=(2, 1, 0),
    )

    for kh in range(KH):
        for kw in range(KW):
            h = window_h + kh * dilation_h
            w = window_w + kw * dilation_w
            valid_h = (h >= 0) & (h < IH)
            valid_w = (w >= 0) & (w < IW)
            valid = valid_h & valid_w
            for c_start in range(0, C, BLOCK_SIZE_C):
                if valid:
                    input_ptr_cur = tl.advance(input_block_ptr, (0, h, w, c_start))
                    vals = tl.load(input_ptr_cur, boundary_check=(0, 1, 2, 3))
                    vals = tl.reshape(vals, (1, 1, BLOCK_SIZE_C))
                else:
                    vals = tl.zeros(
                        (1, 1, BLOCK_SIZE_C), dtype=input_ptr.dtype.element_ty
                    )
                col_idx = c_start * KK + kh * KW + kw
                output_ptr_cur = tl.advance(output_col_base_ptr, (0, 0, col_idx))
                tl.store(output_ptr_cur, vals, boundary_check=(0, 1, 2))


@libentry()
@libtuner(
configs=runtime.get_tuned_config("conv2d_spacemit"),
    key=["GEMM_M", "OC", "GEMM_K"],
)
@triton.jit
def bmm_kernel(
    im2col_buf_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    N,
    OC,
    GEMM_M,
    GEMM_K,
    im2col_stride_n,
    im2col_stride_m,
    im2col_stride_k,
    weight_stride_oc,
    weight_stride_k,
    output_stride_n,
    output_stride_oc,
    output_stride_m,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SUB_BLK_M: tl.constexpr,
    MICRO_M: tl.constexpr,
    MICRO_K: tl.constexpr,
    MICRO_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tiles_m = tl.cdiv(GEMM_M, TILE_M)
    num_tiles_n = tl.cdiv(OC, TILE_N)
    NUM_BMM_TILES_PER_BATCH = num_tiles_m * num_tiles_n
    pid_b = pid // NUM_BMM_TILES_PER_BATCH
    local_tile = pid % NUM_BMM_TILES_PER_BATCH
    pid_m = local_tile // num_tiles_n
    pid_n = local_tile % num_tiles_n
    block_m = pid_m * TILE_M
    block_n = pid_n * TILE_N

    a_ptr = tl.make_block_ptr(
        base=im2col_buf_ptr + pid_b * im2col_stride_n,
        shape=(GEMM_M, GEMM_K),
        strides=(im2col_stride_m, im2col_stride_k),
        offsets=(block_m, 0),
        block_shape=(TILE_M, TILE_K),
        order=(1, 0),
    )

    b_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(GEMM_K, OC),
        strides=(weight_stride_k, weight_stride_oc),
        offsets=(0, block_n),
        block_shape=(TILE_K, TILE_N),
        order=(1, 0),
    )

    if HAS_BIAS:
        bias_block_ptr = tl.make_block_ptr(
            base=bias_ptr,
            shape=(OC,),
            strides=(1,),
            offsets=(block_n,),
            block_shape=(TILE_N,),
            order=(0,),
        )
        bias_vals = tl.load(bias_block_ptr, boundary_check=(0,))
    output_ptr = output_ptr + pid_b * output_stride_n

    if EVEN_K:
        a_descriptor_load = smt.descriptor_load(a_ptr, (0, 0))
        a = smt.view(
            a_descriptor_load,
            (0, 0),
            (TILE_M, TILE_K),
            (MICRO_M, MICRO_K),
        )
        b_descriptor_load = smt.descriptor_load(b_ptr, (0, 0))
        b = smt.view(
            b_descriptor_load, (0, 0), (TILE_K, TILE_N), (MICRO_K, MICRO_N)
        )
        acc = smt.dot(a, b)
        acc = smt.view(acc, (0, 0), (TILE_M, TILE_N), (1, 1))
        if HAS_BIAS:
            acc += bias_vals[None, :]
        acc = acc.to(output_ptr.dtype.element_ty)
        o_ptr = tl.make_block_ptr(
            base=output_ptr,
            shape=(GEMM_M, OC),
            strides=(output_stride_m, output_stride_oc),
            offsets=(block_m, block_n),
            block_shape=(TILE_M, TILE_N),
            order=(1, 0),
        )
        tl.store(o_ptr, acc, boundary_check=(0, 1))
    else:
        sub_num = (min(TILE_M, GEMM_M - TILE_M * pid_m) + SUB_BLK_M - 1) // SUB_BLK_M
        for s in smt.parallel(0, sub_num):
            acc = tl.zeros((SUB_BLK_M, TILE_N), dtype=tl.float32)
            acc = smt.view(acc, (0, 0), (SUB_BLK_M, TILE_N), (MICRO_M, MICRO_N))
            for k_start in tl.range(0, GEMM_K, TILE_K):
                a_descriptor_load = smt.descriptor_load(a_ptr, (0, 0))
                a = smt.view(
                    a_descriptor_load,
                    (s * SUB_BLK_M, k_start),
                    (SUB_BLK_M, TILE_K),
                    (MICRO_M, MICRO_K),
                )
                b_descriptor_load = smt.descriptor_load(b_ptr, (0, 0))
                b = smt.view(
                    b_descriptor_load,
                    (k_start, 0),
                    (TILE_K, TILE_N),
                    (MICRO_K, MICRO_N),
                )
                acc += smt.dot(a, b)
            acc = smt.view(acc, (0, 0), (SUB_BLK_M, TILE_N), (1, 1))
            if HAS_BIAS:
                acc += bias_vals[None, :]
            acc = acc.to(output_ptr.dtype.element_ty)
            o_ptr = tl.make_block_ptr(
                base=output_ptr,
                shape=(GEMM_M, OC),
                strides=(output_stride_m, output_stride_oc),
                offsets=(block_m + s * SUB_BLK_M, block_n),
                block_shape=(SUB_BLK_M, TILE_N),
                order=(1, 0),
            )
            tl.store(o_ptr, acc, boundary_check=(0, 1))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("conv2d_spacemit"),
    key=["GEMM_M", "OC", "GEMM_K"],
)
@triton.jit
def conv2d_fused_kernel(
    input_ptr,
    im2col_buf_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    bar,
    N,
    C,
    IH,
    IW,
    KH,
    KW,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    OH,
    OW,
    OC,
    GEMM_M,
    GEMM_K,
    KK,
    NUM_IM2COL_BLOCKS,
    NUM_BMM_BLOCKS,
    input_stride_n,
    input_stride_h,
    input_stride_w,
    input_stride_c,
    im2col_stride_n,
    im2col_stride_m,
    im2col_stride_k,
    weight_stride_oc,
    weight_stride_k,
    output_stride_n,
    output_stride_oc,
    output_stride_m,
    BLOCK_SIZE_C: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SUB_BLK_M: tl.constexpr,
    MICRO_M: tl.constexpr,
    MICRO_K: tl.constexpr,
    MICRO_N: tl.constexpr,
):
    pid = tl.program_id(1)

    if pid < NUM_IM2COL_BLOCKS:
        n_im2col = pid // (OH * OW)
        ohow = pid % (OH * OW)
        oh = ohow // OW
        ow = ohow % OW
        window_h = oh * stride_h - pad_h
        window_w = ow * stride_w - pad_w

        input_block_ptr = tl.make_block_ptr(
            base=input_ptr,
            shape=(N, IH, IW, C),
            strides=(input_stride_n, input_stride_h, input_stride_w, input_stride_c),
            offsets=(n_im2col, 0, 0, 0),
            block_shape=(1, 1, 1, BLOCK_SIZE_C),
            order=(3, 2, 1, 0),
        )
        output_col_base_ptr = tl.make_block_ptr(
            base=im2col_buf_ptr,
            shape=(N, GEMM_M, GEMM_K),
            strides=(im2col_stride_n, im2col_stride_m, im2col_stride_k),
            offsets=(n_im2col, ohow, 0),
            block_shape=(1, 1, BLOCK_SIZE_C),
            order=(2, 1, 0),
        )

        for kh in range(KH):
            for kw in range(KW):
                h = window_h + kh * dilation_h
                w = window_w + kw * dilation_w
                valid_h = (h >= 0) & (h < IH)
                valid_w = (w >= 0) & (w < IW)
                valid = valid_h & valid_w
                for c_start in range(0, C, BLOCK_SIZE_C):
                    if valid:
                        input_ptr_cur = tl.advance(input_block_ptr, (0, h, w, c_start))
                        vals = tl.load(input_ptr_cur, boundary_check=(0, 1, 2, 3))
                        vals = tl.reshape(vals, (1, 1, BLOCK_SIZE_C))
                    else:
                        vals = tl.zeros(
                            (1, 1, BLOCK_SIZE_C), dtype=input_ptr.dtype.element_ty
                        )
                    col_idx = c_start * KK + kh * KW + kw
                    output_ptr_cur = tl.advance(output_col_base_ptr, (0, 0, col_idx))
                    tl.store(output_ptr_cur, vals, boundary_check=(0, 1, 2))

    smt.barrier_arrive(bar)
    smt.barrier_wait(bar, flag=1)

    if pid < NUM_BMM_BLOCKS:
        num_tiles_m = tl.cdiv(GEMM_M, TILE_M)
        num_tiles_n = tl.cdiv(OC, TILE_N)
        NUM_BMM_TILES_PER_BATCH = num_tiles_m * num_tiles_n
        pid_b = pid // NUM_BMM_TILES_PER_BATCH
        local_tile = pid % NUM_BMM_TILES_PER_BATCH
        pid_m = local_tile // num_tiles_n
        pid_n = local_tile % num_tiles_n
        block_m = pid_m * TILE_M
        block_n = pid_n * TILE_N

        a_ptr = tl.make_block_ptr(
            base=im2col_buf_ptr + pid_b * im2col_stride_n,
            shape=(GEMM_M, GEMM_K),
            strides=(im2col_stride_m, im2col_stride_k),
            offsets=(block_m, 0),
            block_shape=(TILE_M, TILE_K),
            order=(1, 0),
        )

        b_ptr = tl.make_block_ptr(
            base=weight_ptr,
            shape=(GEMM_K, OC),
            strides=(weight_stride_k, weight_stride_oc),
            offsets=(0, block_n),
            block_shape=(TILE_K, TILE_N),
            order=(1, 0),
        )

        if HAS_BIAS:
            bias_block_ptr = tl.make_block_ptr(
                base=bias_ptr,
                shape=(OC,),
                strides=(1,),
                offsets=(block_n,),
                block_shape=(TILE_N,),
                order=(0,),
            )
            bias_vals = tl.load(bias_block_ptr, boundary_check=(0,))
        output_ptr = output_ptr + pid_b * output_stride_n

        if EVEN_K:
            a_descriptor_load = smt.descriptor_load(a_ptr, (0, 0))
            a = smt.view(
                a_descriptor_load,
                (0, 0),
                (TILE_M, TILE_K),
                (MICRO_M, MICRO_K),
            )
            b_descriptor_load = smt.descriptor_load(b_ptr, (0, 0))
            b = smt.view(
                b_descriptor_load, (0, 0), (TILE_K, TILE_N), (MICRO_K, MICRO_N)
            )
            acc = smt.dot(a, b)
            acc = smt.view(acc, (0, 0), (TILE_M, TILE_N), (1, 1))
            if HAS_BIAS:
                acc += bias_vals[None, :]
            acc = acc.to(output_ptr.dtype.element_ty)
            o_ptr = tl.make_block_ptr(
                base=output_ptr,
                shape=(GEMM_M, OC),
                strides=(output_stride_m, output_stride_oc),
                offsets=(block_m, block_n),
                block_shape=(TILE_M, TILE_N),
                order=(1, 0),
            )
            tl.store(o_ptr, acc, boundary_check=(0, 1))
        else:
            sub_num = (min(TILE_M, GEMM_M - TILE_M * pid_m) + SUB_BLK_M - 1) // SUB_BLK_M
            for s in smt.parallel(0, sub_num):
                acc = tl.zeros((SUB_BLK_M, TILE_N), dtype=tl.float32)
                acc = smt.view(acc, (0, 0), (SUB_BLK_M, TILE_N), (MICRO_M, MICRO_N))
                for k_start in tl.range(0, GEMM_K, TILE_K):
                    a_descriptor_load = smt.descriptor_load(a_ptr, (0, 0))
                    a = smt.view(
                        a_descriptor_load,
                        (s * SUB_BLK_M, k_start),
                        (SUB_BLK_M, TILE_K),
                        (MICRO_M, MICRO_K),
                    )
                    b_descriptor_load = smt.descriptor_load(b_ptr, (0, 0))
                    b = smt.view(
                        b_descriptor_load,
                        (k_start, 0),
                        (TILE_K, TILE_N),
                        (MICRO_K, MICRO_N),
                    )
                    acc += smt.dot(a, b)
                acc = smt.view(acc, (0, 0), (SUB_BLK_M, TILE_N), (1, 1))
                if HAS_BIAS:
                    acc += bias_vals[None, :]
                acc = acc.to(output_ptr.dtype.element_ty)
                o_ptr = tl.make_block_ptr(
                    base=output_ptr,
                    shape=(GEMM_M, OC),
                    strides=(output_stride_m, output_stride_oc),
                    offsets=(block_m + s * SUB_BLK_M, block_n),
                    block_shape=(SUB_BLK_M, TILE_N),
                    order=(1, 0),
                )
                tl.store(o_ptr, acc, boundary_check=(0, 1))


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    logger.debug("GEMS_SPACEMIT CONV2D")

    N, C, H, W = input.shape
    OC, _, KH, KW = weight.shape

    str_h, str_w = (stride, stride) if isinstance(stride, int) else stride
    pad_h, pad_w = (padding, padding) if isinstance(padding, int) else padding
    dil_h, dil_w = (dilation, dilation) if isinstance(dilation, int) else dilation

    OH = (H + 2 * pad_h - dil_h * (KH - 1) - 1) // str_h + 1
    OW = (W + 2 * pad_w - dil_w * (KW - 1) - 1) // str_w + 1

    GEMM_M = OH * OW
    KK = KH * KW
    GEMM_K = C * KK

    im2col_buf = torch.empty(
        (N, GEMM_M, GEMM_K), dtype=input.dtype, device=input.device
    )

    output = torch.empty((N, OC, OH, OW), dtype=input.dtype, device=input.device)

    input_nhwc = input.permute(0, 2, 3, 1).contiguous()
    weight_flat = weight.view(OC, -1).contiguous()

    NUM_IM2COL_BLOCKS = N * OH * OW

    TILE_K = triton.next_power_of_2(GEMM_K)

    def im2col_grid(_META):
        return (NUM_IM2COL_BLOCKS,)

    def bmm_grid(META):
        num_tiles_m = triton.cdiv(GEMM_M, META["TILE_M"])
        num_tiles_n = triton.cdiv(OC, META["TILE_N"])
        num_bmm_tiles_per_batch = num_tiles_m * num_tiles_n
        return (N * num_bmm_tiles_per_batch,)

    def fused_grid(META):
        num_tiles_m = triton.cdiv(GEMM_M, META["TILE_M"])
        num_tiles_n = triton.cdiv(OC, META["TILE_N"])
        num_bmm_tiles_per_batch = num_tiles_m * num_tiles_n
        num_bmm_blocks = N * num_bmm_tiles_per_batch
        return (1, max(NUM_IM2COL_BLOCKS, num_bmm_blocks))

    if bias is not None:
        bias_ptr = bias.contiguous()
    else:
        bias_ptr = torch.empty(0, device=input.device, dtype=input.dtype)

    output_3d = output.view(N, OC, GEMM_M)

    with torch_device_fn.device(input.device):
        fused_blocks = fused_grid({"TILE_M": 32, "TILE_N": 32})[1]
        if alloc_mbarrier is not None and release_mbarrier is not None and fused_blocks <= 32767:
            bar = alloc_mbarrier(fused_blocks)
            try:
                conv2d_fused_kernel[fused_grid](
                    input_nhwc,
                    im2col_buf,
                    weight_flat,
                    bias_ptr,
                    output_3d,
                    bar,
                    N,
                    C,
                    H,
                    W,
                    KH,
                    KW,
                    str_h,
                    str_w,
                    pad_h,
                    pad_w,
                    dil_h,
                    dil_w,
                    OH,
                    OW,
                    OC,
                    GEMM_M,
                    GEMM_K,
                    KK,
                    NUM_IM2COL_BLOCKS,
                    fused_blocks,
                    input_nhwc.stride(0),
                    input_nhwc.stride(1),
                    input_nhwc.stride(2),
                    input_nhwc.stride(3),
                    im2col_buf.stride(0),
                    im2col_buf.stride(1),
                    im2col_buf.stride(2),
                    weight_flat.stride(0),
                    weight_flat.stride(1),
                    output_3d.stride(0),
                    output_3d.stride(1),
                    output_3d.stride(2),
                    BLOCK_SIZE_C=256,
                    TILE_K=TILE_K,
                    HAS_BIAS=(bias is not None),
                )
            finally:
                release_mbarrier(bar)
        else:
            im2col_kernel[im2col_grid](
                input_nhwc,
                im2col_buf,
                N,
                C,
                H,
                W,
                KH,
                KW,
                str_h,
                str_w,
                pad_h,
                pad_w,
                dil_h,
                dil_w,
                OH,
                OW,
                GEMM_M,
                GEMM_K,
                KK,
                input_nhwc.stride(0),
                input_nhwc.stride(1),
                input_nhwc.stride(2),
                input_nhwc.stride(3),
                im2col_buf.stride(0),
                im2col_buf.stride(1),
                im2col_buf.stride(2),
            )
            bmm_kernel[bmm_grid](
                im2col_buf,
                weight_flat,
                bias_ptr,
                output_3d,
                N,
                OC,
                GEMM_M,
                GEMM_K,
                im2col_buf.stride(0),
                im2col_buf.stride(1),
                im2col_buf.stride(2),
                weight_flat.stride(0),
                weight_flat.stride(1),
                output_3d.stride(0),
                output_3d.stride(1),
                output_3d.stride(2),
                TILE_K=TILE_K,
                HAS_BIAS=(bias is not None),
            )

    return output
