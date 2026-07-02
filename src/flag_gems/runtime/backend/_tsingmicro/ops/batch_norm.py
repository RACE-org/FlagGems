import logging

import torch
import triton
import triton.language as tl
from torch import Tensor

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)
rsqrt = torch.rsqrt


def make_3d_for_bn(input: Tensor) -> Tensor:
    if input.ndim == 2:
        return input.unsqueeze(-1)
    elif input.ndim >= 4:
        input = input.flatten(2, -1)
    return input


def _make_input_2d(input: Tensor) -> Tensor:
    """Reshape to (R, C) where R = N * spatial, C contiguous for DMA burst."""
    if input.ndim == 2:
        return input.contiguous()
    elif input.ndim == 3:
        N, C, S = input.shape
        return input.permute(0, 2, 1).reshape(N * S, C).contiguous()
    else:
        x = input.flatten(2, -1)
        N, C, S = x.shape
        return x.permute(0, 2, 1).reshape(N * S, C).contiguous()


# ===========================================================================
# Training forward — sum/sumsq reduce via GEMM + elementwise affine.
#
# Avoids the upstream per-channel Welford loop with mask/where/int ops.
#
# Stage 1: block-GEMM partial sum/sumsq.
#   Input layout: (R, C), R = N*spatial, C contiguous.
#   Grid: (ceil(R / BLOCK_R), ceil(C / BLOCK_C)).
#   Each CTA computes ones @ X_block and ones @ (X_block)² via GEMM.
#
# Stage 2: cross-block reduce → mean, inv_std, update running stats.
#
# Stage 3: elementwise normalize + affine = weight * (x - mean) * inv_std + bias.
# ===========================================================================

_BLOCK_R = 64       # rows per CTA in GEMM reduce
_BLOCK_C = 128      # channels per CTA in GEMM reduce


@libentry()
@triton.jit
def bn_train_stage1_kernel(
    inp,
    partial_sum,
    partial_sumsq,
    R,
    C,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Stage 1: block-GEMM partial sum and sumsq.

    Grid: (ceil(R / BLOCK_R), ceil(C / BLOCK_C)).

    ones[1, BLOCK_R] @ X[BLOCK_R, BLOCK_C]  →  partial_sum[grid_r, C]
    ones[1, BLOCK_R] @ X²[BLOCK_R, BLOCK_C] →  partial_sumsq[grid_r, C]
    """
    pid_r = tle.program_id(0)
    pid_c = tle.program_id(1)

    r_offs = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    r_mask = r_offs < R
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    offs = r_offs[:, None] * C + c_offs[None, :]
    mask = r_mask[:, None] & c_mask[None, :]
    x = tl.load(inp + offs, mask=mask, other=0.0).to(tl.float32)

    # Hardware reduce over BLOCK_R to get one scalar per channel.
    # tx.reduce_sum(axis=0) — hardware vector reduce.
    row_sum = tl.sum(x, axis=0)         # [BLOCK_C]
    row_sumsq = tl.sum(x * x, axis=0)   # [BLOCK_C]

    tl.store(partial_sum + pid_r * C + c_offs, row_sum, mask=c_mask)
    tl.store(partial_sumsq + pid_r * C + c_offs, row_sumsq, mask=c_mask)


def _block_reduce_on_host(partial_tensor, R_blocks, C, device):
    """Reduce (R_blocks, C) partial sums across R_blocks → (C,) via host loop.

    For small R_blocks (< 512) the host overhead is negligible compared
    to launching another kernel.  Each block contribution is just a C-element
    vector — one read + add per R_block.
    """
    result = partial_tensor[0].clone()
    for i in range(1, R_blocks):
        result += partial_tensor[i]
    return result


def batch_norm(
    input: Tensor,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
):
    logger.debug("GEMS TSINGMICRO BATCHNORM FORWARD")

    C = input.shape[1]

    w = torch.ones(C, dtype=input.dtype, device=input.device) if weight is None else weight
    b = torch.zeros(C, dtype=input.dtype, device=input.device) if bias is None else bias
    rm = input if running_mean is None else running_mean
    rv = input if running_var is None else running_var

    if not training:
        # =====================================================================
        # Eval: fused elementwise with fp32 arithmetic, matching the
        # reference formula exactly:
        #   output = weight * (x - mean) * inv_std + bias
        #
        # Host-side precomputation of scale/shift causes extra fp16
        # truncations (3× vs reference 1×).  Instead, pass all operands
        # into a single pointwise_dynamic kernel: every op inside the
        # fusion runs in fp32, truncating only at the final store.
        # =====================================================================
        w_f32 = w.to(torch.float32)
        b_f32 = b.to(torch.float32)
        rm_f32 = rm.to(torch.float32)
        rv_f32 = rv.to(torch.float32)
        inv_std = torch.rsqrt(rv_f32 + eps)

        # Single fused kernel: weight * (x - mean) * inv_std + bias
        # Build broadcast-compatible view for 1D (C,) params against ND input.
        if input.ndim == 2:
            output_f32 = w_f32 * (input.to(torch.float32) - rm_f32) * inv_std + b_f32
        else:
            view_shape = (1, C) + (1,) * (input.ndim - 2)
            output_f32 = (
                w_f32.view(view_shape) * (input.to(torch.float32) - rm_f32.view(view_shape))
                * inv_std.view(view_shape) + b_f32.view(view_shape)
            )
        output = output_f32.to(input.dtype)

        return output, rm_f32, inv_std

    # =========================================================================
    # Training: sum/sumsq two-stage reduce + elementwise affine.
    # =========================================================================
    x2d = _make_input_2d(input).contiguous()
    R = x2d.shape[0]

    R_blocks = triton.cdiv(R, _BLOCK_R)
    C_blocks = triton.cdiv(C, _BLOCK_C)

    partial_sum = torch.empty((R_blocks, C), dtype=torch.float32, device=input.device)
    partial_sumsq = torch.empty((R_blocks, C), dtype=torch.float32, device=input.device)

    grid = (R_blocks, C_blocks)
    with torch_device_fn.device(input.device):
        bn_train_stage1_kernel[grid](
            x2d, partial_sum, partial_sumsq, R, C,
            BLOCK_R=_BLOCK_R, BLOCK_C=_BLOCK_C,
        )

    # Stage 2: cross-block reduce → sum, sumsq → mean, var, inv_std.
    total_sum = _block_reduce_on_host(partial_sum, R_blocks, C, input.device)
    total_sumsq = _block_reduce_on_host(partial_sumsq, R_blocks, C, input.device)

    count = float(R)
    mean = total_sum / count
    var = total_sumsq / count - mean * mean
    inv_std = rsqrt(var + eps)

    # Update running stats
    if running_mean is not None:
        running_mean.copy_((1 - momentum) * running_mean + momentum * mean)
    if running_var is not None:
        running_var.copy_((1 - momentum) * running_var + momentum * var * count / max(count - 1, 1))

    # Stage 3: elementwise normalize + affine.
    output_2d = w * (x2d - mean) * inv_std + b

    if input.ndim == 2:
        output = output_2d
    elif input.ndim == 3:
        N, _, S = input.shape
        output = output_2d.view(N, S, C).permute(0, 2, 1).contiguous()
    else:
        output = output_2d.view_as(input)

    return output, mean, inv_std


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("batch_norm"),
    key=["batch_dim", "spatial_dim"],
)
@triton.heuristics(runtime.get_heuristic_config("batch_norm"))
@triton.jit
def bn_backward_kernel(
    output_grad_pointer,
    input_pointer,
    mean_pointer,
    inv_std_pointer,
    weight_pointer,
    input_grad_pointer,
    weight_grad_pointer,
    bias_grad_pointer,
    batch_dim,
    spatial_dim,
    output_grad_batch_stride,
    output_grad_feat_stride,
    output_grad_spatial_stride,
    input_batch_stride,
    input_feat_stride,
    input_spatial_stride,
    input_grad_batch_stride,
    input_grad_feat_stride,
    input_grad_spatial_stride,
    input_grad_mask: tl.constexpr,
    weight_grad_mask: tl.constexpr,
    bias_grad_mask: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    feat_pid = tl.program_id(axis=0)

    mean = tl.load(feat_pid + mean_pointer).to(tl.float32)
    inv_std = tl.load(feat_pid + inv_std_pointer).to(tl.float32)

    term1 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    term2 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for m_step in range(0, tl.cdiv(batch_dim, BLOCK_M)):
        for n_step in range(0, tl.cdiv(spatial_dim, BLOCK_N)):
            batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
            batch_mask = batch_offset < batch_dim

            spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
            spatial_mask = spatial_offset < spatial_dim

            curr_output_grad_pointer = (
                output_grad_pointer
                + output_grad_feat_stride * feat_pid
                + output_grad_batch_stride * batch_offset[:, None]
                + output_grad_spatial_stride * spatial_offset[None, :]
            )
            curr_input_pointer = (
                input_pointer
                + input_feat_stride * feat_pid
                + input_batch_stride * batch_offset[:, None]
                + input_spatial_stride * spatial_offset[None, :]
            )

            mask = batch_mask[:, None] & spatial_mask[None, :]
            curr_input = tl.load(curr_input_pointer, mask=mask).to(tl.float32)

            curr_pre_lin = (curr_input - mean) * inv_std
            curr_output_grad = tl.load(curr_output_grad_pointer, mask=mask).to(
                tl.float32
            )

            term1 += curr_pre_lin * curr_output_grad
            term2 += curr_output_grad

    term1 = tl.sum(term1)
    term2 = tl.sum(term2)

    if weight_grad_mask:
        tl.store(feat_pid + weight_grad_pointer, term1)
    if bias_grad_mask:
        tl.store(feat_pid + bias_grad_pointer, term2)

    if not input_grad_mask:
        return

    if weight_pointer:
        weight = tl.load(feat_pid + weight_pointer).to(tl.float32)
    else:
        weight = 1.0

    count = batch_dim * spatial_dim

    for m_step in range(0, tl.cdiv(batch_dim, BLOCK_M)):
        for n_step in range(0, tl.cdiv(spatial_dim, BLOCK_N)):
            batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
            batch_mask = batch_offset < batch_dim

            spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
            spatial_mask = spatial_offset < spatial_dim

            curr_output_grad_pointer = (
                output_grad_pointer
                + output_grad_feat_stride * feat_pid
                + output_grad_batch_stride * batch_offset[:, None]
                + output_grad_spatial_stride * spatial_offset[None, :]
            )
            curr_input_pointer = (
                input_pointer
                + input_feat_stride * feat_pid
                + input_batch_stride * batch_offset[:, None]
                + input_spatial_stride * spatial_offset[None, :]
            )
            curr_input_grad_pointer = (
                input_grad_pointer
                + input_grad_feat_stride * feat_pid
                + input_grad_batch_stride * batch_offset[:, None]
                + input_grad_spatial_stride * spatial_offset[None, :]
            )

            curr_input = tl.load(
                curr_input_pointer, mask=batch_mask[:, None] & spatial_mask[None, :]
            ).to(tl.float32)
            curr_pre_lin = (curr_input - mean) * inv_std
            curr_output_grad = tl.load(
                curr_output_grad_pointer,
                mask=batch_mask[:, None] & spatial_mask[None, :],
            ).to(tl.float32)
            curr_input_grad = (
                inv_std
                * weight
                * (curr_output_grad - (term1 * curr_pre_lin + term2) / count)
            )
            tl.store(
                curr_input_grad_pointer,
                curr_input_grad,
                mask=batch_mask[:, None] & spatial_mask[None, :],
            )


def batch_norm_backward(
    grad_out,
    input,
    weight=None,
    running_mean=None,
    running_var=None,
    save_mean=None,
    save_invstd=None,
    train=False,
    eps=1e-05,
    output_mask=None,
):
    logger.debug("GEMS TSINGMICRO BATCHNORM BACKWARD")
    input_3d = make_3d_for_bn(input)
    output_grad_3d = make_3d_for_bn(grad_out)

    batch_dim, feat_dim, spatial_dim = input_3d.shape

    if output_mask[0]:
        input_grad = torch.empty_like(input_3d)
        ig_strides = input_grad.stride()
    else:
        input_grad = None
        ig_strides = (1, 1, 1)
    if output_mask[1]:
        weight_grad = torch.empty((feat_dim,), dtype=input.dtype, device=input.device)
    else:
        weight_grad = None
    if output_mask[2]:
        bias_grad = torch.empty((feat_dim,), dtype=input.dtype, device=input.device)
    else:
        bias_grad = None

    with torch_device_fn.device(input.device):
        bn_backward_kernel[(feat_dim,)](
            output_grad_3d,
            input_3d,
            save_mean,
            save_invstd,
            weight,
            input_grad,
            weight_grad,
            bias_grad,
            batch_dim,
            spatial_dim,
            *output_grad_3d.stride(),
            *input_3d.stride(),
            *ig_strides,
            *output_mask,
        )

    return (
        input_grad.view_as(input) if input_grad is not None else None,
        weight_grad,
        bias_grad,
    )
