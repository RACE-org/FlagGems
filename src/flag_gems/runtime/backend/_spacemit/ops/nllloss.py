import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss_forward_kernel(
    inp_ptr,
    tgt_ptr,
    wgt_ptr,
    out_ptr,
    ignore_index,
    N,
    C,
    reduction: tl.constexpr = 1,
    BLOCK_N: tl.constexpr = 128,
):
    pid_nd = tl.program_id(0)
    offset_nd = pid_nd * BLOCK_N + tl.arange(0, BLOCK_N)
    offset_d = offset_nd % 1
    offset_n = offset_nd // 1

    mask_block = offset_nd < N

    tgt_ptrs = tgt_ptr + offset_n + offset_d
    tgt = tl.load(tgt_ptrs, mask=mask_block, other=0)
    assert tgt >= 0 and tgt < C, "Invalid target value"
    ignore_mask = (tgt != ignore_index) & mask_block

    if wgt_ptr is None:
        wgt_tgt = ignore_mask.to(tl.float32)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    inp_tgt_ptrs = inp_ptr + offset_n * C + tgt + offset_d
    inp_tgt = tl.load(inp_tgt_ptrs, mask=ignore_mask, other=0).to(tl.float32)
    out = inp_tgt * wgt_tgt * -1

    # none
    if reduction == 0:
        out_ptrs = out_ptr + offset_n + offset_d
        tl.store(out_ptrs, out, mask=mask_block)
    # mean
    elif reduction == 1:
        total_out = tl.sum(out)
        total_wgt = tl.sum(wgt_tgt)
        tl.store(out_ptr, total_out)
        tl.store(out_ptr + 1, total_wgt)
        tl.store(out_ptr + 3, total_out / total_wgt)
    # sum
    else:
        total_out = tl.sum(out)
        tl.store(out_ptr, total_out)


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss_backward_kernel(
    out_grad_ptr,
    tgt_ptr,
    wgt_ptr,
    inp_grad_ptr,
    ignore_index,
    total_weight,
    N,
    C,
    reduction: tl.constexpr = 1,
    BLOCK_N: tl.constexpr = 128,
):
    pid_n = tl.program_id(0)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_n = offsets_n < N

    tgt = tl.load(tgt_ptr + offsets_n, mask=mask_n, other=0)
    ignore_mask = (tgt != ignore_index) & mask_n

    if wgt_ptr is None:
        wgt_tgt = ignore_mask.to(tl.float32)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    if reduction == 0:
        out_grad_ptrs = out_grad_ptr + offsets_n
        out_grad = tl.load(out_grad_ptrs, mask=mask_n, other=0).to(tl.float32)
    else:
        out_grad = tl.load(out_grad_ptr).to(tl.float32)
    if reduction == 1:
        total_w = tl.load(total_weight).to(tl.float32)
    else:
        total_w = 1

    inp_grad = tl.where(ignore_mask, -1 * out_grad * wgt_tgt / total_w, 0)
    inp_grad_ptrs = inp_grad_ptr + offsets_n * C + tgt
    tl.store(inp_grad_ptrs, inp_grad, mask=mask_n)


@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss2d_forward_kernel(
    inp_ptr,
    tgt_ptr,
    wgt_ptr,
    out_ptr,
    ignore_index,
    N,
    C,
    D,
    reduction: tl.constexpr = 1,
    BLOCK_ND: tl.constexpr = 128,
):
    pid_nd = tl.program_id(0)
    offset_nd = pid_nd * BLOCK_ND + tl.arange(0, BLOCK_ND)
    offset_d = offset_nd % D
    offset_n = offset_nd // D

    mask_block = offset_nd < N * D

    tgt_ptrs = tgt_ptr + offset_n * D + offset_d
    tgt = tl.load(tgt_ptrs, mask=mask_block, other=0)
    assert tgt >= 0 and tgt < C, "Invalid target value"
    ignore_mask = (tgt != ignore_index) & mask_block

    if wgt_ptr is None:
        wgt_tgt = ignore_mask.to(tl.float32)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    inp_tgt_ptrs = inp_ptr + offset_n * C * D + tgt * D + offset_d
    inp_tgt = tl.load(inp_tgt_ptrs, mask=ignore_mask, other=0).to(tl.float32)
    out = inp_tgt * wgt_tgt * -1

    # none
    if reduction == 0:
        out_ptrs = out_ptr + offset_n * D + offset_d
        tl.store(out_ptrs, out, mask=mask_block)
    # mean
    elif reduction == 1:
        total_out = tl.sum(out)
        total_wgt = tl.sum(wgt_tgt)
        tl.store(out_ptr, total_out)
        tl.store(out_ptr + 1, total_wgt)
        tl.store(out_ptr + 3, total_out / total_wgt)
    # sum
    else:
        total_out = tl.sum(out)
        tl.store(out_ptr, total_out)


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss2d_backward_kernel(
    out_grad_ptr,
    tgt_ptr,
    wgt_ptr,
    inp_grad_ptr,
    ignore_index,
    total_weight,
    N,
    C,
    D,
    reduction: tl.constexpr = 1,
    BLOCK_ND: tl.constexpr = 128,
):
    pid_nd = tl.program_id(0)
    offset_nd = pid_nd * BLOCK_ND + tl.arange(0, BLOCK_ND)
    offset_d = offset_nd % D
    offset_n = offset_nd // D

    mask_block = offset_nd < N * D

    tgt_ptrs = tgt_ptr + offset_n * D + offset_d
    tgt = tl.load(tgt_ptrs, mask=mask_block, other=0)
    ignore_mask = (tgt != ignore_index) & mask_block

    if wgt_ptr is None:
        wgt_tgt = ignore_mask.to(tl.float32)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    if reduction == 0:
        out_grad_ptrs = out_grad_ptr + offset_n * D + offset_d
        out_grad = tl.load(out_grad_ptrs, mask=mask_block, other=0).to(tl.float32)
    else:
        out_grad = tl.load(out_grad_ptr).to(tl.float32)

    if reduction == 1:
        total_w = tl.load(total_weight).to(tl.float32)
    else:
        total_w = 1
    inp_grad = tl.where(ignore_mask, -1 * out_grad * wgt_tgt / total_w, 0)
    inp_grad_ptrs = inp_grad_ptr + offset_n * C * D + tgt * D + offset_d
    tl.store(inp_grad_ptrs, inp_grad, mask=mask_block)


# 1d & 2d tensor
def nll_loss_forward(self, target, weight=None, reduction=1, ignore_index=-100):
    logger.debug("GEMS_SPACEMIT NLL_LOSS_FORWARD")
    assert self.ndim <= 2, "Invalid input ndim"
    shape = list(target.shape)
    N = 1 if self.ndim == 1 else self.shape[0]
    C = self.shape[-1]
    assert target.numel() == N, "Invalid target size"

    self = self.contiguous()
    target = target.contiguous()
    weight = None if weight is None else weight.contiguous()

    # reduction: 0-None, 1-mean, 2-sum
    if reduction == 0:
        out = torch.empty(shape, dtype=self.dtype, device=self.device)
    elif reduction == 1:
        out = torch.zeros([4], dtype=torch.float32, device=self.device)
    else:
        out = torch.zeros([], dtype=torch.float32, device=self.device)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    with torch_device_fn.device(self.device):
        nll_loss_forward_kernel[grid](
            self, target, weight, out, ignore_index, N, C, reduction,
        )

    if reduction == 0:
        output = out
        total_weight = torch.empty([], dtype=self.dtype, device=self.device)
    elif reduction == 1:
        out = out.to(self.dtype)
        output = out[3]
        total_weight = out[1]
    else:
        output = out.to(self.dtype)
        total_weight = torch.empty([], dtype=self.dtype, device=self.device)

    return output, total_weight


def nll_loss_backward(
    grad_output, self, target, weight=None, reduction=1,
    ignore_index=-100, total_weight=None,
):
    logger.debug("GEMS_SPACEMIT NLL_LOSS_BACKWARD")
    N = 1 if self.ndim == 1 else self.shape[0]
    C = self.shape[-1]

    grad_output = grad_output.contiguous()
    target = target.contiguous()
    weight = None if weight is None else weight.contiguous()

    grad_input = torch.zeros_like(self).contiguous()

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    with torch_device_fn.device(self.device):
        nll_loss_backward_kernel[grid](
            grad_output, target, weight, grad_input,
            ignore_index, total_weight, N, C, reduction,
        )

    return grad_input


# 3d+ tensor
def nll_loss2d_forward(self, target, weight=None, reduction=1, ignore_index=-100):
    logger.debug("GEMS_SPACEMIT NLL_LOSS2D_FORWARD")
    assert self.ndim == 4, "Invalid input ndim"

    shape = list(target.shape)
    N, C, _, D = self.shape
    assert shape == [N, 1, D], "Invalid target size"

    self = self.contiguous()
    target = target.contiguous()
    weight = None if weight is None else weight.contiguous()

    if reduction == 0:
        out = torch.empty(shape, dtype=self.dtype, device=self.device)
    elif reduction == 1:
        out = torch.zeros([4], dtype=torch.float32, device=self.device)
    else:
        out = torch.zeros([], dtype=torch.float32, device=self.device)

    grid = lambda meta: (triton.cdiv(N * D, meta["BLOCK_ND"]),)
    with torch_device_fn.device(self.device):
        nll_loss2d_forward_kernel[grid](
            self, target, weight, out, ignore_index, N, C, D, reduction
        )

    if reduction == 0:
        output = out
        total_weight = torch.empty([], dtype=self.dtype, device=self.device)
    elif reduction == 1:
        out = out.to(self.dtype)
        output = out[3]
        total_weight = out[1]
    else:
        output = out.to(self.dtype)
        total_weight = torch.empty([], dtype=self.dtype, device=self.device)

    return output, total_weight


def nll_loss2d_backward(
    grad_output, self, target, weight=None, reduction=1,
    ignore_index=-100, total_weight=None,
):
    logger.debug("GEMS_SPACEMIT NLL_LOSS2D_BACKWARD")
    N, C, _, D = self.shape

    grad_output = grad_output.contiguous()
    target = target.contiguous()
    weight = None if weight is None else weight.contiguous()

    grad_input = torch.zeros_like(self).contiguous()

    grid = lambda meta: (triton.cdiv(N * D, meta["BLOCK_ND"]),)
    with torch_device_fn.device(self.device):
        nll_loss2d_backward_kernel[grid](
            grad_output, target, weight, grad_input,
            ignore_index, total_weight, N, C, D, reduction,
        )

    return grad_input
