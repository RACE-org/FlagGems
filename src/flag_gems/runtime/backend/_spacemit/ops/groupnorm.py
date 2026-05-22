import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

rsqrt = tl_extra_shim.rsqrt
logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def group_norm_kernel(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    group_size,
    C,
    HW,
    num_groups,
    num_tasks,
    eps,
    BLOCK_GROUP_SIZE: tl.constexpr,
    BLOCK_HW_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    sub_num = tl.cdiv(max(num_tasks - pid, 0), num_ctas)

    num_elements = group_size * HW
    group_offset = tl.arange(0, BLOCK_GROUP_SIZE)
    hw_offset = tl.arange(0, BLOCK_HW_SIZE)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        group = task_idx % num_groups

        wb_offset = group * group_size + group_offset
        xy_mask = (wb_offset[:, None] < C) & (hw_offset[None, :] < HW)

        X_ptr = tl.make_block_ptr(
            base=X + task_idx * num_elements,
            shape=(group_size, HW),
            strides=(HW, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_GROUP_SIZE, BLOCK_HW_SIZE),
            order=(1, 0),
        )
        Y_ptr = tl.make_block_ptr(
            base=Y + task_idx * num_elements,
            shape=(group_size, HW),
            strides=(HW, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_GROUP_SIZE, BLOCK_HW_SIZE),
            order=(1, 0),
        )

        Mean_ptr = tl.make_block_ptr(
            base=Mean,
            shape=(num_tasks,),
            strides=(1,),
            offsets=(task_idx,),
            block_shape=(1,),
            order=(0,),
        )
        Rstd_ptr = tl.make_block_ptr(
            base=Rstd,
            shape=(num_tasks,),
            strides=(1,),
            offsets=(task_idx,),
            block_shape=(1,),
            order=(0,),
        )

        X_val = tl.load(X_ptr, boundary_check=(0, 1))
        x_dtype = X_val.dtype
        X_val = X_val.to(tl.float32)
        mean = tl.sum(X_val) / num_elements
        x = tl.where(xy_mask, X_val - mean, 0.0)

        var = tl.sum(x * x) / num_elements
        rstd = rsqrt(var + eps)
        x_hat = x * rstd

        if W is None:
            weight = 1
        else:
            w_block_ptr = tl.make_block_ptr(
                base=W + group * group_size,
                shape=(group_size,),
                strides=(1,),
                offsets=(0,),
                block_shape=(BLOCK_GROUP_SIZE,),
                order=(0,),
            )
            weight = tl.load(w_block_ptr, boundary_check=(0,))
            weight = tl.view(weight, (BLOCK_GROUP_SIZE, 1))
        if B is None:
            bias = 0
        else:
            b_block_ptr = tl.make_block_ptr(
                base=B + group * group_size,
                shape=(group_size,),
                strides=(1,),
                offsets=(0,),
                block_shape=(BLOCK_GROUP_SIZE,),
                order=(0,),
            )
            bias = tl.load(b_block_ptr, boundary_check=(0,))
            bias = tl.view(bias, (BLOCK_GROUP_SIZE, 1))
        Y_val = x_hat * weight + bias
        Y_val = Y_val.to(x_dtype)

        tl.store(Y_ptr, Y_val, boundary_check=(0, 1))
        tl.store(Mean_ptr, mean.to(x_dtype))
        tl.store(Rstd_ptr, rstd.to(x_dtype))


@libentry()
@triton.jit(do_not_specialize=["eps"])
def group_norm_fulltile_kernel(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    group_size,
    HW,
    num_groups,
    num_tasks,
    eps,
    BLOCK_GROUP_SIZE: tl.constexpr,
    BLOCK_HW_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    sub_num = tl.cdiv(max(num_tasks - pid, 0), num_ctas)

    num_elements = group_size * HW
    group_offset = tl.arange(0, BLOCK_GROUP_SIZE)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        group = task_idx % num_groups

        X_ptr = tl.make_block_ptr(
            base=X + task_idx * num_elements,
            shape=(group_size, HW),
            strides=(HW, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_GROUP_SIZE, BLOCK_HW_SIZE),
            order=(1, 0),
        )
        Y_ptr = tl.make_block_ptr(
            base=Y + task_idx * num_elements,
            shape=(group_size, HW),
            strides=(HW, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_GROUP_SIZE, BLOCK_HW_SIZE),
            order=(1, 0),
        )

        X_val = tl.load(X_ptr, boundary_check=(0, 1))
        x_dtype = X_val.dtype
        X_val = X_val.to(tl.float32)
        mean = tl.sum(X_val) / num_elements
        x = X_val - mean

        var = tl.sum(x * x) / num_elements
        rstd = rsqrt(var + eps)
        x_hat = x * rstd

        if W is None:
            weight = 1
        else:
            weight = tl.load(W + group * group_size + group_offset)
            weight = tl.view(weight, (BLOCK_GROUP_SIZE, 1))
        if B is None:
            bias = 0
        else:
            bias = tl.load(B + group * group_size + group_offset)
            bias = tl.view(bias, (BLOCK_GROUP_SIZE, 1))

        Y_val = (x_hat * weight + bias).to(x_dtype)
        tl.store(Y_ptr, Y_val, boundary_check=(0, 1))
        tl.store(Mean + task_idx, mean.to(x_dtype))
        tl.store(Rstd + task_idx, rstd.to(x_dtype))


@libentry()
@triton.jit(do_not_specialize=["eps"])
def group_norm_small_group_kernel(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    group_size,
    C,
    HW,
    num_groups,
    num_tasks,
    eps,
    BLOCK_GROUP_SIZE: tl.constexpr,
    BLOCK_HW_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    sub_num = tl.cdiv(max(num_tasks - pid, 0), num_ctas)

    num_elements = group_size * HW
    group_offset = tl.arange(0, BLOCK_GROUP_SIZE)
    hw_offset = tl.arange(0, BLOCK_HW_SIZE)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        group = task_idx % num_groups

        wb_offset = group * group_size + group_offset
        wb_mask = wb_offset < C
        out_dtype = Mean.dtype.element_ty

        Mean_ptr = tl.make_block_ptr(
            base=Mean,
            shape=(num_tasks,),
            strides=(1,),
            offsets=(task_idx,),
            block_shape=(1,),
            order=(0,),
        )
        Rstd_ptr = tl.make_block_ptr(
            base=Rstd,
            shape=(num_tasks,),
            strides=(1,),
            offsets=(task_idx,),
            block_shape=(1,),
            order=(0,),
        )

        row_sum = tl.zeros([BLOCK_GROUP_SIZE], dtype=tl.float32)
        row_sq_sum = tl.zeros([BLOCK_GROUP_SIZE], dtype=tl.float32)

        for off in range(0, HW, BLOCK_HW_SIZE):
            cur_hw = off + hw_offset
            hw_mask = cur_hw < HW
            xy_offset = task_idx * num_elements + group_offset[:, None] * HW + cur_hw[None, :]
            xy_mask = wb_mask[:, None] & hw_mask[None, :]
            x_val = tl.load(X + xy_offset, mask=xy_mask, other=0.0).to(tl.float32)
            row_sum += tl.sum(x_val, axis=1)
            row_sq_sum += tl.sum(x_val * x_val, axis=1)

        sum_val = tl.sum(row_sum)
        sumsq_val = tl.sum(row_sq_sum)
        mean = sum_val / num_elements
        var = sumsq_val / num_elements - mean * mean
        rstd = rsqrt(var + eps)

        if W is None:
            weight = 1
        else:
            weight = tl.load(W + wb_offset, mask=wb_mask, other=0.0).to(tl.float32)[:, None]
        if B is None:
            bias = 0
        else:
            bias = tl.load(B + wb_offset, mask=wb_mask, other=0.0).to(tl.float32)[:, None]

        for off in range(0, HW, BLOCK_HW_SIZE):
            cur_hw = off + hw_offset
            hw_mask = cur_hw < HW
            xy_offset = task_idx * num_elements + group_offset[:, None] * HW + cur_hw[None, :]
            xy_mask = wb_mask[:, None] & hw_mask[None, :]
            x_val = tl.load(X + xy_offset, mask=xy_mask, other=0.0)
            x_dtype = x_val.dtype
            x_val_f32 = x_val.to(tl.float32)
            x_hat = tl.where(xy_mask, (x_val_f32 - mean) * rstd, 0.0)
            y_val = (x_hat * weight + bias).to(x_dtype)
            tl.store(Y + xy_offset, y_val, mask=xy_mask)

        tl.store(Mean_ptr, mean.to(out_dtype))
        tl.store(Rstd_ptr, rstd.to(out_dtype))


@libentry()
@triton.jit
def group_norm_backward_kernel(
    grad_y,
    X,
    W,
    Mean,
    Rstd,
    num_groups,
    group_size,
    grad_x,
    C,
    HW,
    num_tasks,
    BLOCK_GROUP_SIZE: tl.constexpr,
    BLOCK_HW_SIZE: tl.constexpr = 128,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    sub_num = tl.cdiv(max(num_tasks - pid, 0), num_ctas)
    num_elements = group_size * HW
    group_offset = tl.arange(0, BLOCK_GROUP_SIZE)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        group = task_idx % num_groups
        wb_offset = group * group_size + group_offset
        wb_mask = wb_offset < C

        rstd = tl.load(Rstd + task_idx).to(tl.float32)
        mean = tl.load(Mean + task_idx).to(tl.float32)
        if W is None:
            weight = 1
        else:
            weight = tl.load(W + wb_offset, mask=wb_mask, other=0.0).to(tl.float32)[:, None]

        dx_part2 = tl.zeros([BLOCK_GROUP_SIZE, BLOCK_HW_SIZE], dtype=tl.float32)
        dx_part3 = tl.zeros([BLOCK_GROUP_SIZE, BLOCK_HW_SIZE], dtype=tl.float32)
        for off in range(0, HW, BLOCK_HW_SIZE):
            hw_offset = off + tl.arange(0, BLOCK_HW_SIZE)
            hw_mask = hw_offset < HW
            xy_offset = task_idx * num_elements + group_offset[:, None] * HW + hw_offset[None, :]
            xy_mask = wb_mask[:, None] & hw_mask[None, :]

            dY_val = tl.load(grad_y + xy_offset, mask=xy_mask, other=0.0).to(tl.float32)
            X_val = tl.load(X + xy_offset, mask=xy_mask, other=0.0).to(tl.float32)

            x_hat = tl.where(xy_mask, rstd * (X_val - mean), 0.0)
            dx_hat = weight * dY_val
            dx_part2 += dx_hat
            dx_part3 += dx_hat * x_hat

        dx_2 = tl.sum(dx_part2)
        dx_3 = tl.sum(dx_part3)

        for off in range(0, HW, BLOCK_HW_SIZE):
            hw_offset = off + tl.arange(0, BLOCK_HW_SIZE)
            hw_mask = hw_offset < HW
            xy_offset = task_idx * num_elements + group_offset[:, None] * HW + hw_offset[None, :]
            xy_mask = wb_mask[:, None] & hw_mask[None, :]

            dY_val = tl.load(grad_y + xy_offset, mask=xy_mask, other=0.0).to(tl.float32)
            X_val = tl.load(X + xy_offset, mask=xy_mask, other=0.0).to(tl.float32)

            x_hat = tl.where(xy_mask, rstd * (X_val - mean), 0.0)
            dx_hat = weight * dY_val
            dx = rstd * (dx_hat - (dx_2 + x_hat * dx_3) / num_elements)

            tl.store(grad_x + xy_offset, dx, xy_mask)


@libentry()
@triton.jit
def weight_bias_backward_kernel(
    dY,
    X,
    Mean,
    Rstd,
    dW,
    dB,
    num_groups,
    group_size,
    N,
    C,
    HW,
    BLOCK_N: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    group = pid // group_size
    n_offset = tl.arange(0, BLOCK_N)
    hw_offset = tl.arange(0, BLOCK_HW)
    xy_mask = (n_offset[:, None] < N) & (hw_offset[None, :] < HW)
    mr_mask = n_offset < N

    mean_ptr = Mean + group + n_offset * num_groups
    rstd_ptr = Rstd + group + n_offset * num_groups

    dY_ptr = dY + pid * HW + n_offset[:, None] * C * HW + hw_offset[None, :]
    x_ptr = X + pid * HW + n_offset[:, None] * C * HW + hw_offset[None, :]

    grad_y = tl.load(dY_ptr, mask=xy_mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr, mask=xy_mask, other=0.0)
    x_f32 = x.to(tl.float32)
    mean = tl.load(mean_ptr, mask=mr_mask, other=0.0).to(tl.float32)[:, None]
    rstd = tl.load(rstd_ptr, mask=mr_mask, other=0.0).to(tl.float32)[:, None]

    if dW is not None:
        dw = tl.sum((x_f32 - mean) * rstd * grad_y)
        tl.store(dW + pid, dw)
    if dB is not None:
        db = tl.sum(grad_y)
        tl.store(dB + pid, db)


def group_norm(input, weight, bias, N, C, HxW, group, eps=1e-05):
    logger.debug("GEMS_SPACEMIT GROUPNORM_FORWARD")

    group_size = triton.cdiv(C, group)
    input = input.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()

    y = torch.empty_like(input)
    mean = torch.empty((N, group), dtype=input.dtype, device=input.device)
    rstd = torch.empty((N, group), dtype=input.dtype, device=input.device)

    num_tasks = N * group
    with torch_device_fn.device(input.device):
        if (
            C % group == 0
            and group_size == triton.next_power_of_2(group_size)
            and HxW == triton.next_power_of_2(HxW)
            and group_size <= 4
            and HxW <= 2048
        ):
            num_ctas = min(16, max(1, num_tasks))
            group_norm_fulltile_kernel[(num_ctas,)](
                input,
                y,
                weight,
                bias,
                mean,
                rstd,
                group_size,
                HxW,
                group,
                num_tasks,
                eps,
                BLOCK_GROUP_SIZE=group_size,
                BLOCK_HW_SIZE=HxW,
            )
        elif group_size <= 4 and HxW >= 4096:
            num_ctas = min(16, max(1, num_tasks))
            group_norm_small_group_kernel[(num_ctas,)](
                input,
                y,
                weight,
                bias,
                mean,
                rstd,
                group_size,
                C,
                HxW,
                group,
                num_tasks,
                eps,
                BLOCK_GROUP_SIZE=triton.next_power_of_2(group_size),
                BLOCK_HW_SIZE=128,
            )
        else:
            num_ctas = min(16, max(1, num_tasks))
            group_norm_kernel[(num_ctas,)](
                input,
                y,
                weight,
                bias,
                mean,
                rstd,
                group_size,
                C,
                HxW,
                group,
                num_tasks,
                eps,
                BLOCK_GROUP_SIZE=triton.next_power_of_2(group_size),
                BLOCK_HW_SIZE=triton.next_power_of_2(HxW),
            )
    return y, mean, rstd


def group_norm_backward(
    grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask
):
    logger.debug("GEMS_SPACEMIT GROUPNORM_BACKWARD")

    grad_out = grad_out.contiguous()
    input = input.contiguous()
    mean = mean.contiguous()
    rstd = rstd.contiguous()
    weight = None if weight is None else weight.contiguous()
    group_size = triton.cdiv(C, group)

    if output_mask[0]:
        grad_inp = torch.empty_like(input)
        num_tasks = N * group
        num_ctas = min(16, max(1, num_tasks))
        grid = (num_ctas,)
        with torch_device_fn.device(input.device):
            group_norm_backward_kernel[grid](
                grad_out,
                input,
                weight,
                mean,
                rstd,
                group,
                group_size,
                grad_inp,
                C,
                HxW,
                num_tasks,
                BLOCK_GROUP_SIZE=triton.next_power_of_2(group_size),
            )
    else:
        grad_inp = None

    if output_mask[1] is False and output_mask[2] is False:
        return grad_inp, None, None

    weight_grad = torch.empty_like(weight) if output_mask[1] else None
    bias_grad = torch.empty_like(weight) if output_mask[2] else None
    with torch_device_fn.device(input.device):
        weight_bias_backward_kernel[(C, 1, 1)](
            grad_out,
            input,
            mean,
            rstd,
            weight_grad,
            bias_grad,
            group,
            group_size,
            N,
            C,
            HxW,
            BLOCK_N=triton.next_power_of_2(N),
            BLOCK_HW=triton.next_power_of_2(HxW),
        )
    return grad_inp, weight_grad, bias_grad
