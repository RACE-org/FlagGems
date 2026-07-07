import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.randn import randn_kernel
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.random_utils import philox_backend_seed_offset
from flag_gems.utils.shape_utils import broadcast_shapes, volume

logger = logging.getLogger(__name__)

UNROLL = 4
NUM_CTAS = 8


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("eq_tensor_tensor"),
    key=["n_elements"],
)
@triton.jit
def normal_transform_kernel_tt(
    Val_ptr,
    Std_ptr,
    Mean_ptr,
    Out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)

    total_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(total_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        val_blk = tl.make_block_ptr(
            base=Val_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        std_blk = tl.make_block_ptr(
            base=Std_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        mean_blk = tl.make_block_ptr(
            base=Mean_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        out_blk = tl.make_block_ptr(
            base=Out_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )

        val = tl.load(val_blk, boundary_check=(0,)).to(tl.float32)
        std = tl.load(std_blk, boundary_check=(0,)).to(tl.float32)
        mean = tl.load(mean_blk, boundary_check=(0,)).to(tl.float32)
        out = val * std + mean
        tl.store(out_blk, out.to(Out_ptr.type.element_ty), boundary_check=(0,))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("eq_tensor_tensor"),
    key=["n_elements"],
)
@triton.jit
def normal_transform_kernel_tf(
    Val_ptr,
    Mean_ptr,
    Out_ptr,
    std,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)

    total_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(total_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        val_blk = tl.make_block_ptr(
            base=Val_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        mean_blk = tl.make_block_ptr(
            base=Mean_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        out_blk = tl.make_block_ptr(
            base=Out_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )

        val = tl.load(val_blk, boundary_check=(0,)).to(tl.float32)
        mean = tl.load(mean_blk, boundary_check=(0,)).to(tl.float32)
        out = val * std + mean
        tl.store(out_blk, out.to(Out_ptr.type.element_ty), boundary_check=(0,))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("eq_tensor_tensor"),
    key=["n_elements"],
)
@triton.jit
def normal_transform_kernel_ft(
    Val_ptr,
    Std_ptr,
    Out_ptr,
    mean,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)

    total_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(total_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        val_blk = tl.make_block_ptr(
            base=Val_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        std_blk = tl.make_block_ptr(
            base=Std_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        out_blk = tl.make_block_ptr(
            base=Out_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )

        val = tl.load(val_blk, boundary_check=(0,)).to(tl.float32)
        std = tl.load(std_blk, boundary_check=(0,)).to(tl.float32)
        out = val * std + mean
        tl.store(out_blk, out.to(Out_ptr.type.element_ty), boundary_check=(0,))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("eq_tensor_scalar"),
    key=["n_elements"],
)
@triton.jit
def normal_transform_kernel_ff(
    Val_ptr,
    Out_ptr,
    std,
    mean,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)

    total_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(total_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        val_blk = tl.make_block_ptr(
            base=Val_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        out_blk = tl.make_block_ptr(
            base=Out_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )

        val = tl.load(val_blk, boundary_check=(0,)).to(tl.float32)
        out = val * std + mean
        tl.store(out_blk, out.to(Out_ptr.type.element_ty), boundary_check=(0,))


def normal_distribution(shape, device, *, generator=None, out=None):
    if out is None:
        out = torch.empty(shape, device=device, dtype=torch.float32)
    N = volume(shape)
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)

    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(device):
        randn_kernel[grid_fn](out, N, philox_seed, philox_offset)
    return out


def normal_tensor_tensor(mean, std, *, generator=None):
    logger.debug("GEMS_SPACEMIT NORMAL_TENSOR_TENSOR")
    shape = broadcast_shapes([mean.shape, std.shape])
    device = mean.device
    out = normal_distribution(shape, device)
    mean = torch.broadcast_to(mean, shape).contiguous()
    std = torch.broadcast_to(std, shape).contiguous()
    n = out.numel()
    normal_transform_kernel_tt[(NUM_CTAS,)](out, std, mean, out, n)
    return out


def normal_tensor_float(mean, std, *, generator=None):
    logger.debug("GEMS_SPACEMIT NORMAL_TENSOR_FLOAT")
    shape = mean.shape
    device = mean.device
    out = normal_distribution(shape, device)
    mean = mean.contiguous()
    n = out.numel()
    normal_transform_kernel_tf[(NUM_CTAS,)](out, mean, out, float(std), n)
    return out


def normal_float_tensor(mean, std, *, generator=None):
    logger.debug("GEMS_SPACEMIT NORMAL_FLOAT_TENSOR")
    shape = std.shape
    device = std.device
    out = normal_distribution(shape, device)
    std = std.contiguous()
    n = out.numel()
    normal_transform_kernel_ft[(NUM_CTAS,)](out, std, out, float(mean), n)
    return out


def normal_(self, mean=0, std=1, *, generator=None):
    logger.debug("GEMS_SPACEMIT NORMAL_")
    shape = self.shape
    device = self.device
    self = normal_distribution(shape, device, generator=generator, out=self)
    n = self.numel()
    normal_transform_kernel_ff[(NUM_CTAS,)](self, self, float(std), float(mean), n)
    return self