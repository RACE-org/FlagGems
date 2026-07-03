import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

div_rn = tl_extra_shim.div_rn
exp2 = tl_extra_shim.exp2

logger = logging.getLogger(__name__)

from flag_gems.utils.codegen_config_utils import CodeGenConfig


from flag_gems.utils.pointwise_dynamic import pointwise_dynamic

my_config = CodeGenConfig(
    max_tile_size= 65536,
    max_grid_size=(16, 16, 16),
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=True,
)

@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=my_config)
@triton.jit
def silu_and_mul_kernel(x, y):
    x_fp32 = x.to(tl.float32)
    log2e: tl.constexpr = 1.4426950408889634
    x_silu = x_fp32 / (1 + exp2(-x.to(tl.float32) * log2e))
    return x_silu * y


@pointwise_dynamic(
    promotion_methods=[(0, 1, 2, "DEFAULT"), (0, 1, 2, "DEFAULT")], num_outputs=2, config=my_config
)
@triton.jit
def silu_and_mul_grad_kernel(x, y, dgrad):
    x_fp32 = x.to(tl.float32)
    sig = 1 / (1 + tl.exp(-x_fp32))
    x_silu = x_fp32 * sig
    d_x_silu = sig * (1 + x_fp32 * (1 - sig))
    dx = d_x_silu * dgrad * y
    dy = dgrad * x_silu
    return dx, dy


class SiluAndMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        logger.debug("GEMS SILU AND MUL FORWARD")
        return silu_and_mul_kernel(A, B)

    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A, grad_B = silu_and_mul_grad_kernel(A, B, grad_output)
        return grad_A, grad_B


def silu_and_mul(A, B):
    return SiluAndMul.apply(A, B)
