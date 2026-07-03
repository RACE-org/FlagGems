import logging

import torch
import triton

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import CodeGenConfig
from flag_gems.utils.tensor_wrapper import StridedBuffer

# max_tile_size=8192 的原因：
# flip 多维度反转向量 (negative strides) 时，kernel 生成 rdma→wdma 纯 DMA 异步指令管线。
# >8192 时 CGRA 反向 DMA 处理时间超出流水线协调窗口，前序 DMA 未完成后续指令即读取 SPM，
# 导致非确定性错误（0.8%~19% 元素不匹配）。ENABLE_SYNCHRONOUS_INTRINSIC=1
# (每条指令后 RcsWaitfinish) 可消除但损害性能，故编译器侧用 tile 上限规避。
# 实测安全阈值：8192 稳定，16384 偶发错误。
my_config = CodeGenConfig(
    max_tile_size=4096,
    max_grid_size=(16, 16, 16),
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=False,
)

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")], config=my_config)
@triton.jit
def copy_func(x):
    return x


def flip(A: torch.Tensor, dims) -> torch.Tensor:
    logger.debug("GEMS FLIP")
    strides = list(A.stride())
    flip_dims_b = [False for _ in A.stride()]
    for dim in dims:
        dim = dim % A.ndim
        flip_dims_b[dim] = not flip_dims_b[dim]
    n = 0
    offset = 0
    for i in range(len(flip_dims_b)):
        if flip_dims_b[i] and A.size(i) > 1 and A.stride(i) != 0:
            offset += strides[i] * (A.shape[i] - 1)
            strides[i] = -strides[i]
            n += 1
    if n == 0 or A.numel() <= 1:
        return A.clone()
    out = torch.empty_like(A)
    flipped_A = StridedBuffer(A, strides=strides, offset=offset)
    overload = copy_func.instantiate(A.ndim)
    overload(flipped_A, out0=out)
    return out
