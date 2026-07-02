import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim
from flag_gems.utils.codegen_config_utils import CodeGenConfig

# max_tile_size=4096 的原因：
# angle_float_and_int 路径生成 rdma→boolgreatrequalvv→bit2fp→mask_move→wdma 异步指令管线。
# >4096 时 CGRA 处理时间超出流水线协调窗口，后序指令读到前序未完成写入的 SPM 数据，
# 导致 compare+select 非确定性错误（0.01%~4.4% 元素）。ENABLE_SYNCHRONOUS_INTRINSIC=1
# (每条指令后 RcsWaitfinish) 可消除但损害性能，故编译器侧用 tile 上限规避。
my_config = CodeGenConfig(
    max_tile_size=4096,
    max_grid_size=(16, 16, 16),
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=False,
)

atan2 = tl_extra_shim.atan2

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, "DEFAULT")], config=my_config)
@triton.jit
def angle_func(real, imag):
    real_last, imag_last = (
        (real.to(tl.float32), imag.to(tl.float32))
        if real.dtype == tl.float16
        else (real, imag)
    )
    result = atan2(imag_last, real_last)
    return result


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "INT_TO_FLOAT")], config=my_config)
@triton.jit
def angle_float_and_int(real):
    zero = 0.0
    pi = math.pi
    real_positive = real >= zero
    result = tl.where(real_positive, zero, pi)
    return result


def angle(input_tensor: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS ANGLE")
    if input_tensor.dtype == torch.complex32 or input_tensor.dtype == torch.complex64:
        real = input_tensor.real
        imag = input_tensor.imag
        return angle_func(real, imag)
    else:
        real = input_tensor
        return angle_float_and_int(real)
