import logging

import triton
import triton.language as tl

from flag_gems.utils.codegen_config_utils import CodeGenConfig
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic

erf_config = CodeGenConfig(
    max_tile_size=65536,
    max_grid_size=(16, 16, 16),
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=False,
)

logger = logging.getLogger(__name__)

@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=erf_config)
@triton.jit
def erf_func(x):
    x_f32 = x.to(tl.float32)
    sign = x_f32 >= 0.0
    sign = (2.0 * sign - 1.0)

    P: tl.constexpr = 0.3275911
    A1: tl.constexpr = 0.254829592
    A2: tl.constexpr = -0.284496736
    A3: tl.constexpr = 1.421413741
    A4: tl.constexpr = -1.453152027
    A5: tl.constexpr = 1.061405429
    x_abs = tl.abs(x_f32)
    t = 1 / (1.0 + x_abs * P)
    exp = tl.exp(-x_abs * x_abs)
    mul_add = ((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t
    y = 1.0 - mul_add * exp

    return sign * y


def erf(x):
    logger.debug("GEMS ERF")
    return erf_func(x)


def erf_(x):
    logger.debug("GEMS ERF_")
    return erf_func(x, out0=x)
