import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


from flag_gems.utils.codegen_config_utils import CodeGenConfig

my_config = CodeGenConfig(
    max_tile_size= 65536,
    max_grid_size=(16, 16, 16),
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=False,
)

@pointwise_dynamic(
    is_tensor=[True, False, False, False], promotion_methods=[(0, "DEFAULT")], config=my_config
)
@triton.jit
def elu_forward_kernel(x, alpha, scale, input_scale):
    return tl.where(
        x > 0,
        scale * input_scale * x,
        scale * alpha * (tl.exp(x.to(tl.float32) * input_scale) - 1),
    )


def elu(A, alpha=1.0, scale=1.0, input_scale=1.0):
    logger.debug("GEMS ELU")
    return elu_forward_kernel(A, alpha, scale, input_scale)
