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

@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")], config=my_config)
@triton.jit
def log_sigmoid_forward(x):
    return tl.minimum(x, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(x).to(tl.float32)))


def log_sigmoid(x):
    logger.debug("GEMS LOG_SIGMOID FORWARD")

    return log_sigmoid_forward(x)
