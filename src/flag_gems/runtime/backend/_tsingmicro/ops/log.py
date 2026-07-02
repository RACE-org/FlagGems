import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

from flag_gems.utils.codegen_config_utils import CodeGenConfig
my_config = CodeGenConfig(
    max_tile_size= 512 * 512,
    max_grid_size=(16, 16, 16),
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=False,
)

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")], config=my_config)
@triton.jit
def log_func(x):
    return tl.log(x.to(tl.float32))


def log(A):
    logger.debug("[TSING] GEMS LOG")
    return log_func(A)
