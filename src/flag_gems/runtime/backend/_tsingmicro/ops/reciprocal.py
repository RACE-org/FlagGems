import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

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


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=my_config)
@triton.jit
def reciprocal_func(x):
    return 1.0 / x.to(tl.float32)


def reciprocal(A):
    logger.debug("GEMS RECIPROCAL")
    return reciprocal_func(A)


def reciprocal_(A):
    logger.debug("GEMS RECIPROCAL_")
    return reciprocal_func(A, out0=A)
