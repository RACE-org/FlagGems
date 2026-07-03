import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

from flag_gems.utils.codegen_config_utils import CodeGenConfig

my_config = CodeGenConfig(
    max_tile_size= 65536,
    max_grid_size=(16, 16, 16),
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=True,
)

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")], config=my_config)
@triton.jit
def ne_func(x, y):
    return x.to(tl.float32) != y.to(tl.float32)


def ne(A, B):
    logger.debug("GEMS NE")
    return ne_func(A, B)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")], config=my_config)
@triton.jit
def ne_func_scalar(x, y):
    return x.to(tl.float32) != y.to(tl.float32)


def ne_scalar(A, B):
    logger.debug("GEMS NE SCALAR")
    return ne_func_scalar(A, B)
