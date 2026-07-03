import logging

import triton
import triton.language as tl

from flag_gems.utils.pointwise_dynamic import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import CodeGenConfig

op_config = CodeGenConfig(
    max_tile_size=65536,
    max_grid_size=(16, 16, 16),
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=False,
)


logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")], config=op_config)
@triton.jit
def logical_or_func(x, y):
    return x.to(tl.int1).logical_or(y.to(tl.int1))


def logical_or(A, B):
    logger.debug("GEMS LOGICAL_OR")
    return logical_or_func(A, B)
