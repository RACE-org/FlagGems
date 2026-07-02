import logging

import triton
import triton.language as tl

from flag_gems.utils.codegen_config_utils import CodeGenConfig
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

op_config = CodeGenConfig(
    max_tile_size=65536,
    max_grid_size=(16, 16, 16),
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=False,
)


@pointwise_dynamic(is_tensor=[True, False, False], promotion_methods=[(0, "DEFAULT")], config=op_config)
@triton.jit
def threshold_kernel(self, threshold, value):
    return tl.where(self > threshold, self, value)


def threshold(self, threshold, value):
    logger.debug("GEMS THRESHOLD FORWARD")
    output = threshold_kernel(self, threshold, value)
    return output
