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

@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=my_config)
@triton.jit
def relu_forward(x):
    return tl.where(x > 0, x, 0)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=my_config)
@triton.jit
def relu_backward(x, dy):
    return tl.where(x > 0, dy, 0)


def relu(self):
    logger.debug("GEMS RELU FORWARD")
    output = relu_forward(self)
    return output


def relu_(A):
    logger.debug("GEMS RELU_ FORWARD")
    out = relu_forward(A, out0=A)
    return out
