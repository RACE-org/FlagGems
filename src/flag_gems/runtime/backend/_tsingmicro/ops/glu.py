import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils.pointwise_dynamic import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import CodeGenConfig

glu_config = CodeGenConfig(
    max_tile_size=65536,
    max_grid_size=(16, 16, 16),
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=False,
)

@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=glu_config)
@triton.jit
def glu_kernel(a, b):
    sigmoid_b = 1 / (1 + tl.exp(-b.to(tl.float32)))
    result = a * sigmoid_b

    return result


def glu(self, dim=-1):
    assert self.shape[dim] % 2 == 0, "Split dimension must be even"
    logging.debug("GLU FORWARD")
    # Split into a and b
    a, b = torch.chunk(self, 2, dim=dim)
    out = glu_kernel(a, b)

    return out
