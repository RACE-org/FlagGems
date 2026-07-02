import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import CodeGenConfig

my_config = CodeGenConfig(
    max_tile_size=65536,
    max_grid_size=(16, 16, 16),
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=False,
)

logger = logging.getLogger(__name__)


# Use the `(x != 0) & (y != 0)` form rather than `x.to(tl.int1).logical_and(...)`:
# the explicit `!= 0` lowers directly to tx.boolunequalvv (fp-SIMD on Tx81),
# and the `&` to tx.boolandvv — matches the int1-vector fast path without
# relying on the dtype cast taking the same route.
@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")], config=my_config)
@triton.jit
def logical_and_func(x, y):
    return (x != 0) & (y != 0)


def logical_and(A, B):
    logger.debug("GEMS LOGICAL_AND")
    return logical_and_func(A, B)


# Inplace writes 0/1 ints back into A's dtype, so wrap the bool mask in
# tl.where to produce the right output type.
@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")], config=my_config)
@triton.jit
def logical_and_func_(x, y):
    return tl.where((x != 0) & (y != 0), 1, 0)


def logical_and_(A, B):
    logger.debug("GEMS LOGICAL_AND_")
    logical_and_func_(A, B, out0=A)
    return A
