import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.shape_utils import broadcast_shapes

logger = logging.getLogger(__name__)


# ===========================================================================
# CPU pre-generated randn + DMA upload — replaces NPU-side Philox + Box-Muller.
#
# Philox integer ops (mului_extended, xori, addi) + Box-Muller (log, sqrt,
# sin, cos) all fall back to RISC-V scalar on Tx81.  Pre-generating the
# standard-normal samples on CPU eliminates the entire RNG chain from the
# kernel, replacing it with a single DMA load.
#
# The DMA bandwidth cost (~4 bytes per element) is negligible compared to
# the ~31k scalar integer iterations per CTA that Philox would require.
# ===========================================================================


@pointwise_dynamic(
    is_tensor=[True, True, True], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def transform_func_tensor_tensor(val, std, mean):
    return val * std + mean


@pointwise_dynamic(
    is_tensor=[True, False, True], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def transform_func_tensor_float(val, std, mean):
    return val * std + mean


@pointwise_dynamic(
    is_tensor=[True, True, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def transform_func_float_tensor(val, std, mean):
    return val * std + mean


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def transform_func_float_float(val, std, mean):
    return val * std + mean


def _randn_cpu(shape, device, *, generator=None):
    """CPU pre-generated standard-normal samples, DMA-uploaded to device.

    .to(device) is blocking for non-CUDA devices — DMA completes before
    the call returns, so the tensor is immediately usable by device kernels.
    """
    return torch.randn(shape, dtype=torch.float32, generator=generator).to(device)


def normal_tensor_tensor(mean, std, *, generator=None):
    logger.debug("GEMS TSINGMICRO NORMAL_TENSOR_TENSOR")
    shape = broadcast_shapes([mean.shape, std.shape])
    device = mean.device
    out = _randn_cpu(shape, device, generator=generator)
    return transform_func_tensor_tensor(out, std, mean)


def normal_tensor_float(mean, std, *, generator=None):
    logger.debug("GEMS TSINGMICRO NORMAL_TENSOR_FLOAT")
    shape = mean.shape
    device = mean.device
    out = _randn_cpu(shape, device, generator=generator)
    return transform_func_tensor_float(out, std, mean)


def normal_float_tensor(mean, std, *, generator=None):
    logger.debug("GEMS TSINGMICRO NORMAL_FLOAT_TENSOR")
    shape = std.shape
    device = std.device
    out = _randn_cpu(shape, device, generator=generator)
    return transform_func_float_tensor(out, std, mean)
