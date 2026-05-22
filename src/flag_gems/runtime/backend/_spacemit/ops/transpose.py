import logging

import torch

from .permute import permute

logger = logging.getLogger(__name__)


def transpose(input, dim0, dim1):
    """Transpose via permute kernel."""
    logger.debug("GEMS_SPACEMIT TRANSPOSE")
    ndim = input.dim()
    dim0 = dim0 % ndim
    dim1 = dim1 % ndim
    dims = list(range(ndim))
    dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
    return permute(input, dims)
