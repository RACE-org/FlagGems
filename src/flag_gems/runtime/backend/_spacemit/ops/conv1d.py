import logging

from .conv2d import conv2d

logger = logging.getLogger(__name__)


def conv1d(input, weight, bias=None, padding=0, stride=1, dilation=1, groups=1):
    logger.debug("GEMS_SPACEMIT CONV1D")

    if isinstance(stride, (list, tuple)):
        stride_width = stride[0]
    else:
        stride_width = stride

    if isinstance(padding, (list, tuple)):
        padding_width = padding[0]
    else:
        padding_width = padding

    if isinstance(dilation, (list, tuple)):
        dilation_width = dilation[0]
    else:
        dilation_width = dilation

    return conv2d(
        input.unsqueeze(-1),
        weight.unsqueeze(-1),
        bias,
        (stride_width, 1),
        (padding_width, 0),
        (dilation_width, 1),
        groups,
    ).squeeze(-1)
