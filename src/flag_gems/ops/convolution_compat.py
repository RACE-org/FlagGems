from flag_gems.ops.conv2d import conv2d


def convolution(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    transposed=False,
    output_padding=0,
    groups=1,
):
    if transposed:
        raise NotImplementedError("convolution wrapper does not support transposed=True")
    if output_padding not in (0, (0, 0), [0, 0]):
        raise NotImplementedError("convolution wrapper does not support output_padding")
    if input.ndim != 4:
        raise NotImplementedError(
            f"convolution wrapper expects 4D input, got ndim={input.ndim}"
        )
    return conv2d(input, weight, bias, stride, padding, dilation, groups)


def _convolution(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    transposed=False,
    output_padding=0,
    groups=1,
    benchmark=False,
    deterministic=False,
    cudnn_enabled=True,
    allow_tf32=True,
):
    del benchmark, deterministic, cudnn_enabled, allow_tf32
    return convolution(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        transposed=transposed,
        output_padding=output_padding,
        groups=groups,
    )


def cudnn_convolution(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    return conv2d(input, weight, bias, stride, padding, dilation, groups)