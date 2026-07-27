import importlib
import itertools

import torch

import flag_gems

import random
import numpy as np

from .conftest import QUICK_MODE, TO_CPU


def SkipVersion(module_name, skip_pattern):
    cmp = skip_pattern[0]
    assert cmp in ("=", "<", ">"), f"Invalid comparison operator: {cmp}"
    try:
        M, N = skip_pattern[1:].split(".")
        M, N = int(M), int(N)
    except Exception:
        raise ValueError("Cannot parse version number from skip_pattern.")

    try:
        module = importlib.import_module(module_name)
        version = module.__version__
        major, minor = map(int, version.split(".")[:2])
    except Exception:
        raise ImportError(f"Cannot determine version of module: {module_name}")

    if cmp == "=":
        return major == M and minor == N
    elif cmp == "<":
        return (major, minor) < (M, N)
    else:
        return (major, minor) > (M, N)


INT16_MIN = torch.iinfo(torch.int16).min
INT16_MAX = torch.iinfo(torch.int16).max
INT32_MIN = torch.iinfo(torch.int32).min
INT32_MAX = torch.iinfo(torch.int32).max

sizes_one = [1]
sizes_pow_2 = [16, 64, 256]
sizes_noalign = [d + 17 for d in sizes_pow_2]
sizes_1d = [1, 64, 512]
sizes_2d_nc = [1] if QUICK_MODE else [64, 512]
sizes_2d_nr = [1] if QUICK_MODE else [1, 32]

UT_SHAPES_1D = [(1,), (32,), ]
UT_SHAPES_2D = [(1, 32), (32, 32), (64, 32)]
POINTWISE_SHAPES = (
    [(2, 19, 7)]
    if QUICK_MODE
    else [(1,), (32, 32), (20, 64, 15)]
)
SPECIAL_SHAPES = (
    [(2, 19, 7)]
    if QUICK_MODE
    else [(1,), (32, 32), (20, 64, 15)]
)
DISTRIBUTION_SHAPES = [(20, 32, 15)]
REDUCTION_SHAPES = [(2, 32)] if QUICK_MODE else [(32, 32), (32, 32, 32)]
REDUCTION_SMALL_SHAPES = (
    [(1, 32)] if QUICK_MODE else [(32, 32), (32, 32, 32)]
)
STACK_SHAPES = [
    [(16,), (16,)],
    [(16, 256), (16, 256)],
]
CONTIGUOUS_SHAPE_STRIDES_1D = [
    ((256,), (1,)),
]
DILATED_SHAPE_STRIDES_1D = [
    ((256,), (2,)),
]
CONTIGUOUS_SHAPE_STRIDES_2D = [
    ((1, 512), (512, 1)),
]
TRANSPOSED_SHAPE_STRIDES_2D = [
    ((128, 32), (1, 128)),
]
CONTIGUOUS_SHAPE_STRIDES_3D = [
    ((20, 256, 15), (512, 15, 1)),
]
TRANSPOSED_SHAPE_STRIDES_3D = [
    ((256, 20, 15), (15, 512, 1)),
]
SHAPE_STRIDES = [
    ((256,), (1,)),
    ((1, 512), (512, 1)),
    ((256, 20, 15), (15, 512, 1)),
]

IRREGULAR_SHAPE_STRIDES = [((10, 10, 10, 8, 4))]

UPSAMPLE_SHAPES = [
    (3, 5, 16, 4),
    (3, 7, 16, 4),
]


FLOAT_DTYPES = [torch.float16, torch.float32]
ALL_FLOAT_DTYPES = FLOAT_DTYPES + [torch.float64]
INT_DTYPES = [torch.int16, torch.int32]
ALL_INT_DTYPES = INT_DTYPES + [torch.int64]
BOOL_TYPES = [torch.bool]

SCALARS = [0.001, 0.002, 0.003, 0.009]
STACK_DIM_LIST = [-2, -1, 0, 1]


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    ref_inp = inp
    if TO_CPU:
        ref_inp = ref_inp.to("cpu")
    if upcast:
        ref_inp = ref_inp.to(torch.float64)
    return ref_inp


def to_cpu(res, ref):
    if TO_CPU:
        res = res.to("cpu")
        assert ref.device == torch.device("cpu")
    return res


def gems_assert_close(res, ref, dtype, equal_nan=False, reduce_dim=1):
    res = to_cpu(res, ref)
    flag_gems.testing.assert_close(
        res, ref, dtype, equal_nan=equal_nan, reduce_dim=reduce_dim
    )


def gems_assert_equal(res, ref, equal_nan=False):
    res = to_cpu(res, ref)
    flag_gems.testing.assert_equal(res, ref, equal_nan=equal_nan)


def unsqueeze_tuple(t, max_len):
    for _ in range(len(t), max_len):
        t = t + (1,)
    return t


def unsqueeze_tensor(inp, max_ndim):
    for _ in range(inp.ndim, max_ndim):
        inp = inp.unsqueeze(-1)
    return inp


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
