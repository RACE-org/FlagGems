import os

if os.environ.get("FLAG_GEMS_CUSTOM_OPS", "1") != "0":
    # Override the official pointwise_dynamic with the tsingmicro
    # broadcast-aware codegen BEFORE any op module is loaded.  Op files use
    # `from flag_gems.utils import pointwise_dynamic` (or
    # `from flag_gems.utils.pointwise_dynamic import pointwise_dynamic`),
    # so patch both the package attribute and the symbol inside the base
    # module to cover every import style.
    import flag_gems.utils as _gems_utils
    import flag_gems.utils.pointwise_dynamic as _base_pd
    from ..utils.pointwise_dynamic import pointwise_dynamic as _txda_pd

    _gems_utils.pointwise_dynamic = _txda_pd
    _base_pd.pointwise_dynamic = _txda_pd

    from . import add, gelu

    from .add import add, add_
    from .cat import cat
    from .abs import abs, abs_
    from .cos import cos, cos_
    from .tanh import tanh, tanh_, tanh_backward
    from .to import to_dtype
    from .topk import topk
    from .eq import eq, eq_scalar
    from .exp import exp, exp_
    from .eye_m import eye_m
    from .elu import elu
    from .mul import mul, mul_
    from .sub import sub, sub_
    from .neg import neg, neg_
    from .ne import ne, ne_scalar
    from .ge import ge, ge_scalar
    from .gt import gt, gt_scalar
    from .le import le, le_scalar
    from .lt import lt, lt_scalar
    from .log import log
    from .logical_and import logical_and, logical_and_
    from .logical_not import logical_not
    from .log_sigmoid import log_sigmoid
    from .isinf import isinf
    from .isnan import isnan
    from .isclose import allclose, isclose
    from .ones import ones
    from .ones_like import ones_like
    from .fill import fill_scalar, fill_scalar_, fill_tensor, fill_tensor_
    from .flip import flip
    from .full import full
    from .full_like import full_like
    from .reciprocal import reciprocal, reciprocal_
    from .relu import relu, relu_
    from .rsqrt import rsqrt, rsqrt_
    from .sin import sin, sin_
    from .silu import silu, silu_, silu_backward
    from .silu_and_mul import silu_and_mul
    from .sigmoid import sigmoid, sigmoid_, sigmoid_backward
    from .zeros import zeros
    from .zeros_like import zeros_like
    from .maximum import maximum
    from .minimum import minimum
    from .bitwise_and import (
        bitwise_and_scalar,
        bitwise_and_scalar_,
        bitwise_and_scalar_tensor,
        bitwise_and_tensor,
        bitwise_and_tensor_,
    )
    from .bitwise_not import bitwise_not, bitwise_not_
    from .bitwise_or import (
        bitwise_or_scalar,
        bitwise_or_scalar_,
        bitwise_or_scalar_tensor,
        bitwise_or_tensor,
        bitwise_or_tensor_,
    )
    from .all import all_dim, all_dims, all

    from .outer import outer

    from .pow import pow_tensor_tensor
    from .gelu_and_mul import gelu_and_mul
    from .gelu import gelu, gelu_, gelu_backward
    from .erf import erf
    from .glu import glu
    from .isfinite import isfinite
    from .angle import angle
    from .logical_or import logical_or
    from .nan_to_num import nan_to_num
    from .log_softmax import log_softmax
    from .rms_norm import rms_norm
    from .logical_xor import logical_xor
    from .div import true_divide, floor_divide, remainder
    from .hstack import hstack
    from .stack import stack
    from .layernorm import layer_norm
    from .batch_norm import batch_norm
    from .sum import sum
    from .max import max
    from .mean import mean
    from .min import min
    # from .argmin import argmin

    from .dropout import dropout, dropout_backward
    from .masked_fill import masked_fill, masked_fill_
    from .masked_select import masked_select
    from .multinomial import multinomial
    from .normal import (
        normal_float_tensor,
        normal_tensor_float,
        normal_tensor_tensor,
    )
    from .rand import rand
    from .rand_like import rand_like
    from .randn import randn
    from .scatter import scatter, scatter_
    from .nonzero import nonzero
    from .vector_norm import vector_norm
    from .clamp import  clamp, clamp_, clamp_tensor, clamp_tensor_
    from .any import any, any_dim, any_dims
    from .argmax import argmax
    from .where import (
        where_self,
        where_self_out,
        where_scalar_other,
        where_scalar_self,
    )
    # from .quantile import quantile

    from .arange import arange, arange_start
    # from .index_put import index_put, index_put_
    from .index_select import index_select
    from .isin import isin
    from .cumsum import cumsum, cumsum_out, normed_cumsum
    from .cummax import cummax
    from .mse_loss import mse_loss
    from .prod import prod, prod_dim
    from .embedding import embedding, embedding_backward
    from .bmm import bmm
    from .contiguous import contiguous
    from .exponential_ import exponential_
    # from .repeat_interleave import (
    #     repeat_interleave_self_int,
    #     repeat_interleave_self_tensor,
    #     repeat_interleave_tensor,
    # )
    from .gather import gather, gather_backward
    from .pad import pad, constant_pad_nd
    from .tile import tile
    # from .sort import sort, sort_stable
    from .count_nonzero import count_nonzero
    from .cummin import cummin
    from .diag import diag
    from .diag_embed import diag_embed
    from .diagonal import diagonal_backward
    from .kron import kron
    from .lerp import lerp_scalar, lerp_scalar_, lerp_tensor, lerp_tensor_
    from .linspace import linspace
    from .var_mean import var_mean
    from .weightnorm import weight_norm_interface, weight_norm_interface_backward

def get_specific_ops():
    return {}


def get_unused_ops():
    return ()


__all__ = [
    "get_specific_ops",
    "get_unused_ops",
]
