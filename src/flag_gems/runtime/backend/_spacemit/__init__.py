import os
import importlib
import importlib.util
from collections import namedtuple
from typing import Any

from backend_utils import VendorInfoBase  # noqa: E402
from flag_gems.runtime.commom_utils import Autograd

if importlib.util.find_spec("triton.backends.spine_triton") is not None:
    from .utils.config_pre_hook import setup_triton_config

    setup_triton_config()

    import triton  # noqa: E402
    from triton.backends.spine_triton.driver import CPUDriver  # noqa: E402

    triton.runtime.driver.set_active(CPUDriver())  # noqa: E402


vendor_info = VendorInfoBase(
    vendor_name="spacemit",
    device_name="cpu",
    device_query_cmd="spacemit-tcm-smi",
)


def _load_op(module_name, op_name=None):
    module = importlib.import_module(f"{__package__}.ops.{module_name}")
    op_name = module_name if op_name is None else op_name
    return getattr(module, op_name)


def get_register_op_config():
    DISABLE_SPACEMIT_OPS = os.environ.get("DISABLE_SPACEMIT_OPS", "")
    if DISABLE_SPACEMIT_OPS:
        return ()
    return (
        ("abs", _load_op("abs"), Autograd.disable),
        ("abs_", _load_op("abs", "abs_"), Autograd.disable),
        ("add.Tensor", _load_op("add"), Autograd.disable),
        ("add_.Tensor", _load_op("add", "add_"), Autograd.disable),
        ("addmm", _load_op("addmm"), Autograd.disable),
        ("all", _load_op("all"), Autograd.disable),
        ("all.dim", _load_op("all", "all_dim"), Autograd.disable),
        ("all.dims", _load_op("all", "all_dims"), Autograd.disable),
        ("allclose", _load_op("isclose", "allclose"), Autograd.disable),
        ("amax", _load_op("amax"), Autograd.disable),
        ("any", _load_op("any"), Autograd.disable),
        ("any.dim", _load_op("any", "any_dim"), Autograd.disable),
        ("any.dims", _load_op("any", "any_dims"), Autograd.disable),
        ("argmax", _load_op("argmax"), Autograd.disable),
        ("argmin", _load_op("argmin"), Autograd.disable),
        ("avg_pool2d", _load_op("mean", "avg_pool2d"), Autograd.disable),
        ("native_batch_norm", _load_op("batch_norm", "batch_norm"), Autograd.disable),
        ("bitwise_and.Tensor", _load_op("bitwise_and", "bitwise_and_tensor"), Autograd.disable),
        ("bitwise_and_.Tensor", _load_op("bitwise_and", "bitwise_and_tensor_"), Autograd.disable),
        ("bitwise_and.Scalar", _load_op("bitwise_and", "bitwise_and_scalar"), Autograd.disable),
        ("bitwise_and_.Scalar", _load_op("bitwise_and", "bitwise_and_scalar_"), Autograd.disable),
        ("bitwise_and.Scalar_Tensor", _load_op("bitwise_and", "bitwise_and_scalar_tensor"), Autograd.disable),
        ("bitwise_not", _load_op("bitwise_not"), Autograd.disable),
        ("bitwise_not_", _load_op("bitwise_not", "bitwise_not_"), Autograd.disable),
        ("bitwise_or.Tensor", _load_op("bitwise_or", "bitwise_or_tensor"), Autograd.disable),
        ("bitwise_or_.Tensor", _load_op("bitwise_or", "bitwise_or_tensor_"), Autograd.disable),
        ("bitwise_or.Scalar", _load_op("bitwise_or", "bitwise_or_scalar"), Autograd.disable),
        ("bitwise_or_.Scalar", _load_op("bitwise_or", "bitwise_or_scalar_"), Autograd.disable),
        ("bitwise_or.Scalar_Tensor", _load_op("bitwise_or", "bitwise_or_scalar_tensor"), Autograd.disable),
        ("bmm", _load_op("bmm"), Autograd.disable),
        ("cat", _load_op("cat"), Autograd.disable),
        ("clamp", _load_op("clamp"), Autograd.disable),
        ("clamp_", _load_op("clamp", "clamp_"), Autograd.disable),
        ("clamp.Tensor", _load_op("clamp", "clamp_tensor"), Autograd.disable),
        ("clamp.Tensor_out", _load_op("clamp", "clamp_tensor"), Autograd.disable),
        ("clamp_.Tensor", _load_op("clamp", "clamp_tensor_"), Autograd.disable),
        ("clamp_min", _load_op("clamp", "clamp_min"), Autograd.disable),
        ("clamp_min_", _load_op("clamp", "clamp_min_"), Autograd.disable),
        ("conv1d", _load_op("conv1d"), Autograd.disable),
        ("conv2d", _load_op("conv2d"), Autograd.disable),
        ("_conv_depthwise2d", _load_op("conv_depthwise2d", "_conv_depthwise2d"), Autograd.disable),
        ("_convolution", _load_op("convolution_compat", "_convolution"), Autograd.disable),
        ("convolution", _load_op("convolution_compat", "convolution"), Autograd.disable),
        ("cudnn_convolution", _load_op("convolution_compat", "cudnn_convolution"), Autograd.disable),
        ("cos", _load_op("cos"), Autograd.disable),
        ("cos_", _load_op("cos", "cos_"), Autograd.disable),
        ("cumsum", _load_op("cumsum"), Autograd.disable),
        ("cumsum.out", _load_op("cumsum", "cumsum_out"), Autograd.disable),
        ("div.Tensor", _load_op("div", "true_divide"), Autograd.disable),
        ("div_.Tensor", _load_op("div", "true_divide_"), Autograd.disable),
        ("div.Scalar", _load_op("div", "true_divide"), Autograd.disable),
        ("div_.Scalar", _load_op("div", "true_divide_"), Autograd.disable),
        ("divide.Tensor", _load_op("div", "true_divide"), Autograd.disable),
        ("divide_.Tensor", _load_op("div", "true_divide_"), Autograd.disable),
        ("divide.Scalar", _load_op("div", "true_divide"), Autograd.disable),
        ("divide_.Scalar", _load_op("div", "true_divide_"), Autograd.disable),
        ("true_divide.Tensor", _load_op("div", "true_divide"), Autograd.disable),
        ("true_divide_.Tensor", _load_op("div", "true_divide_"), Autograd.disable),
        ("true_divide.Scalar", _load_op("div", "true_divide"), Autograd.disable),
        ("true_divide_.Scalar", _load_op("div", "true_divide_"), Autograd.disable),
        ("floor_divide", _load_op("div", "floor_divide"), Autograd.disable),
        ("floor_divide.Scalar", _load_op("div", "floor_divide"), Autograd.disable),
        ("div.Tensor_mode", _load_op("div", "div_mode"), Autograd.disable),
        ("div.Scalar_mode", _load_op("div", "div_mode"), Autograd.disable),
        ("eq.Tensor", _load_op("eq"), Autograd.disable),
        ("eq.Scalar", _load_op("eq", "eq_scalar"), Autograd.disable),
        ("erf", _load_op("erf"), Autograd.disable),
        ("erf_", _load_op("erf", "erf_"), Autograd.disable),
        ("exp", _load_op("exp"), Autograd.disable),
        ("exp_", _load_op("exp", "exp_"), Autograd.disable),
        ("fill.Scalar", _load_op("fill", "fill_scalar"), Autograd.disable),
        ("fill.Tensor", _load_op("fill", "fill_tensor"), Autograd.disable),
        ("fill_.Scalar", _load_op("fill", "fill_scalar_"), Autograd.disable),
        ("fill_.Tensor", _load_op("fill", "fill_tensor_"), Autograd.disable),
        ("ge.Tensor", _load_op("ge"), Autograd.disable),
        ("ge.Scalar", _load_op("ge", "ge_scalar"), Autograd.disable),
        ("gelu", _load_op("gelu"), Autograd.disable),
        ("native_group_norm", _load_op("groupnorm", "group_norm"), Autograd.disable),
        ("gt.Tensor", _load_op("gt"), Autograd.disable),
        ("gt.Scalar", _load_op("gt", "gt_scalar"), Autograd.disable),
        ("index_select", _load_op("index_select"), Autograd.disable),
        ("isclose", _load_op("isclose"), Autograd.disable),
        ("isinf", _load_op("isinf"), Autograd.disable),
        ("isnan", _load_op("isnan"), Autograd.disable),
        ("native_layer_norm", _load_op("layernorm", "layer_norm"), Autograd.disable),
        ("le.Tensor", _load_op("le"), Autograd.disable),
        ("le.Scalar", _load_op("le", "le_scalar"), Autograd.disable),
        ("log_sigmoid_forward", _load_op("log_sigmoid", "log_sigmoid"), Autograd.disable),
        ("log_softmax", _load_op("log_softmax", "log_softmax"), Autograd.disable),
        ("_log_softmax", _load_op("log_softmax", "log_softmax"), Autograd.disable),
        ("log_softmax.int", _load_op("log_softmax", "log_softmax"), Autograd.disable),
        ("lt.Tensor", _load_op("lt"), Autograd.disable),
        ("lt.Scalar", _load_op("lt", "lt_scalar"), Autograd.disable),
        ("masked_fill.Tensor", _load_op("masked_fill", "masked_fill"), Autograd.disable),
        ("masked_fill.Scalar", _load_op("masked_fill", "masked_fill"), Autograd.disable),
        ("masked_fill_.Tensor", _load_op("masked_fill", "masked_fill_"), Autograd.disable),
        ("masked_fill_.Scalar", _load_op("masked_fill", "masked_fill_"), Autograd.disable),
        ("max", _load_op("max"), Autograd.disable),
        ("max.dim", _load_op("max", "max_dim"), Autograd.disable),
        ("maximum", _load_op("maximum"), Autograd.disable),
        ("max_pool2d_with_indices", _load_op("maxpool", "maxpool2d"), Autograd.disable),
        ("mean", _load_op("mean"), Autograd.disable),
        ("mean.dim", _load_op("mean", "mean_dim"), Autograd.disable),
        ("min", _load_op("min"), Autograd.disable),
        ("min.dim", _load_op("min", "min_dim"), Autograd.disable),
        ("minimum", _load_op("minimum"), Autograd.disable),
        ("mm", _load_op("mm"), Autograd.disable),
        ("mul.Tensor", _load_op("mul"), Autograd.disable),
        ("mul_.Tensor", _load_op("mul", "mul_"), Autograd.disable),
        ("multinomial", _load_op("multinomial"), Autograd.disable),
        ("mv", _load_op("mv"), Autograd.disable),
        ("ne.Tensor", _load_op("ne"), Autograd.disable),
        ("ne.Scalar", _load_op("ne", "ne_scalar"), Autograd.disable),
        ("nll_loss_forward", _load_op("nllloss", "nll_loss_forward"), Autograd.disable),
        ("nll_loss2d_forward", _load_op("nllloss", "nll_loss2d_forward"), Autograd.disable),
        ("normal.Tensor_float", _load_op("normal", "normal_tensor_float"), Autograd.disable),
        ("normal.float_Tensor", _load_op("normal", "normal_float_tensor"), Autograd.disable),
        ("normal.Tensor_Tensor", _load_op("normal", "normal_tensor_tensor"), Autograd.disable),
        ("normal_", _load_op("normal", "normal_"), Autograd.disable),
        ("nonzero", _load_op("nonzero"), Autograd.disable),
        ("ones", _load_op("ones"), Autograd.disable),
        ("outer", _load_op("outer"), Autograd.enable),
        # ("permute", _load_op("permute"), Autograd.disable),
        ("zeros", _load_op("zeros"), Autograd.disable),
        ("pow.Scalar", _load_op("pow", "pow_scalar"), Autograd.disable),
        ("pow.Tensor_Scalar", _load_op("pow", "pow_tensor_scalar"), Autograd.disable),
        ("pow_.Tensor_Scalar", _load_op("pow", "pow_tensor_scalar_"), Autograd.disable),
        ("pow.Tensor_Tensor", _load_op("pow", "pow_tensor_tensor"), Autograd.disable),
        ("pow_.Tensor_Tensor", _load_op("pow", "pow_tensor_tensor_"), Autograd.disable),
        ("relu", _load_op("relu"), Autograd.disable),
        ("relu_", _load_op("relu", "relu_"), Autograd.disable),
        ("reciprocal", _load_op("reciprocal"), Autograd.disable),
        ("rsqrt", _load_op("rsqrt"), Autograd.disable),
        ("rsub.Scalar", _load_op("rsub"), Autograd.disable),
        ("sigmoid", _load_op("sigmoid"), Autograd.disable),
        ("silu", _load_op("silu"), Autograd.disable),
        ("silu_", _load_op("silu", "silu_"), Autograd.disable),
        ("sin", _load_op("sin"), Autograd.disable),
        ("sin_", _load_op("sin", "sin_"), Autograd.disable),
        ("softmax", _load_op("softmax"), Autograd.disable),
        ("_softmax", _load_op("softmax"), Autograd.disable),
        ("softmax.int", _load_op("softmax"), Autograd.disable),
        ("sort", _load_op("sort"), Autograd.disable),
        ("sub.Tensor", _load_op("sub"), Autograd.disable),
        ("sub_.Tensor", _load_op("sub", "sub_"), Autograd.disable),
        ("sum", _load_op("sum"), Autograd.disable),
        ("sum.dim_IntList", _load_op("sum", "sum_dim"), Autograd.disable),
        ("sum.IntList_out", _load_op("sum", "sum_dim_out"), Autograd.disable),
        ("sum.out", _load_op("sum", "sum_out"), Autograd.disable),
        ("tanh", _load_op("tanh"), Autograd.enable),
        ("tanh_", _load_op("tanh", "tanh_"), Autograd.disable),
        ("_to_copy", _load_op("to", "to_dtype"), Autograd.disable),
        ("transpose.int", _load_op("transpose"), Autograd.disable),
        ("triu", _load_op("triu"), Autograd.disable),
        ("unfold", _load_op("unfold"), Autograd.disable),
        ("var_mean.correction", _load_op("var_mean"), Autograd.disable),
        ("linalg_vector_norm", _load_op("vector_norm", "vector_norm"), Autograd.disable),
        ("where.self_out", _load_op("where", "where_self_out"), Autograd.disable),
        ("where.self", _load_op("where", "where_self"), Autograd.disable),
        ("where.ScalarSelf", _load_op("where", "where_scalar_self"), Autograd.disable),
        ("where.ScalarOther", _load_op("where", "where_scalar_other"), Autograd.disable),
        ("_scaled_dot_product_flash_attention", _load_op("flash_attention", "scaled_dot_product_attention"), Autograd.enable),
        ("_scaled_dot_product_efficient_attention", _load_op("attention_compat", "_scaled_dot_product_efficient_attention"), Autograd.enable),
    )



def get_unused_op():
    return ()


class _DeviceGuard:
    def __init__(self, index: int):
        self.idx = index
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = self.idx

    def __exit__(self, type: Any, value: Any, traceback: Any):
        self.idx = self.prev_idx
        return False


class _DeviceWrapper:
    def __init__(self, device: Any):
        ...

    def __enter__(self):
        ...

    def __exit__(self, type: Any, value: Any, traceback: Any):
        ...
        return False

    @staticmethod
    def current_device():
        """Return device index for kernel cache. CPU backend always uses device 0."""
        return 0

    @staticmethod
    def get_device_properties(device: Any = None):
        DeviceProperties = namedtuple(
            "DeviceProperties", ["multi_processor_count", "total_memory"]
        )
        return DeviceProperties(
            multi_processor_count=os.cpu_count() or 1,
            total_memory=64 * 1024**3,
        )


def empty_cache():
    pass

CUSTOMIZED_UNUSED_OPS = ()


__all__ = ["*"]