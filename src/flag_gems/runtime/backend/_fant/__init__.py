from backend_utils import Autograd, VendorInfoBase  # noqa: E402

vendor_info = VendorInfoBase(
    vendor_name="fant", device_name="fant", device_query_cmd="clinfo"
)

CUSTOMIZED_UNUSED_OPS = ()


def get_register_op_config():
    from .ops import (  # noqa: F811
        all,
        all_dim,
        all_dims,
        native_dropout,
        exponential_,
        rand_like,
        rand,
        randn_like,
        randn,
        softmax,
        uniform_,
        normal_float_tensor,
        normal_tensor_float,
        normal_tensor_tensor,
        fill_scalar,
        fill_tensor,
        arange,
        arange_start,
    )
    return (
        ("native_dropout", native_dropout, Autograd.enable),
        ("exponential_", exponential_, Autograd.disable),
        ("rand", rand, Autograd.disable),
        ("rand_like", rand_like, Autograd.disable),
        ("randn", randn, Autograd.disable),
        ("randn_like", randn_like, Autograd.disable),
        ("uniform_", uniform_, Autograd.disable),
        ("softmax.int", softmax, Autograd.enable),
        ("all", all, Autograd.disable),
        ("all.dim", all_dim, Autograd.disable),
        ("all.dims", all_dims, Autograd.disable),
        ("normal.Tensor_float", normal_tensor_float, Autograd.disable),
        ("normal.float_Tensor", normal_float_tensor, Autograd.disable),
        ("normal.Tensor_Tensor", normal_tensor_tensor, Autograd.disable),
        ("fill.Scalar", fill_scalar, Autograd.disable),
        ("fill.Tensor", fill_tensor, Autograd.disable),
        ("arange.start_step", arange_start, Autograd.disable),
        ("arange.start", arange_start, Autograd.disable),
        ("arange", arange, Autograd.disable),
    )

def get_unused_op():
    return CUSTOMIZED_UNUSED_OPS


__all__ = ["*"]
