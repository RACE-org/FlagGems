from backend_utils import Autograd, VendorInfoBase  # noqa: E402

# NOTE: do NOT `from .ops import *` at module top level.
# `flag_gems.runtime.backend.DeviceDetector` imports this package during vendor
# probing (before `flag_gems` itself has finished initializing).  Importing ops
# here would chain into `flag_gems.utils -> flag_gems.runtime`, which is still
# being initialized, and raise a circular-import error.  All op imports must be
# deferred to functions that run AFTER `flag_gems` is fully loaded — i.e.
# `get_register_op_config()` below, which is called from `Register.__init__`.

vendor_info = VendorInfoBase(
    vendor_name="tsingmicro",
    device_name="txda",
    device_query_cmd="tsm_smi",
)


def get_register_op_config():
    # Lazy import: by the time Register asks for this list, flag_gems is fully
    # initialized so importing ops triggers no circular import.  Each tuple is
    # `(aten_op_key, impl_fn, autograd_mode)` and overrides whatever base
    # implementation `flag_gems.enable()` registered for the same key.
    #
    # Use the SAME ``Autograd`` enum that ``runtime/register.py`` compares against
    # (``commom_utils.Autograd``).  The module-level ``from backend_utils import
    # Autograd`` above resolves ``backend_utils`` as a *bare top-level* module,
    # which is a different object than ``flag_gems.runtime.backend.backend_utils``;
    # its ``Autograd.enable`` then fails the ``is commom_utils.Autograd.enable``
    # check in ``register_impl``, so an ``Autograd.enable`` op would be registered
    # at the plain backend key "TXDA" instead of "AutogradTXDA".  For a multi-output
    # op with a native derivative (``_weight_norm_interface`` -> (output, norm)),
    # that leaves PyTorch's native autograd node wrapping ``output`` while gems only
    # owns ``norm``; the native backward then hits a CPU fallback that requires an
    # fp32 ``norm`` and raises "expected scalar type Float but found BFloat16".
    from flag_gems.runtime.commom_utils import Autograd

    from .ops.gelu import gelu
    from .ops.argmax import argmax
    from .ops.vector_norm import vector_norm
    from .ops.cummin import cummin
    from .ops.masked_fill import masked_fill
    from .ops.randperm import randperm
    from .ops.index_add import index_add
    from .ops.var_mean import var_mean
    from .ops.count_nonzero import count_nonzero
    from .ops.weightnorm import weight_norm, weight_norm_interface

    return (
        ("gelu", gelu, Autograd.enable),
        ("argmax", argmax, Autograd.disable),
        ("linalg_vector_norm", vector_norm, Autograd.disable),
        ("cummin", cummin, Autograd.disable),
        ("masked_fill.Tensor", masked_fill, Autograd.disable),
        ("masked_fill.Scalar", masked_fill, Autograd.disable),
        ("randperm", randperm, Autograd.disable),
        ("index_add", index_add, Autograd.disable),
        ("var_mean.correction", var_mean, Autograd.disable),
        ("count_nonzero", count_nonzero, Autograd.disable),
        ("_weight_norm_interface", weight_norm_interface, Autograd.enable),
        ("_weight_norm", weight_norm, Autograd.enable),
    )


def get_unused_op():
    return ()


__all__ = ["*"]
