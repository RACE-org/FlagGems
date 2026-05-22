from . import backend, commom_utils
from .backend.device import DeviceDetector
from .configloader import ConfigLoader

config_loader = ConfigLoader()
device = DeviceDetector()

"""
The dependency order of the sub-directory is strict, and changing the order arbitrarily may cause errors.
"""

# torch_device_fn is like 'torch.cuda' object
backend.set_torch_backend_device_fn(device.vendor_name)
torch_device_fn = backend.gen_torch_device_object()

# torch_backend_device is like 'torch.backend.cuda' object
torch_backend_device = backend.get_torch_backend_device_fn()


def get_triton_config(op_name):
    return config_loader.get_triton_config(op_name)


def get_tuned_config(op_name):
    return get_triton_config(op_name)


def get_heuristic_config(op_name):
    if device.vendor_name == "spacemit":
        from .backend._spacemit.heuristics_config_utils import HEURISTICS_CONFIGS

        return HEURISTICS_CONFIGS[op_name]
    return {}


def replace_customized_ops(_globals):
    if device.vendor_name == "nvidia":
        return
    for item in backend.get_curent_device_extend_op(device.vendor_name):
        fn_name, fn = item[:2]
        _globals[fn_name] = fn


__all__ = ["commom_utils", "backend", "device", "get_triton_config"]
