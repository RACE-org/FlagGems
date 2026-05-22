import os
import importlib.util
from collections import namedtuple
from pathlib import Path
from typing import Any

from backend_utils import VendorInfoBase  # noqa: E402
from flag_gems.runtime.commom_utils import Autograd

from .utils.config_pre_hook import setup_triton_config

# from .heuristics_config_utils import HEURISTICS_CONFIGS

setup_triton_config()

import triton  # noqa: E402
from triton.backends.spine_triton.driver import CPUDriver  # noqa: E402

triton.runtime.driver.set_active(CPUDriver())  # noqa: E402

vendor_info = VendorInfoBase(
    vendor_name="spacemit",
    device_name="cpu",
    device_query_cmd="spacemit-tcm-smi",
)


def _load_op(op_name):
    module_path = Path(__file__).with_name("ops") / f"{op_name}.py"
    spec = importlib.util.spec_from_file_location(
        f"flag_gems.runtime.backend._spacemit._custom_{op_name}", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, op_name)


def get_register_op_config():
    return (
        ("addmm", _load_op("addmm"), Autograd.disable),
        ("bmm", _load_op("bmm"), Autograd.disable),
        ("mm", _load_op("mm"), Autograd.disable),
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
        DeviceProperties = namedtuple("DeviceProperties", ["multi_processor_count"])
        return DeviceProperties(multi_processor_count=os.cpu_count() or 1)


CUSTOMIZED_UNUSED_OPS = ()


__all__ = ["*"]
