from backend_utils import VendorInfoBase  # noqa: E402

from .heuristics_config_utils import HEURISTICS_CONFIGS

global specific_ops, unused_ops
specific_ops = None
unused_ops = None
vendor_info = VendorInfoBase(
    vendor_name="tsingmicro", 
    device_name="txda",
    device_query_cmd="tsm_smi",
    dispatch_key="PrivateUse1",
)


CUSTOMIZED_UNUSED_OPS = (
)

def OpLoader():
    import os

    global specific_ops, unused_ops
    if specific_ops is None:
        if os.environ.get("FLAG_GEMS_CUSTOM_OPS", "1") == "0":
            specific_ops = {}
            unused_ops = []
            return
        from . import ops  # noqa: F403

        specific_ops = ops.get_specific_ops()
        unused_ops = ops.get_unused_ops()


__all__ = ["HEURISTICS_CONFIGS", "vendor_info", "OpLoader"]

