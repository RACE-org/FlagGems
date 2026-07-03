import itertools
import logging
from typing import List, Tuple, Union

from flag_gems.utils.codegen_config_utils import CodeGenConfig
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic
from flag_gems.utils.tensor_wrapper import StridedBuffer

import torch
import triton

my_config = CodeGenConfig(
    max_tile_size=1024 * 512,
    max_grid_size=(16, 16, 16),
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=False,
)

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")], config=my_config)
@triton.jit
def copy_func(x):
    return x


def _is_1d_empty_tensor(tensor: torch.Tensor) -> bool:
    return tensor.ndim == 1 and tensor.numel() == 0


def _find_reference_tensor(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]
) -> torch.Tensor:
    for tensor in tensors:
        if not _is_1d_empty_tensor(tensor):
            return tensor
    return tensors[0]


def cat(
    A: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logger.debug("GEMS CAT")
    if len(A) == 0:
        raise RuntimeError("torch.cat(): expected a non-empty list of Tensors")
    if len(A) == 1:
        return A[0]

    ref_tensor = _find_reference_tensor(A)

    assert dim >= -ref_tensor.ndim and dim < ref_tensor.ndim, f"Invalid dim: {dim}"
    dim = dim % ref_tensor.ndim

    # PyTorch allows 1-D empty tensors with shape (0,) in torch.cat inputs.
    # They should not decide the reference rank. This matters for KV-cache init:
    # torch.cat([empty_cache, key_states], dim=-2), where empty_cache is (0,).
    if _is_1d_empty_tensor(ref_tensor):
        return torch.empty((0,), dtype=ref_tensor.dtype, device=ref_tensor.device)

    ref_shape = list(ref_tensor.shape)
    valid_tensors = []

    for tensor_idx, tensor in enumerate(A):
        if _is_1d_empty_tensor(tensor):
            continue

        inp_shape = list(tensor.shape)
        if len(inp_shape) != len(ref_shape):
            raise RuntimeError(
                f"Tensors must have same number of dimensions: got {len(ref_shape)} and {len(inp_shape)}"
            )

        for idx, (common_length, length) in enumerate(zip(ref_shape, inp_shape)):
            if idx == dim:
                continue
            if length != common_length:
                raise RuntimeError(
                    f"Sizes of tensors must match except in dimension {dim}. "
                    f"Expected size {common_length} but got size {length} for tensor number "
                    f"{tensor_idx} in the list"
                )

        valid_tensors.append(tensor)

    if len(valid_tensors) == 0:
        return torch.empty((0,), dtype=ref_tensor.dtype, device=ref_tensor.device)

    out_shape = list(ref_shape)
    out_shape[dim] = sum(tensor.shape[dim] for tensor in valid_tensors)
    out0 = torch.empty(out_shape, dtype=ref_tensor.dtype, device=ref_tensor.device)

    # No data-copy kernel is needed for zero-numel tensors. Skipping them also
    # avoids creating invalid StridedBuffer views for empty placeholders.
    copy_tensors = [tensor for tensor in valid_tensors if tensor.numel() > 0]
    if len(copy_tensors) == 0:
        return out0

    out0_strides = out0.stride()
    out0_offsets = list(
        itertools.accumulate(
            [tensor.shape[dim] * out0_strides[dim] for tensor in copy_tensors[:-1]],
            initial=0,
        )
    )

    for tensor, out0_offset in zip(copy_tensors, out0_offsets):
        in_view = StridedBuffer(tensor, tensor.shape, tensor.stride())
        out_view = StridedBuffer(
            out0,
            tensor.shape,
            out0.stride(),
            offset=out0_offset,
        )
        copy_func.instantiate(tensor.ndim)(in_view, out0=out_view)

    return out0