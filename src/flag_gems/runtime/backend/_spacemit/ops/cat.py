import logging
from typing import List, Tuple, Union

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)

NUM_CTAS = 8


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("cat"),
    key=["n_elements"],
)
@triton.jit
def copy_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple copy kernel using make_block_ptr + boundary_check."""
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)

    total_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(total_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        src_blk = tl.make_block_ptr(
            base=src_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        dst_blk = tl.make_block_ptr(
            base=dst_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )

        x = tl.load(src_blk, boundary_check=(0,))
        tl.store(dst_blk, x, boundary_check=(0,))


def cat(
    A: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logger.debug("GEMS_SPACEMIT CAT")

    if len(A) == 0:
        raise RuntimeError("torch.cat(): expected a non-empty list of Tensors")
    if len(A) == 1:
        return A[0]

    # Find first non-empty tensor
    first_non_empty_idx = 0
    for i, t in enumerate(A):
        if t.numel() > 0:
            first_non_empty_idx = i
            break

    ndim = A[first_non_empty_idx].ndim

    if dim < -ndim or dim >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-ndim}, {ndim-1}], but got {dim})"
        )
    if dim < 0:
        dim = dim + ndim

    # Validate shapes
    inp_shapes = [list(_.shape) for _ in A]
    reference_shape = inp_shapes[first_non_empty_idx]
    reference_ndim = len(reference_shape)

    for tensor_idx, s in enumerate(inp_shapes):
        if A[tensor_idx].numel() == 0:
            continue
        if len(s) != reference_ndim:
            raise RuntimeError(
                f"Tensors must have same number of dimensions: got {reference_ndim} and {len(s)}"
            )

    for tensor_idx, inp_shape in enumerate(inp_shapes):
        if A[tensor_idx].numel() == 0:
            continue
        for idx, (common_length, length) in enumerate(zip(reference_shape, inp_shape)):
            if idx == dim:
                continue
            elif length != common_length:
                raise RuntimeError(
                    f"Sizes of tensors must match except in dimension {dim}. "
                    f"Expected size {common_length} but got size {length} for tensor number "
                    f"{tensor_idx} in the list"
                )

    # Build output shape
    out_shape = list(reference_shape)
    out_shape[dim] = sum(s[dim] if len(s) > dim else 0 for s in inp_shapes)

    reference_tensor = A[first_non_empty_idx]
    out = torch.empty(out_shape, dtype=reference_tensor.dtype, device=reference_tensor.device)

    # Copy each tensor into the output using flat copy with offset
    offset = 0
    for a in A:
        if a.numel() == 0:
            continue
        # Use narrow to get the slice of output, then copy
        n = a.shape[dim]
        out_slice = out.narrow(dim, offset, n)
        # If both are contiguous and same layout, use kernel
        a_contig = a.contiguous()
        out_slice_contig = out_slice.contiguous()
        if out_slice.is_contiguous() and a_contig.numel() == out_slice.numel():
            n_elements = a_contig.numel()
            with torch_device_fn.device(out.device):
                copy_kernel[(NUM_CTAS,)](
                    a_contig, out_slice, n_elements
                )
        else:
            out_slice.copy_(a_contig)
        offset += n

    return out
