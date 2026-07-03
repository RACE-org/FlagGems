import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

TOTAL_CORE_NUM = 16


@libentry()
@triton.jit(do_not_specialize=["M"])
def embedding_kernel(
    out_ptr,  # pointer to the output
    in_ptr,  # pointer to the input (indices)
    weight_ptr,  # pointer to the weights
    M,  # number of rows = indices.numel()
    N: tl.constexpr,  # number of columns in X
    BLOCK_SIZE: tl.constexpr,
):
    # 16-CTA full-tile: each tile owns a contiguous chunk of rows and loops over
    # them, instead of launching one CTA per row (M-CTA explosion on Tx81).
    pid = tle.program_id(0)
    rows_per_tile = tl.cdiv(M, TOTAL_CORE_NUM)
    start = pid * rows_per_tile
    end = min(start + rows_per_tile, M)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    for row in range(start, end):
        row_idx = tl.load(in_ptr + row)
        embedding_weight = tl.load(weight_ptr + row_idx * N + cols, mask, other=0.0)
        tl.store(out_ptr + row * N + cols, embedding_weight, mask)


@libentry()
@triton.jit
def indice_freq_kernel(
    indices_freq,
    indices,  # pointer to the input
    elem_cnt: tl.constexpr,  # number of columns in X
    INDICE_BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    block_start = pid * INDICE_BLOCK_SIZE

    offsets = block_start + tl.arange(0, INDICE_BLOCK_SIZE)
    mask = offsets < elem_cnt

    index_element = tl.load(indices + offsets, mask=mask)
    tl.atomic_add(indices_freq + index_element, 1, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["padding_idx", "M", "W"])
def embedding_backward_kernel(
    grad_in,  # pointer to the gradient input  (num_weights, N)
    grad_out,  # pointer to the gradient output (M, N), contiguous
    indices,  # pointer to the input
    padding_idx,  # padding_idx
    M,  # number of rows = indices.numel()
    W,  # num_weights (number of output rows)
    HAS_PADDING_IDX: tl.constexpr,
    N: tl.constexpr,  # number of columns in X
    BLOCK_SIZE: tl.constexpr,
):
    # Output-partition / owner-computes (atomic-free): each tile owns a disjoint
    # slice of the OUTPUT rows [lo, hi). It scans every input row but only
    # accumulates those whose index lands in its slice. Because each output row
    # has exactly one owner tile and the per-tile loop is sequential, the
    # accumulation is a plain (non-atomic) load-add-store -- no cross-tile
    # contention, so no atomic_add and none of its discrete-RMW overhead.
    pid = tle.program_id(0)
    out_per_tile = tl.cdiv(W, TOTAL_CORE_NUM)
    lo = pid * out_per_tile
    hi = min(lo + out_per_tile, W)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    for m in range(0, M):
        j = tl.load(indices + m).to(tl.int32)
        owned = (j >= lo) and (j < hi)
        if HAS_PADDING_IDX:
            owned = owned and (j != padding_idx)
        if owned:
            embedding_grad = tl.load(grad_out + m * N + cols, mask, other=0.0)
            if tl.constexpr(embedding_grad.dtype.is_bf16()):
                embedding_grad = embedding_grad.to(tl.float32)
            cur = tl.load(grad_in + j * N + cols, mask, other=0.0)
            tl.store(grad_in + j * N + cols, cur + embedding_grad, mask)


@libentry()
@triton.jit(do_not_specialize=["n_rows"])
def embedding_grad_scale_kernel(
    grad_out,
    indice_freq,
    n_rows,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row_start = tle.program_id(0)
    row_step = tle.num_programs(0)

    for row_idx in range(row_start, n_rows, row_step):
        embedding_scale = 1.0
        indice_freq_val = tl.load(indice_freq + row_idx)
        if indice_freq_val > 1:
            embedding_scale = 1.0 / indice_freq_val

        cols = tl.arange(0, BLOCK_SIZE)
        mask = tl.arange(0, BLOCK_SIZE) < N
        embedding_grad = tl.load(grad_out + row_idx * N + cols, mask=mask)
        scaled_embedding_grad = embedding_grad * embedding_scale
        tl.store(grad_out + row_idx * N + cols, scaled_embedding_grad, mask=mask)


def embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    logger.debug("GEMS TSINGMICRO EMBEDDING FORWARD")
    assert not sparse, "Currently do not support sparse format"

    M = indices.numel()
    N = weight.shape[-1]

    BLOCK_SIZE = triton.next_power_of_2(N)
    indices = indices.contiguous()
    weight = weight.contiguous()
    output = torch.empty((*indices.shape, N), device=indices.device, dtype=weight.dtype)

    with torch_device_fn.device(weight.device):
        embedding_kernel[(TOTAL_CORE_NUM,)](output, indices, weight, M, N, BLOCK_SIZE)

    return output


def embedding_backward(
    grad_outputs,
    indices,
    num_weights,
    padding_idx=-1,
    scale_grad_by_freq=False,
    sparse=False,
):
    logger.debug("GEMS TSINGMICRO EMBEDDING BACKWARD")
    assert not sparse, "Currently do not support sparse format"

    M = indices.numel()
    N = grad_outputs.shape[-1]

    # Tx81 kernels address grad_out linearly as (M, N); enforce contiguity so a
    # strided/permuted grad (e.g. stride (1024,4096,1)) is laid out correctly.
    grad_outputs = grad_outputs.contiguous()
    indices = indices.contiguous()

    grad_inputs = torch.zeros(
        (num_weights, N),
        device=grad_outputs.device,
        dtype=(
            torch.float32
            if grad_outputs.dtype is torch.bfloat16
            else grad_outputs.dtype
        ),
    )

    if scale_grad_by_freq:
        indice_freq = torch.zeros(
            (num_weights,),
            requires_grad=False,
            device=grad_outputs.device,
            dtype=torch.int32,
        )
        INDICE_BLOCK_SIZE = 256
        indice_grid = (triton.cdiv(M, INDICE_BLOCK_SIZE),)

        with torch_device_fn.device(grad_outputs.device):
            indice_freq_kernel[indice_grid](indice_freq, indices, M, INDICE_BLOCK_SIZE)
    else:
        indice_freq = None

    BLOCK_SIZE = triton.next_power_of_2(N)
    HAS_PADDING_IDX = padding_idx is not None

    with torch_device_fn.device(grad_outputs.device):
        embedding_backward_kernel[(TOTAL_CORE_NUM,)](
            grad_inputs,
            grad_outputs,
            indices,
            padding_idx,
            M,
            num_weights,
            HAS_PADDING_IDX,
            N,
            BLOCK_SIZE,
        )

    if scale_grad_by_freq:
        with torch_device_fn.device(grad_outputs.device):
            embedding_grad_scale_kernel[(TOTAL_CORE_NUM,)](
                grad_inputs, indice_freq, num_weights, N, BLOCK_SIZE
            )

    return (
        grad_inputs.to(torch.bfloat16)
        if grad_outputs.dtype is torch.bfloat16
        else grad_inputs
    )
