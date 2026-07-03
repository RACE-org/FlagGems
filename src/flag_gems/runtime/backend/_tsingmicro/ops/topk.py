import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.codegen_config_utils import CodeGenConfig

logger = logging.getLogger(__name__)

_MIN_FLOAT32_VAL = tl.constexpr(torch.finfo(torch.float32).min)
_MAX_FLOAT32_VAL = tl.constexpr(torch.finfo(torch.float32).max)
_MIN_FLOAT16_VAL = tl.constexpr(torch.finfo(torch.float16).min)
_MAX_FLOAT16_VAL = tl.constexpr(torch.finfo(torch.float16).max)
_MIN_BFLOAT16_VAL = tl.constexpr(torch.finfo(torch.bfloat16).min)
_MAX_BFLOAT16_VAL = tl.constexpr(torch.finfo(torch.bfloat16).max)
_MIN_INT32_VAL = tl.constexpr(torch.iinfo(torch.int32).min)
_MAX_INT32_VAL = tl.constexpr(torch.iinfo(torch.int32).max)

TOTAL_TILES = 16

op_config = CodeGenConfig(
    max_tile_size=65536,
    max_grid_size=(16, 16, 16),
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=False,
)


@triton.jit
def _get_finfo_val(dtype, return_max):
    if dtype is tl.float32:
        if return_max:
            return _MAX_FLOAT32_VAL
        else:
            return _MIN_FLOAT32_VAL
    elif dtype is tl.float16:
        if return_max:
            return _MAX_FLOAT16_VAL
        else:
            return _MIN_FLOAT16_VAL
    elif dtype is tl.bfloat16:
        if return_max:
            return _MAX_BFLOAT16_VAL
        else:
            return _MIN_BFLOAT16_VAL


# ===========================================================================
# Path 1: K=1 dedicated kernel — one argmax per row, minimal overhead
# ===========================================================================


@libentry()
@triton.jit
def topk_k1_kernel(
    y_ptr,
    index_ptr,
    x_ptr,
    batch_size,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DESCENDING: tl.constexpr,
):
    """grid=(16,), row-partition.  One hardware argmax per row — no loop body."""
    TOTAL_TILES: tl.constexpr = 16
    pid = tle.program_id(0)
    rows_per_tile = tl.cdiv(batch_size, TOTAL_TILES)
    start = pid * rows_per_tile
    end = tl.minimum(start + rows_per_tile, batch_size)

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    for row in range(start, end):
        sentinel = _get_finfo_val(x_ptr.dtype.element_ty, return_max=not DESCENDING)
        x_val = tl.load(x_ptr + row * N + cols, mask=mask, other=sentinel).to(tl.float32)

        if DESCENDING:
            cur_val = tl.max(x_val)
            cur_idx = tl.argmax(x_val, axis=0)
        else:
            cur_val = tl.min(x_val)
            cur_idx = tl.argmin(x_val, axis=0)

        tl.store(y_ptr + row, cur_val)
        tl.store(index_ptr + row, cur_idx)


# ===========================================================================
# Path 2: Single-stage repeated max (N fits in SPM, K > 1)
#
#   Per row: DMA full row → SPM, then K iterations of (argmax → store → mask).
#   No intermediate DDR round-trip, no bitonic sort.  O(N*K) per row with
#   hardware-accelerated argmax/max/reduce on every pass.
# ===========================================================================


@libentry()
@triton.jit
def topk_singlestage_kernel(
    y_ptr,
    index_ptr,
    x_ptr,
    batch_size,
    k,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DESCENDING: tl.constexpr,
):
    TOTAL_TILES: tl.constexpr = 16
    pid = tle.program_id(0)
    rows_per_tile = tl.cdiv(batch_size, TOTAL_TILES)
    start = pid * rows_per_tile
    end = tl.minimum(start + rows_per_tile, batch_size)

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    for row in range(start, end):
        sentinel = _get_finfo_val(x_ptr.dtype.element_ty, return_max=not DESCENDING)
        x_val = tl.load(x_ptr + row * N + cols, mask=mask, other=sentinel).to(tl.float32)

        for k_idx in range(k):
            if DESCENDING:
                cur_val = tl.max(x_val)
                cur_idx = tl.argmax(x_val, axis=0)
                x_val = tl.where(
                    cols == cur_idx,
                    _get_finfo_val(tl.float32, return_max=False),
                    x_val,
                )
            else:
                cur_val = tl.min(x_val)
                cur_idx = tl.argmin(x_val, axis=0)
                x_val = tl.where(
                    cols == cur_idx,
                    _get_finfo_val(tl.float32, return_max=True),
                    x_val,
                )

            tl.store(y_ptr + row * k + k_idx, cur_val)
            tl.store(index_ptr + row * k + k_idx, cur_idx)


# ===========================================================================
# Path 3: Two-stage fallback (N too large for SPM, > 2 MB)
#
#   Same two-stage structure as upstream, but stage 2 uses repeated max
#   instead of bitonic sort on the candidate pool (num_chunks * K elements).
#   Stage 2 grid = (16,) with row-partition, matching paths 1 & 2.
# ===========================================================================


@libentry()
@triton.jit
def topk_stage1_kernel(
    y_ptr,
    index_ptr,
    x_ptr,
    k,
    N: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
):
    """grid=(batch_size, chunk_num).  Each CTA: load one chunk, K × argmax → local top-K."""
    cur_batch = tle.program_id(0)
    cur_chunk_idx = tle.program_id(1)
    chunk_num = tle.num_programs(1)

    y_ptr += cur_batch * chunk_num * k + cur_chunk_idx * k
    index_ptr += cur_batch * chunk_num * k + cur_chunk_idx * k

    chunk_offset = cur_chunk_idx * CHUNK_SIZE
    x_ptr += cur_batch * N + chunk_offset

    cols = tl.arange(0, CHUNK_SIZE)
    mask = (chunk_offset + cols) < N

    mask_val = _get_finfo_val(x_ptr.dtype.element_ty, return_max=not DESCENDING)
    x_val = tl.load(x_ptr + cols, mask=mask, other=mask_val).to(tl.float32)
    for k_idx in range(k):
        if DESCENDING:
            chunk_select_val = tl.max(x_val)
            chunk_select_idx = tl.argmax(x_val, axis=0)
        else:
            chunk_select_val = tl.min(x_val)
            chunk_select_idx = tl.argmin(x_val, axis=0)

        tl.store(y_ptr + k_idx, chunk_select_val)
        tl.store(index_ptr + k_idx, chunk_select_idx + chunk_offset)

        if DESCENDING:
            x_val = tl.where(
                cols == chunk_select_idx,
                _get_finfo_val(tl.float32, return_max=False),
                x_val,
            )
        else:
            x_val = tl.where(
                cols == chunk_select_idx,
                _get_finfo_val(tl.float32, return_max=True),
                x_val,
            )


@libentry()
@triton.jit
def topk_stage2_kernel(
    y_ptr,
    index_ptr,
    chunk_x,
    chunk_index,
    batch_size,
    k,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
):
    """grid=(16,), row-partition.  Repeated max on num_chunks * K candidates → global top K."""
    TOTAL_TILES: tl.constexpr = 16
    pid = tle.program_id(0)
    rows_per_tile = tl.cdiv(batch_size, TOTAL_TILES)
    start = pid * rows_per_tile
    end = tl.minimum(start + rows_per_tile, batch_size)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    for row in range(start, end):
        sentinel = _get_finfo_val(chunk_x.dtype.element_ty, return_max=not DESCENDING)

        x_val = tl.load(chunk_x + row * N + cols, mask=mask, other=sentinel).to(tl.float32)

        for k_idx in range(k):
            if DESCENDING:
                cur_val = tl.max(x_val)
                cur_idx = tl.argmax(x_val, axis=0)
                # Direct scalar load — avoids tl.sum on int32 (Tx81 has no int
                # vector reduce; hardware argmax already gave us the position).
                actual_idx = tl.load(chunk_index + row * N + cur_idx)
                x_val = tl.where(
                    cols == cur_idx,
                    _get_finfo_val(tl.float32, return_max=False),
                    x_val,
                )
            else:
                cur_val = tl.min(x_val)
                cur_idx = tl.argmin(x_val, axis=0)
                actual_idx = tl.load(chunk_index + row * N + cur_idx)
                x_val = tl.where(
                    cols == cur_idx,
                    _get_finfo_val(tl.float32, return_max=True),
                    x_val,
                )

            tl.store(y_ptr + row * k + k_idx, cur_val)
            tl.store(index_ptr + row * k + k_idx, actual_idx)


# ===========================================================================
# Host helpers
# ===========================================================================


def _topk_two_stage(x, k, dim, largest, sorted):
    """Fallback when a full row doesn't fit in SPM (> 2 MB)."""
    descending = True if largest else False
    topk_elem_cnt = x.shape[dim]
    batch_size = math.prod(x.shape) // topk_elem_cnt

    if topk_elem_cnt < 1024:
        chunk_size = 256
    else:
        chunk_size = 1024
    if chunk_size < k:
        chunk_size = triton.next_power_of_2(k)

    chunk_num = triton.cdiv(topk_elem_cnt, chunk_size)

    stage1_out = torch.empty(
        batch_size * chunk_num * k, device=x.device, dtype=x.dtype
    )
    stage1_out_idx = torch.empty(
        batch_size * chunk_num * k, device=x.device, dtype=torch.int32
    )

    out_shape = x.shape[:-1] + (k,)
    stage2_out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    stage2_out_idx = torch.empty(out_shape, device=x.device, dtype=torch.int32)

    with torch_device_fn.device(x.device):
        topk_stage1_kernel[batch_size, chunk_num](
            stage1_out, stage1_out_idx, x, k,
            topk_elem_cnt, chunk_size, descending,
        )

    stage2_elem_cnt = chunk_num * k
    BLOCK_SIZE = triton.next_power_of_2(stage2_elem_cnt)

    with torch_device_fn.device(x.device):
        topk_stage2_kernel[(TOTAL_TILES,)](
            stage2_out, stage2_out_idx,
            stage1_out, stage1_out_idx, batch_size, k,
            stage2_elem_cnt, BLOCK_SIZE, descending,
        )

    return (stage2_out, stage2_out_idx.to(torch.int64))


def topk(x, k, dim=-1, largest=True, sorted=True):
    logger.debug("GEMS TSINGMICRO TOPK")
    if dim < 0:
        dim = dim + x.ndim
    assert dim == x.ndim - 1, "Currently only support topk in last dimension"

    descending = True if largest else False
    topk_elem_cnt = x.shape[dim]
    k = min(k, topk_elem_cnt)
    batch_size = math.prod(x.shape) // topk_elem_cnt

    # SPM capacity guard: if a full row > 2 MB, fall back to two-stage
    spm_limit = 2 * 1024 * 1024
    if topk_elem_cnt * x.element_size() > spm_limit:
        return _topk_two_stage(x, k, dim, largest, sorted)

    BLOCK_N = triton.next_power_of_2(topk_elem_cnt)
    out_shape = x.shape[:-1] + (k,)
    grid = (TOTAL_TILES,)

    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    out_idx = torch.empty(out_shape, device=x.device, dtype=torch.int32)

    with torch_device_fn.device(x.device):
        if k == 1:
            topk_k1_kernel[grid](out, out_idx, x, batch_size,
                                  topk_elem_cnt, BLOCK_N, descending)
        else:
            topk_singlestage_kernel[grid](out, out_idx, x, batch_size, k,
                                           topk_elem_cnt, BLOCK_N, descending)

    return (out, out_idx.to(torch.int64))
