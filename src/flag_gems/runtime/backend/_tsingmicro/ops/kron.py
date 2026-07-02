import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.kron import (
    calculate_indices,
    kron as _generic_kron,
    prepare_tensor_for_kron,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_TILE_GRID = 16
_BLOCK_BM = 16
_BLOCK_BN = 16
_INT32_INDEX_LIMIT = 1 << 31


@libentry()
@triton.jit(
    do_not_specialize=[
        "TOTAL_JOBS",
        "JOBS_PER_BATCH",
        "B_BLOCKS_N",
        "B_BLOCKS_PER_A",
        "A_ELEMS",
        "A_ROWS",
        "A_COLS",
        "B_ROWS",
        "B_COLS",
        "OUT_COLS",
        "C_BATCH_STRIDE",
    ]
)
def _kron_a_scalar_b_tile_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    map_ptr,
    TOTAL_JOBS: int,
    JOBS_PER_BATCH: int,
    B_BLOCKS_N: int,
    B_BLOCKS_PER_A: int,
    A_ELEMS: int,
    A_ROWS: int,
    A_COLS: int,
    B_ROWS: int,
    B_COLS: int,
    OUT_COLS: int,
    C_BATCH_STRIDE: int,
    HAS_BATCH_MAP: tl.constexpr,
    BLOCK_BM: tl.constexpr,
    BLOCK_BN: tl.constexpr,
):
    pid = tle.program_id(0)
    b_row_lane = tl.arange(0, BLOCK_BM)
    b_col_lane = tl.arange(0, BLOCK_BN)

    for job in tl.range(pid, TOTAL_JOBS, tl.num_programs(0)):
        if tl.constexpr(HAS_BATCH_MAP):
            batch_id = job // JOBS_PER_BATCH
            local_job = job - batch_id * JOBS_PER_BATCH
            a_batch_idx = tl.load(map_ptr + batch_id * 2)
            b_batch_idx = tl.load(map_ptr + batch_id * 2 + 1)
        else:
            batch_id = 0
            local_job = job
            a_batch_idx = 0
            b_batch_idx = 0

        a_elem = local_job // B_BLOCKS_PER_A
        b_tile = local_job - a_elem * B_BLOCKS_PER_A
        b_block_row = b_tile // B_BLOCKS_N
        b_block_col = b_tile - b_block_row * B_BLOCKS_N

        # These divisions are scalar per CTA job, not vector lane work. The hot
        # lane path below only uses linear B tile offsets and contiguous stores.
        a_row = a_elem // A_COLS
        a_col = a_elem - a_row * A_COLS

        b_rows = b_block_row * BLOCK_BM + b_row_lane
        b_cols = b_block_col * BLOCK_BN + b_col_lane
        b_mask = (b_rows[:, None] < B_ROWS) & (b_cols[None, :] < B_COLS)

        a_val = tl.load(a_ptr + a_batch_idx * A_ELEMS + a_elem)
        b_vals = tl.load(
            b_ptr
            + b_batch_idx * (B_ROWS * B_COLS)
            + b_rows[:, None] * B_COLS
            + b_cols[None, :],
            mask=b_mask,
            other=0,
        )

        out_rows = a_row * B_ROWS + b_rows
        out_cols = a_col * B_COLS + b_cols
        out_offsets = (
            batch_id * C_BATCH_STRIDE
            + out_rows[:, None] * OUT_COLS
            + out_cols[None, :]
        )
        tl.store(c_ptr + out_offsets, a_val * b_vals, mask=b_mask)


def _numel(shape):
    n_elements = 1
    for size in shape:
        n_elements *= int(size)
    return n_elements


def _build_batch_map(a_shape, b_shape, batch_size, device):
    if batch_size == 1:
        return None

    values = []
    for batch_idx in range(batch_size):
        a_idx, b_idx = calculate_indices(batch_idx, a_shape, b_shape)
        values.extend((a_idx, b_idx))
    return torch.tensor(values, dtype=torch.int32, device=device)


def _kron_fast_path(A, B):
    A_prepared, B_prepared, out_shape = prepare_tensor_for_kron(A, B)
    M1, N1 = A_prepared.shape[-2:]
    M2, N2 = B_prepared.shape[-2:]
    M, N = M1 * M2, N1 * N2
    batch_size = math.prod(out_shape[:-2]) if out_shape[:-2] else 1

    output_numel = _numel(out_shape)
    if output_numel >= _INT32_INDEX_LIMIT:
        return None

    output_dtype = torch.promote_types(A.dtype, B.dtype)
    C = torch.empty(out_shape, device=A.device, dtype=output_dtype)

    A_view = A_prepared.reshape(-1, M1, N1).contiguous()
    B_view = B_prepared.reshape(-1, M2, N2).contiguous()
    C_view = C.view(-1, M, N)

    a_elems = M1 * N1
    b_blocks_n = triton.cdiv(N2, _BLOCK_BN)
    b_blocks_per_a = triton.cdiv(M2, _BLOCK_BM) * b_blocks_n
    jobs_per_batch = a_elems * b_blocks_per_a
    total_jobs = batch_size * jobs_per_batch
    if total_jobs == 0:
        return C.reshape(-1) if A.dim() <= 1 and B.dim() <= 1 else C

    batch_map = _build_batch_map(A_prepared.shape, B_prepared.shape, batch_size, A.device)
    has_batch_map = batch_map is not None
    map_arg = batch_map if has_batch_map else A_view

    with torch_device_fn.device(A.device):
        _kron_a_scalar_b_tile_kernel[(min(_TILE_GRID, total_jobs),)](
            A_view,
            B_view,
            C_view,
            map_arg,
            total_jobs,
            jobs_per_batch,
            b_blocks_n,
            b_blocks_per_a,
            a_elems,
            M1,
            N1,
            M2,
            N2,
            N,
            M * N,
            HAS_BATCH_MAP=has_batch_map,
            BLOCK_BM=_BLOCK_BM,
            BLOCK_BN=_BLOCK_BN,
            num_warps=8,
        )

    if A.dim() <= 1 and B.dim() <= 1:
        return C.reshape(-1)
    return C


def kron(A, B):
    logger.debug("GEMS TSINGMICRO KRON")

    if A.is_complex() or B.is_complex():
        return _generic_kron(A, B)

    if A.dim() == 0 or B.dim() == 0 or A.numel() == 0 or B.numel() == 0:
        return _generic_kron(A, B)

    result = _kron_fast_path(A, B)
    if result is None:
        return _generic_kron(A, B)
    return result
