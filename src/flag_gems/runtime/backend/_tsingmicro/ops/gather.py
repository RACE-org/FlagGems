import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.gather import gather as _generic_gather
from flag_gems.ops.scatter import scatter_
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.shape_utils import restride_dim

logger = logging.getLogger(__name__)

_MAX_RANK = 5


@libentry()
@triton.jit
def _gather_lastdim_kernel(
    inp_ptr: tl.tensor,
    index_ptr: tl.tensor,
    out_ptr: tl.tensor,
    M: int,
    K: int,
    N: int,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """out[row, col] = inp[row, index[row, col]].

    This path is only valid when the non-last dimensions of input and index
    match exactly.  Then both tensors can be viewed as 2D without changing row
    identity, and no per-element multi-dimensional decomposition is needed.
    """
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (cols[None, :] < N)

    idx = tl.load(
        index_ptr + rows[:, None] * N + cols[None, :],
        mask=mask,
        other=0,
    ).to(tl.int32)
    vals = tl.load(inp_ptr + rows[:, None] * K + idx, mask=mask, other=0)
    tl.store(out_ptr + rows[:, None] * N + cols[None, :], vals, mask=mask)


@libentry()
@triton.jit
def _gather_anydim_kernel(
    inp_ptr: tl.tensor,
    index_ptr: tl.tensor,
    out_ptr: tl.tensor,
    N_total: int,
    dim_stride: int,
    is0: int,
    is1: int,
    is2: int,
    is3: int,
    is4: int,
    ins0: int,
    ins1: int,
    ins2: int,
    ins3: int,
    ins4: int,
    ips0: int,
    ips1: int,
    ips2: int,
    ips3: int,
    ips4: int,
    ops0: int,
    ops1: int,
    ops2: int,
    ops3: int,
    ops4: int,
    BLOCK_N: tl.constexpr,
):
    """Generic gather with int32 arithmetic and no vector i64 hot path.

    This follows the upstream codegen algorithm, but keeps all flat-index
    decomposition in int32.  Tx81 lowers vector i64 arithmetic poorly.
    """
    pid = tle.program_id(0)
    ctas = tle.num_programs(0)

    for j in tl.range(0, tl.cdiv(tl.cdiv(N_total, BLOCK_N), ctas)):
        block_id = pid + j * ctas
        off = block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = off < N_total

        cur = off
        i4 = cur % is4
        cur = cur // is4
        i3 = cur % is3
        cur = cur // is3
        i2 = cur % is2
        cur = cur // is2
        i1 = cur % is1
        i0 = cur // is1

        idx_off = i0 * ins0 + i1 * ins1 + i2 * ins2 + i3 * ins3 + i4 * ins4
        idx_val = tl.load(index_ptr + idx_off, mask=mask, other=0).to(tl.int32)

        inp_off = (
            i0 * ips0
            + i1 * ips1
            + i2 * ips2
            + i3 * ips3
            + i4 * ips4
            + idx_val * dim_stride
        )
        vals = tl.load(inp_ptr + inp_off, mask=mask, other=0)

        out_off = i0 * ops0 + i1 * ops1 + i2 * ops2 + i3 * ops3 + i4 * ops4
        tl.store(out_ptr + out_off, vals, mask=mask)


def _pad_shapes_strides(shape, stride, max_rank):
    rank = len(shape)
    pad = max_rank - rank
    shapes = (1,) * pad + tuple(shape)
    strides = (0,) * pad + tuple(stride)
    return tuple(int(s) for s in shapes), tuple(int(s) for s in strides)


def _can_use_fast_path(inp, dim, index, out, sparse_grad):
    if sparse_grad or inp.is_complex() or inp.dtype == torch.float64:
        return False
    if not isinstance(dim, int) or dim < -inp.ndim or dim >= inp.ndim:
        return False
    if inp.ndim == 0 or inp.ndim > _MAX_RANK or index.ndim != inp.ndim:
        return False
    if index.numel() == 0:
        return False
    if out is not None and (not out.is_contiguous() or out.dtype != inp.dtype):
        return False
    if index.dtype not in (torch.int32, torch.int64):
        return False
    if inp.numel() >= (1 << 31) or index.numel() >= (1 << 31):
        return False
    return True


def _can_use_lastdim_fast_path(inp, dim, index):
    dim = dim % inp.ndim
    return dim == inp.ndim - 1 and tuple(index.shape[:-1]) == tuple(inp.shape[:-1])


def _gather_lastdim(inp_c, index_c, out):
    M = out.numel() // out.shape[-1]
    K = inp_c.shape[-1]
    N = out.shape[-1]

    inp_2d = inp_c.reshape(M, K)
    index_2d = index_c.reshape(M, N)
    out_2d = out.reshape(M, N)

    block_m = min(triton.next_power_of_2(M), 256)
    block_n = min(triton.next_power_of_2(N), 512)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    _gather_lastdim_kernel[grid](
        inp_2d,
        index_2d,
        out_2d,
        M,
        K,
        N,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
    )


def _gather_anydim(inp_strided, dim, index_c, out, dim_stride):
    n_total = index_c.numel()

    idx_shape, idx_stride = _pad_shapes_strides(index_c.shape, index_c.stride(), _MAX_RANK)
    inp_shape, inp_stride = _pad_shapes_strides(
        inp_strided.shape, inp_strided.stride(), _MAX_RANK
    )
    out_shape, out_stride = _pad_shapes_strides(out.shape, out.stride(), _MAX_RANK)

    block_n = min(512, triton.next_power_of_2(n_total))
    grid = (min(16, triton.cdiv(n_total, block_n)),)

    _gather_anydim_kernel[grid](
        inp_strided,
        index_c,
        out,
        n_total,
        int(dim_stride),
        idx_shape[0],
        idx_shape[1],
        idx_shape[2],
        idx_shape[3],
        idx_shape[4],
        idx_stride[0],
        idx_stride[1],
        idx_stride[2],
        idx_stride[3],
        idx_stride[4],
        inp_stride[0],
        inp_stride[1],
        inp_stride[2],
        inp_stride[3],
        inp_stride[4],
        out_stride[0],
        out_stride[1],
        out_stride[2],
        out_stride[3],
        out_stride[4],
        BLOCK_N=block_n,
        num_warps=4,
    )


def gather(inp, dim, index, out=None, sparse_grad=False):
    logger.debug("GEMS_TSINGMICRO GATHER")

    if not _can_use_fast_path(inp, dim, index, out, sparse_grad):
        return _generic_gather(inp, dim, index, out, sparse_grad)

    dim = dim % inp.ndim
    inp_c = inp.contiguous()
    index_c = (
        index.to(torch.int32).contiguous()
        if index.dtype == torch.int64
        else index.contiguous()
    )

    if out is None:
        out = torch.empty_like(index, dtype=inp.dtype, device=inp.device)
    else:
        out = out.contiguous()

    if _can_use_lastdim_fast_path(inp_c, dim, index_c):
        _gather_lastdim(inp_c, index_c, out)
        return out

    dim_stride = inp_c.stride(dim)
    inp_strided = restride_dim(inp_c, dim, index_c.shape)
    _gather_anydim(inp_strided, dim, index_c, out, dim_stride)
    return out


def gather_backward(grad, self, dim, index, sparse_grad):
    logger.debug("GEMS_TSINGMICRO GATHER BACKWARD")
    result = grad.new_zeros(self.shape)
    return scatter_(result, dim, index, grad, reduce="add")
