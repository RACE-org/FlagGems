# import logging

# import torch
# import triton
# import triton.language as tl

# from flag_gems.ops.linspace import linspace as _generic_linspace
# from flag_gems.runtime import torch_device_fn
# from flag_gems.utils import libentry

# logger = logging.getLogger(__name__)

# _TILE_GRID = 16
# _BLOCK_SIZE = 256
# _FP32_EXACT_INDEX_LIMIT = 1 << 24


# @libentry()
# @triton.jit(do_not_specialize=["STEPS", "START", "END", "STEP_SIZE"])
# def _linspace_f32_kernel(
#     out_ptr,
#     STEPS: int,
#     START: float,
#     END: float,
#     STEP_SIZE: float,
#     MID: tl.constexpr,
#     BLOCK_SIZE: tl.constexpr,
# ):
#     pid = tl.program_id(0)
#     lane = tl.arange(0, BLOCK_SIZE)

#     for base in tl.range(pid * BLOCK_SIZE, STEPS, tl.num_programs(0) * BLOCK_SIZE):
#         idx = base + lane
#         mask = idx < STEPS
#         idx_f32 = idx.to(tl.float32)
#         rev_idx_f32 = (STEPS - idx - 1).to(tl.float32)

#         # Use the same forward/backward split as the generic implementation so
#         # both endpoints are generated from their exact scalar values.
#         fw_values = START + STEP_SIZE * idx_f32
#         bd_values = END - STEP_SIZE * rev_idx_f32
#         out_val = tl.where(idx < MID, fw_values, bd_values)
#         tl.store(out_ptr + idx, out_val, mask=mask)


# def _requested_dtype(dtype):
#     return dtype if dtype is not None else torch.get_default_dtype()


# def _as_scalar(value):
#     if isinstance(value, torch.Tensor):
#         return value.item()
#     return value


# def _can_use_fast_path(dtype, layout, pin_memory, steps):
#     requested_dtype = _requested_dtype(dtype)
#     return (
#         layout in (None, torch.strided)
#         and pin_memory in (None, False)
#         and requested_dtype in (torch.float16, torch.bfloat16, torch.float32)
#         and steps <= _FP32_EXACT_INDEX_LIMIT
#     )


# def linspace(
#     start, end, steps, *, dtype=None, layout=None, device=None, pin_memory=None
# ) -> torch.Tensor:
#     logger.debug("GEMS TSINGMICRO LINSPACE")
#     assert steps >= 1, "steps must be >= 1"

#     if not _can_use_fast_path(dtype, layout, pin_memory, steps):
#         return _generic_linspace(
#             start,
#             end,
#             steps,
#             dtype=dtype,
#             layout=layout,
#             device=device,
#             pin_memory=pin_memory,
#         )

#     out = torch.empty(
#         steps,
#         dtype=dtype,
#         layout=layout,
#         device=device,
#         pin_memory=pin_memory,
#     )

#     if steps == 1:
#         return torch.fill(out, start)

#     start = _as_scalar(start)
#     end = _as_scalar(end)
#     step_size = (float(end) - float(start)) / (steps - 1)
#     mid = steps // 2

#     grid = (min(_TILE_GRID, triton.cdiv(steps, _BLOCK_SIZE)),)
#     with torch_device_fn.device(out.device):
#         _linspace_f32_kernel[grid](
#             out,
#             steps,
#             float(start),
#             float(end),
#             float(step_size),
#             MID=mid,
#             BLOCK_SIZE=_BLOCK_SIZE,
#             num_warps=8,
#         )
#     return out


# import logging

# import torch
# import triton
# import triton.language as tl

# from flag_gems.ops.linspace import linspace as _generic_linspace
# from flag_gems.runtime import torch_device_fn
# from flag_gems.utils import libentry

# logger = logging.getLogger(__name__)

# _TILE_GRID = 16
# _MIN_BLOCK_SIZE = 16
# _MAX_BLOCK_SIZE = 4096
# _FP32_EXACT_INDEX_LIMIT = 1 << 24


# @libentry()
# @triton.jit(do_not_specialize=["STEPS", "START", "END", "STEP_SIZE"])
# def _linspace_f32_kernel(
#     out_ptr,
#     STEPS: int,
#     START: float,
#     END: float,
#     STEP_SIZE: float,
#     MID: tl.constexpr,
#     BLOCK_SIZE: tl.constexpr,
# ):
#     pid = tl.program_id(0)
#     lane = tl.arange(0, BLOCK_SIZE)

#     for base in tl.range(pid * BLOCK_SIZE, STEPS, tl.num_programs(0) * BLOCK_SIZE):
#         idx = base + lane
#         mask = idx < STEPS
#         idx_f32 = idx.to(tl.float32)
#         rev_idx_f32 = (STEPS - idx - 1).to(tl.float32)

#         # Use the same forward/backward split as the generic implementation so
#         # both endpoints are generated from their exact scalar values.
#         fw_values = START + STEP_SIZE * idx_f32
#         bd_values = END - STEP_SIZE * rev_idx_f32
#         out_val = tl.where(idx < MID, fw_values, bd_values)
#         tl.store(out_ptr + idx, out_val, mask=mask)


# def _requested_dtype(dtype):
#     return dtype if dtype is not None else torch.get_default_dtype()


# def _as_scalar(value):
#     if isinstance(value, torch.Tensor):
#         return value.item()
#     return value


# def _can_use_fast_path(dtype, layout, pin_memory, steps):
#     requested_dtype = _requested_dtype(dtype)
#     return (
#         layout in (None, torch.strided)
#         and pin_memory in (None, False)
#         and requested_dtype in (torch.float16, torch.bfloat16, torch.float32)
#         and steps <= _FP32_EXACT_INDEX_LIMIT
#     )


# def _pick_block_size(steps):
#     # Pick a block so normal-size linspace launches about one CTA per Tx81
#     # tile.  A fixed large block makes steps=256/512 run on only one CTA.
#     per_tile = triton.cdiv(steps, _TILE_GRID)
#     block = triton.next_power_of_2(max(per_tile, 1))
#     return min(_MAX_BLOCK_SIZE, max(_MIN_BLOCK_SIZE, block))


# def _num_warps(block_size):
#     if block_size >= 2048:
#         return 8
#     if block_size >= 512:
#         return 4
#     if block_size >= 128:
#         return 2
#     return 1


# def linspace(
#     start, end, steps, *, dtype=None, layout=None, device=None, pin_memory=None
# ) -> torch.Tensor:
#     logger.debug("GEMS TSINGMICRO LINSPACE")
#     assert steps >= 1, "steps must be >= 1"

#     if not _can_use_fast_path(dtype, layout, pin_memory, steps):
#         return _generic_linspace(
#             start,
#             end,
#             steps,
#             dtype=dtype,
#             layout=layout,
#             device=device,
#             pin_memory=pin_memory,
#         )

#     out = torch.empty(
#         steps,
#         dtype=dtype,
#         layout=layout,
#         device=device,
#         pin_memory=pin_memory,
#     )

#     if steps == 1:
#         return torch.fill(out, start)

#     start = _as_scalar(start)
#     end = _as_scalar(end)
#     step_size = (float(end) - float(start)) / (steps - 1)
#     mid = steps // 2

#     block_size = _pick_block_size(steps)
#     grid = (min(_TILE_GRID, triton.cdiv(steps, block_size)),)
#     with torch_device_fn.device(out.device):
#         _linspace_f32_kernel[grid](
#             out,
#             steps,
#             float(start),
#             float(end),
#             float(step_size),
#             MID=mid,
#             BLOCK_SIZE=128,
#             num_warps=_num_warps(block_size),
#         )
#     return out

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.linspace import linspace as _generic_linspace
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_TILE_GRID = 16
_MIN_BLOCK_SIZE = 16
_MAX_BLOCK_SIZE = 4096
_SMALL_STEPS_FALLBACK_THRESHOLD = 512
_FP32_EXACT_INDEX_LIMIT = 1 << 24


@libentry()
@triton.jit(do_not_specialize=["STEPS", "START", "END", "STEP_SIZE"])
def _linspace_f32_kernel(
    out_ptr,
    STEPS: int,
    START: float,
    END: float,
    STEP_SIZE: float,
    MID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK_SIZE)

    for base in tl.range(pid * BLOCK_SIZE, STEPS, tl.num_programs(0) * BLOCK_SIZE):
        idx = base + lane
        mask = idx < STEPS
        idx_f32 = idx.to(tl.float32)
        rev_idx_f32 = (STEPS - idx - 1).to(tl.float32)

        # Use the same forward/backward split as the generic implementation so
        # both endpoints are generated from their exact scalar values.
        fw_values = START + STEP_SIZE * idx_f32
        bd_values = END - STEP_SIZE * rev_idx_f32
        out_val = tl.where(idx < MID, fw_values, bd_values)
        tl.store(out_ptr + idx, out_val, mask=mask)


def _requested_dtype(dtype):
    return dtype if dtype is not None else torch.get_default_dtype()


def _as_scalar(value):
    if isinstance(value, torch.Tensor):
        return value.item()
    return value


def _can_use_fast_path(dtype, layout, pin_memory, steps):
    requested_dtype = _requested_dtype(dtype)
    return (
        layout in (None, torch.strided)
        and pin_memory in (None, False)
        and requested_dtype in (torch.float16, torch.bfloat16, torch.float32)
        and steps > _SMALL_STEPS_FALLBACK_THRESHOLD
        and steps <= _FP32_EXACT_INDEX_LIMIT
    )


def _pick_block_size(steps):
    # Pick a block so normal-size linspace launches about one CTA per Tx81
    # tile.  Very small steps fall back to the generic 128-block kernel because
    # launch/scheduling cost dominates and it is faster for steps=256.
    per_tile = triton.cdiv(steps, _TILE_GRID)
    block = triton.next_power_of_2(max(per_tile, 1))
    return min(_MAX_BLOCK_SIZE, max(_MIN_BLOCK_SIZE, block))


def _num_warps(block_size):
    if block_size >= 2048:
        return 8
    if block_size >= 512:
        return 4
    if block_size >= 128:
        return 2
    return 1


def linspace(
    start, end, steps, *, dtype=None, layout=None, device=None, pin_memory=None
) -> torch.Tensor:
    logger.debug("GEMS TSINGMICRO LINSPACE")
    assert steps >= 1, "steps must be >= 1"

    if not _can_use_fast_path(dtype, layout, pin_memory, steps):
        return _generic_linspace(
            start,
            end,
            steps,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
        )

    out = torch.empty(
        steps,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )

    if steps == 1:
        return torch.fill(out, start)

    start = _as_scalar(start)
    end = _as_scalar(end)
    step_size = (float(end) - float(start)) / (steps - 1)
    mid = steps // 2

    block_size = _pick_block_size(steps)
    grid = (min(_TILE_GRID, triton.cdiv(steps, block_size)),)
    with torch_device_fn.device(out.device):
        _linspace_f32_kernel[grid](
            out,
            steps,
            float(start),
            float(end),
            float(step_size),
            MID=mid,
            BLOCK_SIZE=block_size,
            num_warps=_num_warps(block_size),
        )
    return out
