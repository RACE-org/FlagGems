import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry
from flag_gems.utils import libtuner

logger = logging.getLogger(__name__)

NUM_CTAS = 8
MAX_DIMS = 8


@libentry()
@libtuner(
	configs=runtime.get_tuned_config("permute"),
	key=["n_elements"],
)
@triton.jit
def permute_kernel(
	inp,
	out,
	n_elements,
	total_tiles,
	BLOCK_SIZE: tl.constexpr,
	NDIM: tl.constexpr,
	IN_SHAPE: tl.constexpr,
	IN_STRIDES: tl.constexpr,
	OUT_SHAPE: tl.constexpr,
	OUT_STRIDES: tl.constexpr,
	DIMS: tl.constexpr,
	OUT_TILE_SHAPE: tl.constexpr,
	IN_TILE_SHAPE: tl.constexpr,
	OUT_TILE_COUNTS: tl.constexpr,
	INV_DIMS: tl.constexpr,
	MEM_ORDER: tl.constexpr,
	BOUNDARY: tl.constexpr,
):
	pid = tl.program_id(0)
	num_ctas = tl.num_programs(0)

	sub_num = tl.cdiv(max(total_tiles - pid, 0), num_ctas)

	for block_idx in tl.range(0, sub_num):
		task_idx = pid + num_ctas * block_idx

		if NDIM == 1:
			out_offsets = (task_idx * OUT_TILE_SHAPE[0],)
			in_offsets = (out_offsets[INV_DIMS[0]],)
		elif NDIM == 2:
			tmp = task_idx
			tile_1 = tmp % OUT_TILE_COUNTS[1]
			tmp = tmp // OUT_TILE_COUNTS[1]
			tile_0 = tmp % OUT_TILE_COUNTS[0]
			out_offsets = (
				tile_0 * OUT_TILE_SHAPE[0],
				tile_1 * OUT_TILE_SHAPE[1],
			)
			in_offsets = (out_offsets[INV_DIMS[0]], out_offsets[INV_DIMS[1]])
		elif NDIM == 3:
			tmp = task_idx
			tile_2 = tmp % OUT_TILE_COUNTS[2]
			tmp = tmp // OUT_TILE_COUNTS[2]
			tile_1 = tmp % OUT_TILE_COUNTS[1]
			tmp = tmp // OUT_TILE_COUNTS[1]
			tile_0 = tmp % OUT_TILE_COUNTS[0]
			out_offsets = (
				tile_0 * OUT_TILE_SHAPE[0],
				tile_1 * OUT_TILE_SHAPE[1],
				tile_2 * OUT_TILE_SHAPE[2],
			)
			in_offsets = (
				out_offsets[INV_DIMS[0]],
				out_offsets[INV_DIMS[1]],
				out_offsets[INV_DIMS[2]],
			)
		elif NDIM == 4:
			tmp = task_idx
			tile_3 = tmp % OUT_TILE_COUNTS[3]
			tmp = tmp // OUT_TILE_COUNTS[3]
			tile_2 = tmp % OUT_TILE_COUNTS[2]
			tmp = tmp // OUT_TILE_COUNTS[2]
			tile_1 = tmp % OUT_TILE_COUNTS[1]
			tmp = tmp // OUT_TILE_COUNTS[1]
			tile_0 = tmp % OUT_TILE_COUNTS[0]
			out_offsets = (
				tile_0 * OUT_TILE_SHAPE[0],
				tile_1 * OUT_TILE_SHAPE[1],
				tile_2 * OUT_TILE_SHAPE[2],
				tile_3 * OUT_TILE_SHAPE[3],
			)
			in_offsets = (
				out_offsets[INV_DIMS[0]],
				out_offsets[INV_DIMS[1]],
				out_offsets[INV_DIMS[2]],
				out_offsets[INV_DIMS[3]],
			)
		else:
			out_offsets = (0,)
			in_offsets = (0,)

		inp_blk = tl.make_block_ptr(
			base=inp,
			shape=IN_SHAPE,
			strides=IN_STRIDES,
			offsets=in_offsets,
			block_shape=IN_TILE_SHAPE,
			order=MEM_ORDER,
		)
		out_blk = tl.make_block_ptr(
			base=out,
			shape=OUT_SHAPE,
			strides=OUT_STRIDES,
			offsets=out_offsets,
			block_shape=OUT_TILE_SHAPE,
			order=MEM_ORDER,
		)

		vals = tl.load(inp_blk, boundary_check=BOUNDARY)
		if NDIM == 1:
			vals = tl.permute(vals, (0,))
		elif NDIM == 2:
			vals = tl.permute(vals, (DIMS[0], DIMS[1]))
		elif NDIM == 3:
			vals = tl.permute(vals, (DIMS[0], DIMS[1], DIMS[2]))
		else:
			vals = tl.permute(vals, (DIMS[0], DIMS[1], DIMS[2], DIMS[3]))
		tl.store(out_blk, vals, boundary_check=BOUNDARY)


def _normalize_dims(dims, ndim: int) -> tuple[int, ...]:
	if isinstance(dims, torch.Tensor):
		dims = dims.tolist()
	dims = tuple(int(dim) for dim in dims)

	if len(dims) != ndim:
		raise ValueError(
			f"permute(): dims length {len(dims)} does not match tensor ndim {ndim}"
		)

	normalized = tuple(dim % ndim for dim in dims)
	if sorted(normalized) != list(range(ndim)):
		raise ValueError(
			"permute(): dims must be a permutation of [0..ndim-1] with no repeats"
		)
	return normalized


def _make_tile_shape(shape: tuple[int, ...], block_size: int) -> tuple[int, ...]:
	tile_shape = [1] * len(shape)
	remaining = block_size

	for axis in range(len(shape) - 1, -1, -1):
		extent = max(1, int(shape[axis]))
		tile = max(1, min(triton.next_power_of_2(extent), remaining))
		tile_shape[axis] = tile
		remaining = max(1, remaining // tile)

	return tuple(tile_shape)


def _parse_dims(args, kwargs):
	dims = kwargs.get("dims", None)
	if dims is not None:
		return dims
	if len(args) == 2 and isinstance(args[1], (list, tuple, torch.Tensor)):
		return args[1]
	return args[1:]


def permute(*args, **kwargs) -> torch.Tensor:
	logger.debug("GEMS_SPACEMIT PERMUTE")

	if len(args) == 0:
		raise TypeError("permute() missing required argument: 'input'")

	inp = args[0]
	if not isinstance(inp, torch.Tensor):
		raise TypeError("permute(): input must be a torch.Tensor")

	ndim = inp.dim()
	dims = _normalize_dims(_parse_dims(args, kwargs), ndim)

	if ndim == 0:
		return inp.clone()
	if ndim > MAX_DIMS:
		raise NotImplementedError(f"permute(): ndim > {MAX_DIMS} is not supported")

	inp = inp.clone().contiguous()
	out_shape = tuple(int(inp.shape[dim]) for dim in dims)
	n_elements = inp.numel()

	if n_elements == 0:
		return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

	with torch.inference_mode(False):
		if inp.is_inference():
			inp = inp.clone()
		out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
	in_shape = tuple(int(size) for size in inp.shape)
	in_strides = tuple(int(stride) for stride in inp.stride())
	out_strides = tuple(int(stride) for stride in out.stride())
	out_tile_shape = _make_tile_shape(out_shape, 1024)
	in_tile_shape = tuple(out_tile_shape[dims.index(axis)] for axis in range(ndim))
	out_tile_counts = tuple(
		triton.cdiv(out_shape[axis], out_tile_shape[axis]) for axis in range(ndim)
	)
	inv_dims = tuple(dims.index(axis) for axis in range(ndim))
	total_tiles = 1
	for tile_count in out_tile_counts:
		total_tiles *= int(tile_count)

	mem_order = tuple(range(ndim - 1, -1, -1))
	boundary = tuple(range(ndim))

	permute_kernel[(NUM_CTAS,)](
		inp,
		out,
		n_elements,
		total_tiles,
		NDIM=ndim,
		IN_SHAPE=in_shape,
		IN_STRIDES=in_strides,
		OUT_SHAPE=out_shape,
		OUT_STRIDES=out_strides,
		DIMS=dims,
		OUT_TILE_SHAPE=out_tile_shape,
		IN_TILE_SHAPE=in_tile_shape,
		OUT_TILE_COUNTS=out_tile_counts,
		INV_DIMS=inv_dims,
		MEM_ORDER=mem_order,
		BOUNDARY=boundary,
	)
	return out
