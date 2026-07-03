# import logging

# import torch
# import triton
# import triton.language as tl

# from flag_gems.runtime import torch_device_fn
# from flag_gems.utils import libentry
# from flag_gems.utils import triton_lang_extension as tle

# logger = logging.getLogger(__name__)

# # ===========================================================================
# # Tx81 pad optimisation
# #
# # Core idea from doc/ops/pad.md: don't do per-element "am I in padding?" for
# # every output element.  Split by region — center is copy, padding is fill or
# # fixed mapping.
# #
# # 1. Constant pad → fill + copy.  No per-element if_pad, no mode logic.
# #     - Last-dim only  → 2D copy kernel (zero % or //)
# #     - Multi-dim      → rank-specialised copy (RANK: tl.constexpr, exact
# #                        %// count per rank, no wasted corrective chains)
# #
# # 2. Non-constant pad → region-split.
# #     - Last-dim only  → 2D kernel (left mapping | center copy | right mapping)
# #     - Multi-dim      → rank-specialised elementwise kernel
# #
# # All arithmetic is int32.  int64 is never introduced.
# #
# # Variable rank: kernels use RANK: tl.constexpr with per-rank if/elif
# # branches.  Only the matching branch compiles — no dead arithmetic.
# # Supported ranks: 1-5.  Rank 1 always hits the last-dim-only fast path.
# # Rank > 5 falls back to upstream.
# # ===========================================================================

# _MAX_RANK = 5


# # ===========================================================================
# # Kernel 1 — Constant pad, last-dim only (2D, zero %//, any rank)
# # ===========================================================================


# @libentry()
# @triton.jit
# def _constant_pad_copy_lastdim_kernel(
#     inp_ptr: tl.tensor,
#     out_ptr: tl.tensor,
#     M: int,
#     N: int,
#     pad_left: int,
#     out_row_stride: int,
#     BLOCK_M: tl.constexpr,
#     BLOCK_N: tl.constexpr,
# ):
#     """Copy inp[row, col] → out[row, pad_left + col].  Pure 2D, zero %//."""
#     pid_m = tle.program_id(0)
#     pid_n = tle.program_id(1)

#     rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
#     cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
#     rm = rows < M
#     cm = cols < N

#     val = tl.load(
#         inp_ptr + rows[:, None] * N + cols[None, :],
#         mask=rm[:, None] & cm[None, :],
#         other=0.0,
#     )
#     tl.store(
#         out_ptr + rows[:, None] * out_row_stride + pad_left + cols[None, :],
#         val,
#         mask=rm[:, None] & cm[None, :],
#     )


# # ===========================================================================
# # Kernel 2 — Constant pad, general dims (rank-specialised via constexpr)
# # ===========================================================================


# @libentry()
# @triton.jit
# def _constant_pad_copy_general_kernel(
#     inp_ptr: tl.tensor,
#     out_ptr: tl.tensor,
#     N_total: int,
#     RANK: tl.constexpr,
#     # shapes, strides, pad_before  — always _MAX_RANK args; only RANK used
#     is0: int, is1: int, is2: int, is3: int, is4: int,
#     irst0: int, irst1: int, irst2: int, irst3: int, irst4: int,
#     pb0: int, pb1: int, pb2: int, pb3: int, pb4: int,
#     os0: int, os1: int, os2: int, os3: int, os4: int,
#     BLOCK_SIZE: tl.constexpr,
# ):
#     """Copy each input element to its padded output position.

#     Iterates over *input* elements (fewer than output when padding > 0).
#     RANK selects the decomposition path at compile time — only the exact
#     number of %// ops for the actual rank is emitted.
#     """
#     pid = tle.program_id(0)
#     ctas = tle.num_programs(0)
#     for j in range(tl.cdiv(tl.cdiv(N_total, BLOCK_SIZE), ctas)):
#         block_id = pid + j * ctas
#         off = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#         mask = off < N_total

#         # ---- coordinate decomposition: exact RANK %// ops, no waste ----
#         if RANK == 1:
#             i0 = off
#             inp_off = i0 * irst0
#             out_off = (i0 + pb0) * os0
#         elif RANK == 2:
#             cur = off
#             i1 = cur % is1; cur = cur // is1
#             i0 = cur
#             inp_off = i0 * irst0 + i1 * irst1
#             out_off = (i0 + pb0) * os0 + (i1 + pb1) * os1
#         elif RANK == 3:
#             cur = off
#             i2 = cur % is2; cur = cur // is2
#             i1 = cur % is1; cur = cur // is1
#             i0 = cur
#             inp_off = i0 * irst0 + i1 * irst1 + i2 * irst2
#             out_off = (i0 + pb0) * os0 + (i1 + pb1) * os1 + (i2 + pb2) * os2
#         elif RANK == 4:
#             cur = off
#             i3 = cur % is3; cur = cur // is3
#             i2 = cur % is2; cur = cur // is2
#             i1 = cur % is1; cur = cur // is1
#             i0 = cur
#             inp_off = i0 * irst0 + i1 * irst1 + i2 * irst2 + i3 * irst3
#             out_off = (i0 + pb0) * os0 + (i1 + pb1) * os1 + (i2 + pb2) * os2 + (i3 + pb3) * os3
#         elif RANK == 5:
#             cur = off
#             i4 = cur % is4; cur = cur // is4
#             i3 = cur % is3; cur = cur // is3
#             i2 = cur % is2; cur = cur // is2
#             i1 = cur % is1; cur = cur // is1
#             i0 = cur
#             inp_off = i0 * irst0 + i1 * irst1 + i2 * irst2 + i3 * irst3 + i4 * irst4
#             out_off = (i0 + pb0) * os0 + (i1 + pb1) * os1 + (i2 + pb2) * os2 + (i3 + pb3) * os3 + (i4 + pb4) * os4

#         val = tl.load(inp_ptr + inp_off, mask=mask, other=0.0)
#         tl.store(out_ptr + out_off, val, mask=mask)


# # ===========================================================================
# # Kernel 3 — Non-constant pad, last-dim only (2D region-split, any rank)
# # ===========================================================================


# @libentry()
# @triton.jit
# def _pad_nonconstant_lastdim_kernel(
#     inp_ptr: tl.tensor,
#     out_ptr: tl.tensor,
#     M: int,
#     N: int,
#     pad_left: int,
#     pad_right: int,
#     out_row_stride: int,
#     MODE: tl.constexpr,  # 1=reflect, 2=replicate, 3=circular
#     BLOCK_M: tl.constexpr,
#     BLOCK_N: tl.constexpr,
# ):
#     """Last-dim non-constant pad with three regions per row:
#     left border (mode mapping) | center (direct copy) | right border (mapping).

#     MODE is constexpr → dead branches eliminated at compile time.
#     """
#     pid_m = tle.program_id(0)
#     pid_n = tle.program_id(1)

#     rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
#     cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
#     rm = rows < M
#     out_row = rows[:, None]
#     out_col = cols[None, :]
#     out_len = pad_left + N + pad_right
#     out_mask = rm[:, None] & (out_col < out_len)

#     is_left = out_col < pad_left
#     is_center = (out_col >= pad_left) & (out_col < pad_left + N)

#     if MODE == 1:  # reflect
#         src_col = tl.where(
#             is_left,
#             pad_left - out_col,
#             tl.where(is_center, out_col - pad_left,
#                      2 * N + pad_left - out_col - 2),
#         )
#     elif MODE == 2:  # replicate
#         src_col = tl.where(
#             is_left,
#             tl.zeros_like(out_col),
#             tl.where(is_center, out_col - pad_left,
#                      tl.full_like(out_col, N - 1)),
#         )
#     elif MODE == 3:  # circular
#         src_col = tl.where(
#             is_left,
#             out_col + N - pad_left,
#             tl.where(is_center, out_col - pad_left,
#                      out_col - N - pad_left),
#         )

#     val = tl.load(inp_ptr + out_row * N + src_col, mask=out_mask, other=0.0)
#     tl.store(out_ptr + out_row * out_row_stride + out_col, val, mask=out_mask)


# # ===========================================================================
# # Kernel 4 — Non-constant pad, general dims (rank-specialised via constexpr)
# # ===========================================================================


# @libentry()
# @triton.jit
# def _pad_nonconstant_general_kernel(
#     inp_ptr: tl.tensor,
#     out_ptr: tl.tensor,
#     N_total: int,
#     RANK: tl.constexpr,
#     MODE: tl.constexpr,
#     # output shapes, input shapes, input strides, pad_before, pad_after
#     os0: int, os1: int, os2: int, os3: int, os4: int,
#     is0: int, is1: int, is2: int, is3: int, is4: int,
#     irst0: int, irst1: int, irst2: int, irst3: int, irst4: int,
#     pb0: int, pb1: int, pb2: int, pb3: int, pb4: int,
#     pa0: int, pa1: int, pa2: int, pa3: int, pa4: int,
#     BLOCK_SIZE: tl.constexpr,
# ):
#     """General non-constant pad.  Iterates over output elements, decomposes
#     flat index → coords, applies per-dim mode mapping, loads from input.

#     RANK: tl.constexpr → only matching branch compiles.
#     MODE: tl.constexpr → dead tl.where branches eliminated in each block.
#     """
#     pid = tle.program_id(0)
#     ctas = tle.num_programs(0)
#     for j in range(tl.cdiv(tl.cdiv(N_total, BLOCK_SIZE), ctas)):
#         block_id = pid + j * ctas
#         off = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#         mask = off < N_total

#         # ----  Per-rank coordinate decomposition + per-dim mode mapping  ----
#         # Each branch is self-contained — no variables leak across branches.
#         if RANK == 1:
#             d0 = off
#             left0 = d0 < pb0
#             right0 = d0 >= (pb0 + is0)
#             if MODE == 1:
#                 s0 = tl.where(left0, pb0 - d0,
#                               tl.where(right0, 2 * is0 + pb0 - d0 - 2, d0 - pb0))
#             elif MODE == 2:
#                 s0 = tl.where(left0, 0,
#                               tl.where(right0, is0 - 1, d0 - pb0))
#             else:
#                 s0 = tl.where(left0, d0 + is0 - pb0,
#                               tl.where(right0, d0 - is0 - pb0, d0 - pb0))
#             inp_off = s0 * irst0

#         elif RANK == 2:
#             cur = off
#             d1 = cur % os1; cur = cur // os1
#             d0 = cur

#             left1 = d1 < pb1; right1 = d1 >= (pb1 + is1)
#             left0 = d0 < pb0; right0 = d0 >= (pb0 + is0)

#             if MODE == 1:
#                 s1 = tl.where(left1, pb1 - d1,
#                               tl.where(right1, 2 * is1 + pb1 - d1 - 2, d1 - pb1))
#                 s0 = tl.where(left0, pb0 - d0,
#                               tl.where(right0, 2 * is0 + pb0 - d0 - 2, d0 - pb0))
#             elif MODE == 2:
#                 s1 = tl.where(left1, 0,
#                               tl.where(right1, is1 - 1, d1 - pb1))
#                 s0 = tl.where(left0, 0,
#                               tl.where(right0, is0 - 1, d0 - pb0))
#             else:
#                 s1 = tl.where(left1, d1 + is1 - pb1,
#                               tl.where(right1, d1 - is1 - pb1, d1 - pb1))
#                 s0 = tl.where(left0, d0 + is0 - pb0,
#                               tl.where(right0, d0 - is0 - pb0, d0 - pb0))
#             inp_off = s0 * irst0 + s1 * irst1

#         elif RANK == 3:
#             cur = off
#             d2 = cur % os2; cur = cur // os2
#             d1 = cur % os1; cur = cur // os1
#             d0 = cur

#             left2_3 = d2 < pb2; right2_3 = d2 >= (pb2 + is2)
#             left1_3 = d1 < pb1; right1_3 = d1 >= (pb1 + is1)
#             left0_3 = d0 < pb0; right0_3 = d0 >= (pb0 + is0)

#             if MODE == 1:
#                 s2 = tl.where(left2_3, pb2 - d2,
#                               tl.where(right2_3, 2 * is2 + pb2 - d2 - 2, d2 - pb2))
#                 s1 = tl.where(left1_3, pb1 - d1,
#                               tl.where(right1_3, 2 * is1 + pb1 - d1 - 2, d1 - pb1))
#                 s0 = tl.where(left0_3, pb0 - d0,
#                               tl.where(right0_3, 2 * is0 + pb0 - d0 - 2, d0 - pb0))
#             elif MODE == 2:
#                 s2 = tl.where(left2_3, 0,
#                               tl.where(right2_3, is2 - 1, d2 - pb2))
#                 s1 = tl.where(left1_3, 0,
#                               tl.where(right1_3, is1 - 1, d1 - pb1))
#                 s0 = tl.where(left0_3, 0,
#                               tl.where(right0_3, is0 - 1, d0 - pb0))
#             else:
#                 s2 = tl.where(left2_3, d2 + is2 - pb2,
#                               tl.where(right2_3, d2 - is2 - pb2, d2 - pb2))
#                 s1 = tl.where(left1_3, d1 + is1 - pb1,
#                               tl.where(right1_3, d1 - is1 - pb1, d1 - pb1))
#                 s0 = tl.where(left0_3, d0 + is0 - pb0,
#                               tl.where(right0_3, d0 - is0 - pb0, d0 - pb0))
#             inp_off = s0 * irst0 + s1 * irst1 + s2 * irst2

#         elif RANK == 4:
#             cur = off
#             d3 = cur % os3; cur = cur // os3
#             d2 = cur % os2; cur = cur // os2
#             d1 = cur % os1; cur = cur // os1
#             d0 = cur

#             left3 = d3 < pb3; right3 = d3 >= (pb3 + is3)
#             left2 = d2 < pb2; right2 = d2 >= (pb2 + is2)
#             left1 = d1 < pb1; right1 = d1 >= (pb1 + is1)
#             left0 = d0 < pb0; right0 = d0 >= (pb0 + is0)

#             if MODE == 1:
#                 s3 = tl.where(left3, pb3 - d3,
#                               tl.where(right3, 2 * is3 + pb3 - d3 - 2, d3 - pb3))
#                 s2 = tl.where(left2, pb2 - d2,
#                               tl.where(right2, 2 * is2 + pb2 - d2 - 2, d2 - pb2))
#                 s1 = tl.where(left1, pb1 - d1,
#                               tl.where(right1, 2 * is1 + pb1 - d1 - 2, d1 - pb1))
#                 s0 = tl.where(left0, pb0 - d0,
#                               tl.where(right0, 2 * is0 + pb0 - d0 - 2, d0 - pb0))
#             elif MODE == 2:
#                 s3 = tl.where(left3, 0,
#                               tl.where(right3, is3 - 1, d3 - pb3))
#                 s2 = tl.where(left2, 0,
#                               tl.where(right2, is2 - 1, d2 - pb2))
#                 s1 = tl.where(left1, 0,
#                               tl.where(right1, is1 - 1, d1 - pb1))
#                 s0 = tl.where(left0, 0,
#                               tl.where(right0, is0 - 1, d0 - pb0))
#             else:
#                 s3 = tl.where(left3, d3 + is3 - pb3,
#                               tl.where(right3, d3 - is3 - pb3, d3 - pb3))
#                 s2 = tl.where(left2, d2 + is2 - pb2,
#                               tl.where(right2, d2 - is2 - pb2, d2 - pb2))
#                 s1 = tl.where(left1, d1 + is1 - pb1,
#                               tl.where(right1, d1 - is1 - pb1, d1 - pb1))
#                 s0 = tl.where(left0, d0 + is0 - pb0,
#                               tl.where(right0, d0 - is0 - pb0, d0 - pb0))
#             inp_off = s0 * irst0 + s1 * irst1 + s2 * irst2 + s3 * irst3

#         elif RANK == 5:
#             cur = off
#             d4 = cur % os4; cur = cur // os4
#             d3 = cur % os3; cur = cur // os3
#             d2 = cur % os2; cur = cur // os2
#             d1 = cur % os1; cur = cur // os1
#             d0 = cur

#             left4 = d4 < pb4; right4 = d4 >= (pb4 + is4)
#             left3 = d3 < pb3; right3 = d3 >= (pb3 + is3)
#             left2 = d2 < pb2; right2 = d2 >= (pb2 + is2)
#             left1 = d1 < pb1; right1 = d1 >= (pb1 + is1)
#             left0 = d0 < pb0; right0 = d0 >= (pb0 + is0)

#             if MODE == 1:
#                 s4 = tl.where(left4, pb4 - d4,
#                               tl.where(right4, 2 * is4 + pb4 - d4 - 2, d4 - pb4))
#                 s3 = tl.where(left3, pb3 - d3,
#                               tl.where(right3, 2 * is3 + pb3 - d3 - 2, d3 - pb3))
#                 s2 = tl.where(left2, pb2 - d2,
#                               tl.where(right2, 2 * is2 + pb2 - d2 - 2, d2 - pb2))
#                 s1 = tl.where(left1, pb1 - d1,
#                               tl.where(right1, 2 * is1 + pb1 - d1 - 2, d1 - pb1))
#                 s0 = tl.where(left0, pb0 - d0,
#                               tl.where(right0, 2 * is0 + pb0 - d0 - 2, d0 - pb0))
#             elif MODE == 2:
#                 s4 = tl.where(left4, 0,
#                               tl.where(right4, is4 - 1, d4 - pb4))
#                 s3 = tl.where(left3, 0,
#                               tl.where(right3, is3 - 1, d3 - pb3))
#                 s2 = tl.where(left2, 0,
#                               tl.where(right2, is2 - 1, d2 - pb2))
#                 s1 = tl.where(left1, 0,
#                               tl.where(right1, is1 - 1, d1 - pb1))
#                 s0 = tl.where(left0, 0,
#                               tl.where(right0, is0 - 1, d0 - pb0))
#             else:
#                 s4 = tl.where(left4, d4 + is4 - pb4,
#                               tl.where(right4, d4 - is4 - pb4, d4 - pb4))
#                 s3 = tl.where(left3, d3 + is3 - pb3,
#                               tl.where(right3, d3 - is3 - pb3, d3 - pb3))
#                 s2 = tl.where(left2, d2 + is2 - pb2,
#                               tl.where(right2, d2 - is2 - pb2, d2 - pb2))
#                 s1 = tl.where(left1, d1 + is1 - pb1,
#                               tl.where(right1, d1 - is1 - pb1, d1 - pb1))
#                 s0 = tl.where(left0, d0 + is0 - pb0,
#                               tl.where(right0, d0 - is0 - pb0, d0 - pb0))
#             inp_off = s0 * irst0 + s1 * irst1 + s2 * irst2 + s3 * irst3 + s4 * irst4

#         val = tl.load(inp_ptr + inp_off, mask=mask, other=0.0)
#         tl.store(out_ptr + off, val, mask=mask)


# # ===========================================================================
# # Host helpers
# # ===========================================================================


# def _parse_pad(pad, ndim):
#     """Convert PyTorch pad tuple → pad_before / pad_after arrays.

#     `pad` is specified from the last dimension backward:
#     (left_last, right_last, left_second_last, right_second_last, ...)
#     """
#     pad_size = len(pad)
#     assert pad_size % 2 == 0
#     pad_pair = pad_size // 2
#     pad_before = [0] * ndim
#     pad_after = [0] * ndim
#     for i in range(pad_pair):
#         pad_before[ndim - i - 1] = pad[2 * i]
#         pad_after[ndim - i - 1] = pad[2 * i + 1]
#     return pad_before, pad_after


# def _is_only_lastdim(pad_before, pad_after, ndim):
#     """True when only the last dimension has non-zero padding."""
#     for i in range(ndim - 1):
#         if pad_before[i] != 0 or pad_after[i] != 0:
#             return False
#     return True


# def _pad_values(values, max_rank):
#     """Pad a list of per-dim values to max_rank (trailing-pad with 0s).

#     TRAILING pad so kernel accesses indices 0..RANK-1 (actual values),
#     not the padding.
#     """
#     rank = len(values)
#     pad_n = max_rank - rank
#     return tuple(values) + (0,) * pad_n if pad_n > 0 else tuple(values)


# def _pad_tuple(tup, max_rank, pad_val=0):
#     """Trailing-pad a tuple to max_rank.

#     TRAILING pad — kernel accesses indices 0..RANK-1 as actual values,
#     padding values at RANK..max_rank-1 are never touched (RANK: tl.constexpr).

#     pad_val=1 for shapes (%1=0, //1=no-op if accidentally accessed),
#     pad_val=0 for strides/pad values.
#     """
#     rank = len(tup)
#     pad_n = max_rank - rank
#     if pad_n == 0:
#         return tuple(tup)
#     return tuple(tup) + (pad_val,) * pad_n


# # ===========================================================================
# # Main entry points
# # ===========================================================================


# def pad(self, pad, mode="constant", value=None):
#     logger.debug("GEMS TSINGMICRO PAD")

#     if value is None:
#         value = 0.0
#     value = float(value)

#     ndim = self.ndim

#     # ---- validation (mirrors upstream) ----
#     if mode == "reflect":
#         assert len(pad) == 2 * ndim, (
#             f"padding size is expected to be {2 * ndim}, but got {len(pad)}"
#         )
#         for i in range(ndim):
#             pad_l = pad[2 * i]
#             pad_r = pad[2 * i + 1]
#             input_size = self.shape[ndim - i - 1]
#             assert pad_l < input_size and pad_r < input_size, (
#                 "padding size should be less than the corresponding "
#                 "input dimension"
#             )

#     if mode == "circular":
#         assert len(pad) == 2 * ndim, (
#             f"padding size is expected to be {2 * ndim}, but got {len(pad)}"
#         )
#         for i in range(ndim):
#             pad_l = pad[2 * i]
#             pad_r = pad[2 * i + 1]
#             input_size = self.shape[ndim - i - 1]
#             assert pad_l <= input_size and pad_r <= input_size, (
#                 "Padding value causes wrapping around more than once."
#             )

#     pad_before, pad_after = _parse_pad(pad, ndim)
#     dst_shape = [
#         self.shape[i] + pad_before[i] + pad_after[i] for i in range(ndim)
#     ]

#     # Rank > 5 pad is essentially non-existent in practice.
#     assert ndim <= _MAX_RANK, (
#         f"Tx81 pad supports rank ≤ {_MAX_RANK}, got {ndim}"
#     )

#     with torch_device_fn.device(self.device):
#         out = torch.empty(dst_shape, dtype=self.dtype, device=self.device)

#         # =================================================================
#         # Constant mode — fill + copy
#         # =================================================================
#         if mode == "constant":
#             out.fill_(value)

#             if _is_only_lastdim(pad_before, pad_after, ndim):
#                 _launch_constant_copy_lastdim(self, out, pad_before)
#             else:
#                 _launch_constant_copy_general(
#                     self, out, pad_before, pad_after
#                 )
#             return out

#         # =================================================================
#         # Non-constant mode — region-split or elementwise
#         # =================================================================
#         if mode in ("reflect", "replicate", "circular"):
#             if _is_only_lastdim(pad_before, pad_after, ndim) and self.is_contiguous():
#                 _launch_nonconstant_lastdim(
#                     self, out, pad_before, pad_after, mode
#                 )
#             else:
#                 _launch_nonconstant_general(
#                     self, out, pad_before, pad_after, mode
#                 )
#             return out

#         raise ValueError(f"Unsupported pad mode: {mode}")


# def constant_pad_nd(self, pad_spec, value=0):
#     return pad(self, pad_spec, mode="constant", value=value)


# # ===========================================================================
# # Launch helpers
# # ===========================================================================


# def _launch_constant_copy_lastdim(inp, out, pad_before):
#     """2D copy of input into output interior, last-dim padding only.
#     Flatten all prefix dims → M, last dim → N.  Zero %//."""
#     N = inp.shape[-1]
#     pad_l = pad_before[-1]
#     out_row_stride = out.shape[-1]
#     M = inp.numel() // N

#     BLOCK_M = min(triton.next_power_of_2(M), 256)
#     BLOCK_N = min(triton.next_power_of_2(N), 512)
#     grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

#     _constant_pad_copy_lastdim_kernel[grid](
#         inp.reshape(M, N), out.reshape(M, out_row_stride),
#         M, N, pad_l, out_row_stride,
#         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
#     )


# def _launch_constant_copy_general(inp, out, pad_before, pad_after):
#     """Elementwise copy with exact-rank coordinate decomposition.
#     Only RANK %// ops per element — no waste on padded dims."""
#     N_total = inp.numel()
#     rank = inp.ndim

#     # Pad to _MAX_RANK for uniform kernel signature.  Unused trailing args
#     # are never accessed (RANK: tl.constexpr branch eliminates them).
#     ish = _pad_tuple(inp.shape, _MAX_RANK, pad_val=1)
#     ist = inp.stride()
#     ist_padded = _pad_tuple(ist, _MAX_RANK, pad_val=0)
#     pb = _pad_values(pad_before, _MAX_RANK)
#     os_padded = _pad_tuple(out.stride(), _MAX_RANK, pad_val=0)

#     BLOCK_SIZE = min(512, triton.next_power_of_2(N_total))
#     grid = (min(16, triton.cdiv(N_total, BLOCK_SIZE)),)

#     _constant_pad_copy_general_kernel[grid](
#         inp, out, N_total, rank,
#         int(ish[0]), int(ish[1]), int(ish[2]), int(ish[3]), int(ish[4]),
#         int(ist_padded[0]), int(ist_padded[1]), int(ist_padded[2]),
#         int(ist_padded[3]), int(ist_padded[4]),
#         int(pb[0]), int(pb[1]), int(pb[2]), int(pb[3]), int(pb[4]),
#         int(os_padded[0]), int(os_padded[1]), int(os_padded[2]),
#         int(os_padded[3]), int(os_padded[4]),
#         BLOCK_SIZE=BLOCK_SIZE,
#     )


# def _launch_nonconstant_lastdim(inp, out, pad_before, pad_after, mode):
#     """2D region-split for last-dim-only non-constant pad.  Any rank."""
#     N = inp.shape[-1]
#     pad_l = pad_before[-1]
#     pad_r = pad_after[-1]
#     out_row_stride = out.shape[-1]
#     M = inp.numel() // N

#     MODE_MAP = {"reflect": 1, "replicate": 2, "circular": 3}
#     mode_const = MODE_MAP[mode]

#     BLOCK_M = min(triton.next_power_of_2(M), 256)
#     BLOCK_N = min(triton.next_power_of_2(out_row_stride), 512)
#     grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(out_row_stride, BLOCK_N))

#     _pad_nonconstant_lastdim_kernel[grid](
#         inp.reshape(M, N), out.reshape(M, out_row_stride),
#         M, N, pad_l, pad_r, out_row_stride,
#         MODE=mode_const,
#         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
#     )


# def _launch_nonconstant_general(inp, out, pad_before, pad_after, mode):
#     """Elementwise non-constant pad with exact-rank coordinate decomposition
#     and per-dim mode mapping."""
#     N_total = out.numel()
#     rank = inp.ndim

#     osh = _pad_tuple(out.shape, _MAX_RANK, pad_val=1)
#     ish = _pad_tuple(inp.shape, _MAX_RANK, pad_val=1)
#     ist_padded = _pad_tuple(inp.stride(), _MAX_RANK, pad_val=0)
#     pb = _pad_values(pad_before, _MAX_RANK)
#     pa = _pad_values(pad_after, _MAX_RANK)

#     MODE_MAP = {"reflect": 1, "replicate": 2, "circular": 3}
#     mode_const = MODE_MAP[mode]

#     BLOCK_SIZE = min(512, triton.next_power_of_2(N_total))
#     grid = (min(16, triton.cdiv(N_total, BLOCK_SIZE)),)

#     _pad_nonconstant_general_kernel[grid](
#         inp, out, N_total, rank, mode_const,
#         int(osh[0]), int(osh[1]), int(osh[2]), int(osh[3]), int(osh[4]),
#         int(ish[0]), int(ish[1]), int(ish[2]), int(ish[3]), int(ish[4]),
#         int(ist_padded[0]), int(ist_padded[1]), int(ist_padded[2]),
#         int(ist_padded[3]), int(ist_padded[4]),
#         int(pb[0]), int(pb[1]), int(pb[2]), int(pb[3]), int(pb[4]),
#         int(pa[0]), int(pa[1]), int(pa[2]), int(pa[3]), int(pa[4]),
#         BLOCK_SIZE=BLOCK_SIZE,
#     )


import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.pad import pad as _generic_pad
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_MAX_RANK = 5


@libentry()
@triton.jit
def _constant_pad_copy_lastdim_kernel(
    inp_ptr: tl.tensor,
    out_ptr: tl.tensor,
    M: int,
    N: int,
    OUT_N: int,
    PAD_LEFT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Copy contiguous interior when only the last dimension is padded."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (cols[None, :] < N)

    vals = tl.load(inp_ptr + rows[:, None] * N + cols[None, :], mask=mask, other=0.0)
    tl.store(
        out_ptr + rows[:, None] * OUT_N + (cols[None, :] + PAD_LEFT),
        vals,
        mask=mask,
    )


@libentry()
@triton.jit
def _pad_lastdim_nonconstant_kernel(
    inp_ptr: tl.tensor,
    out_ptr: tl.tensor,
    M: int,
    N: int,
    OUT_N: int,
    PAD_LEFT: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Reflect/replicate/circular pad when only the last dimension is padded."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_end = PAD_LEFT + N

    src = cols - PAD_LEFT
    if MODE == 0:  # reflect
        src = tl.where(cols < PAD_LEFT, PAD_LEFT - cols, src)
        src = tl.where(cols >= valid_end, (N + PAD_LEFT - 1) * 2 - cols - PAD_LEFT, src)
    elif MODE == 1:  # replicate
        src = tl.where(cols < PAD_LEFT, 0, src)
        src = tl.where(cols >= valid_end, N - 1, src)
    else:  # circular
        src = tl.where(cols < PAD_LEFT, cols + N - PAD_LEFT, src)
        src = tl.where(cols >= valid_end, cols - valid_end, src)

    mask = (rows[:, None] < M) & (cols[None, :] < OUT_N)
    vals = tl.load(inp_ptr + rows[:, None] * N + src[None, :], mask=mask, other=0.0)
    tl.store(out_ptr + rows[:, None] * OUT_N + cols[None, :], vals, mask=mask)


@libentry()
@triton.jit
def _constant_pad_copy_general_kernel(
    inp_ptr: tl.tensor,
    out_ptr: tl.tensor,
    n_elements: int,
    s0: int,
    s1: int,
    s2: int,
    s3: int,
    s4: int,
    p0: int,
    p1: int,
    p2: int,
    p3: int,
    p4: int,
    os0: int,
    os1: int,
    os2: int,
    os3: int,
    os4: int,
    BLOCK: tl.constexpr,
):
    """Copy input elements into the already-filled output interior.

    This avoids the upstream per-output-element if_pad branch.  All padded
    elements are produced by the prior out.fill_(value); the kernel only writes
    the valid interior.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    rem = offsets
    i4 = rem % s4
    rem = rem // s4
    i3 = rem % s3
    rem = rem // s3
    i2 = rem % s2
    rem = rem // s2
    i1 = rem % s1
    i0 = rem // s1

    out_offsets = (
        (i0 + p0) * os0
        + (i1 + p1) * os1
        + (i2 + p2) * os2
        + (i3 + p3) * os3
        + (i4 + p4) * os4
    )
    vals = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + out_offsets, vals, mask=mask)


def _split_pad(shape, pad):
    ndim = len(shape)
    pad_size = len(pad)
    assert pad_size % 2 == 0
    pad_pair = pad_size // 2
    assert pad_pair <= ndim

    pad_before = [0 for _ in range(ndim)]
    pad_after = [0 for _ in range(ndim)]
    for i in range(pad_pair):
        pad_before[ndim - i - 1] = int(pad[2 * i])
        pad_after[ndim - i - 1] = int(pad[2 * i + 1])
    return pad_before, pad_after


def _pad_front(values, target, fill):
    values = tuple(int(v) for v in values)
    return (fill,) * (target - len(values)) + values


def _make_out_shape(shape, pad_before, pad_after):
    return [
        int(size) + int(before) + int(after)
        for size, before, after in zip(shape, pad_before, pad_after)
    ]


def _can_use_constant_fast_path(inp, pad_before, pad_after, out_shape):
    if inp.ndim == 0 or inp.ndim > _MAX_RANK:
        return False
    if any(p < 0 for p in pad_before) or any(p < 0 for p in pad_after):
        return False
    if any(dim < 0 for dim in out_shape):
        return False
    return True


def _only_lastdim_padded(pad_before, pad_after):
    return all(p == 0 for p in pad_before[:-1]) and all(
        p == 0 for p in pad_after[:-1]
    )


def _constant_pad_copy_lastdim(inp_c, out, pad_before, pad_after):
    n = inp_c.shape[-1]
    out_n = out.shape[-1]
    m = inp_c.numel() // n
    pad_left = pad_before[-1]

    block_n = min(triton.next_power_of_2(n), 1024)
    block_m = min(triton.next_power_of_2(max(m, 1)), max(1, 32768 // block_n), 32)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))

    with torch_device_fn.device(inp_c.device):
        _constant_pad_copy_lastdim_kernel[grid](
            inp_c,
            out,
            m,
            n,
            out_n,
            PAD_LEFT=pad_left,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=4,
    )


def _pad_lastdim_nonconstant(inp, out, pad_before, mode):
    inp_c = inp.contiguous()
    n = inp_c.shape[-1]
    out_n = out.shape[-1]
    m = inp_c.numel() // n
    pad_left = pad_before[-1]
    mode_id = {"reflect": 0, "replicate": 1, "circular": 2}[mode]

    block_n = min(triton.next_power_of_2(out_n), 1024)
    block_m = min(triton.next_power_of_2(max(m, 1)), max(1, 32768 // block_n), 32)
    grid = (triton.cdiv(m, block_m), triton.cdiv(out_n, block_n))

    with torch_device_fn.device(inp.device):
        _pad_lastdim_nonconstant_kernel[grid](
            inp_c,
            out,
            m,
            n,
            out_n,
            PAD_LEFT=pad_left,
            MODE=mode_id,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=4,
        )


def _constant_pad_copy_general(inp_c, out, pad_before):
    shapes = _pad_front(inp_c.shape, _MAX_RANK, 1)
    pads = _pad_front(pad_before, _MAX_RANK, 0)
    out_strides = _pad_front(out.stride(), _MAX_RANK, 0)

    block = 2048
    grid = (triton.cdiv(inp_c.numel(), block),)
    with torch_device_fn.device(inp_c.device):
        _constant_pad_copy_general_kernel[grid](
            inp_c,
            out,
            inp_c.numel(),
            shapes[0],
            shapes[1],
            shapes[2],
            shapes[3],
            shapes[4],
            pads[0],
            pads[1],
            pads[2],
            pads[3],
            pads[4],
            out_strides[0],
            out_strides[1],
            out_strides[2],
            out_strides[3],
            out_strides[4],
            BLOCK=block,
            num_warps=4,
        )


def _constant_pad_nd(inp, pad, value):
    pad_before, pad_after = _split_pad(inp.shape, pad)
    out_shape = _make_out_shape(inp.shape, pad_before, pad_after)
    if not _can_use_constant_fast_path(inp, pad_before, pad_after, out_shape):
        return _generic_pad(inp, pad, mode="constant", value=value)

    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    out.fill_(value)

    if inp.numel() == 0:
        return out

    inp_c = inp.contiguous()
    if _only_lastdim_padded(pad_before, pad_after):
        _constant_pad_copy_lastdim(inp_c, out, pad_before, pad_after)
    else:
        _constant_pad_copy_general(inp_c, out, pad_before)
    return out


def _validate_nonconstant_mode(inp, pad_before, pad_after, mode):
    if any(p < 0 for p in pad_before) or any(p < 0 for p in pad_after):
        return False
    n = inp.shape[-1]
    pad_left = pad_before[-1]
    pad_right = pad_after[-1]
    if mode == "reflect":
        return pad_left < n and pad_right < n
    if mode == "circular":
        return pad_left <= n and pad_right <= n
    return mode == "replicate"


def _nonconstant_pad_lastdim(inp, pad, mode):
    pad_before, pad_after = _split_pad(inp.shape, pad)
    out_shape = _make_out_shape(inp.shape, pad_before, pad_after)
    if (
        inp.ndim == 0
        or inp.ndim > _MAX_RANK
        or not _only_lastdim_padded(pad_before, pad_after)
        or any(dim < 0 for dim in out_shape)
        or not _validate_nonconstant_mode(inp, pad_before, pad_after, mode)
    ):
        return _generic_pad(inp, pad, mode=mode, value=0.0)

    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    if inp.numel() == 0:
        return out
    _pad_lastdim_nonconstant(inp, out, pad_before, mode)
    return out


def pad(self, pad, mode="constant", value=None):
    logger.debug("GEMS_TSINGMICRO PAD")
    if value is None:
        value = 0.0

    if mode == "constant":
        return _constant_pad_nd(self, pad, value)

    if mode in ("reflect", "replicate", "circular"):
        return _nonconstant_pad_lastdim(self, pad, mode)

    return _generic_pad(self, pad, mode=mode, value=value)


def constant_pad_nd(self, padding, value=0):
    return pad(self, padding, mode="constant", value=value)
