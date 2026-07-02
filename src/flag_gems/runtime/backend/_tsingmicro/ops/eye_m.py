import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry

device_ = device


@libentry()
@triton.jit
def eye_kernel(
    out_ptr,
    LIMIT,        # min(N, M) — diagonal length
    M,            # row stride for flat indexing
    BLOCK: tl.constexpr,
):
    # 1D kernel: each CTA writes a BLOCK-long segment of the diagonal.
    # Off-diagonal cells stay zero (pre-filled by torch.zeros) — we never
    # touch them, eliminating the bulk of the previous write traffic.
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < LIMIT
    # fp32 vector path (Tx81 SIMD is fp-only).
    ones = tl.full([BLOCK], 1.0, dtype=tl.float32)
    diag_idx = off * M + off  # diagonal flat offset
    tl.store(out_ptr + diag_idx, ones.to(out_ptr.dtype.element_ty), mask=mask)


def eye_m(n, m, *, dtype=None, layout=torch.strided, device=None, pin_memory=None):
    """
    Triton-based implementation of torch.eye_m(n, m).

    Strategy: pre-zero the output (one fast tx.memset), then only launch
    enough CTAs to cover the diagonal — each CTA writes BLOCK ones along
    the i==j line. Off-diagonal tiles are skipped entirely.
    """
    logging.debug("GEMS EYE_M")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)
    if layout != torch.strided:
        raise ValueError("Currently only strided layout is supported for eye_m.")

    out = torch.zeros(
        (n, m), dtype=dtype, device=device, layout=layout, pin_memory=pin_memory
    )

    limit = min(n, m)
    if limit == 0:
        return out

    BLOCK = 256
    grid = (triton.cdiv(limit, BLOCK),)
    eye_kernel[grid](out, limit, m, BLOCK)
    return out
