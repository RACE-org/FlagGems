import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn

try:
    uint_to_uniform_float = tl.uint_to_uniform_float
except AttributeError:
    # Copied from triton.language package for compatibility
    @triton.jit
    def uint_to_uniform_float(x):
        """
        Numerically stable function to convert a random uint into a random float uniformly sampled in [0, 1).
        """
        if tl.constexpr(x.dtype == tl.uint32) or tl.constexpr(x.dtype == tl.int32):
            x = x.to(tl.int32, bitcast=True)
            scale = 4.6566127342e-10
        else:
            tl.static_assert(
                tl.constexpr(x.dtype == tl.uint64) or tl.constexpr(x.dtype == tl.int64)
            )
            x = x.to(tl.int64, bitcast=True)
            scale = 1.0842020432385337e-19
        x = tl.where(x < 0, -x - 1, x)
        return x * scale


def philox_backend_seed_offset(increment, generator=None):
    if generator is None:
        device = torch_device_fn.current_device()
        generator = torch_device_fn.default_generators[device]
    state_copy = generator.get_state()
    if flag_gems.vendor_name in ("kunlunxin", "aipu"):
        c0, c1 = state_copy.view(torch.int64)[-2], state_copy.view(torch.int64)[-1]
    else:
        c0, c1 = state_copy.view(torch.int64)

    seed, offset = int(c0), int(c1)
    increment = (increment + 3) // 4 * 4
    c1 += increment
    generator.set_state(state_copy)
    return seed, offset


@triton.jit
def uniform(seed, philox_offset, offset):
    seed = seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint64)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint64)
    i4 = offset
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(seed, c0, c1, _O, _O)
    r0 = uint_to_uniform_float(r0)
    r1 = uint_to_uniform_float(r1)
    r2 = uint_to_uniform_float(r2)
    r3 = uint_to_uniform_float(r3)
    return r0, r1, r2, r3
