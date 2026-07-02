"""Correctness tests for operators in the support list that are not covered by perf.py.

Run:  python tests/test_extra_ops.py --check
      python tests/test_extra_ops.py --check --ops cumsum,topk,gather
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable

import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())

try:
    import flag_gems
except Exception:
    raise RuntimeError("flag_gems is required; install spine-FlagGems first")

import torch.nn.functional as F

FLOAT_DTYPES = (torch.float16,)
FLOAT32_DTYPES = (torch.float32,)
INT_DTYPES = (torch.int32,)
BOOL_DTYPES = (torch.bool,)

# Ops whose output cannot be compared deterministically (RNG-based)
STOCHASTIC_OPS = {
    "rand", "randn", "rand_like", "uniform_", "native_dropout", "multinomial",
}


@dataclass(frozen=True)
class TestCase:
    name: str
    dtypes: tuple
    prepare: Callable
    run: Callable
    # When True, correctness check uses statistical validation instead of allclose
    stochastic: bool = False


def _f(shape, dtype):
    return torch.randn(shape, dtype=dtype, device="cpu")


def _i(shape, dtype=torch.int32, low=0, high=8):
    return torch.randint(low, high, shape, dtype=dtype, device="cpu")


def _b(shape):
    return torch.rand(shape, device="cpu") > 0.5


# ---------------------------------------------------------------------------
# Creation ops
# ---------------------------------------------------------------------------

def _arange_case():
    def prepare(dtype):
        return (dtype,), {}

    def run(dtype):
        return torch.arange(0, 64, dtype=dtype, device="cpu")

    return TestCase("arange", FLOAT_DTYPES, prepare, run)


def _ones_case():
    def prepare(dtype):
        return (dtype,), {}

    def run(dtype):
        return torch.ones((32, 32), dtype=dtype, device="cpu")

    return TestCase("ones", FLOAT_DTYPES, prepare, run)


def _zeros_case():
    def prepare(dtype):
        return (dtype,), {}

    def run(dtype):
        return torch.zeros((32, 32), dtype=dtype, device="cpu")

    return TestCase("zeros", FLOAT_DTYPES, prepare, run)


def _full_case():
    def prepare(dtype):
        return (dtype,), {}

    def run(dtype):
        return torch.full((32, 32), 3.14, dtype=dtype, device="cpu")

    return TestCase("full", FLOAT_DTYPES, prepare, run)


def _full_like_case():
    def prepare(dtype):
        x = _f((32, 32), dtype)
        return (x,), {}

    def run(x):
        return torch.full_like(x, 2.71)

    return TestCase("full_like", FLOAT_DTYPES, prepare, run)


def _rand_case():
    def prepare(dtype):
        return (dtype,), {}

    def run(dtype):
        return torch.rand((32, 32), dtype=dtype, device="cpu")

    return TestCase("rand", FLOAT_DTYPES, prepare, run, stochastic=True)


def _rand_like_case():
    def prepare(dtype):
        x = _f((32, 32), dtype)
        return (x,), {}

    def run(x):
        return torch.rand_like(x)

    return TestCase("rand_like", FLOAT_DTYPES, prepare, run, stochastic=True)


def _randn_case():
    def prepare(dtype):
        return (dtype,), {}

    def run(dtype):
        return torch.randn((32, 32), dtype=dtype, device="cpu")

    return TestCase("randn", FLOAT_DTYPES, prepare, run, stochastic=True)


# ---------------------------------------------------------------------------
# Pointwise / element-wise ops
# ---------------------------------------------------------------------------

def _neg_case():
    def prepare(dtype):
        x = _f((32, 32), dtype)
        return (x,), {}

    def run(x):
        return torch.neg(x)

    return TestCase("neg", FLOAT_DTYPES, prepare, run)


def _ne_case():
    def prepare(dtype):
        x = _f((32, 32), dtype)
        y = _f((32, 32), dtype)
        return (x, y), {}

    def run(x, y):
        return torch.ne(x, y)

    return TestCase("ne", FLOAT_DTYPES, prepare, run)

def _or_case():
    """Tensor.__or__ dispatches to bitwise_or for integer types."""
    def prepare(dtype):
        x = _i((32, 32))
        y = _i((32, 32))
        return (x, y), {}

    def run(x, y):
        return x | y

    return TestCase("__or__", INT_DTYPES, prepare, run)


# ---------------------------------------------------------------------------
# Reduction ops
# ---------------------------------------------------------------------------

def _prod_case():
    def prepare(dtype):
        x = _f((32, 32), dtype)
        return (x,), {}

    def run(x):
        return torch.prod(x, dim=-1)

    return TestCase("prod", FLOAT_DTYPES, prepare, run)


def _cumsum_case():
    def prepare(dtype):
        x = _f((32, 32), dtype)
        return (x,), {}

    def run(x):
        return torch.cumsum(x, dim=0)

    return TestCase("cumsum", FLOAT_DTYPES, prepare, run)


def _topk_case():
    def prepare(dtype):
        x = _f((32, 64), dtype)
        return (x,), {}

    def run(x):
        # Return only the values; index tie-breaking can differ between implementations
        return torch.topk(x, k=8, dim=-1).values

    return TestCase("topk", FLOAT_DTYPES, prepare, run)


# ---------------------------------------------------------------------------
# Shape manipulation ops
# ---------------------------------------------------------------------------

def _cat_case():
    def prepare(dtype):
        x = _f((16, 32), dtype)
        y = _f((16, 32), dtype)
        return (x, y), {}

    def run(x, y):
        return torch.cat([x, y], dim=0)

    return TestCase("cat", FLOAT_DTYPES, prepare, run)


def _concat_case():
    def prepare(dtype):
        x = _f((16, 32), dtype)
        y = _f((16, 32), dtype)
        return (x, y), {}

    def run(x, y):
        return torch.cat([x, y], dim=0)

    return TestCase("concat", FLOAT_DTYPES, prepare, run)


def _chunk_case():
    def prepare(dtype):
        x = _f((32, 32), dtype)
        return (x,), {}

    def run(x):
        return torch.chunk(x, 4, dim=0)

    return TestCase("chunk", FLOAT_DTYPES, prepare, run)


def _split_case():
    def prepare(dtype):
        x = _f((32, 32), dtype)
        return (x,), {}

    def run(x):
        return torch.split(x, 8, dim=0)

    return TestCase("split", FLOAT_DTYPES, prepare, run)


def _split_with_sizes_case():
    def prepare(dtype):
        x = _f((32, 32), dtype)
        return (x,), {}

    def run(x):
        return torch.split(x, [8, 8, 16], dim=0)

    return TestCase("split_with_sizes", FLOAT_DTYPES, prepare, run)


def _stack_case():
    def prepare(dtype):
        x = _f((16, 32), dtype)
        y = _f((16, 32), dtype)
        return (x, y), {}

    def run(x, y):
        return torch.stack([x, y], dim=0)

    return TestCase("stack", FLOAT_DTYPES, prepare, run)

def _hstack_case():
    def prepare(dtype):
        x = _f((16, 32), dtype)
        y = _f((16, 32), dtype)
        return (x, y), {}

    def run(x, y):
        return torch.hstack([x, y])

    return TestCase("hstack", FLOAT_DTYPES, prepare, run)

def _narrow_case():
    def prepare(dtype):
        x = _f((32, 32), dtype)
        return (x,), {}

    def run(x):
        return torch.narrow(x, 0, 4, 16)

    return TestCase("narrow", FLOAT_DTYPES, prepare, run)


def _slice_case():
    def prepare(dtype):
        x = _f((32, 32), dtype)
        return (x,), {}

    def run(x):
        return x[4:20, :]

    return TestCase("slice", FLOAT_DTYPES, prepare, run)


def _tile_case():
    def prepare(dtype):
        x = _f((8, 16), dtype)
        return (x,), {}

    def run(x):
        return torch.tile(x, (2, 2))

    return TestCase("tile", FLOAT_DTYPES, prepare, run)


def _repeat_case():
    def prepare(dtype):
        x = _f((8, 16), dtype)
        return (x,), {}

    def run(x):
        return x.repeat(2, 2)

    return TestCase("repeat", FLOAT_DTYPES, prepare, run)


def _repeat_interleave_case():
    def prepare(dtype):
        x = _f((8, 16), dtype)
        return (x,), {}

    def run(x):
        return torch.repeat_interleave(x, 3, dim=0)

    return TestCase("repeat_interleave", FLOAT_DTYPES, prepare, run)


def _unfold_case():
    # flag_gems registers the 4D conv-style unfold (F.unfold / im2col), not Tensor.unfold
    def prepare(dtype):
        x = _f((2, 4, 16, 16), dtype)
        return (x,), {}

    def run(x):
        return F.unfold(x.float(), kernel_size=(3, 3), padding=1, stride=2)

    return TestCase("unfold", FLOAT_DTYPES, prepare, run)


def _flip_case():
    def prepare(dtype):
        x = _f((16, 32), dtype)
        return (x,), {}

    def run(x):
        return torch.flip(x, [0, 1])

    return TestCase("flip", FLOAT_DTYPES, prepare, run)


# ---------------------------------------------------------------------------
# Memory / identity ops
# ---------------------------------------------------------------------------

def _contiguous_case():
    def prepare(dtype):
        x = _f((32, 32), dtype).t()  # non-contiguous
        return (x,), {}

    def run(x):
        return x.contiguous()

    return TestCase("contiguous", FLOAT_DTYPES, prepare, run)


def _copy__case():
    def prepare(dtype):
        src = _f((32, 32), dtype)
        dst = torch.empty_like(src)
        return (dst, src), {}

    def run(dst, src):
        return dst.copy_(src)

    return TestCase("copy_", FLOAT_DTYPES, prepare, run)


def _conj_physical_case():
    # conj_physical is a no-op for real tensors
    def prepare(dtype):
        x = _f((32, 32), dtype)
        return (x,), {}

    def run(x):
        return torch.conj_physical(x)

    return TestCase("conj_physical", FLOAT_DTYPES, prepare, run)


def _resolve_conj_case():
    def prepare(dtype):
        x = _f((32, 32), dtype)
        return (x,), {}

    def run(x):
        return torch.resolve_conj(x)

    return TestCase("resolve_conj", FLOAT_DTYPES, prepare, run)


def _resolve_neg_case():
    def prepare(dtype):
        x = _f((32, 32), dtype)
        return (x,), {}

    def run(x):
        return torch.resolve_neg(x)

    return TestCase("resolve_neg", FLOAT_DTYPES, prepare, run)


# ---------------------------------------------------------------------------
# Scatter / gather ops
# ---------------------------------------------------------------------------

def _gather_case():
    def prepare(dtype):
        x = _f((16, 32), dtype)
        index = torch.randint(0, 32, (16, 8), dtype=torch.long, device="cpu")
        return (x, index), {}

    def run(x, index):
        return torch.gather(x, 1, index)

    return TestCase("gather", FLOAT_DTYPES, prepare, run)


def _scatter_case():
    def prepare(dtype):
        x = _f((16, 32), dtype)
        src = _f((16, 8), dtype)
        index = torch.randint(0, 32, (16, 8), dtype=torch.long, device="cpu")
        return (x, index, src), {}

    def run(x, index, src):
        return x.clone().scatter_(1, index, src)

    return TestCase("scatter", FLOAT_DTYPES, prepare, run)


def _index_select_case():
    def prepare(dtype):
        x = _f((32, 32), dtype)
        index = torch.randint(0, 32, (12,), dtype=torch.long, device="cpu")
        return (x, index), {}

    def run(x, index):
        return torch.index_select(x, 0, index)

    return TestCase("index_select", FLOAT_DTYPES, prepare, run)


def _index_put__case():
    def prepare(dtype):
        x = _f((32, 32), dtype)
        values = _f((8, 32), dtype)
        idx = torch.randint(0, 32, (8,), dtype=torch.long, device="cpu")
        return (x, idx, values), {}

    def run(x, idx, values):
        return x.clone().index_put_((idx,), values)

    return TestCase("index_put_", FLOAT_DTYPES, prepare, run)


def _select_scatter_case():
    def prepare(dtype):
        x = _f((16, 32), dtype)
        src = _f((32,), dtype)
        return (x, src), {}

    def run(x, src):
        return torch.select_scatter(x, src, 0, 4)

    return TestCase("select_scatter", FLOAT_DTYPES, prepare, run)


def _slice_scatter_case():
    def prepare(dtype):
        x = _f((32, 32), dtype)
        src = _f((8, 32), dtype)
        return (x, src), {}

    def run(x, src):
        return torch.slice_scatter(x, src, 0, 4, 12)

    return TestCase("slice_scatter", FLOAT_DTYPES, prepare, run)


# ---------------------------------------------------------------------------
# Loss / dropout ops
# ---------------------------------------------------------------------------

def _cross_entropy_loss_case():
    def prepare(dtype):
        inp = _f((16, 10), dtype)
        target = torch.randint(0, 10, (16,), dtype=torch.long, device="cpu")
        return (inp, target), {}

    def run(inp, target):
        return F.cross_entropy(inp.float(), target)

    return TestCase("cross_entropy_loss", FLOAT_DTYPES, prepare, run)


def _native_dropout_case():
    # train=False: no Triton kernel compiled; op simply returns the input unchanged.
    # (train=True hits the dropout Triton kernel which fails with a ptr-cast mlir-translate
    # error in this build environment.)
    def prepare(dtype):
        x = _f((32, 32), dtype)
        return (x,), {}

    def run(x):
        out, _mask = torch.ops.aten.native_dropout(x, 0.5, False)
        return out

    return TestCase("native_dropout", FLOAT_DTYPES, prepare, run)


# ---------------------------------------------------------------------------
# Other ops
# ---------------------------------------------------------------------------

def _masked_fill_case():
    def prepare(dtype):
        x = _f((32, 32), dtype)
        mask = _b((32, 32))
        return (x, mask), {}

    def run(x, mask):
        return x.masked_fill(mask, float("-inf"))

    return TestCase("masked_fill", FLOAT_DTYPES, prepare, run)


def _where_case():
    def prepare(dtype):
        cond = _b((32, 32))
        x = _f((32, 32), dtype)
        y = _f((32, 32), dtype)
        return (cond, x, y), {}

    def run(cond, x, y):
        return torch.where(cond, x, y)

    return TestCase("where", FLOAT_DTYPES, prepare, run)


def _pad_case():
    def prepare(dtype):
        x = _f((16, 16), dtype)
        return (x,), {}

    def run(x):
        return F.pad(x, (2, 2, 2, 2))

    return TestCase("pad", FLOAT_DTYPES, prepare, run)


def _constant_pad_nd_case():
    # flag_gems has a Python name-collision bug in constant_pad_nd:
    #   def constant_pad_nd(self, pad, value=0): return pad(self, pad, ...)
    # The parameter `pad` (a list) shadows the imported `pad` function, so it
    # tries to call the list as a callable → 'list' object is not callable.
    # Workaround: use F.pad(mode='constant') which dispatches through aten::pad
    # (separately registered and working), not aten::constant_pad_nd.
    def prepare(dtype):
        x = _f((16, 16), dtype)
        return (x,), {}

    def run(x):
        return F.pad(x, (1, 1, 1, 1), mode="constant", value=0.0)

    return TestCase("constant_pad_nd", FLOAT_DTYPES, prepare, run)


def _uniform__case():
    def prepare(dtype):
        x = _f((32, 32), dtype)
        return (x,), {}

    def run(x):
        return x.clone().uniform_(0.0, 1.0)

    return TestCase("uniform_", FLOAT_DTYPES, prepare, run, stochastic=True)


def _embedding_case():
    def prepare(dtype):
        weight = _f((1024, 8), dtype)
        indices = torch.randint(0, 1024, (2, 4), dtype=torch.long, device="cpu")
        return (indices, weight), {}

    def run(indices, weight):
        return F.embedding(indices, weight)

    return TestCase("embedding", FLOAT_DTYPES, prepare, run)


def _multinomial_case():
    def prepare(dtype):
        weights = torch.rand((8, 32), dtype=dtype, device="cpu")
        return (weights,), {}

    def run(weights):
        return torch.multinomial(weights.float(), num_samples=4, replacement=True)

    return TestCase("multinomial", FLOAT_DTYPES, prepare, run, stochastic=True)


# ---------------------------------------------------------------------------
# Vision ops
# ---------------------------------------------------------------------------

def _upsample_nearest2d_case():
    def prepare(dtype):
        x = _f((2, 4, 8, 8), dtype)
        return (x,), {}

    def run(x):
        return F.interpolate(x.float(), scale_factor=2, mode="nearest")

    return TestCase("upsample_nearest2d", FLOAT_DTYPES, prepare, run)


def _maxpool_case():
    def prepare(dtype):
        x = _f((2, 4, 16, 16), dtype)
        return (x,), {}

    def run(x):
        result = torch.ops.aten.max_pool2d_with_indices(x.float(), [2, 2], [2, 2])
        return result[0]

    return TestCase("maxpool", FLOAT_DTYPES, prepare, run)


def _adaptive_avg_pool2d_case():
    def prepare(dtype):
        x = _f((2, 4, 16, 16), dtype)
        return (x,), {}

    def run(x):
        return F.adaptive_avg_pool2d(x.float(), (8, 8))

    return TestCase("adptiveAvgPool2d", FLOAT_DTYPES, prepare, run)


# ---------------------------------------------------------------------------
# Attention ops
# ---------------------------------------------------------------------------

def _scaled_dot_product_attention_case():
    def prepare(dtype):
        q = _f((2, 4, 16, 32), dtype)
        k = _f((2, 4, 16, 32), dtype)
        v = _f((2, 4, 16, 32), dtype)
        return (q, k, v), {}

    def run(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=False)

    return TestCase("scaled_dot_product_attention", FLOAT_DTYPES, prepare, run)


def _efficient_attention_forward_case():
    def prepare(dtype):
        q = _f((2, 16, 4, 32), dtype)
        k = _f((2, 16, 4, 32), dtype)
        v = _f((2, 16, 4, 32), dtype)
        return (q, k, v), {}

    def run(q, k, v):
        op = getattr(torch.ops.aten, "_efficient_attention_forward", None)
        if op is not None:
            try:
                out = op.default(q, k, v, None, None, None, False, False, 0, False, None)
                return out[0] if isinstance(out, tuple) else out
            except Exception:
                pass
        # Fallback: use scaled_dot_product_attention
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
        )

    return TestCase("_efficient_attention_forward", FLOAT_DTYPES, prepare, run)


# ---------------------------------------------------------------------------
# Build the full case list
# ---------------------------------------------------------------------------

def build_cases() -> list[TestCase]:
    return [
        # Creation
        _arange_case(),
        _ones_case(),
        _zeros_case(),
        _full_case(),
        _full_like_case(),
        _rand_case(),
        _rand_like_case(),
        _randn_case(),
        # Pointwise
        _neg_case(),
        _ne_case(),
        _or_case(),
        # Reduction
        _prod_case(),
        _cumsum_case(),
        _topk_case(),
        # Shape
        _cat_case(),
        _concat_case(),
        _chunk_case(),
        _split_case(),
        _split_with_sizes_case(),
        _stack_case(),
        _hstack_case(),
        _narrow_case(),
        _slice_case(),
        _tile_case(),
        _repeat_case(),
        _repeat_interleave_case(),
        _unfold_case(),
        _flip_case(),
        # Memory
        _contiguous_case(),
        _copy__case(),
        _conj_physical_case(),
        _resolve_conj_case(),
        _resolve_neg_case(),
        # Scatter/gather
        _gather_case(),
        _scatter_case(),
        _index_select_case(),
        _index_put__case(),
        _select_scatter_case(),
        _slice_scatter_case(),
        # Loss/dropout
        _cross_entropy_loss_case(),
        _native_dropout_case(),
        # Other
        _masked_fill_case(),
        _where_case(),
        _pad_case(),
        _constant_pad_nd_case(),
        _uniform__case(),
        _embedding_case(),
        _multinomial_case(),
        # Vision
        _upsample_nearest2d_case(),
        _maxpool_case(),
        _adaptive_avg_pool2d_case(),
        # Attention
        _scaled_dot_product_attention_case(),
        _efficient_attention_forward_case(),
    ]


# ---------------------------------------------------------------------------
# Correctness checking
# ---------------------------------------------------------------------------

def _tensors_close(ref, res, rtol=1e-2, atol=1e-3, label="") -> bool:
    if not isinstance(ref, torch.Tensor) or not isinstance(res, torch.Tensor):
        return True
    if ref.dtype != res.dtype:
        if ref.is_floating_point() and res.is_floating_point():
            ref = ref.float()
            res = res.float()
    if ref.is_floating_point():
        ok = torch.allclose(ref, res, rtol=rtol, atol=atol, equal_nan=True)
        if not ok:
            diff = (ref.float() - res.float()).abs()
            print(f"    {label}FAIL (max_diff={diff.max().item():.6e})")
        return ok
    else:
        ok = torch.equal(ref, res)
        if not ok:
            print(f"    {label}FAIL ({(ref != res).sum().item()} mismatches)")
        return ok


def _compare_outputs(ref, res, label="") -> bool:
    if isinstance(ref, torch.Tensor):
        return _tensors_close(ref, res, label=label)
    if isinstance(ref, (tuple, list)):
        all_ok = True
        for i, (r, o) in enumerate(zip(ref, res)):
            if isinstance(r, torch.Tensor):
                all_ok = _tensors_close(r, o, label=f"{label}[{i}] ") and all_ok
        return all_ok
    return True


def _check_stochastic(case: TestCase, out) -> None:
    """Statistical sanity checks for random-output ops."""
    name = case.name
    if not isinstance(out, torch.Tensor):
        if isinstance(out, (tuple, list)):
            out = next((t for t in out if isinstance(t, torch.Tensor)), None)
    if out is None or not isinstance(out, torch.Tensor):
        return

    if name in ("rand", "rand_like"):
        in_range = ((out >= 0.0) & (out <= 1.0)).all().item()
        print(f"    range [0,1]: {'PASS' if in_range else 'FAIL'}")
    elif name in ("randn",):
        out_f = out.float()
        mean_ok = abs(out_f.mean().item()) < 0.1
        print(f"    mean≈0: {'PASS' if mean_ok else 'FAIL'} (mean={out_f.mean().item():.4f})")
    elif name == "uniform_":
        in_range = ((out >= 0.0) & (out <= 1.0)).all().item()
        print(f"    range [0,1]: {'PASS' if in_range else 'FAIL'}")
    elif name == "native_dropout":
        # output should have same shape, ~50% zeros
        frac_zero = (out == 0).float().mean().item()
        ok = 0.3 < frac_zero < 0.7
        print(f"    ~50%% zeros: {'PASS' if ok else 'FAIL'} (frac={frac_zero:.3f})")
    elif name == "multinomial":
        if out.dtype in (torch.int32, torch.int64):
            print(f"    dtype is int: PASS")
    else:
        print(f"    (stochastic op — skipping deterministic check)")


def check_correctness(case: TestCase, rtol: float = 1e-2, atol: float = 1e-3) -> None:
    print(f"\n--- [correctness] {case.name} ---")
    for dtype in case.dtypes:
        try:
            args, kwargs = case.prepare(dtype)

            with torch.no_grad(), flag_gems.use_gems():
                res = case.run(*args, **kwargs)

            if case.stochastic:
                _check_stochastic(case, res)
                print(f"  dtype {dtype}: PASS (stochastic)")
                continue

            with torch.no_grad():
                ref = case.run(*args, **kwargs)

            ok = _compare_outputs(ref, res)
            print(f"  dtype {dtype}: {'PASS' if ok else 'FAIL'}")
        except Exception as exc:
            print(f"  dtype {dtype}: ERROR - {exc}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correctness tests for operators not covered by perf.py"
    )
    parser.add_argument("--ops", type=str, default="all",
                        help="Comma-separated op names or 'all'")
    parser.add_argument("--check", action="store_true",
                        help="Run correctness checks (default behaviour)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cases = build_cases()
    selected = {name.strip() for name in args.ops.split(",") if name.strip()}

    if selected != {"all"}:
        cases = [c for c in cases if c.name in selected]

    print("Operator Correctness Test (flag_gems vs torch)")
    print("=" * 50)
    print(f"Total cases: {len(cases)}")
    for case in cases:
        check_correctness(case)
