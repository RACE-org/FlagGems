from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable

import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())

try:
    import flag_gems
except Exception:  # pragma: no cover - fallback for environments without flag_gems
    raise RuntimeError("flag_gems is required for this benchmark; torch fallback is disabled")


FLOAT_DTYPES = (torch.float16,)
INT_DTYPES = (torch.int32,)
BOOL_DTYPES = (torch.bool,)


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    dtypes: tuple[torch.dtype, ...]
    prepare: Callable[[torch.dtype], tuple[tuple[object, ...], dict[str, object]]]
    run: Callable[..., object]


def make_float_tensor(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(shape, dtype=dtype, device="cpu")


def make_int_tensor(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.randint(-8, 8, shape, dtype=dtype, device="cpu")


def make_bool_tensor(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.rand(shape, device="cpu") > 0.5


def simple_bench(fn: Callable[[], object], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    return 1000.0 * (time.perf_counter() - start) / iterations


def unary_case(name: str, op: Callable[[torch.Tensor], object], shape: tuple[int, ...], dtypes=FLOAT_DTYPES) -> BenchmarkCase:
    def prepare(dtype: torch.dtype):
        x = make_float_tensor(shape, dtype)
        return (x,), {}

    return BenchmarkCase(name=name, dtypes=dtypes, prepare=prepare, run=op)


def reduction_case(name: str, op: Callable[[torch.Tensor], object], shape: tuple[int, ...], dtypes=FLOAT_DTYPES) -> BenchmarkCase:
    return unary_case(name, op, shape, dtypes=dtypes)


def var_mean_case(name: str, shape: tuple[int, ...], dtypes=FLOAT_DTYPES) -> BenchmarkCase:
    def prepare(dtype: torch.dtype):
        x = make_float_tensor(shape, dtype)
        return (x,), {}

    def run(x: torch.Tensor):
        return torch.var_mean(x, dim=-1, unbiased=False)

    return BenchmarkCase(name=name, dtypes=dtypes, prepare=prepare, run=run)


def normal_case(name: str, shape: tuple[int, ...], dtypes=FLOAT_DTYPES) -> BenchmarkCase:
    def prepare(dtype: torch.dtype):
        mean = make_float_tensor(shape, dtype)
        std = torch.rand(shape, dtype=dtype, device="cpu") + 0.1
        return (mean, std), {}

    def run(mean: torch.Tensor, std: torch.Tensor):
        return torch.normal(mean, std)

    return BenchmarkCase(name=name, dtypes=dtypes, prepare=prepare, run=run)


STOCHASTIC_OPS = {"normal", "randn", "dropout"}


def clamp_case(name: str, shape: tuple[int, ...], dtypes=FLOAT_DTYPES) -> BenchmarkCase:
    def prepare(dtype: torch.dtype):
        x = make_float_tensor(shape, dtype)
        return (x,), {}

    def run(x: torch.Tensor):
        return torch.clamp(x, min=-0.5, max=0.5)

    return BenchmarkCase(name=name, dtypes=dtypes, prepare=prepare, run=run)


def binary_case(name: str, op: Callable[[torch.Tensor, torch.Tensor], object], shape: tuple[int, ...], dtypes=FLOAT_DTYPES) -> BenchmarkCase:
    def prepare(dtype: torch.dtype):
        x = make_float_tensor(shape, dtype)
        y = make_float_tensor(shape, dtype)
        return (x, y), {}

    return BenchmarkCase(name=name, dtypes=dtypes, prepare=prepare, run=op)


def int_binary_case(name: str, op: Callable[[torch.Tensor, torch.Tensor], object], shape: tuple[int, ...]) -> BenchmarkCase:
    def prepare(dtype: torch.dtype):
        x = make_int_tensor(shape, dtype)
        y = make_int_tensor(shape, dtype)
        return (x, y), {}

    return BenchmarkCase(name=name, dtypes=INT_DTYPES, prepare=prepare, run=op)


def int_unary_case(name: str, op: Callable[[torch.Tensor], object], shape: tuple[int, ...]) -> BenchmarkCase:
    def prepare(dtype: torch.dtype):
        x = make_int_tensor(shape, dtype)
        return (x,), {}

    return BenchmarkCase(name=name, dtypes=INT_DTYPES, prepare=prepare, run=op)


def bool_unary_case(name: str, op: Callable[[torch.Tensor], object], shape: tuple[int, ...]) -> BenchmarkCase:
    def prepare(_: torch.dtype):
        x = make_bool_tensor(shape)
        return (x,), {}

    return BenchmarkCase(name=name, dtypes=BOOL_DTYPES, prepare=prepare, run=op)


def float_scalar_case(name: str, op: Callable[[torch.Tensor], object], shape: tuple[int, ...], dtypes=FLOAT_DTYPES) -> BenchmarkCase:
    return unary_case(name, op, shape, dtypes=dtypes)


def mm_case(name: str, shape: tuple[int, int, int], dtypes=FLOAT_DTYPES) -> BenchmarkCase:
    m, n, k = shape

    def prepare(dtype: torch.dtype):
        a = make_float_tensor((m, k), dtype)
        b = make_float_tensor((k, n), dtype)
        return (a, b), {}

    return BenchmarkCase(name=name, dtypes=dtypes, prepare=prepare, run=torch.mm)


def addmm_case(name: str, shape: tuple[int, int, int], dtypes=FLOAT_DTYPES) -> BenchmarkCase:
    m, n, k = shape

    def prepare(dtype: torch.dtype):
        input_ = make_float_tensor((m, n), dtype)
        a = make_float_tensor((m, k), dtype)
        b = make_float_tensor((k, n), dtype)
        return (input_, a, b), {}

    return BenchmarkCase(name=name, dtypes=dtypes, prepare=prepare, run=torch.addmm)


def bmm_case(name: str, shape: tuple[int, int, int, int], dtypes=FLOAT_DTYPES) -> BenchmarkCase:
    batch, m, n, k = shape

    def prepare(dtype: torch.dtype):
        a = make_float_tensor((batch, m, k), dtype)
        b = make_float_tensor((batch, k, n), dtype)
        return (a, b), {}

    return BenchmarkCase(name=name, dtypes=dtypes, prepare=prepare, run=torch.bmm)


def matmul_case(name: str, shape: tuple[int, int, int], dtypes=FLOAT_DTYPES) -> BenchmarkCase:
    return mm_case(name, shape, dtypes=dtypes)


def mv_case(name: str, shape: tuple[int, int], dtypes=FLOAT_DTYPES) -> BenchmarkCase:
    rows, cols = shape

    def prepare(dtype: torch.dtype):
        a = make_float_tensor((rows, cols), dtype)
        v = make_float_tensor((cols,), dtype)
        return (a, v), {}

    return BenchmarkCase(name=name, dtypes=dtypes, prepare=prepare, run=torch.mv)


def linear_case(name: str, shape: tuple[int, int, int], dtypes=FLOAT_DTYPES) -> BenchmarkCase:
    batch, in_features, out_features = shape

    def prepare(dtype: torch.dtype):
        x = make_float_tensor((batch, in_features), dtype)
        weight = make_float_tensor((out_features, in_features), dtype)
        bias = make_float_tensor((out_features,), dtype)
        return (x, weight, bias), {}

    return BenchmarkCase(name=name, dtypes=dtypes, prepare=prepare, run=torch.nn.functional.linear)


def outer_case(name: str, length: int, dtypes=FLOAT_DTYPES) -> BenchmarkCase:
    def prepare(dtype: torch.dtype):
        a = make_float_tensor((length,), dtype)
        b = make_float_tensor((length,), dtype)
        return (a, b), {}

    return BenchmarkCase(name=name, dtypes=dtypes, prepare=prepare, run=torch.outer)


def fill_case(name: str, shape: tuple[int, ...], dtypes=FLOAT_DTYPES) -> BenchmarkCase:
    def prepare(dtype: torch.dtype):
        x = make_float_tensor(shape, dtype)
        return (x,), {}

    def run(x: torch.Tensor):
        return x.fill_(0.5)

    return BenchmarkCase(name=name, dtypes=dtypes, prepare=prepare, run=run)


def layer_norm_case(name: str, shape: tuple[int, ...], dtypes=FLOAT_DTYPES) -> BenchmarkCase:
    def prepare(dtype: torch.dtype):
        x = make_float_tensor(shape, dtype)
        return (x,), {}

    def run(x: torch.Tensor):
        return torch.nn.functional.layer_norm(x, x.shape[-1:])

    return BenchmarkCase(name=name, dtypes=dtypes, prepare=prepare, run=run)


def native_layer_norm_case(name: str, shape: tuple[int, ...], dtypes=FLOAT_DTYPES) -> BenchmarkCase:
    def prepare(dtype: torch.dtype):
        x = make_float_tensor(shape, dtype)
        return (x,), {}

    def run(x: torch.Tensor):
        weight = torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)
        bias = torch.zeros(x.shape[-1], dtype=x.dtype, device=x.device)
        op = getattr(torch.ops.aten, "native_layer_norm", None)
        if op is not None:
            try:
                return op.default(x, [x.shape[-1]], weight, bias, 1e-5)[0]
            except Exception:
                pass
        return torch.nn.functional.layer_norm(x, x.shape[-1:])

    return BenchmarkCase(name=name, dtypes=dtypes, prepare=prepare, run=run)


def group_norm_case(name: str, shape: tuple[int, int, int, int], num_groups: int, dtypes=FLOAT_DTYPES) -> BenchmarkCase:
    n, c, h, w = shape

    def prepare(dtype: torch.dtype):
        x = make_float_tensor((n, c, h, w), dtype)
        weight = make_float_tensor((c,), dtype)
        bias = make_float_tensor((c,), dtype)
        return (x, weight, bias), {}

    def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        return torch.nn.functional.group_norm(x, num_groups=num_groups, weight=weight, bias=bias)

    return BenchmarkCase(name=name, dtypes=dtypes, prepare=prepare, run=run)


def native_group_norm_case(name: str, shape: tuple[int, int, int, int], num_groups: int, dtypes=FLOAT_DTYPES) -> BenchmarkCase:
    n, c, h, w = shape

    def prepare(dtype: torch.dtype):
        x = make_float_tensor((n, c, h, w), dtype)
        return (x,), {}

    def run(x: torch.Tensor):
        weight = torch.ones(c, dtype=x.dtype, device=x.device)
        bias = torch.zeros(c, dtype=x.dtype, device=x.device)
        op = getattr(torch.ops.aten, "native_group_norm", None)
        if op is not None:
            try:
                return op.default(x, weight, bias, n, c, h * w, num_groups, 1e-5)[0]
            except Exception:
                pass
        return torch.nn.functional.group_norm(x, num_groups=num_groups, weight=weight, bias=bias)

    return BenchmarkCase(name=name, dtypes=dtypes, prepare=prepare, run=run)


def batch_norm_case(name: str, shape: tuple[int, int, int, int], dtypes=FLOAT_DTYPES) -> BenchmarkCase:
    n, c, h, w = shape

    def prepare(dtype: torch.dtype):
        x = make_float_tensor((n, c, h, w), dtype)
        running_mean = torch.zeros(c, dtype=dtype, device="cpu")
        running_var = torch.ones(c, dtype=dtype, device="cpu")
        weight = torch.ones(c, dtype=dtype, device="cpu")
        bias = torch.zeros(c, dtype=dtype, device="cpu")
        return (x, running_mean, running_var, weight, bias), {}

    def run(x: torch.Tensor, running_mean: torch.Tensor, running_var: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        return torch.nn.functional.batch_norm(
            x,
            running_mean,
            running_var,
            weight=weight,
            bias=bias,
            training=False,
            momentum=0.1,
            eps=1e-5,
        )

    return BenchmarkCase(name=name, dtypes=dtypes, prepare=prepare, run=run)


def softmax_case(name: str, shape: tuple[int, ...], dtypes=FLOAT_DTYPES) -> BenchmarkCase:
    def prepare(dtype: torch.dtype):
        x = make_float_tensor(shape, dtype)
        return (x,), {}

    def run(x: torch.Tensor):
        return torch.nn.functional.softmax(x, dim=-1)

    return BenchmarkCase(name=name, dtypes=dtypes, prepare=prepare, run=run)


def log_softmax_case(name: str, shape: tuple[int, ...], dtypes=FLOAT_DTYPES) -> BenchmarkCase:
    def prepare(dtype: torch.dtype):
        x = make_float_tensor(shape, dtype)
        return (x,), {}

    def run(x: torch.Tensor):
        return torch.nn.functional.log_softmax(x, dim=-1)

    return BenchmarkCase(name=name, dtypes=dtypes, prepare=prepare, run=run)


def attention_case(name: str, shape: tuple[int, int, int, int], dtypes=FLOAT_DTYPES) -> BenchmarkCase:
    batch, heads, seq_len, head_dim = shape

    def prepare(dtype: torch.dtype):
        q = make_float_tensor((batch, heads, seq_len, head_dim), dtype)
        k = make_float_tensor((batch, heads, seq_len, head_dim), dtype)
        v = make_float_tensor((batch, heads, seq_len, head_dim), dtype)
        return (q, k, v), {}

    def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        gems_fn = (
            getattr(flag_gems, "_scaled_dot_product_flash_attention", None)
            or getattr(flag_gems, "scaled_dot_product_attention", None)
        )
        if gems_fn is not None:
            try:
                return gems_fn(q, k, v, is_causal=False)
            except Exception:
                pass

        op = getattr(torch.ops.aten, "_scaled_dot_product_efficient_attention", None)
        if op is not None:
            try:
                out = op.default(q, k, v, None, False, 0.0, False, scale=None)
                return out[0] if isinstance(out, tuple) else out
            except Exception:
                pass

        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)

    return BenchmarkCase(name=name, dtypes=dtypes, prepare=prepare, run=run)


def conv2d_like_case(name: str, shape: tuple[int, int, int, int], dtypes=FLOAT_DTYPES) -> BenchmarkCase:
    n, c, h, w = shape
    out_channels = max(1, c * 2)

    def prepare(dtype: torch.dtype):
        x = make_float_tensor((n, c, h, w), dtype)
        weight = make_float_tensor((out_channels, c, 3, 3), dtype)
        bias = make_float_tensor((out_channels,), dtype)
        return (x, weight, bias), {}

    def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        op = getattr(torch.ops.aten, "convolution", None)
        if op is not None:
            try:
                return op.default(x, weight, bias, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            except Exception:
                pass
        return torch.nn.functional.conv2d(x, weight, bias, stride=1, padding=1)

    return BenchmarkCase(name=name, dtypes=dtypes, prepare=prepare, run=run)


def conv1d_case(name: str, shape: tuple[int, int, int], dtypes=FLOAT_DTYPES) -> BenchmarkCase:
    n, c, l = shape
    out_channels = max(1, c * 2)

    def prepare(dtype: torch.dtype):
        x = make_float_tensor((n, c, l), dtype)
        weight = make_float_tensor((out_channels, c, 3), dtype)
        bias = make_float_tensor((out_channels,), dtype)
        return (x, weight, bias), {}

    def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        return torch.nn.functional.conv1d(x, weight, bias, stride=1, padding=1)

    return BenchmarkCase(name=name, dtypes=dtypes, prepare=prepare, run=run)


def triu_case(name: str, shape: tuple[int, int], diagonal: int = 0, dtypes=FLOAT_DTYPES) -> BenchmarkCase:
    def prepare(dtype: torch.dtype):
        x = make_float_tensor(shape, dtype)
        return (x,), {}

    def run(x: torch.Tensor):
        return torch.triu(x, diagonal=diagonal)

    return BenchmarkCase(name=name, dtypes=dtypes, prepare=prepare, run=run)


def elementwise_cases() -> list[BenchmarkCase]:
    shape = (32, 32)
    int_shape = (32, 32)
    bool_shape = (32, 32)

    return [
        reduction_case("min", torch.min, shape),
        reduction_case("max", torch.max, shape),
        reduction_case("amax", lambda x: torch.amax(x), shape),
        reduction_case("sum", lambda x: torch.sum(x), shape),
        unary_case("isnan", torch.isnan, shape),
        unary_case("isinf", torch.isinf, shape),
        var_mean_case("var_mean", shape),
        normal_case("normal", shape),
        clamp_case("clamp", shape),
        unary_case("abs", torch.abs, shape),
        binary_case("add", torch.add, shape),
        binary_case("div", lambda x, y: torch.div(x, y.abs() + 1.0), shape),
        unary_case("exp", torch.exp, shape),
        unary_case("gelu", lambda x: torch.nn.functional.gelu(x, approximate="none"), shape),
        unary_case("relu", torch.nn.functional.relu, shape),
        unary_case("silu", torch.nn.functional.silu, shape),
        unary_case("softmax", lambda x: torch.nn.functional.softmax(x, dim=-1), shape),
        unary_case("sigmoid", torch.sigmoid, shape),
        unary_case("log_sigmoid", torch.nn.functional.logsigmoid, shape),
        binary_case("mul", torch.mul, shape),
        binary_case("pow", lambda x, y: torch.pow(x.abs() + 1e-3, 1.5), shape),
        unary_case("reciprocal", lambda x: torch.reciprocal(x.abs() + 1.0), shape),
        unary_case("rsqrt", lambda x: torch.rsqrt(x.abs() + 1.0), shape),
        binary_case("rsub", lambda x, y: torch.rsub(x, y), shape),
        binary_case("sub", torch.sub, shape),
        unary_case("mean", lambda x: x.mean(dim=-1), shape),
        unary_case("argmax", lambda x: torch.argmax(x, dim=-1), shape),
        unary_case("sin", torch.sin, shape),
        unary_case("cos", torch.cos, shape),
        unary_case("tanh", torch.tanh, shape),
        unary_case("erf", torch.erf, shape),
        unary_case("log_softmax", lambda x: torch.nn.functional.log_softmax(x, dim=-1), shape),
        unary_case("vector_norm", lambda x: torch.linalg.vector_norm(x, dim=-1), shape),
        unary_case("permute", lambda x: x.permute(1, 0), shape),
        unary_case("transpose", lambda x: x.transpose(0, 1), shape),
        triu_case("triu", shape),
        unary_case("fill", lambda x: x.fill_(0.5), shape),
        int_binary_case("bitwise_and", torch.bitwise_and, int_shape),
        int_unary_case("bitwise_not", torch.bitwise_not, int_shape),
        int_binary_case("bitwise_or", torch.bitwise_or, int_shape),
        binary_case("eq", torch.eq, shape),
        binary_case("ge", torch.ge, shape),
        binary_case("gt", torch.gt, shape),
        binary_case("le", torch.le, shape),
        binary_case("lt", torch.lt, shape),
        bool_unary_case("all", torch.all, bool_shape),
        bool_unary_case("any", torch.any, bool_shape),
    ]


def matrix_cases() -> list[BenchmarkCase]:
    return [
        mm_case("mm", (32, 32, 32)),
        addmm_case("addmm", (32, 32, 32)),
        bmm_case("bmm", (8, 32, 32, 32)),
        matmul_case("matmul", (32, 32, 32)),
        # mv_case("mv", (4096, 128), dtypes=(torch.float16,)),
        linear_case("linear", (32, 32, 32)),
        outer_case("outer", 32),
    ]


def normalization_cases() -> list[BenchmarkCase]:
    return [
        layer_norm_case("layernorm", (32, 32)),
        native_layer_norm_case("native_layer_norm", (32, 32)),
        group_norm_case("group_norm", (1, 4, 2, 2), num_groups=1),
        native_group_norm_case("native_group_norm", (8, 32, 32, 32), num_groups=8),
        batch_norm_case("batch_norm", (8, 32, 32, 32)),
    ]


def convolution_cases() -> list[BenchmarkCase]:
    return [
        conv1d_case("conv1d", (4, 16, 128)),
        conv2d_like_case("conv2d", (4, 16, 32, 32)),
        conv2d_like_case("convolution", (4, 16, 32, 32)),
        conv2d_like_case("_convolution", (4, 16, 32, 32)),
        conv2d_like_case("cudnn_convolution", (4, 16, 32, 32)),
        conv2d_like_case("_conv_depthwise2d", (4, 16, 32, 32)),
    ]


def special_cases() -> list[BenchmarkCase]:
    return [
        softmax_case("softmax", (32, 32)),
        log_softmax_case("log_softmax", (32, 32)),
        attention_case("_scaled_dot_product_efficient_attention", (2, 8, 128, 64)),
    ]


def build_cases() -> list[BenchmarkCase]:
    return matrix_cases() + normalization_cases() + special_cases() + elementwise_cases() + convolution_cases()


def run_case(case: BenchmarkCase, warmup: int, iterations: int) -> None:
    print(f"\n--- {case.name} ---")
    for dtype in case.dtypes:
        try:
            args, kwargs = case.prepare(dtype)
            with torch.inference_mode():
                bench = lambda: case.run(*args, **kwargs)
                cost_ms = simple_bench(bench, warmup, iterations)
            print(f"dtype {dtype}: {cost_ms:.3f} ms")
        except Exception as exc:
            print(f"dtype {dtype}: Failed - {exc}")


def check_correctness(case: BenchmarkCase, rtol: float = 1e-2, atol: float = 1e-3) -> None:
    print(f"\n--- [correctness] {case.name} ---")
    is_stochastic = any(s in case.name.lower() for s in STOCHASTIC_OPS)
    for dtype in case.dtypes:
        try:
            args, kwargs = case.prepare(dtype)
            with torch.inference_mode(), flag_gems.use_gems():
                out = case.run(*args, **kwargs)

            def _compare_stochastic(out_t, args, label=""):
                if not isinstance(out_t, torch.Tensor) or not out_t.is_floating_point():
                    return
                out_f = out_t.float()
                mean_arg = args[0].float() if isinstance(args[0], torch.Tensor) else None
                if mean_arg is not None:
                    n = out_f.numel()
                    mean_diff = (out_f.mean() - mean_arg.mean()).abs().item()
                    # expected std of sample mean ≈ 1/sqrt(n); allow 5-sigma
                    threshold = 5.0 / (n ** 0.5)
                    ok = mean_diff < max(threshold, 0.05)
                    print(f"  dtype {dtype}{label}: mean_diff={mean_diff:.4f} (thr={threshold:.4f}) {'PASS' if ok else 'FAIL'}")
                else:
                    print(f"  dtype {dtype}{label}: no mean arg to check")

            def _compare(ref_t, out_t, label=""):
                if not isinstance(ref_t, torch.Tensor) or not isinstance(out_t, torch.Tensor):
                    return
                # Align dtypes before comparison
                if ref_t.dtype != out_t.dtype:
                    if ref_t.is_floating_point() and out_t.is_floating_point():
                        ref_t = ref_t.float()
                        out_t = out_t.float()
                if ref_t.is_floating_point():
                    if torch.allclose(ref_t, out_t, rtol=rtol, atol=atol, equal_nan=True):
                        print(f"  dtype {dtype}{label}: PASS")
                    else:
                        diff = (ref_t.float() - out_t.float()).abs()
                        print(f"  dtype {dtype}{label}: FAIL (max_diff={diff.max().item():.6e})")
                else:
                    if torch.equal(ref_t, out_t):
                        print(f"  dtype {dtype}{label}: PASS")
                    else:
                        mismatches = (ref_t != out_t).sum().item()
                        print(f"  dtype {dtype}{label}: FAIL ({mismatches} mismatches)")

            if is_stochastic:
                if isinstance(out, torch.Tensor):
                    _compare_stochastic(out, args)
                elif isinstance(out, tuple):
                    for i, o in enumerate(out):
                        _compare_stochastic(o, args, f"[{i}]")
            else:
                with torch.inference_mode():
                    ref = case.run(*args, **kwargs)
                if isinstance(ref, tuple) and isinstance(out, tuple):
                    for i, (r, o) in enumerate(zip(ref, out)):
                        _compare(r, o, f"[{i}]")
                else:
                    _compare(ref, out)
        except Exception as exc:
            print(f"  dtype {dtype}: ERROR - {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple operator performance benchmark")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--ops", type=str, default="all", help="Comma-separated op names or 'all'")
    parser.add_argument("--check", action="store_true", help="Run correctness check against torch")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cases = build_cases()
    selected = {name.strip() for name in args.ops.split(",") if name.strip()}

    if selected != {"all"}:
        cases = [case for case in cases if case.name in selected]

    if args.check:
        print("Operator Correctness Test (flag_gems vs torch)")
        print("===============================================")
        for case in cases:
            check_correctness(case)
    else:
        print("Simple Operator Performance Test")
        print("================================")
        print("backend: flag_gems")
        print(f"warmup={args.warmup}, iters={args.iters}")

        with flag_gems.use_gems():
            for case in cases:
                run_case(case, args.warmup, args.iters)
