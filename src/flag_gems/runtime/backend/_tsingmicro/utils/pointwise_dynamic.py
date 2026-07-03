"""Tsingmicro-specific PointwiseDynamic codegen.

Inherits the base PointwiseDynamic codegen and adds broadcast-aware
block-pointer loads: for each input dim whose stride is 0 (i.e. broadcast),
we issue ``tl.make_block_ptr`` with ``block_shape[j] = 1`` instead of the
full ``tile_size[j]``, load a single element, then ``tl.broadcast_to`` it
back to the full tile shape.  That collapses per-tile DRAM accesses on the
broadcast dim from ``tile_size[j]`` to 1 (typically 65536× fewer loads for
a ``(M, N)/(M, 1)`` broadcast).  See ``div_broadcast_perf_analysis.md`` §6
for the design and measured speedups.

Wiring (4-class inheritance chain):

* ``KernelGenerator`` — overrides ``gen_signature`` and
  ``gen_body_one_tile_per_cta_with_bptr``, adds
  ``_gen_broadcast_aware_load_bptr``.
* ``WrapperGenerator`` — overrides ``gen_kernel_launch`` to compute the
  per-input broadcast-flag tuple and pass it as a ``tl.constexpr``.
* ``ModuleGenerator`` — overrides ``__init__`` to plug in the two above.
* ``PointwiseDynamicFunction`` — overrides ``instantiate`` to construct
  the tsingmicro ``ModuleGenerator``.

The on-disk cache filename matches the base codegen, so callers that
switch between base and tsingmicro codegen must clear
``~/.flaggems/code_cache/`` between runs to avoid loading a stale module.
"""

import importlib
import importlib.util
from itertools import product
from typing import List, Optional, Tuple

from triton.runtime.jit import JITFunction

from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic
from flag_gems.utils.codegen_config_utils import CodeGenConfig
from flag_gems.utils.pointwise_dynamic import (
    FunctionSchema,
    KernelGenerator as _BaseKernelGenerator,
    ModuleGenerator as _BaseModuleGenerator,
    PointwiseDynamicFunction as _BasePointwiseDynamicFunction,
    WrapperGenerator as _BaseWrapperGenerator,
    _cs,
    _tuple_content,
    _type_name,
)


class KernelGenerator(_BaseKernelGenerator):
    def gen_signature(self, code, with_block_pointer=False):
        code.writeline(f"def {self.name}(")
        with code.indent():
            input_tensor_index = 0
            non_tensor_index = 0
            output_tensor_index = 0

            schema = self.fx
            # signature: inputs ptrs & non tensor inputs
            for i in range(schema.num_inputs()):
                if schema.is_tensor(i):
                    code.writeline(
                        f"in{input_tensor_index}_ptr: tl.tensor, # of tl.pointer_type"
                    )
                    input_tensor_index += 1
                else:
                    if schema.input_type(i) is not None:
                        code.writeline(
                            f"val{non_tensor_index}: {_type_name(schema.input_type(i))},"
                        )
                    else:
                        code.writeline(f"val{non_tensor_index},")
                    non_tensor_index += 1

            # signature: output ptrs
            for i in range(schema.num_outputs()):
                code.writeline(
                    f"out{output_tensor_index}_ptr: tl.tensor, # of tl.pointer_type"
                )
                output_tensor_index += 1

            # signature: strides, for each tensor arguments
            ndim = self.ndim
            if ndim > 0:
                # strides for inputs
                for i in range(schema.num_input_tensors()):
                    stride_args = _cs(f"in{i}_stride{j}: int" for j in range(ndim))
                    code.writeline(f"{stride_args}, # strides for in{i}")
                    if with_block_pointer:
                        stride_order_args = _cs(
                            f"in{i}_stride_order{j}: tl.constexpr" for j in range(ndim)
                        )
                        code.writeline(f"{stride_order_args}, # stride order for in{i}")
                        # broadcast flags (constexpr so the kernel specializes on the
                        # broadcast pattern and prunes dead branches at compile time)
                        if ndim >= 2:
                            bcast_args = _cs(
                                f"in{i}_bcast{j}: tl.constexpr" for j in range(ndim)
                            )
                            code.writeline(f"{bcast_args}, # broadcast flags for in{i}")

                # strides for outputs
                for i in range(schema.num_output_tensors()):
                    stride_args = _cs(f"out{i}_stride{j}: int" for j in range(ndim))
                    code.writeline(f"{stride_args}, # strides for out{i}")
                    if with_block_pointer:
                        stride_order_args = _cs(
                            f"out{i}_stride_order{j}: tl.constexpr" for j in range(ndim)
                        )
                        code.writeline(
                            f"{stride_order_args}, # stride order for out{i}"
                        )

                # task space, used to reconstruct multi index
                task_space_args = _cs(f"s{i}: int" for i in range(ndim))
                code.writeline(f"{task_space_args}, # task_space")

                # number of tasks, used to compute mask
                code.writeline("num_tasks: int,")

            # tile size & tiles_per_cta, gsl style
            if ndim > 0:
                code.writeline("tiles_per_cta: int,")
                tile_sizes = _cs(f"tile_size{i}: tl.constexpr" for i in range(ndim))
                code.writeline(f"{tile_sizes},")
                code.writeline("one_tile_per_cta: tl.constexpr,")
        code.writeline("):")

    def _gen_broadcast_aware_load_bptr(self, code, input_idx, ndim):
        """Generate a block-pointer load for one input, with stride=0 broadcast handling.

        We rely on per-input ``in{i}_bcast{j}: tl.constexpr`` flags passed by the wrapper
        (1 if that dim has stride==0, 0 otherwise).  Because they are constexpr the
        ``if/else`` chain below is resolved at compile time, so each kernel
        specialization sees a single, statically-shaped block_ptr — no mismatched
        if-branch types.

        For an input that is broadcast along some dims, we load with block_shape=1 on
        those dims (one element, one DRAM access), then ``tl.broadcast_to`` to the
        full tile shape so the downstream compute sees the same shape across inputs.
        """
        shape = _tuple_content(tuple(f"s{j}" for j in range(ndim)))
        offsets = _tuple_content(tuple(f"offset{j}" for j in range(ndim)))
        order = _tuple_content(tuple(f"in{input_idx}_stride_order{j}" for j in range(ndim)))
        strides = _tuple_content(tuple(f"in{input_idx}_stride{j}" for j in range(ndim)))
        tile_sizes = _tuple_content(tuple(f"tile_size{j}" for j in range(ndim)))

        first = True
        for mask in product([False, True], repeat=ndim):
            if not any(mask):
                continue  # the "all-False" fallback comes last in the else: branch

            # constexpr-resolvable predicate: only constants, no chained logical ops.
            # Use ``&`` on constexpr ints instead of chained ``and`` (unsupported by Triton).
            pred_parts = [
                f"(in{input_idx}_bcast{j} == {1 if mask[j] else 0})"
                for j in range(ndim)
            ]
            pred = " & ".join(pred_parts)
            block = _tuple_content(
                tuple("1" if mask[j] else f"tile_size{j}" for j in range(ndim))
            )
            non_broadcast_dims = [j for j in range(ndim) if not mask[j]]
            bcheck = _tuple_content(
                tuple(f"in{input_idx}_stride_order{j}" for j in non_broadcast_dims)
            )

            keyword = "if" if first else "elif"
            code.writeline(f"{keyword} {pred}:")
            with code.indent():
                code.writeline(
                    f"_bptr = tl.make_block_ptr("
                    f"in{input_idx}_ptr, ({shape}), ({strides}), ({offsets}), "
                    f"({block}), order=({order}))"
                )
                if non_broadcast_dims:
                    code.writeline(
                        f"_v = tl.load(_bptr, boundary_check=({bcheck})).to(in{input_idx}_ptr.type.element_ty)"
                    )
                else:
                    code.writeline(
                        f"_v = tl.load(_bptr).to(in{input_idx}_ptr.type.element_ty)"
                    )
                code.writeline(
                    f"in{input_idx} = tl.broadcast_to(_v, ({tile_sizes}))"
                )
            first = False

        # No broadcast at all — original full-block load
        code.writeline("else:")
        with code.indent():
            code.writeline(
                f"_bptr = tl.make_block_ptr("
                f"in{input_idx}_ptr, ({shape}), ({strides}), ({offsets}), "
                f"({tile_sizes}), order=({order}))"
            )
            code.writeline(
                f"in{input_idx} = tl.load(_bptr, boundary_check=({order})).to(in{input_idx}_ptr.type.element_ty)"
            )

    # nd tile 1d grid kernel with block pointer
    def gen_body_one_tile_per_cta_with_bptr(self, code):
        ndim = self.ndim
        schema = self.fx

        # block pointer for each operand
        shape = _tuple_content(tuple(f"s{i}" for i in range(ndim)))
        offsets = _tuple_content(tuple(f"offset{i}" for i in range(ndim)))
        tile_sizes = _tuple_content(tuple(f"tile_size{i}" for i in range(ndim)))

        # reconstruct pid multi index
        code.writeline(
            "# pid multi index recontruction: we use c ordering, right axes changes fastest"
        )
        for i in reversed(range(ndim)):
            if i > 0:
                code.writeline(f"tile_id{i} = tile_id % num_tiles{i}")
                code.writeline(f"tile_id //= num_tiles{i}")
            else:
                code.writeline(f"tile_id{i} = tile_id")
        code.newline()

        # cta_offsets
        code.writeline("# tile offsets")
        for i in range(ndim):
            # Or else: AssertionError: Block pointers only support 32 bit
            # `offsets/block_shape`, add a `.to(tl.int32)` or use regular indexing
            # for 64 bit support
            code.writeline(f"offset{i} = (tile_id{i} * tile_size{i}).to(tl.int32)")

        # loads — broadcast-aware path only kicks in for ndim>=2 (a 1D input
        # never has a stride==0 dim in this codegen).
        code.writeline("# loads")
        if ndim >= 2:
            for i in range(schema.num_input_tensors()):
                self._gen_broadcast_aware_load_bptr(code, i, ndim)
        else:
            for i in range(schema.num_input_tensors()):
                strides = _tuple_content(tuple(f"in{i}_stride{j}" for j in range(ndim)))
                order = _tuple_content(
                    tuple(f"in{i}_stride_order{j}" for j in range(ndim))
                )
                code.writeline(
                    f"in{i}_bptr = tl.make_block_ptr("
                    f"in{i}_ptr, ({shape}), ({strides}), ({offsets}), ({tile_sizes}), order=({order}))"
                )
                code.writeline(
                    f"in{i} = tl.load(in{i}_bptr, boundary_check=({order})).to(in{i}_ptr.type.element_ty) "
                    "# workaround the bug on bool"
                )
        code.newline()

        # compute
        # TODO: sepearate this part
        inputs_to_scalar_fn = [self.input_name(i) for i in range(schema.num_inputs())]
        outputs_to_scalar_fn = [
            self.output_name(i) for i in range(schema.num_output_tensors())
        ]
        inputs_to_scalar_fn = _cs(inputs_to_scalar_fn)
        outputs_to_scalar_fn = _cs(outputs_to_scalar_fn)

        code.writeline("# compute")
        code.writeline(
            f"{outputs_to_scalar_fn} = {self.fn_name}({inputs_to_scalar_fn})"
        )
        code.newline()

        # stores
        code.writeline(
            "# stores, note that store to block pointer does not automatically cast the value to the pointer's dtype"
        )
        for i in range(schema.num_output_tensors()):
            strides = _tuple_content(tuple(f"out{i}_stride{j}" for j in range(ndim)))
            order = _tuple_content(
                tuple(f"out{i}_stride_order{j}" for j in range(ndim))
            )
            code.writeline(
                f"out{i}_bptr = tl.make_block_ptr("
                f"out{i}_ptr, ({shape}), ({strides}), ({offsets}), ({tile_sizes}), order=({order}))"
            )
            code.writeline(
                f"tl.store(out{i}_bptr, out{i}.to(out{i}_bptr.type.element_ty), boundary_check=({order}))"
            )


class WrapperGenerator(_BaseWrapperGenerator):
    def gen_kernel_launch(
        self,
        code: IndentedBuffer,
    ):
        schema = self.fx
        ndim = self.ndim

        with_block_pointer = self.config.prefer_block_pointer

        code.writeline("# kernel launch")
        for i in range(schema.num_input_tensors()):
            code.writeline(f"in{i}_strides = in{i}.stride()")
            if not with_block_pointer:
                continue
            if ndim >= 2:  # where ndim is 1, we don't need to compute stride order
                code.writeline(f"in{i}_stride_order = stride_order(in{i}_strides)")
                # broadcast flags: stride==0 indicates a broadcast dim
                code.writeline(
                    f"in{i}_bcast = tuple(1 if s == 0 else 0 for s in in{i}_strides)"
                )
            else:
                code.writeline(f"in{i}_stride_order = (0,)")
        for i in range(schema.num_output_tensors()):
            code.writeline(f"out{i}_strides = out{i}.stride()")
            if not with_block_pointer:
                continue
            if ndim >= 2:
                code.writeline(f"out{i}_stride_order = stride_order(out{i}_strides)")
            else:
                code.writeline(f"out{i}_stride_order = (0,)")

        code.writeline("with torch_device_fn.device(in0.device.index):")
        with code.indent():
            code.writeline(f"{self.jit_fn_name}[grid](")
            with code.indent():
                params = []
                # NOTE: WRAP
                for i in range(schema.num_inputs()):
                    if schema.is_tensor(i):
                        params.append(f"{self.input_name(i)}")
                    else:
                        params.append(self.input_name(i))
                for i in range(schema.num_output_tensors()):
                    params.append(f"{self.output_name(i)}")

                code.writeline(f"{_cs(params)},")

                if ndim > 0:
                    for i in range(schema.num_input_tensors()):
                        s = ", ".join(f"in{i}_strides[{j}]" for j in range(ndim))
                        code.writeline(f"{s}, # stride for in{i}")
                        if not with_block_pointer:
                            continue
                        order = ", ".join(
                            f"in{i}_stride_order[{j}]" for j in range(ndim)
                        )
                        code.writeline(f"{order}, # stride order for in{i}")
                        if ndim >= 2:
                            bcast = ", ".join(
                                f"in{i}_bcast[{j}]" for j in range(ndim)
                            )
                            code.writeline(f"{bcast}, # broadcast flags for in{i}")

                    for i in range(schema.num_output_tensors()):
                        s = ", ".join(f"out{i}_strides[{j}]" for j in range(ndim))
                        code.writeline(f"{s}, # stride for out{i}")
                        if not with_block_pointer:
                            continue
                        order = ", ".join(
                            f"out{i}_stride_order[{j}]" for j in range(ndim)
                        )
                        code.writeline(f"{order}, # stride orderfor out{i}")

                    shape_args: str = ", ".join(f"shape[{i}]" for i in range(ndim))
                    code.writeline(f"{shape_args}, # task indexing space")
                    code.writeline("num_tasks, # num tasks")
                    code.writeline("tiles_per_cta=tiles_per_cta, # tiles_per_cta")
                    for i in range(ndim):
                        code.writeline(f"tile_size{i}=tile_sizes[{i}],")
                    code.writeline("one_tile_per_cta=one_tile_per_cta,")
                code.writeline("num_warps=num_warps,")
            code.writeline(")")


class ModuleGenerator(_BaseModuleGenerator):
    # Same as base, but wires in the tsingmicro KernelGenerator/WrapperGenerator
    # defined above.
    def __init__(
        self,
        function_schema: FunctionSchema,
        scalar_fn,
        ndim: int,
        jit_fn_name: str,
        wrapper_name: str,
        config: CodeGenConfig,
    ):
        self.config = config
        self.wrapper_gen = WrapperGenerator(
            function_schema, jit_fn_name, ndim, wrapper_name, config
        )
        self.kernel_gen = KernelGenerator(
            function_schema, scalar_fn, ndim, jit_fn_name, config
        )


class PointwiseDynamicFunction(_BasePointwiseDynamicFunction):
    # Same as base.instantiate, but constructs the tsingmicro ModuleGenerator.
    # NOTE: the cache filename matches the base codegen, so callers that switch
    # between base and tsingmicro codegen must clear ~/.flaggems/code_cache/
    # to avoid loading a stale module.
    def instantiate(self, ndim):
        if ndim in self.overloads:
            return self.overloads[ndim]

        code = IndentedBuffer()

        scalar_fn_name = self._scalar_fn.__name__
        kernel_name = f"{scalar_fn_name}_kernel_rank_{ndim}"
        wrapper_name = f"{scalar_fn_name}_wrapper_rank_{ndim}"
        module_gen = ModuleGenerator(
            self.fx,
            self._scalar_fn,
            ndim,
            kernel_name,
            wrapper_name,
            self.config,
        )
        module_gen.codegen(code)

        file_name = (
            f"pointwise_dynamic_{self._scalar_fn_cache_key}_{kernel_name}_"
            f"{'1d_tile_' if self.config.prefer_1d_tile else ''}"
            f"{'bptr' if (not self.config.prefer_1d_tile and self.config.prefer_block_pointer) else ''}"
            ".py"
        )

        file_path = code_cache_dir() / file_name
        write_atomic(file_path, code.getvalue())

        spec = importlib.util.spec_from_file_location(
            f"_gen_module_{self._scalar_fn_cache_key}_rank_{ndim}",
            file_path,
        )
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        m.__dict__.update(self._scalar_fn.__globals__)
        m.__dict__[self._scalar_fn.__name__] = self._scalar_fn

        overload = getattr(m, wrapper_name)
        self.overloads[ndim] = overload
        return overload


def pointwise_dynamic(
    f: Optional[JITFunction] = None,
    *,
    num_inputs: Optional[int] = None,
    is_tensor: Optional[List[bool]] = None,
    dtypes: Optional[List[Optional[type]]] = None,
    num_outputs: Optional[int] = None,
    promotion_methods: Optional[Tuple[int, ...]] = None,
    config: Optional[CodeGenConfig] = None,
):
    def decorator(fn):
        nonlocal num_inputs
        if (num_inputs is None) and (is_tensor is None) and (dtypes is None):
            num_inputs = len(fn.arg_names)
        op_desc = FunctionSchema(
            num_inputs=num_inputs,
            is_tensor=is_tensor,
            dtypes=dtypes,
            num_outputs=num_outputs,
            promotion_methods=promotion_methods,
        )
        return PointwiseDynamicFunction(op_desc, fn, config)

    if f is not None:
        return decorator(f)
    return decorator

