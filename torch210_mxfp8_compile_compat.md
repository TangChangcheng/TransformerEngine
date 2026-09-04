# PyTorch 2.10 MXFP8 `torch.compile` compatibility plan

## Decision

Transformer Engine (TE) needs a PyTorch 2.10 compatibility path for `te.Linear` with `MXFP8BlockScaling` under `torch.compile(fullgraph=True)`. PyTorch cannot be upgraded to 2.11 or newer.

Do not backport or emulate PyTorch's newer `torch._library.opaque_object` protocol. Do not pass quantizer objects, quantized-storage objects, opaque Python metadata, or global-registry tokens through a custom-op boundary.

Instead, implement a narrow TE-only backend that uses the native PyTorch 2.10 custom-op APIs and a primitive/raw-buffer ABI:

```text
Linear.forward
  ├─ PyTorch with the newer opaque-object protocol
  │    └─ existing #3053 generic _linear_op path
  ├─ PyTorch 2.10 and a supported MXFP8 configuration
  │    └─ dedicated _linear_mxfp8_210_op path
  └─ all other configurations
       └─ existing eager fallback; fullgraph reports the unsupported reason
```

The 2.10 path should accept only tensors, tensor lists, and dispatcher schema primitives (`int`, `bool`, `float`, and optional tensors). It reconstructs temporary MXFP8 quantizers and internal MXFP8 storage inside the real custom-op forward/backward implementation.

This is a deliberately separate backend, not a compatibility mode for the
generic #3053 argument planner. The generic planner remains the implementation
for PyTorch builds with the newer opaque-object protocol.

## Root cause

PR #3053 adds a generic `torch.compile` path for `te.Linear`. Its generic custom-op framework relies on a value-opaque Python object protocol:

```python
from torch._library.opaque_object import register_opaque_type
```

The current TE implementation uses this protocol for both:

1. Quantizer objects, including `MXFP8Quantizer`.
2. `OpaqueValueBundle`, which carries scalar Linear settings and quantized storage metadata across the custom-op boundary.

The current design expects the newer `__fx_repr__() -> (expression, globals)` contract. PyTorch 2.10's private opaque-object implementation follows an older `__repr__` reconstruction contract and is not compatible with this TE path. TE's tests intentionally mark the opaque-object behavior as requiring PyTorch 2.11 or newer.

Without a compatible opaque-object path, Dynamo can trace into mutable `MXFP8Quantizer` operations such as:

```python
quantizer.internal = True
quantizer.optimize_for_gemm = True
quantizer.set_usage(rowwise=True, columnwise=True)
```

and fail with `UnsafeScriptObjectError` / `_setattr_`.

## Why a small opaque-object shim is not sufficient

Adding only `register_opaque_type`, `is_opaque_value_type`, and `get_opaque_type_name` as Python shims would only make imports progress. It does not provide the required Dynamo guards, FX reconstruction, schema support, or correct handling of quantized saved tensors.

Likewise, replacing only `Quantizer` arguments with scalar fields is not sufficient in the current generic framework. `OpaqueValueBundle` is also used for:

- Linear scalar and enum configuration;
- process-group names;
- `QuantizedTensorStorage.__tensor_flatten__` metadata;
- quantized inner-buffer names and layout;
- storage classes and outer shape;
- output and saved-tensor reconstruction.

Training backward must restore quantized input and weight buffers. The existing generic `TensorOrQuantized` representation is:

```text
Tensor?      original tensor or tensor subclass
Tensor[]     inner quantized buffers
Opaque meta  storage/quantizer/layout context
```

Therefore, removing only the quantizer object still leaves an opaque metadata dependency in forward, autograd setup, and backward.

## Scope of the first implementation

The first version should deliberately be a narrow, explicit feature:

```text
PyTorch 2.10.x
te.Linear
MXFP8BlockScaling
plain BF16 or FP16 input
plain non-quantized parameter weight
single GPU
prewarmed FP8 metadata and quantizer state
standard training forward and backward
torch.compile(fullgraph=True)
```

The following must be rejected with an explicit fallback/error rather than silently taking the compatibility path:

- DelayedScaling or custom/non-MXFP8 quantizers;
- `recipe.backward_override` equal to `high_precision` or `dequantized`;
- quantized input or quantized-model-init weight;
- `fp8_output=True` and `fp8_grad=True`;
- `is_first_microbatch != None` / FP8 weight caching;
- fused weight-gradient accumulation (`main_grad`);
- delayed weight-gradient compute;
- FSDP/FSDP2, tensor parallelism, sequence parallelism, and Userbuffers;
- CPU activation offload, DistributedWeight/GTP, and debug instrumentation.
- activation recomputation, CUDA graph capture, and `reduce-overhead` until they
  have dedicated validation;
- MXFP8 2D quantization until it has a separate, passing validation suite.

These exclusions are intentional. Several are already unsupported by PR #3053's generic compiled path, and distributed features require independently designed and validated custom-op state/effect contracts.

The compatibility path must branch before `Linear._get_quantizers()`,
`Linear._get_weight_quantizers()`, `_enable_weight_preswizzle()`, generic
`LinearFwdArgs` construction, and generic `_linear_op` selection. Each of these
can mutate a module-owned quantizer or enter the opaque-object path. The branch
must happen after the existing Userbuffers logic has derived `fp8_output` and
`fp8_grad`, so that those automatically enabled configurations can be rejected
explicitly. It must also account for graph-capture code that rewrites
`is_first_microbatch`.

`prepare_forward()` itself initializes FP8 metadata and can mutate module state
on a cold call. Version 1 therefore requires an eager prewarm before compiling:
enter the same MXFP8 autocast context and invoke the module eagerly once, then
compile subsequent calls. A cold `fullgraph=True` call must either fail with an
explicit "FP8 metadata must be prewarmed" diagnostic or be supported later by
a separate trace-safe initialization design.

## Primitive MXFP8 ABI

No `MXFP8Quantizer`, `MXFP8TensorStorage`, or Python flatten context crosses the 2.10 op boundary. A logical MXFP8 value is represented by raw buffers plus primitive layout/configuration fields:

```text
rowwise_data: Tensor?
rowwise_scale_inv: Tensor?
columnwise_data: Tensor?
columnwise_scale_inv: Tensor?

fp8_dtype: int
logical_dtype: int
rowwise_usage: bool
columnwise_usage: bool
with_gemm_swizzled_scales: bool
with_2d_quantization: bool
```

The corresponding immutable Python helper may be named `MXFP8QuantizerSpec`, but the registered custom-op schema itself must receive only the primitive fields above.

`internal` and `optimize_for_gemm` are quantizer-construction inputs, not a
complete description of an already materialized MXFP8 value. In particular,
`with_gemm_swizzled_scales` must be carried separately for every saved value.
Compact and GEMM-swizzled scales can have the same dtype and shape while having
different semantics; backward must rebuild the layout actually produced by
forward, not infer it from the current quantizer configuration.

Inside the real forward/backward op body, reconstruct a temporary quantizer:

```python
q = MXFP8Quantizer(
    fp8_dtype=DType.cast(spec.fp8_dtype),
    rowwise=spec.rowwise_usage,
    columnwise=spec.columnwise_usage,
    with_2d_quantization=spec.with_2d_quantization,
)
q.internal = spec.internal
q.optimize_for_gemm = spec.optimize_for_gemm
```

This mutation is safe because it executes inside the custom-op implementation, not in the Dynamo-traced Python region.

The fake implementation must consume the same primitive spec and use the existing MXFP8 layout primitives. In particular, it must reuse `MXFP8Quantizer.get_scale_shape()` rather than duplicate scale padding rules. Real and fake paths must share the same normalized usage/layout helper.

For the standard training path, the fixed saved-buffer ABI must cover these
directions, even if a particular backward call does not request every gradient:

| Operand | Forward use | Backward use |
| --- | --- | --- |
| `x` input | rowwise data and scale for fprop | columnwise data and scale for wgrad |
| `w` weight | rowwise data and scale for fprop | columnwise data and scale for dgrad |
| `dy` grad output | rowwise data and scale for dgrad | columnwise data and scale for wgrad |

Forward should take plain `inp`, `weight`, and optional `bias`, then explicitly
return the raw saved `x`/`w` buffers in a fixed order. Backward receives those
buffers plus `grad_output`, explicitly constructs both required `dy` views, and
returns plain dgrad/wgrad/bgrad. It must not accept a generic raw-buffer list
plus a Python-side descriptor. Each materialized operand carries its own dtype
and layout fields: `Format.HYBRID` can use different forward and backward FP8
dtypes.

## Proposed files

Add a dedicated module rather than version-forking the generic opaque-object framework:

```text
transformer_engine/pytorch/dynamo/mxfp8_linear_210.py
```

It should contain:

- PyTorch capability predicate for the 2.10 compatibility path;
- `MXFP8QuantizerSpec` encode/decode helpers;
- raw MXFP8 buffer pack/unpack helpers;
- dedicated forward custom op and fake forward;
- dedicated backward custom op and fake backward;
- `register_autograd` setup/context glue;
- strict unsupported-configuration predicate.

Modify:

- `transformer_engine/pytorch/module/linear.py`
  - select the 2.10 backend before `_get_quantizers()`,
    `_get_weight_quantizers()`, and `_enable_weight_preswizzle()` mutate a
    quantizer;
  - add `_compile_mxfp8_210_unsupported_reason()`;
  - retain the existing #3053 `_linear_op` path for newer PyTorch versions.
- `tests/pytorch/test_torch_compile.py`
  - add a PyTorch-2.10-only compatibility test group.

Do not modify the current generic `dynamo/custom_op.py` argument planner to remove opaque objects. That would expand the patch from a narrowly scoped backend into a rewrite of generic scalar, ProcessGroup, storage-metadata, and 2.11+ behavior.

The new backend should keep raw-buffer pack/rebuild private to
`mxfp8_linear_210.py`. Modifying `mxfp8_tensor.py`, `MXFP8TensorStorage`, or
their public APIs is not required for version 1 and would widen the regression
surface. Retain `register_value_opaque_quantizer(MXFP8Quantizer)` unchanged for
the PyTorch 2.11+ path.

## Required validation

Run against a real CUDA PyTorch 2.10 wheel. The local development version alone is not a substitute.

1. **API preflight:** confirm `torch.library.custom_op`, `register_fake`, `register_autograd`, and the required native schemas (`Tensor`, `Tensor?`, `Tensor[]`, `int`, `bool`) are available.
2. **Numerical parity:** compare eager and compiled MXFP8 Linear for output, input gradient, weight gradient, and bias gradient. Start with 1D MXFP8; add 2D only after a separate validation suite passes. Cover BF16 and FP16, bias enabled/disabled, only dgrad, only wgrad, both gradients, rank-2 input, and rank-3 input. Cover E4M3 and HYBRID format.
3. **Graph behavior:** use `torch.compile(fullgraph=True)` and assert no `MXFP8Quantizer._setattr_` graph break occurs. Test prewarm, first compilation, and repeated calls. The suite must not use the generic `_opaque_available` skip.
4. **Fake/layout parity:** verify every saved raw slot's number, order, shape, dtype, presence, scale padding, and swizzle bit for fake forward, real forward, saved tensors, and backward reconstruction.
5. **Negative coverage:** every excluded configuration, including `backward_override`, cold initialization, 2D MXFP8, and invalid MX block dimensions, must provide a clear unsupported reason. It must not silently take a global registry path or run with an incomplete layout.
6. **Optional follow-up:** test dynamic leading shapes and `mode="reduce-overhead"` only after the standard path is correct. Add a pinned PyTorch 2.10 CUDA CI job.

## Alternatives rejected

### Backport or shim PyTorch opaque objects

Rejected for a TE-only solution. The visible API is small, but correct support depends on Dynamo guards, FX code generation, object reconstruction, and custom-op schema integration. Maintaining a Torch fork would be required for a robust solution.

### Global registry token for a Quantizer

Rejected as an op ABI. A token does not encode graph semantics, so Dynamo cannot guard against quantizer mutation or registry reuse. It also creates graph-cache, multi-process, lifecycle, and re-entrant compile hazards. A runtime cache may be used internally after primitive spec has established correctness, but cannot be the source of truth for fake or graph behavior.

### Remove opaque objects from the generic custom-op framework

Rejected as the initial implementation. It risks regressions in the currently working PyTorch 2.11+ path and requires a generic representation of every Quantizer, storage layout, ProcessGroup, and scalar metadata type.

## Estimated delivery

| Phase | Deliverable | Estimate |
| --- | --- | --- |
| Preflight | Validate exact PyTorch 2.10 CUDA custom-op APIs | 0.5-1 day |
| Prototype | Single-GPU MXFP8 forward-only fullgraph proof | 1-2 days |
| v1 training | Forward/backward raw-buffer ABI and parity tests | 5-8 engineering days |
| Hardening | Diagnostics, unsupported cases, and pinned 2.10 CI | 2-4 engineering days |
| Optional | Dynamic shapes and reduce-overhead verification | 1-3 days |

Expected v1: approximately 1.5-2 weeks for a stable single-GPU standard training path. Tensor/sequence parallelism, FSDP, Userbuffers, quantized weight initialization, and weight caching should be planned as separate projects.

## Relevant code

- `transformer_engine/pytorch/dynamo/custom_op.py` -- generic #3053 opaque-object custom-op framework.
- `transformer_engine/pytorch/dynamo/quantizer_opaque.py` -- quantizer value representation for newer PyTorch.
- `transformer_engine/pytorch/module/linear.py` -- existing Linear real/fake forward/backward implementations and custom-op dispatch point.
- `transformer_engine/pytorch/tensor/mxfp8_tensor.py` -- MXFP8 quantizer and buffer geometry.
- `transformer_engine/pytorch/quantized_tensor.py` -- mutable Quantizer state and quantized storage abstractions.
- `tests/pytorch/test_torch_compile.py` -- current #3053 compile coverage.
