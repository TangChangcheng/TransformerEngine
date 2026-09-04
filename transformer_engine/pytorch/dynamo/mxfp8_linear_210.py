# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch 2.10-only MXFP8 ``Linear`` custom-op backend.

This deliberately does not use the generic TE ``dynamo.custom_op`` framework.
That framework transports quantizers and quantized-storage metadata as PyTorch
value-opaque Python objects, whose protocol is not compatible with PyTorch
2.10.  This module has a fixed Tensor/primitive ABI instead: the op quantizes
plain input and weight tensors internally, returns raw MXFP8 buffers required
by backward, and reconstructs internal MXFP8 storage only inside custom-op
implementations.

The backend is intentionally narrow.  ``Linear`` owns feature gating; this file
only implements single-device, 1D MXFP8 fprop/bprop mechanics.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..constants import DType
from ..cpp_extensions import general_gemm
from ..tensor.mxfp8_tensor import MXFP8Quantizer
from ..tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage
from ..torch_version import torch_version


_NAMESPACE = "transformer_engine_compile_210"
_EMPTY_DTYPE = torch.uint8
_DTYPE_FP16 = 0
_DTYPE_BF16 = 1
_TORCH_DTYPE_FROM_CODE = {_DTYPE_FP16: torch.float16, _DTYPE_BF16: torch.bfloat16}


def _encode_logical_dtype(dtype: torch.dtype) -> int:
    if dtype == torch.float16:
        return _DTYPE_FP16
    if dtype == torch.bfloat16:
        return _DTYPE_BF16
    raise ValueError(f"MXFP8 Linear 2.10 backend does not support dtype {dtype}")


def is_mxfp8_linear_210_available() -> bool:
    """Whether this fixed-schema backend is applicable to this Torch build.

    Do not use presence of ``opaque_object`` as a capability test: PyTorch 2.10
    has an older incompatible private implementation.  The native custom-op API
    is the only runtime dependency of this backend.
    """

    version = torch_version()
    if not ((2, 10, 0) <= version < (2, 11, 0)):
        return False
    if not hasattr(torch.library, "custom_op"):
        return False
    # These are methods on the CustomOpDef returned by ``custom_op``.  The
    # public decorators are present in all target 2.10 builds we support.
    return hasattr(torch.library, "register_fake") and hasattr(
        torch.library, "register_autograd"
    )


def _empty(device: torch.device) -> torch.Tensor:
    return torch.empty(0, dtype=_EMPTY_DTYPE, device=device)


def _optional(tensor: Optional[torch.Tensor], device: torch.device) -> torch.Tensor:
    return tensor if tensor is not None else _empty(device)


def _present(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    return None if tensor.numel() == 0 else tensor


def _make_quantizer(
    fp8_dtype: int,
    *,
    rowwise: bool,
    columnwise: bool,
    optimize_for_gemm: bool = True,
) -> MXFP8Quantizer:
    """Build a short-lived internal quantizer inside the custom op."""

    quantizer = MXFP8Quantizer(
        DType.cast(fp8_dtype),
        rowwise=rowwise,
        columnwise=columnwise,
        with_2d_quantization=False,
    )
    quantizer.internal = True
    quantizer.optimize_for_gemm = optimize_for_gemm
    return quantizer


def _make_storage(
    rowwise_data: torch.Tensor,
    rowwise_scale_inv: torch.Tensor,
    columnwise_data: torch.Tensor,
    columnwise_scale_inv: torch.Tensor,
    *,
    fp8_dtype: int,
    logical_dtype: torch.dtype,
    with_gemm_swizzled_scales: bool,
) -> MXFP8TensorStorage:
    """Rebuild internal MXFP8 storage from the fixed raw-buffer ABI."""

    rowwise_data_opt = _present(rowwise_data)
    columnwise_data_opt = _present(columnwise_data)
    return MXFP8TensorStorage(
        rowwise_data=rowwise_data_opt,
        rowwise_scale_inv=_present(rowwise_scale_inv),
        columnwise_data=columnwise_data_opt,
        columnwise_scale_inv=_present(columnwise_scale_inv),
        fp8_dtype=DType.cast(fp8_dtype),
        quantizer=None,
        with_gemm_swizzled_scales=with_gemm_swizzled_scales,
        fake_dtype=logical_dtype,
    )


def _saved_scales_are_swizzled() -> bool:
    """Layout flag for this backend's materialized fprop MXFP8 buffers.

    Version 1 always quantizes its internal GEMM operands with
    ``optimize_for_gemm=True``.  Keep this as an explicit helper instead of
    inferring the saved layout later from a freshly-created quantizer.
    """

    return True


def _storage_buffers(
    storage: MXFP8TensorStorage, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return MXFP8 buffers in the fixed ABI order."""

    return (
        _optional(storage._rowwise_data, device),
        _optional(storage._rowwise_scale_inv, device),
        _optional(storage._columnwise_data, device),
        _optional(storage._columnwise_scale_inv, device),
    )


def _quantize(
    tensor: torch.Tensor,
    *,
    fp8_dtype: int,
    rowwise: bool,
    columnwise: bool,
) -> MXFP8TensorStorage:
    quantizer = _make_quantizer(
        fp8_dtype,
        rowwise=rowwise,
        columnwise=columnwise,
    )
    result = quantizer(tensor)
    assert isinstance(result, MXFP8TensorStorage)
    return result


def _out_shape(inp: torch.Tensor, out_features: int) -> Tuple[int, ...]:
    return (*tuple(inp.shape[:-1]), out_features)


@torch.library.custom_op(
    f"{_NAMESPACE}::mxfp8_linear_fwd",
    mutates_args=(),
    device_types="cuda",
)
def _mxfp8_linear_fwd(
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    fwd_fp8_dtype: int,
    bwd_fp8_dtype: int,
    needs_dgrad: bool,
    needs_wgrad: bool,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """MXFP8 fprop plus explicit raw tensors saved for backward."""

    # x is consumed rowwise by fprop and columnwise by wgrad.  w is consumed
    # rowwise by fprop and columnwise by dgrad.  Keeping a fixed return arity
    # makes the custom-op autograd ABI independent of which gradients are used.
    x_q = _quantize(
        inp,
        fp8_dtype=fwd_fp8_dtype,
        rowwise=True,
        columnwise=needs_wgrad,
    )
    w_q = _quantize(
        weight,
        fp8_dtype=fwd_fp8_dtype,
        rowwise=True,
        columnwise=needs_dgrad,
    )
    out, *_ = general_gemm(
        w_q,
        x_q,
        out_dtype=inp.dtype,
        bias=bias,
        use_split_accumulator=True,
    )
    out = out.view(_out_shape(inp, weight.shape[0]))
    del bwd_fp8_dtype
    return (out, *_storage_buffers(x_q, inp.device), *_storage_buffers(w_q, weight.device))


@_mxfp8_linear_fwd.register_fake
def _mxfp8_linear_fwd_fake(
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    fwd_fp8_dtype: int,
    bwd_fp8_dtype: int,
    needs_dgrad: bool,
    needs_wgrad: bool,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Data-free fprop twin with the same fixed raw-buffer ABI."""

    del bias, bwd_fp8_dtype
    q_x = _make_quantizer(
        fwd_fp8_dtype, rowwise=True, columnwise=needs_wgrad, optimize_for_gemm=True
    )
    q_w = _make_quantizer(
        fwd_fp8_dtype, rowwise=True, columnwise=needs_dgrad, optimize_for_gemm=True
    )

    def _buffers(q: MXFP8Quantizer, tensor: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        specs = q.inner_tensor_specs(tuple(tensor.shape))
        return (
            torch.empty(*specs["_rowwise_data"][0], dtype=torch.uint8, device=tensor.device),
            torch.empty(
                *specs["_rowwise_scale_inv"][0], dtype=torch.uint8, device=tensor.device
            ),
            (
                torch.empty(
                    *specs["_columnwise_data"][0], dtype=torch.uint8, device=tensor.device
                )
                if q.columnwise_usage
                else _empty(tensor.device)
            ),
            (
                torch.empty(
                    *specs["_columnwise_scale_inv"][0], dtype=torch.uint8, device=tensor.device
                )
                if q.columnwise_usage
                else _empty(tensor.device)
            ),
        )

    out = torch.empty(_out_shape(inp, weight.shape[0]), dtype=inp.dtype, device=inp.device)
    return (out, *_buffers(q_x, inp), *_buffers(q_w, weight))


@torch.library.custom_op(
    f"{_NAMESPACE}::mxfp8_linear_bwd",
    mutates_args=(),
    device_types="cuda",
)
def _mxfp8_linear_bwd(
    grad_output: torch.Tensor,
    x_rowwise_data: torch.Tensor,
    x_rowwise_scale_inv: torch.Tensor,
    x_columnwise_data: torch.Tensor,
    x_columnwise_scale_inv: torch.Tensor,
    w_rowwise_data: torch.Tensor,
    w_rowwise_scale_inv: torch.Tensor,
    w_columnwise_data: torch.Tensor,
    w_columnwise_scale_inv: torch.Tensor,
    fwd_fp8_dtype: int,
    bwd_fp8_dtype: int,
    logical_dtype: int,
    needs_dgrad: bool,
    needs_wgrad: bool,
    needs_bgrad: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MXFP8 bprop using raw fprop saves and a newly quantized grad output."""

    dtype = _TORCH_DTYPE_FROM_CODE[logical_dtype]
    x_q = _make_storage(
        x_rowwise_data,
        x_rowwise_scale_inv,
        x_columnwise_data,
        x_columnwise_scale_inv,
        fp8_dtype=fwd_fp8_dtype,
        logical_dtype=dtype,
        with_gemm_swizzled_scales=_saved_scales_are_swizzled(),
    )
    w_q = _make_storage(
        w_rowwise_data,
        w_rowwise_scale_inv,
        w_columnwise_data,
        w_columnwise_scale_inv,
        fp8_dtype=fwd_fp8_dtype,
        logical_dtype=dtype,
        with_gemm_swizzled_scales=_saved_scales_are_swizzled(),
    )
    dy_q = _quantize(
        grad_output,
        fp8_dtype=bwd_fp8_dtype,
        rowwise=needs_dgrad or needs_bgrad,
        columnwise=needs_wgrad,
    )

    dgrad = _empty(grad_output.device)
    wgrad = _empty(grad_output.device)
    bgrad = _empty(grad_output.device)
    if needs_dgrad:
        dgrad, *_ = general_gemm(
            w_q,
            dy_q,
            layout="NN",
            out_dtype=dtype,
            grad=True,
            use_split_accumulator=True,
        )
        dgrad = dgrad.view(*grad_output.shape[:-1], w_rowwise_data.shape[-1])
    if needs_wgrad:
        wgrad, *_ = general_gemm(
            x_q,
            dy_q,
            layout="NT",
            out_dtype=dtype,
            grad=True,
            use_split_accumulator=True,
        )
    if needs_bgrad:
        # The generic FP8 path computes bgrad during grad-output preprocessing.
        # This primitive-only backend has no Python args object for that helper,
        # so use the mathematically identical reduction directly.
        bgrad = grad_output.reshape(-1, grad_output.shape[-1]).sum(dim=0)
    return dgrad, wgrad, bgrad


@_mxfp8_linear_bwd.register_fake
def _mxfp8_linear_bwd_fake(
    grad_output: torch.Tensor,
    x_rowwise_data: torch.Tensor,
    x_rowwise_scale_inv: torch.Tensor,
    x_columnwise_data: torch.Tensor,
    x_columnwise_scale_inv: torch.Tensor,
    w_rowwise_data: torch.Tensor,
    w_rowwise_scale_inv: torch.Tensor,
    w_columnwise_data: torch.Tensor,
    w_columnwise_scale_inv: torch.Tensor,
    fwd_fp8_dtype: int,
    bwd_fp8_dtype: int,
    logical_dtype: int,
    needs_dgrad: bool,
    needs_wgrad: bool,
    needs_bgrad: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Data-free bprop twin."""

    del (
        x_rowwise_scale_inv,
        x_columnwise_data,
        x_columnwise_scale_inv,
        w_rowwise_scale_inv,
        w_columnwise_data,
        w_columnwise_scale_inv,
        fwd_fp8_dtype,
        bwd_fp8_dtype,
    )
    dtype = _TORCH_DTYPE_FROM_CODE[logical_dtype]
    dgrad = (
        torch.empty(
            (*grad_output.shape[:-1], x_rowwise_data.shape[-1]),
            dtype=dtype,
            device=grad_output.device,
        )
        if needs_dgrad
        else _empty(grad_output.device)
    )
    wgrad = (
        torch.empty(
            (grad_output.shape[-1], x_rowwise_data.shape[-1]),
            dtype=dtype,
            device=grad_output.device,
        )
        if needs_wgrad
        else _empty(grad_output.device)
    )
    bgrad = (
        torch.empty((grad_output.shape[-1],), dtype=dtype, device=grad_output.device)
        if needs_bgrad
        else _empty(grad_output.device)
    )
    return dgrad, wgrad, bgrad


def _mxfp8_linear_setup_context(ctx, inputs, output) -> None:
    inp, weight, bias, fwd_fp8_dtype, bwd_fp8_dtype, needs_dgrad, needs_wgrad = inputs
    del inp, weight
    (
        _out,
        x_rowwise_data,
        x_rowwise_scale_inv,
        x_columnwise_data,
        x_columnwise_scale_inv,
        w_rowwise_data,
        w_rowwise_scale_inv,
        w_columnwise_data,
        w_columnwise_scale_inv,
    ) = output
    ctx.save_for_backward(
        x_rowwise_data,
        x_rowwise_scale_inv,
        x_columnwise_data,
        x_columnwise_scale_inv,
        w_rowwise_data,
        w_rowwise_scale_inv,
        w_columnwise_data,
        w_columnwise_scale_inv,
    )
    ctx.fwd_fp8_dtype = fwd_fp8_dtype
    ctx.bwd_fp8_dtype = bwd_fp8_dtype
    ctx.logical_dtype = _encode_logical_dtype(output[0].dtype)
    ctx.needs_dgrad = needs_dgrad
    ctx.needs_wgrad = needs_wgrad
    ctx.needs_bgrad = bias is not None and bias.requires_grad


def _mxfp8_linear_backward_wrapper(ctx, grad_output, *unused_output_grads):
    del unused_output_grads
    saved = ctx.saved_tensors
    dgrad, wgrad, bgrad = _mxfp8_linear_bwd(
        grad_output,
        *saved,
        ctx.fwd_fp8_dtype,
        ctx.bwd_fp8_dtype,
        ctx.logical_dtype,
        ctx.needs_dgrad,
        ctx.needs_wgrad,
        ctx.needs_bgrad,
    )
    return (
        dgrad if ctx.needs_dgrad else None,
        wgrad if ctx.needs_wgrad else None,
        bgrad if ctx.needs_bgrad else None,
        None,
        None,
        None,
        None,
    )


_mxfp8_linear_fwd.register_autograd(
    _mxfp8_linear_backward_wrapper,
    setup_context=_mxfp8_linear_setup_context,
)


def mxfp8_linear_210(
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    fwd_fp8_dtype: DType,
    bwd_fp8_dtype: DType,
) -> torch.Tensor:
    """Run the public-facing first result of the fixed-schema forward op."""

    needs_dgrad = inp.requires_grad
    needs_wgrad = weight.requires_grad
    out, *_ = _mxfp8_linear_fwd(
        inp,
        weight,
        bias,
        int(fwd_fp8_dtype),
        int(bwd_fp8_dtype),
        needs_dgrad,
        needs_wgrad,
    )
    return out


__all__ = ["is_mxfp8_linear_210_available", "mxfp8_linear_210"]
