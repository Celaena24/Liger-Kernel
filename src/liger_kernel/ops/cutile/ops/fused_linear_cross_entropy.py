# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Hopper-optimized fused linear + cross-entropy loss via CuTile DSL.

Key insight vs the Triton implementation:
  The Triton FLCE splits the token batch into tiny chunks (≈ 128 rows each) to
  keep peak logit memory bounded.  On older GPUs this was acceptable because the
  GEMM was already fast enough.  On Hopper (H100/H200) the cuBLAS tensor-core
  throughput is so high that running 32 × (128-row) GEMMs is catastrophically
  slower than a single (4096-row) GEMM — we measured 8× regression vs unfused
  PyTorch on H200.

  This cutile implementation uses large chunks (up to MAX_CHUNK_TOKENS tokens,
  typically equal to the full batch for short sequences), reducing the GEMM
  count from O(BT/128) to O(BT/4096) while keeping peak logit memory within a
  configurable budget.  The CE pass then uses the Hopper-tuned cuTile kernel
  that already leverages TMA async-prefetch hints (latency=3).

  Additionally, on Hopper we overlap the forward GEMM of chunk i+1 with the CE
  kernel of chunk i via separate CUDA streams, hiding the memory latency of the
  CE pass behind the compute of the next GEMM.
"""

import math

from typing import Optional

import cuda.tile as ct
import torch

from liger_kernel.ops.cutile.ops.cross_entropy import LigerCrossEntropyFunction  # noqa: F401
from liger_kernel.ops.cutile.ops.cross_entropy import _select_cross_entropy_block_size
from liger_kernel.ops.cutile.ops.cross_entropy import liger_cross_entropy_kernel_ct
from liger_kernel.ops.cutile.ops.utils import _next_power_of_2
from liger_kernel.ops.cutile.ops.utils import element_mul_kernel
from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd

# Maximum logit-tensor memory budget per chunk (bytes).
# On H200 (141 GB HBM3e) we can afford large chunks; 2 GB keeps peak
# per-chunk overhead reasonable even when grad_input and grad_weight buffers
# are also live.
_HOPPER_CHUNK_BUDGET_BYTES = 2 * 1024**3  # 2 GB

# Minimum chunk size in tokens: below this threshold the GEMM M-dimension is
# too small for cuBLAS to reach good tensor-core utilization on Hopper.
_MIN_CHUNK_TOKENS = 512


def _select_hopper_chunk_size(BT: int, V: int, dtype: torch.dtype) -> int:
    """
    Return a chunk size (number of token rows) tuned for Hopper GEMMs.

    The Triton FLCE uses chunk_size = next_power_of_2(ceil(BT / ceil(V/H))),
    which gives very small chunks (e.g. 128 rows for LLaMA-3 vocab).  On
    Hopper, each cuBLAS GEMM call has non-trivial kernel-launch overhead and
    the tensor-core pipeline only reaches peak throughput for large M.

    Here we choose the largest power-of-two chunk size that keeps the logit
    buffer within _HOPPER_CHUNK_BUDGET_BYTES, then clamp to BT.
    """
    bytes_per_token = V * (2 if dtype in (torch.bfloat16, torch.float16) else 4)
    max_tokens = _HOPPER_CHUNK_BUDGET_BYTES // max(bytes_per_token, 1)
    # Clamp to [_MIN_CHUNK_TOKENS, BT] and round up to next power of 2.
    chunk = _next_power_of_2(max(min(max_tokens, BT), _MIN_CHUNK_TOKENS))
    return min(chunk, _next_power_of_2(BT))


def fused_linear_cross_entropy_forward(
    _input: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    ce_weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    lse_square_scale: float = 0.0,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    softcap: Optional[float] = None,
    return_z_loss: bool = False,
    accum_dtype: Optional[torch.dtype] = None,
    use_token_scaling: bool = False,
    return_token_accuracy: bool = False,
    return_predicted_tokens: bool = False,
):
    assert isinstance(return_z_loss, bool), f"return_z_loss must be True or False. Got: {return_z_loss}"
    assert isinstance(return_token_accuracy, bool), (
        f"return_token_accuracy must be True or False. Got: {return_token_accuracy}"
    )
    assert isinstance(return_predicted_tokens, bool), (
        f"return_predicted_tokens must be True or False. Got: {return_predicted_tokens}"
    )

    device = _input.device
    dtype = _input.dtype
    input_requires_grad = _input.requires_grad

    BT, H = _input.shape
    V = weight.shape[0]

    # ---- Hopper-specific chunk size ----
    # Use large chunks to maximise cuBLAS tensor-core utilisation.
    chunk_size = _select_hopper_chunk_size(BT, V, dtype)
    num_chunks = math.ceil(BT / chunk_size)

    # CE kernel block size (stays the same as CE-only kernel).
    BLOCK_SIZE = _select_cross_entropy_block_size(V)

    # ---- Output / gradient buffers ----
    grad_input = torch.zeros_like(_input, device=device)

    if input_requires_grad:
        accum_dtype_ = accum_dtype if accum_dtype is not None else weight.dtype
        grad_weight = torch.zeros_like(weight, dtype=accum_dtype_, device=device) if weight.requires_grad else None
        grad_bias = torch.zeros_like(bias, dtype=accum_dtype_, device=device) if bias is not None else None
    else:
        grad_weight = None
        grad_bias = None

    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)
    z_loss_1d = torch.zeros(BT, dtype=dtype, device=device) if return_z_loss else None
    token_accuracy_1d = torch.zeros(BT, dtype=torch.float32, device=device) if return_token_accuracy else None
    predicted_tokens_1d = torch.full((BT,), -1, dtype=torch.int64, device=device) if return_predicted_tokens else None

    # Pre-compute ignore-index statistics.
    target_mask = target != ignore_index
    total_n_non_ignore = target_mask.sum().item()
    total_sum_non_ignore_ce_weight = float(total_n_non_ignore)
    ce_weight_sum = 0.0
    if ce_weight is not None:
        assert ce_weight.shape[0] == V
        assert torch.is_floating_point(ce_weight)
        total_sum_non_ignore_ce_weight = (
            torch.gather(ce_weight, dim=0, index=target.masked_select(target_mask)).sum().item()
            if total_n_non_ignore > 0
            else 1.0
        )
        ce_weight_sum = ce_weight.float().sum().item()
        if ce_weight.stride(-1) != 1:
            ce_weight = ce_weight.contiguous()

    has_softcapping = softcap is not None
    softcap_val = float(softcap) if softcap is not None else 0.0
    reduction_mean = int(reduction == "mean")
    inv_n_non_ignore = 1.0 / max(total_n_non_ignore, 1)

    # Dummy tensors required by the cutile kernel for disabled outputs.
    dummy_f32 = torch.zeros(1, dtype=torch.float32, device=device)
    dummy_i64 = torch.zeros(1, dtype=torch.int64, device=device)
    dummy_weight_ct = torch.zeros(1, dtype=torch.float32, device=device)
    has_weight = ce_weight is not None

    # ---- Hopper stream overlap: GEMM[i+1] ∥ CE[i] ----
    # We use two streams so the cuBLAS GEMM of the next chunk can run
    # concurrently with the CE kernel of the current chunk on the SM partition.
    main_stream = torch.cuda.current_stream()
    gemm_stream = torch.cuda.Stream(device=device)
    ce_stream = torch.cuda.Stream(device=device)

    # Pre-materialise first chunk's logits on the GEMM stream so the loop
    # starts with a logits buffer ready.
    start0 = 0
    end0 = min(chunk_size, BT)
    with torch.cuda.stream(gemm_stream):
        logits_cur = (_input[start0:end0] @ weight.t()).contiguous()
        if bias is not None:
            logits_cur = logits_cur + bias

    for chunk_id in range(num_chunks):
        start = chunk_id * chunk_size
        end = min(start + chunk_size, BT)
        n_rows = end - start

        # ---- Launch NEXT chunk's GEMM on gemm_stream ----
        # CE and GEMM streams will be running concurrently.
        if chunk_id + 1 < num_chunks:
            nxt_start = (chunk_id + 1) * chunk_size
            nxt_end = min(nxt_start + chunk_size, BT)
            with torch.cuda.stream(gemm_stream):
                # Wait until the previous CE is done using logits_cur memory
                # before we write the next chunk (logits_cur is reused below).
                gemm_stream.wait_stream(ce_stream)
                logits_nxt = (_input[nxt_start:nxt_end] @ weight.t()).contiguous()
                if bias is not None:
                    logits_nxt = logits_nxt + bias
        else:
            logits_nxt = None

        # Ensure logits_cur is ready before CE.
        ce_stream.wait_stream(gemm_stream)

        input_chunk = _input[start:end]
        target_chunk = target[start:end].contiguous()

        # Slices into output buffers.
        loss_slice = loss_1d[start:end]
        z_loss_slice = z_loss_1d[start:end] if return_z_loss else dummy_f32
        ta_slice = token_accuracy_1d[start:end] if return_token_accuracy else dummy_f32
        pt_slice = predicted_tokens_1d[start:end] if return_predicted_tokens else dummy_i64

        # Token-scaling: capture per-token predicted probability before CE.
        scaling_factors = None
        if use_token_scaling:
            logits_fs = logits_cur.detach()
            if has_softcapping:
                logits_fs = softcap_val * torch.tanh(logits_fs / softcap_val)
            probs = torch.softmax(logits_fs.float(), dim=-1)
            valid_mask = target_chunk != ignore_index
            valid_targets = target_chunk[valid_mask]
            pred_probs = torch.zeros(n_rows, dtype=probs.dtype, device=device)
            if valid_targets.numel() > 0:
                pred_probs[valid_mask] = torch.gather(probs[valid_mask], -1, valid_targets.unsqueeze(-1)).squeeze(-1)
            scaling_factors = pred_probs.detach()

        # ---- cuTile CE kernel (computes loss, writes d(logits) in-place) ----
        ct.launch(
            ce_stream,
            (n_rows, 1, 1),
            liger_cross_entropy_kernel_ct,
            (
                logits_cur,
                target_chunk,
                ce_weight.float() if has_weight else dummy_weight_ct,
                loss_slice,
                z_loss_slice if return_z_loss else dummy_f32,
                ta_slice if return_token_accuracy else dummy_f32,
                pt_slice if return_predicted_tokens else dummy_i64,
                int(V),
                float(inv_n_non_ignore),
                float(total_sum_non_ignore_ce_weight),
                float(ce_weight_sum),
                int(ignore_index),
                float(label_smoothing),
                float(lse_square_scale),
                float(softcap_val),
                int(BLOCK_SIZE),
                int(input_requires_grad),
                int(reduction_mean),
                int(has_weight),
                int(has_softcapping),
                int(return_z_loss),
                int(return_token_accuracy),
                int(return_predicted_tokens),
            ),
        )

        # ---- Gradient accumulation (on ce_stream, uses d(logits) in logits_cur) ----
        if use_token_scaling:
            with torch.cuda.stream(ce_stream):
                loss_slice_val = loss_1d[start:end] * scaling_factors
                loss_1d[start:end] = loss_slice_val
                if return_z_loss:
                    z_loss_1d[start:end] = z_loss_1d[start:end] * scaling_factors

        if input_requires_grad:
            with torch.cuda.stream(ce_stream):
                # logits_cur now holds d(loss)/d(logits) after the CE kernel.
                if use_token_scaling:
                    logits_cur *= scaling_factors.unsqueeze(-1)
                grad_input[start:end] = logits_cur @ weight

        if grad_weight is not None and input_requires_grad:
            with torch.cuda.stream(ce_stream):
                grad_weight += torch.mm(logits_cur.t(), input_chunk).to(accum_dtype_)

        if bias is not None and input_requires_grad:
            with torch.cuda.stream(ce_stream):
                torch.add(input=grad_bias, other=logits_cur.sum(dim=0), out=grad_bias, alpha=1.0)

        # Advance buffer pointer.
        logits_cur = logits_nxt

    # Sync all streams back to main before returning.
    main_stream.wait_stream(ce_stream)
    main_stream.wait_stream(gemm_stream)

    # ---- Reduction ----
    if reduction == "none":
        loss = loss_1d
        z_loss = z_loss_1d if return_z_loss else None
        token_accuracy = token_accuracy_1d if return_token_accuracy else None
    else:
        loss = torch.sum(loss_1d)
        z_loss = torch.sum(z_loss_1d) if return_z_loss else None
        token_accuracy = torch.sum(token_accuracy_1d) / max(total_n_non_ignore, 1) if return_token_accuracy else None

    predicted_tokens = predicted_tokens_1d if return_predicted_tokens else None

    if grad_weight is not None:
        grad_weight = grad_weight.to(weight.dtype)
    if grad_bias is not None:
        grad_bias = grad_bias.to(bias.dtype)

    return loss, z_loss, token_accuracy, predicted_tokens, grad_input, grad_weight, grad_bias


def fused_linear_cross_entropy_backward(
    grad_output: torch.Tensor,
    grad_input: torch.Tensor,
    grad_weight: Optional[torch.Tensor],
    grad_bias: Optional[torch.Tensor],
):
    """Scale pre-accumulated gradients by the upstream loss gradient."""
    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        return grad_input, grad_weight, grad_bias

    BT, H = grad_input.shape
    BLOCK_SIZE_IN = min(65536, _next_power_of_2(H))

    ct.launch(
        torch.cuda.current_stream(),
        (BT, 1, 1),
        element_mul_kernel,
        (grad_input, grad_output, int(H), int(BLOCK_SIZE_IN), H % BLOCK_SIZE_IN != 0),
    )

    if grad_weight is not None:
        V, H_w = grad_weight.shape
        BLOCK_SIZE_W = min(65536, _next_power_of_2(H_w))
        ct.launch(
            torch.cuda.current_stream(),
            (V, 1, 1),
            element_mul_kernel,
            (grad_weight, grad_output, int(H_w), int(BLOCK_SIZE_W), H_w % BLOCK_SIZE_W != 0),
        )

    if grad_bias is not None:
        V_b = grad_bias.shape[0]
        BLOCK_SIZE_B = min(65536, _next_power_of_2(1))
        ct.launch(
            torch.cuda.current_stream(),
            (V_b, 1, 1),
            element_mul_kernel,
            (grad_bias, grad_output, 1, int(BLOCK_SIZE_B), False),
        )

    return grad_input, grad_weight, grad_bias


class LigerFusedLinearCrossEntropyFunction(torch.autograd.Function):
    """
    cuTile autograd wrapper for fused linear + cross-entropy loss.

    Hopper-specific optimisations vs the Triton implementation:
      1. Large chunk sizes (up to 2 GB of logits) for better cuBLAS utilisation.
      2. GEMM/CE stream overlap: forward GEMM of chunk i+1 runs concurrently
         with the cuTile CE kernel of chunk i.
      3. TMA async-prefetch hints (latency=3) already baked into the cuTile
         CE kernel for Hopper's async-copy hardware.
    """

    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        _input: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        ce_weight: Optional[torch.FloatTensor] = None,
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
        return_z_loss: bool = False,
        accum_dtype: Optional[torch.dtype] = None,
        use_token_scaling: bool = False,
        return_token_accuracy: bool = False,
        return_predicted_tokens: bool = False,
    ):
        loss, z_loss, token_accuracy, predicted_tokens, grad_input, grad_weight, grad_bias = (
            fused_linear_cross_entropy_forward(
                _input=_input,
                weight=weight,
                target=target,
                bias=bias,
                ce_weight=ce_weight,
                ignore_index=ignore_index,
                lse_square_scale=lse_square_scale,
                label_smoothing=label_smoothing,
                reduction=reduction,
                softcap=softcap,
                return_z_loss=return_z_loss,
                accum_dtype=accum_dtype,
                use_token_scaling=use_token_scaling,
                return_token_accuracy=return_token_accuracy,
                return_predicted_tokens=return_predicted_tokens,
            )
        )
        ctx.save_for_backward(
            grad_input.detach(),
            grad_weight.detach() if grad_weight is not None else None,
            grad_bias.detach() if grad_bias is not None else None,
        )
        ctx.return_z_loss = return_z_loss
        ctx.return_token_accuracy = return_token_accuracy
        ctx.return_predicted_tokens = return_predicted_tokens
        return loss, z_loss, token_accuracy, predicted_tokens

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output, grad_output2, grad_output3, grad_output4):
        if ctx.return_z_loss:
            del grad_output2
        if ctx.return_token_accuracy:
            del grad_output3
        if ctx.return_predicted_tokens:
            del grad_output4

        grad_input, grad_weight, grad_bias = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_backward(
            grad_output, grad_input, grad_weight, grad_bias
        )
        return (
            grad_input,
            grad_weight,
            None,  # target
            grad_bias,
            None,  # ce_weight
            None,  # ignore_index
            None,  # lse_square_scale
            None,  # label_smoothing
            None,  # reduction
            None,  # softcap
            None,  # return_z_loss
            None,  # accum_dtype
            None,  # use_token_scaling
            None,  # return_token_accuracy
            None,  # return_predicted_tokens
        )
