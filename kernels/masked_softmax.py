# kernels/masked_softmax.py
"""Triton kernels for the MCTS policy prior: legal-move masking + softmax.

Two versions, both matched against kernels.reference.masked_softmax_ref:

masked_softmax_dense (v1)
    The textbook row softmax with a mask: one program per row, the whole
    4672-wide row in one block, illegal entries set to -inf. Reads and writes
    every entry, so at large batch it is bandwidth-bound -- the roofline
    reference point for v2.

masked_softmax_gather (v2)
    Reads only the legal entries and writes them packed. Legal moves come in a
    fixed-shape layout -- idx (B, 256) int16, count (B,) -- rather than as a
    variable-length CSR list. A CSR list's length changes with every batch,
    and CUDA graphs need static shapes; with the padded layout every buffer
    keeps its shape, so the kernel can be captured into the same graph as the
    forward pass and only the buffer contents change between replays. Chess
    has at most 218 legal moves, so 256 always fits.

Both compute in fp32 whatever the logits' dtype (fp16 under AMP), and a row
with no legal moves comes out as all zeros rather than NaN: with every entry
at -inf, the max is -inf and x - max would be -inf - (-inf).
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import triton
import triton.language as tl

from engine import ACTION_SIZE

DENSE_BLOCK = triton.next_power_of_2(ACTION_SIZE)  # 8192
GATHER_BLOCK = 256                                 # >= 218, the most legal moves a chess position has


@triton.jit
def _masked_softmax_dense_kernel(
    logits_ptr, mask_ptr, out_ptr,
    n_cols, stride_logits, stride_mask, stride_out,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    in_row = offs < n_cols

    x = tl.load(logits_ptr + row * stride_logits + offs, mask=in_row, other=float("-inf"))
    x = x.to(tl.float32)
    legal = tl.load(mask_ptr + row * stride_mask + offs, mask=in_row, other=0) != 0
    x = tl.where(legal, x, float("-inf"))

    m = tl.max(x, axis=0)
    m = tl.where(m == float("-inf"), 0.0, m)   # no legal entry: keep exp() at 0, not NaN
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = e / tl.where(s > 0, s, 1.0)
    tl.store(out_ptr + row * stride_out + offs, p, mask=in_row)


@triton.jit
def _masked_softmax_gather_kernel(
    logits_ptr, idx_ptr, count_ptr, out_ptr,
    stride_logits, stride_idx, stride_out,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    n = tl.load(count_ptr + row)
    valid = offs < n

    # Padding slots are never dereferenced: both loads are masked by count.
    idx = tl.load(idx_ptr + row * stride_idx + offs, mask=valid, other=0).to(tl.int32)
    x = tl.load(logits_ptr + row * stride_logits + idx, mask=valid, other=float("-inf"))
    x = x.to(tl.float32)

    m = tl.max(x, axis=0)
    m = tl.where(m == float("-inf"), 0.0, m)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = e / tl.where(s > 0, s, 1.0)
    # Write the whole row, padding included, so slots past count are 0 rather
    # than whatever the buffer held before.
    tl.store(out_ptr + row * stride_out + offs, p)


def _check_logits(logits: torch.Tensor) -> None:
    if not logits.is_cuda:
        raise ValueError("logits must be on a CUDA device")
    if logits.dim() != 2 or logits.shape[1] != ACTION_SIZE:
        raise ValueError(f"logits must be (B, {ACTION_SIZE}), got {tuple(logits.shape)}")
    if logits.stride(1) != 1:
        raise ValueError("logits rows must be contiguous")
    if logits.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported logits dtype {logits.dtype}")


def masked_softmax_dense(logits: torch.Tensor, mask: torch.Tensor,
                         out: Optional[torch.Tensor] = None, num_warps: int = 8) -> torch.Tensor:
    """v1. logits (B, 4672) fp32/fp16/bf16, mask (B, 4672) bool or uint8 ->
    probs (B, 4672) fp32, 0 at illegal entries."""
    _check_logits(logits)
    if mask.shape != logits.shape or mask.device != logits.device or mask.stride(1) != 1:
        raise ValueError("mask must match logits' shape and device, with contiguous rows")
    if mask.dtype == torch.bool:
        mask = mask.view(torch.uint8)   # same bytes; Triton reads it as u8
    elif mask.dtype != torch.uint8:
        raise ValueError(f"mask must be bool or uint8, got {mask.dtype}")
    B = logits.shape[0]
    if out is None:
        out = torch.empty((B, ACTION_SIZE), device=logits.device, dtype=torch.float32)
    if B:
        _masked_softmax_dense_kernel[(B,)](
            logits, mask, out,
            ACTION_SIZE, logits.stride(0), mask.stride(0), out.stride(0),
            BLOCK=DENSE_BLOCK, num_warps=num_warps,
        )
    return out


def masked_softmax_gather(logits: torch.Tensor, idx: torch.Tensor, count: torch.Tensor,
                          out: Optional[torch.Tensor] = None, num_warps: int = 2) -> torch.Tensor:
    """v2. logits (B, 4672), idx (B, 256) int16/int32, count (B,) int32 ->
    out (B, 256) fp32 where out[i, :count[i]] are the priors of idx[i, :count[i]]
    in that order and the rest is 0. Pass a preallocated out to keep the call
    allocation-free, e.g. inside a captured CUDA graph."""
    _check_logits(logits)
    B = logits.shape[0]
    if idx.shape != (B, GATHER_BLOCK) or idx.dtype not in (torch.int16, torch.int32) or idx.stride(1) != 1:
        raise ValueError(f"idx must be (B, {GATHER_BLOCK}) int16/int32 with contiguous rows")
    if count.shape != (B,) or count.dtype != torch.int32:
        raise ValueError("count must be (B,) int32")
    if idx.device != logits.device or count.device != logits.device:
        raise ValueError("idx and count must be on logits' device")
    if out is None:
        out = torch.empty((B, GATHER_BLOCK), device=logits.device, dtype=torch.float32)
    if B:
        _masked_softmax_gather_kernel[(B,)](
            logits, idx, count, out,
            logits.stride(0), idx.stride(0), out.stride(0),
            BLOCK=GATHER_BLOCK, num_warps=num_warps,
        )
    return out


def pack_legal(legal_lists: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Lay out per-position legal action indices for masked_softmax_gather:
    idx (B, 256) int16 zero-padded, count (B,) int32."""
    B = len(legal_lists)
    idx = np.zeros((B, GATHER_BLOCK), dtype=np.int16)
    count = np.zeros((B,), dtype=np.int32)
    for i, legal in enumerate(legal_lists):
        n = len(legal)
        if n > GATHER_BLOCK:
            raise ValueError(f"{n} legal moves exceeds {GATHER_BLOCK}")
        idx[i, :n] = legal
        count[i] = n
    return idx, count
