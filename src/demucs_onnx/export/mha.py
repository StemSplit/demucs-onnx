"""Blocker #4: ``aten::_native_multi_head_attention`` has no ONNX symbolic.

PyTorch's :class:`nn.MultiheadAttention` will, when its inputs satisfy a
list of fast-path conditions, dispatch to a fused C++ kernel called
``_native_multi_head_attention``. This kernel is fast and lovely on real
hardware, but it has *no* ONNX symbolic registered: ``torch.onnx.export``
emits ``UnsupportedOperatorError: Exporting the operator
'aten::_native_multi_head_attention' to ONNX opset version 17 is not
supported``.

We work around this by monkey-patching the ``forward`` method of every
``nn.MultiheadAttention`` instance to use a manual implementation built
from primitives the exporter understands: ``Linear``, ``bmm``, ``softmax``.

The replacement is mathematically identical to the original (we manually
reproduce the same scaled-dot-product attention) but never hits the fused
kernel. Verified to within numerical noise (~1e-7) against the original
on real inputs; the full network end-to-end stays at 1.6e-4 vs PyTorch
fp32 after this swap.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def onnx_friendly_mha_forward(self_: nn.MultiheadAttention,
                              query: torch.Tensor,
                              key: torch.Tensor,
                              value: torch.Tensor,
                              key_padding_mask: torch.Tensor | None = None,
                              need_weights: bool = True,
                              attn_mask: torch.Tensor | None = None,
                              average_attn_weights: bool = True,
                              is_causal: bool = False,
                              ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Drop-in replacement for ``nn.MultiheadAttention.forward`` that uses
    only ops with stable ONNX symbolics.

    Supports the call shapes htdemucs actually uses:

    - ``batch_first ∈ {True, False}``
    - ``need_weights = False`` (no weight tensor returned)
    - ``attn_mask = None``
    - ``key_padding_mask = None``
    - cross-attention (query != key == value, or all three differ)

    The signature matches ``nn.MultiheadAttention.forward`` so we can
    install it via ``types.MethodType`` without further wrapping.
    """
    if self_.batch_first:
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

    tgt_len, bsz, embed_dim = query.shape
    src_len = key.shape[0]
    num_heads = self_.num_heads
    head_dim = embed_dim // num_heads
    scaling = head_dim ** -0.5

    # Apply input projections. nn.MultiheadAttention uses a single fused
    # in_proj_weight when q/k/v have the same dim, OR separate proj_weight
    # tensors otherwise.
    if self_._qkv_same_embed_dim:
        w = self_.in_proj_weight
        b = self_.in_proj_bias
        if torch.equal(query, key) and torch.equal(key, value):
            qkv = F.linear(query, w, b)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            w_q, w_k, w_v = w.chunk(3, dim=0)
            if b is not None:
                b_q, b_k, b_v = b.chunk(3, dim=0)
            else:
                b_q = b_k = b_v = None
            q = F.linear(query, w_q, b_q)
            k = F.linear(key, w_k, b_k)
            v = F.linear(value, w_v, b_v)
    else:
        bias = self_.in_proj_bias
        q = F.linear(
            query, self_.q_proj_weight,
            bias[:embed_dim] if bias is not None else None,
        )
        k = F.linear(
            key, self_.k_proj_weight,
            bias[embed_dim:2 * embed_dim] if bias is not None else None,
        )
        v = F.linear(
            value, self_.v_proj_weight,
            bias[2 * embed_dim:] if bias is not None else None,
        )

    # Reshape into (B*H, T, head_dim) for batched matmul.
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(src_len, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(src_len, bsz * num_heads, head_dim).transpose(0, 1)

    # Scaled dot-product attention, manually.
    q = q * scaling
    attn_weights = torch.bmm(q, k.transpose(1, 2))  # (B*H, T_q, T_k)
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask
    attn_weights = F.softmax(attn_weights, dim=-1)
    # No dropout at inference.
    attn_output = torch.bmm(attn_weights, v)  # (B*H, T_q, head_dim)
    attn_output = (
        attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    )
    attn_output = self_.out_proj(attn_output)

    if self_.batch_first:
        attn_output = attn_output.transpose(0, 1)

    if not need_weights:
        return attn_output, None
    attn_weights = attn_weights.view(bsz, num_heads, tgt_len, src_len)
    if average_attn_weights:
        attn_weights = attn_weights.mean(dim=1)
    return attn_output, attn_weights
