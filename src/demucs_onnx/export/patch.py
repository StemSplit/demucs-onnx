"""Apply all four htdemucs ONNX-export patches in one call.

This is the orchestration layer that ties together the focused fixes in
:mod:`demucs_onnx.export.stft`, :mod:`~demucs_onnx.export.segment`,
:mod:`~demucs_onnx.export.pos_embed`, and :mod:`~demucs_onnx.export.mha`.

Most users should call :func:`~demucs_onnx.export.export_to_onnx`
directly, which calls this internally. Reach for
:func:`patch_htdemucs_for_onnx` only when you want to keep the patched
PyTorch model around to run alternative tracers (``torch.fx``,
``torch.export``, custom shape-inference, etc).
"""
from __future__ import annotations

import math
import types
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mha import onnx_friendly_mha_forward
from .pos_embed import disable_random_pos_shift
from .segment import coerce_segment_to_float
from .stft import RealISTFT, RealSTFT


def patch_htdemucs_for_onnx(model: nn.Module) -> nn.Module:
    """Mutate an htdemucs sub-model in place so it has no complex tensors and
    no Python-dynamic code paths.

    Returns the same model so calls can be chained.

    The four patches applied:

    1. ``model.segment``: ``Fraction`` → ``float``
       (:func:`demucs_onnx.export.coerce_segment_to_float`)
    2. ``CrossTransformerEncoder._get_pos_embedding``: drops ``random.randrange``
       (:func:`demucs_onnx.export.disable_random_pos_shift`)
    3. ``nn.MultiheadAttention.forward``: replaced with a primitive-only impl
       (:func:`demucs_onnx.export.onnx_friendly_mha_forward`)
    4. ``model._spec`` / ``_ispec`` / ``_magnitude`` / ``_mask``: rewritten to
       thread a real ``(B, C, 2, F, T)`` tensor instead of complex tensors,
       backed by :class:`~demucs_onnx.export.RealSTFT` and
       :class:`~demucs_onnx.export.RealISTFT`.
    """
    coerce_segment_to_float(model)
    disable_random_pos_shift(model)

    # Install the manual MHA forward on every nn.MultiheadAttention instance.
    for m in model.modules():
        if isinstance(m, nn.MultiheadAttention):
            m.forward = types.MethodType(onnx_friendly_mha_forward, m)

    # Real STFT/iSTFT replacements. The originals are the methods _spec /
    # _ispec / _magnitude / _mask on the model itself; we swap all four.
    n_fft = 4096
    hop_length = n_fft // 4
    real_stft = RealSTFT(n_fft, hop_length)
    real_istft = RealISTFT(n_fft, hop_length)
    # Move kernels onto the model so they migrate cleanly with .to(device).
    model.real_stft = real_stft
    model.real_istft = real_istft

    def _spec_real(self_: Any, x: torch.Tensor) -> torch.Tensor:
        # Mirrors HTDemucs._spec but emits (B, C, 2, F, T) real tensors.
        hl = self_.hop_length
        nfft = self_.nfft
        if hl != nfft // 4:
            raise AssertionError(f"unexpected hop {hl} for nfft {nfft}")
        le = math.ceil(x.shape[-1] / hl)
        pad = hl // 2 * 3
        x = F.pad(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")
        z = self_.real_stft(x)[..., :-1, :]  # drop the Nyquist bin
        if z.shape[-1] != le + 4:
            raise AssertionError((z.shape, x.shape, le))
        return z[..., 2: 2 + le]

    def _ispec_real(self_: Any, z: torch.Tensor, length: int = 0,
                    scale: int = 0) -> torch.Tensor:
        hl = self_.hop_length // (4 ** scale)
        z = F.pad(z, (0, 0, 0, 1))   # restore the Nyquist bin we dropped
        z = F.pad(z, (2, 2))         # symmetric pad on time axis
        pad = hl // 2 * 3
        le = hl * math.ceil(length / hl) + 2 * pad
        x = self_.real_istft(z, length=le)
        return x[..., pad: pad + length]

    def _magnitude_real(self_: Any, z: torch.Tensor) -> torch.Tensor:
        # cac=True path. Original cac=True flow does:
        #   m = view_as_real(z_complex).permute(0,1,4,2,3) -> (B, C, 2, F, T)
        #   m = m.reshape(B, C*2, F, T)
        # Our z is already (B, C, 2, F, T) real, so just reshape.
        B, C, two, Fr, T = z.shape
        if two != 2:
            raise AssertionError(f"expected 2 real channels, got {two}")
        return z.reshape(B, C * two, Fr, T)

    def _mask_real(self_: Any, z: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # cac=True path. Original:
        #   B,S,C,Fr,T = m.shape
        #   out = m.view(B,S,-1,2,Fr,T).permute(0,1,2,4,5,3)  -> (..,Fr,T,2)
        #   out = view_as_complex(out)                         -> (..,Fr,T) complex
        # Our equivalent stays real: (B, S, C', 2, F, T).
        B, S, C, Fr, T = m.shape
        return m.view(B, S, C // 2, 2, Fr, T)

    model._spec = types.MethodType(_spec_real, model)
    model._ispec = types.MethodType(_ispec_real, model)
    model._magnitude = types.MethodType(_magnitude_real, model)
    model._mask = types.MethodType(_mask_real, model)

    # Force eval + cpu (avoids some MPS-roundtrip branches in demucs).
    model.eval()
    model.to("cpu")
    return model
