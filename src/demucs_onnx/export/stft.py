"""Blocker #1: replace complex-valued STFT/iSTFT with conv1d real ops.

HT-Demucs is "complex-as-channels" (``cac=True``) — internally it converts
the complex STFT output to a real ``(B, C*2, F, T)`` tensor at the input,
and re-converts a real ``(B, S, C*2, F, T)`` prediction back to complex
just to feed iSTFT at the output. So the complex tensor is never actually
*used* as a complex tensor by the network proper; it's just a notational
artefact of using ``torch.stft(return_complex=True)``.

We bypass the complex round-trip entirely by emitting two real channels
directly from a 1-D convolution, with sin/cos DFT kernels chosen to match
``torch.stft(window=hann, n_fft=4096, hop_length=1024, normalized=True,
center=True)``. The same scheme runs in reverse with ``ConvTranspose1d``
plus an overlap-add window-squared envelope for the inverse.

Verified to within 5e-6 max abs diff against ``torch.stft`` on real audio.
The full network is verified to within 1.6e-4 vs the original PyTorch model.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_stft_kernels(n_fft: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(cos, sin)`` DFT kernels of shape ``(n_fft//2 + 1, 1, n_fft)``.

    The kernels reproduce ``torch.stft(window=hann, win_length=n_fft,
    n_fft=n_fft, normalized=True, center=True)`` with hop = ``n_fft // 4``.

    Computation is done in float64 to keep the high-frequency bins precise,
    then cast to float32 for the actual conv weights.
    """
    n_bins = n_fft // 2 + 1
    n = torch.arange(n_fft, dtype=torch.float64)
    window = torch.hann_window(n_fft, periodic=True, dtype=torch.float64)
    norm = 1.0 / math.sqrt(n_fft)  # matches normalized=True

    k = torch.arange(n_bins, dtype=torch.float64).unsqueeze(1)  # (F, 1)
    angles = 2 * math.pi * k * n.unsqueeze(0) / n_fft  # (F, N)

    cos = (window * torch.cos(angles)) * norm
    sin = (window * -torch.sin(angles)) * norm  # negative sign for forward STFT
    return cos.float().unsqueeze(1), sin.float().unsqueeze(1)


class RealSTFT(nn.Module):
    """ONNX-exportable STFT that emits real/imag as two channel groups.

    Matches the demucs ``spectro(x, n_fft, hop_length, pad=0)`` API but
    returns a real tensor of shape ``(..., 2, F, T)`` instead of a complex
    ``(..., F, T)`` tensor.

    Layout matches what ``view_as_real`` would produce: channel 0 = real,
    channel 1 = imag.
    """

    def __init__(self, n_fft: int = 4096, hop_length: int | None = None) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        cos, sin = make_stft_kernels(n_fft)
        self.register_buffer("cos_kernel", cos, persistent=False)
        self.register_buffer("sin_kernel", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *other, length = x.shape
        x = x.reshape(-1, 1, length)
        pad = self.n_fft // 2
        x = F.pad(x, (pad, pad), mode="reflect")
        real = F.conv1d(x, self.cos_kernel, stride=self.hop_length)  # (BN, F, T)
        imag = F.conv1d(x, self.sin_kernel, stride=self.hop_length)
        out = torch.stack([real, imag], dim=1)  # (BN, 2, F, T)
        _, two, freqs, frames = out.shape
        return out.view(*other, two, freqs, frames)


class RealISTFT(nn.Module):
    """ONNX-exportable inverse STFT, the inverse of :class:`RealSTFT`.

    Input shape: ``(..., 2, F, T)``. Output shape: ``(..., L)``.

    Reconstruction formula (per sample ``n`` of frame OLA):

    .. math::

        x[n] = \\frac{1}{\\sqrt{N}} w[n] \\Big( R_0 + (-1)^n R_{N/2}
              + 2 \\sum_{k=1}^{N/2-1} R_k \\cos\\frac{2\\pi k n}{N}
              - I_k \\sin\\frac{2\\pi k n}{N} \\Big)

    The factor of 2 doubles the contribution from bins ``1..N/2-1`` to
    account for the dropped negative frequencies. DC and Nyquist appear
    once, so we halve their entries. After OLA we divide by the
    ``OLA(window^2)`` envelope to undo the analysis windowing.
    """

    def __init__(self, n_fft: int = 4096, hop_length: int | None = None) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4

        n = torch.arange(n_fft, dtype=torch.float64)
        window = torch.hann_window(n_fft, periodic=True, dtype=torch.float64)
        norm = 1.0 / math.sqrt(n_fft)
        n_bins = n_fft // 2 + 1
        k = torch.arange(n_bins, dtype=torch.float64).unsqueeze(1)
        angles = 2 * math.pi * k * n.unsqueeze(0) / n_fft

        inv_cos = 2.0 * (window * torch.cos(angles)) * norm
        inv_sin = -2.0 * (window * torch.sin(angles)) * norm
        inv_cos[0] *= 0.5    # DC: appears once, not twice
        inv_cos[-1] *= 0.5   # Nyquist: same
        inv_sin[0] *= 0.0    # imag part of DC and Nyquist is zero by construction
        inv_sin[-1] *= 0.0

        self.register_buffer("inv_cos", inv_cos.float().unsqueeze(1), persistent=False)
        self.register_buffer("inv_sin", inv_sin.float().unsqueeze(1), persistent=False)
        # The OLA(window^2) envelope is shape-dependent; cache by (frames, length).
        self._envelope_cache: dict[tuple[int, int], torch.Tensor] = {}

    def _envelope(self, n_frames: int, length: int,
                  device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (n_frames, length)
        if key not in self._envelope_cache:
            win_sq = torch.hann_window(self.n_fft, periodic=True,
                                       dtype=torch.float64) ** 2
            env = torch.zeros(length + self.n_fft, dtype=torch.float64)
            for i in range(n_frames):
                start = i * self.hop_length
                env[start:start + self.n_fft] += win_sq
            pad = self.n_fft // 2
            env = env[pad: pad + length]
            env = torch.clamp(env, min=1e-11)
            self._envelope_cache[key] = env.float().to(device=device, dtype=dtype)
        return self._envelope_cache[key]

    def forward(self, z: torch.Tensor, length: int | None = None) -> torch.Tensor:
        *other, two, freqs, frames = z.shape
        if two != 2:
            raise ValueError(f"expected (...,2,F,T), got {z.shape}")
        z = z.reshape(-1, 2, freqs, frames)
        real = z[:, 0]
        imag = z[:, 1]
        x_cos = F.conv_transpose1d(real, self.inv_cos, stride=self.hop_length)
        x_sin = F.conv_transpose1d(imag, self.inv_sin, stride=self.hop_length)
        x = x_cos + x_sin  # (BN, 1, L_padded)
        pad = self.n_fft // 2
        end = pad + (length if length is not None else x.shape[-1] - 2 * pad)
        x = x[:, 0, pad:end]
        env = self._envelope(frames, x.shape[-1], x.device, x.dtype)
        x = x / env
        return x.view(*other, x.shape[-1])
