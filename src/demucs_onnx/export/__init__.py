"""Export path: convert a PyTorch demucs/htdemucs checkpoint to ONNX.

The vanilla ``torch.onnx.export`` call will not work on htdemucs as of
PyTorch 2.4. Four blockers, all defeated by patches in this sub-package:

1. **Complex64 STFT outputs** — :mod:`demucs_onnx.export.stft`.
   ``torch.stft(return_complex=True)`` cannot be lowered to ONNX. We
   substitute :class:`~demucs_onnx.export.stft.RealSTFT` (a ``Conv1d`` with
   sin/cos kernels) and a matching :class:`~demucs_onnx.export.stft.RealISTFT`
   inverse.

2. **fractions.Fraction in model.segment** — :mod:`demucs_onnx.export.segment`.
   The original config sets ``segment = Fraction(39, 5)``. ``torch._dynamo``
   refuses to trace through Python ``Fraction`` arithmetic. We coerce it to a
   plain float; the math is identical at inference time.

3. **random.randrange in pos-embedding** — :mod:`demucs_onnx.export.pos_embed`.
   ``CrossTransformerEncoder._get_pos_embedding`` calls ``random.randrange``
   to apply a random shift during *training*. At eval time
   ``sin_random_shift = 0`` so the call is a no-op, but neither dynamo nor
   the legacy exporter can see through ``random``. We monkey-patch the
   method to hardcode ``shift = 0``.

4. **aten::_native_multi_head_attention** — :mod:`demucs_onnx.export.mha`.
   PyTorch's fused MHA C++ kernel has no ONNX symbolic. We replace
   ``nn.MultiheadAttention.forward`` with a drop-in implementation built
   from plain ``Linear`` / ``bmm`` / ``softmax`` ops that the exporter
   already supports.

Use :func:`export_to_onnx` for the high-level "give me an ONNX file"
entry point, or apply :func:`patch_htdemucs_for_onnx` yourself if you
want to keep the patched PyTorch model around.

Importing this sub-package only requires ``torch`` and ``demucs`` if you
actually call one of the public functions — we use a lazy ``__getattr__``
so ``import demucs_onnx.export`` works in inference-only environments
and gives a clear error at first use.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "RealISTFT",
    "RealSTFT",
    "coerce_segment_to_float",
    "disable_random_pos_shift",
    "export_to_onnx",
    "make_stft_kernels",
    "onnx_friendly_mha_forward",
    "patch_htdemucs_for_onnx",
    "verify_onnx_parity",
]

# Map public symbol -> (submodule, attribute) so we can lazy-load.
_LAZY_MAP: dict[str, tuple[str, str]] = {
    "RealISTFT":                 ("demucs_onnx.export.stft", "RealISTFT"),
    "RealSTFT":                  ("demucs_onnx.export.stft", "RealSTFT"),
    "make_stft_kernels":         ("demucs_onnx.export.stft", "make_stft_kernels"),
    "coerce_segment_to_float":   ("demucs_onnx.export.segment", "coerce_segment_to_float"),
    "disable_random_pos_shift":  ("demucs_onnx.export.pos_embed", "disable_random_pos_shift"),
    "onnx_friendly_mha_forward": ("demucs_onnx.export.mha", "onnx_friendly_mha_forward"),
    "patch_htdemucs_for_onnx":   ("demucs_onnx.export.patch", "patch_htdemucs_for_onnx"),
    "export_to_onnx":            ("demucs_onnx.export.exporter", "export_to_onnx"),
    "verify_onnx_parity":        ("demucs_onnx.export.exporter", "verify_onnx_parity"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_MAP:
        mod_path, attr = _LAZY_MAP[name]
        try:
            module = import_module(mod_path)
        except ImportError as exc:
            raise ImportError(
                f"demucs_onnx.export.{name} requires the 'export' extra. "
                "Install with: pip install 'demucs-onnx[export]'\n"
                f"underlying ImportError: {exc}",
            ) from exc
        return getattr(module, attr)
    raise AttributeError(f"module 'demucs_onnx.export' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
