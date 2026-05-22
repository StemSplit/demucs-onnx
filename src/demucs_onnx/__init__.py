"""demucs-onnx — run and export HT-Demucs / Demucs as ONNX.

The two flat entry points everyone uses:

    from demucs_onnx import separate
    stems = separate("song.mp3")           # dict[str, np.ndarray]

    from demucs_onnx.export import export_to_onnx
    export_to_onnx("htdemucs_ft", "out/", stem="drums")

That's the whole API. Everything else is a knob on those two.

New in v0.3.0:

- ``separate(model="htdemucs")`` — single-file 4-stem flavor, ~30%
  faster than the FT bag.
- ``separate(model="htdemucs_6s")`` — single-file 6-stem flavor with
  guitar and piano. ``separate_stem(..., stem="guitar")`` works too.
- :func:`prewarm` — pre-download and pre-compile ONNX sessions so the
  first :func:`separate` call doesn't pay the CoreML-graph-compile or
  download tax.
- :mod:`demucs_onnx.browser` — utilities for generating Vite / Webpack
  / esbuild config snippets that get ``onnxruntime-web`` running with
  these models in the browser.
"""
from __future__ import annotations

from ._audio import load_audio, write_audio, write_mp3, write_wav
from .inference import (
    ALL_KNOWN_STEMS,
    DEFAULT_BAG_MODEL,
    SOURCES,
    SessionPool,
    list_models,
    prewarm,
    separate,
    separate_all,
    separate_stem,
    session_pool,
)
from .providers import auto_select_providers, describe_runtime

__version__ = "0.3.2"

__all__ = [
    "ALL_KNOWN_STEMS",
    "DEFAULT_BAG_MODEL",
    "SOURCES",
    "SessionPool",
    "__version__",
    "auto_select_providers",
    "describe_runtime",
    "list_models",
    "load_audio",
    "prewarm",
    "separate",
    "separate_all",
    "separate_stem",
    "session_pool",
    "write_audio",
    "write_mp3",
    "write_wav",
]
