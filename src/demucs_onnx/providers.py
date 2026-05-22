"""Auto-select an :mod:`onnxruntime` execution-provider list for this host.

In v0.1.0 callers had to know which short alias (``"cpu"`` / ``"coreml"`` /
``"cuda"`` / ``"dml"``) to pass to :func:`~demucs_onnx.separate`. In
v0.2.0 the default is ``providers="auto"`` and we pick the right one
based on the platform, hardware, and which providers ``onnxruntime``
was actually built with.

The detection is intentionally conservative — we only return a GPU EP if
that EP is in :func:`onnxruntime.get_available_providers()`. That means
on macOS arm64 you have to install ``onnxruntime`` 1.17+ (CoreML EP is
shipped by default in modern wheels). On Linux NVIDIA you need
``onnxruntime-gpu``. On Windows DML you need ``onnxruntime-directml``.

If the detection thinks you *should* have a GPU EP but ``onnxruntime``
doesn't expose it, we warn once per session with a fix-it hint.
"""
from __future__ import annotations

import platform
import sys
import warnings
from collections.abc import Sequence
from functools import lru_cache

import onnxruntime as ort

__all__ = [
    "auto_select_providers",
    "describe_runtime",
    "in_browser",
    "resolve_providers",
]

#: Short aliases the CLI / Python API accept. Each maps to the ordered
#: provider list we pass to :class:`onnxruntime.InferenceSession`. The CPU
#: provider always appears last as the fallback, in keeping with the ORT
#: convention.
PROVIDER_ALIASES: dict[str, list[str]] = {
    "cpu":    ["CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    "cuda":   ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "dml":    ["DmlExecutionProvider", "CPUExecutionProvider"],
    "wasm":   ["WasmExecutionProvider", "CPUExecutionProvider"],
}


def in_browser() -> bool:
    """Detect Pyodide / browser environments (deferred full support to v0.3)."""
    return "pyodide" in sys.modules or "js" in sys.modules


@lru_cache(maxsize=1)
def _available_providers() -> tuple[str, ...]:
    """Cache ``ort.get_available_providers()`` — calling it spins ORT up briefly."""
    return tuple(ort.get_available_providers())


def describe_runtime() -> dict[str, object]:
    """Return a flat dict describing the runtime environment.

    Useful for debugging — print it from your code if ``auto`` selects
    something surprising.
    """
    return {
        "system":            platform.system(),
        "machine":           platform.machine(),
        "python":            platform.python_version(),
        "onnxruntime":       ort.__version__,
        "available_providers": list(_available_providers()),
        "in_browser":        in_browser(),
    }


_WARNED_NO_GPU = False


def _warn_no_gpu_once(reason: str, hint: str) -> None:
    """Emit a one-shot warning when auto-detection lands on CPU on a host
    we expected to have a GPU EP."""
    global _WARNED_NO_GPU
    if _WARNED_NO_GPU:
        return
    _WARNED_NO_GPU = True
    warnings.warn(
        f"demucs-onnx: {reason} Falling back to CPU. {hint}",
        RuntimeWarning,
        stacklevel=3,
    )


def auto_select_providers() -> list[str]:
    """Return the best ORT provider list for this host.

    Decision tree (first match wins):

    1. **Browser** (``pyodide`` / ``js``) — WASM. Browser support is not
       wired through the inference path yet; this is a forward-looking
       hook so callers can branch off it. Real browser support lands in
       v0.3.
    2. **macOS arm64** with CoreML EP available — ``CoreMLExecutionProvider``.
       If CoreML EP is missing we warn once with the upgrade hint.
    3. **Linux** with CUDA EP available — ``CUDAExecutionProvider``. If
       we're on Linux x86_64 but the GPU EP is missing we *don't* warn
       (it's normal to run inference on a CPU-only Linux box).
    4. **Windows** with DML EP available — ``DmlExecutionProvider``.
    5. **Fallback** — ``CPUExecutionProvider``.
    """
    if in_browser():
        return ["WasmExecutionProvider", "CPUExecutionProvider"]

    avail = _available_providers()
    system = platform.system()
    machine = platform.machine().lower()

    if system == "Darwin":
        if machine in ("arm64", "aarch64"):
            if "CoreMLExecutionProvider" in avail:
                return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            _warn_no_gpu_once(
                "running on Apple Silicon but CoreMLExecutionProvider is not "
                "exposed by onnxruntime.",
                "Upgrade with `pip install -U 'onnxruntime>=1.17'` — modern "
                "wheels include CoreML EP by default.",
            )
        return ["CPUExecutionProvider"]

    if system == "Linux":
        if "CUDAExecutionProvider" in avail:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    if system == "Windows":
        if "DmlExecutionProvider" in avail:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in avail:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    return ["CPUExecutionProvider"]


def resolve_providers(providers: str | Sequence[str] | None) -> list[str]:
    """Translate a user-facing providers spec into a concrete ORT list.

    Accepts:

    - ``None`` — same as ``"auto"`` (back-compat: v0.1.0 also accepted
      ``None`` but mapped it to ``["CPUExecutionProvider"]``; we now
      auto-detect because that's almost always what callers wanted).
    - ``"auto"`` — call :func:`auto_select_providers`.
    - A short alias (``"cpu" | "coreml" | "cuda" | "dml" | "wasm"``).
    - An explicit ORT provider name (``"CUDAExecutionProvider"``).
    - A list of any of the above. Entries are flattened and de-duped
      while preserving order.
    """
    if providers is None or providers == "auto":
        return auto_select_providers()

    if isinstance(providers, str):
        if providers in PROVIDER_ALIASES:
            return list(PROVIDER_ALIASES[providers])
        return [providers]

    expanded: list[str] = []
    for entry in providers:
        if entry == "auto":
            expanded.extend(auto_select_providers())
        elif entry in PROVIDER_ALIASES:
            expanded.extend(PROVIDER_ALIASES[entry])
        else:
            expanded.append(entry)

    seen: set[str] = set()
    deduped: list[str] = []
    for p in expanded:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped
