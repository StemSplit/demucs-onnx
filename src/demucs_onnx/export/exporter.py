"""High-level ``export_to_onnx`` entry point.

Loads a demucs/htdemucs checkpoint, applies the four ONNX-friendly
patches, and exports a parity-verified ``.onnx`` file. By default it
also runs a numerical parity check against the original PyTorch model
*before* writing the ONNX file — never publish a numerically wrong model.

Usage:

    from demucs_onnx.export import export_to_onnx

    # All 4 specialists from the htdemucs_ft bag, into out/:
    export_to_onnx("htdemucs_ft", "out/")

    # Just the drums specialist:
    export_to_onnx("htdemucs_ft", "drums.onnx", stem="drums")

    # A custom checkpoint file:
    export_to_onnx(Path("my_finetune.th"), "my_finetune.onnx")
"""
from __future__ import annotations

import copy
import time
from collections.abc import Iterable
from pathlib import Path

import torch

from .patch import patch_htdemucs_for_onnx

# Default segment length baked into the exported ONNX graph.
SAMPLE_RATE = 44100
SEGMENT_S = 7.8
N_SAMPLES = int(SEGMENT_S * SAMPLE_RATE)  # 343,980

STEM_TO_INDEX = {"drums": 0, "bass": 1, "other": 2, "vocals": 3}
#: Default stems for the 4-stem htdemucs family (htdemucs, htdemucs_ft).
DEFAULT_4STEM_SOURCES = ("drums", "bass", "other", "vocals")
#: Default stems for the 6-stem htdemucs_6s variant (4 + guitar + piano).
DEFAULT_6STEM_SOURCES = ("drums", "bass", "other", "vocals", "guitar", "piano")
DEFAULT_PARITY_TOLERANCE = 1e-3


def export_to_onnx(checkpoint: str | Path,
                   output: str | Path,
                   *,
                   stem: str | None = None,
                   stems: Iterable[str] | None = None,
                   opset: int = 17,
                   parity_check: bool = True,
                   parity_tolerance: float = DEFAULT_PARITY_TOLERANCE,
                   sample_rate: int = SAMPLE_RATE,
                   segment_seconds: float = SEGMENT_S,
                   verbose: bool = True,
                   ) -> dict[str, Path]:
    """Export a demucs/htdemucs checkpoint to one or more ONNX files.

    Args:
        checkpoint: Either a name accepted by ``demucs.pretrained.get_model``
            (e.g. ``"htdemucs_ft"``, ``"htdemucs"``, ``"mdx_extra"``) or a
            path to a local ``.th`` checkpoint file.

            Bag models (``htdemucs_ft``) are 4-model ensembles. By default we
            export every specialist; pass ``stem`` or ``stems`` to filter.

        output: For a single model, a ``.onnx`` file path. For a bag, a
            directory that will be populated with one file per stem.
        stem: Optional single stem name (drums / bass / other / vocals).
            Only valid when ``checkpoint`` is a bag.
        stems: Optional iterable of stem names to export from a bag.
            Default: all 4.
        opset: ONNX opset to target. 17 is the lowest with native STFT
            support; we don't actually use ONNX's STFT op (we replace it
            with conv1d) but staying ≥17 is a good baseline for ORT EPs.
        parity_check: When True (default), run the patched and the original
            PyTorch models on the same dummy input and abort if their
            outputs differ by more than ``parity_tolerance``. Set to False
            only if you know what you're doing — exporting an unverified
            model is the most common way to ship a silently broken pipeline.
        parity_tolerance: Max allowed abs diff between the patched
            PyTorch model and the original. Default 1e-3 matches the
            tolerance the StemSplit team uses for the published HF models.
        sample_rate: Sample rate the exported model expects. Changing this
            requires a matching change to the host inference code; demucs
            is trained at 44100 Hz.
        segment_seconds: Segment length baked into the exported graph.
        verbose: Print progress to stdout.

    Returns:
        ``{stem_name: output_path}``, or for a single non-bag model
        ``{"model": output_path}``.
    """
    bag, sub_models, sources = _load_checkpoint(checkpoint, verbose=verbose)

    # Resolve which sub-models to export.
    is_bag = sub_models is not None
    # The htdemucs_ft "specialist bag" has 4 sub-models (one per stem); each
    # sub-model nominally predicts all 4 stems but only one row is the real
    # prediction. By contrast, the plain `htdemucs` and `htdemucs_6s` models
    # arrive wrapped in a BagOfModels with a single sub-model that predicts
    # every stem row meaningfully. We treat that single-submodel case as a
    # single-file export, not as a 4-specialist export.
    is_specialist_bag = is_bag and len(sub_models) > 1
    if not is_bag or not is_specialist_bag:
        if stem is not None or stems is not None:
            raise ValueError(
                f"checkpoint {checkpoint!r} is a single model "
                f"(sources={sources!r}); cannot pass `stem` or `stems`. "
                "Export the whole model and pick the row at inference time.",
            )
        # Treat as one model with multi-stem output. Use the checkpoint name
        # as the filename stem (e.g. "htdemucs.onnx", "htdemucs_6s.onnx").
        ckpt_name = (
            Path(checkpoint).stem
            if isinstance(checkpoint, Path) or "/" in str(checkpoint)
            else str(checkpoint)
        )
        targets: list[tuple[str, int]] = [(ckpt_name, 0)]
    else:
        wanted: list[str]
        if stem is not None and stems is not None:
            raise ValueError("pass either `stem` OR `stems`, not both.")
        if stem is not None:
            wanted = [stem]
        elif stems is not None:
            wanted = list(stems)
        else:
            wanted = list(STEM_TO_INDEX)
        for s in wanted:
            if s not in STEM_TO_INDEX:
                raise ValueError(
                    f"unknown stem {s!r}; expected one of {list(STEM_TO_INDEX)}",
                )
        targets = [(s, STEM_TO_INDEX[s]) for s in wanted]

    n_samples = int(segment_seconds * sample_rate)
    out_paths: dict[str, Path] = {}

    out_root = Path(output)
    if is_specialist_bag and len(targets) > 1:
        out_root.mkdir(parents=True, exist_ok=True)
    else:
        out_root.parent.mkdir(parents=True, exist_ok=True)

    for stem_name, idx in targets:
        # Pick the right output filename.
        if is_specialist_bag and len(targets) > 1:
            file_path = out_root / f"htdemucs_ft_{stem_name}.onnx"
        elif is_specialist_bag:
            # Single-stem export: use `output` directly if it ends in .onnx,
            # otherwise treat it as a directory.
            if out_root.suffix.lower() == ".onnx":
                file_path = out_root
            else:
                out_root.mkdir(parents=True, exist_ok=True)
                file_path = out_root / f"htdemucs_ft_{stem_name}.onnx"
        else:
            file_path = out_root if out_root.suffix.lower() == ".onnx" else out_root / f"{stem_name}.onnx"

        if verbose:
            print(f"\n=== Exporting {stem_name} (index {idx}) -> {file_path} ===")

        if is_specialist_bag:
            original = sub_models[idx].eval().to("cpu")
            parity_idx: int | None = idx
        elif is_bag:
            original = sub_models[0].eval().to("cpu")
            parity_idx = None
        else:
            original = bag.eval().to("cpu")
            parity_idx = None
        if parity_check:
            _verify_pytorch_parity(
                original, n_samples=n_samples, bag_index=parity_idx, stem=stem_name,
                tolerance=parity_tolerance, verbose=verbose,
            )
        patched = patch_htdemucs_for_onnx(copy.deepcopy(original))

        _export_one(patched, file_path, n_samples=n_samples,
                    opset=opset, verbose=verbose)
        _onnx_check(file_path, verbose=verbose)

        if parity_check:
            _verify_onnx_parity(original, file_path, n_samples=n_samples,
                                bag_index=parity_idx, stem=stem_name,
                                tolerance=parity_tolerance, verbose=verbose)

        out_paths[stem_name] = file_path

    return out_paths


def verify_onnx_parity(checkpoint: str | Path, onnx_path: str | Path, *,
                       stem: str | None = None,
                       tolerance: float = DEFAULT_PARITY_TOLERANCE,
                       sample_rate: int = SAMPLE_RATE,
                       segment_seconds: float = SEGMENT_S,
                       verbose: bool = True) -> float:
    """Compare an exported ONNX model against the original PyTorch checkpoint.

    Returns the max abs diff. Raises ``AssertionError`` if it exceeds
    ``tolerance``.
    """
    bag, sub_models, _sources = _load_checkpoint(checkpoint, verbose=verbose)
    if sub_models is not None:
        if stem is None:
            raise ValueError(
                f"checkpoint {checkpoint!r} is a bag; pass `stem=` to pick a sub-model.",
            )
        idx = STEM_TO_INDEX[stem]
        original = sub_models[idx].eval().to("cpu")
        bag_index = idx
        stem_name = stem
    else:
        original = bag.eval().to("cpu")
        bag_index = STEM_TO_INDEX.get(stem or "", 0)
        stem_name = stem or "model"
    n_samples = int(segment_seconds * sample_rate)
    return _verify_onnx_parity(
        original, Path(onnx_path), n_samples=n_samples, bag_index=bag_index,
        stem=stem_name, tolerance=tolerance, verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _load_checkpoint(checkpoint: str | Path, *, verbose: bool):
    """Load a demucs checkpoint by name or path.

    Returns ``(bag_or_model, sub_models_or_None, sources_list)``. If the
    loaded object is a bag of sub-models, ``sub_models`` is the list of
    children. Otherwise the same object is returned and ``sub_models`` is
    None (single-model export).
    """
    from demucs.pretrained import get_model

    if isinstance(checkpoint, Path) or (
        isinstance(checkpoint, str) and "/" in checkpoint
    ):
        # Treat as a file path. demucs's loader doesn't have a clean
        # "load arbitrary .th" entry point, so we use torch.load directly.
        if verbose:
            print(f"Loading checkpoint from path: {checkpoint}")
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        if hasattr(state, "models"):  # already a Bag-shaped object
            return state, list(state.models), list(state.sources)
        if hasattr(state, "sources"):
            return state, None, list(state.sources)
        raise RuntimeError(
            f"Don't know how to handle checkpoint object: {type(state).__name__}. "
            "Expected a demucs Bag or single Model with a `.sources` attribute.",
        )

    # Otherwise assume a name from demucs.pretrained.get_model.
    if verbose:
        print(f"Loading pretrained model: {checkpoint}")
    obj = get_model(str(checkpoint))
    if hasattr(obj, "models"):
        return obj, list(obj.models), list(obj.sources)
    return obj, None, list(getattr(obj, "sources", []))


def _verify_pytorch_parity(original: torch.nn.Module, *,
                           n_samples: int, bag_index: int | None, stem: str,
                           tolerance: float, verbose: bool) -> float:
    """Confirm the patched PyTorch model is numerically equivalent to the
    original BEFORE we attempt the ONNX export.

    If ``bag_index`` is None we compare the full multi-stem output;
    otherwise we pick only ``out[0, bag_index]`` (specialist export).
    """
    patched = patch_htdemucs_for_onnx(copy.deepcopy(original))
    torch.manual_seed(0)
    dummy = torch.randn(1, 2, n_samples, dtype=torch.float32)
    with torch.no_grad():
        out_orig = original(dummy)
        out_patch = patched(dummy)
    if bag_index is None:
        diff = float((out_orig - out_patch).abs().max())
    else:
        diff = float((out_orig[0, bag_index] - out_patch[0, bag_index]).abs().max())
    if verbose:
        print(f"  PyTorch parity ({stem}): max abs diff {diff:.6f}", end=" ")
    if diff > tolerance:
        if verbose:
            print(f"FAIL (>{tolerance})")
        raise AssertionError(
            f"patched PyTorch model diverged from original on {stem}: "
            f"max abs diff {diff:.6f} > {tolerance}",
        )
    if verbose:
        print("PASS")
    return diff


def _export_one(patched: torch.nn.Module, out_path: Path, *,
                n_samples: int, opset: int, verbose: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 2, n_samples, dtype=torch.float32)
    t0 = time.perf_counter()
    with torch.no_grad():
        torch.onnx.export(
            patched, dummy, str(out_path),
            opset_version=opset,
            input_names=["mix"], output_names=["stems"],
            do_constant_folding=True, export_params=True,
            dynamo=False,  # legacy exporter: dynamo trips on too much demucs Python
        )
    elapsed = time.perf_counter() - t0
    size_mb = out_path.stat().st_size / 1e6
    if verbose:
        print(f"  exported in {elapsed:.1f}s ({size_mb:.1f} MB)")


def _onnx_check(out_path: Path, *, verbose: bool) -> None:
    import onnx

    model = onnx.load(str(out_path))
    onnx.checker.check_model(model)
    if verbose:
        print(f"  onnx.checker: PASS  ({len(model.graph.node)} nodes)")


def _verify_onnx_parity(original: torch.nn.Module, onnx_path: Path, *,
                        n_samples: int, bag_index: int | None, stem: str,
                        tolerance: float, verbose: bool) -> float:
    import numpy as np
    import onnxruntime as ort

    torch.manual_seed(0)
    dummy = torch.randn(1, 2, n_samples, dtype=torch.float32)
    with torch.no_grad():
        out_torch = original(dummy).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out_onnx = sess.run(["stems"], {"mix": dummy.numpy()})[0]
    if bag_index is None:
        diff = float(np.abs(out_torch - out_onnx).max())
    else:
        diff = float(np.abs(out_torch[0, bag_index] - out_onnx[0, bag_index]).max())
    if verbose:
        print(f"  ONNX parity ({stem}): max abs diff {diff:.6f}", end=" ")
    if diff > tolerance:
        if verbose:
            print(f"FAIL (>{tolerance})")
        raise AssertionError(
            f"exported ONNX diverged from PyTorch on {stem}: "
            f"max abs diff {diff:.6f} > {tolerance}",
        )
    if verbose:
        print("PASS")
    return diff


# Re-export sources list for callers that want to know stem ordering.
__all__ = [
    "N_SAMPLES",
    "SAMPLE_RATE",
    "SEGMENT_S",
    "STEM_TO_INDEX",
    "export_to_onnx",
    "verify_onnx_parity",
]
