"""Hugging Face Hub model registry + download helpers.

The package hosts model repos under the ``StemSplitio`` org. In v0.3.0
the registry covers three families:

**htdemucs-ft specialist bag** (v0.1.0, four files):

    htdemucs-ft-drums-onnx    -> sub-model 0 (drums specialist)
    htdemucs-ft-bass-onnx     -> sub-model 1 (bass specialist)
    htdemucs-ft-other-onnx    -> sub-model 2 (other specialist)
    htdemucs-ft-vocals-onnx   -> sub-model 3 (vocals specialist)
    htdemucs-ft-onnx          -> the full 4-stem bag (all 4 .onnx files)

**htdemucs** (v0.3.0, single file with all 4 stems, ~30% faster than FT):

    htdemucs-onnx             -> one file ``htdemucs.onnx`` (4 stems)

**htdemucs_6s** (v0.3.0, single file with 6 stems incl. guitar + piano):

    htdemucs-6s-onnx          -> one file ``htdemucs_6s.onnx`` (6 stems)

Each repo ships **two** ONNX variants:

- The fp32 weights file (default, ~316 MB for FT specialists, ~150 MB
  for the htdemucs single-file model, ~150 MB for htdemucs_6s).
- A ``*_fp16weights.onnx`` variant (~half the download).
  The graph still computes in fp32 at runtime, so latency and RAM are
  identical to fp32; only the *download* and *on-disk* size shrink.
  Max abs diff vs fp32 weights: ~6e-5.

Each repo is parity-verified against the original PyTorch model to
< 1e-3 max abs diff.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ORG = "StemSplitio"

#: ``Literal`` of the supported weight-precision variants.
Precision = Literal["fp32", "fp16weights"]


@dataclass(frozen=True)
class ModelInfo:
    """Metadata describing one ONNX model available on the Hub.

    Attributes:
        canonical: Canonical model name (e.g. ``"htdemucs_ft"``).
        repo: Hugging Face Hub ``org/repo`` id, e.g.
            ``"StemSplitio/htdemucs-onnx"``.
        sources: Tuple of stems the model predicts, in output order.
        kind: ``"specialist_bag"`` (htdemucs_ft — 4 separate ONNX files
            each predicting one stem), ``"single"`` (one ONNX file with
            all stems predicted), or ``"placeholder"`` (the bag-repo
            alias that re-uses the specialist files).
        filename_fp32: The fp32 ONNX filename hosted on the repo. For
            specialist bags this is the per-stem name and ``{stem}``
            placeholder is filled at lookup time.
        filename_fp16weights: Same for the fp16weights variant.
    """

    canonical: str
    repo: str
    sources: tuple[str, ...]
    kind: Literal["specialist_bag", "single", "placeholder"]
    filename_fp32: str
    filename_fp16weights: str


#: The 4 stems of the htdemucs_ft specialist bag.
BAG_STEMS = ("drums", "bass", "other", "vocals")
BAG_REPO_ID = f"{ORG}/htdemucs-ft-onnx"

# Per-specialist repo ids for htdemucs_ft.
SPECIALIST_REPOS: dict[str, str] = {
    "drums":  f"{ORG}/htdemucs-ft-drums-onnx",
    "bass":   f"{ORG}/htdemucs-ft-bass-onnx",
    "other":  f"{ORG}/htdemucs-ft-other-onnx",
    "vocals": f"{ORG}/htdemucs-ft-vocals-onnx",
}

#: All v0.3.0 models, keyed by canonical name.
MODEL_REGISTRY: dict[str, ModelInfo] = {
    "htdemucs_ft": ModelInfo(
        canonical="htdemucs_ft",
        repo=BAG_REPO_ID,
        sources=BAG_STEMS,
        kind="specialist_bag",
        filename_fp32="htdemucs_ft_{stem}.onnx",
        filename_fp16weights="htdemucs_ft_{stem}_fp16weights.onnx",
    ),
    "htdemucs": ModelInfo(
        canonical="htdemucs",
        repo=f"{ORG}/htdemucs-onnx",
        sources=BAG_STEMS,
        kind="single",
        filename_fp32="htdemucs.onnx",
        filename_fp16weights="htdemucs_fp16weights.onnx",
    ),
    "htdemucs_6s": ModelInfo(
        canonical="htdemucs_6s",
        repo=f"{ORG}/htdemucs-6s-onnx",
        sources=("drums", "bass", "other", "vocals", "guitar", "piano"),
        kind="single",
        filename_fp32="htdemucs_6s.onnx",
        filename_fp16weights="htdemucs_6s_fp16weights.onnx",
    ),
}

#: Legacy per-specialist canonical names (back-compat with v0.1/0.2 API).
MODEL_REPOS: dict[str, str] = {
    f"htdemucs_ft_{stem}": repo for stem, repo in SPECIALIST_REPOS.items()
}

# Convenience aliases — what users actually type.
ALIASES: dict[str, str] = {
    "drums":      "htdemucs_ft_drums",
    "bass":       "htdemucs_ft_bass",
    "other":      "htdemucs_ft_other",
    "vocals":     "htdemucs_ft_vocals",
    "ft":         "htdemucs_ft",
    "ft_drums":   "htdemucs_ft_drums",
    "ft_bass":    "htdemucs_ft_bass",
    "ft_other":   "htdemucs_ft_other",
    "ft_vocals":  "htdemucs_ft_vocals",
    "6s":         "htdemucs_6s",
    "htdemucs6s": "htdemucs_6s",
}


def resolve_model_name(name: str) -> str:
    """Map a user-facing alias (``"drums"``, ``"6s"``) to a canonical key."""
    return ALIASES.get(name, name)


def stem_model_filename(stem: str, precision: Precision = "fp32") -> str:
    """Build the per-specialist on-disk ONNX filename for ``(stem, precision)``.

    Only meaningful for the htdemucs_ft specialist family. Single-file
    models (``htdemucs``, ``htdemucs_6s``) use :func:`model_filename`.
    """
    if precision == "fp32":
        return f"htdemucs_ft_{stem}.onnx"
    if precision == "fp16weights":
        return f"htdemucs_ft_{stem}_fp16weights.onnx"
    raise ValueError(
        f"unknown precision {precision!r}; expected 'fp32' or 'fp16weights'",
    )


def model_filename(canonical: str, precision: Precision = "fp32",
                   stem: str | None = None) -> str:
    """Resolve the on-disk filename for any model in the registry.

    For ``htdemucs_ft`` you must pass ``stem``. For single-file models
    (``htdemucs``, ``htdemucs_6s``) the ``stem`` argument is ignored.
    """
    info = MODEL_REGISTRY.get(canonical)
    if info is None:
        raise ValueError(
            f"unknown canonical model {canonical!r}. "
            f"Known: {list(MODEL_REGISTRY)}",
        )
    template = info.filename_fp32 if precision == "fp32" else info.filename_fp16weights
    if "{stem}" in template:
        if stem is None:
            raise ValueError(
                f"model {canonical!r} is a specialist bag — pass `stem=`.",
            )
        return template.format(stem=stem)
    return template


def download_stem_model(name: str, *, cache_dir: str | Path | None = None,
                        token: str | None = None,
                        precision: Precision = "fp32") -> Path:
    """Download a single specialist ONNX file from htdemucs_ft.

    Cached by ``huggingface_hub``. For non-specialist models
    (``htdemucs``, ``htdemucs_6s``) use :func:`download_single_model`.
    """
    canonical = resolve_model_name(name)
    if canonical not in MODEL_REPOS:
        choices = ", ".join(sorted(MODEL_REPOS))
        raise ValueError(
            f"unknown specialist {name!r}. Known specialists: {choices}. "
            f"Pass `htdemucs_ft` for the full 4-stem bag, or `htdemucs` "
            f"/ `htdemucs_6s` for single-file models.",
        )
    stem = canonical.replace("htdemucs_ft_", "")
    return _hub_download(
        MODEL_REPOS[canonical], stem_model_filename(stem, precision),
        cache_dir=cache_dir, token=token,
    )


def download_single_model(name: str, *, cache_dir: str | Path | None = None,
                          token: str | None = None,
                          precision: Precision = "fp32") -> Path:
    """Download a single-file multi-stem ONNX model (``htdemucs`` /
    ``htdemucs_6s``).

    For ``htdemucs_ft`` see :func:`download_stem_model` and
    :func:`download_bag` instead.
    """
    canonical = resolve_model_name(name)
    info = MODEL_REGISTRY.get(canonical)
    if info is None or info.kind != "single":
        raise ValueError(
            f"{name!r} is not a single-file model. "
            f"Available single-file models: htdemucs, htdemucs_6s.",
        )
    return _hub_download(
        info.repo, model_filename(canonical, precision),
        cache_dir=cache_dir, token=token,
    )


def download_bag(*, cache_dir: str | Path | None = None,
                 token: str | None = None,
                 precision: Precision = "fp32") -> dict[str, Path]:
    """Download all 4 specialist ONNX files for the full htdemucs_ft bag."""
    paths: dict[str, Path] = {}
    for stem in BAG_STEMS:
        paths[stem] = download_stem_model(
            stem, cache_dir=cache_dir, token=token, precision=precision,
        )
    return paths


def _hub_download(repo_id: str, filename: str,
                  *, cache_dir: str | Path | None = None,
                  token: str | None = None) -> Path:
    """Thin wrapper around ``huggingface_hub.hf_hub_download`` with helpful
    errors. Imported lazily so ``import demucs_onnx`` stays cheap."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import (
        HfHubHTTPError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
    )

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(cache_dir) if cache_dir else None,
            token=token,
        )
    except RepositoryNotFoundError as exc:
        raise RuntimeError(
            f"Hugging Face repo {repo_id!r} not found. "
            "If you're behind auth, set HF_TOKEN.",
        ) from exc
    except LocalEntryNotFoundError as exc:
        raise RuntimeError(
            f"Could not download {filename} from {repo_id!r} and no local "
            "cache available. Check your internet connection.",
        ) from exc
    except HfHubHTTPError as exc:
        raise RuntimeError(
            f"Hub download for {repo_id}/{filename} failed: {exc}",
        ) from exc
    return Path(path)


def list_models() -> dict[str, dict[str, str]]:
    """Return ``{alias: {variant: hf_url, ...}}`` for every supported model.

    Each entry maps a variant name (``"repo"`` / ``"fp32"`` /
    ``"fp16weights"`` / ``"sources"``) to a discoverable string — direct
    file URL where possible, or the repo URL for ``"repo"`` and a
    comma-joined stem list for ``"sources"``.
    """
    out: dict[str, dict[str, str]] = {}
    for canonical, info in MODEL_REGISTRY.items():
        if info.kind == "specialist_bag":
            # Emit the full bag and each specialist separately, the way the
            # CLI used to in v0.2.0.
            out[canonical] = {
                "repo":        f"https://huggingface.co/{info.repo}",
                "fp32":        f"https://huggingface.co/{info.repo}",
                "fp16weights": f"https://huggingface.co/{info.repo}",
                "sources":     ",".join(info.sources),
                "kind":        info.kind,
            }
            for stem in info.sources:
                stem_canonical = f"htdemucs_ft_{stem}"
                stem_repo = MODEL_REPOS[stem_canonical]
                out[stem_canonical] = {
                    "repo":        f"https://huggingface.co/{stem_repo}",
                    "fp32":        f"https://huggingface.co/{stem_repo}/resolve/main/{stem_model_filename(stem, 'fp32')}",
                    "fp16weights": f"https://huggingface.co/{stem_repo}/resolve/main/{stem_model_filename(stem, 'fp16weights')}",
                    "sources":     stem,
                    "kind":        "specialist",
                }
        else:
            out[canonical] = {
                "repo":        f"https://huggingface.co/{info.repo}",
                "fp32":        f"https://huggingface.co/{info.repo}/resolve/main/{info.filename_fp32}",
                "fp16weights": f"https://huggingface.co/{info.repo}/resolve/main/{info.filename_fp16weights}",
                "sources":     ",".join(info.sources),
                "kind":        info.kind,
            }
    return out
