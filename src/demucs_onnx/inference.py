"""Inference path: pure numpy + onnxruntime, no PyTorch.

The exported ONNX graphs have a fixed segment shape ``(1, 2, 343980)``
(7.8 s @ 44.1 kHz stereo) — this is baked into the model at export-time
because dynamic axes blow the graph up by ~3x with no runtime benefit.
We chunk longer audio with quarter-segment overlap-add and a triangular
window. Same scheme demucs uses internally.

The htdemucs_ft specialist bag aggregates 4 per-stem networks via a
one-hot weight matrix (drums-model contributes only to drums, etc), so
running 4 specialists and picking the matching row is bit-exact
identical to running the official PyTorch ``htdemucs_ft`` ensemble. The
``htdemucs`` and ``htdemucs_6s`` single-file variants emit all stems
from a single network and are loaded once per call.
"""
from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort

from ._audio import (
    MODEL_SAMPLE_RATE,
    PathLike,
    load_audio,
    resample_to_native,
    write_audio,
    write_wav,
)
from ._hub import (
    BAG_STEMS,
    MODEL_REGISTRY,
    Precision,
    download_single_model,
    download_stem_model,
    resolve_model_name,
)
from ._hub import (
    list_models as _list_models,
)
from .providers import resolve_providers

# These constants are baked into the exported graph. Re-exporting with a
# different segment length means re-publishing the model and bumping these
# constants in lockstep.
SAMPLE_RATE = MODEL_SAMPLE_RATE
SEGMENT_S = 7.8
N_SAMPLES = int(SEGMENT_S * SAMPLE_RATE)  # 343,980
N_CHANNELS = 2

#: Stem order in the htdemucs_ft / htdemucs models.
SOURCES: tuple[str, ...] = ("drums", "bass", "other", "vocals")

#: Stems available across every model in the registry. Used by the CLI
#: ``--stems`` argument so users can pick guitar/piano on htdemucs_6s.
ALL_KNOWN_STEMS: tuple[str, ...] = (
    "drums", "bass", "other", "vocals", "guitar", "piano",
)

#: Default model when the caller asks for the full 4-stem bag.
DEFAULT_BAG_MODEL = "htdemucs_ft"

log = logging.getLogger("demucs_onnx")


def list_models() -> dict[str, dict[str, str]]:
    """Public re-export so users don't need to dig into ``_hub``.

    Returns ``{alias: {"repo": url, "fp32": url, "fp16weights": url,
    "sources": comma_joined, "kind": "specialist_bag"|"single"|"specialist"}}``.
    """
    return _list_models()


# ---------------------------------------------------------------------------
# Session pool (v0.3.0 — caches sessions across separate*() calls so bag-mode
# users don't repay CoreML graph-compile latency on every call).
# ---------------------------------------------------------------------------


class SessionPool:
    """Cache ``ort.InferenceSession`` objects keyed by (repo, precision, providers).

    Inference sessions are expensive to create — particularly the first
    time on the CoreML EP, which compiles the graph (~5+ min for
    htdemucs on M-series macs). The pool keeps sessions alive across
    successive :func:`separate` calls so subsequent runs reuse the same
    compiled graph.

    The pool is process-local and thread-safe under CPython's GIL for the
    common case (get-or-create). Eviction is manual via :meth:`clear`.
    """

    def __init__(self) -> None:
        self._sessions: dict[tuple[str, str, tuple[str, ...]], ort.InferenceSession] = {}

    def get(self, onnx_path: PathLike, providers: Sequence[str]) -> ort.InferenceSession:
        """Return a session for ``onnx_path``, creating one if absent."""
        key = (str(onnx_path), "", tuple(providers))
        sess = self._sessions.get(key)
        if sess is None:
            sess = _make_session(onnx_path, providers)
            self._sessions[key] = sess
        return sess

    def clear(self) -> None:
        """Drop every cached session. Frees the underlying ORT memory."""
        self._sessions.clear()

    def __len__(self) -> int:
        return len(self._sessions)


_DEFAULT_POOL = SessionPool()


def session_pool() -> SessionPool:
    """Return the process-wide default :class:`SessionPool`."""
    return _DEFAULT_POOL


def prewarm(models: Iterable[str] | None = None, *,
            precision: Precision = "fp32",
            providers: str | Sequence[str] | None = "auto",
            cache_dir: PathLike | None = None,
            token: str | None = None) -> dict[str, Path]:
    """Pre-download + compile sessions for one or more models.

    Useful at app startup or in a server context where the first
    :func:`separate` call would otherwise trigger a multi-minute CoreML
    graph-compile or a 300 MB download. ``models`` defaults to
    ``["htdemucs_ft"]`` (the full 4-stem specialist bag).

    Returns ``{model_or_stem: local_path}``.
    """
    if models is None:
        targets: list[str] = [DEFAULT_BAG_MODEL]
    else:
        targets = [resolve_model_name(m) for m in models]

    onnx_providers = resolve_providers(providers)
    pool = session_pool()
    paths: dict[str, Path] = {}
    for canonical in targets:
        if canonical == "htdemucs_ft":
            for stem in BAG_STEMS:
                p = download_stem_model(
                    stem, cache_dir=cache_dir, token=token, precision=precision,
                )
                pool.get(p, onnx_providers)
                paths[stem] = p
        elif canonical in MODEL_REGISTRY and MODEL_REGISTRY[canonical].kind == "single":
            p = download_single_model(
                canonical, cache_dir=cache_dir, token=token, precision=precision,
            )
            pool.get(p, onnx_providers)
            paths[canonical] = p
        elif canonical.startswith("htdemucs_ft_"):
            stem = canonical.replace("htdemucs_ft_", "")
            p = download_stem_model(
                stem, cache_dir=cache_dir, token=token, precision=precision,
            )
            pool.get(p, onnx_providers)
            paths[stem] = p
        else:
            raise ValueError(
                f"unknown model {canonical!r} for prewarm. "
                f"Known: {sorted(MODEL_REGISTRY)}",
            )
    return paths


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_transition_window(segment: int, overlap_frac: float = 0.25) -> np.ndarray:
    """Triangular fade-in/fade-out window for overlap-add chunked inference."""
    transition = int(segment * overlap_frac)
    window = np.ones(segment, dtype=np.float32)
    fade = np.linspace(0, 1, transition, dtype=np.float32)
    window[:transition] = fade
    window[-transition:] = fade[::-1]
    return window


def _make_session(onnx_path: PathLike,
                  providers: Sequence[str]) -> ort.InferenceSession:
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path), sess_options=sess_opts,
                                providers=list(providers))


def _iter_with_progress(iterable, *, total: int, desc: str, enabled: bool):
    """Wrap ``iterable`` with a tqdm progress bar when ``enabled``.

    Falls back to a plain iterator when tqdm isn't installed or when
    stdout isn't a TTY (the bar would just clutter logs in CI).
    """
    if not enabled:
        return iterable
    if not sys.stdout.isatty():
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="chunk",
                leave=False, dynamic_ncols=True)


def _chunked_separate_specialists(
    sessions: dict[str, ort.InferenceSession], mix: np.ndarray, *,
    verbose: bool = False, progress: bool = True,
) -> dict[str, np.ndarray]:
    """Run overlap-add chunked inference across the htdemucs_ft specialist bag.

    ``sessions`` keys must be a subset of :data:`SOURCES`. Each session runs
    its specialist and we pick the row that matches the specialist's target
    stem (bag aggregation = one-hot, so this is mathematically equivalent to
    the original ensemble).
    """
    if mix.ndim != 2 or mix.shape[0] != N_CHANNELS:
        raise ValueError(f"expected (channels=2, samples), got {mix.shape}")
    for stem in sessions:
        if stem not in SOURCES:
            raise ValueError(f"unknown stem {stem!r}; expected one of {SOURCES}")

    total_len = mix.shape[1]
    overlap = N_SAMPLES // 4
    stride = N_SAMPLES - overlap
    n_chunks = max(1, (total_len + stride - 1) // stride)

    window = _make_transition_window(N_SAMPLES)
    out: dict[str, np.ndarray] = {
        stem: np.zeros((N_CHANNELS, total_len), dtype=np.float32)
        for stem in sessions
    }
    weight = np.zeros(total_len, dtype=np.float32)

    if verbose:
        provs = next(iter(sessions.values())).get_providers()
        print(f"  input:  {total_len:,} samples ({total_len / SAMPLE_RATE:.1f} s)")
        print(f"  chunks: {n_chunks} x {N_SAMPLES:,} samples ({SEGMENT_S} s)")
        print(f"  models: {sorted(sessions)}  on {provs[0]}")

    t0 = time.perf_counter()
    work = _iter_with_progress(
        range(n_chunks), total=n_chunks * len(sessions),
        desc="separating", enabled=progress and not verbose,
    )
    for i in work:
        start = i * stride
        end = min(start + N_SAMPLES, total_len)
        chunk = mix[:, start:end]
        if chunk.shape[1] < N_SAMPLES:
            chunk = np.pad(
                chunk, ((0, 0), (0, N_SAMPLES - chunk.shape[1])), mode="constant",
            )
        x = chunk[np.newaxis, ...].astype(np.float32, copy=False)
        chunk_len = end - start
        w = window[:chunk_len]

        for stem, sess in sessions.items():
            stems = sess.run(["stems"], {"mix": x})[0][0]  # (4, 2, N)
            row = SOURCES.index(stem)
            out[stem][:, start:end] += stems[row, :, :chunk_len] * w

        weight[start:end] += w
        if verbose:
            elapsed = time.perf_counter() - t0
            print(f"    chunk {i + 1}/{n_chunks}: {elapsed:.1f}s elapsed")

    weight = np.maximum(weight, 1e-8)
    for stem in out:
        out[stem] /= weight

    if verbose:
        elapsed = time.perf_counter() - t0
        rtf = elapsed / (total_len / SAMPLE_RATE)
        n_runs = len(sessions) * n_chunks
        print(
            f"  total:  {elapsed:.2f}s (RTF {rtf:.2f}, "
            f"{len(sessions)} models x {n_chunks} chunks = {n_runs} ONNX runs)",
        )
    return out


def _chunked_separate_single(
    session: ort.InferenceSession, sources: Sequence[str], mix: np.ndarray,
    *, wanted: Sequence[str] | None = None,
    verbose: bool = False, progress: bool = True,
) -> dict[str, np.ndarray]:
    """Run overlap-add inference on a single-file multi-stem ONNX model.

    Used for ``htdemucs`` and ``htdemucs_6s``. ``sources`` is the model's
    stem ordering (e.g. ``("drums","bass","other","vocals","guitar","piano")``).
    ``wanted`` filters which stems to materialize in the output dict (the
    model still computes all of them — there's no per-stem-skip in this mode).
    """
    if mix.ndim != 2 or mix.shape[0] != N_CHANNELS:
        raise ValueError(f"expected (channels=2, samples), got {mix.shape}")
    keep = list(wanted) if wanted else list(sources)
    for s in keep:
        if s not in sources:
            raise ValueError(
                f"stem {s!r} not in model sources {tuple(sources)}",
            )

    total_len = mix.shape[1]
    overlap = N_SAMPLES // 4
    stride = N_SAMPLES - overlap
    n_chunks = max(1, (total_len + stride - 1) // stride)

    window = _make_transition_window(N_SAMPLES)
    out: dict[str, np.ndarray] = {
        s: np.zeros((N_CHANNELS, total_len), dtype=np.float32) for s in keep
    }
    weight = np.zeros(total_len, dtype=np.float32)

    if verbose:
        provs = session.get_providers()
        print(f"  input:  {total_len:,} samples ({total_len / SAMPLE_RATE:.1f} s)")
        print(f"  chunks: {n_chunks} x {N_SAMPLES:,} samples ({SEGMENT_S} s)")
        print(f"  stems:  {keep}  on {provs[0]}")

    t0 = time.perf_counter()
    work = _iter_with_progress(
        range(n_chunks), total=n_chunks,
        desc="separating", enabled=progress and not verbose,
    )
    for i in work:
        start = i * stride
        end = min(start + N_SAMPLES, total_len)
        chunk = mix[:, start:end]
        if chunk.shape[1] < N_SAMPLES:
            chunk = np.pad(
                chunk, ((0, 0), (0, N_SAMPLES - chunk.shape[1])), mode="constant",
            )
        x = chunk[np.newaxis, ...].astype(np.float32, copy=False)
        chunk_len = end - start
        w = window[:chunk_len]

        stems = session.run(["stems"], {"mix": x})[0][0]  # (S, 2, N)
        for s in keep:
            row = sources.index(s)
            out[s][:, start:end] += stems[row, :, :chunk_len] * w

        weight[start:end] += w
        if verbose:
            elapsed = time.perf_counter() - t0
            print(f"    chunk {i + 1}/{n_chunks}: {elapsed:.1f}s elapsed")

    weight = np.maximum(weight, 1e-8)
    for s in out:
        out[s] /= weight

    if verbose:
        elapsed = time.perf_counter() - t0
        rtf = elapsed / (total_len / SAMPLE_RATE)
        print(
            f"  total:  {elapsed:.2f}s (RTF {rtf:.2f}, "
            f"{n_chunks} chunks = {n_chunks} ONNX runs)",
        )
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def separate(input: PathLike,
             output_dir: PathLike | None = None,
             *,
             model: str = DEFAULT_BAG_MODEL,
             stems: Iterable[str] | None = None,
             providers: str | Sequence[str] | None = "auto",
             precision: Precision = "fp32",
             cache_dir: PathLike | None = None,
             token: str | None = None,
             verbose: bool = False,
             progress: bool = True,
             output_format: Literal["wav", "mp3"] = "wav",
             bitrate_kbps: int = 192,
             mix_stems: Sequence[str] | None = None,
             mix_output_name: str = "mix",
             ) -> dict[str, np.ndarray]:
    """Run htdemucs ONNX separation on an audio file.

    Args:
        input: Path to an audio file. Any rate ``soundfile`` can decode
            (WAV/FLAC/OGG/MP3 via libsndfile 1.1+). Mono input is
            upmixed to stereo; non-44.1 kHz rates are transparently
            resampled in *and* back out.
        output_dir: If given, write each stem under here. If ``None``,
            results are returned in memory only.
        model: Which model to run. Supported (v0.3.0):

            - ``"htdemucs_ft"`` (default) — full 4-stem specialist bag.
            - ``"htdemucs_ft_<stem>" | "<stem>"`` — single FT specialist.
            - ``"htdemucs"`` — single-file 4-stem model. ~30% faster
              than the FT bag on the same hardware, slightly lower SDR.
            - ``"htdemucs_6s"`` — single-file 6-stem model. Same 4 stems
              plus ``"guitar"`` and ``"piano"``. The only ONNX export of
              the 6-stem variant on the Hub.

        stems: Subset of stems to materialize. For ``htdemucs_ft`` this
            also skips the specialists you don't need (saves time). For
            ``htdemucs`` and ``htdemucs_6s`` the model always computes
            all stems internally; this just filters the returned dict.
        providers: ONNX Runtime execution providers. ``"auto"`` (default)
            picks the best available EP for the host (CoreML on macOS
            arm64, CUDA on Linux+NVIDIA, DML on Windows, CPU otherwise).
            Pass a short alias (``"cpu"``, ``"coreml"``, ``"cuda"``,
            ``"dml"``), an explicit ORT provider name, or a list of
            either to override.
        precision: ``"fp32"`` (default) or ``"fp16weights"``. The latter
            downloads a smaller variant (~1.9x smaller) but is otherwise
            identical at runtime — same RAM, same latency, max abs diff
            vs fp32 is ~6e-5.
        cache_dir: Override the huggingface_hub model cache location.
        token: Hugging Face access token (only needed if you've made the
            model repos private).
        verbose: Print chunk-by-chunk progress to stdout.
        progress: Render a ``tqdm`` progress bar (only when ``verbose``
            is False and stdout is a TTY).
        output_format: ``"wav"`` (default) or ``"mp3"``. MP3 requires
            ``pip install 'demucs-onnx[mp3]'``.
        bitrate_kbps: MP3 bitrate when ``output_format='mp3'``. 32-320,
            default 192.
        mix_stems: Optional list of stem names to additionally sum into
            a single output file (alongside the individual stem files).
            For ``htdemucs_6s`` you can mix any of the 6 stems, e.g.
            ``("drums","bass","guitar","piano")`` for a full backing
            track with guitar and piano kept in.
        mix_output_name: Filename stem for the mixed output (default
            ``"mix"``, becomes ``mix.wav`` / ``mix.mp3``).

    Returns:
        ``{stem_name: numpy.ndarray of shape (channels=2, samples)}`` in
        float32, at the **input file's native sample rate** (auto-resampled).
    """
    audio, native_sr = load_audio(input, target_sr=SAMPLE_RATE)
    onnx_providers = resolve_providers(providers)
    pool = session_pool()

    canonical = resolve_model_name(model)

    if canonical == "htdemucs_ft":
        wanted = list(stems) if stems is not None else list(BAG_STEMS)
        for s in wanted:
            if s not in BAG_STEMS:
                raise ValueError(
                    f"unknown stem {s!r}; htdemucs_ft predicts {BAG_STEMS}",
                )
        model_paths = {
            stem: download_stem_model(
                stem, cache_dir=cache_dir, token=token, precision=precision,
            )
            for stem in wanted
        }
        sessions = {
            stem: pool.get(path, onnx_providers)
            for stem, path in model_paths.items()
        }
        out_model_sr = _chunked_separate_specialists(
            sessions, audio, verbose=verbose, progress=progress,
        )

    elif canonical in MODEL_REGISTRY and MODEL_REGISTRY[canonical].kind == "single":
        info = MODEL_REGISTRY[canonical]
        path = download_single_model(
            canonical, cache_dir=cache_dir, token=token, precision=precision,
        )
        session = pool.get(path, onnx_providers)
        wanted = list(stems) if stems is not None else list(info.sources)
        for s in wanted:
            if s not in info.sources:
                raise ValueError(
                    f"unknown stem {s!r}; {canonical} predicts "
                    f"{tuple(info.sources)}",
                )
        out_model_sr = _chunked_separate_single(
            session, info.sources, audio, wanted=wanted,
            verbose=verbose, progress=progress,
        )

    elif canonical.startswith("htdemucs_ft_"):
        target_stem = canonical.replace("htdemucs_ft_", "")
        if target_stem not in BAG_STEMS:
            raise ValueError(
                f"unknown model {model!r}. Known specialists: "
                f"htdemucs_ft_{{drums,bass,other,vocals}}.",
            )
        if stems is not None and list(stems) != [target_stem]:
            raise ValueError(
                f"model {model!r} only predicts {target_stem!r}; pass "
                f"`model='htdemucs_ft'` or `model='htdemucs'` to get other stems.",
            )
        path = download_stem_model(
            target_stem, cache_dir=cache_dir, token=token, precision=precision,
        )
        sessions = {target_stem: pool.get(path, onnx_providers)}
        out_model_sr = _chunked_separate_specialists(
            sessions, audio, verbose=verbose, progress=progress,
        )

    else:
        raise ValueError(
            f"unknown model {model!r}. Known: {sorted(MODEL_REGISTRY)} "
            f"or any htdemucs_ft specialist (drums/bass/other/vocals).",
        )

    if native_sr != SAMPLE_RATE:
        log.info("demucs-onnx: resampling output %d Hz -> %d Hz", SAMPLE_RATE, native_sr)
        out_native = {
            stem: resample_to_native(arr, SAMPLE_RATE, native_sr)
            for stem, arr in out_model_sr.items()
        }
    else:
        out_native = out_model_sr

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        ext = "mp3" if output_format == "mp3" else "wav"
        for stem, audio_out in out_native.items():
            target = out_path / f"{stem}.{ext}"
            write_audio(target, audio_out, native_sr, bitrate_kbps=bitrate_kbps)
            if verbose:
                print(f"  wrote {target}")

        if mix_stems:
            missing = [s for s in mix_stems if s not in out_native]
            if missing:
                raise ValueError(
                    f"mix_stems references stems we did not produce: {missing}. "
                    f"Available: {sorted(out_native)}",
                )
            mixed = np.sum(
                np.stack([out_native[s] for s in mix_stems], axis=0), axis=0,
            ).astype(np.float32)
            target = out_path / f"{mix_output_name}.{ext}"
            write_audio(target, mixed, native_sr, bitrate_kbps=bitrate_kbps)
            if verbose:
                print(f"  wrote {target} (sum of {list(mix_stems)})")

    return out_native


def separate_all(input: PathLike,
                 output_dir: PathLike | None = None,
                 **kwargs: object) -> dict[str, np.ndarray]:
    """Shorthand for :func:`separate` with the full htdemucs_ft bag.

    Equivalent to ``separate(input, output_dir, model="htdemucs_ft", ...)``.
    For the faster single-file ``htdemucs`` flavor pass ``model="htdemucs"``
    directly to :func:`separate`.
    """
    return separate(input, output_dir, model="htdemucs_ft", **kwargs)  # type: ignore[arg-type]


def separate_stem(input: PathLike,
                  stem: str,
                  output_dir: PathLike | None = None,
                  **kwargs: object) -> np.ndarray:
    """Run one specialist and return only that stem as a numpy array.

    For ``stem`` ∈ {drums, bass, other, vocals} this picks the
    htdemucs_ft specialist (4x faster than the bag). For ``stem`` ∈
    {guitar, piano} this transparently switches to ``htdemucs_6s`` and
    returns the requested row.
    """
    if stem in BAG_STEMS:
        out = separate(input, output_dir, model=f"htdemucs_ft_{stem}", **kwargs)  # type: ignore[arg-type]
        return out[stem]
    if stem in ("guitar", "piano"):
        out = separate(
            input, output_dir, model="htdemucs_6s", stems=[stem], **kwargs,  # type: ignore[arg-type]
        )
        return out[stem]
    raise ValueError(
        f"stem must be one of {(*BAG_STEMS, 'guitar', 'piano')}, got {stem!r}",
    )


__all__ = [
    "ALL_KNOWN_STEMS",
    "DEFAULT_BAG_MODEL",
    "N_CHANNELS",
    "N_SAMPLES",
    "SAMPLE_RATE",
    "SEGMENT_S",
    "SOURCES",
    "SessionPool",
    "list_models",
    "load_audio",
    "prewarm",
    "separate",
    "separate_all",
    "separate_stem",
    "session_pool",
    "write_wav",
]
