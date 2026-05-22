"""Tiny audio I/O helpers.

We deliberately depend on ``soundfile`` only for decode and encode of
WAV / FLAC / OGG. Optional extras add MP3 encode (``lameenc``) and
high-quality resampling (``soxr``); both are imported lazily so the
base install stays under ~50 MB.

The htdemucs ONNX graph is bound to 44.1 kHz stereo. In v0.2.0 we
transparently:

- Up-mix mono input to stereo (duplicate the single channel).
- Down-mix >2-channel input to the first two channels (unchanged from
  v0.1.0).
- Resample anything that's not 44.1 kHz to 44.1 kHz on the way in, and
  resample the output back to the input's native rate on the way out.

If ``soxr`` is not installed we fall back to a built-in
``scipy.signal.resample_poly`` implementation; if *that's* also missing
we fall back to a numpy-only linear-phase polyphase resampler (slow but
correct). The first two are several orders of magnitude faster than the
naive fallback for typical audio rates.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import soundfile as sf

PathLike = str | Path

log = logging.getLogger("demucs_onnx")

#: The htdemucs model's hard-coded sample rate. Resample to this for
#: inference, resample back to the input rate when writing.
MODEL_SAMPLE_RATE = 44100


def _resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample ``(channels, samples)`` float32 audio from ``src_sr`` -> ``dst_sr``.

    Tries backends in this order: ``soxr`` (C, fastest, highest quality),
    then ``scipy.signal.resample_poly``, then a numpy fallback. The
    numpy fallback is intentionally slow but dependency-free so the
    base install still resamples correctly.
    """
    if src_sr == dst_sr:
        return audio
    if audio.ndim != 2:
        raise ValueError(f"expected (channels, samples), got {audio.shape}")

    try:
        import soxr
        out = soxr.resample(audio.T, src_sr, dst_sr, quality="HQ").T
        return np.ascontiguousarray(out, dtype=np.float32)
    except ImportError:
        pass

    try:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(src_sr, dst_sr)
        up, down = dst_sr // g, src_sr // g
        out = resample_poly(audio, up, down, axis=1).astype(np.float32, copy=False)
        return np.ascontiguousarray(out)
    except ImportError:
        pass

    log.warning(
        "demucs-onnx is resampling with a slow numpy fallback. "
        "Install `soxr` for a 50x speedup: pip install soxr",
    )
    out_len = math.ceil(audio.shape[1] * dst_sr / src_sr)
    src_t = np.arange(audio.shape[1], dtype=np.float64) / src_sr
    dst_t = np.arange(out_len, dtype=np.float64) / dst_sr
    out = np.empty((audio.shape[0], out_len), dtype=np.float32)
    for c in range(audio.shape[0]):
        out[c] = np.interp(dst_t, src_t, audio[c]).astype(np.float32)
    return out


def load_audio(path: PathLike, target_sr: int = MODEL_SAMPLE_RATE,
               ) -> tuple[np.ndarray, int]:
    """Load audio as float32 stereo at ``target_sr``, return ``(audio, native_sr)``.

    Returns the audio at ``target_sr`` (the model's sample rate) for use
    by the inference loop, and the file's *native* sample rate alongside
    so the caller can resample the output back to the original rate
    before writing.

    - Mono inputs are duplicated to L/R.
    - >2-channel inputs are downmixed to the first two channels.
    - Sample rates < 8000 Hz log a quality warning but are still
      processed (the model output will sound rough).
    - Sample rates > 44.1 kHz are silently down-sampled.
    """
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.T
    if audio.shape[0] == 1:
        audio = np.tile(audio, (2, 1))
        log.info("demucs-onnx: mono input upmixed to stereo")
    elif audio.shape[0] > 2:
        audio = audio[:2]
        log.info("demucs-onnx: >2-channel input downmixed to first 2 channels")

    if sr != target_sr:
        if sr < 8000:
            log.warning(
                "demucs-onnx: very low input sample rate (%d Hz) — "
                "output quality will be poor.", sr,
            )
        log.info("demucs-onnx: resampling %d Hz -> %d Hz for inference", sr, target_sr)
        audio_model = _resample_audio(audio, sr, target_sr)
    else:
        audio_model = audio
    return np.ascontiguousarray(audio_model, dtype=np.float32), sr


def resample_to_native(audio: np.ndarray, model_sr: int, native_sr: int,
                       ) -> np.ndarray:
    """Resample model-rate output back to the input file's native rate."""
    return _resample_audio(audio, model_sr, native_sr)


def write_wav(path: PathLike, audio: np.ndarray, sample_rate: int) -> None:
    """Write a ``(channels, samples)`` float32 array as a 16-bit PCM WAV."""
    if audio.ndim != 2:
        raise ValueError(f"expected (channels, samples), got {audio.shape}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.T, sample_rate, subtype="PCM_16")


def write_mp3(path: PathLike, audio: np.ndarray, sample_rate: int,
              bitrate_kbps: int = 192) -> None:
    """Write a ``(channels, samples)`` float32 array as a CBR MP3.

    Requires the ``lameenc`` extra (``pip install demucs-onnx[mp3]``).
    """
    try:
        import lameenc
    except ImportError as exc:
        raise ImportError(
            "MP3 output requires the 'mp3' extra. "
            "Install with: pip install 'demucs-onnx[mp3]'",
        ) from exc

    if audio.ndim != 2 or audio.shape[0] not in (1, 2):
        raise ValueError(f"expected (channels=1|2, samples), got {audio.shape}")
    if not 32 <= bitrate_kbps <= 320:
        raise ValueError(f"bitrate must be in [32, 320] kbps, got {bitrate_kbps}")

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate_kbps)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(audio.shape[0])
    encoder.set_quality(2)

    pcm = (np.clip(audio.T, -1.0, 1.0) * 32767.0).astype(np.int16)
    pcm_bytes = pcm.tobytes()
    mp3_data = encoder.encode(pcm_bytes)
    mp3_data += encoder.flush()
    Path(path).write_bytes(mp3_data)


def write_audio(path: PathLike, audio: np.ndarray, sample_rate: int, *,
                bitrate_kbps: int = 192) -> None:
    """Write audio, dispatching by ``path`` suffix. Defaults to WAV."""
    suffix = Path(path).suffix.lower()
    if suffix == ".mp3":
        write_mp3(path, audio, sample_rate, bitrate_kbps=bitrate_kbps)
    else:
        write_wav(path, audio, sample_rate)
