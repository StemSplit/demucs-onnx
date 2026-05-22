# Install

`demucs-onnx` ships as a single PyPI package with three optional
extras. Pick the install that matches what you want to do.

## Inference only (default)

```bash
pip install demucs-onnx
```

Pulls in:

- `onnxruntime>=1.17` — the ONNX Runtime Python wheel.
  Modern wheels include the CoreML EP on macOS arm64; pick
  `onnxruntime-gpu` instead if you want CUDA on Linux.
- `numpy>=1.24`
- `soundfile>=0.12` — WAV/FLAC/OGG decode + encode via libsndfile.
- `soxr>=0.3` — fast resampler (used when input sample rate != 44.1 kHz).
- `huggingface-hub>=0.24` — model downloads from
  [StemSplitio](https://huggingface.co/StemSplitio).
- `tqdm>=4.65` — progress bar.

Install footprint: ~50 MB. The 316 MB ONNX model downloads lazily on
first use and is cached forever.

## With MP3 output

```bash
pip install "demucs-onnx[mp3]"
```

Adds `lameenc>=1.6` (~50 KB wheel) for `--mp3` output and the
`write_mp3` / `write_audio` Python helpers. No ffmpeg required.

## With the export pipeline

```bash
pip install "demucs-onnx[export]"
```

Adds the PyTorch-side deps you need to call `demucs_onnx.export.export_to_onnx`:

- `torch>=2.4,<2.5`
- `torchaudio>=2.4,<2.5`
- `demucs==4.0.1`
- `onnx>=1.16`
- `onnxscript>=0.1.0`

This is ~2 GB of wheels (mostly PyTorch). Skip it if you only need to
*run* models — the [StemSplitio HF repos](https://huggingface.co/StemSplitio)
already ship the ONNX files.

## Everything

```bash
pip install "demucs-onnx[all]"
# == demucs-onnx[mp3] for now (no other end-user extras yet)
```

## GPU execution providers

`demucs-onnx` auto-detects what's available. To unlock GPU EPs you may
need a non-default `onnxruntime` wheel:

| Platform | Recommended wheel | EP picked by `auto` |
|---|---|---|
| macOS arm64 (M-series) | `onnxruntime>=1.17` (default) | CoreML |
| Linux + NVIDIA | `onnxruntime-gpu` | CUDA |
| Windows + DX12 GPU | `onnxruntime-directml` | DirectML |
| anywhere else | `onnxruntime` | CPU |

```bash
# Linux + NVIDIA
pip install onnxruntime-gpu demucs-onnx

# Windows + DML
pip install onnxruntime-directml demucs-onnx
```

The `providers="auto"` default in `separate()` and the `--providers
auto` CLI flag will pick the best available EP for the host. Force
CPU-only with `--providers cpu` if you don't trust the auto-selection.

## Python version

Supported Python: **3.10, 3.11, 3.12, 3.13**.

## Verifying the install

```bash
python -c "from demucs_onnx import describe_runtime; \
  import json; print(json.dumps(describe_runtime(), indent=2))"
```

Should print:

```json
{
  "system": "Darwin",
  "machine": "arm64",
  "python": "3.13.0",
  "onnxruntime": "1.19.0",
  "available_providers": ["CoreMLExecutionProvider", "AzureExecutionProvider", "CPUExecutionProvider"],
  "in_browser": false
}
```

If `CoreMLExecutionProvider` (macOS arm64), `CUDAExecutionProvider`
(Linux + NVIDIA), or `DmlExecutionProvider` (Windows + DX12) is missing
when you expected it, upgrade `onnxruntime`.
