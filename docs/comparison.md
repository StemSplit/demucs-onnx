# Comparison

Honest comparison of `demucs-onnx` vs the alternatives, organized by
what you're actually trying to do.

## Vs other ONNX-export attempts for demucs

| Project | Working `.onnx` artifact? | Working inference? | On PyPI? | Status |
|---|---|---|---|---|
| **`demucs-onnx`** *(this)* | ✅ all 4 blockers patched, parity-verified to 1.6 × 10⁻⁴ | ✅ pure numpy + ORT | ✅ PyPI + 7 HF model repos | Maintained. |
| `facebookresearch/demucs` | ❌ none of the 4 blockers fixed | n/a | ✅ (PyTorch only) | Maintained. |
| `lstm-mode/demucs-onnx` (GH fork) | ❌ stuck on STFT complex blocker | n/a | ❌ | Abandoned. |
| Stack Overflow gists | ❌ each stuck on one of the 4 blockers | n/a | ❌ | n/a |
| `mvsep` / Audio Separator GUIs | ✅ but for bundled MDX/UVR, **not htdemucs** | ✅ for MDX | n/a | Maintained. |

If you find a comparable working solution for htdemucs ONNX after this
package was published — [open an issue](https://github.com/StemSplit/demucs-onnx/issues)
so we can update this table.

## Vs Spleeter

| Aspect | `demucs-onnx` | `spleeter` |
|---|---|---|
| Vocal SDR (MUSDB18-HQ median) | **9.19 dB** | 6.9 dB |
| Drums SDR | **10.11 dB** | 6.4 dB |
| Model size | 316 MB (single specialist) | ~100 MB |
| Latency on CPU (3-min song) | ~22 s | ~12 s |
| Latency on GPU | ~5 s on T4 | ~3 s on T4 |
| Dependencies | onnxruntime + numpy + soundfile | TensorFlow + ffmpeg |
| Maintained? | Yes (htdemucs is from 2023) | Effectively no (2019, last release 2020) |

**When to pick spleeter:** you need TensorFlow integration or you're
fine with 2.3 dB lower vocals SDR for ~2× faster CPU inference.

**When to pick demucs-onnx:** you want SOTA quality. That's what most
people want most of the time.

## Vs raw `demucs` (PyTorch)

| Aspect | `demucs-onnx` | `demucs` (PyTorch) |
|---|---|---|
| Install footprint | ~50 MB | ~2 GB |
| Cold-start time | ~3 s | ~12 s |
| Inference speed on CPU | **1.31× faster** (onnxruntime CPU EP) | baseline |
| Inference speed on GPU | 5-10× faster (CUDA / DML / CoreML EPs) | depends on PyTorch GPU support |
| Memory footprint | ~1.1 GB (specialist) / ~4 GB (bag) | similar |
| Mobile / browser support | ✅ via ORT iOS / Android / Web | ❌ |
| Quality | parity-verified to 1.6 × 10⁻⁴ vs PyTorch fp32 | reference |

**When to pick PyTorch demucs:** you need training, custom losses, or
you already have a PyTorch model-serving pipeline.

**When to pick demucs-onnx:** anything inference-only — production
APIs, mobile apps, browser demos, low-footprint Docker images.

## Vs cloud separation APIs

| Aspect | `demucs-onnx` self-hosted | StemSplit API | Other cloud APIs (LALAL.AI, Moises, etc) |
|---|---|---|---|
| Per-song cost (at scale) | electricity + amortized hardware | ~$0.05 / minute | $0.20-$1.00 / song |
| Latency | sync, on your hardware | ~10 s for a 3-min song | varies |
| Privacy | files stay on your machine | files stay in your StemSplit project | audio uploaded to third party |
| Setup work | `pip install`, write 30 lines of overlap-add | one API key | one API key |
| Quality | identical to htdemucs (the SOTA open-source model) | identical (we use htdemucs_ft) | varies, sometimes proprietary models |

**When to pick a cloud API:** you don't want to manage GPUs or
sub-second latency targets and you trust a third party with the audio.

**When to pick `demucs-onnx`:** privacy-sensitive content, batch
processing at scale (where per-song API cost dominates), or you want
to bundle separation into a mobile app / browser tab.

If you specifically don't want to bundle a 316 MB model but also need
self-hosting flexibility, see the [StemSplit API](https://stemsplit.io/developers)
— same model, same quality, REST endpoint.

## Vs HT-Demucs FT in PyTorch

`htdemucs_ft` is a **bag of 4 specialist models**, one per stem. Each
specialist is the same architecture as `htdemucs` but with weights
fine-tuned to be best at one stem. The bag aggregates outputs with a
one-hot weight matrix (drums-model contributes only to drums, etc), so
the bag's drums output IS the drums specialist's drums output.

This is why `demucs-onnx` ships the 4 specialists as **independent**
ONNX files: you can mix-and-match, run only the one you need (4× faster),
or run all 4 in parallel sessions for the equivalent of the full bag.

| Variant | When to use | Cost |
|---|---|---|
| `model="htdemucs_ft"` | Full bag — best SDR, all 4 stems | 4 sessions, 4× inference cost |
| `model="htdemucs_ft", stems=("vocals",)` | Best SDR, vocals only | 1 session, 1× inference cost |
| `model="vocals"` (alias) | same as above | 1 session, 1× inference cost |
| `model="htdemucs"` | Faster startup, 1 session for all 4 stems | 1 session, slightly lower SDR than the bag |
| `model="htdemucs_6s"` | Need guitar / piano stems | 1 session, 6 stems |

## Vs htdemucs vs htdemucs_ft vs htdemucs_6s

| Aspect | `htdemucs` | `htdemucs_ft` | `htdemucs_6s` |
|---|---|---|---|
| Stems | 4 | 4 | **6** (+ guitar, piano) |
| Distribution | single .onnx file | 4 specialist .onnx files (bag) | single .onnx file |
| fp32 size | 316 MB | 1.26 GB total | 258 MB |
| Vocal SDR (MUSDB18-HQ) | ~8.8 dB | **9.19 dB** | ~8.5 dB |
| Drum SDR | ~9.5 dB | **10.11 dB** | ~9.5 dB |
| Inference cost | 1× | 4× (full bag) | 1× |
| Best for | Mobile apps, browser demos | Production-quality 4-stem | Guitar / piano isolation |

If you need the **best vocal SDR** and don't mind 1.26 GB of model
download: `htdemucs_ft`. Otherwise: `htdemucs` for 4-stem,
`htdemucs_6s` for 6-stem.
