# Models

`demucs-onnx` v0.3.0 publishes 7 ONNX repos under the
[StemSplitio](https://huggingface.co/StemSplitio) Hugging Face org. All
auto-download on first use; nothing else is required.

## At a glance

| Alias | Kind | Stems | fp32 size | fp16weights size | Best for |
|---|---|---|---:|---:|---|
| `htdemucs_ft` | 4-specialist bag | drums, bass, other, vocals | 1.26 GB (4×316 MB) | 660 MB (4×166 MB) | Best SDR. Default. |
| `htdemucs` | Single file | drums, bass, other, vocals | 316 MB | 166 MB | Fastest 4-stem startup, ~30% faster than the bag. |
| `htdemucs_6s` | Single file | drums, bass, other, vocals, **guitar**, **piano** | 258 MB | 136 MB | Need guitar / piano stems. The only ONNX export of the 6-stem variant. |
| `htdemucs_ft_drums` | Single specialist | drums | 316 MB | 166 MB | Drum extraction, beat transcription. |
| `htdemucs_ft_bass` | Single specialist | bass | 316 MB | 166 MB | Bassline isolation, mix rebalancing. |
| `htdemucs_ft_other` | Single specialist | other | 316 MB | 166 MB | Karaoke instrumental (pair with vocals). |
| `htdemucs_ft_vocals` | Single specialist | vocals | 316 MB | 166 MB | **#1 open-source vocal SDR** — vocal removal, acapella, karaoke. |

`fp16weights` files compute in fp32 at runtime — same RAM, same speed,
max abs diff vs fp32 weights is ~6 × 10⁻⁵. Use them.

---

## Quality

Parity vs PyTorch fp32 (random `(1, 2, 343980)` input):

| Repo | max abs diff | tolerance |
|---|---:|---:|
| `htdemucs-ft-drums-onnx` | 1.63 × 10⁻⁴ | 1 × 10⁻³ |
| `htdemucs-ft-bass-onnx`  | 1.42 × 10⁻⁴ | 1 × 10⁻³ |
| `htdemucs-ft-other-onnx` | 1.71 × 10⁻⁴ | 1 × 10⁻³ |
| `htdemucs-ft-vocals-onnx` | 1.55 × 10⁻⁴ | 1 × 10⁻³ |
| `htdemucs-onnx`     | 6.62 × 10⁻⁴ | 1 × 10⁻³ |
| `htdemucs-6s-onnx`  | 2.42 × 10⁻⁴ | 1 × 10⁻³ |

Official MUSDB18-HQ SDR (median across 50 test songs, BSS Eval v4):

| Model | drums | bass | other | vocals |
|---|---:|---:|---:|---:|
| `htdemucs_ft` | **10.11 dB** | **9.7 dB** | **7.0 dB** | **9.19 dB** |
| `htdemucs`    | ~9.5 dB | ~9.0 dB | ~5.5 dB | ~8.8 dB |
| `htdemucs_6s` | ~9.5 dB | ~9.0 dB | ~5.5 dB | ~8.5 dB |

`htdemucs_6s` has lower vocals/other SDR because the model also
predicts guitar and piano — "other" becomes more specific (less of a
catchall), trading vocal-recall for guitar/piano clarity.

Full benchmark across every popular open-source separator:
[StemSplitio/stem-separation-benchmark-2026](https://huggingface.co/datasets/StemSplitio/stem-separation-benchmark-2026).

---

## Performance

Single 7.8 s segment on Apple M4 Pro (CPU EP):

| Mode | Per segment | Per 3-min song | RTF |
|---|---:|---:|---:|
| `separate(model="htdemucs_ft", stems=("vocals",))` | 1.6 s | ~22 s | 0.20 |
| `separate(model="htdemucs_ft")` (bag, 4 stems) | 6.4 s | ~88 s | 0.49 |
| `separate(model="htdemucs")` (single file, 4 stems) | 1.6 s | ~22 s | 0.20 |
| `separate(model="htdemucs_6s")` (single file, 6 stems) | 1.6 s | ~22 s | 0.20 |

CUDA / DirectML / CoreML EPs are typically ≥ 5× faster on real GPUs.

---

## Programmatic listing

```python
from demucs_onnx import list_models
import json
print(json.dumps(list_models(), indent=2))
```

Or via the CLI:

```bash
demucs-onnx list-models
```

---

## Direct download URLs

Every model is downloadable without `huggingface_hub`. The URLs are
stable and CDN-hosted by HF.

```bash
# Single-file 4-stem htdemucs (fp16weights variant).
curl -OL https://huggingface.co/StemSplitio/htdemucs-onnx/resolve/main/htdemucs_fp16weights.onnx

# 6-stem variant.
curl -OL https://huggingface.co/StemSplitio/htdemucs-6s-onnx/resolve/main/htdemucs_6s_fp16weights.onnx

# Vocals specialist (the most-used file).
curl -OL https://huggingface.co/StemSplitio/htdemucs-ft-vocals-onnx/resolve/main/htdemucs_ft_vocals_fp16weights.onnx
```

---

## License

All ONNX files are **MIT-licensed**, matching the original HT-Demucs.
Please cite the original authors if you use them in research; see the
[Skip the infrastructure](index.md#skip-the-infrastructure) section
on the home page for the bibtex.
