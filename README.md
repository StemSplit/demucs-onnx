<p align="center">
  <a href="https://pypi.org/project/demucs-onnx/">
    <img src="https://raw.githubusercontent.com/StemSplit/demucs-onnx/main/assets/banner.svg" alt="demucs-onnx — HT-Demucs FT exported to ONNX for iOS, Android, and the browser" width="100%">
  </a>
</p>

# demucs-onnx

[![PyPI version](https://img.shields.io/pypi/v/demucs-onnx.svg?label=pypi&color=blue)](https://pypi.org/project/demucs-onnx/)
[![Python versions](https://img.shields.io/pypi/pyversions/demucs-onnx.svg)](https://pypi.org/project/demucs-onnx/)
[![License: MIT](https://img.shields.io/pypi/l/demucs-onnx.svg?color=green)](https://github.com/StemSplit/demucs-onnx/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/demucs-onnx.svg?color=informational)](https://pypistats.org/packages/demucs-onnx)
[![Docs](https://img.shields.io/badge/docs-stemsplit.github.io-blueviolet)](https://stemsplit.github.io/demucs-onnx/)
[![GitHub stars](https://img.shields.io/github/stars/StemSplit/demucs-onnx?style=social)](https://github.com/StemSplit/demucs-onnx)

**Run and export HT-Demucs / Demucs music source separation as ONNX —
Python, browser, iOS, and Android.** Pure numpy + onnxruntime at
inference (no PyTorch), a one-call export pipeline that fixes the four
known blockers in `torch.onnx.export`, pre-built ONNX models on Hugging
Face, and a copy-pasteable `onnxruntime-web` path for the browser.
Powers the [StemSplit](https://stemsplit.io) production stack.

```bash
pip install 'demucs-onnx[mp3]'
demucs-onnx separate song.mp3 out/ --karaoke --mp3
# writes out/karaoke.mp3  (drums + bass + other, vocals removed)
```

## Quick links

- [Documentation](https://stemsplit.github.io/demucs-onnx/) — full guides, API reference, and walkthroughs
- [GitHub](https://github.com/StemSplit/demucs-onnx) — source, issues, and discussions
- [Hugging Face models](https://huggingface.co/StemSplitio) — 7 pre-built ONNX repos
- [CLI reference](https://stemsplit.github.io/demucs-onnx/cli/) — every flag, every example
- [Browser guide](https://stemsplit.github.io/demucs-onnx/browser/) — `onnxruntime-web` integration
- [Hosted API](https://stemsplit.io/developers) — skip the infra, get stems via HTTP
- [`stemsplit-python`](https://pypi.org/project/stemsplit-python/) — official Python SDK for the StemSplit API (`pip install stemsplit-python`)

## Why use this instead of `X`?

Snapshot 2026-05-21. Distilled from
[`COMPETITIVE_LANDSCAPE.md`](https://github.com/StemSplit/demucs-onnx/blob/main/COMPETITIVE_LANDSCAPE.md).

| Want to … | `demucs-onnx` | `facebookresearch/demucs` | `nomadkaraoke/audio-separator` | `deezer/spleeter` | `sevagh/demucs.onnx` |
|---|:---:|:---:|:---:|:---:|:---:|
| `pip install` it | ✅ | ✅ | ✅ | ✅ | ❌ (C++ build) |
| Run **HT-Demucs as ONNX** with no PyTorch at inference | ✅ | ❌ | ❌ (uses torch for Demucs) | ❌ (TensorFlow, not Demucs) | ✅ (C++ only) |
| Pre-built models on Hugging Face | ✅ 7 repos | — | — | — | ❌ |
| Browser / `onnxruntime-web` support | ✅ scaffold + bundler configs | ❌ | ❌ | ❌ | ❌ |
| Mobile-friendly (iOS / Android via ORT) | ✅ | ❌ | ❌ | ❌ | ⚠️ build yourself |
| 6-stem (drums, bass, other, vocals, **guitar**, **piano**) | ✅ ONNX | ✅ PyTorch | ⚠️ via Demucs/PyTorch | ❌ | ⚠️ all variants |
| Karaoke / mix-stems CLI shortcut | ✅ `--karaoke --mp3` | ❌ | ⚠️ scriptable | ❌ | ❌ |
| Auto-resample any sample rate / mono | ✅ | ❌ (44.1 kHz stereo only) | ✅ | ❌ | ⚠️ |
| Export your own Demucs checkpoint to ONNX | ✅ one call, parity-verified | ❌ | ❌ | ❌ | ❌ (uses pre-export) |

`demucs-onnx` is the only pip-installable Python package that runs
HT-Demucs as ONNX cross-platform with no PyTorch dependency at
inference. If you find a comparable working solution after this package
was published, please [open an issue](https://github.com/StemSplit/demucs-onnx/issues)
so we can update this table.

## Quick start

### Install

```bash
pip install demucs-onnx                # inference only — onnxruntime + numpy + soundfile + soxr
pip install "demucs-onnx[mp3]"         # adds the lameenc encoder for --mp3 output
pip install "demucs-onnx[export]"      # adds torch + demucs for the export pipeline
```

### Separate (Python)

```python
from demucs_onnx import separate, separate_stem

# Full 4-stem bag (default). Auto-downloads from HF on first run, auto
# picks the best execution provider for this host (CoreML / CUDA / DML).
stems = separate("song.mp3")
# stems: {"drums": ndarray (2, S), "bass": ..., "other": ..., "vocals": ...}

# Just one stem — 4× faster, 75% less RAM, model size 316 MB instead of 1.26 GB.
vocals = separate_stem("song.mp3", "vocals")

# Smaller download (166 MB per stem instead of 316 MB), no runtime cost.
stems = separate("song.mp3", precision="fp16weights")

# Write straight to MP3, including a karaoke instrumental mix.
separate(
    "song.mp3", "stems/",
    output_format="mp3", bitrate_kbps=192,
    mix_stems=("drums", "bass", "other"), mix_output_name="karaoke",
)
```

### Separate (CLI)

```bash
# Killer feature — one command -> karaoke.mp3 ready to share.
demucs-onnx separate song.mp3 stems/ --karaoke --mp3

# All 4 stems, auto provider (CoreML on macOS, CUDA on Linux, etc).
demucs-onnx separate song.mp3 stems/

# 6-stem mode with guitar + piano.
demucs-onnx separate song.mp3 stems/ --model htdemucs_6s

# Single specialist mode — 4x faster than the bag.
demucs-onnx separate song.mp3 stems/ --stem vocals

# Smaller download (1.91x), same runtime cost.
demucs-onnx separate song.mp3 stems/ --small

# Custom mix-down: write one file that's vocals + drums only.
demucs-onnx separate song.mp3 stems/ --mix-stems vocals,drums --mp3

# Explicit provider override (auto is the default).
demucs-onnx separate song.mp3 stems/ --providers coreml
demucs-onnx separate song.mp3 stems/ --providers cuda
demucs-onnx separate song.mp3 stems/ --providers dml

demucs-onnx list-models
```

Models auto-download from the Hugging Face Hub on first run and are
cached forever. Inputs at any sample rate (8 kHz – 192 kHz, mono or
stereo) are auto-resampled in for inference and back out so the file
you get matches the file you put in.

### Browser

```bash
# Print a copy-pasteable onnxruntime-web config snippet for any major bundler.
demucs-onnx browser-config --bundler vite      # or webpack, esbuild, next, rollup

# Scaffold a runnable demo into a directory (zero build, just python -m http.server).
demucs-onnx browser-demo /tmp/demo
demucs-onnx browser-demo /tmp/demo --react     # Vite + React + TypeScript variant
```

See the [browser guide](https://stemsplit.github.io/demucs-onnx/browser/)
or the in-tree [`examples/browser/`](https://github.com/StemSplit/demucs-onnx/tree/main/examples/browser)
and [`examples/browser-react/`](https://github.com/StemSplit/demucs-onnx/tree/main/examples/browser-react)
demos.

### Export (Python)

```python
from demucs_onnx.export import export_to_onnx
from pathlib import Path

# Export every specialist of htdemucs_ft into out/ as 4 .onnx files.
paths = export_to_onnx("htdemucs_ft", "out/")
# paths == {"drums": Path("out/htdemucs_ft_drums.onnx"), "bass": ..., ...}

# Export just the vocals specialist to a single file.
export_to_onnx("htdemucs_ft", "vocals.onnx", stem="vocals")

# Export your own fine-tuned checkpoint.
export_to_onnx(Path("my_finetune.th"), "my_finetune.onnx")
```

### Export (CLI)

```bash
demucs-onnx export htdemucs_ft out/                    # all 4 specialists
demucs-onnx export htdemucs_ft drums.onnx --stem drums # one stem -> single file
demucs-onnx export htdemucs_ft out/ --opset 17         # change opset
demucs-onnx export htdemucs_ft out/ --no-parity-check  # advanced (don't)
```

### Mobile / web (after exporting)

```swift
// iOS / Swift, ORT 1.17+
import onnxruntime_objc
let opts = try ORTSessionOptions()
try opts.appendCoreMLExecutionProvider(with: ORTCoreMLExecutionProviderOptions())
let session = try ORTSession(env: env,
                              modelPath: bundle.path(forResource: "htdemucs_ft_vocals",
                                                     ofType: "onnx")!,
                              sessionOptions: opts)
```

```javascript
// Browser / web, onnxruntime-web
import * as ort from "onnxruntime-web";
const session = await ort.InferenceSession.create("htdemucs_ft_vocals.onnx", {
  executionProviders: ["wasm"],
  graphOptimizationLevel: "all",
});
const tensor = new ort.Tensor("float32", audioBuffer, [1, 2, 343980]);
const out = await session.run({ mix: tensor });
```

## What's new in v0.3 — browser, htdemucs/htdemucs_6s, docs site

- **`htdemucs_6s` flavor** — single-file 6-stem ONNX model with **guitar** and **piano** in addition to the standard 4. The only ONNX export of the 6-stem variant on the Hub.
  ([`StemSplitio/htdemucs-6s-onnx`](https://huggingface.co/StemSplitio/htdemucs-6s-onnx))
- **`htdemucs` flavor** — single-file 4-stem ONNX model. ~30% faster than the FT bag (1 session vs 4), slightly lower SDR.
  ([`StemSplitio/htdemucs-onnx`](https://huggingface.co/StemSplitio/htdemucs-onnx))
- **Browser support** via `onnxruntime-web` — copy-pasteable bundler configs and a `demucs-onnx browser-demo PATH` CLI that scaffolds a zero-build vanilla HTML/JS demo or a Vite + React + TS demo.
- **`SessionPool` + `prewarm()`** — process-wide session cache and a one-shot prewarm so the first `separate()` call doesn't pay the CoreML graph-compile tax. Reusing sessions across `htdemucs_ft` bag calls is now automatic.
- **Docs site** at [stemsplit.github.io/demucs-onnx](https://stemsplit.github.io/demucs-onnx/) with the canonical 4-blocker write-up, browser guide, model registry, and autogenerated API reference via `mkdocstrings[python]`.

See [`CHANGELOG.md`](https://github.com/StemSplit/demucs-onnx/blob/main/CHANGELOG.md)
for the full diff vs v0.2.0 and the v0.2.0 UX bundle (`--karaoke`,
`--mix-stems`, `--mp3`, `providers="auto"`, fp16-weight downloads,
auto-resampling, progress bars).

## Why this package exists

For the entire history of the [`demucs`](https://github.com/facebookresearch/demucs)
repo (2021 – 2026), **nobody on PyPI has shipped working ONNX export
tooling** for HT-Demucs. Searching GitHub turns up half a dozen
abandoned forks, all stuck on one of four blockers, all without a
working `.onnx` file to show for it. The official demucs README has no
mention of ONNX. We solved it.

This package ships a pure-numpy + onnxruntime inference path that runs
the official HT-Demucs FT models with no PyTorch dependency (install
footprint drops from ~2 GB to ~50 MB), a one-call export pipeline
(`export_to_onnx("htdemucs_ft", ...)`) that applies all four patches
and parity-checks the output against PyTorch fp32, and the same patches
as independent grep-able modules (`stft.py`, `mha.py`, `pos_embed.py`,
`segment.py`) so you can debug your own exports of related
architectures.

| Want to … | Use this |
|---|---|
| Run htdemucs_ft on CPU / mobile / web with no PyTorch | `from demucs_onnx import separate` |
| Convert your own demucs checkpoint to ONNX | `from demucs_onnx.export import export_to_onnx` |
| Skip the infrastructure entirely | The hosted [StemSplit API](https://stemsplit.io/developers) |

## Used by / integrations

- **[StemSplit](https://stemsplit.io)** — production stack for
  [Vocal Remover](https://stemsplit.io/vocal-remover),
  [Karaoke Maker](https://stemsplit.io/karaoke-maker),
  [Acapella Maker](https://stemsplit.io/acapella-maker), and the
  [hosted API](https://stemsplit.io/developers). The same ONNX
  models on Hugging Face power the production endpoints — what you
  install is what we run.
- **Pre-built ONNX models on Hugging Face** at
  [`StemSplitio`](https://huggingface.co/StemSplitio) — 7 ONNX repos
  + 4 PyTorch source repos, all MIT-licensed and parity-verified.

If you ship `demucs-onnx` in a project and would like to be listed
here, please
[open an issue](https://github.com/StemSplit/demucs-onnx/issues) or
PR a one-line addition.

## The 4 blockers explained

These are the four things that break vanilla `torch.onnx.export` on
HT-Demucs (PyTorch 2.4 / opset 17). Each lives in its own grep-able
module so you can lift the fix into a different project.

### Blocker 1 — `torch.stft` returns complex tensors

```python
# demucs/htdemucs.py
z = torch.stft(x, n_fft, hop_length, return_complex=True)  # complex64 output
```

`torch.onnx.export` raises `Exporting STFT does not currently support
complex types`. The dynamo exporter sometimes lowers it, but the
resulting graph fails ORT shape inference.

**Fix** — [`demucs_onnx/export/stft.py`](https://github.com/StemSplit/demucs-onnx/blob/main/src/demucs_onnx/export/stft.py).
Replace `torch.stft` with a `Conv1d` whose kernels are precomputed
sin/cos DFT bases for `n_fft = 4096`, `hop = 1024`, hann window,
`normalized=True`. The output is two real channels (real, imag) instead
of one complex channel. Inverse: a matching `ConvTranspose1d` plus an
`OLA(window²)` envelope normalisation. The class also overrides
demucs's own `_spec` / `_ispec` / `_magnitude` / `_mask` methods so the
rest of the network sees `(B, C, 2, F, T)` real tensors throughout.
Verified to 5×10⁻⁶ max abs diff against `torch.stft` on real audio.

### Blocker 2 — `model.segment` is a `fractions.Fraction`

```python
# demucs/htdemucs.py
self.segment = Fraction(39, 5)  # = 7.8 seconds
```

`torch._dynamo` allow-lists a small set of "user-defined classes" it
can trace through. `Fraction` is not on it (PyTorch 2.4) and graph
capture crashes. The legacy exporter is more permissive but still
produces a wrong graph because `Fraction` arithmetic is opaque to it.

**Fix** — [`demucs_onnx/export/segment.py`](https://github.com/StemSplit/demucs-onnx/blob/main/src/demucs_onnx/export/segment.py).
Coerce to `float`. Mathematically identical at inference, side-steps
both exporter limitations.

### Blocker 3 — `random.randrange` in the transformer pos-embedding

```python
# demucs/transformer.py
shift = random.randrange(self.sin_random_shift + 1)  # = 0 at eval
```

Used during training for positional-embedding augmentation. At eval,
`sin_random_shift = 0` so the call always returns 0, but neither the
legacy exporter nor dynamo can trace through a call to `random` —
`UnsupportedOperatorError` and graph break, respectively.

**Fix** — [`demucs_onnx/export/pos_embed.py`](https://github.com/StemSplit/demucs-onnx/blob/main/src/demucs_onnx/export/pos_embed.py).
Monkey-patch `CrossTransformerEncoder._get_pos_embedding` with a
deterministic version that hardcodes `shift = 0`. Mathematically
identical at inference time.

### Blocker 4 — `aten::_native_multi_head_attention` has no ONNX symbolic

```python
# torch/nn/functional.py — internally
return torch._native_multi_head_attention(...)  # fused C++ kernel
```

`nn.MultiheadAttention` dispatches to a fast fused C++ kernel when its
inputs satisfy a fast-path check. The fused kernel has no ONNX
symbolic: the exporter raises `UnsupportedOperatorError: Exporting the
operator 'aten::_native_multi_head_attention' to ONNX opset version 17
is not supported`.

**Fix** — [`demucs_onnx/export/mha.py`](https://github.com/StemSplit/demucs-onnx/blob/main/src/demucs_onnx/export/mha.py).
Replace `nn.MultiheadAttention.forward` (per instance, via
`types.MethodType`) with a manual scaled-dot-product attention built
from `Linear` / `bmm` / `softmax`. The exporter handles those
primitives without complaint. Output is bit-identical to the fused
kernel up to fp32 round-off.

### Net result

After all four patches, end-to-end parity vs PyTorch fp32:

| Stem | max abs diff (1×2×343980 random input) |
|---|---:|
| drums | 1.63 × 10⁻⁴ |
| bass | 1.42 × 10⁻⁴ |
| other | 1.71 × 10⁻⁴ |
| vocals | 1.55 × 10⁻⁴ |

…and the ONNX graph runs in `onnxruntime` CPU at **1.31× the speed of
PyTorch CPU** on Apple M4 Pro (no GPU).

## Pre-trained ONNX models on Hugging Face

We host **seven** companion ONNX model repos (plus four PyTorch source
repos for parity-checking your own exports). The Python package
downloads from these automatically on first run; you can also fetch
them by hand.

| Repo | Stems | Size | Use case |
|---|---|---:|---|
| [`StemSplitio/htdemucs-ft-onnx`](https://huggingface.co/StemSplitio/htdemucs-ft-onnx) | all 4 (bag) | 1.26 GB | Full FT bag, best SDR, default |
| [`StemSplitio/htdemucs-onnx`](https://huggingface.co/StemSplitio/htdemucs-onnx) | all 4 (single) | **316 MB** | Fastest 4-stem startup, ~30% faster than the bag |
| [`StemSplitio/htdemucs-6s-onnx`](https://huggingface.co/StemSplitio/htdemucs-6s-onnx) | **6** (incl. guitar + piano) | **258 MB** | The only 6-stem ONNX export on the Hub |
| [`StemSplitio/htdemucs-ft-drums-onnx`](https://huggingface.co/StemSplitio/htdemucs-ft-drums-onnx) | drums | 316 MB | Drum extraction, beat transcription |
| [`StemSplitio/htdemucs-ft-bass-onnx`](https://huggingface.co/StemSplitio/htdemucs-ft-bass-onnx) | bass | 316 MB | Bassline isolation, mix rebalancing |
| [`StemSplitio/htdemucs-ft-other-onnx`](https://huggingface.co/StemSplitio/htdemucs-ft-other-onnx) | other | 316 MB | Karaoke instrumental, sample-flipping |
| [`StemSplitio/htdemucs-ft-vocals-onnx`](https://huggingface.co/StemSplitio/htdemucs-ft-vocals-onnx) | vocals | 316 MB | **#1 open-source vocal SDR** — vocal removal, acapella, karaoke |

Every repo also ships a `*_fp16weights.onnx` variant (~half the
download) with identical runtime memory / latency. All MIT-licensed
and parity-verified to < 1e-3 vs PyTorch fp32. See the
[Models page in the docs](https://stemsplit.github.io/demucs-onnx/models/)
for full size / speed / quality tables.

## Performance

Real measurements on Apple M4 Pro (8-core CPU, no GPU):

| Mode | Per 7.8-s segment | Per 3-min song | RTF |
|---|---:|---:|---:|
| `demucs-onnx`, single specialist (CPU) | **1.59 s** | **~22 s** | 0.20 |
| `demucs-onnx`, full bag (CPU) | 6.4 s | ~88 s | 0.49 |
| PyTorch CPU (single specialist) | 2.09 s | ~29 s | 0.26 |
| PyTorch MPS (full bag) | 1.0 s | ~12 s | 0.07 |

CUDA / DirectML / CoreML ONNX EPs are all ≥ 5× faster than the CPU EP
on real GPUs — see the model card on each HF repo for hardware-specific
numbers.

## API

### `demucs_onnx.separate(input, output_dir=None, *, model="htdemucs_ft", stems=None, providers="auto", precision="fp32", cache_dir=None, token=None, verbose=False, progress=True, output_format="wav", bitrate_kbps=192, mix_stems=None, mix_output_name="mix") -> dict[str, np.ndarray]`

Run separation on an audio file. Returns
`{stem_name: (channels, samples)}` in float32 **at the input file's
native sample rate** (we auto-resample for inference and back). If
`output_dir` is given, also writes `<stem>.wav` (or `.mp3`) files into
it; pass `mix_stems=("drums","bass","other")` to additionally write a
single karaoke instrumental file.

`model` accepts:

- `"htdemucs_ft"` (default) — full 4-stem fine-tuned bag
- `"htdemucs"` — single-file 4-stem, ~30% faster than the bag
- `"htdemucs_6s"` — single-file 6-stem (drums, bass, other, vocals, **guitar**, **piano**)
- `"htdemucs_ft_<stem>"` or just `"<stem>"` — single specialist (`drums` / `bass` / `other` / `vocals`)

`providers` accepts:

- `"auto"` (default) — auto-detect the best EP for this host (CoreML / CUDA / DML / CPU)
- A short alias (`"cpu"`, `"coreml"`, `"cuda"`, `"dml"`), an explicit ORT provider name, or a list of either

`precision` accepts `"fp32"` (default) or `"fp16weights"`. The latter
downloads a 166 MB variant per stem (1.91× smaller) with identical
runtime memory and latency; max abs diff vs fp32 is ~6e-5.

### `demucs_onnx.separate_stem(input, stem, output_dir=None, **kwargs) -> np.ndarray`

Shorthand: run only one specialist and return the single stem as a
numpy array. ~4× faster than running the full bag when you only need
one stem. Accepts `guitar` / `piano` (auto-routes to `htdemucs_6s`).

### `demucs_onnx.separate_all(input, output_dir=None, **kwargs) -> dict[str, np.ndarray]`

Shorthand for `separate(..., model="htdemucs_ft")`.

### `demucs_onnx.prewarm(models=("htdemucs_ft",), **kwargs) -> None`

Pre-download and pre-compile ORT sessions so the first `separate()`
call doesn't pay the CoreML graph-compile or HF-download tax.

### `demucs_onnx.auto_select_providers() -> list[str]`

Return the EP list `separate()` would pick on this host. Useful for
debugging — print it from your code if `auto` selects something
surprising.

### `demucs_onnx.describe_runtime() -> dict[str, object]`

Returns `{system, machine, python, onnxruntime, available_providers,
in_browser}`. Print this if `auto` doesn't pick the EP you expect.

### `demucs_onnx.export.export_to_onnx(checkpoint, output, *, stem=None, stems=None, opset=17, parity_check=True, parity_tolerance=1e-3, ...) -> dict[str, Path]`

Convert a demucs/htdemucs PyTorch checkpoint (by name or `.th` path) to
one or more ONNX files. Applies all four patches, runs a numerical
parity check before writing, and aborts if max abs diff > tolerance.

### `demucs_onnx.export.patch_htdemucs_for_onnx(model) -> nn.Module`

Apply all four patches in place, return the same model. Useful when
you want to keep the patched model around for alternative tracers.

### Individual patches

Each blocker is a single-purpose module so you can pull just one fix
into a different project:

- `demucs_onnx.export.coerce_segment_to_float` — Fraction → float
- `demucs_onnx.export.disable_random_pos_shift` — drop `random.randrange`
- `demucs_onnx.export.onnx_friendly_mha_forward` — manual MHA forward
- `demucs_onnx.export.RealSTFT` / `RealISTFT` — complex STFT replacement

## Skip the infrastructure — use the StemSplit API

Don't want to bundle a 316 MB model in your app, manage a GPU pool, or
write overlap-add chunking? Use the
**[StemSplit API](https://stemsplit.io/developers)** instead — same
models under the hood, hosted for you, with credits and a dashboard.

- [stemsplit.io](https://stemsplit.io)
- [Developer docs](https://stemsplit.io/developers/docs)
- [API reference](https://stemsplit.io/developers/reference)
- **Python SDK:** `pip install stemsplit-python` — [`stemsplit-python` on PyPI](https://pypi.org/project/stemsplit-python/), [source](https://github.com/StemSplit/stemsplit-python)

Or use the no-code tools that ship the same model family:
[Vocal Remover](https://stemsplit.io/vocal-remover) ·
[Karaoke Maker](https://stemsplit.io/karaoke-maker) ·
[Acapella Maker](https://stemsplit.io/acapella-maker) ·
[YouTube Stem Splitter](https://stemsplit.io/youtube-stem-splitter).

## License & attribution

This package is **MIT-licensed**, matching the original HT-Demucs.
Please cite the original authors if you use the model in research:

```bibtex
@inproceedings{rouard2023hybrid,
  title     = {Hybrid Transformers for Music Source Separation},
  author    = {Rouard, Simon and Massa, Francisco and D{\'e}fossez, Alexandre},
  booktitle = {ICASSP},
  year      = {2023}
}
```

- Original PyTorch model: [`facebookresearch/demucs`](https://github.com/facebookresearch/demucs)
- ONNX export, parity verification, packaging, and host inference by [StemSplit](https://stemsplit.io)
