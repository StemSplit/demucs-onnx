# Changelog

All notable changes to `demucs-onnx` are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/).

## [0.3.4] - 2026-05-22 — CI release pipeline, drop Python 3.10

### Added

- **Tag-triggered PyPI publish workflow** (`.github/workflows/publish.yml`)
  using PyPI Trusted Publishing (OIDC, no long-lived API tokens), a
  multi-Python verify matrix, a parity smoke gate that separates a
  1-second sine clip on CPU end-to-end, and a sigstore build-provenance
  attestation on every wheel and sdist. A `workflow_dispatch` dry-run
  path against TestPyPI exists for validating release plumbing without
  cutting a tag.

### Changed

- **Dropped Python 3.10 support.** `onnxruntime` 1.24+ no longer ships
  `cp310` wheels, which broke `uv sync` on the verify matrix. Minimum
  supported Python is now 3.11. The 3.10 trove classifier is gone and
  `tool.ruff.target-version` is `py311`.
- **Bumped GitHub Actions off the deprecated Node 20 runtime.**
  `actions/checkout` → `@v6`, `actions/upload-artifact` → `@v7`,
  `actions/download-artifact` → `@v8`, `actions/attest-build-provenance`
  → `@v4`, `actions/setup-python` → `@v6`, `actions/upload-pages-artifact`
  → `@v5`, `actions/deploy-pages` → `@v5`. Third-party pins refreshed:
  `astral-sh/setup-uv` → `@v8.1.0` (SHA-pinned),
  `softprops/action-gh-release` → `@v3.0.0` (SHA-pinned).
- **`fail-fast: false`** on the verify matrix so a single bad Python
  leg no longer chain-cancels the others.

### Fixed

- **`__version__` matches the published wheel version.** v0.3.3 shipped
  a wheel whose `demucs_onnx.__version__` still read `"0.3.2"`; the new
  workflow's verify gate cross-checks `__version__`, `pyproject.toml`,
  and the pushed git tag on every run, so this can't drift again.

### Docs

- **Corrected README attribution and HF model-repo count** (7 ONNX
  repos, not 9) across `docs/index.md`, `docs/models.md`, and
  `docs/comparison.md`.
- **Rewrote `PUBLISH.md`** to match the new CI release flow — TL;DR
  bump + tag + push; the workflow does the rest. The pre-existing
  `tools/publish.py` is documented as break-glass-only.

## [0.3.3] - 2026-05-21 — Cross-link the official Python SDK

Documentation- and metadata-only patch release. **No runtime changes** —
if you installed v0.3.2 the Python and CLI behavior is byte-identical
to v0.3.3.

### Added

- **Cross-links to [`stemsplit-python`](https://pypi.org/project/stemsplit-python/)** —
  the new official Python SDK for the hosted StemSplit API — in the
  README's "Quick links" section and the "Skip the infrastructure"
  section. The two packages are designed to coexist: prototype against
  the local-inference path with `demucs-onnx`, then swap in
  `stemsplit-python` when you want shared GPU capacity, the YouTube
  ingest, BPM / key detection, or webhook delivery.

## [0.3.2] - 2026-05-21 — PyPI page polish (badges, comparison table, metadata)

Documentation- and metadata-only patch release. **No runtime changes** —
if you installed v0.3.1 the Python and CLI behavior is byte-identical
to v0.3.2.

### Added

- **Banner image** (`assets/banner.svg`) at the top of the README,
  rendered on PyPI and GitHub via an absolute `raw.githubusercontent.com`
  URL.
- **Expanded badges row** in the README — added a downloads badge
  (`pypi/dm/demucs-onnx`) and a GitHub stars badge alongside the
  existing version, Python, license, and docs badges.
- **"Quick links" section** at the top of the README so docs / GitHub
  / Hugging Face / CLI / Browser / API links are visible above the
  fold.
- **Expanded comparison table** distilled from
  `COMPETITIVE_LANDSCAPE.md`, comparing against `facebookresearch/demucs`,
  `nomadkaraoke/audio-separator`, `deezer/spleeter`, and
  `sevagh/demucs.onnx` across nine concrete dimensions.
- **"Used by / integrations" section** in the README listing the
  StemSplit production deployment and the Hugging Face model repos as
  first-party consumers, plus an open invitation for community
  integrations.

### Changed

- **Fixed PyPI rendering** of bullet lists with embedded code blocks
  in the "What's new" and "Why this package exists" sections — PyPI's
  CommonMark renderer was breaking the hanging indent on continuation
  lines. Restructured to keep bullets single-paragraph and promote
  code examples to top-level fences.
- **Fixed PyPI rendering** of the ordered list under "We solved it.
  This package ships:" — the renderer was renumbering each item as
  `1.` because of blank-line separators. Replaced with prose +
  fenced examples.
- **Corrected the model-repo count** in the README — the prose said
  "nine" but only seven ONNX repos exist on Hugging Face (plus four
  PyTorch source repos for parity-checking).
- **All in-README image and link URLs are now absolute**, pointing at
  `raw.githubusercontent.com` or the docs site, so PyPI's renderer
  resolves them correctly.
- **`[project.urls]`** expanded with `Discussions`, `Source Code`,
  `Hosted API`, `Hugging Face Models`, `StemSplit App`. `Homepage`
  now points at the docs site (the most useful destination for a
  PyPI visitor) rather than the company landing page.
- **`keywords`** expanded from 23 to 36 entries to cover task names
  (`vocal-removal`, `instrumental-extraction`, `karaoke-maker`,
  `acapella-extractor`), generic audio terms (`audio`, `music`,
  `audio-ml`, `audio-processing`, `stems`), and browser/runtime
  surface (`browser-audio`, `wasm`, `webgpu`, `htdemucs_ft`).
- **`classifiers`** expanded from 14 to 28 entries to cover all four
  major OS targets (`MacOS X`, `Windows`, `POSIX :: Linux`),
  `Environment :: Console` and `Environment :: Web Environment`,
  `Intended Audience :: End Users/Desktop`,
  `Programming Language :: Python :: 3 :: Only`,
  `Topic :: Multimedia :: Sound/Audio :: Editors / Mixers`, and
  `Topic :: Software Development :: Libraries`.
- **`maintainers`** field added to `pyproject.toml` (same as
  `authors`) so PyPI's sidebar surfaces a maintainer block.

## [0.3.1] - 2026-05-21

Documentation-only re-release of v0.3.0 with corrected README and
CHANGELOG framing. No functional changes — if you installed v0.3.0,
the runtime is identical to v0.3.1.

## [0.3.0] - 2026-05-21 — Browser support, more models, docs site

This release adds browser inference via `onnxruntime-web`, two new
model flavors (`htdemucs` and `htdemucs_6s`), a published documentation
site, and a session-pooling fix for repeated calls. Highlights:

### Added

- **New `htdemucs` flavor** — single-file 4-stem ONNX model. ~30%
  faster than the FT bag (1 session instead of 4), slightly lower SDR.
  New HF repo:
  [`StemSplitio/htdemucs-onnx`](https://huggingface.co/StemSplitio/htdemucs-onnx).
  Use via `separate(model="htdemucs")`.
- **New `htdemucs_6s` flavor** — single-file 6-stem ONNX model with
  **guitar** and **piano** in addition to the standard 4 stems. The
  **only ONNX export of the 6-stem variant on the Hub.** New HF repo:
  [`StemSplitio/htdemucs-6s-onnx`](https://huggingface.co/StemSplitio/htdemucs-6s-onnx).
  Use via `separate(model="htdemucs_6s")` or
  `separate_stem("song.mp3", "guitar")`.
- **`demucs_onnx.browser` module + CLI commands** —
  `demucs-onnx browser-config --bundler {vite|webpack|esbuild|next|rollup}`
  prints a ready-to-paste `onnxruntime-web` config snippet for each
  major bundler. `demucs-onnx browser-demo PATH [--react]`
  scaffolds the in-tree vanilla-HTML or Vite + React + TS demo into a
  directory so users can run `python -m http.server` and try it
  locally without cloning anything.
- **`examples/browser/`** — zero-build vanilla HTML/JS demo that loads
  the 166 MB fp16weights vocals model from HF and separates a WAV via
  FileReader. Chunked overlap-add ported from the Python `infer.py`.
- **`examples/browser-react/`** — minimal Vite + React + TS demo with
  the same flow, prettier UI, multi-thread WASM via COOP/COEP.
- **`prewarm(models=[...])`** — pre-download and pre-compile ORT
  sessions so the first `separate()` call doesn't pay the CoreML
  graph-compile or HF-download tax. New `demucs-onnx prewarm` CLI too.
- **`SessionPool` + `session_pool()`** — process-wide session cache so
  repeated `separate*` calls reuse compiled graphs. Especially
  valuable for CoreML EP (the first `htdemucs_ft` bag call previously
  triggered 4 separate CoreML graph compilations).
- **Docs site** at
  [stemsplit.github.io/demucs-onnx](https://stemsplit.github.io/demucs-onnx)
  built with MkDocs Material + `mkdocstrings[python]`. Pages:
  Install, CLI, Python API (autogenerated), Browser support, Models,
  Export your own, Comparison, Changelog. Deploys from
  `.github/workflows/docs.yml` on push to main.
- **`ALL_KNOWN_STEMS`** — re-exported as a top-level constant
  (`drums`, `bass`, `other`, `vocals`, `guitar`, `piano`).

### Changed

- **`list_models()` shape** — each entry now also carries `kind`
  (``"specialist_bag"`` / ``"single"`` / ``"specialist"``) and
  ``sources`` (comma-joined stem list). CLI `list-models` output
  updated to a five-column table.
- **`separate_stem(..., "guitar")` and `separate_stem(..., "piano")`**
  auto-route to `htdemucs_6s` instead of erroring out.
- **CLI `--stem`** accepts `guitar` / `piano` (auto-routes).
- **Quiet `huggingface_hub` HTTP logs** in non-verbose mode and disable
  hf-hub progress bars when `--quiet`.
- **`[project.urls]`** adds `Documentation =
  "https://stemsplit.github.io/demucs-onnx/"` and a `Changelog` URL.

### Fixed

- **Specialist bag re-compilation on every call** — `separate()` now
  reuses sessions from the process-wide pool, eliminating the
  multi-minute CoreML graph-compile tax on every call after the first
  when using `model="htdemucs_ft"` in a long-running process.

### New model parity (vs PyTorch fp32, random 1×2×343980 input)

| Model | max abs diff | tolerance |
|---|---:|---:|
| `htdemucs.onnx` | 6.62 × 10⁻⁴ | 1 × 10⁻³ |
| `htdemucs_fp16weights.onnx` | + 4.6 × 10⁻⁵ (vs fp32 weights) | 1 × 10⁻³ |
| `htdemucs_6s.onnx` | 2.42 × 10⁻⁴ | 1 × 10⁻³ |
| `htdemucs_6s_fp16weights.onnx` | + 1.06 × 10⁻⁴ (vs fp32 weights) | 1 × 10⁻³ |

## [0.2.0] - 2026-05-21

The **UX bundle**: turn "it works if you know what you're doing" into
"it just works". Four headline features, all opt-in or backwards
compatible with the v0.1.0 API surface.

### Added

- **`auto_select_providers()`** in the new `demucs_onnx.providers`
  module. Detects the runtime and returns the optimal ORT provider
  list — CoreML on macOS arm64, CUDA on Linux with NVIDIA, DML on
  Windows with DirectX 12, CPU otherwise. Browser (Pyodide / `js`)
  detection is wired up but full WASM support is deferred to v0.3.
- **`providers="auto"`** is now the default for `separate`,
  `separate_all`, `separate_stem`, and the CLI's `--providers` flag.
  The old short-alias (`"cpu"` / `"coreml"` / `"cuda"` / `"dml"`),
  explicit-list, and `None` overrides all still work unchanged.
  We emit a one-time `RuntimeWarning` when the auto-selected EP list
  is CPU-only on a host where we expected a GPU EP (e.g. macOS arm64
  without CoreML EP — the user almost certainly wants to upgrade
  `onnxruntime`).
- **`precision` kwarg** (`"fp32"` / `"fp16weights"`) on `separate*` and
  a `--precision` / `--small` CLI flag. `fp16weights` downloads the
  166 MB variant from the same HF repo as the 316 MB fp32 file
  (1.91× smaller download). Runtime memory and latency are
  unchanged — the graph still computes in fp32 — and the max abs diff
  vs the fp32 weights is ~6 × 10⁻⁵.
- **`--mp3` CLI flag** with `--bitrate {32-320}k`. Powered by the tiny
  `lameenc` wheel via the new `demucs-onnx[mp3]` extra — no ffmpeg
  system dep required. Same `lameenc` also exposed via
  `demucs_onnx.write_mp3` and `demucs_onnx.write_audio` (auto-dispatch
  by file suffix).
- **`--mix-stems vocals,drums` CLI flag** + the equivalent `mix_stems`
  Python kwarg. Writes a single output file that's the sum of the
  named stems alongside the individual stem files. Use
  `mix_output_name` (Python) or implicit `mix.{wav,mp3}` (CLI) to
  rename the output.
- **`--karaoke` shortcut**: alias for `--mix-stems drums,bass,other
  --mix-output-name karaoke`. Combined with `--mp3` this is the
  one-command "give me a karaoke instrumental MP3" feature.
- **Progress bar** via `tqdm` for chunked inference. Disabled when
  stdout is not a TTY, when `--verbose` is set (the per-chunk log
  takes over), or when `--quiet` is set.
- **`--quiet` / `-q` flag**: silences INFO logs and the progress bar.
- **Auto-resample inputs**: any sample rate `soundfile` can decode now
  works. We resample to 44.1 kHz for inference and back to the input's
  native rate before writing. Backend priority: `soxr` (default,
  added to base deps), then `scipy.signal.resample_poly`, then a
  numpy-only linear-interpolation fallback. Mono inputs are
  duplicated to stereo (previously rejected). Inputs below 8 kHz log
  a quality warning but are still processed.
- **`list_models()` now returns variants per stem**:
  `{alias: {"repo": url, "fp32": url, "fp16weights": url}}`.
- **`demucs_onnx.describe_runtime()`**: introspection helper that
  prints system, machine, Python version, ORT version, available EPs,
  and browser detection.

### Changed

- `separate()` now returns audio at the **input file's native sample
  rate** (auto-resampled), not always 44.1 kHz. If you were relying on
  the old behavior, pass `target_sr=44100` to `load_audio` directly
  and write the result yourself.
- `providers=None` now means "auto" (previously "CPU only"). Explicit
  `providers="cpu"` is still the way to force CPU.
- `list_models()` return shape changed from `{alias: url}` to
  `{alias: {variant: url}}`. The CLI `list-models` output now shows
  one line per `(model, precision)` pair.

### Dependencies

- Added `soxr>=0.3` to base deps (~3 MB wheel, C-fast resampler).
- Added `tqdm>=4.65` to base deps (was already pulled in transitively
  via `huggingface_hub`; pinned directly so the progress bar still
  works if hf-hub is downgraded).
- New `mp3` optional extra: `lameenc>=1.6`.
- New `all` optional extra: includes `[mp3]`.

### Fixed

- Mono input no longer raises `ValueError`; it's transparently
  duplicated to stereo.
- Non-44.1 kHz input no longer raises `ValueError`; it's transparently
  resampled.

## [0.1.0] - 2026-05-21

Initial release.

- Pure numpy + onnxruntime inference path for `htdemucs_ft` at fp32
  parity to the original PyTorch model (max abs diff < 1.71 × 10⁻⁴).
- One-call `export_to_onnx(checkpoint, output)` that applies the four
  blocker patches (complex STFT, `fractions.Fraction`,
  `random.randrange`, `aten::_native_multi_head_attention`) and
  parity-checks before writing.
- Five companion model repos under
  [`StemSplitio`](https://huggingface.co/StemSplitio) on the
  Hugging Face Hub, auto-downloaded on first use.
- CLI: `demucs-onnx separate`, `demucs-onnx export`,
  `demucs-onnx list-models`.

[0.3.4]: https://github.com/StemSplit/demucs-onnx/releases/tag/v0.3.4
[0.3.3]: https://github.com/StemSplit/demucs-onnx/releases/tag/v0.3.3
[0.3.2]: https://github.com/StemSplit/demucs-onnx/releases/tag/v0.3.2
[0.3.1]: https://github.com/StemSplit/demucs-onnx/releases/tag/v0.3.1
[0.3.0]: https://github.com/StemSplit/demucs-onnx/releases/tag/v0.3.0
[0.2.0]: https://github.com/StemSplit/demucs-onnx/releases/tag/v0.2.0
[0.1.0]: https://github.com/StemSplit/demucs-onnx/releases/tag/v0.1.0
