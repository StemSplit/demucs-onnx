# CLI reference

The `demucs-onnx` package installs a single executable, `demucs-onnx`,
with five subcommands.

```bash
demucs-onnx --help
demucs-onnx separate --help
demucs-onnx export --help
demucs-onnx list-models --help
demucs-onnx browser-config --help
demucs-onnx browser-demo --help
demucs-onnx prewarm --help
```

---

## `separate`

Run separation on an audio file. Auto-downloads the model from Hugging
Face on first use and caches forever.

```bash
demucs-onnx separate INPUT OUTPUT_DIR [options]
```

| Option | Default | Description |
|---|---|---|
| `--model` | `htdemucs_ft` | Model name. Choices: `htdemucs_ft`, `htdemucs`, `htdemucs_6s`, or any specialist alias (`drums` / `bass` / `other` / `vocals`). |
| `--stem` | — | Shortcut: run only this one stem. For `drums`/`bass`/`other`/`vocals` picks the FT specialist (~4× faster). For `guitar`/`piano` routes to `htdemucs_6s`. |
| `--stems STEM ...` | — | Subset of stems to materialize. |
| `--providers` | `auto` | One of `auto`, `cpu`, `coreml`, `cuda`, `dml`, `wasm`. |
| `--precision` | `fp32` | `fp32` or `fp16weights` (half the download size). |
| `--small` | off | Alias for `--precision fp16weights`. |
| `--mp3` | off | Write `.mp3` instead of `.wav`. Needs `pip install 'demucs-onnx[mp3]'`. |
| `--bitrate` | `192k` | MP3 bitrate. 32-320. |
| `--mix-stems` | — | Comma-separated stems to sum into one extra file. |
| `--karaoke` | off | Shortcut for `--mix-stems drums,bass,other --mix-output-name karaoke`. |
| `--cache-dir` | — | Override hf-hub cache. |
| `-q, --quiet` | off | Errors only. |
| `-v, --verbose` | off | Per-chunk log instead of the progress bar. |

### Examples

```bash
# Full 4-stem separation, auto execution provider.
demucs-onnx separate song.mp3 out/

# Killer feature — one-command karaoke instrumental MP3.
demucs-onnx separate song.mp3 out/ --karaoke --mp3

# Just vocals, 4× faster than the bag, 166 MB download.
demucs-onnx separate song.mp3 out/ --stem vocals --small

# 6-stem mode with guitar + piano (new in v0.3).
demucs-onnx separate song.mp3 out/ --model htdemucs_6s

# Just the guitar stem (auto-routes to htdemucs_6s).
demucs-onnx separate song.mp3 out/ --stem guitar

# Custom mix: write only vocals + drums as a single file.
demucs-onnx separate song.mp3 out/ --mix-stems vocals,drums --mp3

# Force CPU even on a GPU box (for reproducibility).
demucs-onnx separate song.mp3 out/ --providers cpu
```

---

## `export`

Convert a demucs/htdemucs PyTorch checkpoint to ONNX. Requires
`pip install 'demucs-onnx[export]'` (adds PyTorch + demucs).

```bash
demucs-onnx export CHECKPOINT OUTPUT [options]
```

| Option | Default | Description |
|---|---|---|
| `--stem` | — | Single stem to export from the FT bag. |
| `--stems STEM ...` | — | Subset to export from the FT bag. |
| `--opset` | `17` | ONNX opset version. |
| `--no-parity-check` | off | Skip the numerical parity check. **Not recommended.** |
| `--parity-tolerance` | `1e-3` | Max abs diff allowed against PyTorch fp32. |
| `-q, --quiet` | off | Suppress progress output. |

### Examples

```bash
# All 4 specialists of the FT bag into out/.
demucs-onnx export htdemucs_ft out/

# Single specialist into a named file.
demucs-onnx export htdemucs_ft drums.onnx --stem drums

# The single-file 4-stem flavor (~316 MB, 1 file).
demucs-onnx export htdemucs out/htdemucs.onnx

# The 6-stem flavor with guitar + piano (~258 MB).
demucs-onnx export htdemucs_6s out/htdemucs_6s.onnx

# Your own fine-tune (.th path).
demucs-onnx export ./my_finetune.th out/my_finetune.onnx
```

---

## `list-models`

Print every model alias, its kind (specialist bag vs single-file vs
specialist), its stems, and the direct HF download URL for each
precision variant.

```bash
demucs-onnx list-models
```

Sample output (truncated):

```
alias                   kind            sources                           precision     url
----------------------  --------------  --------------------------------  ------------  --------
htdemucs                single          drums,bass,other,vocals           fp32          https://huggingface.co/StemSplitio/htdemucs-onnx/resolve/main/htdemucs.onnx
htdemucs                single          drums,bass,other,vocals           fp16weights   https://huggingface.co/StemSplitio/htdemucs-onnx/resolve/main/htdemucs_fp16weights.onnx
htdemucs_6s             single          drums,bass,other,vocals,guitar,piano  fp32      https://huggingface.co/StemSplitio/htdemucs-6s-onnx/resolve/main/htdemucs_6s.onnx
htdemucs_ft             specialist_bag  drums,bass,other,vocals           fp32          https://huggingface.co/StemSplitio/htdemucs-ft-onnx
...
```

---

## `browser-config`

Print a copy-pasteable bundler config snippet for `onnxruntime-web`.

```bash
demucs-onnx browser-config --bundler {vite,webpack,esbuild,next,rollup}
```

See [Browser support](browser.md) for the rendered snippets and what
each one fixes.

---

## `browser-demo`

Scaffold the in-tree browser demo files into a directory so you can run
`python -m http.server` and try it locally without cloning the repo.

```bash
demucs-onnx browser-demo TARGET_DIR [--react] [--model-url URL]
```

| Option | Default | Description |
|---|---|---|
| `--react` | off | Emit a Vite + React + TS project instead of vanilla HTML/JS. |
| `--model-url` | htdemucs-ft-vocals fp16weights from HF | Override the ONNX model URL. |

Examples:

```bash
# Zero-build vanilla demo (open index.html directly).
demucs-onnx browser-demo /tmp/demo
cd /tmp/demo && python -m http.server 8080
# open http://localhost:8080/

# React + Vite + TS demo (npm install).
demucs-onnx browser-demo /tmp/react-demo --react
cd /tmp/react-demo && npm install && npm run dev
```

---

## `prewarm`

Pre-download and pre-compile sessions for one or more models. Useful in
CI / server-warmup contexts so the first `separate()` call doesn't pay
the model-download or CoreML graph-compile latency.

```bash
demucs-onnx prewarm [--models MODEL ...] [--precision {fp32,fp16weights}] [--providers ...] [--cache-dir DIR]
```

Examples:

```bash
# Default — htdemucs_ft (4 specialists).
demucs-onnx prewarm

# Single-file 6-stem flavor.
demucs-onnx prewarm --models htdemucs_6s --precision fp16weights

# Multiple models, CoreML EP.
demucs-onnx prewarm --models htdemucs_ft htdemucs_6s --providers coreml
```
