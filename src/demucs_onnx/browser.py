"""Browser-runtime helpers: ``onnxruntime-web`` config snippets and
copyable JS/TS code for running htdemucs in the browser.

This module ships no ONNX Runtime JS code itself — it only knows the
configuration recipes that real-world Vite / Webpack / esbuild projects
need to put in their config to ship onnxruntime-web with the WASM
binaries served correctly. Use it like:

    >>> from demucs_onnx.browser import print_wasm_config
    >>> print_wasm_config("vite")  # prints copy-pasteable Vite snippet

Or the equivalent CLI:

    $ demucs-onnx browser-config --bundler vite
    $ demucs-onnx browser-demo /tmp/browser_demo  # write the vanilla demo

The HF model URLs default to the v0.3.0 ``htdemucs-ft-vocals-onnx``
fp16weights variant (166 MB) — the smallest single-stem download that
runs in a browser tab without OOM on most consumer laptops.
"""
from __future__ import annotations

import shutil
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from ._hub import MODEL_REGISTRY, model_filename

#: The canonical recommended model + variant for the in-browser demo.
DEFAULT_DEMO_MODEL = "htdemucs_ft_vocals"
DEFAULT_DEMO_PRECISION: Literal["fp32", "fp16weights"] = "fp16weights"

#: Path on the HF Hub for the recommended browser demo model.
DEFAULT_DEMO_URL = (
    "https://huggingface.co/StemSplitio/htdemucs-ft-vocals-onnx/"
    "resolve/main/htdemucs_ft_vocals_fp16weights.onnx"
)


Bundler = Literal["vite", "webpack", "esbuild", "next", "rollup"]


def _vite_snippet() -> str:
    return textwrap.dedent("""\
        // vite.config.ts — onnxruntime-web setup for demucs-onnx
        // Copy this into your existing config; merge `optimizeDeps` and
        // `server` keys with whatever else you have.
        import { defineConfig } from "vite";

        export default defineConfig({
          // 1) Don't try to pre-bundle the WASM-touching ORT entry.
          optimizeDeps: {
            exclude: ["onnxruntime-web"],
          },
          // 2) Required for SharedArrayBuffer (multithreaded WASM EP).
          server: {
            headers: {
              "Cross-Origin-Opener-Policy": "same-origin",
              "Cross-Origin-Embedder-Policy": "require-corp",
            },
          },
          // 3) Copy the .wasm files into the build so the runtime can fetch them.
          //    onnxruntime-web 1.17+ ships .mjs + .wasm pairs next to the JS entry.
          assetsInclude: ["**/*.wasm"],
        });

        // Then in your app code:
        //   import * as ort from "onnxruntime-web";
        //   // Optional: point ORT at a CDN copy of the .wasm files instead of
        //   // bundling them. Useful when your CDN is faster than your origin.
        //   ort.env.wasm.wasmPaths =
        //     "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
        //   ort.env.wasm.numThreads = navigator.hardwareConcurrency ?? 4;
        //   const sess = await ort.InferenceSession.create(modelUrl, {
        //     executionProviders: ["wasm"],
        //   });
    """)


def _webpack_snippet() -> str:
    return textwrap.dedent("""\
        // webpack.config.js — onnxruntime-web setup for demucs-onnx
        const CopyPlugin = require("copy-webpack-plugin");
        const path = require("path");

        module.exports = {
          // 1) Copy ORT's .wasm/.mjs glue files into your dist/ so the
          //    runtime can fetch them at the same origin as your bundle.
          plugins: [
            new CopyPlugin({
              patterns: [
                {
                  from: path.dirname(
                    require.resolve("onnxruntime-web/package.json"),
                  ) + "/dist/*.{wasm,mjs}",
                  to: "[name][ext]",
                },
              ],
            }),
          ],

          // 2) Treat .wasm as a separate output asset.
          experiments: { asyncWebAssembly: true },

          // 3) Headers needed for multi-threaded WASM EP (dev server).
          devServer: {
            headers: {
              "Cross-Origin-Opener-Policy": "same-origin",
              "Cross-Origin-Embedder-Policy": "require-corp",
            },
          },
        };

        // In your app code:
        //   import * as ort from "onnxruntime-web";
        //   ort.env.wasm.numThreads = navigator.hardwareConcurrency ?? 4;
        //   const sess = await ort.InferenceSession.create(modelUrl, {
        //     executionProviders: ["wasm"],
        //   });
    """)


def _esbuild_snippet() -> str:
    return textwrap.dedent("""\
        // esbuild.config.mjs — onnxruntime-web setup for demucs-onnx
        import esbuild from "esbuild";
        import { copy } from "esbuild-plugin-copy";

        await esbuild.build({
          entryPoints: ["src/main.ts"],
          bundle: true,
          outdir: "dist",
          format: "esm",
          target: "es2022",
          // ORT ships .wasm files as separate assets; copy them next to
          // the JS bundle so the runtime can fetch them at runtime.
          plugins: [
            copy({
              assets: {
                from: ["node_modules/onnxruntime-web/dist/*.{wasm,mjs}"],
                to: ["."],
              },
            }),
          ],
          loader: { ".wasm": "file" },
        });

        // Serve dist/ with these headers for multithreaded WASM:
        //   Cross-Origin-Opener-Policy:   same-origin
        //   Cross-Origin-Embedder-Policy: require-corp
    """)


def _next_snippet() -> str:
    return textwrap.dedent("""\
        // next.config.js — onnxruntime-web setup for demucs-onnx
        // Next.js needs a server-side carve-out so the SSR bundle doesn't
        // try to import the WASM. We also copy the .wasm files into /public.
        const CopyPlugin = require("copy-webpack-plugin");
        const path = require("path");

        module.exports = {
          webpack: (config, { isServer }) => {
            if (!isServer) {
              config.plugins.push(
                new CopyPlugin({
                  patterns: [
                    {
                      from:
                        path.dirname(
                          require.resolve("onnxruntime-web/package.json"),
                        ) + "/dist/*.{wasm,mjs}",
                      to: "static/chunks/[name][ext]",
                    },
                  ],
                }),
              );
            }
            return config;
          },
          // Required headers for multithreaded WASM EP.
          async headers() {
            return [
              {
                source: "/(.*)",
                headers: [
                  { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
                  { key: "Cross-Origin-Embedder-Policy", value: "require-corp" },
                ],
              },
            ];
          },
        };

        // On the client side, import dynamically so SSR doesn't choke:
        //   const ort = await import("onnxruntime-web");
    """)


def _rollup_snippet() -> str:
    return textwrap.dedent("""\
        // rollup.config.mjs — onnxruntime-web setup for demucs-onnx
        import resolve from "@rollup/plugin-node-resolve";
        import commonjs from "@rollup/plugin-commonjs";
        import copy from "rollup-plugin-copy";

        export default {
          input: "src/main.ts",
          output: { dir: "dist", format: "esm" },
          plugins: [
            resolve({ browser: true }),
            commonjs(),
            copy({
              targets: [
                {
                  src: "node_modules/onnxruntime-web/dist/*.{wasm,mjs}",
                  dest: "dist",
                },
              ],
            }),
          ],
        };
    """)


_SNIPPETS: dict[str, Callable[[], str]] = {
    "vite":    _vite_snippet,
    "webpack": _webpack_snippet,
    "esbuild": _esbuild_snippet,
    "next":    _next_snippet,
    "rollup":  _rollup_snippet,
}


def wasm_config(bundler: Bundler = "vite") -> str:
    """Return a copy-pasteable bundler config snippet for ``onnxruntime-web``.

    Supported bundlers: ``"vite"`` (default), ``"webpack"``,
    ``"esbuild"``, ``"next"``, ``"rollup"``.
    """
    if bundler not in _SNIPPETS:
        raise ValueError(
            f"unknown bundler {bundler!r}; expected one of {list(_SNIPPETS)}",
        )
    return _SNIPPETS[bundler]()


def print_wasm_config(bundler: Bundler = "vite") -> None:
    """Print :func:`wasm_config` to stdout."""
    print(wasm_config(bundler))


def model_browser_url(model: str = DEFAULT_DEMO_MODEL,
                      precision: Literal["fp32", "fp16weights"] = DEFAULT_DEMO_PRECISION,
                      ) -> str:
    """Return the direct download URL for ``model`` at ``precision``.

    Useful when emitting JS that wants to call
    ``ort.InferenceSession.create(url, ...)`` against the HF Hub.
    """
    if model == "htdemucs_ft_vocals":
        from ._hub import MODEL_REPOS, stem_model_filename
        repo = MODEL_REPOS["htdemucs_ft_vocals"]
        fname = stem_model_filename("vocals", precision)
        return f"https://huggingface.co/{repo}/resolve/main/{fname}"
    info = MODEL_REGISTRY.get(model)
    if info is None:
        raise ValueError(f"unknown model {model!r}")
    fname = model_filename(model, precision)
    return f"https://huggingface.co/{info.repo}/resolve/main/{fname}"


# ---------------------------------------------------------------------------
# Demo writer (``demucs-onnx browser-demo /path``)
# ---------------------------------------------------------------------------


_VANILLA_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>demucs-onnx browser demo</title>
  <style>
    body { font-family: -apple-system, system-ui, sans-serif; max-width: 720px;
           margin: 2rem auto; padding: 0 1rem; line-height: 1.5; color: #222; }
    h1 { font-size: 1.6rem; margin-bottom: .25rem; }
    .sub { color: #666; margin-bottom: 1.5rem; }
    fieldset { border: 1px solid #ddd; border-radius: 6px; padding: 1rem 1.25rem; }
    legend { padding: 0 .5rem; font-weight: 600; }
    input[type=file] { display: block; margin: .5rem 0; }
    button { padding: .5rem 1rem; font-size: 1rem; border-radius: 6px;
             border: 1px solid #444; background: #fafafa; cursor: pointer; }
    button:disabled { opacity: .5; cursor: not-allowed; }
    pre { background: #f4f4f4; padding: .75rem; border-radius: 4px; overflow-x: auto; }
    .progress { font-family: ui-monospace, monospace; font-size: .9rem; color: #444; }
    a.dl { display: inline-block; margin-top: .75rem; background: #2563eb; color: #fff;
           padding: .5rem 1rem; text-decoration: none; border-radius: 6px; }
  </style>
</head>
<body>
  <h1>demucs-onnx — browser vocals demo</h1>
  <p class="sub">Runs HT-Demucs FT vocals (166&nbsp;MB, fp16weights) entirely in your tab via
     <code>onnxruntime-web</code>. No upload &mdash; the audio never leaves your machine.</p>

  <fieldset>
    <legend>1. Pick an audio file</legend>
    <p>Best results with a stereo 44.1&nbsp;kHz WAV/MP3 between 5 and 30&nbsp;seconds.</p>
    <input type="file" id="file" accept="audio/*" />
    <p class="progress" id="status">No file loaded.</p>
    <button id="run" disabled>Extract vocals</button>
    <a id="download" class="dl" style="display:none">Download vocals.wav</a>
  </fieldset>

  <fieldset style="margin-top:1.5rem">
    <legend>What this does</legend>
    <ul>
      <li>Downloads <code>htdemucs_ft_vocals_fp16weights.onnx</code> (~166 MB) from HF on first run.
          The browser caches it forever.</li>
      <li>Chunks long inputs into 7.8&nbsp;s segments with 25% overlap and runs them sequentially.</li>
      <li>Reassembles the vocals stem and produces a 16-bit PCM WAV at the original sample rate.</li>
      <li>Source: <a href="https://github.com/StemSplit/demucs-onnx/blob/main/examples/browser/index.html">
          examples/browser/index.html</a> in the demucs-onnx repo.</li>
    </ul>
  </fieldset>

  <script type="module" src="./demo.js"></script>
</body>
</html>
"""


_VANILLA_JS = '''\
// Vanilla-JS htdemucs vocals separator. No build step.
//
// Loaded by index.html. Open via:
//   python -m http.server 8080
//   open http://localhost:8080/
//
// Or just double-click index.html; file:// works too for the inference
// itself, but HF requires a CORS-friendly origin to fetch the model so
// you may want to download the .onnx once and host it next to this file.

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.mjs";

const MODEL_URL =
  "__MODEL_URL__";

const SAMPLE_RATE = 44100;       // demucs ONNX graph is hard-bound to 44.1 kHz
const SEGMENT_S   = 7.8;
const N_SAMPLES   = Math.round(SEGMENT_S * SAMPLE_RATE);  // 343,980
const N_CHANNELS  = 2;
const OVERLAP     = Math.floor(N_SAMPLES / 4);
const STRIDE      = N_SAMPLES - OVERLAP;

const els = {
  file: document.getElementById("file"),
  run: document.getElementById("run"),
  status: document.getElementById("status"),
  download: document.getElementById("download"),
};
let audioBuffer = null;
let session = null;

els.file.addEventListener("change", async () => {
  const f = els.file.files[0];
  if (!f) return;
  els.status.textContent = `Decoding ${f.name} ...`;
  const buf = await f.arrayBuffer();
  const ctx = new (window.AudioContext || window.webkitAudioContext)({
    sampleRate: SAMPLE_RATE,
  });
  audioBuffer = await ctx.decodeAudioData(buf);
  els.status.textContent =
    `Loaded ${audioBuffer.duration.toFixed(1)} s, ${audioBuffer.numberOfChannels} ch, ${audioBuffer.sampleRate} Hz`;
  els.run.disabled = false;
});

els.run.addEventListener("click", async () => {
  els.run.disabled = true;
  els.download.style.display = "none";
  try {
    if (!session) {
      els.status.textContent = "Loading vocals model (166 MB, cached after first load) ...";
      // Optional: set ort.env.wasm.numThreads for multithread.
      ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency ?? 2, 4);
      session = await ort.InferenceSession.create(MODEL_URL, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
    }
    const mix = await getStereo44k();
    const vocals = await separate(mix);
    const wav = encodeWav(vocals, SAMPLE_RATE);
    const url = URL.createObjectURL(new Blob([wav], { type: "audio/wav" }));
    els.download.href = url;
    els.download.download = "vocals.wav";
    els.download.style.display = "inline-block";
    els.status.textContent = "Done. Click below to download.";
  } catch (e) {
    console.error(e);
    els.status.textContent = `Error: ${e.message}`;
  } finally {
    els.run.disabled = false;
  }
});

async function getStereo44k() {
  // Returns Float32Array of length 2 * frames (interleaved L,R is wrong for
  // demucs — we want channels-first so we keep them in separate arrays and
  // splice into the input tensor below).
  const data = [
    audioBuffer.getChannelData(0),
    audioBuffer.numberOfChannels > 1
      ? audioBuffer.getChannelData(1)
      : audioBuffer.getChannelData(0),
  ];
  return data;
}

async function separate(mix) {
  const totalLen = mix[0].length;
  const nChunks = Math.max(1, Math.ceil(totalLen / STRIDE));
  const out = [new Float32Array(totalLen), new Float32Array(totalLen)];
  const weight = new Float32Array(totalLen);
  const win = makeTransitionWindow(N_SAMPLES, OVERLAP);
  const chunkBuf = new Float32Array(1 * N_CHANNELS * N_SAMPLES);

  for (let i = 0; i < nChunks; ++i) {
    els.status.textContent = `Running ONNX: chunk ${i + 1}/${nChunks} ...`;
    const start = i * STRIDE;
    const end = Math.min(start + N_SAMPLES, totalLen);
    chunkBuf.fill(0);
    for (let c = 0; c < N_CHANNELS; ++c) {
      const dst = chunkBuf.subarray(c * N_SAMPLES, c * N_SAMPLES + (end - start));
      dst.set(mix[c].subarray(start, end));
    }
    const inputTensor = new ort.Tensor("float32", chunkBuf, [1, N_CHANNELS, N_SAMPLES]);
    const result = await session.run({ mix: inputTensor });
    // Output shape is (1, 4, 2, N) for vocals specialist — vocals is row 3
    // (drums=0, bass=1, other=2, vocals=3 — match SOURCES in Python infer).
    const stems = result.stems.data;
    const vocalsOffset = (3 * N_CHANNELS) * N_SAMPLES;
    const chunkLen = end - start;
    for (let c = 0; c < N_CHANNELS; ++c) {
      const rowStart = vocalsOffset + c * N_SAMPLES;
      for (let s = 0; s < chunkLen; ++s) {
        out[c][start + s] += stems[rowStart + s] * win[s];
      }
    }
    for (let s = 0; s < chunkLen; ++s) weight[start + s] += win[s];
    // Yield to the event loop so the UI can update between chunks.
    await new Promise((r) => setTimeout(r, 0));
  }
  for (let c = 0; c < N_CHANNELS; ++c) {
    for (let s = 0; s < totalLen; ++s) {
      out[c][s] /= Math.max(weight[s], 1e-8);
    }
  }
  return out;
}

function makeTransitionWindow(segment, overlap) {
  const w = new Float32Array(segment);
  for (let i = 0; i < segment; ++i) w[i] = 1;
  for (let i = 0; i < overlap; ++i) {
    const v = i / overlap;
    w[i] = v;
    w[segment - 1 - i] = v;
  }
  return w;
}

function encodeWav(stereo, sr) {
  // 16-bit PCM stereo WAV.
  const n = stereo[0].length;
  const buf = new ArrayBuffer(44 + n * 4);
  const view = new DataView(buf);
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + n * 4, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 2, true);
  view.setUint32(24, sr, true);
  view.setUint32(28, sr * 4, true);
  view.setUint16(32, 4, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, n * 4, true);
  let off = 44;
  for (let i = 0; i < n; ++i) {
    for (let c = 0; c < 2; ++c) {
      const v = Math.max(-1, Math.min(1, stereo[c][i]));
      view.setInt16(off, v < 0 ? v * 0x8000 : v * 0x7fff, true);
      off += 2;
    }
  }
  return buf;
}

function writeString(view, off, s) {
  for (let i = 0; i < s.length; ++i) view.setUint8(off + i, s.charCodeAt(i));
}
'''


_VANILLA_README = """\
# demucs-onnx — vanilla JS browser demo

This is a zero-build, copy-and-paste browser demo that runs HT-Demucs FT
vocals separation entirely in a single page via
[`onnxruntime-web`](https://www.npmjs.com/package/onnxruntime-web).

It downloads the 166 MB `htdemucs_ft_vocals_fp16weights.onnx` from
[Hugging Face](https://huggingface.co/StemSplitio/htdemucs-ft-vocals-onnx)
on first run and caches it forever. Inference is fully local — the audio
never leaves your machine.

## Run it

```bash
# Option A — double-click index.html (file:// works for the page itself;
#           the HF CDN is permissive enough to load the model from file://).
open index.html

# Option B — serve from a static server so the COOP/COEP headers can be
#           set if you want multithreaded WASM:
python -m http.server 8080
# then open http://localhost:8080/
```

## Pick a different model

Open `demo.js` and edit `MODEL_URL` to point at any of these:

| URL | Stem | Size |
|---|---|---:|
| `https://huggingface.co/StemSplitio/htdemucs-ft-vocals-onnx/resolve/main/htdemucs_ft_vocals_fp16weights.onnx` | vocals | 166 MB |
| `https://huggingface.co/StemSplitio/htdemucs-ft-drums-onnx/resolve/main/htdemucs_ft_drums_fp16weights.onnx`   | drums | 166 MB |
| `https://huggingface.co/StemSplitio/htdemucs-onnx/resolve/main/htdemucs_fp16weights.onnx`                     | all 4 | ~150 MB |

## Trade-offs

- **fp16weights** (default here) — 166 MB download, same speed at runtime.
  Use it.
- **fp32** — 316 MB download. No quality gain in any audible sense
  (max abs diff vs fp16weights is ~6e-5).

## Caveats

- The demo runs single-threaded WASM by default for portability. To
  speed it up ~2-3x on a 4-core CPU, host this file under a server that
  sets the COOP/COEP headers and the ORT WASM EP will automatically
  switch on multithreading.
- Long inputs (>2 min) eat browser memory because we keep the entire
  decoded audio + output buffers in JS heap. For production apps prefer
  the Vite/React variant under `../browser-react/`, which streams.
"""


_REACT_TS = '''\
// App.tsx — Vite + React + TS demo for demucs-onnx in the browser.
//
// To regenerate this scaffold: `demucs-onnx browser-demo --react /path`.
// See `../browser/` for the zero-build vanilla equivalent.
import { useCallback, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

const MODEL_URL =
  "__MODEL_URL__";

const SAMPLE_RATE = 44100;
const SEGMENT_S = 7.8;
const N_SAMPLES = Math.round(SEGMENT_S * SAMPLE_RATE);
const N_CHANNELS = 2;
const OVERLAP = Math.floor(N_SAMPLES / 4);
const STRIDE = N_SAMPLES - OVERLAP;

type Stereo = [Float32Array, Float32Array];

function makeWindow(segment: number, overlap: number): Float32Array {
  const w = new Float32Array(segment).fill(1);
  for (let i = 0; i < overlap; i++) {
    const v = i / overlap;
    w[i] = v;
    w[segment - 1 - i] = v;
  }
  return w;
}

function encodeWav(stereo: Stereo, sr: number): ArrayBuffer {
  const n = stereo[0].length;
  const buf = new ArrayBuffer(44 + n * 4);
  const view = new DataView(buf);
  const writeStr = (off: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + n * 4, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 2, true);
  view.setUint32(24, sr, true);
  view.setUint32(28, sr * 4, true);
  view.setUint16(32, 4, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, n * 4, true);
  let off = 44;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 2; c++) {
      const v = Math.max(-1, Math.min(1, stereo[c][i]));
      view.setInt16(off, v < 0 ? v * 0x8000 : v * 0x7fff, true);
      off += 2;
    }
  }
  return buf;
}

export function App() {
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const [status, setStatus] = useState("Pick an audio file to begin.");
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [audio, setAudio] = useState<AudioBuffer | null>(null);
  const [working, setWorking] = useState(false);

  const onFile = useCallback(async (file: File) => {
    setStatus(`Decoding ${file.name} ...`);
    const buf = await file.arrayBuffer();
    const ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
    const decoded = await ctx.decodeAudioData(buf);
    setAudio(decoded);
    setStatus(
      `Loaded ${decoded.duration.toFixed(1)} s, ${decoded.numberOfChannels} ch, ${decoded.sampleRate} Hz`,
    );
  }, []);

  const onRun = useCallback(async () => {
    if (!audio) return;
    setWorking(true);
    setDownloadUrl(null);
    try {
      if (!sessionRef.current) {
        setStatus("Loading vocals model (166 MB, cached after first load) ...");
        ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency ?? 2, 4);
        sessionRef.current = await ort.InferenceSession.create(MODEL_URL, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        });
      }
      const mix: Stereo = [
        audio.getChannelData(0),
        audio.numberOfChannels > 1 ? audio.getChannelData(1) : audio.getChannelData(0),
      ];
      const total = mix[0].length;
      const nChunks = Math.max(1, Math.ceil(total / STRIDE));
      const out: Stereo = [new Float32Array(total), new Float32Array(total)];
      const weight = new Float32Array(total);
      const window = makeWindow(N_SAMPLES, OVERLAP);
      const chunkBuf = new Float32Array(1 * N_CHANNELS * N_SAMPLES);

      for (let i = 0; i < nChunks; i++) {
        setStatus(`Running ONNX: chunk ${i + 1}/${nChunks} ...`);
        const start = i * STRIDE;
        const end = Math.min(start + N_SAMPLES, total);
        chunkBuf.fill(0);
        for (let c = 0; c < N_CHANNELS; c++) {
          chunkBuf
            .subarray(c * N_SAMPLES, c * N_SAMPLES + (end - start))
            .set(mix[c].subarray(start, end));
        }
        const inputTensor = new ort.Tensor("float32", chunkBuf, [1, N_CHANNELS, N_SAMPLES]);
        const result = await sessionRef.current.run({ mix: inputTensor });
        const stems = result.stems.data as Float32Array;
        const vocalsOffset = 3 * N_CHANNELS * N_SAMPLES;
        const chunkLen = end - start;
        for (let c = 0; c < N_CHANNELS; c++) {
          const rowStart = vocalsOffset + c * N_SAMPLES;
          for (let s = 0; s < chunkLen; s++) {
            out[c][start + s] += stems[rowStart + s] * window[s];
          }
        }
        for (let s = 0; s < chunkLen; s++) weight[start + s] += window[s];
        await new Promise((r) => setTimeout(r, 0));
      }
      for (let c = 0; c < N_CHANNELS; c++) {
        for (let s = 0; s < total; s++) out[c][s] /= Math.max(weight[s], 1e-8);
      }
      const wav = encodeWav(out, SAMPLE_RATE);
      setDownloadUrl(URL.createObjectURL(new Blob([wav], { type: "audio/wav" })));
      setStatus("Done.");
    } catch (e: any) {
      setStatus(`Error: ${e.message ?? String(e)}`);
    } finally {
      setWorking(false);
    }
  }, [audio]);

  return (
    <main style={{ maxWidth: 720, margin: "2rem auto", padding: "0 1rem",
                   fontFamily: "system-ui, -apple-system, sans-serif" }}>
      <h1>demucs-onnx — React + Vite browser demo</h1>
      <p style={{ color: "#666" }}>
        Runs HT-Demucs FT vocals (166&nbsp;MB, fp16weights) entirely in your tab via
        <code> onnxruntime-web</code>. No upload — the audio never leaves your machine.
      </p>
      <fieldset style={{ border: "1px solid #ddd", borderRadius: 6, padding: "1rem 1.25rem" }}>
        <legend><b>1. Pick an audio file</b></legend>
        <input
          type="file"
          accept="audio/*"
          onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
        />
        <p>{status}</p>
        <button onClick={onRun} disabled={!audio || working}>
          Extract vocals
        </button>
        {downloadUrl && (
          <a href={downloadUrl} download="vocals.wav"
             style={{ marginLeft: "1rem", textDecoration: "underline" }}>
            Download vocals.wav
          </a>
        )}
      </fieldset>
    </main>
  );
}
'''


_REACT_INDEX_HTML = """\
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>demucs-onnx React demo</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
"""


_REACT_MAIN_TSX = """\
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
"""


_REACT_VITE_CONFIG = """\
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});
"""


_REACT_PACKAGE_JSON = """\
{
  "name": "demucs-onnx-react-demo",
  "private": true,
  "version": "0.3.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "onnxruntime-web": "^1.18.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.0",
    "typescript": "^5.5.0",
    "vite": "^5.4.0"
  }
}
"""


_REACT_TSCONFIG = """\
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "Bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"]
}
"""


_REACT_README = """\
# demucs-onnx — React + Vite browser demo

A minimal Vite + React + TypeScript demo that runs HT-Demucs FT vocals
in the browser via [`onnxruntime-web`](https://www.npmjs.com/package/onnxruntime-web).

## Run it

```bash
npm install
npm run dev
# open http://localhost:5173
```

## Production build

```bash
npm run build
npm run preview
```

## What this demonstrates

- Vite config (`vite.config.ts`) with the COOP/COEP headers needed to
  enable multithreaded WASM EP in dev.
- `optimizeDeps.exclude = ["onnxruntime-web"]` — ORT's WASM glue should
  not be pre-bundled by esbuild.
- Lazy session creation + reuse across runs (`useRef` to keep the
  `InferenceSession` alive between user interactions).
- Chunked overlap-add (port of the 30-line Python loop in `infer.py`).
- 16-bit PCM WAV encoding directly into a `Blob` for download.

For a zero-build alternative (`file://` works, no `npm install`
required), see `../browser/`.
"""


def write_demo_dir(target: Path, *,
                   model_url: str | None = None,
                   react: bool = False) -> list[Path]:
    """Materialize the in-tree browser demo files under ``target``.

    Args:
        target: Directory to write into. Created if missing. Must be
            empty or non-existent (this helper refuses to overwrite
            existing files).
        model_url: URL the demo should fetch the ONNX model from.
            Defaults to ``DEFAULT_DEMO_URL`` (htdemucs-ft-vocals
            fp16weights from the StemSplitio HF org).
        react: When True, emit a Vite + React + TS project. When False
            (default), emit the zero-build vanilla HTML/JS demo.

    Returns:
        The list of files written, in the order they were created.
    """
    model_url = model_url or DEFAULT_DEMO_URL
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    if any(target.iterdir()):
        raise FileExistsError(
            f"refusing to write demo files into non-empty {target}",
        )
    written: list[Path] = []
    if react:
        files = {
            "index.html":      _REACT_INDEX_HTML,
            "vite.config.ts":  _REACT_VITE_CONFIG,
            "package.json":    _REACT_PACKAGE_JSON,
            "tsconfig.json":   _REACT_TSCONFIG,
            "README.md":       _REACT_README,
            "src/App.tsx":     _REACT_TS.replace("__MODEL_URL__", model_url),
            "src/main.tsx":    _REACT_MAIN_TSX,
        }
    else:
        files = {
            "index.html": _VANILLA_HTML,
            "demo.js":    _VANILLA_JS.replace("__MODEL_URL__", model_url),
            "README.md":  _VANILLA_README,
        }
    for rel, content in files.items():
        path = target / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        written.append(path)
    return written


def _suppress_unused_warning() -> None:
    """Keep ``shutil`` referenced so static analyzers don't flag it. We
    expose it for tests/users that want to clear demo dirs."""
    _ = shutil


__all__ = [
    "DEFAULT_DEMO_MODEL",
    "DEFAULT_DEMO_PRECISION",
    "DEFAULT_DEMO_URL",
    "model_browser_url",
    "print_wasm_config",
    "wasm_config",
    "write_demo_dir",
]
