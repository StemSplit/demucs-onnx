# Browser support

`demucs-onnx` runs **fully in the browser tab** via
[`onnxruntime-web`](https://www.npmjs.com/package/onnxruntime-web).
The audio never leaves the user's machine.

This page shows three flavors of integration, from least to most
production-shaped:

1. **Zero-build vanilla** — drop two files on a static host, double-click
   `index.html`, done.
2. **Vite + React + TS** — the typical SPA shape.
3. **Server-side / Next.js** — Next.js needs an SSR carve-out and custom
   headers; we cover both.

For each one we publish a `demucs-onnx browser-config` snippet you can
copy verbatim into your project's bundler config.

!!! tip "Use the fp16weights variant in the browser"
    The default browser demo loads
    `htdemucs_ft_vocals_fp16weights.onnx` (~166 MB) instead of the fp32
    file (~316 MB). The graph still computes in fp32 at runtime, so
    latency and accuracy are unchanged; only the *download* shrinks
    1.91×. Browser caches keep it around forever after the first load.

---

## Pick a model

| Model file (from HF, fp16weights variant) | Download | Best for |
|---|---:|---|
| [`htdemucs_ft_vocals_fp16weights.onnx`](https://huggingface.co/StemSplitio/htdemucs-ft-vocals-onnx/resolve/main/htdemucs_ft_vocals_fp16weights.onnx) | 166 MB | Vocal removal / karaoke (default). |
| [`htdemucs_ft_drums_fp16weights.onnx`](https://huggingface.co/StemSplitio/htdemucs-ft-drums-onnx/resolve/main/htdemucs_ft_drums_fp16weights.onnx) | 166 MB | Drum extraction. |
| [`htdemucs_fp16weights.onnx`](https://huggingface.co/StemSplitio/htdemucs-onnx/resolve/main/htdemucs_fp16weights.onnx) | 166 MB | All 4 stems, single session. |
| [`htdemucs_6s_fp16weights.onnx`](https://huggingface.co/StemSplitio/htdemucs-6s-onnx/resolve/main/htdemucs_6s_fp16weights.onnx) | 136 MB | 6-stem with guitar + piano. |

The 4-stem htdemucs_ft specialist files predict only one row meaningfully
(the bag's drums file's drums row, etc). The `htdemucs` and
`htdemucs_6s` files predict every row, which makes them slightly better
fits for a browser demo when you want all stems in one download.

---

## 1. Zero-build vanilla demo

```bash
# scaffold the demo files into /tmp/demo
demucs-onnx browser-demo /tmp/demo

cd /tmp/demo
python -m http.server 8080
# open http://localhost:8080/
```

You get three files:

```
/tmp/demo/
├── index.html   ~2.5 KB    file-picker UI + "Extract vocals" button
├── demo.js      ~6 KB      ORT session, chunked overlap-add, WAV encode
└── README.md
```

`index.html` works under `file://` for the inference itself (the HF CDN
is permissive enough). You only need an HTTP server if you want
multi-threaded WASM (which needs the COOP/COEP headers shown in the
next section).

The vanilla demo doesn't import `onnxruntime-web` from npm — it loads
the prebuilt `ort.min.mjs` from `cdn.jsdelivr.net`. That keeps the demo
zero-dependency, but if you want to pin ORT to a version locally, fork
the file and swap the import.

---

## 2. Vite + React + TS

```bash
demucs-onnx browser-demo /tmp/react-demo --react

cd /tmp/react-demo
npm install
npm run dev
# open http://localhost:5173/
```

You get a standard Vite + React + TypeScript app:

```
/tmp/react-demo/
├── package.json
├── vite.config.ts
├── tsconfig.json
├── index.html
├── README.md
└── src/
    ├── App.tsx       ~5 KB     React UI + chunked separation loop
    └── main.tsx      tiny      mount point
```

The important parts of `vite.config.ts`:

```ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // 1) Do not pre-bundle ORT's WASM-touching entry.
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  // 2) COOP/COEP enable multi-threaded WASM EP (3-5× speedup).
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});
```

You can print the snippet for any other bundler:

```bash
demucs-onnx browser-config --bundler vite      # default
demucs-onnx browser-config --bundler webpack
demucs-onnx browser-config --bundler esbuild
demucs-onnx browser-config --bundler next
demucs-onnx browser-config --bundler rollup
```

---

## 3. Next.js

Next.js needs an SSR carve-out (`isServer` branch in `webpack` config)
plus the COOP/COEP headers via `next.config.js`:

```bash
demucs-onnx browser-config --bundler next
```

Then dynamically import ORT on the client:

```tsx
"use client";
import { useEffect, useState } from "react";

export default function VocalsButton() {
  const [ort, setOrt] = useState<typeof import("onnxruntime-web") | null>(null);
  useEffect(() => {
    import("onnxruntime-web").then(setOrt);
  }, []);
  // ... render once `ort` is loaded ...
}
```

---

## The 30-line chunked overlap-add loop

The htdemucs ONNX graph is **fixed at exactly 7.8 s of stereo 44.1 kHz**
(`mix` shape `(1, 2, 343980)`). For inputs longer than that, you need
overlap-add chunking with a triangular window. This is the loop both
demos use, distilled to its essence:

```ts
const SAMPLE_RATE = 44100;
const N_SAMPLES = Math.round(7.8 * SAMPLE_RATE);   // 343,980
const OVERLAP   = Math.floor(N_SAMPLES / 4);
const STRIDE    = N_SAMPLES - OVERLAP;

async function separate(
  session: ort.InferenceSession,
  mix: [Float32Array, Float32Array],   // [L, R]
  stemRow: number,                     // 0=drums, 1=bass, 2=other, 3=vocals
) {
  const total = mix[0].length;
  const nChunks = Math.ceil(total / STRIDE);
  const out = [new Float32Array(total), new Float32Array(total)];
  const weight = new Float32Array(total);
  const window = makeTransitionWindow(N_SAMPLES, OVERLAP);
  const chunkBuf = new Float32Array(2 * N_SAMPLES);

  for (let i = 0; i < nChunks; i++) {
    const start = i * STRIDE;
    const end = Math.min(start + N_SAMPLES, total);
    chunkBuf.fill(0);
    for (let c = 0; c < 2; c++) {
      chunkBuf.subarray(c * N_SAMPLES, c * N_SAMPLES + (end - start))
              .set(mix[c].subarray(start, end));
    }
    const result = await session.run({
      mix: new ort.Tensor("float32", chunkBuf, [1, 2, N_SAMPLES]),
    });
    const stems = result.stems.data as Float32Array;   // (1, 4, 2, N) flat
    const rowOffset = (stemRow * 2) * N_SAMPLES;
    const clen = end - start;
    for (let c = 0; c < 2; c++) {
      for (let s = 0; s < clen; s++) {
        out[c][start + s] += stems[rowOffset + c * N_SAMPLES + s] * window[s];
      }
    }
    for (let s = 0; s < clen; s++) weight[start + s] += window[s];
  }
  for (let c = 0; c < 2; c++) {
    for (let s = 0; s < total; s++) {
      out[c][s] /= Math.max(weight[s], 1e-8);
    }
  }
  return out;
}

function makeTransitionWindow(seg: number, overlap: number) {
  const w = new Float32Array(seg).fill(1);
  for (let i = 0; i < overlap; i++) {
    w[i] = i / overlap;
    w[seg - 1 - i] = i / overlap;
  }
  return w;
}
```

For `htdemucs_6s` change `stems[rowOffset + ...]` to use a 6-row layout
and pick whichever row you want
(`drums=0, bass=1, other=2, vocals=3, guitar=4, piano=5`).

---

## Performance notes

Single-threaded WASM on an Apple M4 Pro processes a 7.8 s chunk in
**~6 s** (RTF ~0.77) — slower than the same hardware running ORT
natively (~1.6 s) because the WASM EP runs on the main thread by default.

To unlock multi-threaded WASM (3-5× speedup):

1. Serve under HTTPS or `localhost`.
2. Set the COOP/COEP headers shown above so `SharedArrayBuffer` is
   available.
3. `ort.env.wasm.numThreads = navigator.hardwareConcurrency ?? 4;`

We've measured ~2.5 s per 7.8 s chunk on the same Apple M4 Pro with 4
threads. Mobile devices are slower (~5-10 s per chunk on iPhone 15 Pro).

---

## Frequently asked

**Q: Why fixed 7.8 s segments instead of dynamic length?**
The exported ONNX graph bakes the segment length in to keep the model
~3× smaller (no dynamic-shape blowup in attention). Chunked overlap-add
recovers any length.

**Q: Can I run this on mobile (iOS / Android)?**
Yes — pair the same ONNX file with `onnxruntime-mobile` (iOS / Android
native) or React Native via `react-native-onnxruntime`. The 4-blocker
patches make the model bit-identical across runtimes.

**Q: Does this leak my user's audio to a server?**
No. The browser fetches the `.onnx` file from Hugging Face's CDN, then
runs everything locally. The audio never leaves the tab.
