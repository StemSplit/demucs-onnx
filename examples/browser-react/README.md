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
