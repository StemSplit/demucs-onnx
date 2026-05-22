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
