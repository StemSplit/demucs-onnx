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
  "https://huggingface.co/StemSplitio/htdemucs-ft-vocals-onnx/resolve/main/htdemucs_ft_vocals_fp16weights.onnx";

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
