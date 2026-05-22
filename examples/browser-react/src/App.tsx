// App.tsx — Vite + React + TS demo for demucs-onnx in the browser.
//
// To regenerate this scaffold: `demucs-onnx browser-demo --react /path`.
// See `../browser/` for the zero-build vanilla equivalent.
import { useCallback, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

const MODEL_URL =
  "https://huggingface.co/StemSplitio/htdemucs-ft-vocals-onnx/resolve/main/htdemucs_ft_vocals_fp16weights.onnx";

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
