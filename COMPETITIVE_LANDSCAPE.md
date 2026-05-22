# `demucs-onnx` — competitive landscape & next-step recommendation

Snapshot date: **2026-05-21**. All counts verified live against
GitHub / PyPI / Hugging Face Hub on this date. Quoted stats are
copy-pasted from `gh repo view`, `pypistats`, and the HF `/api`
endpoints, not paraphrased.

---

## TL;DR (60 seconds)

`demucs-onnx` is **the only pip-installable Python package** that runs
HT-Demucs as ONNX with no PyTorch at inference. That niche is
genuinely uncontested — every other "Demucs as ONNX" project on the
internet is either (a) a C++ binary (`sevagh/demucs.onnx`,
`Intel/demucs-openvino`, `MansfieldPlumbing/Demucs_v4_TRT`),
(b) a single-platform `.mlpackage`/`.mlx` (Apple-only:
`dexxdean/htdemucs-coreml`, `andrade0/demucs-mlx`,
`ssmall256/demucs-mlx`), or (c) a 0-download HF repo with no working
script. Nobody has cross-platform Python + browser + pre-built models
under one install command.

**We are NOT alone** on two specific claims:
(1) "STFT inside the ONNX graph" — `MansfieldPlumbing/Demucs_v4_TRT`
shipped a single-input ONNX with internalized FFT ~3 months before
us, but locked to Windows + TensorRT and never packaged.
(2) "HT-Demucs CoreML" — `dexxdean/htdemucs-coreml` and
`john-rocky/CoreML-Models` both shipped working `.mlpackage`s,
though neither is pip-installable and neither covers the 6-stem
variant.

**The single highest-impact next step is not a feature — it's a 30-min
outreach**: comment on the three multi-year-open ONNX export issues
on `facebookresearch/demucs` (#281, #530, #539). Combined, they are
the top Google result for "demucs onnx" and have been sitting open
without a working answer since 2022. After that, the highest
impact-per-day work is a PR to **`nomadkaraoke/python-audio-separator`
(14 616 PyPI downloads/day)** to register `htdemucs_ft-onnx` as the
default Demucs backend — that one merge would 10× our distribution
without us building anything new.

**Top 5 v0.4+ priorities, ranked by impact-per-day:**

1. Comment on the 3 open demucs ONNX issues + add ourselves to
   `awesome-onnx` / `awesome-onnxruntime` (≤1 day, ∞ leverage).
2. PR `nomadkaraoke/python-audio-separator` to use `demucs-onnx`
   as the Demucs backend (2-3 days, 14k DLs/day funnel).
3. Static HF Space that runs HT-Demucs entirely in-browser via
   `onnxruntime-web` (1-2 days, zero competitors, persistent SEO).
4. Add `WebGPU` EP support to the browser path (2-3 days, no
   competitor has this).
5. Add an ONNX export of `BS-RoFormer` / `Mel-Band-RoFormer`
   (5-7 days, currently SOTA, nobody else has an ONNX version).

Detail and justification below.

---

## 1. Executive summary

**Where we stand:** `demucs-onnx` occupies a real and currently
uncontested niche — *pip-installable Python + cross-platform ONNX +
pre-built models for HT-Demucs FT and 6s on the HF Hub*. That exact
intersection has zero direct competitors. **But it is not a niche
nobody wants** — three open issues on the upstream `demucs` repo from
2022-2023 (#281, #530, #539) and the existence of five separate
solo projects (sevagh, MansfieldPlumbing, dexxdean, Intel, john-rocky)
attempting the same conversion in different runtimes prove sustained
demand.

Where we are *not* unique: STFT-internalized ONNX
(`MansfieldPlumbing/Demucs_v4_TRT` got there first, for TRT only),
CoreML conversion (`dexxdean`, `john-rocky` shipped first), Apple
Silicon native (`andrade0`/`ssmall256` shipped MLX ports). Where we
are clearly behind: total user reach (`Anjok07/UVR` 24 737 ⭐,
`stemrollerapp` 3 104 ⭐, `nomadkaraoke/audio-separator` 14 616 DLs/day),
because those are end-user GUIs/wrappers and we are intentionally a
developer SDK. The strategic move is to stay an SDK but plug into
those wrappers so their users transparently benefit from our work.

---

## 2. Comparison matrix

Legend: ✅ yes · ❌ no · ⚠️ partial · — n/a

| Project | Stars | Last commit | License | PyPI? | ONNX runtime? | Browser? | Mobile-friendly? | htdemucs specifically? | Maintained 2026? |
|---|---:|---|---|---|---|---|---|---|---|
| **`demucs-onnx` (this)** | 0 (new) | 2026-05-21 | MIT | **✅ `demucs-onnx`** | ✅ ORT, all EPs | ✅ ORT-web + scaffold | ✅ iOS/Android | ✅ ft + 4s + 6s | ✅ |
| `sevagh/demucs.onnx` | 63 | 2026-02-08 | MIT | ❌ (C++ build) | ✅ ORT, C++ | ❌ | ⚠️ TBD | ✅ all variants | ✅ |
| `sevagh/demucs.cpp` | 162 | 2024-12-01 | MIT | ❌ (C++ build) | ❌ (ggml) | ❌ | ⚠️ | ✅ | ❌ stale |
| `MansfieldPlumbing/Demucs_v4_TRT` | 0 (GH) / 635 DLs (HF) | 2026-03-18 | unset | ❌ (Win .exe) | ⚠️ TRT only | ❌ | ❌ Win only | ✅ 6s only | ✅ |
| `dexxdean/htdemucs-coreml` | 0 | 2026-04-26 | other | ❌ (script) | ❌ CoreML | ❌ | ✅ iOS/macOS only | ✅ 4s only | ✅ |
| `john-rocky/CoreML-Models` | 1 764 | 2026-05-13 | unset | ❌ (zoo) | ❌ CoreML | ❌ | ✅ iOS/macOS only | ✅ 4s only | ✅ |
| `Intel/demucs-openvino` (model) | — | 2024 | Apache | ❌ (OV IR) | ❌ OpenVINO | ❌ | ❌ desktop | ✅ ft + 4s + 6s | ⚠️ |
| `intel/openvino-plugins-ai-audacity` | 1 987 | 2026-05-04 | GPL-3.0 | ❌ (Audacity plugin) | ❌ OpenVINO | ❌ | ❌ desktop | ✅ | ✅ |
| `andrade0/demucs-mlx` | 12 | 2026-02-27 | MIT | ❌ | ❌ MLX | ❌ | ✅ Apple-only | ✅ ft | ⚠️ |
| `ssmall256/demucs-mlx` | 25 | 2026-03-06 | MIT | ❌ | ❌ MLX | ❌ | ✅ Apple-only | ✅ | ⚠️ |
| `kylehowells/demucs-mlx-swift` | 12 | 2026-03-16 | unset | ❌ | ❌ MLX | ❌ | ✅ Apple-only | ✅ | ⚠️ |
| `gianlourbano/demucs-onnx` | 5 | 2024-10-21 | unset | ❌ | ✅ ORT-web | ✅ Vite demo | ❌ | ✅ | ❌ stale |
| `bengfarrell/demucs-wasm` | 3 | 2025-01-29 | MIT | ❌ | ❌ wasm wrapper | ⚠️ | ❌ | ✅ | ❌ stale |
| `facebookresearch/demucs` | 10 110 | 2024-04-24 | MIT | ✅ `demucs` (211k/mo) | ❌ | ❌ | ❌ | ✅ source-of-truth | ❌ stale |
| `deezer/spleeter` | 28 218 | 2025-04-02 | MIT | ✅ `spleeter` (25k/mo) | ❌ TF | ❌ | ❌ | ❌ uses U-Net | ⚠️ |
| `nomadkaraoke/python-audio-separator` | 1 195 | 2026-05-18 | MIT | ✅ `audio-separator` (411k/mo) | ⚠️ ONNX for MDX, **not** htdemucs | ❌ | ❌ | ⚠️ Demucs via PyTorch | ✅ very active |
| `Anjok07/ultimatevocalremovergui` | 24 737 | 2025-03-13 | MIT | ❌ GUI | ⚠️ ONNX for MDX/UVR, **not** htdemucs | ❌ | ❌ | ⚠️ Demucs via PyTorch | ⚠️ |
| `bytedance/music_source_separation` | 1 383 | 2024-04-18 | other | ❌ | ❌ | ❌ | ❌ | ❌ (uses BS-RoFormer) | ❌ stale |
| `ZFTurbo/Music-Source-Separation-Training` | 1 341 | 2026-04-23 | MIT | ❌ | ❌ training | ❌ | ❌ | ⚠️ trains many archs | ✅ |
| `tsurumeso/vocal-remover` | 1 748 | 2024-07-23 | MIT | ❌ | ❌ | ❌ | ❌ | ❌ own arch | ❌ stale |
| `lucidrains/BS-RoFormer` | 815 | 2026-02-01 | MIT | ✅ `BS-RoFormer-pytorch` (low) | ❌ | ❌ | ❌ | ❌ different arch | ✅ |
| `chenmozhijin/BSRoformer.cpp` | 18 | 2026-02-17 | MIT | ❌ ggml | ❌ | ❌ | ⚠️ | ❌ different arch | ✅ |
| `stemrollerapp/stemroller` | 3 104 | 2026-02-25 | unset | ❌ Electron | ❌ bundled PyTorch | ❌ | ❌ desktop | ✅ uses demucs | ⚠️ |
| `JeffreyCA/spleeter-web` | 544 | 2026-05-21 | MIT | ❌ Django app | ❌ | ⚠️ server-side | ❌ | ✅ via demucs PyTorch | ✅ |

Cloud-only competitors (reference points, not comparable):
`lalal.ai`, `moises.ai`, `mvsep.com`. Closed source, hosted-only,
orthogonal to the OSS package strategy.

---

## 3. Per-category narrative

### A. Direct alternatives — "Demucs exported to ONNX / mobile-friendly format"

**Leader (in stars):** `sevagh/demucs.onnx` at 63 ⭐, last commit
2026-02. **Leader (in HF downloads):** `MansfieldPlumbing/Demucs_v4_TRT`
at 635 downloads/month on the model card.

Across this category, **every project except `MansfieldPlumbing` and
ours moves STFT/iSTFT outside the model graph** (sevagh, Intel,
gianlourbano, headfifong all do this; dexxdean and john-rocky do the
CoreML-equivalent workaround). The "STFT-outside" approach is
strictly easier to export but is a meaningful inference cost on
GPU/TensorRT backends because the FFT cannot fuse with the
surrounding convs (MansfieldPlumbing's docs make this argument
explicitly, citing 25-30× throughput improvement on TRT FP16 from
fusing FFT into the graph). Our export does STFT-inside-the-graph
*and* parity-verifies to 1.6 × 10⁻⁴, *and* ships pre-built models
*and* a pip package — that triple is genuinely unique.

The Apple-only MLX wedge (`andrade0`, `ssmall256`,
`kylehowells/demucs-mlx-swift`, `jasonvassallo/demucs-htdemucs-mlx`
on HF) is small (≤25 ⭐ each) but growing — three different MLX
attempts shipped in the last 90 days. MLX is faster than CoreML on
M-series for transformer workloads, but is Apple-only and not
pip-friendly. We can plausibly ignore this wedge unless StemSplit's
Mac App Store strategy needs it.

There is no working "Demucs in onnxruntime-web" project with more
than 5 stars. The browser path is genuinely uncontested.

### B. Adjacent stem-separation toolkits

Three meaningful clusters:

**Cluster 1 — End-user GUI / wrapper apps (≥ 1 000 ⭐).**
`Anjok07/ultimatevocalremovergui` (24 737 ⭐, stale since
2025-03), `stemrollerapp/stemroller` (3 104 ⭐, last commit
2026-02), `JeffreyCA/spleeter-web` (544 ⭐, active). These are
end-user tools, not SDKs. We do not compete with them — but
their users *do* use our underlying models if we land in the
wrappers they call.

**Cluster 2 — Python wrappers (≥ 1 000 ⭐).**
`nomadkaraoke/python-audio-separator` (1 195 ⭐, 411 661 PyPI
downloads/month, **most active project in the entire category**)
is the de facto Python SDK for music source separation. It ships
ONNX support for MDX-Net / UVR-MDX / VR / MDXC models, but its
Demucs backend still imports PyTorch + the official `demucs`
package. **This is the single largest funnel a v0.4 PR could
unlock** — see recommendation #2.

**Cluster 3 — Upstream / research (4 000-30 000 ⭐).**
`deezer/spleeter` (28 218 ⭐) is the OG but architecturally older
(2-stem / 4-stem U-Net, no transformer). `facebookresearch/demucs`
(10 110 ⭐) is the upstream PyTorch project; last commit was
2024-04 and there are 3 multi-year-open issues asking for ONNX
export (see Recommendation #1). `bytedance/music_source_separation`
(1 383 ⭐) is BS-RoFormer paper code, stale. `lucidrains/BS-RoFormer`
(815 ⭐) is a third-party PyTorch implementation of the SOTA
architecture and is actively maintained.

### C. Web-deployable stem separators

This is the **weakest** competitive category in the entire landscape,
which makes it our largest opportunity.

Server-side Gradio Spaces dominate the HF Spaces listing —
`akhaliq/demucs` (204 likes), `akhaliq/Music_Source_Separation`
(68 likes), `nakas/demucs_playground` (62 likes),
`ahk-d/Spleeter-HT-Demucs-Stem-Separation-2025` (39 likes,
created in 2025 and still gaining likes). **All of them run
PyTorch on the Space's CPU/GPU**. None run client-side via
`onnxruntime-web` or WebGPU.

Standalone browser demos: `gianlourbano/demucs-onnx` (5 ⭐, last
commit 2024-10) is the only published `onnxruntime-web` attempt
and it's stale. `bengfarrell/demucs-wasm` (3 ⭐, 2025-01) is a
wrapper, not a deployment. `stemrollerapp/stemroller` (3 104 ⭐)
is Electron + bundled Python, not browser-native.

Cloud competitors (`lalal.ai`, `moises.ai`, `mvsep.com`) are
closed-source SaaS — not in scope for OSS package strategy.
StemSplit.io is the in-house equivalent.

**Gap we can fill cheaply:** a static HF Space + a copy-paste
WebGPU snippet. No one will be ahead of us if we ship this in
v0.4.

### D. Other ONNX export tools for audio ML

`microsoft/onnxruntime` itself has audio examples (Whisper,
wav2vec2) but no music-source-separation example. `pytorch/audio`
has ONNX bits for tracing simple modules but nothing for
fused-MHA + complex STFT. `Intel/openvino-plugins-ai-audacity`
includes the only production C++ deployment of a Demucs export
inside a major desktop DAW; their `htdemucs_fwd.xml` IR is
OpenVINO, not ONNX, but the export strategy (strip STFT outside)
is the same pattern as `sevagh`.

There is **no published reusable "PyTorch ONNX-export fixers"
library** for fused-MHA, complex STFT, `fractions.Fraction`, or
`random.randrange` blockers — our `demucs_onnx.export.*` modules
are, as far as we can find, the only grep-able place these fixes
live as importable functions. That's a small but real piece of
positioning we can mine in docs/blog posts (see Tactic 6 below).

---

## 4. Where we are unique vs not

### Genuinely uncontested ("we are the only ones who do X")

- **Only pip-installable Python package for HT-Demucs via ONNX with
  no PyTorch at inference.** Verified: `pip search` (and the manual
  audit above) finds no peer. `demucs`, `audio-separator`,
  `spleeter` all require either PyTorch or TensorFlow.
- **Only project shipping pre-built ONNX models for `htdemucs_ft`
  (the 4-stem fine-tuned bag) on HF Hub** with parity verification
  to < 1.7 × 10⁻⁴ and an accompanying `infer.py`. The 6 sibling
  attempts (`gentij/htdemucs-ort`, `ModernMube/HTDemucs_onnx`,
  `smank/htdemucs-onnx`, `kjcpc/htdemucs-onnx`, `Kani95/htdemucs-ft-ort`,
  `rubeniskov/htdemucs-ort`) all sit at **0 downloads** with no
  model card.
- **Only 6-stem (`htdemucs_6s`) ONNX with internalized STFT that
  runs cross-platform via ORT.** `MansfieldPlumbing/Demucs_v4_TRT`
  ships a 6-stem internalized-STFT ONNX but their inference path is
  TRT-only and Windows-only; on other platforms their ONNX still
  needs a runtime, and they don't ship one.
- **Only `demucs-onnx browser-demo` / `browser-config` CLI** — no
  other project scaffolds a working `onnxruntime-web` demo into a
  directory.
- **Only project that exposes the four export blockers as
  independent importable modules** (`stft.py`, `mha.py`,
  `pos_embed.py`, `segment.py`) — useful framing for anyone
  exporting *other* transformer/STFT models.

### Where we are NOT first / NOT best

- **STFT-inside-the-graph for HT-Demucs ONNX: `MansfieldPlumbing`
  shipped this first** (commit 2026-02-18, three months before
  us). Their docs explicitly call out why it matters for TRT
  fusion. We need to be careful not to claim "first" on this
  axis in our README — claim "first pip-installable" instead.
- **Production deployment in a real DAW: `Intel/demucs-openvino`
  is in the Audacity AI plugin** (1 987 ⭐). They will have a
  far larger end-user base than our package for a long time.
- **End-user reach: `Anjok07/UVR` (24 737 ⭐) and `stemroller`
  (3 104 ⭐)** dwarf anything we will plausibly do as an SDK.
  The path to those users is plugging *into* them, not competing.
- **Apple-native speed: `andrade0/demucs-mlx` claims 30× realtime
  on M4 Max** because MLX is genuinely faster than CoreML for
  transformer workloads. Our CoreML EP path doesn't match this.
- **Total Python distribution: `audio-separator` ships 14 616
  downloads/day**, `demucs` ships 9 149/day, `spleeter` ships
  1 162/day. We are at zero. The fastest catch-up is
  Recommendation #2.
- **Sheer model coverage: `audio-separator` supports every UVR /
  MDX / BS-RoFormer / Demucs flavor in one wrapper.** We only do
  the HT-Demucs family. Expanding to BS-RoFormer (Recommendation
  #5) closes part of that gap.

---

## 5. Top strategic recommendations (ranked by impact-per-day)

### #1 — Drop comments on the 3 open Demucs ONNX issues, list ourselves in `awesome-onnx*`

- **What:** Post a polite, technical, link-rich comment on each of
  `facebookresearch/demucs#281` ("Using onnx models", 2022),
  `#530` ("export model to onnx without FFT/IFFT", 2023), and
  `#539` ("onnx conversion", 2023). Then submit a PR adding
  `demucs-onnx` to `onnx/awesome-onnx`, `microsoft/onnxruntime`'s
  awesome list, and `topics/audio-to-audio` on HF.
- **Why it matters:** These three Demucs issues are the top organic
  Google result for "demucs onnx" — they've been open for 2-4 years
  with no working answer. Any developer searching the problem lands
  on them. A single comment with our pip-install one-liner converts
  every future visitor into a potential user with **zero ongoing
  cost**.
- **Effort:** 0.5 day total.
- **Duplication risk:** None. No competitor has commented (verified
  via the issue thread).
- **Bundle:** Ship this *today*, independent of v0.4.

### #2 — PR `nomadkaraoke/python-audio-separator` to use `demucs-onnx` as the Demucs backend

- **What:** Add an opt-in `demucs_onnx_backend` to `audio-separator`'s
  `separator/architectures/demucs_separator.py` that, when chosen,
  uses our pip package instead of importing PyTorch + the `demucs`
  package. Default stays PyTorch; advertise the new backend in the
  README as the "no-torch, smaller-footprint" path.
- **Why it matters:** `audio-separator` shipped **411 661 PyPI
  downloads in the last month** vs our zero. They are the dominant
  Python wrapper for music source separation. Removing PyTorch
  from their dependency closure (saves ~2 GB install) is a
  user-visible win for them and a 14 616-DLs/day funnel for us. The
  maintainer is active (last commit 2026-05-18); this kind of
  contributor-friendly PR is likely to merge.
- **Effort:** 2-3 days (write the adapter, run their test matrix,
  draft the PR, iterate with the maintainer).
- **Duplication risk:** None. `audio-separator` does not currently
  have an ONNX Demucs path — they only have ONNX for MDX/UVR/VR
  architectures. We'd be filling a real gap.
- **Bundle:** v0.4 (target).

### #3 — Static HF Space: `StemSplitio/demucs-onnx-web` running entirely client-side via `onnxruntime-web`

- **What:** Take our `examples/browser-react/` scaffold, deploy as
  a Static HF Space. User uploads a song, browser downloads the
  166 MB fp16weights vocals model from our HF repo, runs the chunked
  inference in WASM (multi-thread), returns the karaoke instrumental
  — all without a server.
- **Why it matters:** None of the 200+ "demucs" Spaces on HF run
  client-side. We become the first. Spaces directory traffic is
  significant and persistent (the top demucs Space has 204 likes
  and has been crawled by Google for years). It's also a perfect
  demo for the StemSplit landing page.
- **Effort:** 1-2 days (browser code already exists in v0.3; new
  work is just deploying as a Static Space, polishing UX, adding a
  COOP/COEP-friendly multi-thread fallback).
- **Duplication risk:** Zero today. **Time-sensitive** — somebody
  could plausibly do this in a weekend now that v0.3 is public.
- **Bundle:** v0.4 (target) or parallel HF play.

### #4 — Add `WebGPU` execution provider support to the browser path

- **What:** ORT-Web has supported the `webgpu` EP since 1.17. Add
  `executionProviders: ["webgpu", "wasm"]` and graph-optimization
  flags to our browser scaffolds; benchmark and document the
  speedup vs `wasm` on Chrome 121+ / Edge / desktop Safari.
- **Why it matters:** Browser WASM is ~3× slower than ORT-CPU
  desktop. WebGPU brings most of that gap back. **No competitor
  has shipped a working Demucs WebGPU demo** (gianlourbano's repo
  mentions webgpu in the description but the actual code uses
  wasm). Performance story for the v0.4 launch post.
- **Effort:** 2-3 days including the benchmark write-up.
- **Duplication risk:** Low but rising — WebGPU adoption is the
  buzz item of 2026 in browser ML. We have a 6-month window before
  this becomes table stakes.
- **Bundle:** v0.4 (target).

### #5 — Ship an ONNX export of `BS-RoFormer` / `Mel-Band-RoFormer`

- **What:** Extend `demucs_onnx.export` (or a sibling package
  `bs-roformer-onnx`) to export `lucidrains/BS-RoFormer`. BS-RoFormer
  is the current SOTA on the MDX leaderboard (used by `mvsep`,
  cited in every paper after 2024) and the wrappers
  (`audio-separator`, `Music-Source-Separation-Training`) all want
  it as ONNX. `chenmozhijin/BSRoformer.cpp` (18 ⭐) is GGML, not
  ONNX. `ostinsolo/BS-RoFormer-freeze` (1 ⭐) is PyInstaller, not
  ONNX. Nobody has shipped working ONNX export for this
  architecture.
- **Why it matters:** Closes the "we only do htdemucs" gap that's
  the biggest argument someone would have against adopting our
  package. Makes us the canonical "give me stems as ONNX" tool,
  not just "give me Demucs as ONNX". Big content angle.
- **Effort:** 5-7 days (BS-RoFormer has its own export quirks:
  rotary positional encoding, banded attention, complex spectral
  output. Similar surface area to the four blockers we already
  solved.)
- **Duplication risk:** Low. The `BS-RoFormer` author
  (`lucidrains`) is famously focused on PyTorch implementations
  and rarely ships export tooling himself. The 4 mid-sized
  attempts on GitHub all moved to GGML or PyInstaller, not ONNX.
- **Bundle:** v0.5 or sibling package
  `bs-roformer-onnx` under StemSplitio.

### Honorable mentions (not in top 5 but cheap)

- **INT8 quantization of `htdemucs_ft` + a `*_int8.onnx` variant
  on each HF repo** (1-2 days). Halves model size from 316 MB to
  ~80 MB per stem with measurable but tolerable SDR loss. Only
  `kjcpc/htdemucs_GGML_int8` has tried — but that's GGML, not
  ONNX, and zero downloads.
- **Submit `demucs-onnx` to the `huggingface/transformers.js`
  audio support tracker** — they don't have a music separation
  example yet.
- **Reach out to `Intel/openvino-plugins-ai-audacity` maintainers**
  about a future ORT-EP backend option alongside OpenVINO. Reduces
  Intel-hardware lock-in for that plugin.

---

## 6. Specific tactics for visibility / citation

These are concrete, named actions — not "improve SEO".

1. **`facebookresearch/demucs#281, #530, #539`** — drop a 6-line
   comment on each with a working `pip install demucs-onnx` +
   parity number + link. Highest leverage single action available.
2. **`nomadkaraoke/python-audio-separator` PR** (see Rec #2).
   Even if rejected, the PR thread itself becomes a high-Google
   landing page for "audio separator demucs onnx".
3. **`sevagh/demucs.onnx` issues #2, #6, and the closed #5** —
   relevant users (one asks about iOS RAM usage, one about UVR
   support, one about FP16). Drop a constructive comment with our
   pip-install solution. `sevagh` himself is a respected author
   in this space (also wrote `demucs.cpp` and `umx.cpp`); a public
   acknowledgement that someone solved STFT-in-graph as a Python
   package is good for both repos.
4. **Comment on the 3 open MDX/Demucs export issues on
   `Anjok07/ultimatevocalremovergui`** (search their issues for
   "onnx demucs"). UVR users are the highest-intent music
   separation audience anywhere.
5. **HF Space deployment** (see Rec #3) — but also: convert our
   benchmark dataset `StemSplitio/stem-separation-benchmark-2026`
   into a leaderboard Space so search ranks both the dataset and
   the package together for "vocal separator benchmark".
6. **Blog posts to write** (StemSplit blog + repurpose on dev.to /
   Hashnode via our existing publishing agents):
   - *"The 4 things that stop `torch.onnx.export` from working on
     HT-Demucs, and how we fixed each one"* — the canonical "I
     googled demucs onnx and found this" landing page. Re-use the
     existing README section as the spine.
   - *"Running HT-Demucs in the browser at 0.5× realtime — without
     a server"* — the v0.4 WebGPU launch post.
   - *"`demucs-onnx` vs `sevagh/demucs.onnx` vs
     `Demucs_v4_TRT` vs `htdemucs-coreml` — when to use which"*
     — definitive comparison, drives "demucs production" queries.
   - *"How to add `demucs-onnx` as a backend to
     `nomadkaraoke/python-audio-separator`"* — companion to the
     PR; tags the maintainer; doubles as a tutorial.
7. **HF Spaces to build** (beyond Rec #3):
   - **"Karaoke Maker (browser-only)"** — uses our `--karaoke`
     mix-down logic client-side.
   - **"Acapella Extractor (browser-only)"** — single-stem vocals,
     fastest model.
   - **"HT-Demucs vs BS-RoFormer benchmark"** — uploads a song,
     runs both, lets the user A/B (after Rec #5).
8. **Papers that cite Demucs and care about deployment** — search
   Semantic Scholar for "HT-Demucs production" or
   "music source separation mobile"; the top 10 cite either the
   original paper or `sevagh/demucs.cpp`. Email the corresponding
   authors a 3-line "we shipped a pip-installable Python ONNX
   path" note. Cheap goodwill, can turn into citations.
9. **`pip install`-able from `requirements.txt` of the top 20
   Demucs-using repos on GitHub** — search `code "import demucs"
   filename:requirements.txt`. PR each to suggest an opt-in
   `demucs-onnx` alternative for users without a GPU.
10. **HF org card on `StemSplitio`** — make sure the org pinned
    section prominently lists `demucs-onnx` as the canonical entry
    point. Right now the org card is set up for the model repos
    only.

---

## Appendix — data sources

- GitHub data: `gh repo view <repo> --json
  stargazerCount,forkCount,pushedAt,licenseInfo,createdAt`, run
  2026-05-21.
- PyPI download counts: `https://pypistats.org/api/packages/<pkg>/recent`
  (`last_day`, `last_week`, `last_month`).
- HF Hub model lists: `https://huggingface.co/api/models?search=<q>&sort=downloads`.
- HF Hub Spaces: `https://huggingface.co/api/spaces?search=<q>&sort=likes`.
- Individual HF model cards fetched live for the four shortlisted
  competitors (`MansfieldPlumbing/Demucs_v4_TRT`,
  `dexxdean/htdemucs-coreml`, `Intel/demucs-openvino`,
  `kjcpc/htdemucs-onnx`).
