# Export your own checkpoint

This is the canonical guide for exporting any `htdemucs` (or compatible
demucs) checkpoint to ONNX. It walks through **all four blockers** in
vanilla `torch.onnx.export`, with code references into the
`demucs-onnx` source so you can lift any single fix into a different
project.

If you only want to *run* models, you can stop reading and just `pip
install demucs-onnx`. This guide is for people who:

- have a fine-tuned htdemucs checkpoint they want to deploy as ONNX,
- are debugging a custom demucs derivative and hit one of the blockers,
- want to understand exactly what the patches do.

## One-liner first

```bash
pip install 'demucs-onnx[export]'

# Existing pretrained model:
demucs-onnx export htdemucs_ft out/                # all 4 specialists
demucs-onnx export htdemucs    out/htdemucs.onnx   # single-file 4-stem
demucs-onnx export htdemucs_6s out/htdemucs_6s.onnx

# Your own .th file:
demucs-onnx export ./my_finetune.th out/my_finetune.onnx
```

```python
from demucs_onnx.export import export_to_onnx
export_to_onnx("htdemucs_ft", "out/")            # all 4 specialists
export_to_onnx("htdemucs_ft", "drums.onnx", stem="drums")
export_to_onnx("htdemucs_6s", "out/htdemucs_6s.onnx")
```

Under the hood that calls `patch_htdemucs_for_onnx`, runs a numerical
parity check vs the original PyTorch model, then writes the `.onnx`
only if max abs diff < 1e-3. **It refuses to write a numerically wrong
file** — never publish an ONNX model you haven't parity-checked.

---

## The four blockers, explained

These are the four things that break vanilla `torch.onnx.export` on
HT-Demucs as of PyTorch 2.4 / opset 17.

### Blocker 1 — `torch.stft` returns complex tensors

```python
# demucs/htdemucs.py
z = torch.stft(x, n_fft, hop_length, return_complex=True)
# z.dtype == torch.complex64
```

`torch.onnx.export` raises `Exporting STFT does not currently support
complex types`. The dynamo exporter sometimes lowers it, but the
resulting graph fails ORT shape inference downstream.

**Fix** — [`demucs_onnx/export/stft.py`](https://github.com/StemSplit/demucs-onnx/blob/main/src/demucs_onnx/export/stft.py)

Replace `torch.stft` with a `Conv1d` whose kernels are precomputed
sin/cos DFT bases. For `n_fft = 4096`, `hop = 1024`, hann window,
`normalized=True`, you get two real output channels (real and
imaginary) instead of one complex channel.

The inverse: a matching `ConvTranspose1d` plus an `OLA(window²)`
envelope normalisation.

```python
from demucs_onnx.export import RealSTFT, RealISTFT, make_stft_kernels

# Drop-in for `torch.stft(..., return_complex=True)`.
stft = RealSTFT(n_fft=4096, hop_length=1024)
istft = RealISTFT(n_fft=4096, hop_length=1024)

# Or build just the kernels:
real_kernel, imag_kernel, window = make_stft_kernels(n_fft=4096, normalized=True)
```

Verified to 5 × 10⁻⁶ max abs diff against `torch.stft` on real audio.

The patch also overrides demucs's own `_spec`, `_ispec`, `_magnitude`,
and `_mask` methods so the rest of the network sees `(B, C, 2, F, T)`
real tensors throughout — no `view_as_real` / `view_as_complex` calls
survive into the graph.

### Blocker 2 — `model.segment` is a `fractions.Fraction`

```python
# demucs/htdemucs.py
self.segment = Fraction(39, 5)   # = 7.8 seconds
```

`torch._dynamo` allow-lists a small set of "user-defined classes" it
can trace through. `Fraction` is not on it (PyTorch 2.4) and graph
capture crashes. The legacy exporter is more permissive but still
produces a wrong graph because `Fraction` arithmetic is opaque to it
(it materializes the numerator/denominator as Python ints).

**Fix** — [`demucs_onnx/export/segment.py`](https://github.com/StemSplit/demucs-onnx/blob/main/src/demucs_onnx/export/segment.py)

```python
from demucs_onnx.export import coerce_segment_to_float
coerce_segment_to_float(model)
# model.segment is now float(model.segment) — 7.8 instead of Fraction(39, 5)
```

Mathematically identical at inference, side-steps both exporter
limitations.

### Blocker 3 — `random.randrange` in pos-embedding

```python
# demucs/transformer.py — CrossTransformerEncoder._get_pos_embedding
shift = random.randrange(self.sin_random_shift + 1)
# At eval, sin_random_shift = 0 → shift always = 0
```

Used during training for positional-embedding augmentation. At eval
the call is a no-op, but neither the legacy exporter nor dynamo can
trace through `random` — `UnsupportedOperatorError` and graph break,
respectively.

**Fix** — [`demucs_onnx/export/pos_embed.py`](https://github.com/StemSplit/demucs-onnx/blob/main/src/demucs_onnx/export/pos_embed.py)

```python
from demucs_onnx.export import disable_random_pos_shift
disable_random_pos_shift(model)
# CrossTransformerEncoder._get_pos_embedding is replaced with a
# deterministic version that hardcodes shift=0.
```

Mathematically identical at inference. Doesn't affect training (we
patch the model object in-place after `.eval()`).

### Blocker 4 — `aten::_native_multi_head_attention` has no ONNX symbolic

```python
# torch/nn/functional.py — multi_head_attention_forward
# When the fast-path conditions are met, dispatches to:
return torch._native_multi_head_attention(...)
```

`nn.MultiheadAttention` dispatches to a fast fused C++ kernel when its
inputs satisfy a fast-path check. The fused kernel has no ONNX
symbolic — the exporter raises:

```
UnsupportedOperatorError: Exporting the operator
'aten::_native_multi_head_attention' to ONNX opset version 17 is not
supported.
```

This bites only on PyTorch 2.x; in 1.x the slow path was always used.
You can't disable the fast path globally (no public flag), and
monkey-patching `torch._native_multi_head_attention` to `None` causes
training crashes.

**Fix** — [`demucs_onnx/export/mha.py`](https://github.com/StemSplit/demucs-onnx/blob/main/src/demucs_onnx/export/mha.py)

```python
from demucs_onnx.export import onnx_friendly_mha_forward
import torch.nn as nn, types

for m in model.modules():
    if isinstance(m, nn.MultiheadAttention):
        m.forward = types.MethodType(onnx_friendly_mha_forward, m)
```

Replaces the `forward` *per instance* with a manual scaled-dot-product
attention built from `Linear` / `bmm` / `softmax`. The exporter
handles those primitives without complaint. Output is bit-identical to
the fused kernel up to fp32 round-off.

---

## Doing it all at once

The four patches in one call:

```python
from demucs_onnx.export import patch_htdemucs_for_onnx
patch_htdemucs_for_onnx(model)  # mutates in place, returns same model
```

Then `torch.onnx.export(patched_model, dummy_input, "out.onnx", opset_version=17)`
just works.

The full reference is in
[`demucs_onnx/export/exporter.py`](https://github.com/StemSplit/demucs-onnx/blob/main/src/demucs_onnx/export/exporter.py)
— that's also what powers the `demucs-onnx export` CLI.

---

## Sanity checks the export pipeline runs for you

When you call `export_to_onnx(...)` we run **three** checks before
writing the file:

1. **PyTorch parity** — run the unpatched and patched models on the
   same dummy input. Abort if `max_abs_diff > parity_tolerance` (default
   1e-3). This catches bugs in the patches themselves.

2. **`onnx.checker.check_model`** — verifies the graph is structurally
   valid (node count, type consistency, opset compatibility).

3. **ONNX runtime parity** — load the freshly-written ONNX file in
   `onnxruntime` CPU EP and re-run against the PyTorch model. Abort if
   `max_abs_diff > parity_tolerance`.

Bypass with `--no-parity-check` (don't).

```python
from demucs_onnx.export import verify_onnx_parity
diff = verify_onnx_parity("htdemucs_ft", "out/htdemucs_ft_drums.onnx",
                          stem="drums", tolerance=1e-3)
print(f"max abs diff: {diff:.6e}")
```

---

## What this does *not* fix (yet)

- **Dynamic segment length.** The exported graph is bound to
  `(1, 2, 343980)`. Re-export with a different `segment_seconds` if you
  need a different segment length; we don't support dynamic axes
  because they ~3× the model size with no runtime benefit on the EPs
  we target.
- **Streaming inference.** v0.3.0 is segmented; streaming is on the
  v0.4 roadmap.
- **INT8 quantization.** ORT's dynamic INT8 either gives no real
  speedup on htdemucs (MatMul-only, max diff 2e-3) or breaks the model
  (with Conv, max diff 0.70, 2× slower). We ship the fp16weights
  half-storage variant instead — same speed, half the disk.

---

## Found a fifth blocker?

[Open an issue](https://github.com/StemSplit/demucs-onnx/issues) — we
add new patches as the demucs codebase evolves. Especially welcome:

- PRs adding patches for related architectures (`hdemucs`, `mdx_extra`).
- Reports of new exporter failures with PyTorch versions > 2.4.
- Quantization paths that beat our fp16weights without breaking parity.
