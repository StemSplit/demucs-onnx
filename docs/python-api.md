# Python API

Auto-generated from docstrings via `mkdocstrings[python]`. For the
hand-curated quick reference, see the [README on GitHub](https://github.com/StemSplit/demucs-onnx#api).

---

## Inference

::: demucs_onnx.separate

::: demucs_onnx.separate_all

::: demucs_onnx.separate_stem

::: demucs_onnx.prewarm

::: demucs_onnx.session_pool

::: demucs_onnx.SessionPool

::: demucs_onnx.list_models

---

## Providers

::: demucs_onnx.auto_select_providers

::: demucs_onnx.describe_runtime

---

## Audio I/O

::: demucs_onnx.load_audio

::: demucs_onnx.write_audio

::: demucs_onnx.write_wav

::: demucs_onnx.write_mp3

---

## Browser helpers

::: demucs_onnx.browser.wasm_config

::: demucs_onnx.browser.print_wasm_config

::: demucs_onnx.browser.write_demo_dir

::: demucs_onnx.browser.model_browser_url

---

## Export pipeline

Requires the `[export]` extra (`pip install 'demucs-onnx[export]'`).

::: demucs_onnx.export.exporter.export_to_onnx

::: demucs_onnx.export.exporter.verify_onnx_parity

::: demucs_onnx.export.patch.patch_htdemucs_for_onnx

### Individual patches

::: demucs_onnx.export.segment.coerce_segment_to_float

::: demucs_onnx.export.pos_embed.disable_random_pos_shift

::: demucs_onnx.export.mha.onnx_friendly_mha_forward

::: demucs_onnx.export.stft.RealSTFT

::: demucs_onnx.export.stft.RealISTFT

::: demucs_onnx.export.stft.make_stft_kernels
