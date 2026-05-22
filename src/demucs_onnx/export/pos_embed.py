"""Blocker #3: ``random.randrange`` in the cross-transformer pos-embedding.

``CrossTransformerEncoder._get_pos_embedding`` calls
``random.randrange(self.sin_random_shift + 1)`` to apply a random shift
during *training*. At eval time the demucs config sets
``sin_random_shift = 0`` so the call is a no-op (``randrange(1) == 0``),
but neither :func:`torch.onnx.export` nor :func:`torch.onnx.dynamo_export`
can trace through a call to ``random``: the legacy exporter raises
``UnsupportedOperatorError``, the dynamo exporter raises a graph break.

Fix: monkey-patch the method to a version that hardcodes ``shift = 0``.
This is mathematically identical when ``sin_random_shift = 0`` (which is
always the case during inference) and lets the exporter trace cleanly.
"""
from __future__ import annotations

import types
from typing import Any

import torch
import torch.nn as nn


def disable_random_pos_shift(model: nn.Module) -> nn.Module:
    """Replace ``CrossTransformerEncoder._get_pos_embedding`` with a
    deterministic version that hardcodes ``shift = 0``.

    Also sets ``sin_random_shift = 0`` on every module that has the attr,
    for belt-and-braces when other code paths read it directly.
    """
    # Importing demucs is deferred to keep the inference path torch-free.
    import demucs.transformer as tr

    for m in model.modules():
        if hasattr(m, "sin_random_shift"):
            m.sin_random_shift = 0

    def _get_pos_embedding_no_random(self_: Any, T: int, B: int, C: int,
                                     device: torch.device) -> torch.Tensor:
        # Mirror tr.CrossTransformerEncoder._get_pos_embedding but never
        # call random.randrange. At inference sin_random_shift is 0 so
        # `shift` is always 0 — this branch is exactly equivalent.
        if self_.emb == "sin":
            return tr.create_sin_embedding(
                T, C, shift=0, device=device, max_period=self_.max_period,
            )
        if self_.emb == "cape":
            return tr.create_sin_embedding_cape(
                T, C, B, device=device, max_period=self_.max_period,
                mean_normalize=self_.cape_mean_normalize,
                augment=False,  # eval mode never augments
                max_global_shift=0.0, max_local_shift=0.0, max_scale=1.0,
            )
        if self_.emb == "scaled":
            pos = torch.arange(T, device=device)
            return self_.position_embeddings(pos)[:, None]
        raise RuntimeError(f"unknown emb {self_.emb!r}")

    for m in model.modules():
        if isinstance(m, tr.CrossTransformerEncoder):
            m._get_pos_embedding = types.MethodType(_get_pos_embedding_no_random, m)

    return model
