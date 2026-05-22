"""Blocker #2: ``model.segment`` is a :class:`fractions.Fraction`.

The HT-Demucs config sets ``segment = Fraction(39, 5)`` (= 7.8 seconds).
The segment value is read inside the model's forward and used to derive
chunk lengths.

``torch._dynamo`` allow-lists a small set of "user-defined classes" it
knows how to trace through. ``Fraction`` is not on that list (PyTorch
2.4); the graph capture crashes the moment it hits ``self.segment``. The
legacy exporter is more permissive but still produces a wrong graph
because ``Fraction`` arithmetic is opaque to it.

Coercing to a plain Python ``float`` is mathematically identical at
inference and side-steps the dynamo/exporter limitation entirely.
"""
from __future__ import annotations

from fractions import Fraction

import torch.nn as nn


def coerce_segment_to_float(model: nn.Module) -> nn.Module:
    """Convert ``model.segment`` from a :class:`Fraction` to a ``float``, if needed.

    Idempotent and safe to call on already-coerced models.
    """
    seg = getattr(model, "segment", None)
    if isinstance(seg, Fraction):
        model.segment = float(seg)
    return model
