"""``element_tags=`` for DomainCapture tests whose domains number elements by FEM id.

An element-scoped ``DomainCapture`` refuses without a ``{fem_eid: ops_tag}``
pairing that covers every recorded id. These tests build their OpenSees
domain (a fake, or a hand-built live model) with each element's tag equal
to the id the spec records, so the identity pairing is the truth there.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def identity_tags(spec: Any) -> dict[int, int]:
    """``{e: e}`` for every element id an element-scoped record names."""
    out: dict[int, int] = {}
    for rec in spec.records:
        if rec.category in ("nodes", "modal"):
            continue
        ids = getattr(rec, "element_ids", None)
        if ids is not None:
            out.update((int(e), int(e)) for e in np.asarray(ids).ravel())
    return out
