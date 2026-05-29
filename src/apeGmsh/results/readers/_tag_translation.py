"""Element ``fem_eid`` ↔ ops-tag relabel for MPCO reads (ADR 0043 slice 1.3).

MPCO (STKO) files key element results by the global OpenSees ops tag — the
bucket ``ID`` dataset. The apeGmsh results API speaks ``fem_eid``. For a
dense, uncomposed model the two coincide (ops tags are allocator-assigned
1-based in element order — see
:class:`apeGmsh.opensees._internal.tag_allocator.TagAllocator`), which is
the only reason the ``tag == fem_eid`` convention has held. After
``g.compose`` ([ADR 0038](../../opensees/architecture/decisions/0038-compose-model-composition.md))
bakes per-module base-tag OFFSETS into element ``fem_eid``s, they diverge:
``ops_tag != fem_eid``.

This translator bridges the two using the per-element pairing the bridge
persists at ``/opensees/element_meta/{type}/`` (``ids`` = ops tag,
``fem_eids`` = mesh id), surfaced on the bound :class:`OpenSeesModel`'s
:meth:`elements`. A per-part base-offset map cannot do this — ops tags are
allocator-assigned, not ``fem_eid + offset`` — so the per-element pairing
is the only correct inverse.

Defensive, all-or-nothing
-------------------------
Translation only fires when the bound model describes **every** id in the
array; otherwise the array passes through unchanged. Rationale:

* Non-composed dense models map identically (``ops == fem``), so
  translation is a behavioural no-op either way.
* A real composed model's ``element_meta`` describes every emitted
  element, so every recorded ops tag (and every queryable ``fem_eid``)
  is present → full relabel.
* Tests (and some callers) pair an MPCO file with an *unrelated* stub
  ``model.h5`` purely to satisfy the required ``model_h5=``. Such a stub
  describes none — or worse, a colliding subset — of the file's
  elements. All-or-nothing means a mismatched model never partially
  relabels (which would silently corrupt ``element_index``); it simply
  does nothing, exactly as before the relabel landed.

Strict fail-loud on a partially-known array (the genuine
typo / drift case) is deferred — it needs read-time compose-detection to
distinguish "wrong id" from "deliberately unrelated stub model". Tracked
under ADR 0043 §slice 1.3.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy import ndarray


class ElementTagTranslator:
    """Bidirectional, defensive-passthrough ``fem_eid`` ↔ ops-tag map."""

    __slots__ = ("_fem_to_ops", "_ops_to_fem")

    def __init__(
        self,
        fem_to_ops: "dict[int, int]",
        ops_to_fem: "dict[int, int]",
    ) -> None:
        self._fem_to_ops = dict(fem_to_ops)
        self._ops_to_fem = dict(ops_to_fem)

    @classmethod
    def from_model(cls, model: Any) -> "ElementTagTranslator":
        """Build from a bound :class:`OpenSeesModel`'s element records.

        Each :class:`ElementRecord` carries ``tag`` (ops) and ``fem_eid``
        (mesh). Sentinel ``fem_eid < 0``
        (``MISSING_FEM_ELEMENT_ID``) rows are skipped. A model with no
        element records yields an empty (identity) translator.
        """
        fem_to_ops: dict[int, int] = {}
        ops_to_fem: dict[int, int] = {}
        elements_attr = getattr(model, "elements", None)
        # ``OpenSeesModel.elements`` is a method returning the records;
        # tolerate either a callable or an already-materialised iterable.
        elements = elements_attr() if callable(elements_attr) else elements_attr
        if elements is not None:
            for rec in elements:
                fem = getattr(rec, "fem_eid", None)
                tag = getattr(rec, "tag", None)
                if fem is None or tag is None or int(fem) < 0:
                    continue
                fem_to_ops[int(fem)] = int(tag)
                ops_to_fem[int(tag)] = int(fem)
        return cls(fem_to_ops, ops_to_fem)

    @property
    def is_empty(self) -> bool:
        """True when no pairing is known — every translation is a no-op."""
        return not self._fem_to_ops

    def to_ops(self, fem_ids: "Optional[ndarray]") -> "Optional[ndarray]":
        """Translate a ``fem_eid`` filter to ops tags (all-or-nothing)."""
        return self._relabel(fem_ids, self._fem_to_ops)

    def to_fem(self, ops_ids: "Optional[ndarray]") -> "Optional[ndarray]":
        """Relabel an ops-tag ``element_index`` to ``fem_eid``s (all-or-nothing)."""
        return self._relabel(ops_ids, self._ops_to_fem)

    @staticmethod
    def _relabel(
        ids: "Optional[ndarray]", mapping: "dict[int, int]",
    ) -> "Optional[ndarray]":
        if ids is None:
            return None
        arr = np.asarray(ids, dtype=np.int64)
        if arr.size == 0 or not mapping:
            return arr
        # All-or-nothing: relabel only when the model describes every id;
        # a single unknown id means the bound model does not match this
        # result, so leave the array untouched (see module docstring).
        if not all(int(x) in mapping for x in arr):
            return arr
        return np.array([mapping[int(x)] for x in arr], dtype=np.int64)
