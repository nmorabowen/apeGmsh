"""Element ``fem_eid`` ↔ ops-tag relabel for MPCO/Ladruno reads (ADR 0043 slice 1.3).

MPCO (STKO) and ``.ladruno`` files key element results by the global
OpenSees ops tag — the bucket ``ID`` dataset. The apeGmsh results API
speaks ``fem_eid``. Only for a model whose fem element ids are dense and
1-based do the two coincide (ops tags are allocator-assigned 1-based in
element order — see
:class:`apeGmsh.opensees._internal.tag_allocator.TagAllocator`). They
diverge in two common cases:

* ``g.compose`` ([ADR 0038](../../opensees/architecture/decisions/0038-compose-model-composition.md))
  bakes per-module base-tag OFFSETS into element ``fem_eid``s;
* gmsh numbers lower-dimensional elements first, so **any** solid model
  whose surfaces are meshed (e.g. carries surface physical groups) has
  its 3-D ``fem_eid``s offset by the count of boundary elements — the
  ordinary, uncomposed case.

This translator bridges the two using the per-element pairing the bridge
persists at ``/opensees/element_meta/{type}/`` (``ids`` = ops tag,
``fem_eids`` = mesh id), surfaced on the bound :class:`OpenSeesModel`'s
:meth:`elements`. A per-part base-offset map cannot do this — ops tags are
allocator-assigned, not ``fem_eid + offset`` — so the per-element pairing
is the only correct inverse.

Defensive, all-or-nothing
-------------------------
Translation only fires when the bound model describes **every** id in the
array; otherwise the array passes through unchanged. Additionally, the
two arrays of one read (the incoming filter and the returned index) are
decided **together** via :meth:`ElementTagTranslator.read_translation` —
never independently — because an untranslated filter's selected index can
be (wrongly) fully covered by the reverse map. Rationale:

* Dense 1-based models map identically (``ops == fem``), so translation
  is a behavioural no-op either way.
* A bridge-emitted model's ``element_meta`` describes every emitted
  element, so every recorded ops tag (and every queryable ``fem_eid``)
  is present → full relabel.
* Tests (and some callers) pair an MPCO file with an *unrelated* stub
  ``model.h5`` purely to satisfy the required ``model_h5=``. Such a stub
  describes none — or worse, a colliding subset — of the file's
  elements. All-or-nothing means a mismatched model never partially
  relabels (which would silently corrupt ``element_index``); it simply
  does nothing, exactly as before the relabel landed.

Strict fail-loud on a partially-known array (the genuine typo / drift
case) is deferred — it cannot be told apart from a "deliberately
unrelated stub model" without extra provenance. Tracked under ADR 0043
§slice 1.3.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy import ndarray


class WarnElementTagPairingMissing(UserWarning):
    """An element-level read cannot be relabelled fem_eid ↔ ops-tag.

    Raised (as a warning) when an element-level read is attempted on a
    :class:`Results` whose bound ``FEMData`` was supplied externally
    (``fem=`` / ``.bind(fem)``) while the reader carries no fem_eid↔
    ops-tag pairing **that actually fires** — no pairing at all (e.g.
    ``Results.from_ladruno`` without ``model_h5=``, or a ``model_h5``
    that records no ``element_meta``), or an attached pairing that does
    not cover the file's recorded element ids (an unrelated stub
    ``model_h5=``), which the all-or-nothing contract leaves inert.
    Element ids in filters (``pg=`` / ``ids=`` / …) and the returned
    ``element_index`` then cannot be translated between the file's ops
    tags and the snapshot's ``fem_eid``s; if the two id spaces differ
    (typical for gmsh solid models, whose 2-D boundary elements consume
    the low ids), element results are silently attributed to the wrong
    elements or dropped. Nodal reads are unaffected. Pass ``model_h5=``
    (the bridge-emitted model archive) to attach the exact pairing.
    """


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

    def covers_ops(self, ops_ids: "ndarray") -> bool:
        """True when the pairing describes **every** given ops tag — the
        all-or-nothing condition under which an unfiltered read's
        returned index fully relabels to ``fem_eid``s. An empty id array
        or empty pairing is ``False`` (nothing would relabel)."""
        arr = np.asarray(ops_ids, dtype=np.int64)
        if arr.size == 0 or not self._ops_to_fem:
            return False
        return all(int(x) in self._ops_to_fem for x in arr)

    def to_ops(self, fem_ids: "Optional[ndarray]") -> "Optional[ndarray]":
        """Translate a ``fem_eid`` filter to ops tags (all-or-nothing)."""
        return self._relabel(fem_ids, self._fem_to_ops)

    def to_fem(self, ops_ids: "Optional[ndarray]") -> "Optional[ndarray]":
        """Relabel an ops-tag ``element_index`` to ``fem_eid``s (all-or-nothing)."""
        return self._relabel(ops_ids, self._ops_to_fem)

    def read_translation(
        self, fem_filter: "Optional[ndarray]",
    ) -> "tuple[Optional[ndarray], Any]":
        """Per-read consistent pair ``(bucket_filter, index_to_fem_fn)``.

        A single read must translate **both directions or neither** —
        the two arrays cannot be judged independently. Counter-example
        (the stub-model trap): with pairing ``{fem 1 ↔ ops 2}`` and a
        filter ``[2]``, ``to_ops`` passes the unknown fem id 2 through,
        the bucket then selects ops element 2, and an independent
        ``to_fem`` on the returned index ``[2]`` — which IS fully
        covered — would mis-relabel it to ``[1]``. So:

        * filter fully described by the model → translate it to ops
          tags and relabel the returned index back (the returned index
          is a subset of the translated filter, so the reverse map
          always covers it);
        * filter with any unknown id (unrelated stub model / ids
          already in ops space) → pass the filter AND the returned
          index through untouched;
        * no filter → all-or-nothing :meth:`to_fem` on the full index
          (a real bound model describes every recorded ops tag; an
          unrelated stub with any non-colliding id passes through).
        """
        if fem_filter is None:
            return None, self.to_fem
        arr = np.asarray(fem_filter, dtype=np.int64)
        if all(int(x) in self._fem_to_ops for x in arr):
            return self._relabel(arr, self._fem_to_ops), self.to_fem
        return arr, lambda index: index

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
