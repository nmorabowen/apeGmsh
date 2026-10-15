"""Element ``fem_eid`` ↔ ops-tag relabel for MPCO/Ladruno reads (ADR 0043 slice 1.3).

MPCO (STKO) and ``.ladruno`` files key element results by the global
OpenSees ops tag — the bucket ``ID`` dataset. The apeGmsh results API
speaks ``fem_eid``. Only for a model whose fem element ids are dense and
1-based do the two coincide (ops tags are allocator-assigned 1-based in
element order — see
:class:`apeGmsh.opensees._internal.tag_allocator.TagAllocator`). They
diverge in two common cases:

* ``g.compose`` ([ADR 0038](../../../../architecture/decisions/0038-compose-model-composition.md))
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

The archive describes the files, or it does not
-----------------------------------------------
Each reader the translator is attached to declares the ops tags its
file records (:meth:`ElementTagTranslator.declare_recorded`; a
multi-partition set declares every partition before any read). The
archive *describes* the files when every recorded ops tag is an element
the archive emitted: one it pairs with a ``fem_eid``, or one it emitted
without a mesh id (``fem_eid`` sentinel, e.g. interface ``zeroLength``
pairs, addressed by their ops tag). The decision is per result set, not
per read or per bucket.

* **Described** (a bridge-emitted ``model.h5`` beside its own run):
  every id translates on its own (#2066). A filter's paired ``fem_eid``
  reads its own ops tag whatever else the filter holds; a sentinel
  element is read by its ops tag; the returned index relabels id by id.
  A requested id that is no element of the archive, or an element the
  files did not record, is not read and
  :class:`WarnElementFilterUnread` names it. A filter id is never read
  as an ops tag in its place (that read another element's row).
* **Not described** (an *unrelated* stub ``model.h5`` passed only to
  satisfy ``model_h5=``, the pattern tests use): the archive is not
  about these files, so both arrays of every read pass through
  untouched, as before the relabel landed. :class:`Results` already
  warns :class:`WarnElementTagPairingMissing` for that join when the
  bound ``FEMData`` speaks ``fem_eid``.

A drifted archive (one from a slightly different run) is judged by the
same rule: it describes the files or passes through. Telling drift from
an unrelated stub needs provenance the archive does not carry (ADR 0043
§slice 1.3).
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

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
    ``model_h5=``), whose reads then pass through untranslated.
    Element ids in filters (``pg=`` / ``ids=`` / …) and the returned
    ``element_index`` then cannot be translated between the file's ops
    tags and the snapshot's ``fem_eid``s; if the two id spaces differ
    (typical for gmsh solid models, whose 2-D boundary elements consume
    the low ids), element results are silently attributed to the wrong
    elements or dropped. Nodal reads are unaffected. Pass ``model_h5=``
    (the bridge-emitted model archive) to attach the exact pairing.
    """


class WarnElementFilterUnread(UserWarning):
    """An element id filter names ids the read cannot return (#2066).

    Raised (as a warning) by an MPCO or ``.ladruno`` element read whose
    bound ``model.h5`` describes the result files, when the filter holds
    an id that is no element of the archive, or an element of the
    archive whose ops tag the files never recorded (a recorder
    ``region`` that left it out). The read returns the other ids' rows;
    the named ids get none, and no id is read as an ops tag in their
    place.
    """


class ElementTagTranslator:
    """Bidirectional ``fem_eid`` <-> ops-tag map for one result set.

    ``to_ops`` / ``to_fem`` are all-or-nothing array maps (DomainCapture
    uses them). Reader reads go through :meth:`read_translation`, which
    needs the result files' recorded ops tags, declared with
    :meth:`declare_recorded` (see the module docstring).
    """

    __slots__ = (
        "_fem_to_ops", "_ops_to_fem", "_ops_only", "_recorded_parts",
        "_recorded", "_described", "_arrays",
    )

    def __init__(
        self,
        fem_to_ops: "dict[int, int]",
        ops_to_fem: "dict[int, int]",
        ops_only: "Iterable[int]" = (),
    ) -> None:
        self._fem_to_ops = dict(fem_to_ops)
        self._ops_to_fem = dict(ops_to_fem)
        # Ops tags the archive emitted without a mesh id (fem_eid
        # sentinel): such an element is addressed by its ops tag.
        self._ops_only = frozenset(int(t) for t in ops_only)
        clash = self._ops_only & self._ops_to_fem.keys()
        if clash:
            raise ValueError(
                f"ElementTagTranslator: ops tags {sorted(clash)[:10]} are "
                "both paired with a fem_eid and listed without one."
            )
        # Ops tags each attached result file records, one array per file
        # (``None`` until a reader declares one), and their sorted union,
        # built once on the first read: a union per declaration would
        # re-sort the whole set for every partition (quadratic).
        self._recorded_parts: "Optional[list[ndarray]]" = None
        self._recorded: "Optional[ndarray]" = None
        self._described: "Optional[bool]" = None  # cache of describes_files
        # Sorted key/value arrays for the vectorised per-id maps, built
        # on the first read (DomainCapture never needs them).
        self._arrays: "Optional[tuple[ndarray, ...]]" = None

    @classmethod
    def from_model(cls, model: Any) -> "ElementTagTranslator":
        """Build from a bound :class:`OpenSeesModel`'s element records.

        Each :class:`ElementRecord` carries ``tag`` (ops) and ``fem_eid``
        (mesh). A sentinel ``fem_eid < 0`` (``MISSING_FEM_ELEMENT_ID``)
        row has no mesh id, so its ops tag joins ``ops_only``. A model
        with no element records yields an empty translator.
        """
        fem_to_ops: dict[int, int] = {}
        ops_to_fem: dict[int, int] = {}
        ops_only: set[int] = set()
        elements_attr = getattr(model, "elements", None)
        # ``OpenSeesModel.elements`` is a method returning the records;
        # tolerate either a callable or an already-materialised iterable.
        elements = elements_attr() if callable(elements_attr) else elements_attr
        if elements is not None:
            for rec in elements:
                fem = getattr(rec, "fem_eid", None)
                tag = getattr(rec, "tag", None)
                if tag is None:
                    continue
                if fem is None or int(fem) < 0:
                    ops_only.add(int(tag))
                    continue
                fem_to_ops[int(fem)] = int(tag)
                ops_to_fem[int(tag)] = int(fem)
        return cls(fem_to_ops, ops_to_fem, ops_only - ops_to_fem.keys())

    @property
    def is_empty(self) -> bool:
        """True when no pairing is known — every translation is a no-op."""
        return not self._fem_to_ops

    def covers_fem(self, fem_ids: "ndarray") -> bool:
        """True when the pairing describes **every** given ``fem_eid``, so
        :meth:`to_ops` translates them all. An empty id array or empty
        pairing is ``False``."""
        arr = np.asarray(fem_ids, dtype=np.int64)
        if arr.size == 0 or not self._fem_to_ops:
            return False
        return all(int(x) in self._fem_to_ops for x in arr)

    def to_ops(self, fem_ids: "Optional[ndarray]") -> "Optional[ndarray]":
        """Translate a ``fem_eid`` filter to ops tags (all-or-nothing)."""
        return self._relabel(fem_ids, self._fem_to_ops)

    def to_fem(self, ops_ids: "Optional[ndarray]") -> "Optional[ndarray]":
        """Relabel an ops-tag ``element_index`` to ``fem_eid``s (all-or-nothing)."""
        return self._relabel(ops_ids, self._ops_to_fem)

    def declare_recorded(self, ops_tags: "ndarray") -> None:
        """Declare the ops tags one attached result file records.

        Each reader calls this from ``attach_tag_map``. A
        multi-partition set attaches every partition before any read,
        so the union is the whole set's recorded elements. The union is
        taken once, on the first read.
        """
        if self._recorded_parts is None:
            self._recorded_parts = []
        self._recorded_parts.append(
            np.asarray(ops_tags, dtype=np.int64).ravel(),
        )
        self._recorded = None
        self._described = None

    def describes_files(self) -> bool:
        """True when every recorded ops tag is an element of the archive
        (paired, or emitted without a mesh id): see the module docstring.
        """
        recorded = self._require_recorded()
        if self._described is None:
            _, _, ops_keys, _, ops_only = self._sorted_arrays()
            known = np.concatenate([ops_keys, ops_only])
            self._described = bool(np.isin(recorded, known).all())
        return self._described

    def read_translation(
        self, fem_filter: "Optional[ndarray]",
    ) -> "tuple[Optional[ndarray], Callable[[ndarray], ndarray]]":
        """Per-read pair ``(bucket_filter, index_to_fem_fn)`` (#2066).

        When the archive does not describe the attached files (an
        unrelated stub ``model_h5=``), both arrays pass through
        untouched. When it does, each id translates on its own:

        * a filter's paired ``fem_eid`` reads its own ops tag, whatever
          else the filter holds; a sentinel element's ops tag reads
          itself;
        * a filter id that is no element of the archive, or an element
          the files did not record, is left out of the bucket filter
          and :class:`WarnElementFilterUnread` names it. It is never
          read as an ops tag, which would return another element's row;
        * the returned index relabels ops tag by ops tag.
        """
        if not self.describes_files():
            return fem_filter, _identity
        if fem_filter is None:
            return None, self._index_to_fem
        fem_keys, fem_vals, _, _, ops_only = self._sorted_arrays()
        ids = np.asarray(fem_filter, dtype=np.int64).ravel()
        paired, ops = _lookup(fem_keys, fem_vals, ids)
        known = paired | np.isin(ids, ops_only)
        recorded = np.isin(ops, self._require_recorded())
        unknown = ids[~known]
        unrecorded = ids[known & ~recorded]
        if unknown.size or unrecorded.size:
            _warn_unread(unknown.tolist(), unrecorded.tolist())
        return ops[known & recorded], self._index_to_fem

    def _index_to_fem(self, index: "ndarray") -> "ndarray":
        """Relabel a returned ops-tag index id by id; a sentinel element
        keeps its ops tag."""
        _, _, ops_keys, ops_vals, _ = self._sorted_arrays()
        arr = np.asarray(index, dtype=np.int64)
        return _lookup(ops_keys, ops_vals, arr.ravel())[1].reshape(arr.shape)

    def _sorted_arrays(self) -> "tuple[ndarray, ...]":
        """``(fem keys, their ops, ops keys, their fem, ops_only)``, each
        key array sorted, for :func:`_lookup`."""
        if self._arrays is None:
            def _pairs(mapping: "dict[int, int]") -> "tuple[ndarray, ndarray]":
                keys = np.fromiter(mapping, dtype=np.int64, count=len(mapping))
                vals = np.fromiter(
                    mapping.values(), dtype=np.int64, count=len(mapping),
                )
                order = np.argsort(keys)
                return keys[order], vals[order]
            self._arrays = (
                *_pairs(self._fem_to_ops), *_pairs(self._ops_to_fem),
                np.array(sorted(self._ops_only), dtype=np.int64),
            )
        return self._arrays

    def _require_recorded(self) -> "ndarray":
        """The sorted union of every declared file's ops tags, built once
        (one ``np.unique`` over all parts) and cached."""
        if self._recorded_parts is None:
            raise RuntimeError(
                "ElementTagTranslator: no result file declared its "
                "recorded element tags (declare_recorded), so a read "
                "cannot tell whether the model archive describes the "
                "files. Attach the translator through the reader's "
                "attach_tag_map."
            )
        if self._recorded is None:
            self._recorded = np.unique(np.concatenate(self._recorded_parts))
        return self._recorded

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


def _identity(index: "ndarray") -> "ndarray":
    return index


def _lookup(
    keys: "ndarray", vals: "ndarray", ids: "ndarray",
) -> "tuple[ndarray, ndarray]":
    """``(hit, mapped)``: ``hit[i]`` when ``ids[i]`` is in the sorted
    ``keys``; ``mapped[i]`` is its value there, else ``ids[i]`` itself."""
    if keys.size == 0:
        return np.zeros(ids.shape, dtype=bool), ids.copy()
    pos = np.minimum(np.searchsorted(keys, ids), keys.size - 1)
    hit = keys[pos] == ids
    return hit, np.where(hit, vals[pos], ids)


def _warn_unread(unknown: "list[int]", unrecorded: "list[int]") -> None:
    import warnings

    parts = []
    if unknown:
        parts.append(
            f"element ids {_head(unknown)} are no element of the model "
            "archive (model_h5=): no ops tag pairs with them"
        )
    if unrecorded:
        parts.append(
            f"element ids {_head(unrecorded)} are elements of the model "
            "archive that the result files did not record (a recorder "
            "region left them out)"
        )
    warnings.warn(
        "Element read: " + "; ".join(parts) + ". The result holds no "
        "rows for them.",
        WarnElementFilterUnread,
        stacklevel=4,
    )


def _head(ids: "list[int]", n: int = 20) -> str:
    shown = ", ".join(str(i) for i in ids[:n])
    more = f", ... ({len(ids)} in all)" if len(ids) > n else ""
    return f"[{shown}{more}]"
