"""Reference reader for the bridge's ``model.h5`` archive.

This module ships:

* :class:`SchemaVersionError` / :class:`MalformedH5Error` — explicit
  exception types the viewer team can catch.
* :func:`open` — entry-point that opens a file, performs the strict
  ``major`` schema-version check, and returns an :class:`H5Model`.
* :class:`H5Model` — a thin read-only accessor over the file. The
  viewer team is expected to subclass or wrap this; the per-feature
  methods here cover the minimal contract.

Phase 8 (ADR 0019) — the dict-style accessors (``materials() -> dict``
etc.) have been pruned; every typed accessor formerly named
``*_typed()`` now uses the unsuffixed name.  Callers receive immutable
typed records from :mod:`apeGmsh.opensees._internal.typed_records`.

Schema-version compatibility (ADR 0023):

* Per-zone schema versioning is the authoritative validation rule.
  The bridge zone is validated against
  :func:`schema_version.reader_version(OPENSEES)` via the two-version
  window: any file whose ``opensees_schema_version`` falls inside
  ``[X.(Y-1).*, X.Y.*]`` is accepted; everything else raises
  :class:`SchemaVersionError`.
* Legacy single-stamp files (only ``/meta/schema_version``, no per-zone
  keys) are accepted via the envelope fallback in
  :func:`schema_version.read_zone_version`.
"""
from __future__ import annotations

import builtins
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from .._internal.schema_version import (
    OPENSEES,
    SchemaVersion,
    SchemaVersionError,
    read_zone_version,
    reader_version,
    validate_zone_version,
)
from .._internal.typed_records import (
    BeamIntegrationRecord,
    DeclContext,
    MaterialRecord,
    PatternRecord,
    RecorderRecord,
    SectionComplexRecord,
    SectionSimpleRecord,
    TimeSeriesRecord,
    TransformRecord,
)

if TYPE_CHECKING:
    import h5py

    from apeGmsh._kernel.records._compose import ComposeRecord


__all__ = [
    "H5Model",
    "MalformedH5Error",
    "PartitionEmittedRecord",
    "SchemaVersionError",
    "open",
]


@dataclass(frozen=True, slots=True)
class PartitionEmittedRecord:
    """Read-side view of one ``/opensees/partitions/partition_NN`` group.

    Schema 2.10.0 (ADR 0027).  The reader returns these as the value
    of :meth:`H5Model.partitions`; they mirror the bridge's
    :class:`apeGmsh.mesh._kernel.records._partitions.PartitionRecord`
    (the broker / mesh side) but are intentionally a separate type —
    that record is the unpartitioned-time topology snapshot used by
    the broker to assign rank ownership; this record is the emitted
    rank's per-block emit log captured by :class:`H5Emitter`.  The
    two need not stay aligned beyond the ``rank`` field — e.g. the
    emitted block can carry foreign-declared nodes from
    cross-partition MP constraints that don't appear on the broker's
    side (ADR 0027 §"Decision" INV-2).

    Attributes
    ----------
    rank
        OpenSeesMP rank id this block scopes.  Matches the value
        passed to :meth:`H5Emitter.partition_open`.
    element_ids
        Tuple of OpenSees element tags owned by this rank, in
        emission order.
    node_ids
        Tuple of OpenSees node tags declared on this rank's block,
        deduped within the block.  Includes any foreign-declared
        nodes from replicated MP constraints.
    boundary_node_ids
        Tuple of node tags shared with at least one other rank
        (per-rank set intersected against the union of every other
        rank's node set).  Empty when the model carries only one
        partition.
    """

    rank: int
    element_ids: tuple[int, ...]
    node_ids: tuple[int, ...]
    boundary_node_ids: tuple[int, ...]


class MalformedH5Error(ValueError):
    """Raised when a file is missing the mandatory ``/meta`` group or
    when a required group is structurally invalid (e.g. a compound
    dataset is missing one of its declared fields)."""


def open(path: str, *, meta_path: str = "meta") -> H5Model:
    """Open ``path`` and version-check it.

    Parameters
    ----------
    path
        File system path to a bridge ``model.h5`` (standalone) or a
        composed ``results.h5`` carrying ``/opensees/`` at root.
    meta_path
        HDF5 path to the bridge meta group inside the file.  Default
        ``"meta"`` reads ``/meta`` at root (the standalone shape).
        For ADR 0020 composed results files, pass ``"model/meta"`` so
        the schema check validates the bridge-stamped meta under
        ``/model/`` instead of the results envelope ``/meta`` at
        root.  Cleanup follow-up to the Phase 4 ``/opensees_archive/``
        removal.

    Returns
    -------
    H5Model
        Owns the underlying h5py handle; closes on
        :meth:`H5Model.close` or context-manager exit.

    Raises
    ------
    SchemaVersionError
        If the file's opensees-zone version falls outside the
        two-version window (ADR 0023).  Includes legacy single-stamp
        files whose envelope ``schema_version`` resolves to an
        unsupported version via the envelope-fallback rule.
    MalformedH5Error
        If the meta group at ``meta_path`` is missing entirely, the
        ``schema_version`` attr is empty, or the version string is not
        semver-shaped.
    """
    import h5py
    f = h5py.File(path, "r")
    try:
        meta_key = meta_path.lstrip("/")
        if meta_key not in f:
            raise MalformedH5Error(
                f"{path}: missing /{meta_key} group; not a bridge model.h5"
            )
        meta_attrs = f[meta_key].attrs
        # Empty / missing envelope still surfaces as MalformedH5Error
        # (existing test contract); semver-malformed value too.
        if "schema_version" not in meta_attrs or not str(
            meta_attrs["schema_version"]
        ):
            raise MalformedH5Error(
                f"{path}: /{meta_key}/schema_version attribute is empty"
            )
        try:
            file_version = read_zone_version(meta_attrs, OPENSEES)
        except ValueError as exc:
            raise MalformedH5Error(
                f"{path}: /{meta_key} schema-version attr is not "
                f"semver-shaped: {exc}"
            ) from exc
        if file_version is None:
            raise MalformedH5Error(
                f"{path}: /{meta_key} carries no opensees zone version"
            )
        # ADR 0023 two-version window — refuses too-old / too-new /
        # wrong-major with explicit upgrade-path text.
        #
        # Exception (broker-only neutral files): when the file
        # carries an explicit ``neutral_schema_version`` (the
        # broker-emit marker stamped by ``FEMData.to_h5``) AND no
        # ``/opensees/`` zone is present, treat it as a neutral-only
        # file: every ``/opensees/`` accessor returns an empty list,
        # and the opensees-zone window check is skipped.  Without
        # this carve-out, neutral-only broker output gets refused as
        # "opensees version too old" whenever the neutral and opensees
        # reader minors drift apart — but those files are perfectly
        # valid for the neutral reader path.
        #
        # The tolerance is intentionally narrow: foreign / hand-rolled
        # files that carry only the envelope ``schema_version`` (no
        # ``neutral_schema_version``, no ``/opensees/`` group) still
        # go through the normal window check, so a 2.5.0 stub gets
        # rejected as before.
        opensees_group_key = "opensees"
        has_opensees_zone = opensees_group_key in f
        has_neutral_key = "neutral_schema_version" in meta_attrs
        skip_opensees_window = (
            not has_opensees_zone and has_neutral_key
        )
        if not skip_opensees_window:
            try:
                validate_zone_version(
                    file_version, reader_version(OPENSEES), zone=OPENSEES,
                )
            except SchemaVersionError as exc:
                raise SchemaVersionError(f"{path}: {exc}") from None
        return H5Model(f, path=path, meta_path=meta_key)
    except Exception:
        f.close()
        raise


class H5Model:
    """Read-only accessor over a bridge model.h5 file.

    Construct via :func:`open`. Iterate or use the typed accessors:

    >>> import apeGmsh.opensees.emitter.h5_reader as reader
    >>> m = reader.open("model.h5")
    >>> m.schema_version
    '2.0.0'
    >>> m.meta()["ndm"]
    3
    >>> m.materials()                          # doctest: +SKIP
    [MaterialRecord(type_token='Steel02', tag=1, params=(...)), ...]
    """

    def __init__(
        self, f: h5py.File, *, path: str, meta_path: str = "meta",
    ) -> None:
        self._f: h5py.File = f
        self._path: str = path
        # Normalise to an h5py key (no leading slash) so ``self._f[key]``
        # works whether the caller passed ``/meta`` or ``meta``.  The
        # default ``meta`` keeps the standalone-file case byte-identical.
        self._meta_path: str = meta_path.lstrip("/")

    # -- Lifecycle -------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HDF5 handle."""
        self._f.close()

    def __enter__(self) -> H5Model:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def handle(self) -> h5py.File:
        """Raw h5py file handle — for advanced use beyond this API."""
        return self._f

    @property
    def path(self) -> Path:
        """The on-disk source the reader was opened against.

        ADR 0026 (H5ModelReader Protocol) requires this property so
        adapters can expose their file backing for callers that need
        a path (subprocess argv, error messages, debugging).  Returns
        a :class:`pathlib.Path` regardless of how the source was
        passed to :func:`open` — the stored ``self._path`` is the
        original ``str``.
        """
        return Path(self._path)

    # -- Capability probes (ADR 0026) -----------------------------------

    def has_opensees_orientation(self) -> bool:
        """``True`` iff the file carries the orientation sidecar.

        Both ``/opensees/transforms`` and ``/opensees/element_meta``
        must be present.  This is the gate
        :func:`apeGmsh.viewers.data._h5_probe.has_opensees_orientation`
        applies — the standalone function exists for path-only
        callers; this method is the equivalent on an already-open
        reader (cheaper: no second ``h5py.File`` open).

        Producer-agnostic: both the bridge writer (``apeSees(fem).h5()``)
        and :class:`apeGmsh.opensees.ModelData` write the same
        byte-equivalent zone (ADR 0018 INV-16), so the probe needs
        no provenance check — it asks the file directly.

        Existence of *both* groups is the contract; the join in
        :meth:`element_local_axes_vecxz` requires both.  A file with
        the zones but only ``-1`` sentinel rows still returns
        ``True`` here and degrades gracefully inside the join
        (ADR 0018 INV-11 — "write+degrade on nothing-injected").
        """
        return (
            "opensees/transforms" in self._f
            and "opensees/element_meta" in self._f
        )

    def has_neutral_zone(self) -> bool:
        """``True`` iff the file carries the rich neutral zone.

        The probe re-uses the ``neutral_schema_version`` marker that
        :func:`open` already inspects at construction time
        (``h5_reader.py:157``) to recognise broker-emit files.  An
        apeGmsh-written ``model.h5`` always returns ``True``.  Foreign
        formats whose adapters synthesise a neutral view from the
        bridge zone return ``True`` once the synthesis is complete;
        bridge-only adapters return ``False`` (the viewer degrades
        to fem-only rendering, mirroring the MPCO asymmetry).
        """
        meta_attrs = _attrs_as_dict(self._f[self._meta_path])
        return "neutral_schema_version" in meta_attrs

    # -- Required reads --------------------------------------------------

    @property
    def schema_version(self) -> str:
        version: str = str(self._f[self._meta_path].attrs["schema_version"])
        return version

    def meta(self) -> dict[str, Any]:
        """Return the bridge meta group's attributes as a plain dict.

        Reads from the ``meta_path`` supplied to :func:`open` (default
        ``/meta``).  For composed results files
        (``meta_path="/model/meta"``) this is the bridge meta under
        ``/model/``, not the results envelope at root.
        """
        return _attrs_as_dict(self._f[self._meta_path])

    # -- Optional reads --------------------------------------------------

    def material(self, family: str, name: str) -> dict[str, Any]:
        """Return one material's attrs.

        Raises :class:`KeyError` if missing.
        """
        return _attrs_as_dict(self._f[f"opensees/materials/{family}/{name}"])

    def transform_arrays(self, name: str) -> dict[str, Any]:
        """Return ``{per_element_vecxz?, per_element_emitted_tag?}`` for
        one ``/opensees/transforms/{name}`` group.

        Companion to :meth:`transforms` (which exposes only attrs).
        Raises ``KeyError`` if the transform group is missing.  The H5
        emitter writes one group per ``geomTransf`` call (documented
        schema deviation), so both datasets are single-row — one for an
        explicit-``vecxz`` transform, one per distinct vecxz for an
        orientation-driven fan-out.
        """
        sub = self._f[f"opensees/transforms/{name}"]
        out: dict[str, Any] = {}
        if "per_element_vecxz" in sub:
            out["per_element_vecxz"] = sub["per_element_vecxz"][:]
        if "per_element_emitted_tag" in sub:
            out["per_element_emitted_tag"] = sub["per_element_emitted_tag"][:]
        return out

    def elements(self) -> dict[str, dict[str, Any]]:
        return self._group_attrs_map("elements")

    def analysis(self) -> dict[str, Any] | None:
        a = self._f.get("opensees/analysis")
        if a is None:
            return None
        return _attrs_as_dict(a)

    def element_meta(self) -> dict[str, dict[str, Any]]:
        """Return ``{type_token: attrs}`` for ``/opensees/element_meta``.

        Phase 8.5 added this OpenSees-specific element-metadata zone
        (positional args, cross-references) alongside the broker's
        neutral ``/elements/{gmsh_alias}`` group.  Companion to
        :meth:`element_meta_arrays` for the actual datasets.
        """
        return self._group_attrs_map("opensees/element_meta")

    def element_meta_arrays(self, type_token: str) -> dict[str, Any]:
        """Return ``{ids, fem_eids?, partition_ids?, args?, args_str?}``
        for one element-meta group.

        Raises ``KeyError`` if the type group is missing.

        ``fem_eids`` is the Phase 8.6 mapping that pairs each
        OpenSees tag with the FEM element id it was fanned out from.
        Sentinel value ``-1`` (matches
        :data:`apeGmsh.opensees._internal.tag_resolution.MISSING_FEM_ELEMENT_ID`)
        marks records emitted outside a bridge fan-out.

        ``partition_ids`` is the schema 2.10.0 (ADR 0027) column
        carrying the rank each element was emitted under.  Sentinel
        ``-1`` marks records emitted outside a ``partition_open`` /
        ``partition_close`` bracket.  Optional — pre-2.10.0 archives
        will not carry it.
        """
        sub = self._f[f"opensees/element_meta/{type_token}"]
        out: dict[str, Any] = {"ids": sub["ids"][:]}
        if "fem_eids" in sub:
            out["fem_eids"] = sub["fem_eids"][:]
        if "partition_ids" in sub:
            out["partition_ids"] = sub["partition_ids"][:]
        if "args" in sub:
            out["args"] = sub["args"][:]
        if "args_str" in sub:
            out["args_str"] = sub["args_str"][:]
        return out

    def element_local_axes_vecxz(self) -> dict[int, Any]:
        """Return ``{fem_element_id: vecxz}`` for every beam-column element.

        Joins three zones the archive keeps separate:

        * ``/opensees/transforms/*`` — each group carries the geomTransf
          ``tag`` (``per_element_emitted_tag``) and its ``vecxz``
          reference vector (``per_element_vecxz``).
        * ``/opensees/element_meta/{type_token}`` — each element's
          positional ``args`` tail (one slot is the geomTransf tag) and
          ``fem_eids`` mapping the row back to the broker FEM element id.
        * The OpenSees element vocabulary
          (:mod:`apeGmsh.opensees._element_capabilities`) — tells us
          *which* ``args`` slot holds the transf tag, per element type.

        Keyed by FEM element id so a viewer (which speaks the broker's
        ``/elements/{gmsh_alias}`` ids) can look up the orientation of
        any line element without knowing OpenSees tags.

        Returns an empty dict when the file carries no transforms or no
        element-meta (mesh-only / pre-bridge archives) — never raises
        for a well-formed but enrichment-light file.
        """
        import numpy as np

        transforms_grp = self._f.get("opensees/transforms")
        meta_grp = self._f.get("opensees/element_meta")
        if transforms_grp is None or meta_grp is None:
            return {}

        # emitted geomTransf tag -> vecxz (3,)
        tag_to_vecxz: dict[int, Any] = {}
        for tname in transforms_grp:
            arr = self.transform_arrays(tname)
            vec = arr.get("per_element_vecxz")
            tags = arr.get("per_element_emitted_tag")
            if vec is None or tags is None:
                continue
            vec = np.asarray(vec, dtype=np.float64).reshape(-1, 3)
            tags = np.asarray(tags).reshape(-1)
            for row in range(min(len(tags), len(vec))):
                tag_to_vecxz[int(tags[row])] = vec[row].copy()
        if not tag_to_vecxz:
            return {}

        ndm = int(self.meta().get("ndm", 3) or 3)

        # h5_reader lives in opensees.emitter, so it may consult the
        # element vocabulary directly (the viewer cannot — that's why
        # this join lives here; see ADR 0014 / viewer-integration.md).
        from apeGmsh.opensees._element_capabilities import _ELEM_REGISTRY

        out: dict[int, Any] = {}
        for type_token in meta_grp:
            idx = _transf_arg_tail_index(type_token, ndm, _ELEM_REGISTRY)
            if idx is None:
                continue
            try:
                arrays = self.element_meta_arrays(type_token)
            except KeyError:
                continue
            args = arrays.get("args")
            fem_eids = arrays.get("fem_eids")
            if args is None or fem_eids is None:
                continue
            args = np.asarray(args, dtype=np.float64)
            fem_eids = np.asarray(fem_eids).reshape(-1)
            if args.ndim != 2 or idx >= args.shape[1]:
                continue
            for row in range(min(args.shape[0], len(fem_eids))):
                fem_eid = int(fem_eids[row])
                if fem_eid < 0:        # -1: emitted outside a bridge fan-out
                    continue
                raw = args[row, idx]
                if not np.isfinite(raw):
                    continue
                vec = tag_to_vecxz.get(int(round(float(raw))))
                if vec is not None:
                    out[fem_eid] = vec
        return out

    # -- Typed accessors (ADR 0019) -------------------------------------
    #
    # These return the public dataclasses from
    # ``apeGmsh.opensees._internal.typed_records``.  Phase 8 (ADR 0019)
    # made the typed accessors the sole read surface; the legacy
    # dict-style accessors were pruned.
    #
    # Every method probes optional children with ``in`` (H5Lexists),
    # NOT ``Group.get()`` — per ADR 0018 INV-15 / the
    # ``project_h5py_optional_child_get_hazard`` PR #261 pattern.

    def materials(self) -> list[MaterialRecord]:
        """Return every ``/opensees/materials/{family}/{name}`` group as
        a flat list of :class:`MaterialRecord`.

        Both uniaxial and nD families are flattened into one list; the
        ``family`` axis is recoverable from the OpenSees domain (the
        bridge would re-issue the right call on
        :meth:`emitter.uniaxialMaterial` vs
        :meth:`emitter.nDMaterial`).  :meth:`materials_by_family`
        preserves the partition when the caller needs it.
        """
        out: list[MaterialRecord] = []
        if "opensees" not in self._f:
            return out
        ops = self._f["opensees"]
        if "materials" not in ops:
            return out
        materials = ops["materials"]
        for family in materials:
            fam_group = materials[family]
            for name in fam_group:
                out.append(self._material_record(fam_group[name]))
        return out

    def materials_by_family(
        self,
    ) -> dict[str, list[MaterialRecord]]:
        """Like :meth:`materials` but returns
        ``{"uniaxial": [...], "nd": [...]}``.

        Empty families are omitted (a file with only uniaxial materials
        produces ``{"uniaxial": [...]}``, not ``{"uniaxial": [...],
        "nd": []}``).
        """
        out: dict[str, list[MaterialRecord]] = {}
        if "opensees" not in self._f:
            return out
        ops = self._f["opensees"]
        if "materials" not in ops:
            return out
        materials = ops["materials"]
        for family in materials:
            fam_group = materials[family]
            recs = [self._material_record(fam_group[name]) for name in fam_group]
            if recs:
                out[family] = recs
        return out

    def sections(
        self,
    ) -> list[SectionSimpleRecord | SectionComplexRecord]:
        """Return every ``/opensees/sections/{name}`` group as a typed
        record.

        Complex sections (with ``patches`` / ``fibers`` / ``layers``
        sub-datasets) become :class:`SectionComplexRecord`; simple
        ones become :class:`SectionSimpleRecord`.  The discriminator
        is the presence of any of the three sub-datasets.
        """
        out: list[SectionSimpleRecord | SectionComplexRecord] = []
        if "opensees" not in self._f:
            return out
        ops = self._f["opensees"]
        if "sections" not in ops:
            return out
        sections = ops["sections"]
        for name in sections:
            g = sections[name]
            out.append(self._section_record(g))
        return out

    def transforms(self) -> list[TransformRecord]:
        """Return every ``/opensees/transforms/{name}`` group as a typed
        record.

        Schema deviation (see :class:`TransformRecord` docstring):
        one record per ``geomTransf`` call, not per spec.  The
        ``vec`` field carries the single emitted ``vecxz`` (first row
        of ``per_element_vecxz``).
        """
        out: list[TransformRecord] = []
        if "opensees" not in self._f:
            return out
        ops = self._f["opensees"]
        if "transforms" not in ops:
            return out
        for name in ops["transforms"]:
            out.append(self._transform_record(ops["transforms"][name]))
        return out

    def beam_integration(self) -> list[BeamIntegrationRecord]:
        """Return every ``/opensees/beam_integration/{name}`` group."""
        out: list[BeamIntegrationRecord] = []
        if "opensees" not in self._f:
            return out
        ops = self._f["opensees"]
        if "beam_integration" not in ops:
            return out
        for name in ops["beam_integration"]:
            g = ops["beam_integration"][name]
            attrs = _attrs_as_dict(g)
            params = self._read_param_array(g, "params")
            out.append(BeamIntegrationRecord(
                type_token=str(attrs.get("type", "")),
                tag=int(attrs.get("tag", 0)),
                args=tuple(params),
            ))
        return out

    def time_series(self) -> list[TimeSeriesRecord]:
        """Return every ``/opensees/time_series/{name}`` group."""
        out: list[TimeSeriesRecord] = []
        if "opensees" not in self._f:
            return out
        ops = self._f["opensees"]
        if "time_series" not in ops:
            return out
        for name in ops["time_series"]:
            g = ops["time_series"][name]
            attrs = _attrs_as_dict(g)
            params = self._read_param_array(g, "params")
            out.append(TimeSeriesRecord(
                type_token=str(attrs.get("type", "")),
                tag=int(attrs.get("tag", 0)),
                args=tuple(params),
            ))
        return out

    def patterns(self) -> list[PatternRecord]:
        """Return every ``/opensees/patterns/{name}`` group.

        ``loads`` / ``sps`` / ``ele_loads`` sub-lists are populated
        from the corresponding compound datasets; the connectivity
        between a pattern's stored args and the time-series it
        references is preserved by the ``args`` tuple (the
        ``series_ref`` attr the schema also writes is consumed only
        by validators).
        """
        out: list[PatternRecord] = []
        if "opensees" not in self._f:
            return out
        ops = self._f["opensees"]
        if "patterns" not in ops:
            return out
        for name in ops["patterns"]:
            out.append(self._pattern_record(ops["patterns"][name]))
        return out

    def recorders(self) -> list[RecorderRecord]:
        """Return every ``/opensees/recorders/{name}`` group.

        Schema 2.3.0 unifies typed and declared recorders; the typed
        record's ``decl_context`` field carries the declaration
        metadata for ``kind="declared"`` entries (None for typed
        primitives).
        """
        out: list[RecorderRecord] = []
        if "opensees" not in self._f:
            return out
        ops = self._f["opensees"]
        if "recorders" not in ops:
            return out
        for name in ops["recorders"]:
            out.append(self._recorder_record(ops["recorders"][name]))
        return out

    def partitions(self) -> list[PartitionEmittedRecord]:
        """Return every ``/opensees/partitions/partition_NN`` sub-group.

        Schema 2.10.0 (ADR 0027).  Returns ``[]`` when no partition
        group is present in the file (back-compat: pre-2.10.0 archives
        and 2.10.x archives written outside any
        :meth:`H5Emitter.partition_open` / :meth:`partition_close`
        bracket both produce an empty list).

        The list is ordered by the ``partition_NN`` group ordinal
        (insertion order on write — h5py preserves group creation
        order).  Each record carries the rank id plus the three
        int64 arrays as immutable tuples.
        """
        out: list[PartitionEmittedRecord] = []
        if "opensees" not in self._f:
            return out
        ops = self._f["opensees"]
        if "partitions" not in ops:
            return out
        partitions = ops["partitions"]
        for name in partitions:
            g = partitions[name]
            attrs = _attrs_as_dict(g)
            element_ids = (
                tuple(int(v) for v in g["element_ids"][:])
                if "element_ids" in g else ()
            )
            node_ids = (
                tuple(int(v) for v in g["node_ids"][:])
                if "node_ids" in g else ()
            )
            boundary = (
                tuple(int(v) for v in g["boundary_node_ids"][:])
                if "boundary_node_ids" in g else ()
            )
            out.append(PartitionEmittedRecord(
                rank=int(attrs.get("rank", 0)),
                element_ids=element_ids,
                node_ids=node_ids,
                boundary_node_ids=boundary,
            ))
        return out

    # -- Private decoders for the typed accessors -----------------------

    def _material_record(self, g: Any) -> MaterialRecord:
        attrs = _attrs_as_dict(g)
        params = self._read_param_array(g, "params")
        return MaterialRecord(
            type_token=str(attrs.get("type", "")),
            tag=int(attrs.get("tag", 0)),
            params=tuple(params),
        )

    def _section_record(
        self, g: Any,
    ) -> SectionSimpleRecord | SectionComplexRecord:
        attrs = _attrs_as_dict(g)
        params = self._read_param_array(g, "params")
        has_complex_body = (
            "patches" in g or "fibers" in g or "layers" in g
        )
        type_token = str(attrs.get("type", ""))
        tag = int(attrs.get("tag", 0))
        if not has_complex_body:
            return SectionSimpleRecord(
                type_token=type_token, tag=tag, params=tuple(params),
            )
        from .._internal.typed_records import (
            FiberRecord, LayerRecord, PatchRecord,
        )
        patches: list[PatchRecord] = []
        fibers: list[FiberRecord] = []
        layers: list[LayerRecord] = []
        if "patches" in g:
            for row in g["patches"][:]:
                kind = _decode_bytes(row["kind"])
                # The reader has no symbolic mat-tag to recover here;
                # the writer stored ``material_ref`` as a path string
                # plus the patch args (ny, nz, coords).  Re-construct
                # the patch ``args`` shape from the dataset row so
                # downstream code can re-emit through the Protocol.
                # ``args = (mat_tag, ny, nz, *coords)`` matches the
                # writer's ``_decode_patch`` reverse.
                mat_tag = self._material_tag_from_ref(
                    _decode_bytes(row["material_ref"])
                )
                coords = [float(c) for c in row["coords"]]
                patches.append(PatchRecord(
                    kind=str(kind),
                    args=(int(mat_tag), int(row["ny"]), int(row["nz"]),
                          *coords),
                ))
        if "fibers" in g:
            for row in g["fibers"][:]:
                mat_tag = self._material_tag_from_ref(
                    _decode_bytes(row["material_ref"])
                )
                fibers.append(FiberRecord(
                    y=float(row["y"]),
                    z=float(row["z"]),
                    area=float(row["area"]),
                    mat_tag=int(mat_tag),
                ))
        if "layers" in g:
            for row in g["layers"][:]:
                kind = _decode_bytes(row["kind"])
                mat_tag = self._material_tag_from_ref(
                    _decode_bytes(row["material_ref"])
                )
                line = [float(v) for v in row["line"]]
                layers.append(LayerRecord(
                    kind=str(kind),
                    args=(int(mat_tag), int(row["n_bars"]),
                          float(row["area"]), *line),
                ))
        return SectionComplexRecord(
            type_token=type_token, tag=tag, params=tuple(params),
            patches=patches, fibers=fibers, layers=layers,
        )

    def _transform_record(self, g: Any) -> TransformRecord:
        import numpy as np

        attrs = _attrs_as_dict(g)
        vec_arr = np.asarray(g["per_element_vecxz"][:]).reshape(-1)
        # Each emitted call produces a single-row ``per_element_vecxz``
        # (1, 3); the first three values are the vecxz tuple.
        vec = tuple(float(v) for v in vec_arr[:3])
        return TransformRecord(
            type_token=str(attrs.get("type", "")),
            tag=int(attrs.get("tag", 0)),
            vec=vec,
        )

    def _pattern_record(self, g: Any) -> PatternRecord:
        from .._internal.typed_records import (
            EleLoadRecord, LoadRecord, SPRecord,
        )

        attrs = _attrs_as_dict(g)
        params = self._read_param_array(g, "params")
        # On read we don't see the original direction attr — the
        # writer surfaces ``direction`` for UniformExcitation as an
        # explicit attr alongside ``params``; we ignore it because
        # ``params`` already carries the direction in slot 0.
        loads: list[LoadRecord] = []
        sps: list[SPRecord] = []
        ele_loads: list[EleLoadRecord] = []
        if "loads" in g:
            for row in g["loads"][:]:
                target = int(_decode_bytes(row["target"]))
                forces = tuple(
                    float(v) for v in row["forces"]
                    if not _is_nan(v)
                )
                loads.append(LoadRecord(target=target, forces=forces))
        if "sps" in g:
            for row in g["sps"][:]:
                sps.append(SPRecord(
                    target=int(_decode_bytes(row["target"])),
                    dof=int(row["dof"]),
                    value=float(row["value"]),
                ))
        if "element_loads" in g:
            for row in g["element_loads"][:]:
                args: list[int | float | str] = []
                for cell in row:
                    s = _decode_bytes(cell)
                    if s == "":
                        continue
                    # The writer stored numeric values via ``repr(v)``;
                    # try to recover the original type.  Fall back to
                    # string on parse failure.
                    args.append(_parse_repr(s))
                ele_loads.append(EleLoadRecord(args=tuple(args)))
        return PatternRecord(
            type_token=str(attrs.get("type", "")),
            tag=int(attrs.get("tag", 0)),
            args=tuple(params),
            loads=loads, sps=sps, ele_loads=ele_loads,
        )

    def _recorder_record(self, g: Any) -> RecorderRecord:
        attrs = _attrs_as_dict(g)
        params = self._read_param_array(g, "params")
        kind_attr = str(attrs.get("kind", "typed"))
        ctx: DeclContext | None = None
        if kind_attr == "declared":
            # All declaration-metadata attrs are present together
            # (writer writes them as one block — see
            # ``H5Emitter._write_recorders``).
            ctx = DeclContext(
                declaration_name=str(attrs.get("declaration_name", "")),
                record_name=str(attrs.get("record_name", "")) or None,
                category=str(attrs.get("category", "")),
                components=tuple(
                    str(s) for s in _as_seq(attrs.get("components"))
                ),
                raw=tuple(str(s) for s in _as_seq(attrs.get("raw"))),
                pg=tuple(str(s) for s in _as_seq(attrs.get("pg"))),
                label=tuple(str(s) for s in _as_seq(attrs.get("label"))),
                selection=tuple(
                    str(s) for s in _as_seq(attrs.get("selection"))
                ),
                ids=None if "ids" not in attrs else tuple(
                    int(i) for i in _as_seq(attrs.get("ids"))
                ),
                dt=_attr_or_none(attrs.get("dt"), float),
                n_steps=_attr_or_none(attrs.get("n_steps"), int),
                file_root=str(attrs.get("file_root", ".")),
            )
        return RecorderRecord(
            kind=str(attrs.get("type", "")),
            args=tuple(params),
            decl_context=ctx,
        )

    def _read_param_array(
        self, g: Any, key: str,
    ) -> tuple[int | float | str, ...]:
        """Decode a ``params`` / ``args`` attr pair back into the original
        positional ``*args`` tuple.

        Mirrors the inverse of :func:`apeGmsh.opensees.emitter.h5._write_param_array`:
        slot ``i`` is the string from ``{key}_str[i]`` when non-empty,
        otherwise the float from ``{key}[i]``.
        """
        import numpy as np

        if key not in g.attrs:
            return ()
        nums = np.asarray(g.attrs[key], dtype=np.float64)
        strs_key = f"{key}_str"
        strs: list[str] = []
        if strs_key in g.attrs:
            raw = g.attrs[strs_key]
            for item in raw:
                if isinstance(item, bytes):
                    strs.append(item.decode("utf-8", "replace"))
                else:
                    strs.append(str(item))
        out: list[int | float | str] = []
        for i, num in enumerate(nums):
            if i < len(strs) and strs[i] != "":
                out.append(strs[i])
            else:
                # Recover int from a float when the value is whole;
                # the writer collapsed both via ``float(v)``, so
                # ``_decode_patch`` etc. cast back to int.  Preserve
                # int-ness here so the replayed args are byte-stable
                # for integer-valued slots.
                if num.is_integer():
                    out.append(int(num))
                else:
                    out.append(float(num))
        return tuple(out)

    def _material_tag_from_ref(self, ref: str) -> int:
        """Reverse ``/opensees/materials/{family}/{type}_{tag}`` → ``tag``.

        Returns ``0`` for an empty / unparseable ref so the round-trip
        produces a stable shape even when a material reference is
        broken; validators flag the bad ref through :meth:`validate`.
        """
        if not ref:
            return 0
        # Group name shape is ``{type_token}_{tag}``; tag is after the
        # last underscore.
        leaf = ref.rsplit("/", 1)[-1]
        suffix = leaf.rsplit("_", 1)
        if len(suffix) != 2:
            return 0
        try:
            return int(suffix[1])
        except ValueError:
            return 0

    # -- Neutral-zone reads (Phase 8.5) ---------------------------------

    def nodes(self) -> dict[str, Any]:
        """Return ``{ids, coords}`` for ``/nodes`` (empty dict if absent).

        ``ids`` is a 1-D int64 array; ``coords`` is ``(N, 3)`` float64.
        """
        g = self._f.get("nodes")
        if g is None:
            return {}
        return {
            "ids": g["ids"][:],
            "coords": g["coords"][:],
        }

    def element_arrays(self, type_name: str) -> dict[str, Any]:
        """Return ``{ids, connectivity}`` for one ``/elements/{type}`` group.

        Companion to :meth:`elements` (which exposes only attrs).
        Raises ``KeyError`` if the type group is missing.
        """
        sub = self._f[f"elements/{type_name}"]
        out: dict[str, Any] = {"ids": sub["ids"][:]}
        if "connectivity" in sub:
            out["connectivity"] = sub["connectivity"][:]
        return out

    def physical_groups(self) -> dict[str, dict[str, Any]]:
        """Return root-level ``/physical_groups`` shape.

        Each entry is ``{dim, tag, name, node_ids, node_coords,
        element_ids?}`` — ``element_ids`` is present only for dim>=1
        groups.  Empty dict if ``/physical_groups`` is absent.
        """
        return self._read_named_index("physical_groups")

    def labels(self) -> dict[str, dict[str, Any]]:
        """Return root-level ``/labels`` shape (same fields as :meth:`physical_groups`)."""
        return self._read_named_index("labels")

    def mesh_selections(self) -> dict[str, dict[str, Any]]:
        """Return root-level ``/mesh_selections`` shape.

        Same fields as :meth:`physical_groups` / :meth:`labels` —
        post-mesh selection sets (``fem.mesh_selection``) the broker
        captures at ``get_fem_data()`` time.  Schema 2.4.0 addition
        (Phase 8.7 commit 2); empty dict for pre-2.4.0 files.
        """
        return self._read_named_index("mesh_selections")

    def constraints(self) -> dict[str, Any]:
        """Return ``{kind: compound_array}`` for ``/constraints/{kind}``.

        Each value is the raw compound h5py dataset content (the
        symmetric ``target_kind`` / ``target`` / ``payload_kind`` /
        ``payload`` rows from :mod:`apeGmsh.mesh._record_h5`).  Empty
        dict if no ``/constraints`` group is present.
        """
        g = self._f.get("constraints")
        if g is None:
            return {}
        return {kind: g[kind][:] for kind in g}

    def loads(self) -> dict[str, dict[str, Any]]:
        """Return ``{kind: {pattern: compound_array}}`` for ``/loads``.

        Kinds are ``"nodal"``, ``"element"``, ``"sp"`` — each maps to
        a per-pattern dict of compound arrays.  Empty dict if no
        ``/loads`` group is present.
        """
        g = self._f.get("loads")
        if g is None:
            return {}
        out: dict[str, dict[str, Any]] = {}
        for kind in g:
            sub = g[kind]
            out[kind] = {pat: sub[pat][:] for pat in sub}
        return out

    def masses(self) -> Any:
        """Return the ``/masses`` compound array, or ``None`` if absent."""
        ds = self._f.get("masses")
        if ds is None:
            return None
        return ds[:]

    # -- Compose provenance (Schema 2.9.0; ADR 0038 / Phase 3D.1) -------
    #
    # The three methods below widen the H5ModelReader Protocol (ADR 0026)
    # with compose-awareness so viewers + cuts + future foreign adapters
    # can query the composition graph of a loaded model without coupling
    # to the writer-side ``FEMData`` API.  Backward compatible: schema
    # 2.8.x files (no ``/composed_from/`` group, no ``module_label``
    # parallel datasets) decode as "uncomposed" — iter yields nothing,
    # ``composed_for_*`` returns ``None`` for every id.
    #
    # Probes optional children with ``in`` (H5Lexists), NOT
    # ``Group.get()`` — per the h5py optional-child .get() hazard
    # (``project_h5py_optional_child_get_hazard`` PR #261).

    def iter_composed_from(self) -> Iterator["ComposeRecord"]:
        """Yield each :class:`ComposeRecord` on this model in module-label order.

        Reads ``/composed_from/{safe_label}/`` per ADR 0038 §"Schema".
        Yields nothing when the file is uncomposed (the ``/composed_from/``
        group is omitted by the writer when ``fem.composed_from`` is
        empty — absence is the canonical "uncomposed" signal on disk)
        or when the file pre-dates schema 2.9.0.

        The order is *sorted by storage key* — the writer's
        ``safe = label.replace("/", "_")`` derivation makes this
        equivalent to "sorted by ``ComposeRecord.label``" for the
        common case where labels don't contain ``/``.  Nested-compose
        labels (Phase 3E) will land their own ordering contract.
        """
        from apeGmsh._kernel.records._compose import ComposeRecord

        if "composed_from" not in self._f:
            return
        parent = self._f["composed_from"]
        # ``hasattr(.keys)`` filters out the case where a stray non-group
        # ended up at this key — mirrors ``_read_composed_from`` in the
        # broker reader.
        if not hasattr(parent, "keys"):
            return
        import numpy as np

        for key in sorted(parent.keys()):
            sub = parent[key]
            if not hasattr(sub, "keys"):
                continue
            attrs = sub.attrs
            label = str(attrs.get("label", key))
            source_path = str(attrs.get("source_path", ""))
            source_fem_hash = str(attrs.get("source_fem_hash", ""))
            source_neutral_schema_version = str(
                attrs.get("source_neutral_schema_version", "")
            )
            translate_raw = attrs.get(
                "translate", np.zeros(3, dtype=np.float64),
            )
            translate = tuple(
                float(x) for x in np.asarray(
                    translate_raw, dtype=np.float64,
                ).reshape(-1)[:3]
            )
            rotate: tuple[float, float, float, float] | None = None
            if "rotate" in attrs:
                rotate_raw = np.asarray(
                    attrs["rotate"], dtype=np.float64,
                ).reshape(-1)
                rotate = tuple(  # type: ignore[assignment]
                    float(x) for x in rotate_raw[:4]
                )
            partition_rank: int | None = None
            if "partition_rank" in attrs:
                partition_rank = int(attrs["partition_rank"])
            composed_at = str(attrs.get("composed_at", ""))

            properties: dict[str, Any] = {}
            if "properties" in sub:
                prop_attrs = sub["properties"].attrs
                for pk in prop_attrs:
                    raw = prop_attrs[pk]
                    if isinstance(raw, np.ndarray) and raw.shape == ():
                        raw = raw.item()
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", errors="replace")
                    properties[str(pk)] = raw

            yield ComposeRecord(
                label=label,
                source_path=source_path,
                source_fem_hash=source_fem_hash,
                source_neutral_schema_version=source_neutral_schema_version,
                translate=translate,  # type: ignore[arg-type]
                rotate=rotate,
                partition_rank=partition_rank,
                composed_at=composed_at,
                properties=properties,
            )

    def composed_for_node(self, node_id: int) -> str | None:
        """Return the module label that owns ``node_id``, or ``None`` for host.

        ``None`` is returned for both "host-owned" (the
        ``/nodes/module_label`` row is the empty string) and "not
        present" (the file pre-dates schema 2.9.0, the
        ``module_label`` parallel dataset is absent, or ``node_id``
        does not appear in ``/nodes/ids``).  Callers that need to
        distinguish the two cases must consult :meth:`nodes` directly.

        The conflated-``None`` return matches the convention of
        :meth:`analysis` and :meth:`masses` (which also collapse
        "absent" and "empty" into ``None``).
        """
        if "nodes" not in self._f:
            return None
        nodes_grp = self._f["nodes"]
        if "ids" not in nodes_grp or "module_label" not in nodes_grp:
            return None
        import numpy as np

        ids = np.asarray(nodes_grp["ids"][:], dtype=np.int64)
        matches = np.flatnonzero(ids == int(node_id))
        if matches.size == 0:
            return None
        row = int(matches[0])
        labels = nodes_grp["module_label"][:]
        if row >= len(labels):
            return None
        raw = labels[row]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        label = str(raw)
        return label if label else None

    def composed_for_element(self, element_id: int) -> str | None:
        """Return the module label that owns ``element_id``, or ``None``.

        Walks every ``/elements/{type}/`` sub-group looking for
        ``element_id`` in the type's ``ids`` dataset; returns the
        matching row's ``module_label`` value when found.  Returns
        ``None`` for the same conflated reasons as
        :meth:`composed_for_node` — host-owned (empty string label),
        schema 2.8.x file (no ``module_label`` dataset), or
        ``element_id`` not present in any element-type group.
        """
        if "elements" not in self._f:
            return None
        elements_grp = self._f["elements"]
        if not hasattr(elements_grp, "keys"):
            return None
        import numpy as np

        target = int(element_id)
        for type_name in elements_grp:
            sub = elements_grp[type_name]
            if not hasattr(sub, "keys"):
                continue
            if "ids" not in sub or "module_label" not in sub:
                continue
            ids = np.asarray(sub["ids"][:], dtype=np.int64)
            matches = np.flatnonzero(ids == target)
            if matches.size == 0:
                continue
            row = int(matches[0])
            labels = sub["module_label"][:]
            if row >= len(labels):
                return None
            raw = labels[row]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            label = str(raw)
            return label if label else None
        return None

    def bulk_module_labels_for_nodes(self) -> dict[int, str]:
        """Return ``{node_id: module_label}`` for every non-host node.

        Bulk equivalent of :meth:`composed_for_node` for the viewer's
        per-entity coloring path (Phase 3F.2a — schema 2.9.0 / ADR
        0038).  Reads the full ``/nodes/ids`` + ``/nodes/module_label``
        parallel datasets in one pass; host-owned rows (empty-string
        label) are excluded from the returned mapping.

        Returns an empty dict for pre-2.9.0 archives (the
        ``module_label`` dataset is absent) and for archives written
        with no compose merges (the dataset is present but every row
        is the empty string — the broker's writer always emits it now,
        but every row is ``""`` until a ``compose()`` call runs).

        Probes optional children with ``in`` (H5Lexists) per the h5py
        optional-child .get() hazard documented in PR #261.
        """
        out: dict[int, str] = {}
        if "nodes" not in self._f:
            return out
        nodes_grp = self._f["nodes"]
        if "ids" not in nodes_grp or "module_label" not in nodes_grp:
            return out
        import numpy as np

        ids = np.asarray(nodes_grp["ids"][:], dtype=np.int64)
        labels = nodes_grp["module_label"][:]
        n = min(len(ids), len(labels))
        for i in range(n):
            raw = labels[i]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            label = str(raw)
            if label == "":
                continue
            out[int(ids[i])] = label
        return out

    def bulk_module_labels_for_elements(self) -> dict[int, str]:
        """Return ``{element_id: module_label}`` for every non-host element.

        Bulk equivalent of :meth:`composed_for_element` — walks every
        ``/elements/{type}/`` sub-group and joins its ``ids`` with its
        ``module_label`` parallel dataset (schema 2.9.0 / ADR 0038).
        Host-owned rows (empty-string label) are excluded.

        Returns an empty dict for pre-2.9.0 archives and for archives
        with no compose merges (same conflation as
        :meth:`bulk_module_labels_for_nodes`).

        Probes optional children with ``in`` (H5Lexists) per the h5py
        optional-child .get() hazard documented in PR #261.
        """
        out: dict[int, str] = {}
        if "elements" not in self._f:
            return out
        elements_grp = self._f["elements"]
        if not hasattr(elements_grp, "keys"):
            return out
        import numpy as np

        for type_name in elements_grp:
            sub = elements_grp[type_name]
            if not hasattr(sub, "keys"):
                continue
            if "ids" not in sub or "module_label" not in sub:
                continue
            ids = np.asarray(sub["ids"][:], dtype=np.int64)
            labels = sub["module_label"][:]
            n = min(len(ids), len(labels))
            for i in range(n):
                raw = labels[i]
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                label = str(raw)
                if label == "":
                    continue
                out[int(ids[i])] = label
        return out

    def _read_named_index(
        self, group_name: str,
    ) -> dict[str, dict[str, Any]]:
        """Walk a root-level named index (``physical_groups`` or ``labels``).

        Each child group becomes one entry in the returned dict.
        """
        parent = self._f.get(group_name)
        if parent is None:
            return {}
        out: dict[str, dict[str, Any]] = {}
        for sub_name in parent:
            sub = parent[sub_name]
            entry: dict[str, Any] = _attrs_as_dict(sub)
            if "node_ids" in sub:
                entry["node_ids"] = sub["node_ids"][:]
            if "node_coords" in sub:
                entry["node_coords"] = sub["node_coords"][:]
            if "element_ids" in sub:
                entry["element_ids"] = sub["element_ids"][:]
            out[sub_name] = entry
        return out

    def _group_attrs_map(
        self, group: str,
    ) -> dict[str, dict[str, Any]]:
        g = self._f.get(group)
        if g is None:
            return {}
        return {name: _attrs_as_dict(g[name]) for name in g}

    # -- Validation ------------------------------------------------------

    def validate(self) -> list[str]:
        """Walk the file checking every schema invariant we can mechanically
        verify. Returns a list of violation messages — empty means valid.

        Checked invariants:

        * ``/meta`` has all the required attrs.
        * ``schema_version`` parses as semver.
        * Material / section names match their internal ``tag`` attr
          (i.e. ``Steel02_3``'s ``tag`` attr is 3) — this is the H5
          emitter's convention and a fast inconsistency check.
        * Patches' ``material_ref`` paths resolve to existing groups.
        * Fibers' ``material_ref`` paths resolve.
        * Pattern ``series_ref`` paths resolve.
        """
        errors: list[str] = []
        errors.extend(self._validate_meta())
        errors.extend(self._validate_materials_naming())
        errors.extend(self._validate_section_refs())
        errors.extend(self._validate_pattern_refs())
        return errors

    def _validate_meta(self) -> list[str]:
        out: list[str] = []
        if self._meta_path not in self._f:
            out.append(f"missing /{self._meta_path} group")
            return out
        meta = self._f[self._meta_path]
        required = (
            "schema_version", "apeGmsh_version", "created_iso",
            "ndm", "ndf", "snapshot_id", "model_name",
        )
        for attr in required:
            if attr not in meta.attrs:
                out.append(
                    f"/{self._meta_path} missing required attr {attr!r}"
                )
        return out

    def _validate_materials_naming(self) -> list[str]:
        out: list[str] = []
        materials = self._f.get("opensees/materials")
        if materials is None:
            return out
        for family in materials:
            fam_group = materials[family]
            for name in fam_group:
                g = fam_group[name]
                attr_tag = int(g.attrs.get("tag", -1))
                expected_suffix = f"_{attr_tag}"
                if not name.endswith(expected_suffix):
                    out.append(
                        f"/opensees/materials/{family}/{name}: name does not "
                        f"end with _{attr_tag} (tag attr says {attr_tag})"
                    )
        return out

    def _validate_section_refs(self) -> list[str]:
        out: list[str] = []
        sections = self._f.get("opensees/sections")
        if sections is None:
            return out
        for name in sections:
            sec = sections[name]
            patches = sec.get("patches")
            if patches is not None:
                for row in patches[:]:
                    ref = _decode_bytes(row["material_ref"])
                    if ref and ref not in self._f:
                        out.append(
                            f"/opensees/sections/{name}/patches: material_ref "
                            f"{ref!r} not found in file"
                        )
            fibers = sec.get("fibers")
            if fibers is not None:
                for row in fibers[:]:
                    ref = _decode_bytes(row["material_ref"])
                    if ref and ref not in self._f:
                        out.append(
                            f"/opensees/sections/{name}/fibers: material_ref "
                            f"{ref!r} not found in file"
                        )
        return out

    def _validate_pattern_refs(self) -> list[str]:
        out: list[str] = []
        patterns = self._f.get("opensees/patterns")
        if patterns is None:
            return out
        for name in patterns:
            p = patterns[name]
            ref = p.attrs.get("series_ref")
            if ref:
                ref_str = ref if isinstance(ref, str) else str(ref)
                if ref_str not in self._f:
                    out.append(
                        f"/opensees/patterns/{name}: series_ref {ref_str!r} "
                        "not found in file"
                    )
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _attrs_as_dict(group: Any) -> dict[str, Any]:
    """Convert an h5py attrs view to a plain dict, decoding bytes."""
    out: dict[str, Any] = {}
    for key, value in group.attrs.items():
        out[key] = _decode_bytes(value)
    return out


def _is_nan(value: Any) -> bool:
    """Return ``True`` iff ``value`` is a NaN (used to strip the
    schema's NaN-padded ``forces`` slots from typed
    :class:`PatternRecord` loads)."""
    try:
        return bool(value != value)
    except Exception:
        return False


def _parse_repr(s: str) -> int | float | str:
    """Reverse ``repr(v)`` for the limited (int / float / str) value
    set the ``element_loads`` writer stores.

    Used by the typed pattern accessor; for tokens the writer left as
    a real string (``"-type"`` / ``"-ele"`` flags), ``s`` is just the
    string and we keep it as-is.
    """
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    # Strip surrounding repr-style quotes if present.
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


def _as_seq(value: Any) -> "list[Any]":
    """Coerce an h5py attr to a list (passes through tuples / lists /
    arrays; wraps scalars in a one-element list)."""
    import numpy as np

    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, np.ndarray):
        return list(value)
    return [value]


def _attr_or_none(value: Any, cast: Any) -> Any:
    """Return ``cast(value)`` if ``value`` is truthy / present, else
    ``None``.

    The H5 emitter writes ``None`` attrs as empty strings (per
    ``_set_attr``); the typed reader should recover ``None`` so the
    typed-record discriminators round-trip.
    """
    if value is None:
        return None
    if isinstance(value, str) and value == "":
        return None
    try:
        return cast(value)
    except (TypeError, ValueError):
        return None


def _decode_bytes(v: Any) -> Any:
    """Decode bytes / arrays-of-bytes to UTF-8 strings. Leave others alone."""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    # h5py sometimes returns 0-D arrays for scalar string attrs.
    try:
        import numpy as np
        if isinstance(v, np.ndarray) and v.dtype.kind in ("O", "S"):
            if v.shape == ():
                item = v.item()
                if isinstance(item, bytes):
                    return item.decode("utf-8", errors="replace")
                return item
    except ImportError:  # pragma: no cover  - numpy is a hard dep
        pass
    return v


# Re-exported from `.._element_capabilities` so the writer
# (`H5Emitter.add_oriented_elements`) and this reader share ONE
# definition of the transf-slot lookup (ADR 0018 INV-3).  Existing
# import path `from apeGmsh.opensees.emitter.h5_reader import
# _transf_arg_tail_index, _FORCE_DISP_BEAMS` is preserved.
from .._element_capabilities import _FORCE_DISP_BEAMS, _transf_arg_tail_index  # noqa: F401


# Re-export under the builtin name shadow so users can write
# ``h5_reader.open(...)``. Keep ``builtins.open`` accessible via a
# different alias inside this module if ever needed.
_ = builtins.open  # silence linters about the shadow
