"""
H5Emitter — buffers the bridge's emit calls and writes a model-definition
HDF5 archive conforming to ``architecture/h5-schema.md``.

Unlike the stream emitters (Tcl, Py, Live), this is a **structured**
emitter: each Protocol method updates an in-memory buffer; the file is
materialized at the end via :meth:`write`. The structured shape is
required because the schema groups by name
(``/opensees/materials/uniaxial/{name}``,
``/opensees/sections/{name}``, ...), and aggregate sections / patterns
need open / close bracketing while patches, fibers, loads, and sps
land inside them.

Design notes
============

**Names from tags (Option B in ADR 0011's discussion).** The bridge's
typed primitives carry no user-supplied ``name`` field. Every name in
the file is generated from ``<type_token>_<tag>`` (e.g. ``Steel02_1``,
``Fiber_2``, ``forceBeamColumn_3``). Cross-references resolve via the
generated names. No Protocol extension was required.

**Schema layout.** Phase 8.4 partitioned ``model.h5`` into two zones
(see ``architecture/h5-schema.md`` and
``architecture/phase-8-untangle.md`` §3):

* ``/meta`` and ``/elements`` stay at the file root.  ``/elements``
  sits in the neutral zone because Phase 8.5 hands the writer from
  the bridge to the broker; for now the bridge still emits it.
* Everything else the bridge writes (materials, sections, transforms,
  beam_integration, time_series, patterns, bcs, recorders, analysis)
  lives under ``/opensees/`` so a future second solver can plug in at
  the same level without colliding.

The reshuffle is a breaking schema change — ``SCHEMA_VERSION`` jumps
``1.1.0 → 2.0.0``.  Phase 7a (ADR 0023) replaced the previous
``EXPECTED_SCHEMA_MAJOR`` constant with per-zone two-version-window
validation in :mod:`apeGmsh.opensees._internal.schema_version`.

**Schema deviation (documented).**  One place where the streaming
Protocol cannot supply the spec-level grouping the schema asks for:

* ``/opensees/transforms/{name}`` — the schema shows one group per
  user-declared transform with a ``per_element_vecxz`` dataset of
  shape ``(n_elements, 3)``.  The orientation-driven fan-out in the bridge's
  build layer emits one ``geomTransf`` line per *distinct* vecxz; the
  H5 emitter sees these as N independent calls and cannot
  reverse-engineer the spec boundary.  We therefore emit one
  ``/opensees/transforms/{type}_{tag}/`` group per ``geomTransf``
  call, each carrying a single-row ``per_element_vecxz`` and
  ``per_element_emitted_tag``.  The viewer's vecxz overlay (which
  only needs ``element_id → vecxz``) still works: it iterates all
  transform groups.

**Elements binned by type token (design choice, not deviation).**  The
schema groups elements by element-type token
(``/elements/forceBeamColumn/``, ``/elements/FourNodeTetrahedron/``,
...).  The streaming Protocol does not surface the PG (``spec.pg`` is
known only inside ``_internal/build.py``'s element fan-out), so the
H5 emitter bins by type.  Each group's ``ids`` and ``connectivity``
datasets stack every element of that type across all PGs; the
``args`` / ``args_str`` dataset pair encodes the element's positional
arg list so a vocabulary-aware reader can recover refs.

Both the transform deviation and the elements-by-type choice are
recorded as ``__deviation__`` attrs on the affected groups so a
reader can detect them and degrade gracefully.

**Lazy h5py import.** ``h5py`` is imported only inside :meth:`write` so
constructing an :class:`H5Emitter` (or driving emit) does not pull
the dependency into import time for users who never call ``ops.h5()``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from .._internal.tag_resolution import (
    ATTR_ELEMENT_NODES,
    MISSING_FEM_ELEMENT_ID,
    current_fem_element_id,
)
from .._internal.typed_records import (
    BeamIntegrationRecord as _BeamIntegrationRecord,
    DeclContext as _DeclContext,
    EleLoadRecord as _EleLoadRecord,
    ElementRecord as _ElementRecord,
    EmbeddedNodeRecord as _EmbeddedNodeRecord,
    EqualDOFRecord as _EqualDOFRecord,
    FiberRecord as _FiberRecord,
    FixRecord as _FixRecord,
    LayerRecord as _LayerRecord,
    LoadRecord as _LoadRecord,
    MassRecord as _MassRecord,
    MaterialRecord as _MaterialRecord,
    PatchRecord as _PatchRecord,
    PatternRecord as _PatternRecord,
    RecorderRecord as _RecorderRecord,
    RegionRecord as _RegionRecord,
    RigidDiaphragmRecord as _RigidDiaphragmRecord,
    RigidLinkRecord as _RigidLinkRecord,
    SPRecord as _SPRecord,
    SectionComplexRecord as _SectionComplexRecord,
    SectionSimpleRecord as _SectionSimpleRecord,
    TimeSeriesRecord as _TimeSeriesRecord,
    TransformRecord as _TransformRecord,
)


__all__ = ["H5Emitter", "SCHEMA_VERSION"]


#: Schema version string emitted in ``/meta/schema_version``. Bump
#: ``MAJOR`` only on a breaking change; ``MINOR`` for additive groups
#: (such as the ``/beam_integration`` group introduced alongside the
#: Protocol's ``beamIntegration`` method in Phase 4.5).
#:
#: History:
#:   * 1.0.0 — Phase 6 initial release.
#:   * 1.1.0 — added ``/beam_integration`` group + widened fiber-layer
#:     ``line`` field from float[4] to float[6].
#:   * 2.0.0 — Phase 8.4: bridge-written groups (materials, sections,
#:     transforms, beam_integration, time_series, patterns, bcs,
#:     recorders, analysis) moved under ``/opensees/``.  ``/meta`` and
#:     ``/elements`` stay at root.
#:   * 2.1.0 — Phase 8.5: broker-side neutral zone added (``/nodes``,
#:     ``/elements/{type}``, ``/physical_groups``, ``/labels``,
#:     ``/constraints/{kind}``, ``/loads/{kind}/{pattern}``,
#:     ``/masses``).  ``/elements`` writing handed from bridge to
#:     broker; the bridge no longer emits this group.  Additive — old
#:     v2.0.0 readers still work, they just don't see the new groups.
#:   * 2.2.0 — Phase 8.6: ``fem_eids`` int64 dataset added under each
#:     ``/opensees/element_meta/{type_token}/`` group, parallel to
#:     ``ids``.  Carries the FEM element id each OpenSees tag was
#:     fanned out from (master plan §3 "tag_map").  Sentinel ``-1``
#:     entries mark records emitted outside a bridge fan-out.
#:     Additive — old v2.1.0 readers ignore the new dataset.
#:   * 2.3.0 — Phase 9 commit 6: unified ``/opensees/recorders/``
#:     archive.  Every record group gains a ``kind`` attr —
#:     ``"typed"`` for Node / Element / MPCO primitives (1:1
#:     OpenSees), ``"declared"`` for fan-out calls produced by
#:     ``ops.recorder.declare(...)``. Declared records also carry
#:     the original declaration metadata as attrs:
#:     ``declaration_name``, ``record_name``, ``category``,
#:     ``components``, ``raw``, ``pg``, ``label``, ``selection``,
#:     ``ids``, ``dt``, ``n_steps``, ``file_root``.  Additive — old
#:     v2.2.0 readers see ``kind="declared"`` records as well-formed
#:     recorder groups (they just ignore the extra attrs).
#:   * 2.4.0 — Phase 8.7 commit 2: ``/mesh_selections/`` neutral-zone
#:     group added, mirroring ``/physical_groups`` / ``/labels`` shape.
#:     Carries post-mesh selection sets (``g.mesh_selection`` →
#:     ``fem.mesh_selection``) so the viewer's ``selection=`` selector
#:     round-trips through ``model.h5``.  Additive — old v2.3.0 readers
#:     ignore the new group and lose only the ``selection=`` round-trip
#:     convenience (live mesh_viewer sessions still consult the live
#:     ``fem.mesh_selection`` directly).
#:   * 2.5.0 — apeGmsh.cuts v4: ``/opensees/cuts/`` and
#:     ``/opensees/sweeps/`` groups added for SectionCutDef and
#:     SectionSweepDef persistence (writer lives in
#:     :mod:`apeGmsh.cuts._h5_io`).  Additive — pre-v4 readers ignore
#:     the new groups.
#:   * 2.6.0 — Phase 6 (ADR 0021): the ``/meta/lineage/`` sub-group
#:     is stamped alongside the bridge zone by
#:     :func:`_compose_model_h5`.  Additive — the lineage attrs are
#:     readable by 2.5.x consumers as opaque extra meta children
#:     (the bridge reader ignores them; the read side surfaces a
#:     "lineage absent" warning when the sub-group is missing on a
#:     legacy file).  Per ADR 0023 two-version reader window, both
#:     2.5.x and 2.6.x files are accepted.
#:   * 2.7.0 — Phase 7b (ADR 0022): MP constraint emission fanout.
#:     The H5 emitter gained the ``/opensees/constraints/`` group
#:     with four compound-dtype datasets — ``equalDOF``,
#:     ``rigidLink``, ``rigidDiaphragm``, ``embeddedNode`` — one
#:     row per :meth:`equalDOF` / :meth:`rigidLink` /
#:     :meth:`rigidDiaphragm` / :meth:`embeddedNode` call.  Each
#:     dataset carries the constraint's ``name`` field (declaration
#:     label round-trip, INV-2) and a ``phantom_node_tags`` int64
#:     dataset records phantom-node tags written via
#:     :meth:`node` with ``ndf=6`` so a reader can identify which
#:     nodes were synthesized by the phantom-emit pre-step (INV-3).
#:     Additive — old 2.6.x readers ignore the new group.  Per ADR
#:     0023 two-version reader window, both 2.6.x and 2.7.x files
#:     are accepted.
#:   * 2.8.0 — embeddedNode field rename: the second compound-dtype
#:     column of ``/opensees/constraints/embeddedNode`` was
#:     ``embedding_ele`` in 2.7.0 (a misnomer — the value is the
#:     constrained / slave node id, not an element id) and is
#:     ``cnode`` from 2.8.0 onward (matches the OpenSees ``$Cnode``
#:     vocabulary).  Stored values are unchanged; only the column
#:     name flips.  No in-repo reader consumed the 2.7.0 name, so
#:     no compat shim is required; the two-version reader window
#:     accepts 2.7.x and 2.8.x files but the column name is
#:     version-dependent.
#:   * 2.9.0 — ADR 0024 (Emitter Protocol widening for ``region()``):
#:     the H5 emitter gained the ``/opensees/regions/`` group, with
#:     one ``region_NNN`` sub-group per :meth:`region` call carrying
#:     the integer ``tag`` attribute plus a ``params`` dataset
#:     mirroring the OpenSees flag tail (``-node n1 n2 ...``,
#:     ``-ele e1 e2 ...``, etc.).  Auto-emitted today by the
#:     ``MPCO`` recorder fan-out (``nodes_pg=`` / ``elements_pg=``)
#:     so MPCO output filtered by ``-R $tag`` round-trips through
#:     ``OpenSeesModel.from_h5(...).build("tcl", ...)``.  Additive —
#:     old 2.8.x readers ignore the new group and lose only the
#:     filter round-trip (the MPCO recorder line still carries the
#:     dangling ``-R`` reference; the bridge raises if the
#:     referenced region is missing at re-emit time).  Per ADR 0023
#:     two-version reader window, both 2.8.x and 2.9.x files are
#:     accepted.
#:   * 2.10.0 — ADR 0027 (cross-partition MP-constraint emission policy):
#:     the H5 emitter gained the ``/opensees/partitions/`` group plus
#:     a parallel ``partition_ids`` int64 column on every
#:     ``/opensees/element_meta/{type_token}/`` group.  The new
#:     :meth:`partition_open` / :meth:`partition_close` Protocol
#:     methods bracket per-rank emit blocks; each block populates one
#:     ``partition_NN`` sub-group with ``element_ids`` /
#:     ``node_ids`` / ``boundary_node_ids`` int64 datasets and
#:     ``rank`` / ``n_elements`` / ``n_nodes`` scalar attrs.  The
#:     parent ``/opensees/partitions/`` group carries the
#:     ``n_partitions`` scalar attr.  Boundary nodes are computed on
#:     write as the per-rank set intersected against every other
#:     rank's node set — symmetric across ranks.  Elements emitted
#:     outside a ``partition_open`` / ``partition_close`` bracket get
#:     ``partition_ids`` row value ``-1``.  Additive — old 2.9.x
#:     readers ignore the new group and the new column.  Per ADR 0023
#:     two-version reader window, both 2.9.x and 2.10.x files are
#:     accepted.
#:   * 2.11.0 — bug fix: the bridge now emits **0-based runtime ranks**
#:     (matching ``OpenSeesMP::getPID()``) instead of Gmsh's 1-based
#:     ``PartitionRecord.id``.  Side effect on this zone: per-partition
#:     group ``rank`` attrs and the parallel ``partition_ids`` column
#:     values become 0-based (``0..N-1``) instead of 1-based (``1..N``).
#:     Group naming (``partition_NN`` is just the loop index) is
#:     unchanged.  The broker's Gmsh-side 1-based ``PartitionRecord.id``
#:     is preserved verbatim (only the runtime-rank seam flipped).
#:     Breaking for any reader that mapped ``rank`` attr / partition_ids
#:     values to ``part.id`` directly; per ADR 0023 two-version reader
#:     window, both 2.10.x and 2.11.x files are accepted.
SCHEMA_VERSION: str = "2.11.0"


# Map known time-series type tokens to "is path-bearing": for a Path
# series the ``args`` carry numeric values; for algorithmic series we
# only record the type + scalar params.
_PATH_SERIES_TOKENS: tuple[str, ...] = ("Path",)


# Pattern type tokens that the bridge opens via ``pattern_open`` and
# closes via ``pattern_close`` with a body of load / sp / eleLoad calls.
_BLOCK_PATTERN_TOKENS: tuple[str, ...] = ("Plain", "MultiSupport")


# ---------------------------------------------------------------------------
# Low-level write helpers
# ---------------------------------------------------------------------------

def _set_attr(target: Any, key: str, value: Any) -> None:
    """Write ``value`` as an HDF5 attribute on ``target``.

    HDF5 stores attributes as native scalars or arrays — never as JSON
    blobs (per the schema's "structured groups, scalar attrs, array
    datasets" rule). This helper coerces Python values to the closest
    h5py-friendly representation:

    * ``str`` → variable-length UTF-8 string
    * ``bool`` / ``int`` → int64
    * ``float`` → float64
    * ``None`` → empty-string attr (h5py rejects ``None``)
    * tuple/list of numbers → 1-D float64 / int64 array
    * tuple/list of strings → 1-D variable-length string array
    """
    import h5py
    import numpy as np

    if value is None:
        target.attrs[key] = ""
        return
    if isinstance(value, bool):
        target.attrs[key] = np.int64(int(value))
        return
    if isinstance(value, int):
        target.attrs[key] = np.int64(value)
        return
    if isinstance(value, float):
        target.attrs[key] = np.float64(value)
        return
    if isinstance(value, str):
        target.attrs.create(key, value, dtype=h5py.string_dtype(encoding="utf-8"))
        return
    if isinstance(value, (tuple, list)):
        if not value:
            target.attrs.create(
                key, np.array([], dtype=np.float64),
            )
            return
        if all(isinstance(v, str) for v in value):
            target.attrs.create(
                key, list(value),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            return
        if all(isinstance(v, bool) or isinstance(v, int) for v in value):
            target.attrs[key] = np.asarray(value, dtype=np.int64)
            return
        # Mixed numeric — coerce to float64.
        target.attrs[key] = np.asarray(
            [float(v) for v in value], dtype=np.float64,
        )
        return
    # Fallback: stringify so we never crash on an unexpected attr type.
    target.attrs.create(
        key, repr(value), dtype=h5py.string_dtype(encoding="utf-8"),
    )


def _scan_flag(
    args: tuple[int | float | str, ...], flag: str,
) -> str | None:
    """Return the argument immediately after ``flag`` (as a string), or
    ``None`` if the flag is not present.

    Used for recorder ``-file`` extraction. Treats only string args as
    flag candidates.
    """
    for i, v in enumerate(args[:-1]):
        if isinstance(v, str) and v == flag:
            nxt = args[i + 1]
            return str(nxt)
    return None


def _write_param_array(
    target: Any, key: str, params: tuple[float | str | int, ...],
) -> None:
    """Write a positional ``*args`` tuple as one or two attributes.

    OpenSees parameter lists are positional: ``Steel02 1 fy E b R0 cR1
    cR2`` is all numeric, but ``forceBeamColumn 1 i j tt it -mass m
    -iter mx tol`` interleaves numerics and flag-string tokens. To
    stay HDF5-native (no JSON blobs), we split into two parallel
    arrays:

    * ``{key}`` — float64 array, ``NaN`` in slots that hold a string.
    * ``{key}_str`` — UTF-8 vlen string array, empty string in slots
      that hold a numeric value. Written ONLY if at least one slot is
      a string (pure-numeric param lists skip ``{key}_str`` entirely).

    Slot ``i`` of the original ``*args`` is reconstructible by reading
    whichever of the two arrays has a non-sentinel value.
    """
    import h5py
    import numpy as np

    if not params:
        target.attrs.create(key, np.array([], dtype=np.float64))
        return
    has_str = any(isinstance(v, str) for v in params)
    nums = np.empty(len(params), dtype=np.float64)
    strs: list[str] = []
    for i, v in enumerate(params):
        if isinstance(v, str):
            nums[i] = float("nan")
            strs.append(v)
        else:
            nums[i] = float(v)
            strs.append("")
    target.attrs.create(key, nums)
    if has_str:
        target.attrs.create(
            f"{key}_str", strs,
            dtype=h5py.string_dtype(encoding="utf-8"),
        )


# ---------------------------------------------------------------------------
# Buffered intermediate representations
# ---------------------------------------------------------------------------
#
# Phase 3 (ADR 0019) lifted the per-record dataclasses out into
# ``apeGmsh.opensees._internal.typed_records`` so both the write path
# (this module) and the read-side broker (``OpenSeesModel``) share one
# definition.  The underscore-prefixed aliases above preserve the
# private import paths that ``model_data.py`` and the test suite
# already depend on.


# ---------------------------------------------------------------------------
# Partition emission (ADR 0027, schema 2.10.0)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _PartitionEmitBlock:
    """In-memory accumulator for a single ``partition_open(rank)`` block.

    Lives only inside :class:`H5Emitter`; the read-side reuses
    :class:`apeGmsh.opensees.emitter.h5_reader.PartitionEmittedRecord`
    which is the public surface.  The two are intentionally separate —
    the writer carries mutable accumulators (lists; node_set for dedupe),
    the reader carries immutable tuples.
    """

    #: OpenSeesMP rank id this block scopes.
    rank: int
    #: Node tags emitted under this block, in insertion order, with
    #: duplicates dropped (a phantom node declared twice in one rank
    #: block from two MP constraints would still count once).
    node_ids: list[int] = field(default_factory=list)
    #: Element tags emitted under this block, in insertion order.
    element_ids: list[int] = field(default_factory=list)
    #: O(1) dedupe membership test for ``node_ids`` (parallel
    #: shadow — never read directly, only via ``add_node``).
    _node_set: set[int] = field(default_factory=set)

    def add_node(self, tag: int) -> None:
        """Append ``tag`` if not already present (dedupe within block)."""
        if tag not in self._node_set:
            self._node_set.add(tag)
            self.node_ids.append(tag)

    def add_element(self, tag: int) -> None:
        """Append ``tag`` to the block's element list."""
        self.element_ids.append(tag)


# ---------------------------------------------------------------------------
# H5Emitter
# ---------------------------------------------------------------------------

class H5Emitter:
    """Structured emitter that writes ``model.h5`` per ``h5-schema.md``.

    Construct, drive via :meth:`BuiltModel.emit`, then call
    :meth:`write` to materialize the HDF5 file. The class buffers all
    state in memory; the file is written exactly once.
    """

    def __init__(
        self,
        *,
        schema_version: str = SCHEMA_VERSION,
        model_name: str | None = None,
        apegmsh_version: str | None = None,
        snapshot_id: str | None = None,
    ) -> None:
        self._schema_version: str = schema_version
        self._model_name: str = model_name or "model"
        self._apegmsh_version: str = apegmsh_version or "unknown"
        self._snapshot_id: str = snapshot_id or ""

        self._ndm: int | None = None
        self._ndf: int | None = None

        # Nodes — stored as parallel arrays for compact write.
        self._node_tags: list[int] = []
        self._node_coords: list[tuple[float, float, float]] = []
        # Phantom-node tags written via ``node(tag, *xyz, ndf=6)`` —
        # the per-node DOF override used by the phantom-emit pre-step in
        # ``emit_mp_constraints`` (ADR 0022 INV-3).  Persisted under
        # ``/opensees/constraints/phantom_node_tags`` so a reader can
        # identify which nodes were synthesized.
        self._phantom_node_tags: list[int] = []

        # MP constraints (Phase 7b, ADR 0022).
        self._equal_dofs: list[_EqualDOFRecord] = []
        self._rigid_links: list[_RigidLinkRecord] = []
        self._rigid_diaphragms: list[_RigidDiaphragmRecord] = []
        self._embedded_nodes: list[_EmbeddedNodeRecord] = []
        # Holds the user's declaration label set via
        # ``mp_constraint_comment(name)``; consumed by the next MP
        # constraint emit (INV-2).
        self._pending_mp_name: str = ""

        # Constitutive.
        self._uniaxial: list[_MaterialRecord] = []
        self._nd: list[_MaterialRecord] = []

        # Sections.
        self._sections_simple: list[_SectionSimpleRecord] = []
        self._sections_complex: list[_SectionComplexRecord] = []
        self._open_section: _SectionComplexRecord | None = None

        # Transforms (one record per geomTransf call — see module docstring).
        self._transforms: list[_TransformRecord] = []

        # Beam integration rules (Phase 4.5).
        self._beam_integrations: list[_BeamIntegrationRecord] = []

        # Elements.
        self._elements: list[_ElementRecord] = []

        # Time series.
        self._time_series: list[_TimeSeriesRecord] = []

        # Patterns.
        self._patterns_complete: list[_PatternRecord] = []
        self._open_pattern: _PatternRecord | None = None

        # BCs (model-level).
        self._fixes: list[_FixRecord] = []
        self._masses: list[_MassRecord] = []

        # Regions (emitted from the recorder fan-out; persisted so MPCO
        # ``-R $tag`` round-trips through ``OpenSeesModel.from_h5``).
        self._regions: list[_RegionRecord] = []

        # Recorders.
        self._recorders: list[_RecorderRecord] = []
        # Active declaration context (set by ``recorder_declaration_begin``,
        # cleared by ``recorder_declaration_end``). While non-None, every
        # ``recorder(...)`` call inherits this context for schema 2.3.0
        # archival.
        self._decl_context: _DeclContext | None = None

        # Analysis chain (collected attrs).
        self._analysis_attrs: dict[str, Any] = {}
        self._analyze_call: tuple[int, float | None] | None = None

        # File-internal tag counter for ``add_oriented_elements`` (ADR
        # 0018 / ModelData declarative front-door).  Must not be mixed
        # with bridge-driven ``element()`` calls on the same instance.
        self._orientation_tag_counter: int = 0

        # Partition emission state (ADR 0027, schema 2.10.0).
        # ``_partition_current`` is the block in flight between
        # ``partition_open(rank)`` and ``partition_close()``; while
        # non-None, every :meth:`node` and :meth:`element` call also
        # appends to it.  ``_partition_blocks`` accumulates closed
        # blocks in emit order (one per rank).  ``_element_ranks`` is
        # a parallel list to ``self._elements`` carrying the rank each
        # element was emitted under (sentinel ``-1`` for outside-bracket
        # emission); it materializes the per-element ``partition_ids``
        # column on ``/opensees/element_meta/{type_token}/``.
        self._partition_current: _PartitionEmitBlock | None = None
        self._partition_blocks: list[_PartitionEmitBlock] = []
        self._element_ranks: list[int] = []

    # =====================================================================
    # Protocol — Model
    # =====================================================================

    def model(self, *, ndm: int, ndf: int) -> None:
        self._ndm = ndm
        self._ndf = ndf

    def node(
        self, tag: int, *coords: float, ndf: int | None = None,
    ) -> None:
        # Normalize to 3-tuple by zero-padding 2-D models.
        cs = tuple(float(c) for c in coords)
        if len(cs) == 2:
            x, y = cs
            triple = (x, y, 0.0)
        elif len(cs) == 3:
            x, y, z = cs
            triple = (x, y, z)
        else:
            raise ValueError(
                f"H5Emitter.node: expected 2 or 3 coordinates, got "
                f"{len(cs)}: {cs!r}"
            )
        self._node_tags.append(int(tag))
        self._node_coords.append(triple)
        # Phantom nodes (ADR 0022 INV-3) — track tags so a reader can
        # identify which nodes were synthesized by the phantom-emit
        # pre-step.  Per S2 (ADR 0033) the per-node ``ndf`` kwarg is
        # now legal on real broker nodes too, so the discriminator is
        # the explicit ``phantom_node_mode`` side-channel set by the
        # phantom-emit helpers in
        # :mod:`apeGmsh.opensees._internal.build`.
        from .._internal.tag_resolution import is_phantom_node_mode
        if is_phantom_node_mode(self):
            self._phantom_node_tags.append(int(tag))
        # Partition emission (ADR 0027) — while a per-rank block is
        # open, also record the tag on the active block so the
        # ``/opensees/partitions/partition_NN/node_ids`` dataset
        # reflects every node declared on that rank (native + foreign
        # MP-constraint declarations both count, per INV-2).
        if self._partition_current is not None:
            self._partition_current.add_node(int(tag))

    def fix(self, tag: int, *dofs: int) -> None:
        self._fixes.append(_FixRecord(tag=int(tag), dofs=tuple(int(d) for d in dofs)))

    def mass(self, tag: int, *values: float) -> None:
        self._masses.append(
            _MassRecord(tag=int(tag), values=tuple(float(v) for v in values))
        )

    # =====================================================================
    # Protocol — MP constraints (ADR 0022, Phase 7b, schema 2.7.0)
    # =====================================================================

    def equalDOF(self, master: int, slave: int, *dofs: int) -> None:
        name = self._consume_pending_mp_name()
        self._equal_dofs.append(
            _EqualDOFRecord(
                master=int(master), slave=int(slave),
                dofs=tuple(int(d) for d in dofs),
                name=name,
            )
        )

    def rigidLink(self, kind: str, master: int, slave: int) -> None:
        name = self._consume_pending_mp_name()
        self._rigid_links.append(
            _RigidLinkRecord(
                kind=str(kind),
                master=int(master), slave=int(slave),
                name=name,
            )
        )

    def rigidDiaphragm(
        self, perp_dir: int, master: int, *slaves: int,
    ) -> None:
        name = self._consume_pending_mp_name()
        self._rigid_diaphragms.append(
            _RigidDiaphragmRecord(
                perp_dir=int(perp_dir),
                master=int(master),
                slaves=tuple(int(s) for s in slaves),
                name=name,
            )
        )

    def embeddedNode(
        self, ele_tag: int, cnode: int,
        *args: int | float,
    ) -> None:
        name = self._consume_pending_mp_name()
        self._embedded_nodes.append(
            _EmbeddedNodeRecord(
                ele_tag=int(ele_tag),
                cnode=int(cnode),
                args=tuple(args),
                name=name,
            )
        )

    def mp_constraint_comment(self, name: str) -> None:
        # Latch the declaration label; the next MP-constraint call will
        # consume it via ``_consume_pending_mp_name`` (INV-2).  This is
        # the H5 emitter's analogue of the Tcl/Py ``# {name}`` line.
        self._pending_mp_name = str(name)

    def _consume_pending_mp_name(self) -> str:
        """Return-and-clear the pending mp_constraint_comment label."""
        name = self._pending_mp_name
        self._pending_mp_name = ""
        return name

    # =====================================================================
    # Protocol — Constitutive
    # =====================================================================

    def uniaxialMaterial(
        self, mat_type: str, tag: int, *params: float | str,
    ) -> None:
        self._uniaxial.append(
            _MaterialRecord(type_token=mat_type, tag=int(tag), params=tuple(params))
        )

    def nDMaterial(
        self, mat_type: str, tag: int, *params: float | str,
    ) -> None:
        self._nd.append(
            _MaterialRecord(type_token=mat_type, tag=int(tag), params=tuple(params))
        )

    def section(
        self, sec_type: str, tag: int, *params: float | str,
    ) -> None:
        self._sections_simple.append(
            _SectionSimpleRecord(
                type_token=sec_type, tag=int(tag), params=tuple(params),
            )
        )

    def geomTransf(self, t_type: str, tag: int, *vec: float) -> None:
        self._transforms.append(
            _TransformRecord(
                type_token=t_type, tag=int(tag),
                vec=tuple(float(v) for v in vec),
            )
        )

    # =====================================================================
    # Protocol — Sections that take blocks (Fiber)
    # =====================================================================

    def section_open(
        self, sec_type: str, tag: int, *params: float | str,
    ) -> None:
        if self._open_section is not None:
            raise RuntimeError(
                "H5Emitter.section_open: a section is already open "
                f"({self._open_section.type_token} tag={self._open_section.tag}); "
                "call section_close first."
            )
        self._open_section = _SectionComplexRecord(
            type_token=sec_type, tag=int(tag), params=tuple(params),
        )

    def section_close(self) -> None:
        if self._open_section is None:
            raise RuntimeError(
                "H5Emitter.section_close called with no open section."
            )
        self._sections_complex.append(self._open_section)
        self._open_section = None

    def patch(self, kind: str, *args: int | float) -> None:
        if self._open_section is None:
            raise RuntimeError(
                "H5Emitter.patch called outside a section_open / "
                "section_close block."
            )
        self._open_section.patches.append(
            _PatchRecord(kind=kind, args=tuple(args))
        )

    def fiber(
        self, y: float, z: float, area: float, mat_tag: int,
    ) -> None:
        if self._open_section is None:
            raise RuntimeError(
                "H5Emitter.fiber called outside a section_open / "
                "section_close block."
            )
        self._open_section.fibers.append(
            _FiberRecord(
                y=float(y), z=float(z),
                area=float(area), mat_tag=int(mat_tag),
            )
        )

    def layer(self, kind: str, *args: int | float) -> None:
        if self._open_section is None:
            raise RuntimeError(
                "H5Emitter.layer called outside a section_open / "
                "section_close block."
            )
        self._open_section.layers.append(
            _LayerRecord(kind=kind, args=tuple(args))
        )

    # =====================================================================
    # Protocol — Beam integration rules
    # =====================================================================

    def beamIntegration(
        self, rule_type: str, tag: int, *args: int | float | str,
    ) -> None:
        self._beam_integrations.append(
            _BeamIntegrationRecord(
                type_token=rule_type, tag=int(tag), args=tuple(args),
            )
        )

    # =====================================================================
    # Protocol — Topology
    # =====================================================================

    def element(
        self, ele_type: str, tag: int, *args: int | float | str,
    ) -> None:
        # The bridge sets _current_element_nodes via
        # set_element_nodes(emitter, ...) right before each element call
        # (see _internal/build.py emit_element_spec). Reading it here
        # gives us the connectivity for the schema's
        # /elements/{type}/connectivity dataset.
        connectivity = getattr(self, ATTR_ELEMENT_NODES, None)
        if connectivity is None:
            # Allow direct test driving without the bridge wrapper —
            # fall back to extracting node tags from args (the first
            # `n` integer args before the first non-int arg are the
            # nodes). We don't try to be clever here; tests that need a
            # specific connectivity install set_element_nodes first.
            connectivity = tuple()
        # Phase 8.6: capture the FEM element id from the side channel
        # (sentinel ``-1`` when called outside a bridge fan-out — see
        # `_internal/tag_resolution.MISSING_FEM_ELEMENT_ID`).
        fem_eid = current_fem_element_id(self)
        self._elements.append(
            _ElementRecord(
                type_token=ele_type, tag=int(tag),
                args=tuple(args),
                connectivity=tuple(int(c) for c in connectivity),
                fem_eid=fem_eid,
            )
        )
        # Partition emission (ADR 0027) — record the per-element rank
        # parallel to ``self._elements`` so the per-type
        # ``partition_ids`` column on ``/opensees/element_meta/`` can
        # be materialised at write time.  Sentinel ``-1`` for elements
        # emitted outside any ``partition_open`` / ``partition_close``
        # bracket.  Also append to the active block's ``element_ids``
        # list so the per-rank dataset reflects emission order.
        if self._partition_current is not None:
            self._element_ranks.append(self._partition_current.rank)
            self._partition_current.add_element(int(tag))
        else:
            self._element_ranks.append(-1)

    # =====================================================================
    # Public — declarative orientation inject (ADR 0018 / ModelData)
    # =====================================================================

    def add_oriented_elements(
        self,
        *,
        type_token: str,
        vecxz: "tuple[float, float, float]",
        elements: "Iterable[tuple[int, Sequence[int]]]",
        ndm: int,
    ) -> None:
        """Append one ``geomTransf`` + N elements as one orientation fact.

        Public schema-owning method used by
        :class:`apeGmsh.opensees.ModelData` to inject the per-element
        ``vecxz`` orientation the viewer's beam join needs
        (``h5_reader.element_local_axes_vecxz``).  Driving the bridge
        Protocol methods (``geomTransf`` / ``element``) from a
        declarative front-door would require pre-setting ``_internal``
        side-channels — exporting bridge-internal coupling into a
        public class.  This method appends a ``_TransformRecord`` and
        per-element ``_ElementRecord`` instances directly, stamping a
        real positive ``fem_eid`` (ADR 0018 INV-6).

        Tag values are file-internal — ``model.h5`` only needs the
        transf record's ``tag`` to match the value stored at the
        registry-defined slot of each element's ``args``.  The reader
        join (h5_reader.py:336-363) never cross-checks against a live
        OpenSees domain.

        Parameters
        ----------
        type_token
            OpenSees element type (``"forceBeamColumn"``,
            ``"elasticBeamColumn"``, ...).  Must yield a transf slot
            via :func:`_transf_arg_tail_index` for ``ndm``; an unknown
            token or one with no transf slot raises (ADR 0018 INV-7).
        vecxz
            Three-tuple ``(vx, vy, vz)``.
        elements
            Iterable of ``(fem_eid, connectivity)`` pairs.  ``fem_eid``
            must be ``> 0`` — sentinel ``-1`` is invalid for a
            declarative inject (ADR 0018 INV-6).
        ndm
            2 or 3.  Selects ``slots_2d`` vs ``slots_3d`` for the
            transf-tag positional slot (ADR 0018 INV-9).

        Raises
        ------
        ValueError
            If ``ndm`` is not in ``{2, 3}``, ``vecxz`` lacks 3
            components, ``type_token`` is unknown / has no transf
            slot, or any ``fem_eid`` is ``<= 0``.
        """
        import math

        from .._element_capabilities import (
            _ELEM_REGISTRY,
            _transf_arg_tail_index,
            known_beam_type_tokens,
        )

        if ndm not in (2, 3):
            raise ValueError(
                f"H5Emitter.add_oriented_elements: ndm must be 2 or 3, "
                f"got {ndm!r}."
            )
        if len(vecxz) != 3:
            raise ValueError(
                f"H5Emitter.add_oriented_elements: vecxz must have 3 "
                f"components, got {len(vecxz)}: {vecxz!r}."
            )
        idx = _transf_arg_tail_index(type_token, ndm, _ELEM_REGISTRY)
        if idx is None:
            valid = ", ".join(known_beam_type_tokens(ndm))
            raise ValueError(
                f"H5Emitter.add_oriented_elements: type_token="
                f"{type_token!r} has no transf slot at ndm={ndm}. "
                f"Valid beam tokens: {valid}."
            )

        items = list(elements)
        for fem_eid, _ in items:
            if int(fem_eid) <= 0:
                raise ValueError(
                    f"H5Emitter.add_oriented_elements: fem_eid must "
                    f"be > 0 (ADR 0018 INV-6 — sentinel -1 is invalid "
                    f"for a declarative inject), got {fem_eid!r}."
                )

        # Allocate file-internal tags.  Counter is monotonic on this
        # emitter; do not mix with bridge-driven ``element()`` calls
        # on the same instance.
        self._orientation_tag_counter += 1
        transf_tag = self._orientation_tag_counter
        self._transforms.append(
            _TransformRecord(
                # geomTransf type-token here is the *class string*
                # (Linear / PDelta / Corotational), independent of the
                # element type.  The reader's orientation join reads
                # only ``vec`` + ``tag`` — geomTransf class does not
                # affect orientation.  Use the schema-stable default.
                type_token="Linear",
                tag=transf_tag,
                vec=(float(vecxz[0]), float(vecxz[1]), float(vecxz[2])),
            )
        )

        # Pad args to (idx + 1) tail length with NaN sentinels — the
        # viewer's orientation join only reads the transf slot; other
        # tail columns are unread for orientation.  See
        # ``_write_element_argstack``: arity = len(connectivity),
        # tail = args[arity:].
        for fem_eid, conn in items:
            conn_t = tuple(int(c) for c in conn)
            self._orientation_tag_counter += 1
            ops_tag = self._orientation_tag_counter
            tail: list[float] = [math.nan] * (idx + 1)
            tail[idx] = float(transf_tag)
            self._elements.append(
                _ElementRecord(
                    type_token=type_token,
                    tag=ops_tag,
                    args=(*conn_t, *tail),
                    connectivity=conn_t,
                    fem_eid=int(fem_eid),
                )
            )
            # Parallel partition_ids row — declarative inject is never
            # rank-scoped (ModelData runs outside the bridge fan-out),
            # so the sentinel -1 always applies.  Keeping the lists
            # parallel is invariant: a future caller mixing this path
            # with bridge-driven element() must not break the
            # element_meta column write.
            self._element_ranks.append(-1)

    # =====================================================================
    # Protocol — Time series
    # =====================================================================

    def timeSeries(
        self, ts_type: str, tag: int, *args: int | float | str,
    ) -> None:
        self._time_series.append(
            _TimeSeriesRecord(
                type_token=ts_type, tag=int(tag), args=tuple(args),
            )
        )

    # =====================================================================
    # Protocol — Patterns
    # =====================================================================

    def pattern_open(
        self, p_type: str, tag: int, *args: int | float | str,
    ) -> None:
        if self._open_pattern is not None:
            # Auto-close a stale open pattern. Defensive; the bridge
            # always pairs open/close, but if a single-line pattern
            # (UniformExcitation) was opened and never explicitly
            # closed before another open, finalize it here.
            self._patterns_complete.append(self._open_pattern)
            self._open_pattern = None
        self._open_pattern = _PatternRecord(
            type_token=p_type, tag=int(tag), args=tuple(args),
        )

    def pattern_close(self) -> None:
        if self._open_pattern is None:
            # Allowed — single-line pattern closes are a no-op in some
            # emitters; mirror that tolerance here.
            return
        self._patterns_complete.append(self._open_pattern)
        self._open_pattern = None

    def load(self, tag: int, *forces: float) -> None:
        if self._open_pattern is None:
            raise RuntimeError(
                "H5Emitter.load called outside an open pattern."
            )
        self._open_pattern.loads.append(
            _LoadRecord(target=int(tag), forces=tuple(float(f) for f in forces))
        )

    def eleLoad(self, *args: int | float | str) -> None:
        if self._open_pattern is None:
            raise RuntimeError(
                "H5Emitter.eleLoad called outside an open pattern."
            )
        self._open_pattern.ele_loads.append(
            _EleLoadRecord(args=tuple(args))
        )

    def sp(self, tag: int, dof: int, value: float) -> None:
        if self._open_pattern is None:
            raise RuntimeError(
                "H5Emitter.sp called outside an open pattern."
            )
        self._open_pattern.sps.append(
            _SPRecord(target=int(tag), dof=int(dof), value=float(value))
        )

    # =====================================================================
    # Protocol — Recorders
    # =====================================================================

    def recorder(self, kind: str, *args: int | float | str) -> None:
        self._recorders.append(_RecorderRecord(
            kind=kind,
            args=tuple(args),
            decl_context=self._decl_context,
        ))

    def region(self, tag: int, *args: int | float | str) -> None:
        self._regions.append(_RegionRecord(tag=int(tag), args=tuple(args)))

    def recorder_declaration_begin(
        self,
        *,
        declaration_name: str,
        record_name: str | None,
        category: str,
        components: tuple[str, ...],
        raw: tuple[str, ...] = (),
        pg: tuple[str, ...] = (),
        label: tuple[str, ...] = (),
        selection: tuple[str, ...] = (),
        ids: tuple[int, ...] | None = None,
        dt: float | None = None,
        n_steps: int | None = None,
        file_root: str = ".",
    ) -> None:
        """Open a recorder-declaration archival context.

        Each subsequent :meth:`recorder` call (until the matching
        :meth:`recorder_declaration_end`) is tagged with this
        metadata, persisted in the unified ``/opensees/recorders/``
        group as schema 2.3.0 ``kind="declared"`` entries.
        """
        if self._decl_context is not None:
            raise RuntimeError(
                "H5Emitter.recorder_declaration_begin: a declaration "
                "context is already open. Nested declarations are not "
                "supported — call recorder_declaration_end() first."
            )
        self._decl_context = _DeclContext(
            declaration_name=declaration_name,
            record_name=record_name,
            category=category,
            components=components,
            raw=raw,
            pg=pg,
            label=label,
            selection=selection,
            ids=ids,
            dt=dt,
            n_steps=n_steps,
            file_root=file_root,
        )

    def recorder_declaration_end(self) -> None:
        """Close the active recorder-declaration archival context."""
        if self._decl_context is None:
            raise RuntimeError(
                "H5Emitter.recorder_declaration_end: no active "
                "declaration context (begin/end mismatch)."
            )
        self._decl_context = None

    # =====================================================================
    # Protocol — Partition emission (ADR 0027, schema 2.10.0)
    # =====================================================================

    def partition_open(self, rank: int) -> None:
        """Enter rank-scoped emit tracking.

        Subsequent :meth:`node` and :meth:`element` calls — until the
        matching :meth:`partition_close` — are mirrored onto an in-memory
        accumulator for the rank.  Each accumulator becomes one
        ``/opensees/partitions/partition_NN`` sub-group at
        :meth:`write` time.

        Unlike the streaming emitters (Tcl / Py), H5 has no notion of a
        "current rank block" in the file format; the call is purely a
        bookkeeping mode switch on the in-memory buffer.

        Raises
        ------
        RuntimeError
            If another partition block is already open.  Nested
            partition brackets are not supported — the bridge fan-out
            in ``_internal/build.py`` calls
            ``partition_open(rank) ... partition_close()`` in strict
            sequence, one rank at a time (ADR 0027 §"Implementation
            pointer").
        """
        if self._partition_current is not None:
            raise RuntimeError(
                "H5Emitter.partition_open: a partition block is "
                f"already open (rank={self._partition_current.rank}); "
                "call partition_close() first."
            )
        self._partition_current = _PartitionEmitBlock(rank=int(rank))

    def partition_close(self) -> None:
        """Close the active partition block and stash it for write-time.

        The accumulated ``node_ids`` / ``element_ids`` are frozen into
        the block; :meth:`_write_partitions` later writes them out and
        computes ``boundary_node_ids`` per-rank as the intersection
        against every other rank's node set (ADR 0027 §"Decision" —
        the boundary-node computation is symmetric across ranks and
        does not require a separate pass).

        Calling :meth:`partition_close` with no open block is a no-op
        — mirrors the tolerance :meth:`pattern_close` already provides.
        """
        if self._partition_current is None:
            return
        self._partition_blocks.append(self._partition_current)
        self._partition_current = None

    # -- Partition runtime-conditional fallback (ADR 0027 INV-5) ----------

    def parallel_runtime_fallback_numberer(
        self, primary: str, fallback: str,
    ) -> None:
        """Record the primary numberer as canonical; stash the fallback
        in ``numberer_runtime_fallback`` so re-emit can reconstruct the
        runtime conditional.

        Minimal additive change to the analysis attrs (ADR 0027 INV-5
        amendment 2026-05-23).
        """
        self._analysis_attrs["numberer"] = primary
        self._analysis_attrs["numberer_runtime_fallback"] = fallback

    def parallel_runtime_fallback_system(
        self, primary: str, fallback: str,
    ) -> None:
        """Record the primary system as canonical; stash the fallback
        in ``system_runtime_fallback`` so re-emit can reconstruct the
        runtime conditional.

        Mirror of :meth:`parallel_runtime_fallback_numberer`.
        """
        self._analysis_attrs["system"] = primary
        self._analysis_attrs["system_runtime_fallback"] = fallback

    # =====================================================================
    # Protocol — Stress control (Phase SSI-1) + Staged analysis (SSI-2)
    # =====================================================================
    #
    # H5 archival of these Protocol methods is deferred (Phase SSI-1
    # for parameter / addToParameter / step_hook_ramp; Phase SSI-2.A
    # for stage_open / stage_close; Phase SSI-2.B for domain_change).
    # Persisting them requires a new ``/opensees/stages/stage_NN``
    # zone (one group per StageRecord) carrying name + activated_pgs
    # + n_increments + dt + analysis-chain tag references + a
    # per-stage initial_stress sub-table, plus matching
    # ``_replay_into`` iteration on the read side and new
    # :class:`OpenSeesModel` accessors (``.stages()`` /
    # ``.initial_stress()``).  Bump SCHEMA_VERSION (2.11.0 → 2.12.0)
    # when that work lands.
    #
    # Until then, :meth:`apeSees.h5` raises ``NotImplementedError``
    # the moment a caller asks for an H5 archive of a build that
    # carries stage records or initial_stress records.  The bridge-
    # level guard means these emitter-side bodies are unreachable
    # from real bridge-driven call sites; they remain no-ops solely
    # so direct unit tests that drive an :class:`H5Emitter` outside
    # the bridge (Protocol-conformance smoke tests, RecordingEmitter
    # replay) still type-check and run.

    def addToParameter(
        self, tag: int, ele_tag: int, response: str,
    ) -> None:
        """No-op — Phase SSI-1 archival deferred (apeSees.h5 fails loud)."""
        del tag, ele_tag, response

    def step_hook_ramp(
        self,
        name: str,
        *,
        targets: tuple[tuple[int, float], ...],
        n_steps_to_full: float,
        phase: Literal["before", "after"] = "before",
    ) -> None:
        """No-op — Phase SSI-1 archival deferred (apeSees.h5 fails loud)."""
        del name, targets, n_steps_to_full, phase

    def stage_open(self, name: str) -> None:
        """No-op — Phase SSI-2.A archival deferred (apeSees.h5 fails loud)."""
        del name

    def stage_close(self) -> None:
        """No-op — Phase SSI-2.A archival deferred (apeSees.h5 fails loud)."""

    def domain_change(self) -> None:
        """No-op — Phase SSI-2.B archival deferred (apeSees.h5 fails loud)."""

    # =====================================================================
    # Protocol — Analysis chain
    # =====================================================================

    def constraints(self, c_type: str, *args: int | float | str) -> None:
        self._analysis_attrs["handler"] = c_type
        if args:
            self._analysis_attrs["handler_args"] = tuple(args)

    def numberer(self, n_type: str) -> None:
        self._analysis_attrs["numberer"] = n_type

    def system(self, s_type: str, *args: int | float | str) -> None:
        self._analysis_attrs["system"] = s_type
        if args:
            self._analysis_attrs["system_args"] = tuple(args)

    def test(self, t_type: str, *args: int | float | str) -> None:
        self._analysis_attrs["test"] = t_type
        if args:
            self._analysis_attrs["test_args"] = tuple(args)

    def algorithm(self, a_type: str, *args: int | float | str) -> None:
        self._analysis_attrs["algorithm"] = a_type
        if args:
            self._analysis_attrs["algorithm_args"] = tuple(args)

    def integrator(self, i_type: str, *args: int | float | str) -> None:
        self._analysis_attrs["integrator"] = i_type
        if args:
            self._analysis_attrs["integrator_args"] = tuple(args)

    def analysis(self, a_type: str) -> None:
        self._analysis_attrs["analysis"] = a_type

    def analyze(self, *, steps: int, dt: float | None = None) -> int:
        self._analyze_call = (int(steps), None if dt is None else float(dt))
        return 0

    def eigen(
        self, num_modes: int, *, solver: str = "-genBandArpack",
    ) -> list[float]:
        # ``eigen`` is a runtime one-shot retrieval — not part of the
        # model definition the H5 archive captures.  No-op here; the
        # bridge's ``apeSees.eigen(...)`` drives a LiveOpsEmitter
        # directly and never routes through H5.
        del num_modes, solver
        return []

    # =====================================================================
    # Output — write the buffered model to disk
    # =====================================================================

    def write(self, path: str) -> None:
        """Materialize the buffered model to an HDF5 file at ``path``.

        Standalone path — no broker integration.  Writes only ``/meta``
        and ``/opensees/...``; the neutral zone (``/nodes``,
        ``/elements/{type}``, …) requires a :class:`FEMData` and is
        emitted by :meth:`apeGmsh.opensees.apeSees.h5` instead.

        h5py is imported here (not at module load) so users who never
        call ``ops.h5()`` do not pay the import cost.
        """
        import h5py  # local import — lazy h5py dep; see module docstring

        with h5py.File(path, "w") as f:
            self._write_meta(f)
            self.write_opensees_into(f)

    def write_opensees_into(self, f: Any) -> None:
        """Write the bridge's ``/opensees/...`` groups into an open file.

        The composition entry point used by
        :meth:`apeGmsh.opensees.apeSees.h5` to layer ``/opensees/``
        content on top of a file the broker has already populated with
        ``/meta`` and the neutral zone.  This method does NOT write
        ``/meta`` and does NOT write ``/elements/{type}`` — both are
        broker-owned post-Phase-8.5.
        """
        self._write_bcs(f)
        self._write_materials(f)
        self._write_sections(f)
        self._write_transforms(f)
        self._write_beam_integration(f)
        self._write_element_meta(f)
        self._write_time_series(f)
        self._write_patterns(f)
        self._write_regions(f)
        self._write_recorders(f)
        self._write_constraints(f)
        self._write_partitions(f)
        self._write_analysis(f)

    # -- Per-group writers (split out so each step adds one) -------------

    def _ops_group(self, f: Any) -> Any:
        """Lazily get or create the ``/opensees/`` namespace group.

        Bridge-written groups (materials, sections, transforms, ...)
        live under ``/opensees/`` since Phase 8.4.  Created on first
        use so the incomplete fixture — no bridge content beyond
        ``/meta`` + ``/elements`` — never carries an empty
        ``/opensees`` group.
        """
        if "opensees" not in f:
            f.create_group("opensees")
        return f["opensees"]

    def _write_meta(self, f: Any) -> None:
        """Create ``/meta`` and populate its attributes."""
        meta = f.create_group("meta")
        for key, value in self._meta_attrs().items():
            _set_attr(meta, key, value)

    def _write_bcs(self, f: Any) -> None:
        """Write ``/opensees/bcs/fix`` and ``/opensees/bcs/mass`` compound datasets.

        Both are emitted only if at least one record exists. The bridge's
        ``fix`` / ``mass`` fan-out has already resolved any ``pg=``
        targets into per-node calls, so every record's ``target_kind``
        is ``"node"`` and ``target`` is the integer tag rendered as a
        string (per the schema's compound-dataset convention).
        """
        if not self._fixes and not self._masses:
            return
        bcs = self._ops_group(f).create_group("bcs")
        if self._fixes:
            self._write_bcs_fix(bcs)
        if self._masses:
            self._write_bcs_mass(bcs)

    def _write_bcs_fix(self, bcs_group: Any) -> None:
        import h5py
        import numpy as np

        ndf = max(int(self._ndf or 0), max(len(r.dofs) for r in self._fixes))
        dt = np.dtype(
            [
                ("target_kind", h5py.string_dtype(encoding="utf-8")),
                ("target", h5py.string_dtype(encoding="utf-8")),
                ("dofs", np.int64, (ndf,)),
            ]
        )
        rows = np.empty(len(self._fixes), dtype=dt)
        for i, rec in enumerate(self._fixes):
            padded = list(rec.dofs) + [0] * (ndf - len(rec.dofs))
            rows[i] = ("node", str(rec.tag), tuple(padded))
        bcs_group.create_dataset("fix", data=rows)

    def _write_bcs_mass(self, bcs_group: Any) -> None:
        import h5py
        import numpy as np

        ndf = max(int(self._ndf or 0), max(len(r.values) for r in self._masses))
        dt = np.dtype(
            [
                ("target_kind", h5py.string_dtype(encoding="utf-8")),
                ("target", h5py.string_dtype(encoding="utf-8")),
                ("values", np.float64, (ndf,)),
            ]
        )
        rows = np.empty(len(self._masses), dtype=dt)
        for i, rec in enumerate(self._masses):
            padded = list(rec.values) + [0.0] * (ndf - len(rec.values))
            rows[i] = ("node", str(rec.tag), tuple(padded))
        bcs_group.create_dataset("mass", data=rows)

    # -- Materials -------------------------------------------------------

    def _write_materials(self, f: Any) -> None:
        if not self._uniaxial and not self._nd:
            return
        materials = self._ops_group(f).create_group("materials")
        if self._uniaxial:
            uni = materials.create_group("uniaxial")
            for rec in self._uniaxial:
                self._write_material_record(uni, rec)
        if self._nd:
            nd = materials.create_group("nd")
            for rec in self._nd:
                self._write_material_record(nd, rec)

    def _write_material_record(
        self, parent: Any, rec: _MaterialRecord,
    ) -> None:
        g = parent.create_group(material_name(rec))
        _set_attr(g, "type", rec.type_token)
        _set_attr(g, "tag", rec.tag)
        _write_param_array(g, "params", rec.params)

    # -- Sections --------------------------------------------------------

    def _write_sections(self, f: Any) -> None:
        if not self._sections_simple and not self._sections_complex:
            return
        sections = self._ops_group(f).create_group("sections")
        for rec_simple in self._sections_simple:
            self._write_section_simple(sections, rec_simple)
        for rec_complex in self._sections_complex:
            self._write_section_complex(sections, rec_complex)

    def _write_section_simple(
        self, parent: Any, rec: _SectionSimpleRecord,
    ) -> None:
        g = parent.create_group(section_name_simple(rec))
        _set_attr(g, "type", rec.type_token)
        _set_attr(g, "tag", rec.tag)
        _write_param_array(g, "params", rec.params)

    def _write_section_complex(
        self, parent: Any, rec: _SectionComplexRecord,
    ) -> None:
        g = parent.create_group(section_name_complex(rec))
        _set_attr(g, "type", rec.type_token)
        _set_attr(g, "tag", rec.tag)
        _write_param_array(g, "params", rec.params)
        if rec.patches:
            self._write_patches(g, rec.patches)
        if rec.fibers:
            self._write_fibers(g, rec.fibers)
        if rec.layers:
            self._write_layers(g, rec.layers)

    def _write_patches(
        self, sec_group: Any, patches: list[_PatchRecord],
    ) -> None:
        """Write the ``patches`` compound dataset.

        Patch arg layout (per OpenSees Tcl manual):

        * ``rect``: ``matTag numSubdivY numSubdivZ yI zI yJ zJ`` → 4 coords
        * ``quad``: ``matTag numSubdivIJ numSubdivJK yI zI yJ zJ yK zK yL zL`` → 8 coords
        * ``circ``: ``matTag numSubdivCirc numSubdivRad yC zC intRad extRad startAng endAng`` → 6 coords (padded to 8)

        Unknown kinds: emit row with all-NaN coords and a
        ``__deviation__`` sibling attr noting the kind.
        """
        import h5py
        import numpy as np

        dt = np.dtype(
            [
                ("kind", h5py.string_dtype(encoding="utf-8")),
                ("material_ref", h5py.string_dtype(encoding="utf-8")),
                ("ny", np.int64),
                ("nz", np.int64),
                ("coords", np.float64, (8,)),
            ]
        )
        rows = np.empty(len(patches), dtype=dt)
        unknown_kinds: list[str] = []
        for i, p in enumerate(patches):
            mat_tag, ny, nz, coords = self._decode_patch(p, unknown_kinds)
            rows[i] = (
                p.kind, self._material_ref(mat_tag),
                ny, nz, coords,
            )
        sec_group.create_dataset("patches", data=rows)
        if unknown_kinds:
            _set_attr(
                sec_group, "__deviation_patches__",
                f"unknown patch kinds with truncated coords: "
                f"{','.join(sorted(set(unknown_kinds)))}",
            )

    def _decode_patch(
        self, p: _PatchRecord, unknown_kinds: list[str],
    ) -> tuple[int, int, int, tuple[float, ...]]:
        """Return ``(mat_tag, ny, nz, coords_padded_to_8)`` for one patch."""
        args = list(p.args)
        # First three args after kind are: matTag, n1, n2 — for all
        # standard kinds (rect / quad / circ).
        if len(args) < 3:
            unknown_kinds.append(p.kind)
            return (0, 0, 0, (float("nan"),) * 8)
        mat_tag = int(args[0])
        ny = int(args[1])
        nz = int(args[2])
        coord_args = [float(x) for x in args[3:]]
        if p.kind not in ("rect", "quad", "circ"):
            unknown_kinds.append(p.kind)
        # Pad with NaN to 8.
        padded = coord_args + [float("nan")] * (8 - len(coord_args))
        return (mat_tag, ny, nz, tuple(padded[:8]))

    def _write_fibers(
        self, sec_group: Any, fibers: list[_FiberRecord],
    ) -> None:
        import h5py
        import numpy as np

        dt = np.dtype(
            [
                ("y", np.float64),
                ("z", np.float64),
                ("area", np.float64),
                ("material_ref", h5py.string_dtype(encoding="utf-8")),
            ]
        )
        rows = np.empty(len(fibers), dtype=dt)
        for i, fiber in enumerate(fibers):
            rows[i] = (
                fiber.y, fiber.z, fiber.area,
                self._material_ref(fiber.mat_tag),
            )
        sec_group.create_dataset("fibers", data=rows)

    def _write_layers(
        self, sec_group: Any, layers: list[_LayerRecord],
    ) -> None:
        """Write the ``layers`` compound dataset.

        Layer arg layout (per OpenSees Tcl manual):

        * ``straight``: ``matTag numBars area yStart zStart yEnd zEnd`` → 4 line floats
        * ``circ``: ``matTag numBars area yC zC radius startAng endAng`` → 6 line floats

        ``line`` is sized to 6 (schema 1.1 widening from the v1.0 [4]
        spec). Straight layers pad the trailing 2 with NaN.
        """
        import h5py
        import numpy as np

        dt = np.dtype(
            [
                ("kind", h5py.string_dtype(encoding="utf-8")),
                ("material_ref", h5py.string_dtype(encoding="utf-8")),
                ("n_bars", np.int64),
                ("area", np.float64),
                ("line", np.float64, (6,)),
            ]
        )
        rows = np.empty(len(layers), dtype=dt)
        unknown_kinds: list[str] = []
        for i, lyr in enumerate(layers):
            mat_tag, n_bars, area, line = self._decode_layer(
                lyr, unknown_kinds,
            )
            rows[i] = (
                lyr.kind, self._material_ref(mat_tag),
                n_bars, area, line,
            )
        sec_group.create_dataset("layers", data=rows)
        if unknown_kinds:
            _set_attr(
                sec_group, "__deviation_layers__",
                f"unknown layer kinds: {','.join(sorted(set(unknown_kinds)))}",
            )

    def _decode_layer(
        self, lyr: _LayerRecord, unknown_kinds: list[str],
    ) -> tuple[int, int, float, tuple[float, ...]]:
        args = list(lyr.args)
        if len(args) < 3:
            unknown_kinds.append(lyr.kind)
            return (0, 0, 0.0, (float("nan"),) * 6)
        mat_tag = int(args[0])
        n_bars = int(args[1])
        area = float(args[2])
        line_args = [float(x) for x in args[3:]]
        if lyr.kind not in ("straight", "circ"):
            unknown_kinds.append(lyr.kind)
        padded = line_args + [float("nan")] * (6 - len(line_args))
        return (mat_tag, n_bars, area, tuple(padded[:6]))

    def _material_ref(self, mat_tag: int) -> str:
        """Resolve a material tag to its ``/opensees/materials/...`` HDF5 path.

        Uniaxial tags shadow nd tags in the OpenSees domain (separate
        namespaces), so we check uniaxial first, then nd. Returns the
        empty string if the tag is unknown — viewers should treat this
        as a missing reference and degrade.
        """
        for rec in self._uniaxial:
            if rec.tag == mat_tag:
                return f"/opensees/materials/uniaxial/{material_name(rec)}"
        for rec in self._nd:
            if rec.tag == mat_tag:
                return f"/opensees/materials/nd/{material_name(rec)}"
        return ""

    # -- Transforms ------------------------------------------------------

    def _write_transforms(self, f: Any) -> None:
        """Write one ``/opensees/transforms/{type}_{tag}/`` group per geomTransf call.

        See module docstring for the schema deviation rationale: the
        H5 emitter sees one call per emitted ``geomTransf`` line — for
        orientation-driven transforms the bridge fans these out across
        distinct per-element vecxz, but the streaming Protocol does
        not surface the spec boundary that would let us aggregate them
        into the schema's per-spec grouping.
        """
        if not self._transforms:
            return
        import numpy as np

        transforms = self._ops_group(f).create_group("transforms")
        for rec in self._transforms:
            g = transforms.create_group(transform_name(rec))
            _set_attr(g, "type", rec.type_token)
            _set_attr(g, "tag", rec.tag)
            # Each emitted geomTransf line corresponds to a single
            # vecxz; per_element_vecxz is therefore (1, 3).
            vec_arr = np.asarray([list(rec.vec)], dtype=np.float64)
            g.create_dataset("per_element_vecxz", data=vec_arr)
            tag_arr = np.asarray([rec.tag], dtype=np.int64)
            g.create_dataset("per_element_emitted_tag", data=tag_arr)
            _set_attr(g, "__deviation__", "per-emitted-call grouping")

    # -- Beam integration -----------------------------------------------

    def _write_beam_integration(self, f: Any) -> None:
        """Write ``/opensees/beam_integration/{type}_{tag}/`` groups."""
        if not self._beam_integrations:
            return
        bi = self._ops_group(f).create_group("beam_integration")
        for rec in self._beam_integrations:
            g = bi.create_group(beam_integration_name(rec))
            _set_attr(g, "type", rec.type_token)
            _set_attr(g, "tag", rec.tag)
            _write_param_array(g, "params", rec.args)

    # -- Element metadata (OpenSees-specific args) ----------------------

    def _write_element_meta(self, f: Any) -> None:
        """Write ``/opensees/element_meta/{type_token}/`` groups.

        Phase 8.5 split element data across two zones: the broker
        owns the neutral ``/elements/{gmsh_alias}/`` group (ids +
        connectivity), and the bridge owns the OpenSees-specific
        parameter tail (positional args, string flags,
        cross-references) under ``/opensees/element_meta/{type_token}/``.
        Phase 8.6 added ``fem_eids`` parallel to ``ids`` so a consumer
        can map an OpenSees tag back to its FEM element id.

        For each OpenSees element type token (``forceBeamColumn``,
        ``FourNodeTetrahedron``, …) seen during emit, this method
        creates one group with:

        * ``ids`` — int64 ``(N,)`` of OpenSees element tags.
        * ``fem_eids`` — int64 ``(N,)`` of FEM element ids the
          bridge fanned each OpenSees tag out from.  Entries are
          ``-1`` (:data:`MISSING_FEM_ELEMENT_ID`) for records emitted
          outside a bridge fan-out (test scenarios).
        * ``args`` — float64 ``(N, max_tail)`` of positional args
          after the connectivity prefix; NaN in slots that hold a
          string token.
        * ``args_str`` — vlen-utf8 ``(N, max_tail)`` of string
          tokens at the matching slot; empty string elsewhere.
          Present only if at least one slot is a string.
        """
        if not self._elements:
            return
        import numpy as np

        # Bin by type token; preserve per-type insertion order.  Each
        # bin entry pairs a record with its index in ``self._elements``
        # so the parallel ``self._element_ranks`` row (schema 2.10.0
        # ``partition_ids`` column) can be looked up by index.
        bins: dict[str, list[tuple[int, _ElementRecord]]] = {}
        for i, rec in enumerate(self._elements):
            bins.setdefault(rec.type_token, []).append((i, rec))

        # Safety net: if the partition tracker fell out of sync with
        # the element list, fall back to all-sentinel.  The element()
        # method is the only writer for both lists, so they should be
        # the same length — this guards against unrelated bridge
        # refactors that might bypass element() (e.g. a future
        # declarative front-door that forgets to append).
        ranks_ok = len(self._element_ranks) == len(self._elements)

        parent = self._ops_group(f).create_group("element_meta")
        for type_token, indexed_recs in bins.items():
            recs = [rec for _, rec in indexed_recs]
            g = parent.create_group(element_group_name(type_token))
            _set_attr(g, "type", type_token)
            g.create_dataset(
                "ids",
                data=np.asarray([r.tag for r in recs], dtype=np.int64),
            )
            # Phase 8.6: parallel array of FEM element ids — the
            # (fem_eid, ops_tag) mapping the master plan put under
            # `/opensees/tag_map/`.  Entries are
            # `MISSING_FEM_ELEMENT_ID` (`-1`) for records emitted
            # outside a bridge fan-out (e.g. standalone test calls).
            g.create_dataset(
                "fem_eids",
                data=np.asarray([r.fem_eid for r in recs], dtype=np.int64),
            )
            # Schema 2.10.0 (ADR 0027): parallel array of partition
            # ranks — one int64 per row of ``ids`` carrying the rank
            # the element was emitted under, or ``-1`` for elements
            # emitted outside a ``partition_open`` / ``partition_close``
            # bracket.  Always emitted (additive minor — old 2.9.x
            # readers ignore the new column).
            if ranks_ok:
                rank_row = [self._element_ranks[i] for i, _ in indexed_recs]
            else:
                rank_row = [-1] * len(indexed_recs)
            g.create_dataset(
                "partition_ids",
                data=np.asarray(rank_row, dtype=np.int64),
            )
            self._write_element_argstack(g, recs)

    def _write_element_argstack(
        self, g: Any, recs: list[_ElementRecord],
    ) -> None:
        """Write the element ``args`` / ``args_str`` array pair.

        The first ``arity`` positional args of every element are the
        node tags (already in the broker's
        ``/elements/{gmsh_alias}/connectivity`` dataset); we drop them
        and record only the tail (parameter / cross-reference payload).
        """
        import h5py
        import numpy as np

        arity = max(len(r.connectivity) for r in recs)
        max_tail = max(len(r.args) - arity for r in recs)
        if max_tail <= 0:
            return

        arg_nums = np.full((len(recs), max_tail), float("nan"), dtype=np.float64)
        arg_strs: list[list[str]] = []
        any_str = False
        for i, r in enumerate(recs):
            tail = r.args[arity:]
            row_strs: list[str] = []
            for j, v in enumerate(tail):
                if isinstance(v, str):
                    row_strs.append(v)
                    any_str = True
                else:
                    arg_nums[i, j] = float(v)
                    row_strs.append("")
            row_strs.extend([""] * (max_tail - len(row_strs)))
            arg_strs.append(row_strs)

        g.create_dataset("args", data=arg_nums)
        if any_str:
            g.create_dataset(
                "args_str",
                data=np.asarray(arg_strs, dtype=object),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )

    # -- Time series -----------------------------------------------------

    def _write_time_series(self, f: Any) -> None:
        if not self._time_series:
            return
        ts = self._ops_group(f).create_group("time_series")
        for rec in self._time_series:
            g = ts.create_group(time_series_name(rec))
            _set_attr(g, "type", rec.type_token)
            _set_attr(g, "tag", rec.tag)
            _write_param_array(g, "params", rec.args)
            # Schema's ``time`` and ``values`` arrays would require
            # vocabulary-aware decoding of args (e.g. -dt / -filePath /
            # inline values). The H5 emitter records the raw args; a
            # vocabulary-aware reader (or a future schema-bumped
            # primitive-side _emit) can populate time/values.

    # -- Patterns --------------------------------------------------------

    def _write_patterns(self, f: Any) -> None:
        # Flush any still-open pattern (defensive; bridge always closes).
        if self._open_pattern is not None:
            self._patterns_complete.append(self._open_pattern)
            self._open_pattern = None
        if not self._patterns_complete:
            return
        patterns = self._ops_group(f).create_group("patterns")
        for rec in self._patterns_complete:
            g = patterns.create_group(pattern_name(rec))
            _set_attr(g, "type", rec.type_token)
            _set_attr(g, "tag", rec.tag)
            series_ref = self._pattern_series_ref(rec)
            if series_ref is not None:
                _set_attr(g, "series_ref", series_ref)
            # Direction is a meaningful pattern-level attr for
            # UniformExcitation; surface it explicitly.
            if rec.type_token == "UniformExcitation" and rec.args:
                _set_attr(g, "direction", int(rec.args[0]))
            _write_param_array(g, "params", rec.args)

            if rec.loads:
                self._write_pattern_loads(g, rec.loads)
            if rec.sps:
                self._write_pattern_sps(g, rec.sps)
            if rec.ele_loads:
                self._write_pattern_ele_loads(g, rec.ele_loads)

    def _pattern_series_ref(self, rec: _PatternRecord) -> str | None:
        """Resolve the time-series tag a pattern references → an HDF5 path.

        For ``Plain``: ``args = (ts_tag, ...)``.
        For ``UniformExcitation``: ``args = (direction, "-accel", ts_tag)``
        (per the typed primitive's emit shape). Search ``args`` for an
        int that matches a known time-series tag; the first match wins.
        Returns ``None`` if no match.
        """
        if rec.type_token == "Plain" and rec.args:
            ts_tag = rec.args[0]
            return self._time_series_ref_for_tag(ts_tag)
        if rec.type_token == "UniformExcitation":
            for v in rec.args:
                if isinstance(v, int) and not isinstance(v, bool):
                    candidate = self._time_series_ref_for_tag(v)
                    if candidate is not None and candidate.endswith(
                        f"_{v}"
                    ) and v != rec.args[0]:
                        return candidate
            # Fallback: scan integer-coercible args other than
            # direction (args[0]) for a known time-series tag.
            for v in rec.args[1:]:
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    ref = self._time_series_ref_for_tag(int(v))
                    if ref is not None:
                        return ref
        return None

    def _time_series_ref_for_tag(self, tag: int | float | str) -> str | None:
        try:
            tag_int = int(tag)
        except (TypeError, ValueError):
            return None
        for rec in self._time_series:
            if rec.tag == tag_int:
                return f"/opensees/time_series/{time_series_name(rec)}"
        return None

    def _write_pattern_loads(
        self, g: Any, loads: list[_LoadRecord],
    ) -> None:
        import h5py
        import numpy as np

        ndf = max(int(self._ndf or 0), max(len(r.forces) for r in loads))
        dt = np.dtype(
            [
                ("target_kind", h5py.string_dtype(encoding="utf-8")),
                ("target", h5py.string_dtype(encoding="utf-8")),
                ("forces", np.float64, (ndf,)),
            ]
        )
        rows = np.empty(len(loads), dtype=dt)
        for i, r in enumerate(loads):
            padded = list(r.forces) + [float("nan")] * (ndf - len(r.forces))
            rows[i] = ("node", str(r.target), tuple(padded))
        g.create_dataset("loads", data=rows)

    def _write_pattern_sps(
        self, g: Any, sps: list[_SPRecord],
    ) -> None:
        import h5py
        import numpy as np

        dt = np.dtype(
            [
                ("target_kind", h5py.string_dtype(encoding="utf-8")),
                ("target", h5py.string_dtype(encoding="utf-8")),
                ("dof", np.int64),
                ("value", np.float64),
            ]
        )
        rows = np.empty(len(sps), dtype=dt)
        for i, r in enumerate(sps):
            rows[i] = ("node", str(r.target), r.dof, r.value)
        g.create_dataset("sps", data=rows)

    # -- Recorders -------------------------------------------------------

    def _write_regions(self, f: Any) -> None:
        """Persist ``/opensees/regions/region_NN`` groups (one per region call).

        Each group carries the integer ``tag`` attribute plus a
        ``params`` dataset capturing the raw OpenSees flag tail
        (``-node n1 n2 ...``, ``-ele e1 e2 ...``, ``-eleOnly``, etc.).
        Empty when no region was emitted — auto-emit from the recorder
        fan-out (MPCO ``pg=`` form) is the only producer today.
        """
        if not self._regions:
            return
        regions = self._ops_group(f).create_group("regions")
        for idx, rec in enumerate(self._regions):
            g = regions.create_group(f"region_{idx:03d}")
            _set_attr(g, "tag", rec.tag)
            _write_param_array(g, "params", rec.args)

    def _write_recorders(self, f: Any) -> None:
        if not self._recorders:
            return
        recorders = self._ops_group(f).create_group("recorders")
        for idx, rec in enumerate(self._recorders):
            g = recorders.create_group(recorder_name(rec, idx))
            # Schema 2.3.0: every record carries a ``kind`` attr.
            # ``typed`` — Node / Element / MPCO primitive (1:1 OpenSees).
            # ``declared`` — fan-out call from a ``RecorderDeclaration``
            # (carries the original declaration metadata alongside).
            _set_attr(g, "kind", "declared" if rec.decl_context else "typed")
            _set_attr(g, "type", rec.kind)
            # Surface the -file flag's value as an explicit attr; it's
            # the most-used identifier across recorders.
            file_path = _scan_flag(rec.args, "-file")
            if file_path is not None:
                _set_attr(g, "file", file_path)
            _write_param_array(g, "params", rec.args)
            # Schema 2.3.0: declaration metadata for ``declared`` records.
            ctx = rec.decl_context
            if ctx is not None:
                _set_attr(g, "declaration_name", ctx.declaration_name)
                _set_attr(g, "record_name", ctx.record_name)
                _set_attr(g, "category", ctx.category)
                _set_attr(g, "components", ctx.components)
                _set_attr(g, "raw", ctx.raw)
                _set_attr(g, "pg", ctx.pg)
                _set_attr(g, "label", ctx.label)
                _set_attr(g, "selection", ctx.selection)
                if ctx.ids is not None:
                    _set_attr(g, "ids", ctx.ids)
                _set_attr(g, "dt", ctx.dt)
                _set_attr(g, "n_steps", ctx.n_steps)
                _set_attr(g, "file_root", ctx.file_root)

    # -- MP constraints (Phase 7b, ADR 0022, schema 2.7.0) ---------------

    def _write_constraints(self, f: Any) -> None:
        """Write ``/opensees/constraints/`` with one compound dataset per
        MP constraint kind, plus a flat ``phantom_node_tags`` array.

        Empty groups are skipped — the group is created only if at least
        one MP constraint or phantom node was emitted.  Per ADR 0023
        the schema bump from 2.6 → 2.7 is additive: 2.6.x readers
        ignore the new group.
        """
        has_any = (
            self._equal_dofs or self._rigid_links
            or self._rigid_diaphragms or self._embedded_nodes
            or self._phantom_node_tags
        )
        if not has_any:
            return

        import numpy as np

        constraints = self._ops_group(f).create_group("constraints")

        if self._phantom_node_tags:
            constraints.create_dataset(
                "phantom_node_tags",
                data=np.asarray(self._phantom_node_tags, dtype=np.int64),
            )

        if self._equal_dofs:
            self._write_constraints_equal_dof(constraints)
        if self._rigid_links:
            self._write_constraints_rigid_link(constraints)
        if self._rigid_diaphragms:
            self._write_constraints_rigid_diaphragm(constraints)
        if self._embedded_nodes:
            self._write_constraints_embedded_node(constraints)

    def _write_constraints_equal_dof(self, parent: Any) -> None:
        import h5py
        import numpy as np

        ndf = max(
            int(self._ndf or 0),
            max(len(r.dofs) for r in self._equal_dofs),
        )
        # ndf must be at least 1 — even a 1-DOF equalDOF needs a slot.
        ndf = max(ndf, 1)
        dt = np.dtype(
            [
                ("master", np.int64),
                ("slave", np.int64),
                ("dofs", np.int64, (ndf,)),
                ("name", h5py.string_dtype(encoding="utf-8")),
            ]
        )
        rows = np.empty(len(self._equal_dofs), dtype=dt)
        for i, rec in enumerate(self._equal_dofs):
            padded = list(rec.dofs) + [0] * (ndf - len(rec.dofs))
            rows[i] = (rec.master, rec.slave, tuple(padded), rec.name)
        parent.create_dataset("equalDOF", data=rows)

    def _write_constraints_rigid_link(self, parent: Any) -> None:
        import h5py
        import numpy as np

        dt = np.dtype(
            [
                ("kind", h5py.string_dtype(encoding="utf-8")),
                ("master", np.int64),
                ("slave", np.int64),
                ("name", h5py.string_dtype(encoding="utf-8")),
            ]
        )
        rows = np.empty(len(self._rigid_links), dtype=dt)
        for i, rec in enumerate(self._rigid_links):
            rows[i] = (rec.kind, rec.master, rec.slave, rec.name)
        parent.create_dataset("rigidLink", data=rows)

    def _write_constraints_rigid_diaphragm(self, parent: Any) -> None:
        import h5py
        import numpy as np

        max_slaves = max(len(r.slaves) for r in self._rigid_diaphragms)
        max_slaves = max(max_slaves, 1)
        dt = np.dtype(
            [
                ("perp_dir", np.int32),
                ("master", np.int64),
                ("slaves", np.int64, (max_slaves,)),
                ("n_slaves", np.int32),
                ("name", h5py.string_dtype(encoding="utf-8")),
            ]
        )
        rows = np.empty(len(self._rigid_diaphragms), dtype=dt)
        for i, rec in enumerate(self._rigid_diaphragms):
            padded = (
                list(rec.slaves)
                + [0] * (max_slaves - len(rec.slaves))
            )
            rows[i] = (
                rec.perp_dir, rec.master,
                tuple(padded), len(rec.slaves), rec.name,
            )
        parent.create_dataset("rigidDiaphragm", data=rows)

    def _write_constraints_embedded_node(self, parent: Any) -> None:
        import h5py
        import numpy as np

        max_args = max(len(r.args) for r in self._embedded_nodes)
        max_args = max(max_args, 1)
        dt = np.dtype(
            [
                ("ele_tag", np.int64),
                ("cnode", np.int64),
                ("args", np.float64, (max_args,)),
                ("n_args", np.int32),
                ("name", h5py.string_dtype(encoding="utf-8")),
            ]
        )
        rows = np.empty(len(self._embedded_nodes), dtype=dt)
        for i, rec in enumerate(self._embedded_nodes):
            padded = (
                [float(v) for v in rec.args]
                + [float("nan")] * (max_args - len(rec.args))
            )
            rows[i] = (
                rec.ele_tag, rec.cnode,
                tuple(padded), len(rec.args), rec.name,
            )
        parent.create_dataset("embeddedNode", data=rows)

    # -- Partition emission (ADR 0027, schema 2.10.0) -------------------

    def _write_partitions(self, f: Any) -> None:
        """Persist ``/opensees/partitions/`` per ADR 0027.

        One ``partition_NN`` sub-group per closed
        :meth:`partition_open` / :meth:`partition_close` block; each
        sub-group carries:

        * ``rank`` (int64 attr) — the OpenSeesMP rank id.
        * ``n_elements`` / ``n_nodes`` (int64 attrs) — convenience
          counters (``len`` of the respective dataset).
        * ``element_ids`` (int64 1-D dataset) — OpenSees element tags
          owned by this rank (insertion order).
        * ``node_ids`` (int64 1-D dataset) — OpenSees node tags
          declared on this rank's block, deduped within block.
          Includes foreign-declared nodes from cross-partition MP
          constraints (ADR 0027 §"Decision" INV-2 / phantom-node
          policy).
        * ``boundary_node_ids`` (int64 1-D dataset) — nodes shared
          with at least one other rank (the per-rank set intersected
          against the union of every other rank's node set).  May be
          empty (e.g. a model with one partition).

        The parent group carries the ``n_partitions`` (int64 attr) so
        a reader can quickly probe the count without enumerating
        children.  When no partition block was ever opened, the group
        is not created at all — old 2.9.x readers see no change to
        the file shape.

        Boundary-node strategy: **one-pass intersection on write** —
        symmetric across ranks because the per-rank node sets are
        all available when this method runs (the bridge has finished
        emitting before :meth:`write_opensees_into` is called).  See
        ADR 0027 §"Decision" INV-3 for the determinism contract.
        """
        # If a caller forgot to close the final partition block, flush
        # it defensively (mirrors ``_write_patterns``'s open-pattern
        # auto-close).  The bridge always pairs open/close; this is a
        # safety net for direct tests / interactive use.
        if self._partition_current is not None:
            self._partition_blocks.append(self._partition_current)
            self._partition_current = None

        if not self._partition_blocks:
            return

        import numpy as np

        # Precompute each block's node set so the per-rank intersection
        # is O(N_blocks * mean_block_size) total.  A second loop then
        # subtracts the per-rank set from the global union to identify
        # the boundary slice for each rank.
        node_sets = [set(blk.node_ids) for blk in self._partition_blocks]

        parent = self._ops_group(f).create_group("partitions")
        _set_attr(parent, "n_partitions", len(self._partition_blocks))

        for idx, blk in enumerate(self._partition_blocks):
            g = parent.create_group(f"partition_{idx:02d}")
            _set_attr(g, "rank", blk.rank)
            _set_attr(g, "n_elements", len(blk.element_ids))
            _set_attr(g, "n_nodes", len(blk.node_ids))
            g.create_dataset(
                "element_ids",
                data=np.asarray(blk.element_ids, dtype=np.int64),
            )
            g.create_dataset(
                "node_ids",
                data=np.asarray(blk.node_ids, dtype=np.int64),
            )
            # Boundary nodes: intersect this rank's node set with the
            # union of every other rank's node set.  A node that lives
            # on rank K and rank L appears in both ranks'
            # ``boundary_node_ids`` (symmetric).  Preserve emission
            # order from this rank's ``node_ids`` so the dataset is
            # deterministic and easy to diff.
            others: set[int] = set()
            for other_idx, other_set in enumerate(node_sets):
                if other_idx == idx:
                    continue
                others.update(other_set)
            boundary = [n for n in blk.node_ids if n in others]
            g.create_dataset(
                "boundary_node_ids",
                data=np.asarray(boundary, dtype=np.int64),
            )

    # -- Analysis chain --------------------------------------------------

    def _write_analysis(self, f: Any) -> None:
        if not self._analysis_attrs and self._analyze_call is None:
            return
        analysis = self._ops_group(f).create_group("analysis")
        for key, value in self._analysis_attrs.items():
            _set_attr(analysis, key, value)
        if self._analyze_call is not None:
            steps, dt = self._analyze_call
            _set_attr(analysis, "analyze_steps", steps)
            if dt is not None:
                _set_attr(analysis, "analyze_dt", dt)

    def _write_pattern_ele_loads(
        self, g: Any, ele_loads: list[_EleLoadRecord],
    ) -> None:
        """Element-load records carry vocabulary-rich ``*args``; we
        store them as a single string array per row (since the
        ``-type``/``-ele``/``-eleRange`` flag tokens make positional
        decoding without vocabulary impractical). One row per call."""
        import h5py
        import numpy as np

        max_len = max(len(r.args) for r in ele_loads)
        rows = np.empty((len(ele_loads), max_len), dtype=object)
        for i, r in enumerate(ele_loads):
            row = [
                str(v) if isinstance(v, str) else repr(v)
                for v in r.args
            ]
            row.extend([""] * (max_len - len(row)))
            rows[i] = row
        g.create_dataset(
            "element_loads", data=rows,
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

    # =====================================================================
    # Helpers used by the writer (and by tests inspecting buffer state)
    # =====================================================================

    def _meta_attrs(self) -> dict[str, Any]:
        """Return the ``/meta`` group's attributes as a dict.

        Per ADR 0023 the bridge stamps its own per-zone marker
        ``/meta/opensees_schema_version`` alongside the legacy envelope
        ``schema_version`` key.  Bridge-only standalone files (no broker
        snapshot) carry only the OpenSees per-zone key; ``_compose_model_h5``
        adds the neutral-zone key when a broker FEM was paired in.
        """
        return {
            "schema_version": self._schema_version,
            "opensees_schema_version": self._schema_version,
            "apeGmsh_version": self._apegmsh_version,
            "created_iso": datetime.now(tz=timezone.utc).isoformat(),
            "ndm": int(self._ndm) if self._ndm is not None else 0,
            "ndf": int(self._ndf) if self._ndf is not None else 0,
            "snapshot_id": self._snapshot_id,
            "model_name": self._model_name,
        }


def material_name(rec: _MaterialRecord) -> str:
    """Generate the H5 group name for a material record (``Steel02_1``)."""
    return f"{rec.type_token}_{rec.tag}"


def section_name_simple(rec: _SectionSimpleRecord) -> str:
    """Generate the H5 group name for a simple section record."""
    return f"{rec.type_token}_{rec.tag}"


def section_name_complex(rec: _SectionComplexRecord) -> str:
    """Generate the H5 group name for a fiber / aggregator section."""
    return f"{rec.type_token}_{rec.tag}"


def transform_name(rec: _TransformRecord) -> str:
    """Generate the H5 group name for a transform record."""
    return f"{rec.type_token}_{rec.tag}"


def beam_integration_name(rec: _BeamIntegrationRecord) -> str:
    """Generate the H5 group name for a beam-integration rule."""
    return f"{rec.type_token}_{rec.tag}"


def element_group_name(type_token: str) -> str:
    """Group name under ``/elements``: keyed by element type token."""
    return type_token


def time_series_name(rec: _TimeSeriesRecord) -> str:
    """Generate the H5 group name for a time-series record."""
    return f"{rec.type_token}_{rec.tag}"


def pattern_name(rec: _PatternRecord) -> str:
    """Generate the H5 group name for a pattern record."""
    return f"{rec.type_token}_{rec.tag}"


def recorder_name(rec: _RecorderRecord, idx: int) -> str:
    """Generate the H5 group name for a recorder record.

    Recorders don't carry a tag in the Protocol (they're side-effects
    on the OpenSees domain); we name them positionally by their order
    in the emit stream as ``{kind}_{idx}``.
    """
    return f"{rec.kind}_{idx}"
