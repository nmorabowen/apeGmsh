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
``1.1.0 → 2.0.0`` and the reference reader's
``EXPECTED_SCHEMA_MAJOR`` jumps ``1 → 2``.

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
from typing import Any

from .._internal.tag_resolution import (
    ATTR_ELEMENT_NODES,
    MISSING_FEM_ELEMENT_ID,
    current_fem_element_id,
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
SCHEMA_VERSION: str = "2.5.0"


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

@dataclass(slots=True)
class _MaterialRecord:
    type_token: str
    tag: int
    params: tuple[float | str, ...]


@dataclass(slots=True)
class _SectionSimpleRecord:
    type_token: str
    tag: int
    params: tuple[float | str, ...]


@dataclass(slots=True)
class _PatchRecord:
    kind: str
    args: tuple[int | float, ...]


@dataclass(slots=True)
class _FiberRecord:
    y: float
    z: float
    area: float
    mat_tag: int


@dataclass(slots=True)
class _LayerRecord:
    kind: str
    args: tuple[int | float, ...]


@dataclass(slots=True)
class _SectionComplexRecord:
    """A section opened with :meth:`section_open` and populated with
    patches / fibers / layers before :meth:`section_close`."""
    type_token: str
    tag: int
    params: tuple[float | str, ...]
    patches: list[_PatchRecord] = field(default_factory=list)
    fibers: list[_FiberRecord] = field(default_factory=list)
    layers: list[_LayerRecord] = field(default_factory=list)


@dataclass(slots=True)
class _TransformRecord:
    type_token: str
    tag: int
    vec: tuple[float, ...]


@dataclass(slots=True)
class _BeamIntegrationRecord:
    type_token: str
    tag: int
    args: tuple[int | float | str, ...]


@dataclass(slots=True)
class _ElementRecord:
    type_token: str
    tag: int
    args: tuple[int | float | str, ...]
    connectivity: tuple[int, ...]
    fem_eid: int = MISSING_FEM_ELEMENT_ID


@dataclass(slots=True)
class _TimeSeriesRecord:
    type_token: str
    tag: int
    args: tuple[int | float | str, ...]


@dataclass(slots=True)
class _LoadRecord:
    target: int
    forces: tuple[float, ...]


@dataclass(slots=True)
class _SPRecord:
    target: int
    dof: int
    value: float


@dataclass(slots=True)
class _EleLoadRecord:
    args: tuple[int | float | str, ...]


@dataclass(slots=True)
class _PatternRecord:
    """A pattern declared via :meth:`pattern_open`. ``loads`` / ``sps``
    / ``ele_loads`` are populated only if the pattern was opened as a
    block (``Plain`` / ``MultiSupport``); single-line patterns
    (``UniformExcitation``) close immediately on the next non-load call
    (the bridge calls ``pattern_close`` itself)."""
    type_token: str
    tag: int
    args: tuple[int | float | str, ...]
    loads: list[_LoadRecord] = field(default_factory=list)
    sps: list[_SPRecord] = field(default_factory=list)
    ele_loads: list[_EleLoadRecord] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _DeclContext:
    """Declaration metadata captured between
    :meth:`Emitter.recorder_declaration_begin` and
    :meth:`Emitter.recorder_declaration_end`. Attached to each
    :class:`_RecorderRecord` emitted while a context is active, so
    schema 2.3.0's unified ``/opensees/recorders/`` group can persist
    the original declaration intent alongside the fan-out call.
    """

    declaration_name: str
    record_name: str | None
    category: str
    components: tuple[str, ...]
    raw: tuple[str, ...]
    pg: tuple[str, ...]
    label: tuple[str, ...]
    selection: tuple[str, ...]
    ids: tuple[int, ...] | None
    dt: float | None
    n_steps: int | None
    file_root: str


@dataclass(slots=True)
class _RecorderRecord:
    kind: str
    args: tuple[int | float | str, ...]
    decl_context: _DeclContext | None = None


@dataclass(slots=True)
class _FixRecord:
    tag: int
    dofs: tuple[int, ...]


@dataclass(slots=True)
class _MassRecord:
    tag: int
    values: tuple[float, ...]


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

    # =====================================================================
    # Protocol — Model
    # =====================================================================

    def model(self, *, ndm: int, ndf: int) -> None:
        self._ndm = ndm
        self._ndf = ndf

    def node(self, tag: int, *coords: float) -> None:
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

    def fix(self, tag: int, *dofs: int) -> None:
        self._fixes.append(_FixRecord(tag=int(tag), dofs=tuple(int(d) for d in dofs)))

    def mass(self, tag: int, *values: float) -> None:
        self._masses.append(
            _MassRecord(tag=int(tag), values=tuple(float(v) for v in values))
        )

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
    # Protocol — Analysis chain
    # =====================================================================

    def constraints(self, c_type: str, *args: float) -> None:
        self._analysis_attrs["handler"] = c_type
        if args:
            self._analysis_attrs["handler_args"] = tuple(float(a) for a in args)

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
        self._write_recorders(f)
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

        # Bin by type token; preserve per-type insertion order.
        bins: dict[str, list[_ElementRecord]] = {}
        for rec in self._elements:
            bins.setdefault(rec.type_token, []).append(rec)

        parent = self._ops_group(f).create_group("element_meta")
        for type_token, recs in bins.items():
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
        """Return the ``/meta`` group's attributes as a dict."""
        return {
            "schema_version": self._schema_version,
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
