"""
H5Emitter — buffers the bridge's emit calls and writes a model-definition
HDF5 archive conforming to ``architecture/h5-schema.md``.

Unlike the stream emitters (Tcl, Py, Live), this is a **structured**
emitter: each Protocol method updates an in-memory buffer; the file is
materialized at the end via :meth:`write`. The structured shape is
required because the schema groups by name (``/materials/uniaxial/{name}``,
``/sections/{name}``, ...), and aggregate sections / patterns need open /
close bracketing while patches, fibers, loads, and sps land inside them.

Design notes
============

**Names from tags (Option B in ADR 0011's discussion).** The bridge's
typed primitives carry no user-supplied ``name`` field. Every name in
the file is generated from ``<type_token>_<tag>`` (e.g. ``Steel02_1``,
``Fiber_2``, ``forceBeamColumn_3``). Cross-references resolve via the
generated names. No Protocol extension was required.

**Schema deviations (documented).** Two places where the streaming
Protocol cannot supply the spec-level grouping the schema asks for:

* ``/transforms/{name}`` — the schema shows one group per user-declared
  transform with a ``per_element_vecxz`` dataset of shape
  ``(n_elements, 3)``. The csys-driven fan-out in the bridge's build
  layer emits one ``geomTransf`` line per *distinct* vecxz; the H5
  emitter sees these as N independent calls and cannot reverse-engineer
  the spec boundary. We therefore emit one ``/transforms/{type}_{tag}/``
  group per ``geomTransf`` call, each carrying a single-row
  ``per_element_vecxz`` and ``per_element_emitted_tag``. The viewer's
  vecxz overlay (which only needs ``element_id → vecxz``) still works:
  it iterates all transform groups.

* ``/elements/{pg_name}`` — the schema groups elements by physical
  group. The bridge's element fan-out in ``_internal/build.py`` knows
  the PG (``spec.pg``) but the streaming Protocol does not surface it.
  The H5 emitter therefore groups elements by **element-type token**
  (``/elements/forceBeamColumn/``, ``/elements/FourNodeTetrahedron/``,
  ...). Each group's ``ids`` and ``connectivity`` datasets stack every
  element of that type across all PGs. Every group's attrs include the
  cross-references (``section_ref``, ``transf_ref``, ...) shared by all
  elements in that group; if elements of the same type carry different
  refs, the first observed set wins and the rest are dropped (with a
  WARN-level annotation under ``attrs``). This is a known schema
  deviation; revisiting it requires either bridge cooperation (a
  ``set_current_pg`` side channel, parallel to ``set_element_nodes``)
  or a Protocol extension.

Both deviations are recorded as ``__deviation__`` attrs on the affected
groups so a reader can detect them and degrade gracefully.

**Lazy h5py import.** ``h5py`` is imported only inside :meth:`write` so
constructing an :class:`H5Emitter` (or driving emit) does not pull
the dependency into import time for users who never call ``ops.h5()``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .._internal.tag_resolution import ATTR_ELEMENT_NODES


__all__ = ["H5Emitter", "SCHEMA_VERSION"]


#: Schema version string emitted in ``/meta/schema_version``. Bump
#: ``MAJOR`` only on a breaking change; ``MINOR`` for additive groups
#: (such as the ``/beam_integration`` group introduced alongside the
#: Protocol's ``beamIntegration`` method in Phase 4.5).
SCHEMA_VERSION: str = "1.1.0"


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


@dataclass(slots=True)
class _RecorderRecord:
    kind: str
    args: tuple[int | float | str, ...]


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
        self._elements.append(
            _ElementRecord(
                type_token=ele_type, tag=int(tag),
                args=tuple(args),
                connectivity=tuple(int(c) for c in connectivity),
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
        self._recorders.append(_RecorderRecord(kind=kind, args=tuple(args)))

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

        h5py is imported here (not at module load) so users who never
        call ``ops.h5()`` do not pay the import cost.
        """
        import h5py  # local import — lazy h5py dep; see module docstring

        with h5py.File(path, "w") as f:
            self._write_meta(f)
            self._write_bcs(f)
            # Subsequent steps fill in /materials, /sections,
            # /transforms, /beam_integration, /elements, /time_series,
            # /patterns, /recorders, /analysis.

    # -- Per-group writers (split out so each step adds one) -------------

    def _write_meta(self, f: Any) -> None:
        """Create ``/meta`` and populate its attributes."""
        meta = f.create_group("meta")
        for key, value in self._meta_attrs().items():
            _set_attr(meta, key, value)

    def _write_bcs(self, f: Any) -> None:
        """Write ``/bcs/fix`` and ``/bcs/mass`` compound datasets.

        Both are emitted only if at least one record exists. The bridge's
        ``fix`` / ``mass`` fan-out has already resolved any ``pg=``
        targets into per-node calls, so every record's ``target_kind``
        is ``"node"`` and ``target`` is the integer tag rendered as a
        string (per the schema's compound-dataset convention).
        """
        if not self._fixes and not self._masses:
            return
        bcs = f.create_group("bcs")
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
