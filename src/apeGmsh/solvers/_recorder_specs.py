"""Recorder spec data types — declarative records and resolved spec.

The user declares recorders against PG/label/selection names via
``g.opensees.recorders.*``. Those declarations live as
:class:`RecorderRecord` instances on the composite. Calling
``recorders.resolve(fem)`` produces a :class:`ResolvedRecorderSpec`
— a snapshot tied to a specific FEMData by ``snapshot_id``.

Only the spec is consumed downstream:
- Phase 5 / 8: emit Tcl/Python recorder commands or ``recorder mpco`` lines
- Phase 7: drive in-process domain capture

The spec is gmsh-independent and OpenSees-independent at the type
level — just a numpy + dataclass payload.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Optional

import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    pass


# Allowed categories — these match the user-facing methods on
# ``Recorders``. ``modal`` is special: no target, has ``n_modes``.
ALL_CATEGORIES: tuple[str, ...] = (
    "nodes",
    "elements",
    "line_stations",
    "gauss",
    "fibers",
    "layers",
    "modal",
)


# =====================================================================
# Declarative record (pre-resolution)
# =====================================================================

@dataclass(frozen=True)
class RecorderRecord:
    """One declarative recorder entry — what the user wrote.

    Selectors (``pg`` / ``label`` / ``selection`` / ``ids``) follow the
    same vocabulary as the read-side composite API. Multiple named
    selectors combine as union; ``ids`` is mutually exclusive with the
    named selectors. ``modal`` records ignore selectors and use
    ``n_modes`` instead.
    """

    category: str                              # one of ALL_CATEGORIES
    components: tuple[str, ...]                # canonical names (shorthand expanded)
    name: str                                  # auto-generated or user-provided

    # Selection (named selectors are tuples of names; ``ids`` is a flat tuple)
    pg: tuple[str, ...] = ()
    label: tuple[str, ...] = ()
    selection: tuple[str, ...] = ()
    ids: Optional[tuple[int, ...]] = None

    # Cadence (at most one set; both ``None`` means every analysis step)
    dt: Optional[float] = None
    n_steps: Optional[int] = None

    # Modal-only
    n_modes: Optional[int] = None


# =====================================================================
# Resolved record (post-resolution against FEMData)
# =====================================================================

@dataclass(frozen=True)
class LayerSectionDef:
    """One LayeredShellFiberSection's per-layer composition.

    Populated from ``g.opensees._sections`` at recorder-resolve time
    for ``category="layers"`` records. Carries the data DomainCapture
    cannot probe from the live openseespy domain — per-layer
    thickness and material tag — so the capture-side
    ``write_layers_group`` can populate the LayerSlab schema fully.

    See :class:`LayerSectionMetadata` for the per-record bundle.
    """
    section_tag: int
    section_name: str
    n_layers: int
    thickness: ndarray              # (n_layers,) float64, mid-plane local-z
    material_tags: ndarray          # (n_layers,) int64


@dataclass(frozen=True)
class LayerSectionMetadata:
    """Session-side info for one ``category="layers"`` resolved record.

    Bundles the LayeredShell section definitions referenced by the
    record's elements plus an ``element_to_section`` lookup map.
    DomainCapture's ``_LayerCapturer`` consumes this to populate
    ``LayerSlab.thickness`` (and to validate the layer count when
    probing the live domain).

    Per-element local-axes quaternions are NOT stored here — those
    are computed at capture time from element node coordinates
    (pure geometry, no session info needed).
    """
    sections: dict[int, LayerSectionDef]      # section_tag → def
    element_to_section: dict[int, int]        # eid → section_tag


@dataclass(frozen=True)
class ResolvedRecorderRecord:
    """A :class:`RecorderRecord` after resolution against a FEMData.

    All selectors collapse to concrete ID arrays. The original
    declaration is preserved on ``source`` for inspection / debugging.
    """

    category: str
    name: str
    components: tuple[str, ...]
    dt: Optional[float]
    n_steps: Optional[int]

    # Concrete IDs after resolution — exactly one is populated
    # depending on category (and ``modal`` has neither).
    node_ids: Optional[ndarray] = None
    element_ids: Optional[ndarray] = None

    # Optional OpenSees C++ class name shared by all elements in this
    # record. Populated by the bridge when the user assigned a single
    # class to the underlying PG/label. Used by the .out transcoder
    # to disambiguate catalog entries that share a flat size for the
    # same response token (e.g. FourNodeTetrahedron vs SSPbrick).
    # Other paths (DomainCapture via ``ops.eleType``, MPCO via the
    # bracket key) get class info directly and do not consume this.
    element_class_name: Optional[str] = None

    # Modal-only
    n_modes: Optional[int] = None

    # ``category="layers"``-only — populated when the resolver has
    # an OpenSees back-reference and the record's elements are
    # assigned a LayeredShellFiberSection. ``None`` otherwise.
    layer_section_metadata: Optional[LayerSectionMetadata] = None

    # The original declarative record (back-reference for inspection)
    source: Optional[RecorderRecord] = None


# =====================================================================
# Resolved spec — collection + manifest serialization
# =====================================================================

@dataclass(frozen=True)
class ResolvedRecorderSpec:
    """A complete recorder spec resolved against a FEMData snapshot.

    Carries:
    - The fem ``snapshot_id`` (drift detection contract)
    - A tuple of :class:`ResolvedRecorderRecord` (one per declaration)

    The spec is the bridge between the declarative side
    (``g.opensees.recorders``) and any execution strategy
    (Tcl emission, in-process capture, MPCO bridge).
    """

    fem_snapshot_id: str
    records: tuple[ResolvedRecorderRecord, ...] = field(default_factory=tuple)

    # ---------- iteration helpers ----------

    def __iter__(self) -> Iterable[ResolvedRecorderRecord]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def by_category(self, category: str) -> tuple[ResolvedRecorderRecord, ...]:
        """Records whose ``category`` matches."""
        return tuple(r for r in self.records if r.category == category)

    # ---------- display ----------

    def __repr__(self) -> str:
        lines = [
            f"ResolvedRecorderSpec(fem_snapshot_id={self.fem_snapshot_id[:8]}…, "
            f"{len(self.records)} records)"
        ]
        for r in self.records:
            sel = self._record_selection_summary(r)
            cad = self._record_cadence_summary(r)
            lines.append(f"  - {r.category} {r.name!r}: {sel}, {cad}, "
                         f"components={list(r.components)}")
        return "\n".join(lines)

    @staticmethod
    def _record_selection_summary(r: ResolvedRecorderRecord) -> str:
        if r.category == "modal":
            return f"n_modes={r.n_modes}"
        if r.category == "nodes":
            n = 0 if r.node_ids is None else r.node_ids.size
            return f"{n} nodes"
        n = 0 if r.element_ids is None else r.element_ids.size
        return f"{n} elements"

    @staticmethod
    def _record_cadence_summary(r: ResolvedRecorderRecord) -> str:
        if r.dt is not None:
            return f"dt={r.dt}"
        if r.n_steps is not None:
            return f"every {r.n_steps} steps"
        return "every step"

    # ---------- HDF5 manifest ----------

    def to_manifest_h5(self, path) -> None:
        """Serialize to a small HDF5 sidecar manifest.

        Used by Phase 5/6 emission and cache layers. The format is
        self-contained: it can be reloaded via ``from_manifest_h5``
        to reconstruct an equivalent spec without an active FEMData.
        """
        import h5py
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            f.attrs["schema_version"] = "1.0"
            f.attrs["fem_snapshot_id"] = self.fem_snapshot_id
            recs_grp = f.create_group("records")
            for i, r in enumerate(self.records):
                sub = recs_grp.create_group(f"record_{i:04d}")
                sub.attrs["category"] = r.category
                sub.attrs["name"] = r.name
                # Cadence
                if r.dt is not None:
                    sub.attrs["dt"] = float(r.dt)
                if r.n_steps is not None:
                    sub.attrs["n_steps"] = int(r.n_steps)
                if r.n_modes is not None:
                    sub.attrs["n_modes"] = int(r.n_modes)
                if r.element_class_name is not None:
                    sub.attrs["element_class_name"] = r.element_class_name
                # Components
                sub.create_dataset(
                    "components",
                    data=np.array(r.components, dtype=h5py.string_dtype()),
                )
                # Resolved IDs
                if r.node_ids is not None:
                    sub.create_dataset("node_ids", data=r.node_ids)
                if r.element_ids is not None:
                    sub.create_dataset("element_ids", data=r.element_ids)

    # ---------- Domain capture (Phase 7) ----------

    def capture(
        self,
        path,
        fem,
        *,
        ndm: int = 3,
        ndf: int = 6,
        ops=None,
    ):
        """Open a :class:`DomainCapture` for in-process recording.

        Returns a context manager that wraps the openseespy domain.
        Use as::

            with spec.capture(path="run.h5", fem=fem) as cap:
                cap.begin_stage("gravity", kind="static")
                for _ in range(n):
                    ops.analyze(1, 1.0)
                    cap.step(t=ops.getTime())
                cap.end_stage()
                cap.capture_modes()
        """
        from ..results.capture._domain import DomainCapture
        return DomainCapture(
            self, path, fem, ndm=ndm, ndf=ndf, ops=ops,
        )

    # ---------- Tcl / Python emission (Phase 5) ----------

    def to_tcl_commands(
        self, *, output_dir: str = "", file_format: str = "out",
    ) -> list[str]:
        """Emit ``recorder ...`` Tcl command lines.

        Parameters
        ----------
        output_dir : str
            Directory prefix for emitted output files (e.g.
            ``"out/"``). Empty string emits filenames in the cwd.
        file_format : str
            ``"out"`` (text) or ``"xml"`` (self-describing). Defaults
            to text.
        """
        from ._recorder_emit import emit_spec_tcl
        return emit_spec_tcl(
            self.records, output_dir=output_dir, file_format=file_format,
        )

    def to_python_commands(
        self, *, output_dir: str = "", file_format: str = "out",
    ) -> list[str]:
        """Emit ``ops.recorder(...)`` Python call lines."""
        from ._recorder_emit import emit_spec_python
        return emit_spec_python(
            self.records, output_dir=output_dir, file_format=file_format,
        )

    # ---------- MPCO bridge (Phase 8) ----------

    def to_mpco_tcl_command(
        self, *, output_dir: str = "", filename: str = "run.mpco",
    ) -> str:
        """Emit the single ``recorder mpco ...`` Tcl command for this spec.

        MPCO records the whole result tensor/vector for each token,
        across the entire model. We aggregate unique MPCO tokens
        across all records (canonical components map to MPCO's
        broader token names — e.g. ``stress_xx`` → ``stress``,
        ``displacement_z`` → ``displacement``).

        Cadence: smallest ``dt`` across records, or smallest
        ``n_steps``, or every step (no ``-T`` flag). MPCO's ``-T``
        accepts both ``dt`` and ``nsteps`` (per the mpco-recorder skill).
        """
        from ._recorder_emit import emit_mpco_tcl
        return emit_mpco_tcl(
            self.records, output_dir=output_dir, filename=filename,
        )

    def to_mpco_python_command(
        self, *, output_dir: str = "", filename: str = "run.mpco",
    ) -> str:
        """Emit the single ``ops.recorder('mpco', ...)`` Python call."""
        from ._recorder_emit import emit_mpco_python
        return emit_mpco_python(
            self.records, output_dir=output_dir, filename=filename,
        )

    # ---------- HDF5 manifest ----------

    @classmethod
    def from_manifest_h5(cls, path) -> "ResolvedRecorderSpec":
        """Load a spec from its HDF5 manifest."""
        import h5py

        with h5py.File(path, "r") as f:
            snapshot_id = str(f.attrs["fem_snapshot_id"])
            records: list[ResolvedRecorderRecord] = []
            recs_grp = f["records"]
            for key in sorted(recs_grp.keys()):
                sub = recs_grp[key]
                attrs = sub.attrs
                comps = tuple(
                    s.decode() if isinstance(s, bytes) else str(s)
                    for s in np.asarray(sub["components"][...])
                )
                node_ids = (
                    np.asarray(sub["node_ids"][...], dtype=np.int64)
                    if "node_ids" in sub else None
                )
                element_ids = (
                    np.asarray(sub["element_ids"][...], dtype=np.int64)
                    if "element_ids" in sub else None
                )
                records.append(ResolvedRecorderRecord(
                    category=str(attrs["category"]),
                    name=str(attrs["name"]),
                    components=comps,
                    dt=float(attrs["dt"]) if "dt" in attrs else None,
                    n_steps=int(attrs["n_steps"]) if "n_steps" in attrs else None,
                    n_modes=int(attrs["n_modes"]) if "n_modes" in attrs else None,
                    node_ids=node_ids,
                    element_ids=element_ids,
                    element_class_name=(
                        str(attrs["element_class_name"])
                        if "element_class_name" in attrs else None
                    ),
                ))
        return cls(fem_snapshot_id=snapshot_id, records=tuple(records))
