"""DomainCapture — in-process recording during an openseespy analysis.

Wraps a ``ResolvedRecorderSpec`` and a target HDF5 path; the user
drives the analysis loop and calls ``step(t)`` after each
``ops.analyze(...)``. The capture queries the openseespy domain
directly (``ops.nodeDisp``, ``ops.nodeReaction``, ``ops.eleResponse``,
…), buffers per-record values in RAM, and writes a chunk to the
native HDF5 file at ``end_stage()``.

Scope
-----
- **Nodes records**: full support for kinematic components
  (``displacement_*``, ``rotation_*``, ``velocity_*``,
  ``acceleration_*``, ``angular_velocity_*``,
  ``angular_acceleration_*``), reactions, and pore pressure.
- **Modal records**: ``capture_modes()`` runs ``ops.eigen()``
  and writes one stage per mode (``kind="mode"``).
- **Gauss records** (Phase 11a): continuum stress/strain at GPs,
  for catalogued element classes. Per-element class is queried
  via ``ops.eleType(eid)``; the response catalog
  (``apeGmsh.solvers._element_response``) maps it to a
  ``ResponseLayout``; per-step flat arrays from
  ``ops.eleResponse`` are unflattened and persisted via
  ``NativeWriter.write_gauss_group``.
- **Line-stations records** (Phase 11b): force-/disp-based beam-
  column section forces, probed via
  ``ops.eleResponse(eid, "section", str(ip), "force")``.
- **Per-element-node forces** (Phase 11b, ``category="elements"``):
  closed-form elastic-beam globalForce / localForce vectors.
- **Fibers records (Phase 11e)**: beam-column fiber sections
  (``FiberSection2d`` / ``FiberSection3d``) are fully supported.
  Per ``(element, section)``, ``ops.eleResponse(eid, "section",
  str(sec), "fiberData2")`` returns a single flat vector of
  ``[y, z, area, material_tag, stress, strain] × n_fibers`` —
  geometry is captured on the first step, stress/strain on every
  step, filtered to the record's components. Indexed variants
  (``fiber_stress_<n>``) are not handled at the capture layer;
  request the bulk ``fiber_stress`` / ``fiber_strain`` and post-
  filter via the read-side composite API.
- **Layers records (Phase 11f)**: layered-shell sections
  (``LayeredShellFiberSection`` on ``ASDShellQ4`` /
  ``ShellMITC4`` / ``ShellDKGQ`` / etc.) are fully supported.
  Per ``(element, surface_gp, layer)``, ``ops.eleResponse(eid,
  "material", str(gp), "fiber", str(layer), "stress"|"strain")``
  returns a flat per-layer-cell vector. Per-layer **component
  count is auto-discovered** on the first probe; output is
  surfaced as bare ``fiber_stress`` / ``fiber_strain`` (when
  N==1) or indexed ``fiber_stress_<k>`` / ``fiber_strain_<k>``
  (when N>1).

  Per-layer thickness + material tags come from
  ``record.layer_section_metadata``, populated at recorder-resolve
  time by reading the OpenSees back-reference's section registry
  (see :class:`apeGmsh.solvers._recorder_specs.LayerSectionMetadata`).
  Per-element local-axes quaternions are computed from element
  node coordinates via :mod:`apeGmsh.results._shell_geometry`.
  Layer records resolved without an OpenSees back-reference (i.e.
  ``Recorders()`` standalone) raise at ``DomainCapture`` open
  time with a clear error.

Usage
-----
::

    spec = g.opensees.recorders.resolve(fem, ndm=3, ndf=6)
    with spec.capture(path="run.h5", fem=fem, ndm=3, ndf=6) as cap:
        cap.begin_stage("gravity", kind="static")
        for _ in range(n_grav):
            ops.analyze(1, 1.0)
            cap.step(t=ops.getTime())
        cap.end_stage()

        cap.begin_stage("dynamic", kind="transient")
        for _ in range(n_dyn):
            ops.analyze(1, dt)
            cap.step(t=ops.getTime())
        cap.end_stage()

        cap.capture_modes()

    results = Results.from_native("run.h5", fem=fem)
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy import ndarray

from ...solvers._element_response import (
    CUSTOM_RULE_CATALOG,
    FIBER_CATALOG,
    INFERRED_SECTION_CODES_TABLE,
    LAYER_CATALOG,
    NODAL_FORCE_CATALOG,
    RESPONSE_CATALOG,
    CustomRuleLayout,
    FiberSectionLayout,
    IntRule,
    LayeredShellLayout,
    NodalForceLayout,
    ResponseLayout,
    catalog_token_for_keyword,
    class_dimension,
    gauss_keyword_for_canonical,
    gauss_routing_for_canonical,
    infer_section_codes,
    is_catalogued,
    is_custom_rule_catalogued,
    is_fiber_catalogued,
    is_layer_catalogued,
    is_line_station_synthesis_catalogued,
    is_nodal_force_catalogued,
    lookup,
    lookup_custom_rule,
    lookup_fiber,
    lookup_layer,
    lookup_nodal_force,
    needs_per_material_strain,
    normalise_integration_points,
    resolve_layout_from_gp_x,
    synthesize_line_station_layout_for_elastic_beam,
    unflatten,
    unflatten_nodal,
)
from .._shell_geometry import shell_quaternion

# Re-exports under their original underscore-prefixed names so the
# Step 2b tests (which import the private names) keep working
# without churn. Public consumers (e.g. the .out transcoder) should
# import these from ``apeGmsh.solvers._element_response`` directly.
_INFERRED_SECTION_CODES = INFERRED_SECTION_CODES_TABLE
_class_dimension = class_dimension
_infer_section_codes = infer_section_codes
_normalise_integration_points = normalise_integration_points

if TYPE_CHECKING:
    from ...mesh.FEMData import FEMData
    from ...solvers._recorder_specs import (
        ResolvedRecorderRecord,
        ResolvedRecorderSpec,
    )


# =====================================================================
# Canonical → openseespy call mapping (nodal)
# =====================================================================

# Translational axis → 1-based DOF; rotational axis → 4/5/6.
_TRANS_DOF = {"x": 1, "y": 2, "z": 3}
_ROT_DOF = {"x": 4, "y": 5, "z": 6}

# Canonical prefix → (ops_function_name, axis_kind)
# axis_kind ∈ {"trans", "rot"} selects which DOF table to use.
_NODE_PREFIX_OPS: dict[str, tuple[str, str]] = {
    "displacement": ("nodeDisp", "trans"),
    "rotation": ("nodeDisp", "rot"),
    "velocity": ("nodeVel", "trans"),
    "angular_velocity": ("nodeVel", "rot"),
    "acceleration": ("nodeAccel", "trans"),
    "angular_acceleration": ("nodeAccel", "rot"),
    "displacement_increment": ("nodeDisp", "trans"),  # OpenSees uses nodeDisp + diff manually; treat as disp
    "reaction_force": ("nodeReaction", "trans"),
    "reaction_moment": ("nodeReaction", "rot"),
    "force": ("nodeUnbalance", "trans"),
    "moment": ("nodeUnbalance", "rot"),
}

# Scalar canonical names with a fixed ops call.
_NODE_SCALAR_OPS: dict[str, tuple[str, Optional[int]]] = {
    "pore_pressure": ("nodePressure", None),
}


def _component_to_ops_call(canonical: str) -> Optional[tuple[str, Optional[int]]]:
    """Map ``"displacement_x"`` → ``("nodeDisp", 1)``, etc.

    Returns ``None`` if the canonical name has no nodal ops mapping.
    """
    if canonical in _NODE_SCALAR_OPS:
        return _NODE_SCALAR_OPS[canonical]
    if "_" not in canonical:
        return None
    prefix, axis = canonical.rsplit("_", 1)
    table = _NODE_PREFIX_OPS.get(prefix)
    if table is None:
        return None
    fn_name, axis_kind = table
    dof_table = _TRANS_DOF if axis_kind == "trans" else _ROT_DOF
    if axis not in dof_table:
        return None
    return (fn_name, dof_table[axis])


def _component_needs_reactions(canonical: str) -> bool:
    """Reaction components require ``ops.reactions()`` to be called first."""
    return canonical.startswith(("reaction_force_", "reaction_moment_"))


# =====================================================================
# Gauss-record token derivation
# =====================================================================

def _gauss_record_tokens(record: "ResolvedRecorderRecord") -> tuple[str, str] | None:
    """Return ``(catalog_token, ops_keyword)`` for a gauss-category record.

    Returns ``None`` if the record has no Gauss-level components.
    Raises ``ValueError`` if components mix work-conjugate families
    (e.g. stress + strain, or membrane_force + curvature) — those land
    under different ``ops.eleResponse`` keywords and can't share one
    call.

    Mixing within one keyword is OK: a shell record with components
    ``("membrane_force_xx", "bending_moment_yy", "transverse_shear_xz")``
    routes through a single ``ops.eleResponse(eid, "stresses")`` call.
    """
    keywords: set[str] = set()
    for comp in record.components:
        keyword = gauss_keyword_for_canonical(comp)
        if keyword is not None:
            keywords.add(keyword)
    if not keywords:
        return None
    if len(keywords) > 1:
        raise ValueError(
            f"Record {record.name!r} mixes work-conjugate families "
            f"({sorted(keywords)}); split into separate gauss records "
            f"(one per ops.eleResponse keyword)."
        )
    keyword = next(iter(keywords))
    catalog_token = catalog_token_for_keyword(keyword)
    if catalog_token is None:
        return None
    return (catalog_token, keyword)


def _class_int_rule(class_name: str) -> int | None:
    """Return the catalog's int_rule for a class (or None if ambiguous/unknown).

    v1 catalog: each class has one int_rule. When future entries add
    multiple rules per class (e.g. force-based beams with different
    IP counts), this helper will need richer context — but for the
    standard rules this scope covers, one rule per class is the truth.
    """
    rules = {rule for (cls, rule, _tok) in RESPONSE_CATALOG if cls == class_name}
    if len(rules) == 1:
        return next(iter(rules))
    return None


# =====================================================================
# DomainCapture
# =====================================================================

class DomainCapture:
    """Context manager for in-process result capture.

    Constructed by :meth:`ResolvedRecorderSpec.capture`. The user
    drives the analysis loop and calls ``step(t)`` to capture a
    snapshot.
    """

    def __init__(
        self,
        spec: "ResolvedRecorderSpec",
        path: str | Path,
        fem: "FEMData",
        *,
        ndm: int = 3,
        ndf: int = 6,
        ops: Any = None,
    ) -> None:
        self._spec = spec
        self._path = Path(path)
        self._fem = fem
        self._ndm = ndm
        self._ndf = ndf
        self._ops = ops    # injected for testing; lazy-loaded otherwise

        self._writer = None
        self._current_stage: Optional[str] = None
        self._stage_kind: Optional[str] = None
        self._buffers: list[_NodesCapturer] = []
        # Element-level capturers — gauss (Phase 11a), line_stations +
        # nodal_forces (Phase 11b), fibers (Phase 11e). Layered shells
        # remain deferred (see __enter__).
        self._gauss_capturers: list[_GaussCapturer] = []
        self._line_station_capturers: list[_LineStationCapturer] = []
        self._nodal_force_capturers: list[_NodalForcesCapturer] = []
        self._fiber_capturers: list[_FiberCapturer] = []
        self._layer_capturers: list[_LayerCapturer] = []
        # Records that fall through every supported capturer — surface
        # in step() for visibility.
        self._element_level_records: list = []
        # Whether any nodes record needs reactions per-step.
        self._needs_reactions: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> "DomainCapture":
        from ..writers._native import NativeWriter

        # Snapshot-id consistency between spec and fem is no longer
        # enforced — it's on the user to pair the right spec with the
        # right FEMData.
        self._writer = NativeWriter(self._path)
        self._writer.open(
            fem=self._fem,
            source_type="domain_capture",
            source_path="<openseespy>",
        )
        # Categorise records up-front
        for rec in self._spec.records:
            if rec.category == "nodes":
                if any(
                    _component_needs_reactions(c) for c in rec.components
                ):
                    self._needs_reactions = True
                self._buffers.append(_NodesCapturer(rec))
            elif rec.category == "modal":
                pass    # captured separately via capture_modes()
            elif rec.category == "gauss":
                self._gauss_capturers.append(_GaussCapturer(rec))
            elif rec.category == "line_stations":
                self._line_station_capturers.append(
                    _LineStationCapturer(rec),
                )
            elif rec.category == "elements":
                self._nodal_force_capturers.append(
                    _NodalForcesCapturer(rec),
                )
            elif rec.category == "fibers":
                self._fiber_capturers.append(_FiberCapturer(rec))
            elif rec.category == "layers":
                self._layer_capturers.append(_LayerCapturer(rec, self._fem))
            else:
                self._element_level_records.append(rec)
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()

    def close(self) -> None:
        """Finalise any open stage and close the writer."""
        if self._current_stage is not None:
            self.end_stage()
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    # ------------------------------------------------------------------
    # Stage lifecycle
    # ------------------------------------------------------------------

    def begin_stage(self, name: str, kind: str = "transient") -> str:
        """Open a new stage. Returns the stage_id."""
        if self._writer is None:
            raise RuntimeError(
                "DomainCapture is not open. Use as a context manager."
            )
        if self._current_stage is not None:
            raise RuntimeError(
                f"Stage {self._current_stage!r} still open — call end_stage()."
            )
        # Reset per-stage buffers
        for cap in self._buffers:
            cap.reset()
        for gc in self._gauss_capturers:
            gc.reset()
        for lc in self._line_station_capturers:
            lc.reset()
        for nfc in self._nodal_force_capturers:
            nfc.reset()
        for fc in self._fiber_capturers:
            fc.reset()
        for lc in self._layer_capturers:
            lc.reset()
        # Stage gets a placeholder time vector that we'll fill at end_stage.
        # We can't pre-create the time dataset here because we don't yet
        # know how many steps. We start the stage with an empty time
        # vector and write everything (including time) at end_stage.
        self._current_stage = name
        self._stage_kind = kind
        return name

    def step(self, t: float) -> None:
        """Capture one snapshot at simulation time ``t``."""
        if self._current_stage is None:
            raise RuntimeError("No stage open — call begin_stage() first.")
        if self._element_level_records:
            cats = sorted({r.category for r in self._element_level_records})
            raise NotImplementedError(
                f"DomainCapture does not support element-level "
                f"records of category {cats}. Supported categories: "
                f"``nodes`` (Phase 3), ``modal`` (Phase 3), ``gauss`` "
                f"(Phase 11a), ``line_stations`` (Phase 11b), "
                f"``elements`` per-element-node forces (Phase 11b). "
                f"Fibers and layers are MPCO-only (Phase 11c) — see "
                f"the module docstring."
            )
        ops = self._lazy_ops()
        if self._needs_reactions:
            # Refresh the cached reactions in the domain.
            ops.reactions()
        for cap in self._buffers:
            cap.step(t, ops)
        for gc in self._gauss_capturers:
            gc.step(t, ops)
        for lc in self._line_station_capturers:
            lc.step(t, ops)
        for nfc in self._nodal_force_capturers:
            nfc.step(t, ops)
        for fc in self._fiber_capturers:
            fc.step(t, ops)
        for lc in self._layer_capturers:
            lc.step(t, ops)

    def end_stage(self) -> None:
        """Flush buffered data for the current stage to disk.

        Multiple ``nodes`` records may target different node subsets
        (e.g. displacements on all nodes + reactions on fixed nodes
        only). The native schema has one ``_ids`` per partition, so
        we merge: take the union of node IDs across records, fill
        each component with ``NaN`` at slots the record didn't visit.
        """
        if self._current_stage is None:
            raise RuntimeError("No stage open.")
        assert self._writer is not None

        # Time vector — they must all match (same step cadence).
        time_vec = np.array([], dtype=np.float64)
        for cap in self._buffers:
            if cap._times:
                time_vec = np.array(cap._times, dtype=np.float64)
                break
        if time_vec.size == 0:
            for gc in self._gauss_capturers:
                if gc._times:
                    time_vec = np.array(gc._times, dtype=np.float64)
                    break
        if time_vec.size == 0:
            for lc in self._line_station_capturers:
                if lc._times:
                    time_vec = np.array(lc._times, dtype=np.float64)
                    break
        if time_vec.size == 0:
            for nfc in self._nodal_force_capturers:
                if nfc._times:
                    time_vec = np.array(nfc._times, dtype=np.float64)
                    break
        if time_vec.size == 0:
            for fc in self._fiber_capturers:
                if fc._times:
                    time_vec = np.array(fc._times, dtype=np.float64)
                    break
        if time_vec.size == 0:
            for lc in self._layer_capturers:
                if lc._times:
                    time_vec = np.array(lc._times, dtype=np.float64)
                    break

        sid = self._writer.begin_stage(
            name=self._current_stage,
            kind=self._stage_kind or "transient",
            time=time_vec,
        )

        try:
            self._flush_nodes_merged(sid, time_vec)
            self._flush_gauss(sid)
            self._flush_line_stations(sid)
            self._flush_nodal_forces(sid)
            self._flush_fibers(sid)
            self._flush_layers(sid)
        finally:
            # Always close the stage — even if the merge raised, we
            # want the stage closed so subsequent ``begin_stage`` works.
            self._writer.end_stage()
            self._current_stage = None
            self._stage_kind = None

        # ── Skip-summary warnings ─────────────────────────────────
        # Surface any elements the line-stations capturers had to
        # drop (typically disp-based beams whose IP coords aren't
        # introspectable from openseespy in OpenSees v3.7.x). Until
        # this warning lands the user had no signal that their
        # recorder spec was partially honoured.
        self._warn_about_skipped_line_station_elements()

    def _warn_about_skipped_line_station_elements(self) -> None:
        """Emit a single consolidated warning if any line-station
        captures dropped elements during this stage.

        ``skipped_elements`` is populated once when ``_build_groups``
        runs on the first ``step()``; group structure persists across
        stages, so the list is the cumulative skip set for the
        capturer's lifetime. We do not clear it here — tests inspect
        it post-``end_stage()`` to verify which elements were dropped,
        and clearing would also be wrong for multi-stage captures
        where the same skips apply every stage.
        """
        rows: list[tuple[int, str]] = []
        for lc in self._line_station_capturers:
            rows.extend(lc.skipped_elements)
        if not rows:
            return
        # Group by the dominant reason text so the message is short.
        by_reason: dict[str, list[int]] = {}
        for eid, reason in rows:
            by_reason.setdefault(reason, []).append(eid)
        n_total = len(rows)
        lines = [
            f"DomainCapture dropped {n_total} element(s) from line-"
            f"stations recording in this stage.",
            "If you expected line-force diagrams for these elements:",
            "  • disp-based beams: ops.eleResponse(integrationPoints) "
            "is unavailable in OpenSees v3.7.x — record via MPCO "
            "instead, or rebuild as ForceBeamColumn.",
            "  • elastic beams: should be auto-synthesised from "
            "localForce; if you see them here it's a regression.",
            "Skipped breakdown:",
        ]
        for reason, ids in by_reason.items():
            sample = ", ".join(str(e) for e in ids[:5])
            more = (
                f", ... ({len(ids) - 5} more)" if len(ids) > 5 else ""
            )
            lines.append(f"  [{len(ids)}] {reason} — eids: {sample}{more}")
        warnings.warn("\n".join(lines), stacklevel=3)

    def _flush_gauss(self, stage_id: str) -> None:
        """Write each gauss capturer's per-class buffers to disk."""
        for gc in self._gauss_capturers:
            gc.write_to(self._writer, stage_id, "partition_0")

    def _flush_line_stations(self, stage_id: str) -> None:
        """Write each line-stations capturer's per-class buffers to disk."""
        for lc in self._line_station_capturers:
            lc.write_to(self._writer, stage_id, "partition_0")

    def _flush_nodal_forces(self, stage_id: str) -> None:
        """Write each nodal-forces capturer's per-class buffers to disk."""
        for nfc in self._nodal_force_capturers:
            nfc.write_to(self._writer, stage_id, "partition_0")

    def _flush_fibers(self, stage_id: str) -> None:
        """Write each fiber capturer's per-class buffers to disk."""
        for fc in self._fiber_capturers:
            fc.write_to(self._writer, stage_id, "partition_0")

    def _flush_layers(self, stage_id: str) -> None:
        """Write each layer capturer's per-class buffers to disk."""
        for lc in self._layer_capturers:
            lc.write_to(self._writer, stage_id, "partition_0")

    def _flush_nodes_merged(
        self, stage_id: str, time_vec: ndarray,
    ) -> None:
        """Merge per-record buffers and write one ``nodes/`` slab."""
        # Collect non-empty per-record data
        per_record = []
        for cap in self._buffers:
            if not cap._times:
                continue
            comp_arrays = {
                comp: np.stack(frames, axis=0)        # (T, N_rec)
                for comp, frames in cap._values.items()
                if frames
            }
            if not comp_arrays:
                continue
            per_record.append((
                np.asarray(cap._rec.node_ids, dtype=np.int64),
                comp_arrays,
            ))

        if not per_record:
            return

        # Master node ID list = union, sorted for stability
        all_ids = np.concatenate([ids for ids, _ in per_record])
        master_ids = np.unique(all_ids)        # also returns sorted
        n_total = master_ids.size
        T = time_vec.size

        # Map node ID → master column index
        id_to_col = {int(n): i for i, n in enumerate(master_ids)}

        merged: dict[str, ndarray] = {}
        for rec_ids, comp_arrs in per_record:
            cols = np.array(
                [id_to_col[int(n)] for n in rec_ids], dtype=np.int64,
            )
            for comp, arr in comp_arrs.items():
                if comp not in merged:
                    merged[comp] = np.full((T, n_total), np.nan, dtype=np.float64)
                merged[comp][:, cols] = arr

        self._writer.write_nodes(
            stage_id, "partition_0",
            node_ids=master_ids,
            components=merged,
        )

    # ------------------------------------------------------------------
    # Modal capture (single-step stages)
    # ------------------------------------------------------------------

    def capture_modes(self, n_modes: Optional[int] = None) -> None:
        """Run ``ops.eigen()`` and write one mode-kind stage per mode.

        ``n_modes`` defaults to the maximum across all ``modal``
        records in the spec. Pass an explicit value to override.
        """
        modal_records = [
            r for r in self._spec.records if r.category == "modal"
        ]
        if n_modes is None:
            if not modal_records:
                return
            n_modes = max(r.n_modes for r in modal_records)
        if n_modes <= 0:
            return

        ops = self._lazy_ops()
        eigenvalues = ops.eigen(n_modes)
        # ``ops.eigen`` returns a list of eigenvalues (length n_modes).

        node_ids = np.asarray(self._fem.nodes.ids, dtype=np.int64)
        for mode_idx, lam in enumerate(eigenvalues, start=1):
            omega = math.sqrt(lam) if lam > 0 else 0.0
            freq_hz = omega / (2.0 * math.pi)
            period_s = (2.0 * math.pi / omega) if omega > 0 else 0.0

            sid = self._writer.begin_stage(
                name=f"mode_{mode_idx}",
                kind="mode",
                time=np.array([0.0]),
                eigenvalue=float(lam),
                frequency_hz=float(freq_hz),
                period_s=float(period_s),
                mode_index=mode_idx,
            )

            components: dict[str, ndarray] = {}
            # Translational
            axes = ("x", "y", "z")
            for axis_idx in range(min(3, self._ndm)):
                axis = axes[axis_idx]
                shape = np.array([
                    ops.nodeEigenvector(int(nid), mode_idx, axis_idx + 1)
                    for nid in node_ids
                ], dtype=np.float64)
                components[f"displacement_{axis}"] = shape[None, :]
            # Rotational (only when the model has rotational DOFs)
            if self._ndf >= 6:
                for axis_idx in range(3):
                    axis = axes[axis_idx]
                    shape = np.array([
                        ops.nodeEigenvector(int(nid), mode_idx, axis_idx + 4)
                        for nid in node_ids
                    ], dtype=np.float64)
                    components[f"rotation_{axis}"] = shape[None, :]

            self._writer.write_nodes(
                sid, "partition_0",
                node_ids=node_ids,
                components=components,
            )
            self._writer.end_stage()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _lazy_ops(self) -> Any:
        if self._ops is not None:
            return self._ops
        try:
            import openseespy.opensees as ops
        except ImportError as exc:
            raise RuntimeError(
                "openseespy is not installed. DomainCapture requires "
                "openseespy at runtime, or pass an ops= mock for testing."
            ) from exc
        return ops


# =====================================================================
# Per-record nodal capturer
# =====================================================================

class _NodesCapturer:
    """Buffers values for one ``ResolvedRecorderRecord`` (category=nodes)."""

    def __init__(self, record: "ResolvedRecorderRecord") -> None:
        self._rec = record
        self._times: list[float] = []
        # Pre-resolve canonical → (ops_fn, dof_or_None) per component.
        self._call_specs: list[tuple[str, str, Optional[int]]] = []
        for comp in record.components:
            mapping = _component_to_ops_call(comp)
            if mapping is None:
                # Unknown component — skip with a warning hook? For
                # now, omit silently; spec-level validation should
                # have caught it.
                continue
            self._call_specs.append((comp, mapping[0], mapping[1]))
        self._values: dict[str, list[ndarray]] = {
            comp: [] for comp, _, _ in self._call_specs
        }

    def reset(self) -> None:
        self._times.clear()
        for buf in self._values.values():
            buf.clear()

    def step(self, t: float, ops: Any) -> None:
        self._times.append(float(t))
        for comp, fn_name, dof in self._call_specs:
            fn = getattr(ops, fn_name)
            if dof is None:
                # Scalar — single-arg call (e.g. nodePressure(nid))
                vals = np.array(
                    [float(fn(int(nid))) for nid in self._rec.node_ids],
                    dtype=np.float64,
                )
            else:
                vals = np.array(
                    [float(fn(int(nid), dof)) for nid in self._rec.node_ids],
                    dtype=np.float64,
                )
            self._values[comp].append(vals)

    def write_to(self, writer: Any, stage_id: str, partition_id: str) -> None:
        if not self._times:
            return
        components: dict[str, ndarray] = {}
        for comp, frames in self._values.items():
            if frames:
                components[comp] = np.stack(frames, axis=0)   # (T, N)
        writer.write_nodes(
            stage_id, partition_id,
            node_ids=np.asarray(self._rec.node_ids, dtype=np.int64),
            components=components,
        )


# =====================================================================
# Per-record gauss capturer (Phase 11a)
# =====================================================================

@dataclass
class _GaussClassGroup:
    """Per-(class_name, int_rule) buffer for one gauss record."""
    layout: ResponseLayout
    element_ids: list[int] = field(default_factory=list)
    steps: list[ndarray] = field(default_factory=list)
    # When True, query the response via per-Gauss-point material
    # delegation (``ops.eleResponse(eid, "material", "<gp>", "strain")``)
    # because this element class lacks a working element-level branch
    # for the requested token. See
    # :func:`apeGmsh.solvers._element_response.needs_per_material_strain`.
    via_per_material: bool = False


class _GaussCapturer:
    """Buffers ``ops.eleResponse(eid, "stresses"/"strains")`` for one record.

    Each capturer holds one ``ResolvedRecorderRecord`` of category
    ``"gauss"``. Element classes are discovered from the live ops
    domain on the first ``step()`` call (``ops.eleType(eid)``);
    elements whose ``(class, int_rule, token)`` tuple is not
    catalogued in :mod:`apeGmsh.solvers._element_response` are skipped
    with a tracked reason.
    """

    def __init__(self, record: "ResolvedRecorderRecord") -> None:
        self._rec = record
        self._times: list[float] = []
        tokens = _gauss_record_tokens(record)
        if tokens is None:
            raise ValueError(
                f"Record {record.name!r} (category=gauss) has no "
                f"continuum stress/strain components."
            )
        self._catalog_token, self._ops_keyword = tokens
        # Built lazily on the first step (needs the live ops domain).
        self._groups: Optional[dict[tuple[str, int], _GaussClassGroup]] = None
        # (eid, reason) for any elements skipped during group-building.
        self.skipped_elements: list[tuple[int, str]] = []

    def reset(self) -> None:
        """Clear per-stage step buffers (preserves group structure)."""
        self._times.clear()
        if self._groups is not None:
            for grp in self._groups.values():
                grp.steps.clear()

    def _build_groups(self, ops: Any) -> None:
        """Group element_ids by ``(class_name, int_rule)`` via ``ops.eleType``."""
        groups: dict[tuple[str, int], _GaussClassGroup] = {}
        skipped: list[tuple[int, str]] = []
        for eid in (int(e) for e in self._rec.element_ids):
            class_name = ops.eleType(eid)
            int_rule = _class_int_rule(class_name)
            if int_rule is None:
                skipped.append((eid, f"class {class_name!r} not in catalog"))
                continue
            if not is_catalogued(class_name, int_rule, self._catalog_token):
                skipped.append((
                    eid,
                    f"({class_name}, {int_rule}, {self._catalog_token}) "
                    f"not catalogued",
                ))
                continue
            layout = lookup(class_name, int_rule, self._catalog_token)
            grp = groups.setdefault(
                (class_name, int_rule), _GaussClassGroup(
                    layout=layout,
                    via_per_material=needs_per_material_strain(
                        class_name, self._catalog_token,
                    ),
                ),
            )
            grp.element_ids.append(eid)
        self._groups = groups
        self.skipped_elements = skipped

    def step(self, t: float, ops: Any) -> None:
        if self._groups is None:
            self._build_groups(ops)
        self._times.append(float(t))
        for grp in self._groups.values():
            buf: list[ndarray] = []
            for eid in grp.element_ids:
                arr = self._query_element(ops, grp, eid)
                if arr.size != grp.layout.flat_size_per_element:
                    raise ValueError(
                        f"ops.eleResponse for element {eid} "
                        f"({'per-material' if grp.via_per_material else self._ops_keyword!r}) "
                        f"returned {arr.size} values but the catalog "
                        f"layout for {grp.layout.class_tag} expects "
                        f"{grp.layout.flat_size_per_element}."
                    )
                buf.append(arr)
            # (E_g, flat_size); stacked along time at write_to.
            grp.steps.append(np.stack(buf, axis=0))

    def _query_element(
        self, ops: Any, grp: "_GaussClassGroup", eid: int,
    ) -> ndarray:
        """Pull one element's flat response vector for the current step.

        Default path is the bulk element-level keyword
        (``ops.eleResponse(eid, "stresses"|"strains"|...)``). Classes
        flagged with ``via_per_material=True`` (currently Tri31's
        strain query) loop over Gauss points and concatenate
        ``ops.eleResponse(eid, "material", "<gp>", "<token>")`` —
        the same path MPCO takes through Tri31's
        ``"material"``/``"integrPoint"`` setResponse branch.
        """
        if grp.via_per_material:
            n_gp = grp.layout.n_gauss_points
            n_comp = grp.layout.n_components_per_gp
            flat: list[float] = []
            for gp in range(1, n_gp + 1):
                vals = ops.eleResponse(
                    eid, "material", str(gp), self._catalog_token,
                )
                if len(vals) != n_comp:
                    raise ValueError(
                        f"ops.eleResponse({eid}, 'material', '{gp}', "
                        f"{self._catalog_token!r}) returned "
                        f"{len(vals)} values; catalog expects "
                        f"{n_comp} per Gauss point."
                    )
                flat.extend(vals)
            return np.asarray(flat, dtype=np.float64)
        vals = ops.eleResponse(eid, self._ops_keyword)
        return np.asarray(vals, dtype=np.float64)

    def write_to(
        self, writer: Any, stage_id: str, partition_id: str,
    ) -> None:
        if not self._times or self._groups is None:
            return
        for i, ((class_name, _int_rule), grp) in enumerate(self._groups.items()):
            if not grp.steps:
                continue
            flat = np.stack(grp.steps, axis=0)        # (T, E_g, flat_size)
            decoded = unflatten(flat, grp.layout)
            writer.write_gauss_group(
                stage_id, partition_id,
                group_id=f"{self._rec.name}_{class_name}_{i}",
                class_tag=grp.layout.class_tag,
                int_rule=_int_rule,
                element_index=np.array(grp.element_ids, dtype=np.int64),
                natural_coords=grp.layout.natural_coords,
                components=decoded,
            )


# =====================================================================
# Per-record line-stations capturer (Phase 11b Step 2b)
# =====================================================================
#
# Force-/disp-based beam-columns expose section forces along the
# element length via ``ops.eleResponse(eid, "section", str(i),
# "force")`` for each integration point ``i``. The integration
# scheme is per-element metadata (the assigned ``beamIntegration``);
# we discover IP locations via Tier 1 ``ops.eleResponse(eid,
# "integrationPoints")`` which returns physical positions ``xi*L``
# along the beam (see ``ForceBeamColumn3d.cpp:3338–3346``).
#
# Section response codes are not exposed by openseespy. We infer them
# from the per-IP force vector's length under the assumption that
# the assigned section uses canonical aggregation order:
#
#   2D: (P, Mz)  or  (P, Mz, Vy)
#   3D: (P, Mz, My, T) [+ optional Vy, Vz]
#
# Non-canonical aggregations land mis-named columns; users with
# such sections should use MPCO recording (where META/COMPONENTS
# carries the actual code names verbatim).
#
# Disp-based beam-columns (``DispBeamColumn{2d,3d}`` and family) do
# not expose ``"integrationPoints"`` in OpenSees v3.7.x — those
# elements are silently skipped by DomainCapture v1 and recorded
# via MPCO instead.

# Section-code inference + parent-coordinate normalisation moved
# to :mod:`apeGmsh.solvers._element_response` (Step 2c). The Phase
# 11b Step 2b underscore-prefixed names above re-export the public
# helpers, so existing imports continue to work.


@dataclass
class _LineStationGroup:
    """Per-(class_name, gp_x signature, section_codes) buffer.

    ``mode`` selects the per-step recording strategy:

    * ``"section_force"`` (default) — call
      ``ops.eleResponse(eid, "section", str(i), "force")`` for each
      integration point ``i = 1..n_IP``. Used for force-based and
      disp-based fiber-section beam-columns.
    * ``"local_force_synthesis"`` — call
      ``ops.eleResponse(eid, "localForce")`` once per step and
      synthesise a 2-station slab at ξ ∈ {-1, +1} from the end-node
      force vector. Used for closed-form elastic beams that have no
      integration points but still expose a meaningful internal
      force diagram (constant axial / shear, linear moment between
      end values).
    """
    layout: ResponseLayout
    gp_x: ndarray
    element_ids: list[int] = field(default_factory=list)
    steps: list[ndarray] = field(default_factory=list)
    mode: str = "section_force"


class _LineStationCapturer:
    """Buffers ``ops.eleResponse(eid, "section", i, "force")`` per IP per step.

    Mirrors :class:`_GaussCapturer` but for force-/disp-based beam-
    columns whose IP layout is per-element (custom integration rule).
    Elements are grouped on first ``step()`` by
    ``(class_name, rounded gp_x signature, section_codes)``; each
    group ends up as one ``write_line_stations_group`` call.
    """

    _GP_X_SIGNATURE_DECIMALS = 10

    def __init__(self, record: "ResolvedRecorderRecord") -> None:
        self._rec = record
        self._times: list[float] = []
        # Built on first step; key is (class_name, gp_x_sig, section_codes).
        self._groups: Optional[
            dict[tuple[str, tuple[float, ...], tuple[int, ...]],
                 _LineStationGroup]
        ] = None
        # (eid, reason) for any element skipped during group-building.
        self.skipped_elements: list[tuple[int, str]] = []

    def reset(self) -> None:
        """Clear per-stage step buffers (preserves group structure)."""
        self._times.clear()
        if self._groups is not None:
            for grp in self._groups.values():
                grp.steps.clear()

    def _probe_element(
        self, eid: int, ops: Any,
    ) -> Optional[tuple[ResponseLayout, ndarray, tuple[int, ...], str]]:
        """Discover layout + gp_x for one element.

        Returns ``(layout, gp_x, section_codes, mode)`` where ``mode``
        is ``"section_force"`` for force-/disp-based beams and
        ``"local_force_synthesis"`` for closed-form elastic beams.
        Returns ``None`` for elements outside both catalogs.
        """
        try:
            class_name = ops.eleType(eid)
        except Exception as exc:
            self.skipped_elements.append(
                (eid, f"ops.eleType failed: {exc}"),
            )
            return None

        # ── Path A: catalogued section.force probe (force-/disp-based) ──
        if is_custom_rule_catalogued(class_name, "section_force"):
            return self._probe_section_force(eid, ops, class_name)

        # ── Path B: elastic-beam localForce synthesis (no IPs) ──────
        if is_line_station_synthesis_catalogued(class_name):
            return self._probe_local_force_synthesis(eid, ops, class_name)

        self.skipped_elements.append((
            eid,
            f"class {class_name!r} has no line-stations capture path "
            f"(not in CUSTOM_RULE_CATALOG for section.force, and not "
            f"an elastic beam in NODAL_FORCE_CATALOG)",
        ))
        return None

    def _probe_section_force(
        self, eid: int, ops: Any, class_name: str,
    ) -> Optional[tuple[ResponseLayout, ndarray, tuple[int, ...], str]]:
        """Force-/disp-based path: section.force per integration point."""
        # Tier 1: integrationPoints. Disp-based beams raise / return
        # empty here in OpenSees v3.7.x — those elements end up in
        # skipped_elements.
        try:
            xi_phys_raw = ops.eleResponse(eid, "integrationPoints")
        except Exception as exc:
            self.skipped_elements.append((
                eid,
                f"ops.eleResponse(integrationPoints) unavailable: {exc}",
            ))
            return None
        xi_phys = np.asarray(xi_phys_raw, dtype=np.float64).flatten()
        if xi_phys.size == 0:
            self.skipped_elements.append((
                eid,
                f"ops.eleResponse(integrationPoints) returned empty",
            ))
            return None

        # Element length from the end-node coordinates.
        try:
            nodes = list(ops.eleNodes(eid))
        except Exception as exc:
            self.skipped_elements.append(
                (eid, f"ops.eleNodes failed: {exc}"),
            )
            return None
        if len(nodes) < 2:
            self.skipped_elements.append((
                eid,
                f"element has {len(nodes)} nodes; expected at least 2",
            ))
            return None
        c1 = np.asarray(ops.nodeCoord(int(nodes[0])), dtype=np.float64)
        c2 = np.asarray(ops.nodeCoord(int(nodes[-1])), dtype=np.float64)
        L = float(np.linalg.norm(c2 - c1))
        try:
            xi_natural = _normalise_integration_points(xi_phys, L)
        except ValueError as exc:
            self.skipped_elements.append((eid, str(exc)))
            return None

        # Probe IP 1's section.force for component count.
        try:
            first_force = ops.eleResponse(eid, "section", "1", "force")
        except Exception as exc:
            self.skipped_elements.append(
                (eid, f"ops.eleResponse(section,1,force) failed: {exc}"),
            )
            return None
        n_comp = int(np.asarray(first_force).size)
        try:
            section_codes = _infer_section_codes(class_name, n_comp)
        except ValueError as exc:
            self.skipped_elements.append((eid, str(exc)))
            return None

        custom = lookup_custom_rule(class_name, "section_force")
        layout = resolve_layout_from_gp_x(custom, xi_natural, section_codes)
        return layout, xi_natural, section_codes, "section_force"

    def _probe_local_force_synthesis(
        self, eid: int, ops: Any, class_name: str,
    ) -> Optional[tuple[ResponseLayout, ndarray, tuple[int, ...], str]]:
        """Elastic-beam path: synthesise a 2-station slab from localForce.

        Closed-form elastic beams have no integration points; we read
        the full end-node resisting-force vector once per step and
        produce a 2-station slab at ξ ∈ {-1, +1}. Sign convention is
        applied at capture time (see :meth:`step`) so the resulting
        slab matches the section-force convention used by force-based
        beams — adjacent elements line up at shared nodes.
        """
        layout = synthesize_line_station_layout_for_elastic_beam(class_name)
        if layout is None:
            self.skipped_elements.append((
                eid,
                f"class {class_name!r} not eligible for line-stations "
                f"synthesis from localForce",
            ))
            return None

        # Verify localForce is actually responsive on this element.
        try:
            test = ops.eleResponse(eid, "localForce")
        except Exception as exc:
            self.skipped_elements.append((
                eid,
                f"ops.eleResponse(localForce) failed: {exc}",
            ))
            return None
        arr = np.asarray(test, dtype=np.float64).flatten()
        expected = 2 * layout.n_components_per_gp
        if arr.size != expected:
            self.skipped_elements.append((
                eid,
                f"ops.eleResponse(localForce) returned {arr.size} "
                f"values; expected {expected} (2 nodes × "
                f"{layout.n_components_per_gp} dofs).",
            ))
            return None

        # Stations sit at ξ ∈ {-1, +1}; section_codes are not used by
        # the synthesis path but we tag the group with a synthetic
        # value so the group key stays distinct from any future
        # section.force entry on the same class.
        gp_x = np.array([-1.0, 1.0], dtype=np.float64)
        section_codes: tuple[int, ...] = ()
        return layout, gp_x, section_codes, "local_force_synthesis"

    def _build_groups(self, ops: Any) -> None:
        groups: dict[
            tuple[str, tuple[float, ...], tuple[int, ...]],
            _LineStationGroup,
        ] = {}
        if self._rec.element_ids is None:
            self._groups = groups
            return
        for eid in (int(e) for e in self._rec.element_ids):
            probed = self._probe_element(eid, ops)
            if probed is None:
                continue
            layout, gp_x, section_codes, mode = probed
            class_name = ops.eleType(eid)
            sig = tuple(np.round(gp_x, self._GP_X_SIGNATURE_DECIMALS).tolist())
            key = (class_name, sig, section_codes)
            grp = groups.get(key)
            if grp is None:
                grp = _LineStationGroup(
                    layout=layout, gp_x=gp_x, mode=mode,
                )
                groups[key] = grp
            grp.element_ids.append(eid)
        self._groups = groups

    def step(self, t: float, ops: Any) -> None:
        if self._groups is None:
            self._build_groups(ops)
        self._times.append(float(t))
        assert self._groups is not None
        for grp in self._groups.values():
            if grp.mode == "local_force_synthesis":
                grp.steps.append(self._step_local_force(grp, ops))
            else:
                grp.steps.append(self._step_section_force(grp, ops))

    def _step_section_force(
        self, grp: _LineStationGroup, ops: Any,
    ) -> ndarray:
        """One step's flat array via per-IP ``section.force`` probes."""
        n_ip = grp.layout.n_gauss_points
        n_comp = grp.layout.n_components_per_gp
        buf: list[ndarray] = []
        for eid in grp.element_ids:
            flat = np.empty(n_ip * n_comp, dtype=np.float64)
            for i in range(1, n_ip + 1):
                vec = ops.eleResponse(eid, "section", str(i), "force")
                arr = np.asarray(vec, dtype=np.float64).flatten()
                if arr.size != n_comp:
                    raise ValueError(
                        f"ops.eleResponse({eid}, 'section', {i}, "
                        f"'force') returned {arr.size} values; "
                        f"layout for class_tag={grp.layout.class_tag} "
                        f"expects {n_comp} per IP."
                    )
                flat[(i - 1) * n_comp:i * n_comp] = arr
            buf.append(flat)
        return np.stack(buf, axis=0)    # (E_g, n_ip * n_comp)

    def _step_local_force(
        self, grp: _LineStationGroup, ops: Any,
    ) -> ndarray:
        """One step's flat array synthesised from ``localForce`` end vectors.

        The OpenSees ``localForce`` response is the joint-on-element
        resisting-force vector laid out per node ``[end_i, end_j]``.
        Internal section-force convention treats ξ=-1 (end i) as the
        section just inside node i and ξ=+1 (end j) as the section
        just inside node j; by Newton's third law the station-2
        signs flip relative to the joint-on-element value. Mirrors
        the read-side flip in ``_mpco_local_force_io`` so a slab
        captured here is component-by-component identical to one
        synthesised from MPCO's ``localForce`` bucket.
        """
        n_per = grp.layout.n_components_per_gp
        flat_size = 2 * n_per
        buf: list[ndarray] = []
        for eid in grp.element_ids:
            vec = ops.eleResponse(eid, "localForce")
            arr = np.asarray(vec, dtype=np.float64).flatten()
            if arr.size != flat_size:
                raise ValueError(
                    f"ops.eleResponse({eid}, 'localForce') returned "
                    f"{arr.size} values; synthesis layout expects "
                    f"{flat_size} (2 nodes × {n_per})."
                )
            flat = np.empty(flat_size, dtype=np.float64)
            flat[0:n_per] = arr[0:n_per]              # station 1 — keep
            flat[n_per:flat_size] = -arr[n_per:flat_size]   # station 2 — flip
            buf.append(flat)
        return np.stack(buf, axis=0)    # (E_g, 2 * n_per)

    def write_to(
        self, writer: Any, stage_id: str, partition_id: str,
    ) -> None:
        if not self._times or self._groups is None:
            return
        for i, (key, grp) in enumerate(self._groups.items()):
            if not grp.steps:
                continue
            class_name = key[0]
            flat = np.stack(grp.steps, axis=0)        # (T, E_g, flat_size)
            decoded = unflatten(flat, grp.layout)
            writer.write_line_stations_group(
                stage_id, partition_id,
                group_id=f"{self._rec.name}_{class_name}_{i}",
                class_tag=grp.layout.class_tag,
                int_rule=IntRule.Custom,
                element_index=np.array(grp.element_ids, dtype=np.int64),
                station_natural_coord=grp.gp_x,
                components=decoded,
            )


# =====================================================================
# Per-record nodal-forces capturer (Phase 11b Step 3b)
# =====================================================================
#
# Closed-form line elements (``ElasticBeam{2d,3d}``,
# ``ElasticTimoshenkoBeam{2d,3d}``, ``ModElasticBeam2d``) expose
# per-element-node force vectors via ``ops.eleResponse(eid,
# "globalForce")`` (or ``"localForce"``). No integration points,
# no section codes — the layout is fully baked in
# :data:`NODAL_FORCE_CATALOG`. The capturer reads one frame per
# record (the record's components dictate which) and writes one
# write_nodal_forces_group per element class encountered.
#
# Frame derivation: a record's components must all map to the same
# ``ops`` keyword (all global, or all local). Mixing the two in one
# record is rejected at construction.

@dataclass
class _NodalForcesGroup:
    """Per-class buffer for one nodal-forces record."""
    layout: NodalForceLayout
    element_ids: list[int] = field(default_factory=list)
    steps: list[ndarray] = field(default_factory=list)


class _NodalForcesCapturer:
    """Buffers ``ops.eleResponse(eid, "globalForce" | "localForce")`` per step.

    Mirrors :class:`_GaussCapturer` and :class:`_LineStationCapturer`
    for the nodal-forces topology. Element classes are discovered on
    first ``step()`` via ``ops.eleType(eid)``; uncatalogued classes
    are tracked in ``skipped_elements``.
    """

    def __init__(self, record: "ResolvedRecorderRecord") -> None:
        self._rec = record
        self._times: list[float] = []
        # Derive (ops_keyword, catalog_token) once from the record's
        # components. v1 requires homogeneous frame within one record.
        self._ops_keyword, self._catalog_token = self._derive_frame_routing()
        # Built lazily on first step — needs the live ops domain.
        self._groups: Optional[dict[str, _NodalForcesGroup]] = None
        self.skipped_elements: list[tuple[int, str]] = []

    def _derive_frame_routing(self) -> tuple[str, str]:
        """Resolve a single (ops_keyword, catalog_token) pair for the record.

        Each component must route through the nodal-forces topology;
        all components must share the same frame (all global, or all
        local). Mixed frames → ValueError.
        """
        keywords: set[str] = set()
        catalog_tokens: set[str] = set()
        for comp in self._rec.components:
            routing = gauss_routing_for_canonical(
                comp, topology="nodal_forces",
            )
            if routing is None:
                continue
            keyword, token = routing
            keywords.add(keyword)
            catalog_tokens.add(token)
        if not keywords:
            raise ValueError(
                f"Record {self._rec.name!r} (category=elements) has "
                f"no recognised nodal-forces components. Expected "
                f"``nodal_resisting_*`` canonicals from "
                f"PER_ELEMENT_NODAL_FORCES."
            )
        if len(keywords) > 1:
            raise ValueError(
                f"Record {self._rec.name!r} mixes global and local "
                f"frames ({sorted(keywords)}); split into separate "
                f"records (one per ops.eleResponse keyword)."
            )
        return next(iter(keywords)), next(iter(catalog_tokens))

    def reset(self) -> None:
        """Clear per-stage step buffers (preserves group structure)."""
        self._times.clear()
        if self._groups is not None:
            for grp in self._groups.values():
                grp.steps.clear()

    def _build_groups(self, ops: Any) -> None:
        groups: dict[str, _NodalForcesGroup] = {}
        skipped: list[tuple[int, str]] = []
        if self._rec.element_ids is None:
            self._groups = groups
            self.skipped_elements = skipped
            return
        for eid in (int(e) for e in self._rec.element_ids):
            try:
                class_name = ops.eleType(eid)
            except Exception as exc:
                skipped.append((eid, f"ops.eleType failed: {exc}"))
                continue
            if not is_nodal_force_catalogued(class_name, self._catalog_token):
                skipped.append((
                    eid,
                    f"({class_name}, {self._catalog_token}) not in "
                    f"NODAL_FORCE_CATALOG",
                ))
                continue
            layout = lookup_nodal_force(class_name, self._catalog_token)
            grp = groups.setdefault(
                class_name, _NodalForcesGroup(layout=layout),
            )
            grp.element_ids.append(eid)
        self._groups = groups
        self.skipped_elements = skipped

    def step(self, t: float, ops: Any) -> None:
        if self._groups is None:
            self._build_groups(ops)
        self._times.append(float(t))
        assert self._groups is not None
        for grp in self._groups.values():
            buf: list[ndarray] = []
            for eid in grp.element_ids:
                vals = ops.eleResponse(eid, self._ops_keyword)
                arr = np.asarray(vals, dtype=np.float64).flatten()
                if arr.size != grp.layout.flat_size_per_element:
                    raise ValueError(
                        f"ops.eleResponse({eid}, {self._ops_keyword!r}) "
                        f"returned {arr.size} values; layout for "
                        f"class_tag={grp.layout.class_tag} expects "
                        f"{grp.layout.flat_size_per_element} "
                        f"({grp.layout.n_nodes_per_element} nodes × "
                        f"{grp.layout.n_components_per_node} comps)."
                    )
                buf.append(arr)
            grp.steps.append(np.stack(buf, axis=0))    # (E_g, flat_size)

    def write_to(
        self, writer: Any, stage_id: str, partition_id: str,
    ) -> None:
        if not self._times or self._groups is None:
            return
        for i, (class_name, grp) in enumerate(self._groups.items()):
            if not grp.steps:
                continue
            flat = np.stack(grp.steps, axis=0)        # (T, E_g, flat_size)
            decoded = unflatten_nodal(flat, grp.layout)
            writer.write_nodal_forces_group(
                stage_id, partition_id,
                group_id=f"{self._rec.name}_{class_name}_{i}",
                class_tag=grp.layout.class_tag,
                frame=grp.layout.frame,
                element_index=np.array(grp.element_ids, dtype=np.int64),
                components=decoded,
            )


# =====================================================================
# Per-record fiber capturer (Phase 11e)
# =====================================================================
#
# Beam-column fiber sections (FiberSection2d / FiberSection3d) expose
# per-fiber geometry + state in one shot through
# ``ops.eleResponse(eid, "section", str(sec), "fiberData2")``, which
# returns a flat vector of ``[y, z, area, material_tag, stress, strain]
# × n_fibers``. We make one call per (element, section) per step:
# geometry is captured on the first step (it's constant); stress and
# strain come along for free on every step.
#
# Records request ``fiber_stress`` and/or ``fiber_strain``; both
# canonical names are read from the same call and filtered by what
# the record asked for. Indexed variants (``fiber_stress_<n>``) are
# not handled here — request the bulk canonicals and post-filter via
# the read-side composite API.
#
# Per-element classes are discovered on the first step via
# ``ops.eleType(eid)`` and grouped — uncatalogued classes (those not
# in :data:`FIBER_CATALOG`) are tracked in ``skipped_elements``.

@dataclass
class _FiberClassGroup:
    """Per-class buffer for one fiber record."""
    layout: FiberSectionLayout
    element_ids: list[int] = field(default_factory=list)
    # Filled on the first step: number of sections (IPs) per element,
    # and number of fibers per section (per element).
    n_sections_per_element: list[int] = field(default_factory=list)
    n_fibers_per_section: list[list[int]] = field(default_factory=list)
    # Flat geometry arrays — final length = sum_F (across all elements
    # × sections × fibers). Filled on the first step.
    element_index: list[int] = field(default_factory=list)
    gp_index: list[int] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    z: list[float] = field(default_factory=list)
    area: list[float] = field(default_factory=list)
    material_tag: list[int] = field(default_factory=list)
    # Per-step component buffers. Each entry is ``{component: ndarray
    # of shape (sum_F,)}`` covering the components requested by the
    # record. Stacked along time at write_to.
    steps: list[dict[str, ndarray]] = field(default_factory=list)


class _FiberCapturer:
    """Buffers ``ops.eleResponse(eid, "section", k, "fiberData2")`` per step.

    One ``fiberData2`` call per ``(element, section)`` per step
    returns ``[y, z, area, material_tag, stress, strain] × n_fibers``;
    geometry is captured on the first step, stress/strain every step,
    filtered to ``record.components``. Element classes are discovered
    on first ``step()`` via ``ops.eleType`` and grouped — uncatalogued
    classes are tracked in ``skipped_elements``.
    """

    def __init__(self, record: "ResolvedRecorderRecord") -> None:
        self._rec = record
        self._times: list[float] = []
        comps = set(record.components)
        self._want_stress = "fiber_stress" in comps
        self._want_strain = "fiber_strain" in comps
        if not (self._want_stress or self._want_strain):
            raise ValueError(
                f"Record {record.name!r} (category=fibers) has no "
                f"recognised components — expected ``fiber_stress`` "
                f"and/or ``fiber_strain``; got "
                f"{list(record.components)}."
            )
        # Built lazily on the first step (needs the live ops domain).
        self._groups: Optional[dict[str, _FiberClassGroup]] = None
        self.skipped_elements: list[tuple[int, str]] = []

    def reset(self) -> None:
        """Clear per-stage step buffers (preserves group structure)."""
        self._times.clear()
        if self._groups is not None:
            for grp in self._groups.values():
                grp.steps.clear()

    def _build_groups(self, ops: Any) -> None:
        """Group element_ids by class name (first step only)."""
        groups: dict[str, _FiberClassGroup] = {}
        skipped: list[tuple[int, str]] = []
        if self._rec.element_ids is None:
            self._groups = groups
            self.skipped_elements = skipped
            return
        for eid in (int(e) for e in self._rec.element_ids):
            try:
                class_name = ops.eleType(eid)
            except Exception as exc:
                skipped.append((eid, f"ops.eleType failed: {exc}"))
                continue
            if not is_fiber_catalogued(class_name, "fiber_stress"):
                skipped.append((
                    eid,
                    f"class {class_name!r} not in FIBER_CATALOG",
                ))
                continue
            layout = lookup_fiber(class_name, "fiber_stress")
            grp = groups.setdefault(class_name, _FiberClassGroup(layout=layout))
            grp.element_ids.append(eid)
        self._groups = groups
        self.skipped_elements = skipped

    def step(self, t: float, ops: Any) -> None:
        if self._groups is None:
            self._build_groups(ops)
        first = (len(self._times) == 0)
        self._times.append(float(t))
        assert self._groups is not None
        for grp in self._groups.values():
            if first:
                # Discover number of sections per element via
                # ``integrationPoints``. Same primitive line-stations uses.
                for eid in grp.element_ids:
                    try:
                        ip_phys = ops.eleResponse(
                            eid, "integrationPoints",
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            f"ops.eleResponse({eid}, 'integrationPoints') "
                            f"failed: {exc}"
                        ) from exc
                    n_sec = int(np.asarray(ip_phys).flatten().size)
                    if n_sec == 0:
                        raise RuntimeError(
                            f"Element {eid} has zero sections "
                            f"(integrationPoints empty)."
                        )
                    grp.n_sections_per_element.append(n_sec)
                    grp.n_fibers_per_section.append([])

            stress_buf: list[float] = []
            strain_buf: list[float] = []
            for eid_idx, eid in enumerate(grp.element_ids):
                n_sec = grp.n_sections_per_element[eid_idx]
                for sec in range(1, n_sec + 1):
                    vec = np.asarray(
                        ops.eleResponse(
                            eid, "section", str(sec), "fiberData2",
                        ),
                        dtype=np.float64,
                    ).flatten()
                    if vec.size == 0 or vec.size % 6 != 0:
                        raise RuntimeError(
                            f"ops.eleResponse({eid}, 'section', {sec}, "
                            f"'fiberData2') returned size {vec.size} at "
                            f"t={t}; expected non-zero multiple of 6 "
                            f"([y, z, area, mat, stress, strain] per fiber)."
                        )
                    n_fibers = vec.size // 6
                    if first:
                        grp.n_fibers_per_section[eid_idx].append(n_fibers)
                    else:
                        expected = grp.n_fibers_per_section[eid_idx][sec - 1]
                        if n_fibers != expected:
                            raise RuntimeError(
                                f"ops.eleResponse({eid}, 'section', "
                                f"{sec}, 'fiberData2') returned "
                                f"{n_fibers} fibers at t={t}; expected "
                                f"{expected} (changed mid-analysis)."
                            )
                    arr = vec.reshape(n_fibers, 6)
                    if first:
                        for f in range(n_fibers):
                            grp.element_index.append(eid)
                            grp.gp_index.append(sec - 1)
                            grp.y.append(float(arr[f, 0]))
                            grp.z.append(float(arr[f, 1]))
                            grp.area.append(float(arr[f, 2]))
                            grp.material_tag.append(
                                int(round(float(arr[f, 3]))),
                            )
                    if self._want_stress:
                        stress_buf.extend(arr[:, 4].tolist())
                    if self._want_strain:
                        strain_buf.extend(arr[:, 5].tolist())

            step_dict: dict[str, ndarray] = {}
            if self._want_stress:
                step_dict["fiber_stress"] = np.asarray(
                    stress_buf, dtype=np.float64,
                )
            if self._want_strain:
                step_dict["fiber_strain"] = np.asarray(
                    strain_buf, dtype=np.float64,
                )
            grp.steps.append(step_dict)

    def write_to(
        self, writer: Any, stage_id: str, partition_id: str,
    ) -> None:
        if not self._times or self._groups is None:
            return
        for i, (class_name, grp) in enumerate(self._groups.items()):
            if not grp.steps or not grp.element_index:
                continue
            comps: dict[str, ndarray] = {}
            if self._want_stress:
                comps["fiber_stress"] = np.stack(
                    [s["fiber_stress"] for s in grp.steps], axis=0,
                )
            if self._want_strain:
                comps["fiber_strain"] = np.stack(
                    [s["fiber_strain"] for s in grp.steps], axis=0,
                )
            # section_tag / section_class are not surfaced by
            # openseespy on a per-element basis; record sentinels.
            # Readers do not depend on these for slab values.
            writer.write_fibers_group(
                stage_id, partition_id,
                group_id=f"{self._rec.name}_{class_name}_{i}",
                section_tag=-1,
                section_class="(domain_capture)",
                element_index=np.asarray(
                    grp.element_index, dtype=np.int64,
                ),
                gp_index=np.asarray(grp.gp_index, dtype=np.int64),
                y=np.asarray(grp.y, dtype=np.float64),
                z=np.asarray(grp.z, dtype=np.float64),
                area=np.asarray(grp.area, dtype=np.float64),
                material_tag=np.asarray(
                    grp.material_tag, dtype=np.int64,
                ),
                components=comps,
            )


# =====================================================================
# Per-record layer capturer (Phase 11f)
# =====================================================================
#
# Layered-shell elements (LayeredShellFiberSection on ASDShellQ4 etc.)
# expose per-layer stress/strain via per-fiber probing:
#
#     ops.eleResponse(eid, "material", str(gp), "fiber",
#                     str(layer), "stress" | "strain")
#
# Per-call cost: one Python ↔ C round trip. Per element per step:
# n_surface_gp × n_layers calls. Matches what MPCO does internally.
#
# Metadata that openseespy doesn't expose:
#   - per-layer thickness        ← record.layer_section_metadata
#   - per-layer material tag     ← record.layer_section_metadata
#   - per-element local axes     ← computed from FEM node coords +
#                                  the element's class-specific
#                                  shell_local_axes / shell_quaternion
#                                  (apeGmsh.results._shell_geometry).
#
# Surface GP count per shell class:

_SHELL_GP_COUNT: dict[int, int] = {
    IntRule.Quad_GL_2: 4,        # ASDShellQ4, ShellMITC4, ShellDKGQ, ShellNLDKGQ
    IntRule.Quad_GL_3: 9,        # ShellMITC9
    IntRule.Triangle_GL_2B: 3,   # ASDShellT3
    IntRule.Triangle_GL_2C: 4,   # ShellDKGT, ShellNLDKGT
}


@dataclass
class _LayerClassGroup:
    """Per-class buffer for one layer record."""
    class_name: str
    layout: LayeredShellLayout
    n_surface_gp: int
    element_ids: list[int] = field(default_factory=list)
    section_tags_per_element: list[int] = field(default_factory=list)
    n_layers_per_element: list[int] = field(default_factory=list)
    quaternions: list[ndarray] = field(default_factory=list)    # each (4,)
    # Discovered on the very first probe. v1 enforces homogeneous N
    # across element / surface_gp / layer within one class group.
    n_components_per_layer: int = 0
    # Component canonicals to write (derived once we know N).
    stress_canonicals: tuple[str, ...] = ()
    strain_canonicals: tuple[str, ...] = ()
    # Per-step buffers: one dict per step, keyed by canonical name.
    steps: list[dict[str, ndarray]] = field(default_factory=list)


class _LayerCapturer:
    """Buffers per-layer ``ops.eleResponse`` for one ``layers`` record.

    Per (eid, surface_gp, layer): probe stress and/or strain via
    ``ops.eleResponse(eid, "material", str(gp), "fiber", str(layer),
    "stress")``. Per-layer component count N is discovered on the
    first probe; components are surfaced as ``fiber_stress`` (when
    N==1) or ``fiber_stress_0`` … ``fiber_stress_<N-1>`` (when N>1).
    Same for strain.

    Per-element local-axes quaternions are computed from element
    node coordinates via :mod:`apeGmsh.results._shell_geometry`
    on the first step. Per-layer thickness comes from the resolved
    record's :class:`LayerSectionMetadata`.
    """

    def __init__(
        self,
        record: "ResolvedRecorderRecord",
        fem: "FEMData",
    ) -> None:
        meta = record.layer_section_metadata
        if meta is None:
            raise ValueError(
                f"Record {record.name!r} (category=layers) has no "
                f"layer_section_metadata. Resolve the spec via "
                f"``g.opensees.recorders.resolve(fem)`` so the "
                f"OpenSees back-reference can populate it."
            )
        self._rec = record
        self._fem = fem
        self._meta = meta
        self._times: list[float] = []
        comps = set(record.components)
        self._want_stress = "fiber_stress" in comps
        self._want_strain = "fiber_strain" in comps
        if not (self._want_stress or self._want_strain):
            raise ValueError(
                f"Record {record.name!r} (category=layers) has no "
                f"recognised components — expected ``fiber_stress`` "
                f"and/or ``fiber_strain``; got "
                f"{list(record.components)}."
            )
        self._groups: Optional[dict[str, _LayerClassGroup]] = None
        self.skipped_elements: list[tuple[int, str]] = []

    def reset(self) -> None:
        self._times.clear()
        if self._groups is not None:
            for grp in self._groups.values():
                grp.steps.clear()

    def _build_groups(self, ops: Any) -> None:
        groups: dict[str, _LayerClassGroup] = {}
        skipped: list[tuple[int, str]] = []
        if self._rec.element_ids is None:
            self._groups = groups
            self.skipped_elements = skipped
            return
        for eid in (int(e) for e in self._rec.element_ids):
            try:
                class_name = ops.eleType(eid)
            except Exception as exc:
                skipped.append((eid, f"ops.eleType failed: {exc}"))
                continue
            if not is_layer_catalogued(class_name, "fiber_stress"):
                skipped.append((
                    eid, f"class {class_name!r} not in LAYER_CATALOG",
                ))
                continue
            if int(eid) not in self._meta.element_to_section:
                skipped.append((
                    eid,
                    "no LayeredShell section assigned in metadata",
                ))
                continue
            layout = lookup_layer(class_name, "fiber_stress")
            n_gp = _SHELL_GP_COUNT.get(layout.surface_int_rule)
            if n_gp is None:
                skipped.append((
                    eid,
                    f"surface_int_rule={layout.surface_int_rule} not "
                    f"in known shell GP table",
                ))
                continue
            grp = groups.get(class_name)
            if grp is None:
                grp = _LayerClassGroup(
                    class_name=class_name, layout=layout, n_surface_gp=n_gp,
                )
                groups[class_name] = grp
            grp.element_ids.append(eid)
            sec_tag = self._meta.element_to_section[int(eid)]
            grp.section_tags_per_element.append(sec_tag)
            grp.n_layers_per_element.append(
                self._meta.sections[sec_tag].n_layers,
            )
        self._groups = groups
        self.skipped_elements = skipped

    def _compute_quaternions_first_step(
        self, grp: _LayerClassGroup, ops: Any,
    ) -> None:
        for eid in grp.element_ids:
            node_tags = list(ops.eleNodes(int(eid)))
            coords = np.array(
                [list(ops.nodeCoord(int(n))) for n in node_tags],
                dtype=np.float64,
            )
            grp.quaternions.append(
                shell_quaternion(coords, grp.class_name),
            )

    def _discover_component_count(
        self, grp: _LayerClassGroup, ops: Any,
    ) -> None:
        """Probe the first (eid, gp=1, layer=1) cell to learn N."""
        if not grp.element_ids:
            grp.n_components_per_layer = 0
            return
        eid = grp.element_ids[0]
        # Try stress first; fall back to strain if stress isn't
        # requested. Result length = components per layer cell.
        token = "stress" if self._want_stress else "strain"
        vec = np.asarray(
            ops.eleResponse(eid, "material", "1", "fiber", "1", token),
            dtype=np.float64,
        ).flatten()
        n = int(vec.size)
        if n == 0:
            raise RuntimeError(
                f"ops.eleResponse({eid}, 'material', 1, 'fiber', 1, "
                f"{token!r}) returned empty — check that the section "
                f"assigned to element {eid} is actually a "
                f"LayeredShellFiberSection."
            )
        grp.n_components_per_layer = n
        if n == 1:
            grp.stress_canonicals = (
                ("fiber_stress",) if self._want_stress else ()
            )
            grp.strain_canonicals = (
                ("fiber_strain",) if self._want_strain else ()
            )
        else:
            grp.stress_canonicals = (
                tuple(f"fiber_stress_{k}" for k in range(n))
                if self._want_stress else ()
            )
            grp.strain_canonicals = (
                tuple(f"fiber_strain_{k}" for k in range(n))
                if self._want_strain else ()
            )

    def step(self, t: float, ops: Any) -> None:
        if self._groups is None:
            self._build_groups(ops)
        first = (len(self._times) == 0)
        self._times.append(float(t))
        assert self._groups is not None
        for grp in self._groups.values():
            if first:
                self._compute_quaternions_first_step(grp, ops)
                self._discover_component_count(grp, ops)

            n_gp = grp.n_surface_gp
            n_comp = grp.n_components_per_layer
            # Pre-allocate per-component flat arrays sized for the
            # full slab (E_g × n_gp × n_layers per element). With
            # heterogeneous n_layers within one group (rare but
            # supported), we accumulate per element.
            stress_buffers: list[list[float]] = [
                [] for _ in grp.stress_canonicals
            ]
            strain_buffers: list[list[float]] = [
                [] for _ in grp.strain_canonicals
            ]
            for eid_idx, eid in enumerate(grp.element_ids):
                n_layers = grp.n_layers_per_element[eid_idx]
                for gp in range(1, n_gp + 1):
                    for layer in range(1, n_layers + 1):
                        if self._want_stress:
                            vec = np.asarray(
                                ops.eleResponse(
                                    eid, "material", str(gp),
                                    "fiber", str(layer), "stress",
                                ),
                                dtype=np.float64,
                            ).flatten()
                            if vec.size != n_comp:
                                raise RuntimeError(
                                    f"ops.eleResponse({eid}, "
                                    f"'material', {gp}, 'fiber', "
                                    f"{layer}, 'stress') returned "
                                    f"{vec.size} components at t={t}; "
                                    f"first-step probe expected "
                                    f"{n_comp}."
                                )
                            for k in range(n_comp):
                                stress_buffers[k].append(float(vec[k]))
                        if self._want_strain:
                            vec = np.asarray(
                                ops.eleResponse(
                                    eid, "material", str(gp),
                                    "fiber", str(layer), "strain",
                                ),
                                dtype=np.float64,
                            ).flatten()
                            if vec.size != n_comp:
                                raise RuntimeError(
                                    f"ops.eleResponse({eid}, "
                                    f"'material', {gp}, 'fiber', "
                                    f"{layer}, 'strain') returned "
                                    f"{vec.size} components at t={t}; "
                                    f"first-step probe expected "
                                    f"{n_comp}."
                                )
                            for k in range(n_comp):
                                strain_buffers[k].append(float(vec[k]))

            step_dict: dict[str, ndarray] = {}
            for canonical, buf in zip(grp.stress_canonicals, stress_buffers):
                step_dict[canonical] = np.asarray(buf, dtype=np.float64)
            for canonical, buf in zip(grp.strain_canonicals, strain_buffers):
                step_dict[canonical] = np.asarray(buf, dtype=np.float64)
            grp.steps.append(step_dict)

    def write_to(
        self, writer: Any, stage_id: str, partition_id: str,
    ) -> None:
        if not self._times or self._groups is None:
            return
        for i, (class_name, grp) in enumerate(self._groups.items()):
            if not grp.steps or not grp.element_ids:
                continue
            n_gp = grp.n_surface_gp
            # Build flat per-row index arrays (length sum_L).
            element_index: list[int] = []
            gp_index: list[int] = []
            layer_index: list[int] = []
            sub_gp_index: list[int] = []
            thickness: list[float] = []
            quaternion_per_row: list[ndarray] = []
            for eid_idx, eid in enumerate(grp.element_ids):
                sec_tag = grp.section_tags_per_element[eid_idx]
                sec = self._meta.sections[sec_tag]
                quat = grp.quaternions[eid_idx]
                for gp in range(n_gp):
                    for layer in range(sec.n_layers):
                        element_index.append(eid)
                        gp_index.append(gp)
                        layer_index.append(layer)
                        sub_gp_index.append(0)    # v1: 1 sub-IP per layer
                        thickness.append(float(sec.thickness[layer]))
                        quaternion_per_row.append(quat)

            sum_L = len(element_index)
            # Stack per-component arrays across time → (T, sum_L)
            comps: dict[str, ndarray] = {}
            all_canonicals = grp.stress_canonicals + grp.strain_canonicals
            for canonical in all_canonicals:
                stacked = np.stack(
                    [s[canonical] for s in grp.steps], axis=0,
                )
                if stacked.shape[1] != sum_L:
                    raise RuntimeError(
                        f"Layer capturer mismatch on group "
                        f"{class_name}: expected {sum_L} flat rows but "
                        f"step buffer has {stacked.shape[1]}."
                    )
                comps[canonical] = stacked

            writer.write_layers_group(
                stage_id, partition_id,
                group_id=f"{self._rec.name}_{class_name}_{i}",
                element_index=np.asarray(element_index, dtype=np.int64),
                gp_index=np.asarray(gp_index, dtype=np.int64),
                layer_index=np.asarray(layer_index, dtype=np.int64),
                sub_gp_index=np.asarray(sub_gp_index, dtype=np.int64),
                thickness=np.asarray(thickness, dtype=np.float64),
                local_axes_quaternion=np.asarray(
                    quaternion_per_row, dtype=np.float64,
                ),
                components=comps,
            )
