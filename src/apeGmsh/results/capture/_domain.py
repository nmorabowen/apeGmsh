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
- **Other element-level records** (``elements`` /
  ``line_stations`` / ``fibers`` / ``layers``): still raise
  ``NotImplementedError`` at ``step()`` time — their catalog
  entries land in later phases.

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
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy import ndarray

from ...solvers._element_response import (
    CUSTOM_RULE_CATALOG,
    INFERRED_SECTION_CODES_TABLE,
    NODAL_FORCE_CATALOG,
    RESPONSE_CATALOG,
    CustomRuleLayout,
    IntRule,
    NodalForceLayout,
    ResponseLayout,
    catalog_token_for_keyword,
    class_dimension,
    gauss_keyword_for_canonical,
    gauss_routing_for_canonical,
    infer_section_codes,
    is_catalogued,
    is_custom_rule_catalogued,
    is_nodal_force_catalogued,
    lookup,
    lookup_custom_rule,
    lookup_nodal_force,
    normalise_integration_points,
    resolve_layout_from_gp_x,
    unflatten,
    unflatten_nodal,
)

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
        # Element-level capturers — gauss (Phase 11a) + line_stations
        # (Phase 11b Step 2b). fibers / layers / per-element-node forces
        # still raise NotImplementedError below.
        self._gauss_capturers: list[_GaussCapturer] = []
        self._line_station_capturers: list[_LineStationCapturer] = []
        self._nodal_force_capturers: list[_NodalForcesCapturer] = []
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

        # Validate spec ↔ fem hash match (catches user error early).
        if self._spec.fem_snapshot_id != self._fem.snapshot_id:
            raise RuntimeError(
                "ResolvedRecorderSpec was resolved against a different "
                "FEMData (snapshot_id mismatch). Re-resolve the spec "
                "with the correct fem before capturing."
            )

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
                f"DomainCapture does not yet support element-level "
                f"records of category {cats}. Gauss-level continuum "
                f"stress/strain (Phase 11a), line-stations beam "
                f"section forces (Phase 11b Step 2b), and per-element-"
                f"node forces (Phase 11b Step 3b) are supported; "
                f"fibers / layers still need their catalog entries."
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
        finally:
            # Always close the stage — even if the merge raised, we
            # want the stage closed so subsequent ``begin_stage`` works.
            self._writer.end_stage()
            self._current_stage = None
            self._stage_kind = None

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
                (class_name, int_rule), _GaussClassGroup(layout=layout),
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
                vals = ops.eleResponse(eid, self._ops_keyword)
                arr = np.asarray(vals, dtype=np.float64)
                if arr.size != grp.layout.flat_size_per_element:
                    raise ValueError(
                        f"ops.eleResponse({eid}, {self._ops_keyword!r}) "
                        f"returned {arr.size} values but the catalog "
                        f"layout for {grp.layout.class_tag} expects "
                        f"{grp.layout.flat_size_per_element}."
                    )
                buf.append(arr)
            # (E_g, flat_size); stacked along time at write_to.
            grp.steps.append(np.stack(buf, axis=0))

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
    """Per-(class_name, gp_x signature, section_codes) buffer."""
    layout: ResponseLayout
    gp_x: ndarray
    element_ids: list[int] = field(default_factory=list)
    steps: list[ndarray] = field(default_factory=list)


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
    ) -> Optional[tuple[ResponseLayout, ndarray, tuple[int, ...]]]:
        """Discover layout + gp_x for one element. ``None`` ⇒ skip."""
        try:
            class_name = ops.eleType(eid)
        except Exception as exc:
            self.skipped_elements.append(
                (eid, f"ops.eleType failed: {exc}"),
            )
            return None
        if not is_custom_rule_catalogued(class_name, "section_force"):
            self.skipped_elements.append((
                eid,
                f"class {class_name!r} not in CUSTOM_RULE_CATALOG",
            ))
            return None

        # Tier 1: integrationPoints. Disp-based beams raise / return
        # empty here — we silently skip them.
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
        return layout, xi_natural, section_codes

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
            layout, gp_x, section_codes = probed
            class_name = ops.eleType(eid)
            sig = tuple(np.round(gp_x, self._GP_X_SIGNATURE_DECIMALS).tolist())
            key = (class_name, sig, section_codes)
            grp = groups.get(key)
            if grp is None:
                grp = _LineStationGroup(layout=layout, gp_x=gp_x)
                groups[key] = grp
            grp.element_ids.append(eid)
        self._groups = groups

    def step(self, t: float, ops: Any) -> None:
        if self._groups is None:
            self._build_groups(ops)
        self._times.append(float(t))
        assert self._groups is not None
        for grp in self._groups.values():
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
            grp.steps.append(np.stack(buf, axis=0))    # (E_g, flat_size)

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
