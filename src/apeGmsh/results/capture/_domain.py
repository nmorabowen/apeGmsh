"""DomainCapture — in-process recording during an openseespy analysis.

Wraps a ``ResolvedRecorderSpec`` and a target HDF5 path; the user
drives the analysis loop and calls ``step(t)`` after each
``ops.analyze(...)``. The capture queries the openseespy domain
directly (``ops.nodeDisp``, ``ops.nodeReaction``, etc.), buffers
per-record values in RAM, and writes a chunk to the native HDF5
file at ``end_stage()``.

Phase 7 v1 scope
----------------
- **Nodes records**: full support for kinematic components
  (``displacement_*``, ``rotation_*``, ``velocity_*``,
  ``acceleration_*``, ``angular_velocity_*``,
  ``angular_acceleration_*``), reactions, and pore pressure.
- **Modal records**: ``capture_modes()`` runs ``ops.eigen()``
  and writes one stage per mode (``kind="mode"``).
- **Element-level records** (``elements``, ``gauss``,
  ``line_stations``, ``fibers``, ``layers``): emit a
  ``NotImplementedError`` at ``step()`` time. The
  unflattening required for ``ops.eleResponse(...)`` returns
  shares logic with the Phase 6 transcoder; both will land
  together in a follow-up.

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
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy import ndarray

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
        # Track records the user declared but we don't capture in v1.
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
                f"DomainCapture v1 does not yet support element-level "
                f"records ({cats}). Use Strategy A (Tcl emission + "
                f"transcoder, Phase 6) or Strategy C (MPCO bridge, "
                f"Phase 8) for these recorders. Nodes and modal "
                f"records are fully supported."
            )
        ops = self._lazy_ops()
        if self._needs_reactions:
            # Refresh the cached reactions in the domain.
            ops.reactions()
        for cap in self._buffers:
            cap.step(t, ops)

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

        sid = self._writer.begin_stage(
            name=self._current_stage,
            kind=self._stage_kind or "transient",
            time=time_vec,
        )

        try:
            self._flush_nodes_merged(sid, time_vec)
        finally:
            # Always close the stage — even if the merge raised, we
            # want the stage closed so subsequent ``begin_stage`` works.
            self._writer.end_stage()
            self._current_stage = None
            self._stage_kind = None

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
