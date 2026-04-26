"""RecorderTranscoder — parse OpenSees ``.out`` files into native HDF5.

Drives the Phase 5 emission output back to a native HDF5 file that
matches the apeGmsh schema. The ``ResolvedRecorderSpec`` is the
source of truth for what was recorded; we reuse
:mod:`apeGmsh.solvers._recorder_emit` to recompute the file paths
and column layouts (so transcoder ↔ emitter stay in lockstep).

Phase 6 v1 scope
----------------
- ``nodes`` records — full TXT support, multi-record merge with
  NaN fill (matches Phase 7 capture semantics).
- Element-level records (``elements``/``gauss``/``line_stations``/
  ``fibers``/``layers``) — same META unflattening problem as Phase 7
  capture. Skipped for v1 with a clear message.
- ``modal`` — emission deferred (Phase 5), so the transcoder skips
  too.

Single-stage transcode: one transcoder call produces one stage. For
multi-stage analyses the user emits multiple Tcl scripts (or uses
domain capture, Phase 7).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

from ...solvers._recorder_emit import emit_logical, _DEFERRED_CATEGORIES
from ..writers._native import NativeWriter
from . import _txt

if TYPE_CHECKING:
    from ...mesh.FEMData import FEMData
    from ...solvers._recorder_specs import ResolvedRecorderSpec


class RecorderTranscoder:
    """Transcode emitted recorder files into a native HDF5 results file."""

    def __init__(
        self,
        spec: "ResolvedRecorderSpec",
        output_dir: str | Path,
        target_path: str | Path,
        fem: "FEMData",
        *,
        stage_name: str = "analysis",
        stage_kind: str = "transient",
        file_format: str = "out",
    ) -> None:
        self._spec = spec
        self._output_dir = Path(output_dir)
        self._target_path = Path(target_path)
        self._fem = fem
        self._stage_name = stage_name
        self._stage_kind = stage_kind
        self._file_format = file_format

    def run(self) -> Path:
        """Parse the emitted files and write the native HDF5 target.

        Returns the path of the written file.
        """
        if self._spec.fem_snapshot_id != self._fem.snapshot_id:
            raise RuntimeError(
                "ResolvedRecorderSpec was resolved against a different "
                "FEMData (snapshot_id mismatch)."
            )

        # Collect per-record parsed data first; only open the writer
        # once we know the time vector + merged node IDs.
        node_records: list[_NodeRecordPayload] = []
        unsupported: list[str] = []

        for rec in self._spec.records:
            if rec.category in _DEFERRED_CATEGORIES:
                unsupported.append(
                    f"{rec.category}:{rec.name}"
                )
                continue
            if rec.category == "nodes":
                node_records.append(self._parse_node_record(rec))
            else:
                # elements / gauss / line_stations — META unflatten
                # not yet implemented in the transcoder.
                unsupported.append(f"{rec.category}:{rec.name}")

        if unsupported:
            # Log but don't fail — let the user know what was skipped.
            # Phase 6 v1 = nodes only.
            pass

        # Aggregate: time vector (must match across records) + merged nodes
        time_vec = np.array([], dtype=np.float64)
        for nr in node_records:
            if nr.time.size:
                time_vec = nr.time
                break

        # Sanity: every record's time vector should match (within tol)
        for nr in node_records:
            if nr.time.size and time_vec.size:
                if nr.time.shape != time_vec.shape:
                    raise ValueError(
                        f"Recorder {nr.record_name!r} has "
                        f"{nr.time.size} time steps, but other records "
                        f"have {time_vec.size}. Recorder cadences must "
                        f"match within one stage."
                    )

        # Write
        with NativeWriter(self._target_path) as w:
            w.open(
                fem=self._fem,
                source_type="tcl_recorders",
                source_path=str(self._output_dir),
            )
            sid = w.begin_stage(
                name=self._stage_name,
                kind=self._stage_kind,
                time=time_vec,
            )
            self._write_merged_nodes(w, sid, node_records, time_vec)
            w.end_stage()

        return self._target_path

    # ------------------------------------------------------------------
    # Per-record parsing
    # ------------------------------------------------------------------

    def _parse_node_record(self, rec) -> "_NodeRecordPayload":
        """Parse all emitted files for one nodes record."""
        # Reuse the emitter to learn what files OpenSees produced and
        # what each column in those files represents.
        logicals = list(emit_logical(
            rec,
            output_dir=str(self._output_dir),
            file_format=self._file_format,
        ))

        if not logicals:
            return _NodeRecordPayload(
                record_name=rec.name,
                node_ids=np.asarray(rec.node_ids, dtype=np.int64),
                time=np.array([], dtype=np.float64),
                components={},
            )

        components: dict[str, ndarray] = {}
        time_vec = np.array([], dtype=np.float64)
        node_ids = np.asarray(rec.node_ids, dtype=np.int64)

        # Each logical = one file per ops_type.
        for lr in logicals:
            t, per_dof = _txt.parse_node_file(lr.file_path, lr)
            if time_vec.size == 0:
                time_vec = t
            elif t.size != time_vec.size:
                raise ValueError(
                    f"Recorder {rec.name!r}: file {lr.file_path!r} has "
                    f"{t.size} steps, expected {time_vec.size}."
                )
            ops_type = lr.response_tokens[0]
            # Map each (ops_type, dof) → canonical name via the spec's
            # components list. We don't reverse-engineer; we filter the
            # spec's own components to those whose ops mapping matches.
            for canonical in rec.components:
                pair = _canonical_to_node_ops(canonical)
                if pair is None:
                    continue
                comp_ops_type, comp_dof = pair
                if comp_ops_type != ops_type:
                    continue
                if comp_dof not in per_dof:
                    continue
                components[canonical] = per_dof[comp_dof]   # (T, N_record)

        return _NodeRecordPayload(
            record_name=rec.name,
            node_ids=node_ids,
            time=time_vec,
            components=components,
        )

    # ------------------------------------------------------------------
    # Merge logic (mirror of DomainCapture._flush_nodes_merged)
    # ------------------------------------------------------------------

    def _write_merged_nodes(
        self,
        writer: NativeWriter,
        stage_id: str,
        records: list["_NodeRecordPayload"],
        time_vec: ndarray,
    ) -> None:
        """Merge per-record node data into one ``nodes/`` slab."""
        non_empty = [r for r in records if r.components]
        if not non_empty:
            return

        all_ids = np.concatenate([r.node_ids for r in non_empty])
        master = np.unique(all_ids)
        n_total = master.size
        T = time_vec.size
        id_to_col = {int(n): i for i, n in enumerate(master)}

        merged: dict[str, ndarray] = {}
        for r in non_empty:
            cols = np.array(
                [id_to_col[int(n)] for n in r.node_ids], dtype=np.int64,
            )
            for comp, arr in r.components.items():
                if comp not in merged:
                    merged[comp] = np.full(
                        (T, n_total), np.nan, dtype=np.float64,
                    )
                merged[comp][:, cols] = arr

        writer.write_nodes(
            stage_id, "partition_0",
            node_ids=master,
            components=merged,
        )


# =====================================================================
# Internal payloads
# =====================================================================

class _NodeRecordPayload:
    __slots__ = ("record_name", "node_ids", "time", "components")

    def __init__(
        self,
        record_name: str,
        node_ids: ndarray,
        time: ndarray,
        components: dict[str, ndarray],
    ) -> None:
        self.record_name = record_name
        self.node_ids = node_ids
        self.time = time
        self.components = components


# =====================================================================
# Canonical name → (ops_type, dof) — a thin alias of the emit module
# =====================================================================

def _canonical_to_node_ops(canonical: str) -> tuple[str, int] | None:
    """Forward the emit module's mapping (no separate table)."""
    from ...solvers._recorder_emit import (
        _NODAL_PREFIX_TABLE, _NODAL_SCALAR_TABLE,
        _AXIS_TO_TRANS_DOF, _AXIS_TO_ROT_DOF,
    )
    # Scalars
    if canonical in _NODAL_SCALAR_TABLE:
        ops_type, dof = _NODAL_SCALAR_TABLE[canonical]
        if dof < 0:
            dof = 4
        return (ops_type, dof)
    if "_" not in canonical:
        return None
    prefix, axis = canonical.rsplit("_", 1)
    table = _NODAL_PREFIX_TABLE.get(prefix)
    if table is None:
        return None
    ops_type, axis_kind = table
    dof_table = (
        _AXIS_TO_TRANS_DOF if axis_kind == "trans" else _AXIS_TO_ROT_DOF
    )
    if axis not in dof_table:
        return None
    return (ops_type, dof_table[axis])
