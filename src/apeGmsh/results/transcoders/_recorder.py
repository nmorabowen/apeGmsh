"""RecorderTranscoder — parse OpenSees ``.out`` files into native HDF5.

Drives the Phase 5 emission output back to a native HDF5 file that
matches the apeGmsh schema. The ``ResolvedRecorderSpec`` is the
source of truth for what was recorded; we reuse
:mod:`apeGmsh.solvers._recorder_emit` to recompute the file paths
and column layouts (so transcoder ↔ emitter stay in lockstep).

Scope
-----
- ``nodes`` records — full TXT support, multi-record merge with
  NaN fill (matches DomainCapture semantics).
- ``gauss`` records (Phase 11a) — continuum stress/strain at GPs,
  for catalogued element classes. Class identity is sniffed from the
  file's column count via the response catalog; v1 requires
  homogeneous-class records (one element class per record).
- ``elements`` (per-element-node forces), ``line_stations``,
  ``fibers``, ``layers`` — still skipped; their catalog entries land
  in later phases.
- ``modal`` — emission deferred, so the transcoder skips too.

Single-stage transcode: one transcoder call produces one stage. For
multi-stage analyses the user emits multiple Tcl scripts (or uses
domain capture).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

from ...solvers._element_response import (
    RESPONSE_CATALOG,
    CustomRuleLayout,
    IntRule,
    NodalForceLayout,
    ResponseLayout,
    catalog_token_for_keyword,
    gauss_keyword_for_canonical,
    gauss_routing_for_canonical,
    infer_section_codes,
    is_custom_rule_catalogued,
    is_nodal_force_catalogued,
    lookup_custom_rule,
    lookup_nodal_force,
    normalise_integration_points,
    resolve_layout_from_gp_x,
    unflatten,
    unflatten_nodal,
)
from ...solvers._recorder_emit import (
    _DEFERRED_CATEGORIES,
    emit_logical,
    line_station_gpx_path,
)
from ..writers._native import NativeWriter
from . import _txt

if TYPE_CHECKING:
    from ...mesh.FEMData import FEMData
    from ...solvers._recorder_specs import (
        ResolvedRecorderRecord,
        ResolvedRecorderSpec,
    )


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
        # Records the run() skipped — populated post-run for inspection.
        self.unsupported: list[str] = []

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
        gauss_records: list[_GaussRecordPayload] = []
        line_records: list[_LineStationRecordPayload] = []
        nodal_records: list[_NodalForceRecordPayload] = []
        unsupported: list[str] = []

        for rec in self._spec.records:
            if rec.category in _DEFERRED_CATEGORIES:
                unsupported.append(f"{rec.category}:{rec.name}")
                continue
            if rec.category == "nodes":
                node_records.append(self._parse_node_record(rec))
            elif rec.category == "gauss":
                payload = self._parse_gauss_record(rec)
                if payload is not None:
                    gauss_records.append(payload)
            elif rec.category == "line_stations":
                payload = self._parse_line_station_record(rec)
                if payload is not None:
                    line_records.append(payload)
            elif rec.category == "elements":
                payload = self._parse_nodal_force_record(rec)
                if payload is not None:
                    nodal_records.append(payload)
            else:
                # fibers / layers — not yet wired.
                unsupported.append(f"{rec.category}:{rec.name}")

        if unsupported:
            # Caller can detect skipped records via ``self.unsupported``
            # if needed; we don't fail the run for them.
            self.unsupported = list(unsupported)

        # Aggregate: time vector (must match across records).
        time_vec = np.array([], dtype=np.float64)
        for nr in node_records:
            if nr.time.size:
                time_vec = nr.time
                break
        if time_vec.size == 0:
            for gr in gauss_records:
                if gr.time.size:
                    time_vec = gr.time
                    break

        # Sanity: every record's time vector should match.
        for nr in node_records:
            if nr.time.size and time_vec.size:
                if nr.time.shape != time_vec.shape:
                    raise ValueError(
                        f"Recorder {nr.record_name!r} has "
                        f"{nr.time.size} time steps, but other records "
                        f"have {time_vec.size}. Recorder cadences must "
                        f"match within one stage."
                    )
        for gr in gauss_records:
            if gr.time.size and time_vec.size:
                if gr.time.shape != time_vec.shape:
                    raise ValueError(
                        f"Recorder {gr.record_name!r} has "
                        f"{gr.time.size} time steps, but other records "
                        f"have {time_vec.size}. Recorder cadences must "
                        f"match within one stage."
                    )
        for lr in line_records:
            if lr.time.size and time_vec.size:
                if lr.time.shape != time_vec.shape:
                    raise ValueError(
                        f"Recorder {lr.record_name!r} has "
                        f"{lr.time.size} time steps, but other records "
                        f"have {time_vec.size}. Recorder cadences must "
                        f"match within one stage."
                    )
        for nr in nodal_records:
            if nr.time.size and time_vec.size:
                if nr.time.shape != time_vec.shape:
                    raise ValueError(
                        f"Recorder {nr.record_name!r} has "
                        f"{nr.time.size} time steps, but other records "
                        f"have {time_vec.size}. Recorder cadences must "
                        f"match within one stage."
                    )
        if time_vec.size == 0:
            for lr in line_records:
                if lr.time.size:
                    time_vec = lr.time
                    break
        if time_vec.size == 0:
            for nr in nodal_records:
                if nr.time.size:
                    time_vec = nr.time
                    break

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
            self._write_gauss_records(w, sid, gauss_records)
            self._write_line_station_records(w, sid, line_records)
            self._write_nodal_force_records(w, sid, nodal_records)
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
    # Per-record parsing — gauss
    # ------------------------------------------------------------------

    def _parse_gauss_record(
        self, rec: "ResolvedRecorderRecord",
    ) -> "_GaussRecordPayload | None":
        """Parse a gauss record's ``.out`` file into ``(T, E, flat_size)``.

        Sniffs the catalog layout from the file's column count: every
        catalogued ``(class, rule, token)`` has a unique
        ``flat_size_per_element`` per token, so a homogeneous-class
        record's column count uniquely identifies the layout.
        """
        logicals = list(emit_logical(
            rec, output_dir=str(self._output_dir),
            file_format=self._file_format,
        ))
        if not logicals:
            return None
        if len(logicals) > 1:
            raise ValueError(
                f"Gauss record {rec.name!r} unexpectedly emitted "
                f"{len(logicals)} logical recorders; v1 transcoder "
                f"assumes one .out file per gauss record."
            )
        lr = logicals[0]
        catalog_token = _record_catalog_token(rec)
        if catalog_token is None:
            return None

        n_elements = len(lr.target_ids)
        # Peek at the first row to learn the column count without
        # paying for the full body.
        first_row = np.loadtxt(lr.file_path, dtype=np.float64, max_rows=1)
        if first_row.ndim == 0:
            raise ValueError(
                f"{lr.file_path}: file is empty or 1-D scalar."
            )
        total_data_cols = first_row.size - 1   # subtract the time column
        if total_data_cols % n_elements != 0:
            raise ValueError(
                f"{lr.file_path}: {total_data_cols} data columns is not "
                f"divisible by {n_elements} elements. v1 transcoder "
                f"requires homogeneous-class records — split this "
                f"record by element class."
            )
        flat_size = total_data_cols // n_elements

        layout, class_name, int_rule = _identify_layout(
            catalog_token, flat_size,
            class_hint=rec.element_class_name,
        )

        time, flat = _txt.parse_element_file(
            lr.file_path, lr, flat_size,
        )

        return _GaussRecordPayload(
            record_name=rec.name,
            element_ids=np.asarray(rec.element_ids, dtype=np.int64),
            class_name=class_name,
            int_rule=int_rule,
            layout=layout,
            time=time,
            flat=flat,
        )

    def _write_gauss_records(
        self,
        writer: NativeWriter,
        stage_id: str,
        records: list["_GaussRecordPayload"],
    ) -> None:
        """Write each gauss payload as a per-record gauss group."""
        for i, gr in enumerate(records):
            decoded = unflatten(gr.flat, gr.layout)
            writer.write_gauss_group(
                stage_id, "partition_0",
                group_id=f"{gr.record_name}_{gr.class_name}_{i}",
                class_tag=gr.layout.class_tag,
                int_rule=gr.int_rule,
                element_index=gr.element_ids,
                natural_coords=gr.layout.natural_coords,
                components=decoded,
            )

    # ------------------------------------------------------------------
    # Per-record parsing — line_stations (Phase 11b Step 2c.2)
    # ------------------------------------------------------------------

    def _parse_line_station_record(
        self, rec: "ResolvedRecorderRecord",
    ) -> "_LineStationRecordPayload | None":
        """Parse a paired ``.out`` + ``_gpx.out`` set into per-group payloads.

        The line-stations record emits two LogicalRecorders (Step
        2c.1): the section.force file and the integrationPoints
        ``_gpx.out`` companion. We read both to reconstruct what
        MPCO carries in META + GP_X attribute:

        1. gpx file → first row of physical IP positions ``xi*L``
           per element (IPs are static, so only step 0 is needed).
        2. fem element-end coords → element length ``L`` per eid.
        3. ``xi*L → [-1, +1]`` natural coords per element.
        4. section.force file → ``(T, E, n_IP * n_components)`` flat,
           with ``n_components`` derived from the column count.
        5. ``infer_section_codes(class_name, n_components)`` → codes.
        6. Group elements by ``(gp_x signature, section_codes)`` so
           heterogeneous beam-integration assignments produce
           separate write_line_stations_group calls.
        """
        if rec.element_class_name is None:
            raise ValueError(
                f"Line-stations record {rec.name!r} requires "
                f"``element_class_name`` on the spec — the .out "
                f"transcoder cannot infer the element class from the "
                f"file alone (no META). Add the class hint via the "
                f"recorder builder or pass it through "
                f"ResolvedRecorderRecord.element_class_name."
            )
        class_name = rec.element_class_name
        if not is_custom_rule_catalogued(class_name, "section_force"):
            raise ValueError(
                f"Class {class_name!r} is not in CUSTOM_RULE_CATALOG; "
                f"add an entry in apeGmsh.solvers._element_response if "
                f"this beam-column class should be supported."
            )

        logicals = list(emit_logical(
            rec, output_dir=str(self._output_dir),
            file_format=self._file_format,
        ))
        if len(logicals) != 2:
            raise ValueError(
                f"Line-stations record {rec.name!r} expected to emit "
                f"2 logical recorders (section.force + "
                f"integrationPoints), got {len(logicals)}."
            )
        section_lr, gpx_lr = logicals[0], logicals[1]
        if "integrationPoints" not in gpx_lr.response_tokens:
            raise ValueError(
                f"Line-stations record {rec.name!r}: second logical "
                f"recorder is not the integrationPoints pair "
                f"(tokens={gpx_lr.response_tokens})."
            )

        # ── Read the gpx file's first row (IPs are static) ─────────
        first_gpx_row = np.loadtxt(
            gpx_lr.file_path, dtype=np.float64, max_rows=1,
        )
        if first_gpx_row.ndim == 0 or first_gpx_row.size < 2:
            raise ValueError(
                f"{gpx_lr.file_path}: integrationPoints file is empty "
                f"or has too few columns."
            )
        n_elements = len(rec.element_ids)
        gpx_data_cols = first_gpx_row.size - 1   # subtract -time column
        if gpx_data_cols % n_elements != 0:
            raise ValueError(
                f"{gpx_lr.file_path}: {gpx_data_cols} data columns is "
                f"not divisible by {n_elements} elements. The integration"
                f"Points file should be element-major with n_IP "
                f"columns per element."
            )
        n_ip = gpx_data_cols // n_elements
        # Reshape to (E, n_IP) of physical xi*L.
        xi_phys = first_gpx_row[1:].reshape(n_elements, n_ip)

        # ── Element lengths from FEM (one length per element) ──────
        element_lengths = _resolve_element_lengths(
            self._fem, rec.element_ids,
        )

        # ── Per-element natural coords ──────────────────────────────
        xi_natural = np.empty_like(xi_phys)
        for i, L in enumerate(element_lengths):
            xi_natural[i] = normalise_integration_points(xi_phys[i], L)

        # ── Read the section.force file: total cols = 1 + E*n_IP*K ──
        first_section_row = np.loadtxt(
            section_lr.file_path, dtype=np.float64, max_rows=1,
        )
        if first_section_row.ndim == 0 or first_section_row.size < 2:
            raise ValueError(
                f"{section_lr.file_path}: section.force file is empty."
            )
        section_data_cols = first_section_row.size - 1
        per_element_cols = section_data_cols // n_elements
        if section_data_cols % n_elements != 0:
            raise ValueError(
                f"{section_lr.file_path}: {section_data_cols} data "
                f"columns is not divisible by {n_elements} elements."
            )
        if per_element_cols % n_ip != 0:
            raise ValueError(
                f"{section_lr.file_path}: {per_element_cols} columns "
                f"per element is not divisible by {n_ip} IPs (from the "
                f"paired integrationPoints file). Check that both "
                f"files were emitted from the same recorder spec."
            )
        n_components = per_element_cols // n_ip
        flat_size_per_element = n_ip * n_components

        section_codes = infer_section_codes(class_name, n_components)
        custom = lookup_custom_rule(class_name, "section_force")

        time, flat = _txt.parse_element_file(
            section_lr.file_path, section_lr, flat_size_per_element,
        )
        # flat shape: (T, E, n_IP * n_components) — element-major.

        # ── Group elements by (gp_x signature, section_codes) ─────
        # Custom-rule beams sharing the same beamIntegration produce
        # the same xi vector; differing beamIntegrations land in
        # different groups (one write_line_stations_group call each).
        groups: dict[
            tuple[tuple[float, ...], tuple[int, ...]],
            list[int],
        ] = {}
        for i in range(n_elements):
            sig = tuple(np.round(xi_natural[i], 10).tolist())
            key = (sig, section_codes)
            groups.setdefault(key, []).append(i)

        payloads: list[_LineStationGroupPayload] = []
        for (sig, codes), idxs in groups.items():
            row_idxs = np.array(idxs, dtype=np.int64)
            eids = np.asarray(
                [rec.element_ids[i] for i in idxs], dtype=np.int64,
            )
            gp_x = xi_natural[idxs[0]]
            layout = resolve_layout_from_gp_x(custom, gp_x, codes)
            group_flat = flat[:, row_idxs, :]
            payloads.append(_LineStationGroupPayload(
                class_name=class_name,
                element_ids=eids,
                gp_x=gp_x,
                layout=layout,
                flat=group_flat,
            ))

        return _LineStationRecordPayload(
            record_name=rec.name,
            time=time,
            groups=payloads,
        )

    def _write_line_station_records(
        self,
        writer: NativeWriter,
        stage_id: str,
        records: list["_LineStationRecordPayload"],
    ) -> None:
        """Write each line-stations payload as one group per (xi, codes)."""
        for record in records:
            for i, gp in enumerate(record.groups):
                decoded = unflatten(gp.flat, gp.layout)
                writer.write_line_stations_group(
                    stage_id, "partition_0",
                    group_id=(
                        f"{record.record_name}_{gp.class_name}_{i}"
                    ),
                    class_tag=gp.layout.class_tag,
                    int_rule=IntRule.Custom,
                    element_index=gp.element_ids,
                    station_natural_coord=gp.gp_x,
                    components=decoded,
                )

    # ------------------------------------------------------------------
    # Per-record parsing — nodal_forces (Phase 11b Step 3c.2)
    # ------------------------------------------------------------------

    def _parse_nodal_force_record(
        self, rec: "ResolvedRecorderRecord",
    ) -> "_NodalForceRecordPayload | None":
        """Parse one ``elements`` record's ``.out`` file.

        Closed-form line elements have a fully-baked layout in
        :data:`NODAL_FORCE_CATALOG` — no IP positions to recover, no
        section codes to infer. The transcoder needs only:

        1. ``rec.element_class_name`` — the .out file carries no
           class info, so the spec hint is required.
        2. The frame (``"global"`` or ``"local"``), derived from the
           record's components like the emit side does. Drives the
           ``catalog_token`` lookup (``"global_force"`` /
           ``"local_force"``).
        3. The .out file emitted by ``_emit_element_simple`` (one
           file per record — no pairing needed).

        Builds one :class:`_NodalForceRecordPayload` (one bucket per
        class — but v1 transcoder requires homogeneous-class records
        like the gauss path does).
        """
        if rec.element_class_name is None:
            raise ValueError(
                f"Elements record {rec.name!r} requires "
                f"``element_class_name`` on the spec — the .out "
                f"transcoder cannot infer the element class from "
                f"the file alone (no META). Add the class hint via "
                f"the recorder builder or pass it through "
                f"ResolvedRecorderRecord.element_class_name."
            )
        class_name = rec.element_class_name

        # Derive the frame (global / local) from the record's
        # components — same logic as _nodal_record_ops_keyword in
        # the emit module.
        frame_keywords: set[str] = set()
        catalog_tokens: set[str] = set()
        for comp in rec.components:
            routing = gauss_routing_for_canonical(
                comp, topology="nodal_forces",
            )
            if routing is None:
                continue
            kw, tok = routing
            frame_keywords.add(kw)
            catalog_tokens.add(tok)
        if not frame_keywords:
            raise ValueError(
                f"Elements record {rec.name!r} has no recognised "
                f"nodal-forces components — expected "
                f"``nodal_resisting_*`` canonicals."
            )
        if len(frame_keywords) > 1:
            raise ValueError(
                f"Elements record {rec.name!r} mixes global and "
                f"local frames ({sorted(frame_keywords)}); split "
                f"into separate records."
            )
        catalog_token = next(iter(catalog_tokens))

        if not is_nodal_force_catalogued(class_name, catalog_token):
            raise ValueError(
                f"({class_name!r}, {catalog_token!r}) is not in "
                f"NODAL_FORCE_CATALOG; add an entry in "
                f"apeGmsh.solvers._element_response if this class "
                f"+ frame should be supported."
            )
        layout = lookup_nodal_force(class_name, catalog_token)

        logicals = list(emit_logical(
            rec, output_dir=str(self._output_dir),
            file_format=self._file_format,
        ))
        if len(logicals) != 1:
            raise ValueError(
                f"Elements record {rec.name!r}: expected 1 logical "
                f"recorder (no pairing for nodal_forces), got "
                f"{len(logicals)}."
            )
        lr = logicals[0]

        time, flat = _txt.parse_element_file(
            lr.file_path, lr, layout.flat_size_per_element,
        )
        # flat shape: (T, E, n_nodes * n_components_per_node).

        return _NodalForceRecordPayload(
            record_name=rec.name,
            element_ids=np.asarray(rec.element_ids, dtype=np.int64),
            class_name=class_name,
            layout=layout,
            time=time,
            flat=flat,
        )

    def _write_nodal_force_records(
        self,
        writer: NativeWriter,
        stage_id: str,
        records: list["_NodalForceRecordPayload"],
    ) -> None:
        """Write each nodal-forces payload as one nodal_forces group."""
        for i, nr in enumerate(records):
            decoded = unflatten_nodal(nr.flat, nr.layout)
            writer.write_nodal_forces_group(
                stage_id, "partition_0",
                group_id=f"{nr.record_name}_{nr.class_name}_{i}",
                class_tag=nr.layout.class_tag,
                frame=nr.layout.frame,
                element_index=nr.element_ids,
                components=decoded,
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


class _GaussRecordPayload:
    __slots__ = (
        "record_name", "element_ids", "class_name", "int_rule",
        "layout", "time", "flat",
    )

    def __init__(
        self,
        record_name: str,
        element_ids: ndarray,
        class_name: str,
        int_rule: int,
        layout: ResponseLayout,
        time: ndarray,
        flat: ndarray,
    ) -> None:
        self.record_name = record_name
        self.element_ids = element_ids
        self.class_name = class_name
        self.int_rule = int_rule
        self.layout = layout
        self.time = time
        self.flat = flat


class _LineStationGroupPayload:
    """One (gp_x signature, section_codes) bucket within a record."""
    __slots__ = ("class_name", "element_ids", "gp_x", "layout", "flat")

    def __init__(
        self,
        class_name: str,
        element_ids: ndarray,
        gp_x: ndarray,
        layout: ResponseLayout,
        flat: ndarray,
    ) -> None:
        self.class_name = class_name
        self.element_ids = element_ids
        self.gp_x = gp_x
        self.layout = layout
        self.flat = flat


class _LineStationRecordPayload:
    __slots__ = ("record_name", "time", "groups")

    def __init__(
        self,
        record_name: str,
        time: ndarray,
        groups: list[_LineStationGroupPayload],
    ) -> None:
        self.record_name = record_name
        self.time = time
        self.groups = groups


class _NodalForceRecordPayload:
    """One ``elements`` record's ``.out`` parse output."""
    __slots__ = (
        "record_name", "element_ids", "class_name", "layout",
        "time", "flat",
    )

    def __init__(
        self,
        record_name: str,
        element_ids: ndarray,
        class_name: str,
        layout: NodalForceLayout,
        time: ndarray,
        flat: ndarray,
    ) -> None:
        self.record_name = record_name
        self.element_ids = element_ids
        self.class_name = class_name
        self.layout = layout
        self.time = time
        self.flat = flat


# =====================================================================
# Element-length resolver (line-stations transcoder helper)
# =====================================================================

def _resolve_element_lengths(fem, element_ids) -> ndarray:
    """Return per-element lengths for line-stations transcoding.

    Tries a duck-typed test-fixture API first
    (``fem.element_end_coords(eid) -> (c1, c2)``), then falls back to
    the real :class:`apeGmsh.mesh.FEMData.FEMData` path
    (``fem.elements.get(tag=eid).resolve()`` + ``fem.nodes.get_coords()``).

    The duck-typed shortcut keeps test fixtures simple — they don't
    need a fully-built ElementComposite when the only thing the
    transcoder needs is per-element length.
    """
    eids = np.asarray(element_ids, dtype=np.int64)
    out = np.empty(eids.size, dtype=np.float64)

    if hasattr(fem, "element_end_coords"):
        for i, eid in enumerate(eids):
            c1, c2 = fem.element_end_coords(int(eid))
            out[i] = float(np.linalg.norm(
                np.asarray(c2, dtype=np.float64)
                - np.asarray(c1, dtype=np.float64),
            ))
        return out

    for i, eid in enumerate(eids):
        ids, conn = fem.elements.get(tag=int(eid)).resolve()
        if conn.size == 0:
            raise ValueError(
                f"Element {eid} has no connectivity in the bound FEMData."
            )
        # ``conn`` is (n_elements_resolved, n_nodes_per_element). For a
        # tag lookup we expect 1 row; take its first and last node.
        node_a = int(conn.flat[0])
        node_b = int(conn.flat[-1])
        c1 = np.asarray(fem.nodes.get_coords(node_a), dtype=np.float64)
        c2 = np.asarray(fem.nodes.get_coords(node_b), dtype=np.float64)
        out[i] = float(np.linalg.norm(c2 - c1))
    return out


# =====================================================================
# Catalog token derivation + layout sniff
# =====================================================================

def _record_catalog_token(rec: "ResolvedRecorderRecord") -> str | None:
    """Return ``"stress"`` / ``"strain"`` for a gauss record (or None).

    Mixed work-conjugate families raise — same contract as
    DomainCapture and the spec emit. Records with no Gauss-level
    components return ``None`` (the caller should skip).

    Recognises both continuum prefixes (``stress`` / ``strain``) and
    shell-resultant prefixes (``membrane_force`` / ``bending_moment``
    / ``transverse_shear`` / ``membrane_strain`` / ``curvature`` /
    ``transverse_shear_strain``) — they share catalog tokens with
    their conjugate continuum families.
    """
    keywords: set[str] = set()
    for comp in rec.components:
        keyword = gauss_keyword_for_canonical(comp)
        if keyword is not None:
            keywords.add(keyword)
    if not keywords:
        return None
    if len(keywords) > 1:
        raise ValueError(
            f"Record {rec.name!r} mixes work-conjugate families "
            f"({sorted(keywords)}); split into separate gauss records "
            f"(one per ops.eleResponse keyword)."
        )
    keyword = next(iter(keywords))
    return catalog_token_for_keyword(keyword)


def _identify_layout(
    catalog_token: str,
    flat_size: int,
    *,
    class_hint: str | None = None,
) -> tuple[ResponseLayout, str, int]:
    """Find the catalog entry matching ``(token, flat_size_per_element)``.

    Disambiguation rules:

    1. If ``class_hint`` is provided, restrict to entries with that
       ``class_name``. This is how the transcoder consumer overrides
       the column-count sniff when two classes share a flat size for
       reasons other than coincidence (e.g. ``SSPbrick`` and
       ``FourNodeTetrahedron`` both produce 6 columns for ``stress``,
       but their geometry is wildly different).
    2. If multiple entries match and they all share the **shape**
       (same ``n_gauss_points``, ``n_components_per_gp``,
       ``natural_coords``, ``component_layout``, ``coord_system``),
       pick the first — they're functionally equivalent for decoding
       purposes (e.g. ``Brick`` vs ``BbarBrick`` have different
       formulations but identical recorder output shape). The
       returned ``class_name`` is the first match's; downstream
       readers that need to distinguish formulations should use
       ``class_hint``.
    3. Otherwise raise ``ValueError`` with the candidate list.

    Returns ``(layout, class_name, int_rule)``.
    """
    matches = [
        (cls, rule, layout)
        for (cls, rule, tok), layout in RESPONSE_CATALOG.items()
        if tok == catalog_token and layout.flat_size_per_element == flat_size
    ]
    if class_hint is not None:
        matches = [m for m in matches if m[0] == class_hint]

    if len(matches) == 1:
        cls, rule, layout = matches[0]
        return layout, cls, rule

    if not matches:
        if class_hint is not None:
            raise ValueError(
                f"No catalog entry matches token={catalog_token!r}, "
                f"flat_size_per_element={flat_size}, "
                f"class_hint={class_hint!r}. Check the class name and "
                f"whether the rule combination is in RESPONSE_CATALOG."
            )
        raise ValueError(
            f"No catalog entry matches token={catalog_token!r} with "
            f"flat_size_per_element={flat_size}. Either the element "
            f"class is not yet in RESPONSE_CATALOG, or the record "
            f"mixes element classes (v1 transcoder requires "
            f"homogeneous-class records)."
        )

    if _all_shape_equivalent(matches):
        cls, rule, layout = matches[0]
        return layout, cls, rule

    candidates = ", ".join(f"{cls}@{rule}" for cls, rule, _ in matches)
    raise ValueError(
        f"Ambiguous catalog match for token={catalog_token!r} with "
        f"flat_size_per_element={flat_size}: {candidates}. Pass "
        f"``class_hint=`` to ``_identify_layout`` to pick one — these "
        f"classes have different shapes (GPs / coords / layout) and "
        f"the transcoder cannot guess from the column count alone."
    )


def _all_shape_equivalent(
    matches: list[tuple[str, int, ResponseLayout]],
) -> bool:
    """True if every match has the same decoding shape modulo class_tag.

    Used to silently pick a representative when multiple entries
    differ only in formulation (``Brick`` vs ``BbarBrick``, etc.).
    """
    if len(matches) < 2:
        return True
    first = matches[0][2]
    for _, _, layout in matches[1:]:
        if layout.n_gauss_points != first.n_gauss_points:
            return False
        if layout.n_components_per_gp != first.n_components_per_gp:
            return False
        if layout.coord_system != first.coord_system:
            return False
        if layout.component_layout != first.component_layout:
            return False
        if not np.array_equal(layout.natural_coords, first.natural_coords):
            return False
    return True


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
