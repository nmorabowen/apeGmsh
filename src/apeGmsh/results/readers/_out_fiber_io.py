"""Fibre and per-Gauss-point material decoding for classic ``.out`` files.

The ``.out`` sibling of :mod:`._mpco_fiber_io` / :mod:`._mpco_material_io`:
the files :func:`apeGmsh.results.spec._emit.emit_logical` names for a
``fibers`` record (``section fiberData2`` plus its ``integrationPoints``
pair) and for a material-state gauss record (one ``material <gp> <token>``
file per Gauss point), decoded into the layouts the MPCO route gives, under
its component names. ``RecorderTranscoder`` calls these and writes the
result; nothing here opens a writer.

Each record's rows are labelled with the spec's FEM ids, so each reader
first proves the model's ``(fem_eid, tag)`` pairing holds every one of them
(:func:`require_translated`): the emit translated through it.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

import numpy as np
from numpy import ndarray

from ...opensees._response_catalog import ResponseLayout, normalise_integration_points
from ..spec._emit import (
    FIBER_OUT_COLUMNS,
    FIBER_OUT_WIDTH,
    element_tag_map,
    fiber_record_class,
    per_material_gauss_layout,
)

if TYPE_CHECKING:
    from ..spec._emit import LogicalRecorder
    from ..spec._resolved import ResolvedRecorderRecord


def require_translated(
    rec: "ResolvedRecorderRecord", model_h5_src: "Path | None",
    fem_snapshot_id: str,
) -> Any:
    """Refuse a fibre or per-material record whose ids no model pairs, and
    return that model (an ``OpenSeesModel``).

    Its rows are labelled with the spec's FEM ids, which the emit
    translated through the model's ``(fem_eid, tag)`` pairing; without
    that model, or with ids it does not hold, the labels are unproven.
    """
    from ...opensees import OpenSeesModel
    if model_h5_src is None:
        raise ValueError(
            f"Record {rec.name!r} ({rec.category}): its .out rows name "
            f"elements by OpenSees tag, translated from FEM ids through the "
            f"model's pairing. Pass the model (model_h5_src=, or "
            f"Results.from_recorders(..., model=)).")
    model = OpenSeesModel.from_h5(model_h5_src)
    tags = element_tag_map(model, fem_snapshot_id=fem_snapshot_id)
    missing = [int(e) for e in np.asarray(rec.element_ids) if int(e) not in tags]
    if missing:
        raise ValueError(
            f"Record {rec.name!r} ({rec.category}): FEM element ids "
            f"{missing[:10]} have no OpenSees element in the model.")
    return model


#: Fibre-section beams whose element args start ``$transfTag $integrationTag``.
_FIBER_BEAM_TOKENS = frozenset({"forceBeamColumn", "dispBeamColumn"})
#: ``beamIntegration`` rules whose args are ``$secTag $N``: ``N`` points.
_FIXED_N_RULES = frozenset({"Lobatto", "Legendre", "NewtonCotes", "Radau", "Trapezoidal"})


def model_integration_points(rec: "ResolvedRecorderRecord", model: Any) -> "int | None":
    """The one integration-point count the model gives every element of a
    fibres record, or ``None`` when they all share one rule whose count its
    args do not state (a hinge rule): uniform, but unchecked.

    ``.out`` carries no per-element count, so a record whose elements the
    model gives different (or unstated, under different rules) counts is
    refused: its columns would divide evenly into the wrong sections. An
    element that is not a fibre-section beam the model can describe is
    refused by its type token.
    """
    rows = {int(r.fem_eid): r for r in model.elements()}
    rules = {int(b.tag): b for b in model.beam_integration()}
    tags: set[int] = set()
    for e in np.asarray(rec.element_ids, dtype=np.int64).tolist():
        row = rows[e]
        if row.type_token not in _FIBER_BEAM_TOKENS or int(row.args[1]) not in rules:
            raise ValueError(
                f"Record {rec.name!r} (category=fibers): FEM element {e} is "
                f"{row.type_token!r} in the model, not one of "
                f"{sorted(_FIBER_BEAM_TOKENS)} with a beamIntegration rule, so "
                f"its integration-point count cannot be checked.")
        tags.add(int(row.args[1]))
    counts = {int(rules[t].args[1]) if rules[t].type_token in _FIXED_N_RULES else None
              for t in tags}
    if len(tags) > 1 and (len(counts) != 1 or None in counts):
        raise ValueError(
            f"Record {rec.name!r} (category=fibers): its elements use "
            f"beamIntegration rules {sorted(tags)}, with integration-point "
            f"counts {sorted(counts, key=str)}; .out carries no per-element "
            f"count, so the record must hold one count. Split the record by "
            f"integration rule.")
    return counts.pop()


def _read_rows(lr: "LogicalRecorder", n_elements: int, unit: int,
               ) -> tuple[ndarray, ndarray, int]:
    """``(time (T,), body (T, E, n * unit), n)``: the file's per-element width
    in whole ``unit``-value entries, or raise."""
    path = Path(lr.file_path)
    if not path.exists():
        raise FileNotFoundError(f"Recorder output file not found: {path}.")
    raw = np.loadtxt(path, dtype=np.float64, ndmin=2)
    n, rem = divmod(raw.shape[1] - 1, n_elements * unit)
    if raw.shape[0] == 0 or n < 1 or rem:
        raise ValueError(
            f"{path}: {raw.shape[1] - 1} data columns are not {n_elements} "
            f"elements x a whole number of {unit}-value entries; the record "
            f"needs one element class and one section layout.")
    return raw[:, 0], raw[:, 1:].reshape(raw.shape[0], n_elements, -1), n


def read_per_material_gauss(
    rec: "ResolvedRecorderRecord", logicals: Sequence["LogicalRecorder"],
    model_h5_src: "Path | None", fem_snapshot_id: str,
) -> tuple[str, int, ResponseLayout, ndarray, ndarray]:
    """``(class_name, int_rule, layout, time, flat)`` of a gauss record's
    ``material <gp> <token>`` files, one per Gauss point.

    ``flat`` is ``(T, E, n_GP * K)``, GP-slow: the ``unflatten`` order, with
    file ``gp`` at catalog point ``gp - 1``. A material whose response
    width is not the catalog's ``K`` fails the column check.
    """
    require_translated(rec, model_h5_src, fem_snapshot_id)
    layout, class_name, int_rule = per_material_gauss_layout(rec)
    k = layout.n_components_per_gp
    if len(logicals) != layout.n_gauss_points:
        raise ValueError(f"Gauss record {rec.name!r}: {len(logicals)} per-point "
                         f"files for {layout.n_gauss_points} Gauss points.")
    n_e = len(np.asarray(rec.element_ids))
    reads = [_read_rows(lr, n_e, k) for lr in logicals]
    time = reads[0][0]
    for lr, (t, _body, n) in zip(logicals, reads):
        if n != 1 or not np.array_equal(t, time):
            raise ValueError(
                f"{lr.file_path}: expected {k} values per element and the "
                f"time column of the record's other Gauss-point files.")
    flat = np.stack([body for _t, body, _n in reads], axis=2)
    return class_name, int_rule, layout, time, flat.reshape(time.size, n_e, -1)


class FiberRecordPayload:
    """One ``fibers`` record's parse: the fibre index columns and values."""
    __slots__ = ("record_name", "class_name", "time", "index", "components")

    def __init__(self, record_name: str, class_name: str, time: ndarray,
                 index: dict[str, ndarray], components: dict[str, ndarray]) -> None:
        self.record_name = record_name
        self.class_name = class_name
        self.time = time
        self.index = index
        self.components = components


def read_fiber_record(
    rec: "ResolvedRecorderRecord", logicals: Sequence["LogicalRecorder"],
    lengths_of: Callable[[ndarray], ndarray], model_h5_src: "Path | None",
    fem_snapshot_id: str,
) -> "FiberRecordPayload | None":
    """A ``section fiberData2`` file and its ``integrationPoints`` pair.

    ``lengths_of(ids)`` gives the elements' lengths, in ``ids`` order;
    ``None`` when the record emitted nothing (no elements). The pair gives
    each element's natural stations; the count is one per record, which
    the model must confirm (:func:`model_integration_points`), and each
    element's stations must increase strictly within ``[-1, 1]``, so a
    record mixing counts is refused, never split evenly into wrong
    sections. The fibre file gives
    ``n_sections * n_fibres * 6`` columns per element, element-major,
    section-mid, fibre-fast: the MPCO fibre-slab order. No response gives a
    section's fibre count, so every (element, section) must carry one fibre
    table (``y, z, area, matTag``) on every row: a record whose elements
    carry different fibre counts, which the column count alone can hide,
    then reads as unequal tables and is refused.
    """
    class_name = fiber_record_class(rec)
    if not logicals:
        return None
    n_model = model_integration_points(
        rec, require_translated(rec, model_h5_src, fem_snapshot_id))
    fib_lr, gpx_lr = logicals
    ids = np.asarray(rec.element_ids, dtype=np.int64)
    _t, gpx, n_ip = _read_rows(gpx_lr, ids.size, 1)
    xi = np.stack([normalise_integration_points(gpx[0, i], float(length))
                   for i, length in enumerate(lengths_of(ids))])
    if (n_model is not None and n_ip != n_model) or not (
            (np.diff(xi, axis=1) > 0.0).all() and (np.abs(xi) <= 1.0 + 1e-12).all()):
        raise ValueError(
            f"{gpx_lr.file_path}: {n_ip} stations per element (the model "
            f"gives {n_model}), not strictly increasing within [-1, 1] on "
            f"every element: the integrationPoints columns do not split "
            f"into the record's elements.")
    time, body, n_fib = _read_rows(fib_lr, ids.size, n_ip * FIBER_OUT_WIDTH)
    rows = body.reshape(time.size, -1, FIBER_OUT_WIDTH)
    geom = rows[0, :, :4]
    if not np.array_equal(rows[:, :, :4].reshape(-1, n_fib, 4), np.broadcast_to(
            geom[:n_fib], (time.size * ids.size * n_ip, n_fib, 4))):
        raise ValueError(
            f"{fib_lr.file_path}: the fibre tables (y, z, area, matTag) differ "
            f"between the record's sections or steps, so its {ids.size} "
            f"elements cannot be split into fibres reliably: .out carries no "
            f"per-section fibre count. Split the record so that each holds "
            f"one fibre section.")
    return FiberRecordPayload(
        record_name=rec.name, class_name=class_name, time=time,
        index=dict(
            element_index=np.repeat(ids, n_ip * n_fib),
            gp_index=np.tile(np.repeat(np.arange(n_ip, dtype=np.int64), n_fib), ids.size),
            y=geom[:, 0], z=geom[:, 1], area=geom[:, 2],
            material_tag=np.round(geom[:, 3]).astype(np.int64),
            station_natural_coord=np.repeat(xi.ravel(), n_fib),
        ),
        components={c: np.ascontiguousarray(rows[:, :, FIBER_OUT_COLUMNS[c]])
                    for c in rec.components},
    )


def write_fiber_records(writer: Any, stage_id: str,
                        records: Iterable[FiberRecordPayload]) -> None:
    """Write each fibre payload as one fibers group."""
    for i, fr in enumerate(records):
        writer.write_fibers_group(
            stage_id, "partition_0",
            group_id=f"{fr.record_name}_{fr.class_name}_{i}",
            section_tag=-1, section_class="(recorder)",
            components=fr.components, **fr.index,
        )
