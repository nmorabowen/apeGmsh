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
(:func:`require_translated`): the emit translated through it. The caller
loads that model once and passes it: the bridge's ``OpenSeesModel``, or the
live pairing a FEMData archive's run proved (#2269), which a per-material
gauss record reads through (#2290). A fibres record reads through it too: the
pairing read each element's section locations and fibre counts from the live
domain before the run, in place of the bridge archive's ``beamIntegration``.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

import numpy as np
from numpy import ndarray

from ...opensees import OpenSeesModel
from ...opensees._recorder_translate import LiveDomainFacts
from ...opensees._response_catalog import ResponseLayout, normalise_integration_points
from ..spec._emit import (
    FIBER_OUT_COLUMNS,
    FIBER_OUT_WIDTH,
    element_tag_map,
    fiber_record_class,
    per_material_gauss_layout,
)
from ._out_station_slicing import (
    STATION_RANGE_TOL,
    line_station_count,
)

if TYPE_CHECKING:
    from ..spec._emit import LogicalRecorder
    from ..spec._resolved import ResolvedRecorderRecord


def require_translated(
    rec: "ResolvedRecorderRecord", model: Any, fem_snapshot_id: str,
) -> None:
    """Refuse a fibre or per-material record whose ids no model pairs.

    Its rows are labelled with the spec's FEM ids, which the emit
    translated through the ``(fem_eid, tag)`` pairing of ``model``: the
    bridge's ``OpenSeesModel``, or a live pairing
    (:class:`~apeGmsh.opensees._recorder_translate.LiveDomainFacts`, its
    lineage and proved ``elements()``). Without one, or with ids it does
    not hold, the labels are unproven.
    """
    if model is None:
        raise ValueError(
            f"Record {rec.name!r} ({rec.category}): its .out rows name "
            f"elements by OpenSees tag, translated from FEM ids through the "
            f"model's pairing. Pass the model (model_h5_src=, or "
            f"Results.from_recorders(..., model=)).")
    tags = element_tag_map(model, fem_snapshot_id=fem_snapshot_id)
    missing = [int(e) for e in np.asarray(rec.element_ids) if int(e) not in tags]
    if missing:
        raise ValueError(
            f"Record {rec.name!r} ({rec.category}): FEM element ids "
            f"{missing[:10]} have no OpenSees element in the model.")


#: Element-family suffixes of the beams a fibres record reads:
#: ``forceBeamColumn``, ``dispBeamColumn`` and its fork variant (matched by
#: suffix, so no fork name is spelled here). Other fibre classes refuse.
_FIBER_BEAM_SUFFIXES = ("forcebeamcolumn", "dispbeamcolumn")


def require_record_class(rec: "ResolvedRecorderRecord", model: Any) -> None:
    """Refuse a fibres record whose elements are not all of its class.

    Each element's type token must be the record's class without its
    ``2d`` / ``3d`` suffix (``ForceBeamColumn3d`` is ``forceBeamColumn``),
    one of the families above; any other element is refused by its token,
    so a record mixing classes is too. The point counts are then checked by
    the rule both ``.out`` readers share
    (:func:`._out_station_slicing.line_station_count`).
    """
    family = str(rec.element_class_name)[:-2].lower()
    rows = {int(r.fem_eid): r for r in model.elements()}
    for e in np.asarray(rec.element_ids, dtype=np.int64).tolist():
        token = rows[e].type_token.lower()
        if token != family or not token.endswith(_FIBER_BEAM_SUFFIXES):
            raise ValueError(
                f"Record {rec.name!r} (category=fibers): FEM element {e} is "
                f"{rows[e].type_token!r} in the model, not the record's class "
                f"{rec.element_class_name!r} as a force or displacement beam, "
                f"so its integration-point count cannot be checked.")


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
    model: Any, fem_snapshot_id: str,
) -> tuple[str, int, ResponseLayout, ndarray, ndarray]:
    """``(class_name, int_rule, layout, time, flat)`` of a gauss record's
    ``material <gp> <token>`` files, one per Gauss point.

    ``model`` is the bridge's model or a live pairing
    (:func:`require_translated`); the layout is the record's class's in
    the catalog either way, and a live pairing has proved that class by
    ``eleType``. ``flat`` is ``(T, E, n_GP * K)``, GP-slow: the
    ``unflatten`` order, with file ``gp`` at catalog point ``gp - 1``. A
    material whose response width is not the catalog's ``K`` fails the
    column check.
    """
    require_translated(rec, model, fem_snapshot_id)
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
    lengths_of: Callable[[ndarray], ndarray], model: Any,
    fem_snapshot_id: str,
) -> "FiberRecordPayload | None":
    """A ``section fiberData2`` file and its ``integrationPoints`` pair.

    ``lengths_of(ids)`` gives the elements' lengths, in ``ids`` order;
    ``None`` when the record emitted nothing (no elements). The pair gives
    each element's natural stations. The count is one per record, which the
    model must confirm by the rule both ``.out`` readers share
    (:func:`._out_station_slicing.line_station_count`, which checks the
    counts before the columns divide and refuses a rule of unknown count),
    and each element's stations must
    increase strictly within ``[-1, 1]``, to the shared slack a file written
    at ``-precision 6`` needs. So a record mixing counts is refused, never
    split evenly into wrong sections. The fibre file gives
    ``n_sections * n_fibres * 6`` columns per element, element-major,
    section-mid, fibre-fast: the MPCO fibre-slab order. No response gives a
    section's fibre count, so every (element, section) must carry one fibre
    table (``y, z, area, matTag``) on every row: a record whose elements
    carry different fibre counts, which the column count alone can hide,
    then reads as unequal tables and is refused.

    Over a live pairing (:class:`LiveDomainFacts`, a FEMData archive's run)
    the counts are the live domain's, read before the run: each element's
    ``sectionLocation`` count stands in for its ``beamIntegration`` by the
    same rule, every section must carry one ``fiberData2`` fibre count that
    the file's columns then confirm, and the file's stations must be the
    live section locations (:func:`_require_live_fibers`). That match places
    each station and so stands in for the order check: a legal rule can
    write its stations out of order (a long-hinge HingeRadau), and the
    line-stations read accepts them on the same proof (#2332).
    """
    class_name = fiber_record_class(rec)
    if not logicals:
        return None
    require_translated(rec, model, fem_snapshot_id)
    if not isinstance(model, (OpenSeesModel, LiveDomainFacts)):
        raise ValueError(
            f"Record {rec.name!r} (category=fibers): its stations are counted "
            f"from each element's beamIntegration in the bridge's model "
            f"archive, or from the sections a live pairing read before the "
            f"run; a {type(model).__name__} carries neither. Record it through "
            f"the bridge's model archive (ops.h5(...)) or emit_default.")
    require_record_class(rec, model)
    fib_lr, gpx_lr = logicals
    ids = np.asarray(rec.element_ids, dtype=np.int64)
    # The whole row as one block; the model's counts then divide it, checked
    # before the columns are, since a mix need not divide by the elements.
    _t, row, n_columns = _read_rows(gpx_lr, 1, 1)
    n_ip = line_station_count(rec, n_columns, gpx_lr.file_path, model,
                              fem_snapshot_id)
    gpx = row.reshape(row.shape[0], ids.size, n_ip)
    xi = np.stack([normalise_integration_points(gpx[0, i], float(length))
                   for i, length in enumerate(lengths_of(ids))])
    out = np.flatnonzero(~(np.abs(xi) <= 1.0 + STATION_RANGE_TOL).all(axis=1))
    if out.size:
        raise ValueError(
            f"{gpx_lr.file_path}: FEM element {int(ids[out[0]])} has a station "
            f"outside the element's natural range [-1, 1] (stations "
            f"{np.round(xi[out[0]], 6).tolist()}); the likely cause is an "
            f"element length or integration-point count that does not match "
            f"the file.")
    if isinstance(model, LiveDomainFacts):
        # The live section locations place each station, in whatever order
        # its rule gives (a long-hinge HingeRadau writes them out of order,
        # legally), as the line-stations read does (#2332).
        live_fib: "int | None" = _require_live_fibers(
            rec, model, xi, lengths_of(ids), gpx_lr.file_path)
    else:
        live_fib = None
        bad = np.flatnonzero(~(np.diff(xi, axis=1) > 0.0).all(axis=1))
        if bad.size:
            raise ValueError(
                f"{gpx_lr.file_path}: FEM element {int(ids[bad[0]])} has a "
                f"station repeated or out of order (stations "
                f"{np.round(xi[bad[0]], 6).tolist()} in natural coordinates; "
                f"they must be strictly increasing within [-1, 1]). A "
                f"HingeRadau rule with lp_i + lp_j >= L/4 writes stations like "
                f"this (its interior weight is not positive), so the fibres "
                f"cannot be placed along the element.")
    time, body, n_fib = _read_rows(fib_lr, ids.size, n_ip * FIBER_OUT_WIDTH)
    if live_fib is not None and n_fib != live_fib:
        raise ValueError(
            f"{fib_lr.file_path}: {n_fib} fibres per section in the file, but "
            f"the live domain gave each section {live_fib} before the run. "
            f"The file does not match the live domain.")
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


def _require_live_fibers(
    rec: "ResolvedRecorderRecord", facts: LiveDomainFacts, xi: ndarray,
    lengths: ndarray, path: "str | Path",
) -> int:
    """The one fibre count of every section of ``rec``'s elements, from
    the sections the live pairing read before the run, or raise.

    Each element's file stations ``xi`` (natural, ``(E, n_IP)``) must be
    its live section locations, to the slack a ``-precision 6`` file needs,
    so each fibre table is placed at the section that wrote it. A record
    whose sections carry different fibre counts is refused naming them: a
    ``.out`` row carries no per-section count, so it cannot be split.
    """
    ids = np.asarray(rec.element_ids, dtype=np.int64)
    by_count: dict[int, list[str]] = {}
    for i, e in enumerate(ids.tolist()):
        locations, fibres = facts.fiber_sections(e)
        live = normalise_integration_points(
            np.asarray(locations, dtype=np.float64), float(lengths[i]))
        off = np.abs(live - xi[i])
        if not (off <= STATION_RANGE_TOL).all():
            raise ValueError(
                f"{path}: FEM element {e} has stations "
                f"{np.round(xi[i], 6).tolist()} in the file, but its live "
                f"sections are at {np.round(live, 6).tolist()} (natural "
                f"coordinates). The file does not match the live domain.")
        for k, n in enumerate(fibres, start=1):
            by_count.setdefault(int(n), []).append(f"{e}:{k}")
    if len(by_count) > 1:
        listing = "; ".join(
            f"{n} fibres: sections {where[:6]}{' ...' if len(where) > 6 else ''}"
            for n, where in sorted(by_count.items()))
        raise ValueError(
            f"Record {rec.name!r} (category=fibers): its sections carry "
            f"different fibre counts in the live domain ({listing}; FEM "
            f"element:section). A .out row carries no per-section fibre count, "
            f"so it cannot be split into fibres reliably. Split the record so "
            f"that each holds one fibre section.")
    return next(iter(by_count))


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
