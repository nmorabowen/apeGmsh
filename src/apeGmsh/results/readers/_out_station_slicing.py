"""Per-element station checks for classic ``.out`` beam records.

A ``.out`` record over beam sections (line stations, or fibres) comes with an
``integrationPoints`` pair. Both files carry one column block per element and
no per-element point count, so a reader slices them into equal blocks. An
element with 3 Lobatto points beside one with 5 also gives 4 columns each,
and the equal slices then cut across the element boundary: one element takes
the next one's first station.

Two checks refuse that, so every ``.out`` reader slices by the same rule:

* :func:`line_station_count`: with a model bound, each element's count, read
  from the model's ``beamIntegration``, must be the same and must fill the
  file's columns. It runs before the columns are divided, so a 3 + 4 record
  is refused for its counts, as the fibre reader refuses it, and never as
  "not divisible" (#2267). Without a model it only divides the columns;
* :func:`require_ordered_stations`: each element's natural stations must
  increase strictly within [-1, 1]. This is the check without a model.

:func:`require_line_stations` applies the second when no model is bound. A
model-proven slice skips the order check: a legal rule
can place its stations out of order (HingeRadau with ``lp_i = lp_j = 0.15``
on ``L = 0.75`` gives ``[-1, 0.067, 0.346, -0.346, -0.067, 1]``).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

from ...opensees import OpenSeesModel
from ...opensees._element_capabilities import _FORCE_DISP_BEAMS
from ..spec._emit import element_tag_map

if TYPE_CHECKING:
    from ..spec._resolved import ResolvedRecorderRecord

#: Element tokens whose argument tail (after the connectivity) starts
#: ``transfTag integrationTag``: the opensees package's own set.
INTEGRATION_ELEMENT_TOKENS = _FORCE_DISP_BEAMS
#: ``beamIntegration`` rules whose arguments are ``(secTag, N)``.
UNIFORM_RULE_TOKENS = frozenset({
    "Legendre", "Lobatto", "NewtonCotes", "Radau", "Trapezoidal",
})
#: The plastic-hinge rules' fixed point counts: OpenSees's
#: ``integrationPoints`` response gives 6, 6, 4 and 4 positions.
HINGE_RULE_POINTS = {
    "HingeRadau": 6, "HingeRadauTwo": 6, "HingeMidpoint": 4, "HingeEndpoint": 4,
}
#: Slack on the natural range [-1, 1]: a ``.out`` written at OpenSees's
#: default ``-precision 6`` puts an end station near 1 + 1e-5.
STATION_RANGE_TOL = 1.0e-4


def model_station_counts(
    rec: "ResolvedRecorderRecord", model: OpenSeesModel, fem_snapshot_id: str,
) -> ndarray:
    """Each element's integration-point count in ``model``, in record order.

    The model must be built from the spec's FEMData (its lineage is checked).
    An element the model does not hold, one that is not a beam with a
    ``beamIntegration``, or a rule this table does not know is refused by
    name: its count would be a guess.
    """
    element_tag_map(model, fem_snapshot_id=fem_snapshot_id)   # lineage, or raise
    by_fem = {int(r.fem_eid): r for r in model.elements() if int(r.fem_eid) >= 0}
    rules = {int(bi.tag): bi for bi in model.beam_integration()}
    assert rec.element_ids is not None
    counts = np.empty(len(rec.element_ids), dtype=np.int64)
    for i, e in enumerate(rec.element_ids):
        what = f"Record {rec.name!r} ({rec.category}): FEM element {int(e)}"
        if int(e) not in by_fem:
            raise ValueError(f"{what} has no element in model=.")
        er = by_fem[int(e)]
        tail = er.args[len(er.connectivity):]
        if er.type_token not in INTEGRATION_ELEMENT_TOKENS or len(tail) < 2:
            raise ValueError(
                f"{what} is a {er.type_token!r} in model=, not a beam with a "
                f"beamIntegration ({sorted(INTEGRATION_ELEMENT_TOKENS)}), so "
                f"its section stations are unknown.")
        integ_tag = int(tail[1])
        if integ_tag not in rules:
            raise ValueError(
                f"{what} names beamIntegration {integ_tag}, which model= "
                f"does not hold.")
        bi = rules[integ_tag]
        if bi.type_token in UNIFORM_RULE_TOKENS:
            counts[i] = int(bi.args[1])
        elif bi.type_token in HINGE_RULE_POINTS:
            counts[i] = HINGE_RULE_POINTS[bi.type_token]
        else:
            raise ValueError(
                f"{what} uses beamIntegration {bi.type_token!r}, whose "
                f"point count is not known here; known: "
                f"{sorted(UNIFORM_RULE_TOKENS | set(HINGE_RULE_POINTS))}.")
    return counts


def line_station_count(
    rec: "ResolvedRecorderRecord", n_columns: int, path: "str | Path",
    model_h5_src: "Path | None", fem_snapshot_id: str,
) -> int:
    """The integration-point count of every element of ``rec``, whose
    ``integrationPoints`` file at ``path`` has ``n_columns`` data columns.

    With ``model_h5_src`` the model's counts decide, and they are checked
    first: unequal counts are refused naming the elements by count, as the
    fibre reader refuses them (#2267), before the columns are divided. Equal
    counts that do not fill ``n_columns`` mean the file is not from this
    model. Without a model, the columns must divide evenly by element.
    """
    assert rec.element_ids is not None
    n_elements = len(rec.element_ids)
    if model_h5_src is None:
        if n_columns % n_elements != 0:
            raise ValueError(
                f"{path}: {n_columns} data columns is not divisible by "
                f"{n_elements} elements. The integrationPoints file should be "
                f"element-major with n_IP columns per element.")
        return n_columns // n_elements
    counts = model_station_counts(
        rec, OpenSeesModel.from_h5(model_h5_src), fem_snapshot_id)
    if np.unique(counts).size > 1:
        by_count: dict[int, list[int]] = {}
        for e, c in zip(rec.element_ids, counts):
            by_count.setdefault(int(c), []).append(int(e))
        listing = "; ".join(
            f"{c} points: FEM elements {ids[:10]}{' ...' if len(ids) > 10 else ''}"
            for c, ids in sorted(by_count.items()))
        raise ValueError(
            f"Record {rec.name!r} ({rec.category}): its elements have "
            f"different integration-point counts in model= ({listing}). A .out "
            f"record carries no per-element count, so it cannot be sliced into "
            f"elements reliably. Split the record so that each holds one "
            f"point count.")
    n_ip = int(counts[0])
    if n_columns != n_elements * n_ip:
        raise ValueError(
            f"{path}: {n_columns} data columns over {n_elements} elements, but "
            f"model= gives its elements {n_ip} integration points each. The "
            f"file does not match the model.")
    return n_ip


def require_ordered_stations(
    rec: "ResolvedRecorderRecord", xi_natural: ndarray, path: "str | Path",
) -> None:
    """Raise unless each row of ``xi_natural`` (one element's natural
    stations) increases strictly within [-1, 1].

    For a slice no model proves: a block that crosses an element boundary
    carries one element's end and the next one's start, so it is out of
    order. A block can also be out of order legally, so the refusal names
    the order and asks for the model, which proves the slice.
    """
    tol = STATION_RANGE_TOL
    bad = ~(np.all(np.diff(xi_natural, axis=1) > 0.0, axis=1)
            & np.all(np.abs(xi_natural) <= 1.0 + tol, axis=1))
    if bad.any():
        assert rec.element_ids is not None
        i = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"{path}: the stations read for FEM element "
            f"{int(rec.element_ids[i])} of record {rec.name!r} "
            f"({rec.category}) are {np.round(xi_natural[i], 6).tolist()}, "
            f"which do not increase strictly within [-1, 1]. Without model= "
            f"the equal slices of a .out record are read only when each "
            f"element's stations are in that order, because the files carry "
            f"no per-element point count. Pass model=, whose beamIntegration "
            f"counts prove the slicing.")


def require_line_stations(
    rec: "ResolvedRecorderRecord", xi_natural: ndarray, path: "str | Path",
    model_h5_src: "Path | None", fem_snapshot_id: str,
) -> None:
    """Check an equal-slice ``(E, n_IP)`` read of ``rec``'s stations.

    With ``model_h5_src`` the model's counts proved the slicing already
    (:func:`line_station_count` gave ``n_IP``), so there is nothing left to
    check; without one (a transcode called with no model), the station order
    must hold.
    """
    if model_h5_src is None:
        require_ordered_stations(rec, xi_natural, path)
