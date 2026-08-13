"""FEMData catalog: MODEL.* and MESH.*."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._catalog import CATALOG_SEVERITY, EVIDENCE_CAP
from ._inverted import coords_are_planar, measure_linear_cell
from ._report import render_text
from ._types import AssessmentReport, Finding

if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData


def assess_fem(fem: FEMData, *, figures: bool = False) -> AssessmentReport:
    """Run the v1 FEM catalog. ``figures=True`` is S3."""
    _refuse_figures(figures)
    findings, skipped = collect_fem(fem)
    return AssessmentReport(
        findings=findings,
        text=render_text(findings, skipped, source="FEMData"),
        figures=(),
        lineage=None,
    )


def collect_fem(
    fem: FEMData,
) -> tuple[tuple[Finding, ...], tuple[tuple[str, str], ...]]:
    findings: list[Finding] = []
    skipped: list[tuple[str, str]] = []
    _check_empty(fem, findings)
    _check_inverted(fem, findings, skipped)
    _check_coincident(fem, findings)
    _check_empty_pg(fem, findings)
    return tuple(findings), tuple(skipped)


def _refuse_figures(figures: bool) -> None:
    if figures:
        raise NotImplementedError(
            "assess(figures=True) is ADR 0094 S3. "
            "S2 implements findings only; pass figures=False (the default)."
        )


def _finding(code: str, message: str, detail: dict[str, object] | None = None) -> Finding:
    return Finding(
        code=code,
        severity=CATALOG_SEVERITY[code],
        message=message,
        detail=detail,
    )


def _check_empty(fem: FEMData, findings: list[Finding]) -> None:
    n_nodes = int(fem.info.n_nodes)
    n_elems = int(fem.info.n_elems)
    if n_nodes == 0 or n_elems == 0:
        findings.append(_finding(
            "MODEL.EMPTY",
            f"{n_nodes} node(s), {n_elems} element(s)",
            {"n_nodes": n_nodes, "n_elems": n_elems},
        ))


def _check_inverted(
    fem: FEMData,
    findings: list[Finding],
    skipped: list[tuple[str, str]],
) -> None:
    ids = np.asarray(fem.nodes.ids)
    coords = np.asarray(fem.nodes.coords, dtype=np.float64)
    if ids.size == 0 or coords.size == 0:
        return
    id_to_idx = {int(nid): i for i, nid in enumerate(ids)}

    inverted: list[int] = []
    skipped_types: list[str] = []
    seen_skip: set[str] = set()
    planar = coords_are_planar(coords)
    skip_2d_nonplanar = False

    for group in fem.elements:
        et = group.element_type
        name = et.name
        if name in {"tri3", "quad4"} and et.order == 1 and not planar:
            skip_2d_nonplanar = True
            continue
        if et.order != 1 or name not in {"tri3", "quad4", "tet4", "hex8"}:
            if et.dim >= 2 and name not in seen_skip:
                seen_skip.add(name)
                if et.order != 1:
                    skipped_types.append(f"{name} (order={et.order}, not judged)")
                else:
                    skipped_types.append(
                        f"{name} (linear, no v1 corner-volume formula)"
                    )
            continue
        for eid, conn in group:
            corners = [id_to_idx.get(int(n)) for n in conn]
            if any(i is None for i in corners):
                continue
            pts = coords[np.asarray(corners, dtype=np.int64)]
            vol = measure_linear_cell(name, pts)
            if vol is None or vol > 0.0:
                continue
            inverted.append(int(eid))

    if skip_2d_nonplanar:
        skipped.append((
            "MESH.INVERTED",
            "2D cells in non-planar 3D space — orientation is not inversion",
        ))
    if skipped_types:
        skipped.append((
            "MESH.INVERTED",
            "not judged: " + ", ".join(skipped_types),
        ))
    if inverted:
        sample = inverted[:EVIDENCE_CAP]
        findings.append(_finding(
            "MESH.INVERTED",
            f"{len(inverted)} inverted / zero-volume linear cell(s)",
            {
                "ids": tuple(sample),
                "n": len(inverted),
            },
        ))


def _check_coincident(fem: FEMData, findings: list[Finding]) -> None:
    pairs = fem.inspect.find_coincident_node_pairs()
    # Branch on emptiness only — never parse the ref strings.
    unbridged = [pair for pair, refs in pairs.items() if not refs]
    if not unbridged:
        return
    sample = unbridged[:EVIDENCE_CAP]
    findings.append(_finding(
        "MESH.COINCIDENT_UNBRIDGED",
        f"{len(unbridged)} unbridged coincident node pair(s)",
        {
            "pairs": tuple((int(a), int(b)) for a, b in sample),
            "n": len(unbridged),
        },
    ))


def _check_empty_pg(fem: FEMData, findings: list[Finding]) -> None:
    pgset = fem.nodes.physical
    empty: list[str] = []
    for name in pgset.names():
        if len(pgset.node_ids(name)) == 0:
            empty.append(name)
    if not empty:
        return
    sample = empty[:EVIDENCE_CAP]
    findings.append(_finding(
        "MESH.EMPTY_PG",
        f"{len(empty)} named physical group(s) with 0 nodes",
        {"names": tuple(sample), "n": len(empty)},
    ))
