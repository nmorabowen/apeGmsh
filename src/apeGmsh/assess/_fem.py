"""FEMData catalog: MODEL.* and MESH.*."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ._catalog import CATALOG_SEVERITY, EVIDENCE_CAP
from ._inverted import (
    _JUDGED_2D,
    _JUDGED_LINEAR,
    coords_are_planar,
    measure_linear_cell,
)
from ._report import NO_STILLS, render_text
from ._types import AssessmentReport, Finding

if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData


def assess_fem(
    fem: FEMData,
    *,
    figures: bool = False,
    out_dir: str | Path | None = None,
) -> AssessmentReport:
    """Run the v1 FEM catalog. ``figures=True`` writes one mesh still."""
    findings, skipped = collect_fem(fem)
    fig_paths, fig_note = _fem_figures(fem, figures, out_dir)
    return AssessmentReport(
        findings=findings,
        text=render_text(
            findings, skipped, source="FEMData",
            figures=fig_paths, figure_note=fig_note,
        ),
        figures=fig_paths,
        lineage=None,
        skipped=skipped,
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


def resolve_out_dir(out_dir: str | Path | None) -> Path:
    """Directory for stills. ``None`` → a fresh temp dir."""
    if out_dir is None:
        import tempfile
        return Path(tempfile.mkdtemp(prefix="apegmsh-assess-"))
    return Path(out_dir)


def _fem_figures(
    fem: FEMData,
    figures: bool,
    out_dir: str | Path | None,
) -> tuple[tuple[Path, ...], str | None]:
    if not figures:
        return (), None
    # Broker method — lazy-imports viewers.render (INV-1).
    written = fem.render(resolve_out_dir(out_dir) / "mesh.png")
    if written is None:
        return (), NO_STILLS
    return (written,), None


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
    judged_any = False

    for group in fem.elements:
        et = group.element_type
        name = et.name
        if name in _JUDGED_2D and et.order == 1 and not planar:
            skip_2d_nonplanar = True
            continue
        if et.order != 1 or name not in _JUDGED_LINEAR:
            if name not in seen_skip:
                seen_skip.add(name)
                if et.dim < 2:
                    skipped_types.append(f"{name} (1D, not judged)")
                elif et.order != 1:
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
            if vol is None:
                continue
            judged_any = True
            if name in _JUDGED_2D:
                if vol == 0.0:
                    inverted.append(int(eid))
            elif vol <= 0.0:
                inverted.append(int(eid))

    # Amendment 1: the verdict's "Not evaluated" line is only for
    # "no judged cell type ran". When solids WERE judged and only
    # side types (couplings, 1D, quadratic) were skipped, the skip
    # entry is coverage disclosure — prefix it so
    # fail_reserved_unevaluated does not read it as unevaluated
    # (same mechanism as the union-merge fill carve-out).
    prefix = "partial coverage — " if judged_any else ""
    if skip_2d_nonplanar:
        skipped.append((
            "MESH.INVERTED",
            prefix + "2D cells in non-planar 3D space — "
            "orientation is not inversion",
        ))
    if skipped_types:
        skipped.append((
            "MESH.INVERTED",
            prefix + "not judged: " + ", ".join(skipped_types),
        ))
    if inverted:
        sample = inverted[:EVIDENCE_CAP]
        findings.append(_finding(
            "MESH.INVERTED",
            f"{len(inverted)} inverted / degenerate linear cell(s)",
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
