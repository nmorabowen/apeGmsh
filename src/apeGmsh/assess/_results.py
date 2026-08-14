"""Results catalog: RES.* (+ bound-FEM mesh checks)."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ._catalog import CATALOG_SEVERITY, EVIDENCE_CAP
from ._fem import collect_fem, resolve_out_dir
from ._report import NO_STILLS, render_text
from ._types import AssessmentReport, Finding

if TYPE_CHECKING:
    from apeGmsh.results.Results import Results
    from apeGmsh.results._slabs import NodeSlab

_DISP = ("displacement_x", "displacement_y", "displacement_z")

_LEVEL_GETTERS = (
    ("nodes", lambda r, c, s: r.nodes.get(component=c, time=-1, stage=s)),
    ("elements", lambda r, c, s: r.elements.get(component=c, time=-1, stage=s)),
    ("line_stations", lambda r, c, s: r.elements.line_stations.get(
        component=c, time=-1, stage=s,
    )),
    ("gauss", lambda r, c, s: r.elements.gauss.get(component=c, time=-1, stage=s)),
    ("fibers", lambda r, c, s: r.elements.fibers.get(component=c, time=-1, stage=s)),
    ("layers", lambda r, c, s: r.elements.layers.get(component=c, time=-1, stage=s)),
    ("springs", lambda r, c, s: r.elements.springs.get(component=c, time=-1, stage=s)),
)


def assess_results(
    results: Results,
    *,
    figures: bool = False,
    out_dir: str | Path | None = None,
) -> AssessmentReport:
    """Run the v1 Results catalog. ``figures=True`` calls ``render_pack``."""
    findings: list[Finding] = []
    skipped: list[tuple[str, str]] = []

    fem = results.fem
    if fem is None:
        findings.append(_finding(
            "RES.UNBOUND_FEM",
            "Results has no bound FEMData",
        ))
        skipped.append(("MESH.*", "no bound FEMData"))
    else:
        fem_findings, fem_skipped = collect_fem(fem)
        findings.extend(fem_findings)
        skipped.extend(fem_skipped)

    # lineage never raises (ADR 0021 INV-2). Do not call assert_clean().
    lineage = results.lineage
    warnings = tuple(lineage.warnings)
    if warnings:
        findings.append(_finding(
            "RES.LINEAGE",
            f"{len(warnings)} lineage warning(s)",
            {"warnings": warnings[:EVIDENCE_CAP], "n": len(warnings)},
        ))

    stages = list(results.stages)
    if not stages:
        findings.append(_finding(
            "RES.NO_STAGE",
            "Results file has no stages",
        ))
        skipped.append(("RES.NAN", "no stages"))
        skipped.append(("RES.INF", "no stages"))
        skipped.append(("RES.U_VS_DIAG", "no stages"))
        skipped.append(("RES.ENERGY_ERR", "no stages"))
        skipped.append(("RES.ENERGY_NONFINITE", "no stages"))
    else:
        _check_nonfinite(results, stages, findings, skipped)
        _check_u_vs_diag(results, stages, findings, skipped)
        _check_energy(results, stages, findings, skipped)

    packed = tuple(findings)
    skipped_t = tuple(skipped)
    fig_paths, fig_note = _results_figures(results, figures, out_dir)
    return AssessmentReport(
        findings=packed,
        text=render_text(
            packed, skipped_t, source="Results",
            figures=fig_paths, figure_note=fig_note,
        ),
        figures=fig_paths,
        lineage=lineage,
        skipped=skipped_t,
    )


def _results_figures(
    results: Results,
    figures: bool,
    out_dir: str | Path | None,
) -> tuple[tuple[Path, ...], str | None]:
    if not figures:
        return (), None
    if results.fem is None:
        return (), NO_STILLS
    # Broker method — lazy-imports viewers.render (INV-1).
    written = results.render_pack(resolve_out_dir(out_dir))
    if not written:
        return (), NO_STILLS
    return written, None


def _finding(code: str, message: str, detail: dict[str, object] | None = None) -> Finding:
    return Finding(
        code=code,
        severity=CATALOG_SEVERITY[code],
        message=message,
        detail=detail,
    )


def _check_nonfinite(
    results: Results,
    stages: list[Any],
    findings: list[Finding],
    skipped: list[tuple[str, str]],
) -> None:
    """Judge last-step ``slab.values``. Node-level NaN uses the union-merge
    fill rule (ADR 0094 Amendment 1): a NaN where any other node-level
    component is finite is fill, not ``RES.NAN``.
    """
    getters = dict(_LEVEL_GETTERS)
    for stage in stages:
        comps = results.inspect.components(stage=stage.id)
        node_names = list(comps.get("nodes") or [])
        node_slabs: dict[str, Any] = {}
        if node_names:
            get_nodes = getters["nodes"]
            for name in node_names:
                node_slabs[name] = get_nodes(results, name, stage.id)
            any_finite = _node_any_finite(node_slabs)
            for name, slab in node_slabs.items():
                _emit_node_nonfinite(
                    slab, name, stage.id, any_finite, findings, skipped,
                )

        for level, names in comps.items():
            if level == "nodes":
                continue
            get = getters.get(level)
            if get is None:
                continue
            for name in names:
                slab = get(results, name, stage.id)
                values = np.asarray(slab.values)
                if values.size == 0:
                    continue
                if np.isnan(values).any():
                    findings.append(_finding(
                        "RES.NAN",
                        (
                            f"NaN in {name!r} ({level}) "
                            f"stage={stage.id!r} at last step"
                        ),
                        _nonfinite_detail(slab, stage.id, "nan"),
                    ))
                if np.isinf(values).any():
                    findings.append(_finding(
                        "RES.INF",
                        (
                            f"Inf in {name!r} ({level}) "
                            f"stage={stage.id!r} at last step"
                        ),
                        _nonfinite_detail(slab, stage.id, "inf"),
                    ))


def _last_row(slab: Any) -> Any:
    values = np.asarray(slab.values)
    if values.size == 0:
        return np.array([], dtype=np.float64)
    row = values[-1] if values.ndim >= 2 else values
    return np.asarray(row, dtype=np.float64).reshape(-1)


def _node_any_finite(slabs: dict[str, Any]) -> dict[int, bool]:
    """Per node id: True if any node-level component is finite at last step."""
    any_finite: dict[int, bool] = {}
    for slab in slabs.values():
        ids = np.asarray(getattr(slab, "node_ids", []))
        row = _last_row(slab)
        if ids.size == 0 or row.size == 0:
            continue
        n = min(int(ids.size), int(row.size))
        for i in range(n):
            nid = int(ids[i])
            if np.isfinite(row[i]):
                any_finite[nid] = True
            else:
                any_finite.setdefault(nid, False)
    return any_finite


def _emit_node_nonfinite(
    slab: Any,
    name: str,
    stage_id: str,
    any_finite: dict[int, bool],
    findings: list[Finding],
    skipped: list[tuple[str, str]],
) -> None:
    values = np.asarray(slab.values)
    if values.size == 0:
        return
    ids = np.asarray(getattr(slab, "node_ids", []))
    row = _last_row(slab)
    if np.isinf(row).any():
        findings.append(_finding(
            "RES.INF",
            f"Inf in {name!r} (nodes) stage={stage_id!r} at last step",
            _nonfinite_detail(slab, stage_id, "inf"),
        ))
    nan = np.isnan(row)
    if not nan.any():
        return
    n = min(int(ids.size), int(row.size)) if ids.size else 0
    fill_n = 0
    unexplained: list[int] = []
    for i in range(n):
        if not nan[i]:
            continue
        nid = int(ids[i])
        if any_finite.get(nid, False):
            fill_n += 1
        else:
            unexplained.append(nid)
    if unexplained:
        findings.append(_finding(
            "RES.NAN",
            f"NaN in {name!r} (nodes) stage={stage_id!r} at last step",
            {
                "component": name,
                "stage": stage_id,
                "n": len(unexplained),
                "ids": tuple(unexplained[:EVIDENCE_CAP]),
            },
        ))
    if fill_n:
        skipped.append((
            "RES.NAN",
            (
                f"union-merge fill in {name!r} "
                f"({fill_n} slot(s), stage={stage_id!r})"
            ),
        ))


def _nonfinite_detail(slab: Any, stage_id: str, kind: str) -> dict[str, object]:
    values = np.asarray(slab.values)
    if kind == "nan":
        mask = np.isnan(values)
    else:
        mask = np.isinf(values)
    # values are (T, …); last step is the only row we asked for.
    n = int(mask.sum())
    ids_attr = getattr(slab, "node_ids", None)
    if ids_attr is None:
        ids_attr = getattr(slab, "element_ids", None)
    if ids_attr is None:
        ids_attr = getattr(slab, "element_index", None)
    sample: list[int] = []
    if ids_attr is not None and values.ndim >= 2:
        hit = np.where(mask[0].reshape(mask[0].shape[0], -1).any(axis=1))[0]
        raw_ids = np.asarray(ids_attr)
        sample = [int(raw_ids[i]) for i in hit[:EVIDENCE_CAP]]
    return {
        "component": slab.component,
        "stage": stage_id,
        "n": n,
        "ids": tuple(sample),
    }


def _check_u_vs_diag(
    results: Results,
    stages: list[Any],
    findings: list[Finding],
    skipped: list[tuple[str, str]],
) -> None:
    fem = results.fem
    if fem is None:
        skipped.append(("RES.U_VS_DIAG", "no bound FEMData"))
        return

    usable = [s for s in stages if s.kind != "mode"]
    if not usable:
        skipped.append(("RES.U_VS_DIAG", "all stages are kind='mode'"))
        return

    from apeGmsh.results._geometry import model_diagonal

    coords = np.asarray(fem.nodes.coords, dtype=np.float64)
    diag = model_diagonal(coords)
    id_to_xyz = {
        int(nid): tuple(float(x) for x in xyz)
        for nid, xyz in zip(fem.nodes.ids, coords)
    }

    best_mag = -1.0
    best_rows: list[dict[str, object]] = []
    saw_disp = False

    for stage in usable:
        comps = results.inspect.components(stage=stage.id).get("nodes", [])
        axes = [c for c in _DISP if c in comps]
        if not axes:
            continue
        saw_disp = True
        slabs = [
            results.nodes.get(component=c, time=-1, stage=stage.id)
            for c in axes
        ]
        mags, node_ids, step = _displacement_mags(slabs)
        if mags.size == 0:
            continue
        order = np.argsort(mags)[::-1]
        top = order[:EVIDENCE_CAP]
        rows = [
            {
                "id": int(node_ids[i]),
                "xyz": id_to_xyz.get(int(node_ids[i])),
                "value": float(mags[i]),
                "step": step,
                "stage": stage.id,
            }
            for i in top
        ]
        peak = float(mags[int(order[0])])
        if peak > best_mag:
            best_mag = peak
            best_rows = rows

    if not saw_disp:
        skipped.append(("RES.U_VS_DIAG", "no displacement_* component"))
        return
    if best_mag < 0.0:
        skipped.append(("RES.U_VS_DIAG", "displacement slab was empty"))
        return

    ratio = best_mag / diag if diag else 0.0
    top = best_rows[0]
    findings.append(_finding(
        "RES.U_VS_DIAG",
        (
            f"‖u‖_max / model_diagonal = {ratio:.4g} "
            f"(‖u‖={best_mag:.4g}, diag={diag:.4g}) "
            f"at node {top['id']} xyz={top['xyz']} step={top['step']}"
        ),
        {
            "ratio": ratio,
            "u_max": best_mag,
            "model_diagonal": diag,
            "extrema": best_rows,
        },
    ))


def _displacement_mags(
    slabs: list[NodeSlab],
) -> tuple[Any, Any, object]:
    """‖u‖ per recorded id from the overlapping last-step slabs."""
    id_sets = [np.asarray(s.node_ids) for s in slabs]
    common = id_sets[0]
    for ids in id_sets[1:]:
        common = np.intersect1d(common, ids, assume_unique=False)
    if common.size == 0:
        return np.array([]), np.array([], dtype=np.int64), None
    acc = np.zeros(common.size, dtype=np.float64)
    step: object = None
    for slab in slabs:
        ids = np.asarray(slab.node_ids)
        lookup = {int(n): i for i, n in enumerate(ids)}
        cols = [lookup[int(n)] for n in common]
        vals = np.asarray(slab.values)
        row = vals[-1] if vals.ndim >= 2 else vals
        acc += np.asarray(row)[cols] ** 2
        if step is None and len(slab.time):
            step = slab.time[-1]
    return np.sqrt(acc), common, step


def _check_energy(
    results: Results,
    stages: list[Any],
    findings: list[Finding],
    skipped: list[tuple[str, str]],
) -> None:
    # TypeError is reader-wide (native / MPCO). ValueError is
    # per-stage (channel absent on this stage) — keep walking.
    try:
        results.energy(stage=stages[0].id)
    except TypeError:
        skipped.append((
            "RES.ENERGY_ERR",
            "no energy channel (Results.energy() raised TypeError)",
        ))
        skipped.append((
            "RES.ENERGY_NONFINITE",
            "no energy channel (Results.energy() raised TypeError)",
        ))
        return
    except ValueError:
        pass

    saw_table = False
    for stage in stages:
        if stage.kind == "mode":
            continue
        try:
            df = results.energy(stage=stage.id)
        except TypeError:
            skipped.append((
                "RES.ENERGY_ERR",
                "no energy channel (Results.energy() raised TypeError)",
            ))
            skipped.append((
                "RES.ENERGY_NONFINITE",
                "no energy channel (Results.energy() raised TypeError)",
            ))
            return
        except ValueError:
            continue
        saw_table = True
        if "ERR" not in getattr(df, "columns", ()):
            skipped.append((
                "RES.ENERGY_ERR",
                f"energy table has no ERR column (stage={stage.id!r})",
            ))
            skipped.append((
                "RES.ENERGY_NONFINITE",
                f"energy table has no ERR column (stage={stage.id!r})",
            ))
            continue
        err = float(df["ERR"].iloc[-1])
        if not np.isfinite(err):
            findings.append(_finding(
                "RES.ENERGY_NONFINITE",
                (
                    f"energy balance ERR is non-finite "
                    f"at last step (stage={stage.id!r})"
                ),
                {"ERR": err, "stage": stage.id},
            ))
            continue
        abs_err = abs(err)
        findings.append(_finding(
            "RES.ENERGY_ERR",
            (
                f"energy balance |ERR| = {abs_err:.4g}% "
                f"at last step (stage={stage.id!r})"
            ),
            {"ERR": err, "stage": stage.id},
        ))
    if not saw_table:
        skipped.append((
            "RES.ENERGY_ERR",
            "energy channel absent on all stages",
        ))
        skipped.append((
            "RES.ENERGY_NONFINITE",
            "energy channel absent on all stages",
        ))
