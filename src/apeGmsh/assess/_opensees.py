"""OpenSeesModel catalog: OSM.* solver-zone checks (+ RES.ZERO_U).

ADR 0094 Amendment 2. ``osm`` is duck-typed — this module must not
import ``apeGmsh.opensees`` (INV-1 extended), so the broker is typed
as ``Any`` rather than the real ``OpenSeesModel`` class.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._catalog import CATALOG_SEVERITY
from ._report import render_text
from ._types import AssessmentReport, Finding

if TYPE_CHECKING:
    from apeGmsh.results.Results import Results

# Pattern type tokens whose record carries a body (loads / sps /
# ele_loads). Everything else (UniformExcitation, H5DRM, ...) is a
# single-line pattern — its body lives in ``args`` instead.
_BLOCK_PATTERN_TYPES = frozenset({"Plain", "MultiSupport"})

_DISP = ("displacement_x", "displacement_y", "displacement_z")

_CATALOG_CODES = (
    "OSM.NO_ANALYZE",
    "OSM.NO_SUPPORT",
    "OSM.NO_PATTERN",
    "OSM.LOADS_UNIMPORTED",
)


def assess_opensees(
    osm: Any,
    results: "Results | None" = None,
) -> AssessmentReport:
    """Run the v1 solver-zone catalog on an ``OpenSeesModel`` broker.

    ``results=`` adds the ``RES.ZERO_U`` cross-check; omitted, it is
    listed in ``skipped``. Staged decks (``osm.stages()`` non-empty)
    put the whole catalog in ``skipped`` — stage-level judging is a
    follow-up.
    """
    findings: list[Finding] = []
    skipped: list[tuple[str, str]] = []

    if tuple(osm.stages()):
        reason = "staged deck — stage-level judging is a follow-up"
        for code in (*_CATALOG_CODES, "RES.ZERO_U"):
            skipped.append((code, reason))
        return _report(findings, skipped)

    analyze_call = osm.analyze_call()
    patterns = tuple(osm.patterns())

    if analyze_call is None:
        findings.append(_finding(
            "OSM.NO_ANALYZE",
            "No ops.analyze archived. eigen calls are not archived, "
            "so an eigen-only / modal deck legitimately trips this.",
        ))
        skipped.append((
            "OSM.NO_PATTERN",
            "OSM.NO_ANALYZE fired — free-vibration transients are legal",
        ))
    elif not patterns:
        findings.append(_finding(
            "OSM.NO_PATTERN",
            "ops.analyze archived but no pattern declared "
            "(free-vibration transients from initial conditions "
            "are legal).",
        ))

    _check_support(osm, findings)
    _check_loads_unimported(osm, patterns, findings, skipped)
    _check_zero_u(results, analyze_call, patterns, findings, skipped)

    return _report(findings, skipped)


def _report(
    findings: list[Finding], skipped: list[tuple[str, str]],
) -> AssessmentReport:
    packed = tuple(findings)
    skipped_t = tuple(skipped)
    return AssessmentReport(
        findings=packed,
        text=render_text(packed, skipped_t, source="OpenSeesModel"),
        lineage=None,
        skipped=skipped_t,
    )


def _finding(code: str, message: str, detail: dict[str, object] | None = None) -> Finding:
    return Finding(
        code=code,
        severity=CATALOG_SEVERITY[code],
        message=message,
        detail=detail,
    )


def _check_support(osm: Any, findings: list[Finding]) -> None:
    if tuple(osm.fixes()):
        return
    if tuple(osm.fem.nodes.sp.homogeneous()):
        return
    for pattern in osm.patterns():
        if tuple(pattern.sps):
            return
    findings.append(_finding(
        "OSM.NO_SUPPORT",
        "No fix / homogeneous sp / pattern sp found. Free-free eigen "
        "and DRM-style models are legal.",
    ))


def _check_loads_unimported(
    osm: Any,
    patterns: tuple[Any, ...],
    findings: list[Finding],
    skipped: list[tuple[str, str]],
) -> None:
    if not patterns:
        # OSM.NO_ANALYZE / OSM.NO_PATTERN already cover "no pattern".
        return
    node_loads = tuple(osm.fem.nodes.loads)
    elem_loads = tuple(osm.fem.elements.loads)
    prescribed_sp = tuple(osm.fem.nodes.sp.prescribed())
    if not node_loads and not elem_loads and not prescribed_sp:
        return

    if any(p.type_token not in _BLOCK_PATTERN_TYPES for p in patterns):
        return  # a single-line pattern (UniformExcitation, ...) exists
    if any(p.loads or p.sps or p.ele_loads for p in patterns):
        return  # at least one pattern block imported something

    findings.append(_finding(
        "OSM.LOADS_UNIMPORTED",
        "Broker carries load / prescribed-displacement records but no "
        "pattern imports them (g.loads declared, from_model forgotten).",
        {
            "n_node_loads": len(node_loads),
            "n_elem_loads": len(elem_loads),
            "n_prescribed_sp": len(prescribed_sp),
            "n_patterns": len(patterns),
        },
    ))
    skipped.append((
        "OSM.LOADS_UNIMPORTED",
        "per-case attribution is undecidable "
        "(from_model provenance is not archived)",
    ))


def _check_zero_u(
    results: "Results | None",
    analyze_call: "tuple[int, float | None] | None",
    patterns: tuple[Any, ...],
    findings: list[Finding],
    skipped: list[tuple[str, str]],
) -> None:
    if results is None:
        skipped.append(("RES.ZERO_U", "results= not passed"))
        return
    if analyze_call is None or not patterns:
        skipped.append((
            "RES.ZERO_U",
            "no archived analyze / pattern to judge against",
        ))
        return

    stages = [s for s in results.stages if getattr(s, "kind", None) != "mode"]
    if not stages:
        skipped.append(("RES.ZERO_U", "no non-mode stages"))
        return

    saw_disp = False
    all_zero = True
    last_stage_id: object = None
    for stage in stages:
        comps = results.inspect.components(stage=stage.id).get("nodes", [])
        axes = [c for c in _DISP if c in comps]
        if not axes:
            continue
        saw_disp = True
        last_stage_id = stage.id
        for component in axes:
            slab = results.nodes.get(component=component, time=-1, stage=stage.id)
            values = np.asarray(slab.values)
            if values.size == 0:
                continue
            row = values[-1] if values.ndim >= 2 else values
            if np.any(np.asarray(row) != 0.0):
                all_zero = False

    if not saw_disp:
        skipped.append(("RES.ZERO_U", "no displacement_* component"))
        return
    if not all_zero:
        return

    findings.append(_finding(
        "RES.ZERO_U",
        f"Last-step displacement is exactly all-zero (stage={last_stage_id!r}) "
        "while an analyze call and >=1 pattern are archived.",
        {"stage": last_stage_id},
    ))
