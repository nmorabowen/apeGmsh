"""OpenSeesModel catalog: OSM.* solver-zone checks (+ RES.ZERO_U).

ADR 0094 Amendment 2 (catalog) / Amendment 3 (run evidence from
``results=``). ``osm`` is duck-typed — this module must not import
``apeGmsh.opensees`` (INV-1 extended), so the broker is typed as
``Any`` rather than the real ``OpenSeesModel`` class.
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
    has_run, evidence_stage = _run_evidence(analyze_call, results)

    if analyze_call is None:
        message = (
            "No ops.analyze archived. eigen calls are not archived, "
            "so an eigen-only / modal deck legitimately trips this."
        )
        if evidence_stage is not None:
            message += (
                f" results= shows a run happened (stage={evidence_stage!r}) "
                "— the deck was driven by a custom analyze loop."
            )
        findings.append(_finding("OSM.NO_ANALYZE", message))
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
    _check_loads_unimported(osm, patterns, has_run, findings, skipped)
    _check_zero_u(results, patterns, findings, skipped)

    return _report(findings, skipped)


def _run_evidence(
    analyze_call: "tuple[int, float | None] | None",
    results: "Results | None",
) -> tuple[bool, object | None]:
    """ADR 0094 Amendment 3: a deck *has run* when either an archived
    ``ops.analyze`` call exists, or ``results=`` carries >=1 non-mode
    stage whose last step is readable — a custom analyze loop leaves
    no ``analyze_call()`` trace, so the results themselves are the
    evidence. Modal-only results are deliberately not evidence.

    Returns ``(has_run, results_stage_id)``. ``results_stage_id`` is
    the non-mode stage that supplied results-based evidence, or
    ``None`` (no such stage — including when ``analyze_call`` alone
    already makes ``has_run`` true), so callers can tell whether the
    evidence was results-based.
    """
    stage_id = _results_evidence_stage(results)
    return analyze_call is not None or stage_id is not None, stage_id


def _results_evidence_stage(results: "Results | None") -> object | None:
    """First non-mode stage with a readable last step, or ``None``.

    Reuses the same ``results.stages`` / ``inspect.components(stage=)``
    access ``_check_zero_u`` uses (INV-9: no step walking) — this just
    asks "does a run archive exist here", not "what is in it".
    """
    if results is None:
        return None
    for stage in results.stages:
        if getattr(stage, "kind", None) == "mode":
            continue
        comps = results.inspect.components(stage=stage.id)
        if any(comps.values()):
            return stage.id
    return None


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
    has_run: bool,
    findings: list[Finding],
    skipped: list[tuple[str, str]],
) -> None:
    node_loads = tuple(osm.fem.nodes.loads)
    elem_loads = tuple(osm.fem.elements.loads)
    prescribed_sp = tuple(osm.fem.nodes.sp.prescribed())
    if not node_loads and not elem_loads and not prescribed_sp:
        return
    if not patterns and not has_run:
        # No run evidence (no archived analyze, no results=) and a
        # zero-pattern deck with declared loads: a legal multi-deck
        # workflow (the loads may target a later static deck) — not
        # judged.
        skipped.append((
            "OSM.LOADS_UNIMPORTED",
            "no run evidence — declared loads may target a later deck",
        ))
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
    patterns: tuple[Any, ...],
    findings: list[Finding],
    skipped: list[tuple[str, str]],
) -> None:
    if results is None:
        skipped.append(("RES.ZERO_U", "results= not passed"))
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

    message = f"Last-step displacement is exactly all-zero (stage={last_stage_id!r})."
    if not patterns:
        message += (
            " No pattern imports the declared loads — see "
            "OSM.LOADS_UNIMPORTED."
        )
    findings.append(_finding(
        "RES.ZERO_U",
        message,
        {"stage": last_stage_id, "n_patterns": len(patterns)},
    ))
