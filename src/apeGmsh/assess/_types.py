"""Frozen assess records — same flavour as ``cuts._preflight``."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Mapping

if TYPE_CHECKING:
    from apeGmsh.opensees._internal.lineage import Lineage


Severity = Literal["error", "warning", "info"]


@dataclass(frozen=True)
class Finding:
    """One catalog verdict.

    ``detail`` carries where/evidence (pg, ids[:K], xyz, step, counts).
    It does not carry Python snippets to exec.
    """

    code: str
    severity: Severity
    message: str
    detail: Mapping[str, object] | None = None


@dataclass(frozen=True)
class AssessmentReport:
    """Result of ``fem.assess()`` / ``results.assess()``.

    ``text`` is what the agent prints. ``findings`` is what it branches
    on for emitted verdicts. ``skipped`` is ``(code, reason)`` pairs
    for checks that did not run (INV-3) — branch here to see that a
    FAIL-reserved check was never evaluated. ``figures`` is empty
    unless ``figures=True`` wrote stills. ``lineage`` is the Results
    object on the results path; ``None`` on ``fem.assess()``.
    """

    findings: tuple[Finding, ...]
    text: str
    figures: tuple[Path, ...] = ()
    lineage: Lineage | None = None
    skipped: tuple[tuple[str, str], ...] = ()
