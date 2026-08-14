"""Markdown report text for :class:`AssessmentReport`."""
from __future__ import annotations

from pathlib import Path

from ._catalog import FAIL_RESERVED
from ._types import Finding

_MAX_LISTED = 24
NO_STILLS = "no stills; numbers only"
_FAIL_ORDER = (
    "MODEL.EMPTY",
    "MESH.INVERTED",
    "RES.UNBOUND_FEM",
    "RES.NO_STAGE",
    "RES.NAN",
    "RES.INF",
)


def fail_reserved_unevaluated(
    skipped: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    """FAIL-reserved codes whose *check did not run* (INV-3, Amendment 1).

    Union-merge fill notes reuse ``RES.NAN`` as the skip key but the
    check did run — those reasons are coverage, not unevaluated.
    """
    keys: set[str] = set()
    for code, reason in skipped:
        if str(reason).startswith("union-merge fill"):
            continue
        keys.add(code)
    family_mesh = "MESH.*" in keys
    out: list[str] = []
    for code in _FAIL_ORDER:
        if code not in FAIL_RESERVED:
            continue
        if code in keys or (family_mesh and code.startswith("MESH.")):
            out.append(code)
    return tuple(out)


def render_text(
    findings: tuple[Finding, ...],
    skipped: tuple[tuple[str, str], ...],
    *,
    source: str,
    figures: tuple[Path, ...] = (),
    figure_note: str | None = None,
) -> str:
    """Build the agent-facing markdown (target ~40–80 lines when busy)."""
    errors = [f for f in findings if f.severity == "error"]
    warnings = [f for f in findings if f.severity == "warning"]
    infos = [f for f in findings if f.severity == "info"]
    uneval = fail_reserved_unevaluated(skipped)

    lines: list[str] = [
        f"# Assessment ({source})",
        "",
        (
            f"**Verdict:** {len(errors)} error(s), "
            f"{len(warnings)} warning(s), {len(infos)} info"
        ),
        "**Scope:** last recorded step only (INV-9).",
    ]
    if uneval:
        named = ", ".join(f"`{c}`" for c in uneval)
        lines.append(f"**Not evaluated (FAIL-reserved):** {named}.")
    lines.append("")

    if not findings:
        lines.append("No findings.")
        lines.append("")
    else:
        for title, group in (
            ("Errors", errors),
            ("Warnings", warnings),
            ("Info", infos),
        ):
            if not group:
                continue
            lines.append(f"## {title}")
            lines.append("")
            shown = group[:_MAX_LISTED]
            for finding in shown:
                lines.append(f"- `{finding.code}` — {finding.message}")
            if len(group) > _MAX_LISTED:
                lines.append(f"- … +{len(group) - _MAX_LISTED} more")
            lines.append("")
            _append_extrema(lines, shown)

    lines.append("## Skipped checks")
    lines.append("")
    if not skipped:
        lines.append("None.")
        lines.append("")
    else:
        for code, reason in skipped:
            lines.append(f"- `{code}` — {reason}")
        lines.append("")

    if figure_note is not None or figures:
        lines.append("## Figures")
        lines.append("")
        if figure_note is not None:
            lines.append(figure_note)
            lines.append("")
        else:
            for path in figures:
                lines.append(f"- `{path}`")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _append_extrema(lines: list[str], findings: list[Finding]) -> None:
    """Top-N extrema appendix only — never a full table dump."""
    rows: list[str] = []
    for finding in findings:
        detail = finding.detail or {}
        extrema = detail.get("extrema")
        if not isinstance(extrema, (list, tuple)):
            continue
        for item in extrema:
            if not isinstance(item, dict):
                continue
            rows.append(
                "  - node {id} xyz={xyz} value={value} step={step}".format(
                    id=item.get("id"),
                    xyz=item.get("xyz"),
                    value=item.get("value"),
                    step=item.get("step"),
                )
            )
    if not rows:
        return
    lines.append("Extrema (top-N):")
    lines.extend(rows)
    lines.append("")
