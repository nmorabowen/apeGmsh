"""Markdown report text for :class:`AssessmentReport`."""
from __future__ import annotations

from ._types import Finding

_MAX_LISTED = 24


def render_text(
    findings: tuple[Finding, ...],
    skipped: tuple[tuple[str, str], ...],
    *,
    source: str,
) -> str:
    """Build the agent-facing markdown (target ~40–80 lines when busy)."""
    errors = [f for f in findings if f.severity == "error"]
    warnings = [f for f in findings if f.severity == "warning"]
    infos = [f for f in findings if f.severity == "info"]

    lines: list[str] = [
        f"# Assessment ({source})",
        "",
        (
            f"**Verdict:** {len(errors)} error(s), "
            f"{len(warnings)} warning(s), {len(infos)} info"
        ),
        "",
    ]

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
