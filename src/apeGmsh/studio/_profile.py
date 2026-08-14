"""Token-budget profiler (ADR 0096 S4).

Sidecar, not an MCP tool. Two inputs: the owned
``.apegmsh/mcp_calls.jsonl`` ledger and an optional Cursor/agent
JSONL transcript. Heuristic tokens = chars/4. The defect metric is
``src_search`` rate.

Optional ``--skill-budget`` is a skill-hygiene gate (character caps),
not a live-session meter. ``--promote`` (S5) lists lookup-miss
eligibility and writes nothing.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

KINDS = (
    "skill_read",
    "index_lookup",
    "habitat_mcp",
    "src_search",
    "src_read",
    "other",
)

HABITAT_TOOLS = frozenset({
    "status",
    "get_selection",
    "run_until",
    "assess",
    "render",
    "animate",
    "results_pin",
    "emit_report",
    "highlight",
    "promote_selection",
})

SKILL_MD_MAX_CHARS = 24_000
REFERENCE_MAX_CHARS = 200_000
MISS_PROMOTE_REPEATS = 3


def heuristic_tokens(chars: int) -> int:
    return chars // 4


def _norm(path: str) -> str:
    return path.replace("\\", "/").lower()


def _is_src(path: str) -> bool:
    p = _norm(path)
    return "/src/apegmsh/" in f"/{p}" or p.startswith("src/apegmsh")


def _is_skill(path: str) -> bool:
    p = _norm(path)
    return (
        "/skills/apegmsh/" in f"/{p}"
        or "/.claude/skills/apegmsh" in f"/{p}"
        or p.endswith("skills/apegmsh/skill.md")
    )


def _path_from(payload: Any) -> str:
    if not isinstance(payload, dict):
        return str(payload or "")
    for key in ("path", "target_directory", "file", "uri"):
        val = payload.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


def classify_tool(name: str, payload: Any) -> str:
    """One tool event → kind. Unknown shapes become ``other``."""
    n = (name or "").strip()
    nlow = n.lower()
    blob = payload if isinstance(payload, dict) else {}
    path = _path_from(blob)
    joined = json.dumps(blob, default=str) if blob else str(payload or "")

    if nlow in {"callmcptool", "call_mcp_tool"}:
        tool = str(blob.get("toolName") or blob.get("tool") or "")
        server = str(blob.get("server") or "")
        if tool == "lookup":
            return "index_lookup"
        if tool in HABITAT_TOOLS or "apegmsh-studio" in server:
            return "habitat_mcp"
        return "other"
    if n == "lookup" or nlow == "lookup":
        return "index_lookup"
    if n in HABITAT_TOOLS:
        return "habitat_mcp"
    if "apegmsh.studio.lookup" in joined.lower():
        return "index_lookup"
    if nlow in {"grep", "glob"} and (_is_src(path) or _is_src(joined)):
        return "src_search"
    if nlow == "read" and _is_skill(path):
        return "skill_read"
    if nlow == "read" and _is_src(path):
        return "src_read"
    return "other"


def _empty_counts() -> dict[str, int]:
    return {kind: 0 for kind in KINDS}


def _empty_chars() -> dict[str, int]:
    return {kind: 0 for kind in KINDS}


def _add_event(
    counts: dict[str, int],
    chars: dict[str, int],
    kind: str,
    nbytes: int,
) -> None:
    counts[kind] = counts.get(kind, 0) + 1
    chars[kind] = chars.get(kind, 0) + max(nbytes, 0)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def summarize_mcp_calls(path: Path) -> tuple[dict[str, int], dict[str, int]]:
    counts = _empty_counts()
    chars = _empty_chars()
    for rec in iter_jsonl(path):
        tool = str(rec.get("tool") or "")
        kind = classify_tool(tool, rec)
        nbytes = int(rec.get("bytes_in") or 0) + int(rec.get("bytes_out") or 0)
        _add_event(counts, chars, kind, nbytes)
    return counts, chars


def _tool_events_from_record(rec: dict[str, Any]) -> list[tuple[str, Any]]:
    """Extract (name, payload) pairs from one transcript JSONL object."""
    if rec.get("type") == "tool_use" and rec.get("name"):
        return [(str(rec["name"]), rec.get("input") or rec.get("arguments") or {})]
    msg = rec.get("message")
    content = None
    if isinstance(msg, dict):
        content = msg.get("content")
    elif isinstance(rec.get("content"), list):
        content = rec["content"]
    if not isinstance(content, list):
        if rec.get("name") and rec.get("input") is not None:
            return [(str(rec["name"]), rec.get("input"))]
        return []
    out: list[tuple[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use" and block.get("name"):
            out.append(
                (str(block["name"]), block.get("input") or block.get("arguments") or {})
            )
    return out


def summarize_transcript(path: Path) -> tuple[dict[str, int], dict[str, int]]:
    counts = _empty_counts()
    chars = _empty_chars()
    for rec in iter_jsonl(path):
        for name, payload in _tool_events_from_record(rec):
            kind = classify_tool(name, payload)
            try:
                nbytes = len(json.dumps(payload, default=str, separators=(",", ":")))
            except (TypeError, ValueError):
                nbytes = len(str(payload))
            _add_event(counts, chars, kind, nbytes)
    return counts, chars


def merge_stats(
    *parts: tuple[dict[str, int], dict[str, int]],
) -> tuple[dict[str, int], dict[str, int]]:
    counts = _empty_counts()
    chars = _empty_chars()
    for c, ch in parts:
        for kind in KINDS:
            counts[kind] += int(c.get(kind) or 0)
            chars[kind] += int(ch.get(kind) or 0)
    return counts, chars


def src_search_rate(counts: dict[str, int]) -> float:
    total = sum(int(counts.get(k) or 0) for k in KINDS)
    if total == 0:
        return 0.0
    return int(counts.get("src_search") or 0) / total


def build_report(
    *,
    mcp_path: Path | None = None,
    transcript_path: Path | None = None,
) -> dict[str, Any]:
    parts: list[tuple[dict[str, int], dict[str, int]]] = []
    if mcp_path is not None:
        parts.append(summarize_mcp_calls(mcp_path))
    if transcript_path is not None:
        parts.append(summarize_transcript(transcript_path))
    if not parts:
        counts, chars = _empty_counts(), _empty_chars()
    else:
        counts, chars = merge_stats(*parts)
    tokens = {k: heuristic_tokens(chars[k]) for k in KINDS}
    total = sum(counts.values())
    return {
        "schema": 1,
        "events": total,
        "counts": counts,
        "chars": chars,
        "tokens": tokens,
        "src_search_rate": round(src_search_rate(counts), 4),
    }


def format_report(report: dict[str, Any]) -> str:
    counts = report.get("counts") or {}
    tokens = report.get("tokens") or {}
    lines = [
        f"events: {report.get('events') or 0}",
        f"src_search_rate: {report.get('src_search_rate') or 0}",
    ]
    for kind in KINDS:
        n = int(counts.get(kind) or 0)
        tok = int(tokens.get(kind) or 0)
        lines.append(f"{kind}: {n}  (~{tok} tok)")
    return "\n".join(lines) + "\n"


def skill_budget_violations(root: Path) -> list[str]:
    """Character-cap gate for ``skills/apegmsh/``. Empty → pass."""
    tree = root / "skills" / "apegmsh"
    skill_md = tree / "SKILL.md"
    out: list[str] = []
    if not skill_md.is_file():
        return [f"missing {skill_md}"]
    n = len(skill_md.read_text(encoding="utf-8"))
    if n > SKILL_MD_MAX_CHARS:
        out.append(f"SKILL.md {n} chars > {SKILL_MD_MAX_CHARS}")
    ref_dir = tree / "references"
    if ref_dir.is_dir():
        for path in sorted(ref_dir.glob("*.md")):
            m = len(path.read_text(encoding="utf-8"))
            if m > REFERENCE_MAX_CHARS:
                out.append(f"{path.name} {m} chars > {REFERENCE_MAX_CHARS}")
    return out


def promote_report(mcp_path: Path | None) -> dict[str, Any]:
    """Eligibility for a reviewed skill/index PR. Writes nothing.

    Counts MCP-logged ``lookup`` records with ``kind=miss`` only.
    Ambiguous hits, CLI lookup, and transcript ``src_search`` are
    out of scope (ADR 0096 S5).
    """
    misses: Counter[str] = Counter()
    if mcp_path is not None and mcp_path.is_file():
        for rec in iter_jsonl(mcp_path):
            if rec.get("tool") != "lookup" or rec.get("kind") != "miss":
                continue
            symbol = rec.get("symbol")
            if isinstance(symbol, str) and symbol.strip():
                misses[symbol.strip()] += 1
    eligible = [
        {"symbol": symbol, "n": n}
        for symbol, n in misses.most_common()
        if n >= MISS_PROMOTE_REPEATS
    ]
    observed = [
        {"symbol": symbol, "n": n}
        for symbol, n in misses.most_common()
        if n < MISS_PROMOTE_REPEATS
    ]
    return {
        "schema": 1,
        "writes": False,
        "bar": MISS_PROMOTE_REPEATS,
        "scope": "MCP lookup kind=miss in model-cwd .apegmsh/mcp_calls.jsonl",
        "eligible_misses": eligible,
        "observed_misses": observed,
        "skill_error": (
            "One confirmed conflict with the generated index or a "
            "# verified: test owes a reviewed PR against skills/apegmsh/. "
            "Do not write skills/ or run sync_skill.py in-session; "
            "humans derive after merge. Do not wait for repetition."
        ),
        "working_steps": (
            "Comments in that .py, and/or a non-normative Recommendation "
            "in skills/apegmsh/references/workflows.md. Never a required "
            "pipeline in SKILL.md."
        ),
        "canonical": "skills/apegmsh/",
        "derive": "scripts/sync_skill.py (humans, after merge)",
        "forbidden": [
            "write skills/apegmsh/ in-session",
            "run scripts/sync_skill.py write path in-session",
            "hand-edit .claude/skills/apegmsh-helper/",
            "MCP CAD verb",
            "fix_skill",
            "remember_steps",
        ],
    }


def format_promote_report(report: dict[str, Any]) -> str:
    lines = [
        "ADR 0096 S5 promotion (observe only; writes=false)",
        f"scope: {report.get('scope')}",
        f"bar: {report.get('bar')} repeated MCP lookup misses (kind=miss)",
        "eligible misses:",
    ]
    eligible = report.get("eligible_misses") or []
    if not eligible:
        lines.append("  (none)")
    else:
        for row in eligible:
            lines.append(f"  {row['symbol']}  n={row['n']}")
    lines.append("observed (below bar):")
    observed = report.get("observed_misses") or []
    if not observed:
        lines.append("  (none)")
    else:
        for row in observed:
            lines.append(f"  {row['symbol']}  n={row['n']}")
    lines.append(f"skill error: {report.get('skill_error')}")
    lines.append(f"working steps: {report.get('working_steps')}")
    lines.append("forbidden: " + "; ".join(report.get("forbidden") or []))
    return "\n".join(lines) + "\n"
