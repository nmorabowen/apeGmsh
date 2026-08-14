"""Resolve a public symbol from the generated API index (ADR 0096 S2)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._index_build import INDEX_PATH, skill_for

_MAX_HITS = 8
_MAX_LINES = 20


def load_index(path: Path | None = None) -> dict[str, Any]:
    dest = path if path is not None else INDEX_PATH
    if not dest.is_file():
        raise FileNotFoundError(
            f"API index missing: {dest}. Run "
            f"`python -m apeGmsh.studio.lookup --build`."
        )
    return json.loads(dest.read_text(encoding="utf-8"))


def match_symbols(query: str, entries: dict[str, Any]) -> list[str]:
    q = query.strip().rstrip("()")
    if q.startswith("g."):
        q = q[2:]
    hits: list[str] = []
    for symbol in entries:
        s = symbol[2:] if symbol.startswith("g.") else symbol
        if s == q or symbol == query.strip() or s.endswith("." + q):
            hits.append(symbol)
    return hits


def format_entry(entry: dict[str, str]) -> str:
    symbol = entry.get("symbol") or ""
    signature = entry.get("signature") or ""
    skill = entry.get("skill") or skill_for(symbol)
    doc = (entry.get("doc") or "").strip()
    lines = [
        symbol,
        signature,
        "",
        skill,
    ]
    if doc:
        lines.extend(["", doc])
    text = "\n".join(lines).rstrip() + "\n"
    clipped = "\n".join(text.splitlines()[:_MAX_LINES])
    if not clipped.endswith("\n"):
        clipped += "\n"
    return clipped


def format_miss(query: str) -> str:
    skill = skill_for(query if query.startswith(("g.", "ops.")) else "g.")
    return (
        f"miss: {query!r}\n"
        f"No index hit. Read {skill} (matching heading). "
        f"`src/` grep is for maintaining apeGmsh, not authoring a model "
        f"(ADR 0096).\n"
    )


def format_ambiguous(
    query: str,
    hits: list[str],
    entries: dict[str, Any] | None = None,
) -> str:
    shown = hits[:_MAX_HITS]
    extra = len(hits) - len(shown)
    lines = [f"ambiguous: {query!r} matches {len(hits)} symbols:"]
    recs = entries or {}
    for hit in shown:
        rec = recs.get(hit) if isinstance(recs.get(hit), dict) else {}
        sig = (rec.get("signature") or "").strip() if rec else ""
        lines.append(f"  {hit}")
        if sig:
            lines.append(f"    {sig}")
    if extra > 0:
        lines.append(f"  … +{extra} more")
    lines.append("Pass a more specific symbol (ADR 0096: no module dump).")
    return "\n".join(lines[:_MAX_LINES]) + "\n"


def lookup(query: str, *, index: dict[str, Any] | None = None) -> tuple[str, int]:
    """Return (text, exit_code). 0 hit, 2 miss/ambiguous, 1 missing index."""
    try:
        payload = index if index is not None else load_index()
    except FileNotFoundError as exc:
        return (f"error: {exc}\n", 1)
    entries = payload.get("entries") or {}
    if not isinstance(entries, dict):
        return ("error: index entries are not a map\n", 1)
    hits = match_symbols(query, entries)
    if not hits:
        return (format_miss(query), 2)
    if len(hits) > 1:
        return (format_ambiguous(query, hits, entries), 2)
    return (format_entry(entries[hits[0]]), 0)
