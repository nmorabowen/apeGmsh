"""ReportBundle + pin + markdown emit (ADR 0095 S4c).

Schema 1. Assembles names / pick / last run / visors / pinned paths.
Markdown is the canonical skin (git-tracked ``docs/``). HTML and canvas
are later skins of the same bundle. Authored output must not land in
``.apegmsh/`` (INV-12).

Pure: no ``gmsh``, no Qt, no viewers.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._ledger import append_run, read_runs
from ._paths import ledger_path, visors_path
from ._status import collect_status

BUNDLE_SCHEMA = 1
EMIT_FORMATS = ("markdown", "html", "canvas")
EMIT_SHIPPED = frozenset({"markdown"})
_VISOR_SUFFIX = frozenset({".png", ".gif", ".mp4", ".jpg", ".jpeg", ".webp"})
_DEFAULT_CHAPTER = Path("docs") / "studio-report.md"


def collect_bundle(
    *,
    root: Path | str | None = None,
    pin: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Frozen ReportBundle (schema 1) from habitat files + optional pin."""
    base = Path.cwd() if root is None else Path(root)
    status = collect_status(base)
    pin_rec = pin if pin is not None else _last_pin(base)
    visors = list_visors(base)
    if pin_rec is not None and pin_rec.get("visors"):
        visors = [str(p) for p in pin_rec["visors"]]
    return {
        "schema": BUNDLE_SCHEMA,
        "root": str(base),
        "names": status.get("names"),
        "selection": status.get("selection"),
        "last_run": status.get("last_run"),
        "pin": pin_rec,
        "visors": visors,
        "assess": (pin_rec or {}).get("assess"),
    }


def results_pin(
    model_h5: str | Path | None = None,
    results: str | Path | None = None,
    *,
    assess: dict[str, Any] | None = None,
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Append a pin line to ``runs.jsonl``. Records paths + hashes, no copy."""
    base = Path.cwd() if root is None else Path(root)
    model_path = _optional_file(model_h5, base)
    results_path = _optional_file(results, base)
    if model_path is None and results_path is None:
        raise ValueError("results_pin needs model_h5= and/or results=")
    visors = list_visors(base)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stamp = ts.replace("-", "").replace(":", "")
    pin_id = f"pin-{stamp}"
    record = {
        "schema": BUNDLE_SCHEMA,
        "kind": "pin",
        "id": pin_id,
        "ts": ts,
        "model_h5": _file_stamp(model_path),
        "results": _file_stamp(results_path),
        "visors": visors,
        "assess": assess,
        "chapter": None,
    }
    append_run(ledger_path(base), record)
    return {
        "ok": True,
        "id": pin_id,
        "path": str(ledger_path(base)),
        "pin": record,
    }


def emit_report(
    *,
    format: str = "markdown",
    output: str | Path | None = None,
    pin_id: str | None = None,
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Write the canonical Markdown chapter under ``docs/`` (INV-12)."""
    if format not in EMIT_FORMATS:
        raise ValueError(
            f"format must be one of {EMIT_FORMATS}; got {format!r}"
        )
    if format not in EMIT_SHIPPED:
        raise ValueError(
            f"emit_report format={format!r} is not shipped "
            "(S4c is markdown only; html/canvas are later skins)"
        )
    base = Path.cwd() if root is None else Path(root)
    pin = _pin_by_id(base, pin_id) if pin_id else _last_pin(base)
    bundle = collect_bundle(root=base, pin=pin)
    dest = Path(output) if output is not None else base / _DEFAULT_CHAPTER
    if not dest.is_absolute():
        dest = (base / dest).resolve()
    else:
        dest = dest.resolve()
    _refuse_generated(dest, base)
    fig_rel: list[tuple[str, str]] = []
    slug = (pin or {}).get("id") or "studio"
    copied = _promote_visors(bundle.get("visors") or [], base, slug)
    for src, dest_fig in copied:
        fig_rel.append((src, Path(os.path.relpath(dest_fig, dest.parent)).as_posix()))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(_render_markdown(bundle, figures=fig_rel), encoding="utf-8")
    return {
        "ok": True,
        "format": format,
        "path": str(dest),
        "pin_id": (pin or {}).get("id"),
        "figures": [str(p) for _, p in copied],
        "bundle_schema": BUNDLE_SCHEMA,
    }


def list_visors(root: Path) -> list[str]:
    folder = visors_path(root)
    if not folder.is_dir():
        return []
    out: list[str] = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in _VISOR_SUFFIX:
            out.append(str(path.resolve()))
    return out


def _last_pin(root: Path) -> dict[str, Any] | None:
    rows = [r for r in read_runs(ledger_path(root)) if r.get("kind") == "pin"]
    return rows[-1] if rows else None


def _pin_by_id(root: Path, pin_id: str) -> dict[str, Any]:
    for row in reversed(read_runs(ledger_path(root))):
        if row.get("kind") == "pin" and row.get("id") == pin_id:
            return row
    raise ValueError(f"no pin with id {pin_id!r}")


def _optional_file(path: str | Path | None, base: Path) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = (base / p).resolve()
    else:
        p = p.resolve()
    if not p.is_file():
        raise ValueError(f"file not found: {p}")
    return p


def _file_stamp(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "sha256": digest,
        "size": path.stat().st_size,
    }


def _refuse_generated(dest: Path, root: Path) -> None:
    generated = (root / ".apegmsh").resolve()
    try:
        dest.relative_to(generated)
    except ValueError:
        return
    raise ValueError(
        "authored chapter must not live in .apegmsh/ (INV-12); "
        "write it under docs/"
    )


def _promote_visors(
    visors: list[str],
    root: Path,
    slug: str,
) -> list[tuple[str, Path]]:
    """Copy working visors to git-tracked ``docs/figures/<slug>/``."""
    dest_root = root / "docs" / "figures" / slug
    copied: list[tuple[str, Path]] = []
    visor_home = visors_path(root).resolve()
    for raw in visors:
        src = Path(raw)
        if not src.is_file():
            continue
        try:
            rel = src.resolve().relative_to(visor_home)
        except ValueError:
            rel = Path(src.name)
        dest = dest_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied.append((src.name, dest))
    return copied


def _render_markdown(
    bundle: dict[str, Any],
    *,
    figures: list[tuple[str, str]],
) -> str:
    names = bundle.get("names") or {}
    sel = bundle.get("selection") or {}
    last = bundle.get("last_run") or {}
    pin = bundle.get("pin") or {}
    assess = bundle.get("assess") or {}
    lines = [
        "# Studio report",
        "",
        f"ReportBundle schema {bundle.get('schema', BUNDLE_SCHEMA)}. "
        "Generated from habitat files; not a substitute for the script.",
        "",
    ]
    if pin:
        lines.extend([
            f"- **pin:** `{pin.get('id')}`  ({pin.get('ts') or ''})".rstrip(),
            "",
        ])
        model = pin.get("model_h5") or {}
        results = pin.get("results") or {}
        if model.get("path"):
            lines.append(
                f"- **model.h5:** `{model['path']}`  "
                f"(sha256 `{model.get('sha256', '')[:12]}`)"
            )
        if results.get("path"):
            lines.append(
                f"- **results:** `{results['path']}`  "
                f"(sha256 `{results.get('sha256', '')[:12]}`)"
            )
        lines.append("")
    lines.extend([
        "## Names",
        "",
        f"- **phase:** {names.get('phase') or last.get('phase') or '(none)'}",
        f"- **labels:** {_csv(names.get('labels') or last.get('labels'))}",
        f"- **physical groups:** {_csv(names.get('physical_groups') or last.get('physical_groups'))}",
        f"- **entities:** {_counts((names.get('counts') or last.get('counts') or {}).get('entities'))}",
        f"- **elements:** {_counts((names.get('counts') or last.get('counts') or {}).get('elements'))}",
        "",
        "## Pick",
        "",
    ])
    if not sel:
        lines.append("(none)")
    else:
        lines.extend([
            f"- **phase:** {sel.get('phase') or '(none)'}",
            f"- **labels:** {_csv(sel.get('labels'))}",
            f"- **physical groups:** {_csv(sel.get('physical_groups'))}",
            f"- **unnamed:** {len(sel.get('unnamed') or [])}",
        ])
    lines.extend(["", "## Last run", ""])
    if not last:
        lines.append("(none)")
    else:
        lines.extend([
            f"- **ts:** {last.get('ts') or '(none)'}",
            f"- **script:** `{last.get('script') or '(none)'}`",
            f"- **phase:** {last.get('phase') or '(none)'}",
            f"- **ok:** {last.get('ok')}",
            f"- **hash:** `{last.get('hash') or '(none)'}`",
        ])
    lines.extend(["", "## Assess", ""])
    text = assess.get("text") if isinstance(assess, dict) else None
    if text:
        lines.append(str(text).rstrip())
    elif assess:
        findings = assess.get("findings") or []
        lines.append(f"{len(findings)} finding(s).")
        for row in findings[:24]:
            if isinstance(row, dict):
                lines.append(
                    f"- `{row.get('code')}` ({row.get('severity')}): "
                    f"{row.get('message')}"
                )
    else:
        lines.append("(none — run assess first)")
    lines.extend(["", "## Figures", ""])
    if not figures:
        lines.append("(none — working visors stay under `.apegmsh/visors/`)")
    else:
        for caption, rel in figures:
            lines.append(f"![{caption}]({rel})")
            lines.append("")
    lines.append("")
    return "\n".join(lines)


def _csv(items: Any) -> str:
    seq = [str(x) for x in (items or []) if x]
    return ", ".join(seq) if seq else "(none)"


def _counts(counts: Any) -> str:
    if not isinstance(counts, dict) or not counts:
        return "(none)"
    return " ".join(f"{k}={v}" for k, v in counts.items())
