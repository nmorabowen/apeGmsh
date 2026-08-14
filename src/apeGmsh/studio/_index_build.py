"""Harvest public composite signatures into ``_api_index.json`` (ADR 0096 S2).

Walks session composites and ``apeSees`` namespaces via ``inspect`` +
``__init__`` AST. Does not grep ``src/`` at lookup time. Never dumps a
module: the index is a symbol → signature map.
"""
from __future__ import annotations

import ast
import inspect
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

INDEX_SCHEMA = 1
INDEX_PATH = Path(__file__).resolve().parent / "_api_index.json"
_MAX_DEPTH = 3

_SKILL_BY_PREFIX: tuple[tuple[str, str], ...] = (
    ("ops.fault", "references/opensees-bridge.md"),
    ("ops.damping", "references/opensees-bridge.md"),
    ("ops.stage", "references/opensees-bridge.md"),
    ("ops.", "references/opensees-bridge.md"),
    ("g.rebar", "references/rebar.md"),
    ("g.compose", "references/compose.md"),
    ("g.mesh.queries.get_fem_data", "references/fem-broker.md"),
    ("g.sections", "references/section-properties.md"),
)


def skill_for(symbol: str) -> str:
    for prefix, path in _SKILL_BY_PREFIX:
        if symbol.startswith(prefix):
            return path
    return "references/api-cheatsheet.md"


def index_path() -> Path:
    return INDEX_PATH


def build_index() -> dict[str, Any]:
    """Import public composites and return the index payload."""
    import os

    os.environ.setdefault("APEGMSH_QUIET", "1")
    entries: dict[str, dict[str, str]] = {}
    _walk_session(entries)
    _walk_apesees(entries)
    return {
        "schema": INDEX_SCHEMA,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entries": dict(sorted(entries.items())),
    }


def write_index(path: Path | None = None) -> Path:
    dest = path if path is not None else INDEX_PATH
    payload = build_index()
    dest.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return dest


def index_drift(committed: dict[str, Any], live: dict[str, Any]) -> list[str]:
    """Signature/skill/key drift. Ignores the ``generated`` timestamp."""
    c = committed.get("entries") or {}
    l = live.get("entries") or {}
    if not isinstance(c, dict) or not isinstance(l, dict):
        return ["entries are not a map"]
    out: list[str] = []
    missing = sorted(set(l) - set(c))
    extra = sorted(set(c) - set(l))
    if missing:
        out.append(f"{len(missing)} live-only (e.g. {', '.join(missing[:3])})")
    if extra:
        out.append(f"{len(extra)} committed-only (e.g. {', '.join(extra[:3])})")
    changed: list[str] = []
    for key in sorted(set(c) & set(l)):
        ce = c[key] if isinstance(c[key], dict) else {}
        le = l[key] if isinstance(l[key], dict) else {}
        if ce.get("signature") != le.get("signature") or ce.get("skill") != le.get(
            "skill"
        ):
            changed.append(key)
    if changed:
        out.append(
            f"{len(changed)} signature/skill drift (e.g. {', '.join(changed[:3])})"
        )
    return out


def committed_index_drift(path: Path | None = None) -> list[str]:
    """Rebuild from live classes and diff against the committed JSON."""
    dest = path if path is not None else INDEX_PATH
    if not dest.is_file():
        return [f"missing {dest}"]
    committed = json.loads(dest.read_text(encoding="utf-8"))
    return index_drift(committed, build_index())


def _walk_session(entries: dict[str, dict[str, str]]) -> None:
    from apeGmsh._core import apeGmsh as Session

    _harvest_class(Session, "g", entries, depth=0, skip_init_nested=True)
    for attr, mod, cls_name, _lazy in Session._COMPOSITES:
        if attr.startswith("_"):
            continue
        cls = _load_class(mod, cls_name)
        if cls is None:
            continue
        _harvest_class(cls, f"g.{attr}", entries, depth=0)


def _walk_apesees(entries: dict[str, dict[str, str]]) -> None:
    from apeGmsh.opensees.apesees import apeSees

    _harvest_class(apeSees, "ops", entries, depth=0)


def _load_class(mod: str, cls_name: str) -> type | None:
    import importlib

    package = "apeGmsh" + mod  # mod is ".core.Model"
    try:
        module = importlib.import_module(package)
    except Exception:
        return None
    return getattr(module, cls_name, None)


def _harvest_class(
    cls: type,
    prefix: str,
    entries: dict[str, dict[str, str]],
    *,
    depth: int,
    skip_init_nested: bool = False,
) -> None:
    for name, func in _public_callables(cls):
        symbol = f"{prefix}.{name}"
        entries[symbol] = _entry(symbol, func)
    if depth >= _MAX_DEPTH or skip_init_nested:
        return
    for attr, nested in _nested_classes(cls).items():
        _harvest_class(
            nested, f"{prefix}.{attr}", entries, depth=depth + 1,
        )


def _public_callables(cls: type) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for name, member in inspect.getmembers(cls):
        if name.startswith("_"):
            continue
        func = inspect.unwrap(member)
        if isinstance(member, (staticmethod, classmethod)):
            func = inspect.unwrap(member.__func__)
        elif not (inspect.isfunction(func) or inspect.ismethod(func)):
            continue
        mod = getattr(func, "__module__", "") or ""
        if not mod.startswith("apeGmsh"):
            continue
        out.append((name, func))
    return out


def _nested_classes(cls: type) -> dict[str, type]:
    try:
        src = textwrap.dedent(inspect.getsource(cls.__init__))
    except (OSError, TypeError):
        return {}
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {}
    mod = inspect.getmodule(cls)
    names: dict[str, type] = {}
    for node in ast.walk(tree):
        target: ast.AST | None = None
        value: ast.AST | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if target is None or value is None:
            continue
        if not (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and not target.attr.startswith("_")
        ):
            continue
        if not isinstance(value, ast.Call):
            continue
        resolved = _resolve_call_func(value.func, mod)
        if resolved is None:
            continue
        if not getattr(resolved, "__module__", "").startswith("apeGmsh"):
            continue
        names[target.attr] = resolved
    return names


def _resolve_call_func(func: ast.AST, mod: Any) -> type | None:
    if mod is None:
        return None
    if isinstance(func, ast.Name):
        obj = getattr(mod, func.id, None)
        return obj if isinstance(obj, type) else None
    if isinstance(func, ast.Attribute):
        # Only simple module.Name from this module's imports.
        if isinstance(func.value, ast.Name):
            obj = getattr(mod, func.attr, None)
            if isinstance(obj, type):
                return obj
            owner = getattr(mod, func.value.id, None)
            inner = getattr(owner, func.attr, None) if owner is not None else None
            return inner if isinstance(inner, type) else None
    return None


def _entry(symbol: str, func: Any) -> dict[str, str]:
    try:
        sig = inspect.signature(func)
        params = [
            p for p in sig.parameters.values()
            if p.name not in ("self", "cls")
        ]
        rendered = str(sig.replace(parameters=params))
    except (TypeError, ValueError):
        rendered = "(...)"
    short = symbol.rsplit(".", 1)[-1]
    doc = inspect.getdoc(func) or ""
    first = doc.strip().split("\n")[0].strip() if doc.strip() else ""
    return {
        "symbol": symbol,
        "signature": f"{short}{rendered}",
        "skill": skill_for(symbol),
        "doc": first[:200],
    }
