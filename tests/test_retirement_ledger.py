"""Lock: the retirement ledger is dated, alive and not overdue (X4-a, #1806).

Panel rule: every retirement ends in deletion, with a dated CI expiry. The
ledger is ``internal_docs/program/retirement_ledger.toml``; its header
documents the row format. This lock fails, closed, when:

* a row is malformed or undated (#1199 X4 ruling 1: every row is dated);
* any target of a row no longer exists: delete the row with the thing;
* a row's date (``expires`` or ``review_by``) has passed while any of its
  targets still exists: delete the thing, or the maintainer re-dates the row.

The ledger is a ``scripts/land_pr.py`` BASELINE file, so a PR that adds,
re-dates or removes a row needs the ``human-gate`` label.
"""
from __future__ import annotations

import ast
import datetime as dt
import re
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "internal_docs" / "program" / "retirement_ledger.toml"

KINDS = frozenset({
    "keep", "gap", "inventory", "spin-out", "prototypes", "facade",
    "shadow", "xfail", "gate",
})
BASES = frozenset({"ruling", "policy"})
TARGET_KEYS = ("paths", "symbols", "texts")
KEYS = frozenset({"id", "kind", "trigger", "review_by", "expires", "basis",
                  "source", *TARGET_KEYS})
ID_RE = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")


def _is_date(value: Any) -> bool:
    # A TOML datetime is a dt.datetime, itself a dt.date subclass: refuse it.
    return isinstance(value, dt.date) and not isinstance(value, dt.datetime)


def _safe_rel(path: str) -> bool:
    p = PurePosixPath(path.rstrip("/"))
    return (bool(path) and "\\" not in path and not p.is_absolute()
            and ".." not in p.parts and ":" not in p.parts[0])


def malformed(rows: Any) -> list[str]:
    """Every format error in ``rows`` (the parsed ``row`` array)."""
    if not isinstance(rows, list) or not rows:
        return ["the ledger has no [[row]] entries"]
    errors: list[str] = []
    seen: set[str] = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"row {i}: not a table")
            continue
        rid = row.get("id")
        where = f"row {i} ({rid!r})"
        if not isinstance(rid, str) or not ID_RE.fullmatch(rid):
            errors.append(f"{where}: id must be lowercase kebab-case")
        elif rid in seen:
            errors.append(f"{where}: duplicate id")
        else:
            seen.add(rid)
        unknown = sorted(set(row) - KEYS)
        if unknown:
            errors.append(f"{where}: unknown keys {unknown}")
        if row.get("kind") not in KINDS:
            errors.append(f"{where}: kind must be one of {sorted(KINDS)}")
        if row.get("basis") not in BASES:
            errors.append(f"{where}: basis must be one of {sorted(BASES)}")
        if not isinstance(row.get("source"), str) or not row["source"].strip():
            errors.append(f"{where}: source must name where the date comes from")
        has_exp = "expires" in row
        has_trig = "trigger" in row or "review_by" in row
        if has_exp == has_trig:
            errors.append(f"{where}: give exactly one of expires, or trigger + review_by")
        elif has_exp and not _is_date(row["expires"]):
            errors.append(f"{where}: expires must be a bare TOML date (YYYY-MM-DD)")
        elif has_trig:
            if not isinstance(row.get("trigger"), str) or not row["trigger"].strip():
                errors.append(f"{where}: a trigger row needs a non-empty trigger")
            if not _is_date(row.get("review_by")):
                errors.append(f"{where}: review_by must be a bare TOML date (YYYY-MM-DD)")
        if not any(k in row for k in TARGET_KEYS):
            errors.append(f"{where}: no target (paths, symbols or texts)")
        for key in TARGET_KEYS:
            if key not in row:
                continue
            vals = row[key]
            if not isinstance(vals, list) or not vals or not all(
                    isinstance(v, str) for v in vals):
                errors.append(f"{where}: {key} must be a non-empty list of strings")
                continue
            for v in vals:
                path = v if key == "paths" else v.partition("::")[0]
                if key != "paths" and ("::" not in v or not v.partition("::")[2]):
                    errors.append(f"{where}: {key} entry {v!r} must be 'file::{key[:-1]}'")
                elif not _safe_rel(path):
                    errors.append(f"{where}: {v!r} is not a repo-relative posix path")
    return errors


def _defines(body: list[ast.stmt], name: str) -> ast.stmt | None:
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                return node
        elif isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                return node
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                return node
    return None


def _symbol_exists(file: Path, dotted: str) -> bool:
    if not file.is_file():
        return False
    body: list[ast.stmt] = ast.parse(file.read_text(encoding="utf-8")).body
    parts = dotted.split(".")
    for i, part in enumerate(parts):
        node = _defines(body, part)
        if node is None:
            return False
        if i < len(parts) - 1:
            if not isinstance(node, ast.ClassDef):
                return False
            body = node.body
    return True


def missing_targets(row: dict[str, Any], root: Path) -> list[str]:
    """The targets of ``row`` that no longer exist under ``root``."""
    gone: list[str] = []
    for path in row.get("paths", []):
        if not (root / path).exists():
            gone.append(path)
    for entry in row.get("symbols", []):
        file, _, dotted = entry.partition("::")
        if not _symbol_exists(root / file, dotted):
            gone.append(entry)
    for entry in row.get("texts", []):
        file, _, text = entry.partition("::")
        f = root / file
        if not f.is_file() or text not in f.read_text(encoding="utf-8"):
            gone.append(entry)
    return gone


def problems(rows: Any, root: Path, today: dt.date) -> list[str]:
    """Every reason the ledger ``rows`` fail against the tree at ``root``."""
    errors = malformed(rows)
    if errors:
        return errors
    for row in rows:
        rid = row["id"]
        gone = missing_targets(row, root)
        if gone:
            errors.append(
                f"{rid}: {gone} no longer exist. Delete the row with the thing "
                "(or drop just those targets), in the same PR")
            continue
        due = row.get("expires") or row["review_by"]
        if due < today:
            what = "expired" if "expires" in row else "passed its review_by date"
            errors.append(
                f"{rid}: {what} on {due.isoformat()} and its targets still exist. "
                "Delete them, or the maintainer re-dates the row (human-gate)")
    return errors


def ledger_problems(doc: dict[str, Any], root: Path, today: dt.date) -> list[str]:
    """``problems`` for a parsed ledger file. Any top-level key but ``row`` fails:
    a ``[[rows]]`` or ``[row]`` typo would otherwise drop its rows silently."""
    stray = sorted(set(doc) - {"row"})
    if stray:
        return [f"unknown top-level keys {stray}: every row is a [[row]] table"]
    return problems(doc.get("row"), root, today)


def load_doc(path: Path = LEDGER) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def test_ledger_is_well_formed_alive_and_not_overdue() -> None:
    errors = ledger_problems(load_doc(), ROOT, dt.date.today())
    assert not errors, "retirement ledger:\n  " + "\n  ".join(errors)


# --- self-tests: the lock fails on each failure mode, and passes otherwise ---

TODAY = dt.date(2026, 10, 12)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(
        "class Box:\n    split: bool\n    def go(self):\n        pass\n"
        "TABLE = {}\n_PATHS = frozenset({'flat', 'split'})\n",
        encoding="utf-8")
    return tmp_path


def _row(**over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": "thing", "kind": "facade", "paths": ["pkg/mod.py"],
        "expires": dt.date(2026, 12, 1), "basis": "policy", "source": "test",
    }
    row.update(over)
    return {k: v for k, v in row.items() if v is not None}


def test_selftest_future_dated_live_rows_pass(tree: Path) -> None:
    rows = [
        _row(),
        _row(id="trig", expires=None, trigger="V4",
             review_by=dt.date(2027, 1, 1), paths=["pkg/"],
             symbols=["pkg/mod.py::Box.split", "pkg/mod.py::Box.go",
                      "pkg/mod.py::TABLE"],
             texts=["pkg/mod.py::'split'"]),
    ]
    assert problems(rows, tree, TODAY) == []


def test_selftest_past_dated_row_with_existing_path_fails(tree: Path) -> None:
    errors = problems([_row(expires=dt.date(2026, 10, 11))], tree, TODAY)
    assert len(errors) == 1 and "expired on 2026-10-11" in errors[0]


def test_selftest_past_review_by_with_existing_symbol_fails(tree: Path) -> None:
    row = _row(expires=None, trigger="V4", review_by=dt.date(2026, 1, 1),
               paths=None, symbols=["pkg/mod.py::Box.split"])
    errors = problems([row], tree, TODAY)
    assert len(errors) == 1 and "passed its review_by date" in errors[0]


@pytest.mark.parametrize("target", [
    {"paths": ["pkg/gone.py"]},
    {"paths": ["gone/"]},
    {"symbols": ["pkg/mod.py::Box.merge"]},
    {"symbols": ["pkg/mod.py::TABLE.split"]},
    {"symbols": ["pkg/gone.py::Box"]},
    {"texts": ["pkg/mod.py::'partitioned'"]},
])
def test_selftest_row_whose_target_vanished_fails(tree: Path, target: dict) -> None:
    errors = problems([_row(**{"paths": None, **target})], tree, TODAY)
    assert len(errors) == 1 and "no longer exist" in errors[0]


def test_selftest_one_vanished_target_among_live_ones_fails(tree: Path) -> None:
    errors = problems([_row(paths=["pkg/mod.py", "pkg/gone.py"])], tree, TODAY)
    assert len(errors) == 1 and "pkg/gone.py" in errors[0]


@pytest.mark.parametrize("over", [
    {"expires": None},                                   # undated
    {"expires": "2026-12-01"},                           # quoted, not a date
    {"expires": dt.datetime(2026, 12, 1)},               # a datetime
    {"trigger": "V4"},                                   # both forms
    {"expires": None, "trigger": "V4"},                  # trigger, no review_by
    {"expires": None, "review_by": dt.date(2027, 1, 1)},  # review_by, no trigger
    {"paths": None},                                     # no target
    {"paths": []},
    {"paths": ["../outside.py"]},
    {"symbols": ["pkg/mod.py"]},                         # no ::
    {"kind": "misc"},
    {"basis": None},
    {"source": " "},
    {"id": "Not Kebab"},
    {"note": "unknown key"},
])
def test_selftest_malformed_or_undated_row_fails(tree: Path, over: dict) -> None:
    assert problems([_row(**over)], tree, TODAY)


def test_selftest_duplicate_ids_and_empty_ledger_fail(tree: Path) -> None:
    assert problems([_row(), _row()], tree, TODAY)
    assert problems([], tree, TODAY)
    assert problems(None, tree, TODAY)


def test_selftest_stray_top_level_key_fails(tree: Path) -> None:
    """A ``[[rows]]`` typo beside valid ``[[row]]`` tables must not drop rows silently."""
    ledger = tree / "ledger.toml"
    ledger.write_text(
        '[[row]]\nid = "ok"\nkind = "facade"\npaths = ["pkg/mod.py"]\n'
        'expires = 2026-12-01\nbasis = "policy"\nsource = "test"\n\n'
        '[[rows]]\nid = "dropped"\nkind = "facade"\npaths = ["pkg/mod.py"]\n'
        'expires = 2026-01-01\nbasis = "policy"\nsource = "test"\n',
        encoding="utf-8")
    errors = ledger_problems(load_doc(ledger), tree, TODAY)
    assert len(errors) == 1 and "['rows']" in errors[0]
    assert ledger_problems({"row": [_row()]}, tree, TODAY) == []
