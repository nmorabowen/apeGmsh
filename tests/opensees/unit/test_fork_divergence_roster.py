"""Lock on ``architecture/fork-divergence-roster.md``.

Every row of the roster names the engine that is correct and the oracle
that decides it, and cites the tests that pin it. This file keeps those
three claims true:

* every row names a correct engine (``stock|fork|policy|pending #<n>``)
  and a non-empty oracle;
* every apeGmsh node id a row cites still exists (an AST check of the
  file and the test function, with no import);
* every fork-side citation has the ``fork:<path>::<name>@<sha>`` form;
* the number of rows with no pinning test does not grow (a ratchet);
* no row disappears: a fixed divergence is marked ``closed``, not deleted.

Pure text and AST work, so it runs in the default suite.
"""
from __future__ import annotations

import ast
import re
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
ROSTER = ROOT / "architecture" / "fork-divergence-roster.md"

#: Rows the roster must keep. Never remove an id: close the row instead.
REQUIRED_IDS = frozenset(f"D{i}" for i in range(1, 10))

#: Rows whose pinning-tests cell is ``none``. Lower it when a row gains a
#: pin; never raise it (PROGRAM.md §7).
UNPINNED_BASELINE = 1

COLUMNS = (
    "id", "symptom", "stock behaviour", "fork behaviour", "correct engine",
    "oracle", "evidence", "pinning tests", "status",
)
ENGINE_RE = re.compile(r"^(stock|fork|policy|pending #\d+)$")
FORK_RE = re.compile(r"^fork:[\w./-]+\.py::\w+@[0-9a-f]{7,40}$")
NODE_RE = re.compile(r"^(?P<path>tests/[\w./-]+\.py)::(?P<name>\w+(::\w+)?)$")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_roster(text: str) -> list[dict[str, str]]:
    """Every table row whose first cell is a roster id ``D<n>``.

    Each table must carry the full header, so a dropped column fails
    instead of shifting every cell one place left.
    """
    rows: list[dict[str, str]] = []
    header: tuple[str, ...] | None = None
    for line in text.splitlines():
        if not line.startswith("|"):
            header = None
            continue
        cells = _cells(line)
        if cells and cells[0] == "id":
            header = tuple(cells)
            assert header == COLUMNS, f"roster header drifted: {header}"
            continue
        if cells and re.fullmatch(r"D\d+", cells[0]):
            assert header is not None, f"row {cells[0]} sits outside a table"
            assert len(cells) == len(COLUMNS), (
                f"row {cells[0]} has {len(cells)} cells, expected "
                f"{len(COLUMNS)} (an unescaped '|' in a cell?)"
            )
            rows.append(dict(zip(COLUMNS, cells)))
    return rows


def citations(cell: str) -> list[str]:
    """The test citations in a pinning-tests cell (``[]`` for ``none``)."""
    if cell.strip() == "none":
        return []
    return [c.strip().strip("`") for c in cell.split("<br>") if c.strip()]


@lru_cache(maxsize=None)
def _defined(path: str) -> frozenset[str]:
    """``name`` and ``Class::method`` for every test in ``path``."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.add(f"{node.name}::{sub.name}")
    return frozenset(names)


def node_id_problem(cite: str) -> str | None:
    """Why an apeGmsh citation does not resolve, or ``None`` if it does."""
    m = NODE_RE.match(cite)
    if m is None:
        return f"{cite!r} is neither a tests/ node id nor fork:<path>::<name>@<sha>"
    path, name = m["path"], m["name"]
    if not (ROOT / path).is_file():
        return f"{cite!r}: no file {path}"
    if not name.split("::")[-1].startswith("test"):
        return f"{cite!r}: {name} is not a test"
    if name not in _defined(path):
        return f"{cite!r}: {path} defines no {name}"
    return None


def check_rows(rows: list[dict[str, str]]) -> list[str]:
    """Every problem with ``rows``; an empty list means the lock holds."""
    problems: list[str] = []
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        problems.append(f"duplicate roster ids: {sorted(ids)}")
    missing = REQUIRED_IDS - set(ids)
    if missing:
        problems.append(
            f"rows {sorted(missing)} disappeared; close a row, never delete it"
        )
    unpinned = 0
    for r in rows:
        rid = r["id"]
        if not ENGINE_RE.match(r["correct engine"]):
            problems.append(
                f"{rid}: correct engine {r['correct engine']!r} is not "
                "stock|fork|policy|pending #<issue>"
            )
        if not r["oracle"] or r["oracle"] in {"-", "—", "?"}:
            problems.append(f"{rid}: no oracle")
        if not r["status"]:
            problems.append(f"{rid}: no status")
        cites = citations(r["pinning tests"])
        if not cites:
            unpinned += 1
        for c in cites:
            if c.startswith("fork:"):
                if not FORK_RE.match(c):
                    problems.append(
                        f"{rid}: {c!r} is not fork:<path>::<name>@<sha>"
                    )
            else:
                why = node_id_problem(c)
                if why:
                    problems.append(f"{rid}: {why}")
    if unpinned > UNPINNED_BASELINE:
        problems.append(
            f"{unpinned} rows have no pinning test, baseline "
            f"{UNPINNED_BASELINE}; pin the new row instead of raising it"
        )
    return problems


@pytest.fixture(scope="module")
def rows() -> list[dict[str, str]]:
    return parse_roster(ROSTER.read_text(encoding="utf-8"))


def test_roster_parses_every_required_row(rows) -> None:
    assert {r["id"] for r in rows} >= REQUIRED_IDS


def test_roster_lock_holds(rows) -> None:
    assert check_rows(rows) == []


def test_unpinned_baseline_is_tight(rows) -> None:
    """The ratchet only ratchets if the baseline equals today's count."""
    unpinned = sum(1 for r in rows if not citations(r["pinning tests"]))
    assert unpinned == UNPINNED_BASELINE, (
        f"{unpinned} unpinned rows, baseline {UNPINNED_BASELINE}: lower it"
    )


# ---------------------------------------------------------------------
# The lock bites: each mutation of a real row must be caught
# ---------------------------------------------------------------------

def test_fork_citations_are_unquoted_and_parse(rows) -> None:
    """Fork citations stay out of backticks (the doc-path quirk rule reads
    a backticked path as a file in this repo) and still parse here."""
    fork = [c for r in rows for c in citations(r["pinning tests"])
            if c.startswith("fork:")]
    assert fork and all(FORK_RE.match(c) for c in fork)


def _mutated(rows, rid: str, column: str, value: str):
    return [dict(r, **{column: value}) if r["id"] == rid else r for r in rows]


def test_blank_correct_engine_is_caught(rows) -> None:
    assert any("D1: correct engine" in p
               for p in check_rows(_mutated(rows, "D1", "correct engine", "")))


def test_unknown_engine_word_is_caught(rows) -> None:
    assert check_rows(_mutated(rows, "D2", "correct engine", "both"))


def test_blank_oracle_is_caught(rows) -> None:
    assert any("D4: no oracle" in p
               for p in check_rows(_mutated(rows, "D4", "oracle", "")))


def test_renamed_test_is_caught(rows) -> None:
    cell = ("`tests/opensees/live/test_tet10_volume_live.py::"
            "test_tet10_block_is_one_sixth_on_stock_RENAMED`")
    problems = check_rows(_mutated(rows, "D1", "pinning tests", cell))
    assert any("defines no" in p for p in problems)


def test_missing_file_is_caught(rows) -> None:
    cell = "`tests/sections/test_gone.py::test_x`"
    assert any("no file" in p
               for p in check_rows(_mutated(rows, "D4", "pinning tests", cell)))


def test_malformed_fork_citation_is_caught(rows) -> None:
    cell = "`fork:tests/test_x.py::test_y`"  # no @sha
    assert check_rows(_mutated(rows, "D3", "pinning tests", cell))


def test_dropping_a_pin_trips_the_ratchet(rows) -> None:
    problems = check_rows(_mutated(rows, "D4", "pinning tests", "none"))
    assert any("baseline" in p for p in problems)


def test_deleted_row_is_caught(rows) -> None:
    assert any("disappeared" in p
               for p in check_rows([r for r in rows if r["id"] != "D9"]))
