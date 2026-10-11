"""Lock on ``architecture/fork-divergence-roster.md``.

Every row of the roster names the engine that is correct and the oracle
that decides it, and cites the tests that pin it. This file keeps those
three claims true:

* every table row is a roster row: a line the parser does not recognise
  (``D9b``, ``d10``, a backticked id) fails instead of being skipped;
* every row names a correct engine (``stock|fork|policy|pending #<n>``)
  and an oracle that opens with a kind from a fixed vocabulary and says
  something after it;
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
#: An oracle opens with its kind. ``closed form`` / ``cross-engine`` /
#: ``reference run`` / ``contract`` and ``n/a (policy)`` need a body that
#: names the answer; ``pending #<n>`` needs the reason in parentheses.
ORACLE_RE = re.compile(
    r"^(?:(?:closed form|cross-engine|reference run|contract): (?P<body>.+)"
    r"|n/a \(policy\)[.:] (?P<policy>.+)"
    r"|pending #\d+ \((?P<why>[^)]+)\))$"
)
PLACEHOLDER_RE = re.compile(r"^(?:tbd|todo|tba|n/?a|none|unknown|\?+|-+|—)\W*$",
                            re.IGNORECASE)
MIN_BODY_WORDS = 4
SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")
FORK_RE = re.compile(r"^fork:[\w./-]+\.py::\w+@[0-9a-f]{7,40}$")
NODE_RE = re.compile(r"^(?P<path>tests/[\w./-]+\.py)::(?P<name>\w+(::\w+)?)$")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_roster(text: str) -> tuple[list[dict[str, str]], list[str]]:
    """The roster rows, and every table line that is not one.

    A table line is the full header, a separator, or a row whose first cell
    is a bare ``D<n>``. Anything else in a table (``D9b``, ``d10``, a
    backticked id, a row before any header) comes back as unrecognised, so
    the lock fails on it instead of skipping it. Each table must carry the
    full header, so a dropped column fails instead of shifting every cell.
    """
    rows: list[dict[str, str]] = []
    unrecognised: list[str] = []
    header: tuple[str, ...] | None = None
    for number, line in enumerate(text.splitlines(), 1):
        if not line.lstrip().startswith("|"):
            header = None
            continue
        cells = _cells(line)
        if cells[0] == "id":
            header = tuple(cells)
            if header != COLUMNS:
                unrecognised.append(f"line {number}: header drifted: {header}")
            continue
        if all(SEPARATOR_RE.match(c) for c in cells):
            continue
        if header is None or not line.startswith("|"):
            unrecognised.append(f"line {number}: table row outside a roster table")
        elif not re.fullmatch(r"D\d+", cells[0]):
            unrecognised.append(
                f"line {number}: {cells[0]!r} is not a roster id D<n>"
            )
        elif len(cells) != len(COLUMNS):
            unrecognised.append(
                f"line {number}: row {cells[0]} has {len(cells)} cells, "
                f"expected {len(COLUMNS)} (an unescaped '|' in a cell?)"
            )
        else:
            rows.append(dict(zip(COLUMNS, cells)))
    return rows, unrecognised


def oracle_problem(cell: str) -> str | None:
    """Why an oracle cell is not substantive, or ``None`` if it is."""
    m = ORACLE_RE.match(cell)
    if m is None:
        return (f"oracle {cell[:40]!r} does not open with closed form: / "
                "cross-engine: / reference run: / contract: / n/a (policy) / "
                "pending #<n> (<why>)")
    body = m["body"] or m["policy"] or m["why"]
    if PLACEHOLDER_RE.match(body.strip()):
        return f"oracle body {body!r} is a placeholder"
    if (m["body"] or m["policy"]) and len(re.findall(r"\w+", body)) < MIN_BODY_WORDS:
        return f"oracle body {body!r} is too thin to name an answer"
    return None


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
        if not r["oracle"]:
            problems.append(f"{rid}: no oracle")
        elif (why := oracle_problem(r["oracle"])) is not None:
            problems.append(f"{rid}: {why}")
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
def text() -> str:
    return ROSTER.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def rows(text) -> list[dict[str, str]]:
    return parse_roster(text)[0]


def test_every_table_line_is_recognised(text) -> None:
    assert parse_roster(text)[1] == []


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


@pytest.mark.parametrize("bad_id", ["D9b", "d10", "`D10`", "X1"])
def test_unrecognised_row_fails_instead_of_being_skipped(text, bad_id) -> None:
    """A row the parser cannot read is a finding, never a silent skip."""
    junk = f"| {bad_id} | s | s | f | | | | none | |"
    mutated = text.replace("\n| D9 |", f"\n{junk}\n| D9 |", 1)
    assert mutated != text
    rows, unrecognised = parse_roster(mutated)
    assert any(repr(bad_id) in u for u in unrecognised)
    assert {r["id"] for r in rows} == {r["id"] for r in parse_roster(text)[0]}


@pytest.mark.parametrize("oracle", [
    "TBD", "closed form: TBD", "closed form: ?", "todo", "n/a",
    "cross-engine: see PR", "pending #1523", "pending #1523 ()",
    "n/a (policy): tbd",
])
def test_placeholder_oracle_is_caught(rows, oracle) -> None:
    assert any("D2: oracle" in p
               for p in check_rows(_mutated(rows, "D2", "oracle", oracle)))
