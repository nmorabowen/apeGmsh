"""#2189 (the viewer half of #1911): the app's 2.32.0 side-channel rule
equals Python's.

``apeGmshViewer/src/state/results.ts`` names the opensees schema 2.32.0
``model_hash`` change with three constants copied from
``src/apeGmsh/results/Results.py`` (K, #2198):

- ``SIDE_CHANNEL_DECL_VERBS``, the six verbs of that step (T's ruling on
  #2189: a TS copy, pinned here);
- ``SIDE_CHANNEL_DECL_FROM``, the step itself, named in the message;
- ``SIDE_CHANNEL_HASH_CAUSE``, the phrase of Python's line, so the banner
  carries the sentence verbatim.

This test parses both files **as text** (``ast`` for Python, a regex for
TypeScript), the way ``test_app_ladruno_gate_drift.py`` does: it imports
nothing across the ADR 0112 D4 wall, and the app imports nothing from here.
Any difference fails.

It also pins the ``.ladruno`` parity: Python names no era cause for a
non-native reader (both results-side probes return ``None``), and the app
never names one for a ``.ladruno`` (``test/results/pairing.test.ts``).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS_TS = ROOT / "apeGmshViewer" / "src" / "state" / "results.ts"
RESULTS_PY = ROOT / "src" / "apeGmsh" / "results" / "Results.py"

_TS_VERBS = re.compile(
    r"^export const SIDE_CHANNEL_DECL_VERBS\s*(?::[^=]+)?=\s*"
    r"\[(?P<body>[^\]]*)\]\s*as const;",
    re.MULTILINE,
)
_TS_FROM = re.compile(
    r"^export const SIDE_CHANNEL_DECL_FROM\s*=\s*"
    r"\{\s*major:\s*(?P<major>\d+),\s*minor:\s*(?P<minor>\d+)\s*\}\s*as const;",
    re.MULTILINE,
)
_TS_CAUSE = re.compile(
    r'^export const SIDE_CHANNEL_HASH_CAUSE\s*=\s*"(?P<text>[^"]*)";', re.MULTILINE,
)
_TS_STRING = re.compile(r'^\s*"(?P<s>[^"]*)"\s*$')


def parse_ts_verbs(source: str) -> tuple[str, ...]:
    """``SIDE_CHANNEL_DECL_VERBS`` from the results.ts text, in file order."""
    matches = list(_TS_VERBS.finditer(source))
    assert len(matches) == 1, (
        f"{RESULTS_TS}: expected one `export const SIDE_CHANNEL_DECL_VERBS ... "
        f"= [...] as const;`, found {len(matches)}"
    )
    parts = [p for p in matches[0].group("body").split(",") if p.strip()]
    out = []
    for p in parts:
        m = _TS_STRING.match(p)
        assert m is not None, f"{RESULTS_TS}: {p.strip()!r} in SIDE_CHANNEL_DECL_VERBS is not a string literal"
        out.append(m.group("s"))
    return tuple(out)


def parse_ts_from(source: str) -> tuple[int, int, int]:
    m = _TS_FROM.search(source)
    assert m is not None, f"{RESULTS_TS}: no `export const SIDE_CHANNEL_DECL_FROM = {{ major: N, minor: N }} as const;`"
    return (int(m.group("major")), int(m.group("minor")), 0)


def parse_ts_cause(source: str) -> str:
    m = _TS_CAUSE.search(source)
    assert m is not None, f'{RESULTS_TS}: no `export const SIDE_CHANNEL_HASH_CAUSE = "...";`'
    return m.group("text")


def parse_py_constants(source: str) -> dict[str, object]:
    """The three module-level constants of Results.py, evaluated as literals."""
    want = {"SIDE_CHANNEL_DECL_VERBS", "SIDE_CHANNEL_DECL_FROM", "SIDE_CHANNEL_HASH_CAUSE"}
    found: dict[str, object] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, value = node.target.id, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name, value = node.targets[0].id, node.value
        else:
            continue
        if name not in want or value is None:
            continue
        assert name not in found, f"{RESULTS_PY}: {name} is assigned twice"
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "frozenset"
            and len(value.args) == 1
        ):
            found[name] = frozenset(ast.literal_eval(value.args[0]))
        else:
            found[name] = ast.literal_eval(value)
    missing = want - set(found)
    assert not missing, f"{RESULTS_PY}: no literal module-level {sorted(missing)}; the drift test must follow the definition"
    return found


def verbs_drift(app: tuple[str, ...], py: frozenset[str]) -> list[str]:
    """Every way the app's verb list disagrees with Python's, one line each."""
    findings: list[str] = []
    if len(app) != len(set(app)):
        findings.append(f"results.ts lists a verb twice: {list(app)}")
    for v in sorted(set(app) - py):
        findings.append(f"results.ts lists {v!r}, which Python's SIDE_CHANNEL_DECL_VERBS {sorted(py)} does not")
    for v in sorted(py - set(app)):
        findings.append(f"Python's SIDE_CHANNEL_DECL_VERBS has {v!r}, and results.ts does not (app list {sorted(app)})")
    return findings


@pytest.fixture(scope="module")
def results_ts() -> str:
    assert RESULTS_TS.is_file(), f"{RESULTS_TS} is missing: the app's pairing rule moved?"
    return RESULTS_TS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def py_constants() -> dict[str, object]:
    return parse_py_constants(RESULTS_PY.read_text(encoding="utf-8"))


def test_app_verbs_equal_python_verbs(results_ts: str, py_constants: dict[str, object]) -> None:
    py = py_constants["SIDE_CHANNEL_DECL_VERBS"]
    assert isinstance(py, frozenset) and len(py) == 6, py
    findings = verbs_drift(parse_ts_verbs(results_ts), py)
    assert findings == [], "\n".join(findings)


def test_app_step_is_pythons_step(results_ts: str, py_constants: dict[str, object]) -> None:
    assert parse_ts_from(results_ts) == tuple(py_constants["SIDE_CHANNEL_DECL_FROM"])  # type: ignore[arg-type]


def test_app_cause_phrase_is_pythons(results_ts: str, py_constants: dict[str, object]) -> None:
    assert parse_ts_cause(results_ts) == py_constants["SIDE_CHANNEL_HASH_CAUSE"]


def test_python_names_no_era_cause_for_a_non_native_reader() -> None:
    """The ``.ladruno`` parity, Python's half: a reader that is not the
    native one (``LadrunoReader``, ``MPCOReader``) gives ``None`` from both
    results-side probes, so ``Results.lineage`` names no era cause. The app's
    half is ``pairing.test.ts`` "a .ladruno never names an era cause"."""
    from apeGmsh.results.Results import Results, _side_channel_hash_cause

    stub = SimpleNamespace(_reader=object())
    assert Results._results_records_program_decl(stub) is None  # type: ignore[arg-type]
    assert Results._results_side_channel_proof(stub) is None  # type: ignore[arg-type]
    assert _side_channel_hash_cause(None, True) is None
    assert _side_channel_hash_cause(None, False) is None


def test_comparison_catches_a_seventh_verb() -> None:
    """Self-test: an app list with a seventh verb is reported, naming it."""
    py = frozenset({"a", "b"})
    findings = verbs_drift(("a", "b", "c"), py)
    assert len(findings) == 1 and "'c'" in findings[0], findings
    findings = verbs_drift(("a",), py)
    assert len(findings) == 1 and "has 'b'" in findings[0], findings
    assert verbs_drift(("a", "a", "b"), py) == ["results.ts lists a verb twice: ['a', 'a', 'b']"]


def test_parser_reads_the_committed_shape() -> None:
    """The regexes read the exact line shapes results.ts uses, so a reformat
    fails here (loudly) rather than parsing nothing."""
    sample = (
        "export const SIDE_CHANNEL_DECL_FROM = { major: 2, minor: 32 } as const;\n"
        "export const SIDE_CHANNEL_DECL_VERBS: readonly string[] = [\n"
        '  "x",\n  "y",\n] as const;\n'
        'export const SIDE_CHANNEL_HASH_CAUSE = "hash change (z)";\n'
    )
    assert parse_ts_verbs(sample) == ("x", "y")
    assert parse_ts_from(sample) == (2, 32, 0)
    assert parse_ts_cause(sample) == "hash change (z)"
    with pytest.raises(AssertionError, match="not a string literal"):
        parse_ts_verbs("export const SIDE_CHANNEL_DECL_VERBS = [VERB] as const;\n")
