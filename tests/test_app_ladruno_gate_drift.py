"""ADR 0122 slice 4 (#2126): the app's ``.ladruno`` gate equals Python's.

``apeGmshViewer/src/reader/ladruno.ts`` hard-codes the ``INFO@FORMAT_VERSION``
values it opens, ``LADRUNO_SUPPORTED_FORMAT_VERSIONS``, and the
``INFO@GENERATOR`` it requires, ``LADRUNO_GENERATOR``. Python's are
``LADRUNO_SUPPORTED_FORMAT_VERSIONS`` in ``results/schema/_versions.py`` and
the literal ``"Ladruno"`` that ``LadrunoReader._validate_identity`` checks.
This test parses ``ladruno.ts`` **as text**, the way
``test_app_floor_drift.py`` parses ``read.ts``: it imports nothing across the
ADR 0112 D4 wall, and the app imports nothing from here.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from apeGmsh.results.schema._versions import LADRUNO_SUPPORTED_FORMAT_VERSIONS

LADRUNO_TS = (
    Path(__file__).resolve().parents[1]
    / "apeGmshViewer" / "src" / "reader" / "ladruno.ts"
)

_VERSIONS = re.compile(
    r"^export const LADRUNO_SUPPORTED_FORMAT_VERSIONS\s*(?::[^=]+)?=\s*"
    r"\[(?P<body>[^\]]*)\]\s*as const;",
    re.MULTILINE,
)
_GENERATOR = re.compile(
    r'^export const LADRUNO_GENERATOR\s*=\s*"(?P<name>[^"]*)";', re.MULTILINE,
)
_INT = re.compile(r"^\s*(\d+)\s*$")


def parse_versions(source: str) -> tuple[int, ...]:
    """``LADRUNO_SUPPORTED_FORMAT_VERSIONS`` from the ladruno.ts text."""
    matches = list(_VERSIONS.finditer(source))
    assert len(matches) == 1, (
        f"{LADRUNO_TS}: expected one `export const LADRUNO_SUPPORTED_FORMAT_"
        f"VERSIONS ... = [...] as const;`, found {len(matches)}"
    )
    body = matches[0].group("body")
    parts = [p for p in body.split(",") if p.strip()]
    assert parts, f"{LADRUNO_TS}: LADRUNO_SUPPORTED_FORMAT_VERSIONS is empty"
    out = []
    for p in parts:
        m = _INT.match(p)
        assert m is not None, f"{LADRUNO_TS}: {p!r} in the version set is not an integer literal"
        out.append(int(m.group(1)))
    return tuple(out)


def parse_generator(source: str) -> str:
    """``LADRUNO_GENERATOR`` from the ladruno.ts text."""
    m = _GENERATOR.search(source)
    assert m is not None, f'{LADRUNO_TS}: no `export const LADRUNO_GENERATOR = "...";`'
    return m.group("name")


def gate_drift(app_versions: tuple[int, ...]) -> list[str]:
    """Every way the app's version set disagrees with Python's, one line each."""
    findings: list[str] = []
    if sorted(app_versions) != sorted(set(app_versions)):
        findings.append(f"ladruno.ts lists a FORMAT_VERSION twice: {list(app_versions)}")
    app, py = set(app_versions), set(LADRUNO_SUPPORTED_FORMAT_VERSIONS)
    for v in sorted(app - py):
        findings.append(
            f"ladruno.ts opens FORMAT_VERSION {v}, which Python's "
            f"LADRUNO_SUPPORTED_FORMAT_VERSIONS {LADRUNO_SUPPORTED_FORMAT_VERSIONS} refuses"
        )
    for v in sorted(py - app):
        findings.append(
            f"Python reads FORMAT_VERSION {v} and ladruno.ts refuses it "
            f"(app set {sorted(app)})"
        )
    return findings


@pytest.fixture(scope="module")
def ladruno_ts() -> str:
    assert LADRUNO_TS.is_file(), f"{LADRUNO_TS} is missing: the app's ladruno reader moved?"
    return LADRUNO_TS.read_text(encoding="utf-8")


def test_app_versions_equal_python_versions(ladruno_ts: str) -> None:
    findings = gate_drift(parse_versions(ladruno_ts))
    assert findings == [], "\n".join(findings)


LADRUNO_PY = (
    Path(__file__).resolve().parents[1]
    / "src" / "apeGmsh" / "results" / "readers" / "_ladruno.py"
)
_PY_GENERATOR = re.compile(r'^\s*if gen != "(?P<name>[^"]*)":', re.MULTILINE)


def test_app_generator_is_pythons_generator(ladruno_ts: str) -> None:
    """The literal ``LadrunoReader._validate_identity`` compares
    ``INFO/GENERATOR`` with, read from its source as text."""
    hits = _PY_GENERATOR.findall(LADRUNO_PY.read_text(encoding="utf-8"))
    assert len(hits) == 1, (
        f"{LADRUNO_PY}: expected one `if gen != \"...\":` in _validate_identity, "
        f"found {hits}; the drift test must follow the check"
    )
    assert parse_generator(ladruno_ts) == hits[0]


def test_comparison_catches_a_widened_set() -> None:
    """Self-test: an app set of ``[1, 2]`` is reported, naming the version."""
    wider = (*LADRUNO_SUPPORTED_FORMAT_VERSIONS, max(LADRUNO_SUPPORTED_FORMAT_VERSIONS) + 1)
    findings = gate_drift(wider)
    assert len(findings) == 1 and f"opens FORMAT_VERSION {wider[-1]}" in findings[0], findings


def test_comparison_catches_a_narrowed_set() -> None:
    py = LADRUNO_SUPPORTED_FORMAT_VERSIONS
    findings = gate_drift(tuple(v + 100 for v in py))
    assert len(findings) == 2 * len(py), findings
    assert any("Python reads FORMAT_VERSION" in f for f in findings), findings


def test_parser_reads_the_committed_shape() -> None:
    """The regex reads the exact line shape ladruno.ts uses, so a reformat
    fails here (loudly) rather than parsing nothing."""
    sample = (
        'export const LADRUNO_GENERATOR = "Ladruno";\n'
        "export const LADRUNO_SUPPORTED_FORMAT_VERSIONS: readonly number[] = [1, 2] as const;\n"
    )
    assert parse_versions(sample) == (1, 2)
    assert parse_generator(sample) == "Ladruno"
    assert parse_versions("export const LADRUNO_SUPPORTED_FORMAT_VERSIONS = [1] as const;\n") == (1,)
    with pytest.raises(AssertionError, match="not an integer literal"):
        parse_versions("export const LADRUNO_SUPPORTED_FORMAT_VERSIONS = [ONE] as const;\n")
