"""ADR 0113 INV-4 (app drift): the app's floor table equals Python's.

``apeGmshViewer/src/reader/read.ts`` hard-codes one floor table,
``ZONE_FLOOR``, whose values are the lowest minor of each zone the app
opens under that zone's target major (ADR 0113 D7). Python's floors are
the writer-owned ``*_FLOOR`` constants that ``reader_floor(zone)`` reads.
This test parses ``read.ts`` **as text** and compares the numbers; it
imports nothing across the ADR 0112 D4 wall, and the app imports nothing
from here.

The results zone is not in the app's table yet (``TODO(V4)``, the app's
results reader). ``test_results_zone_is_absent_until_v4`` pins that, and
``test_comparison_catches_a_zone_once_added`` proves the comparison
reports a results row the day it lands with the wrong value.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from apeGmsh.opensees._internal.schema_version import (
    RESULTS,
    reader_floor,
    reader_version,
)

READ_TS = (
    Path(__file__).resolve().parents[1]
    / "apeGmshViewer" / "src" / "reader" / "read.ts"
)

#: The app's zone names map onto Python's zone identifiers one to one.
_FLOOR_TABLE = re.compile(
    r"^export const ZONE_FLOOR\s*=\s*\{(?P<body>[^}]*)\}\s*as const;",
    re.MULTILINE,
)
_FLOOR_ROW = re.compile(r"(?P<zone>[a-z_]+)\s*:\s*(?P<minor>\d+)")
_TARGET = re.compile(
    r"^export const (?P<zone>[A-Z_]+)_TARGET\s*=\s*\{\s*major:\s*(?P<major>\d+)"
    r"\s*,\s*minor:\s*(?P<minor>\d+)\s*\}\s*as const;",
    re.MULTILINE,
)


def parse_zone_floors(source: str) -> dict[str, int]:
    """``ZONE_FLOOR`` as ``{zone: floor minor}`` from the read.ts text."""
    m = _FLOOR_TABLE.search(source)
    assert m is not None, f"{READ_TS}: no `export const ZONE_FLOOR = {{...}} as const;`"
    body = m.group("body")
    rows = {r.group("zone"): int(r.group("minor")) for r in _FLOOR_ROW.finditer(body)}
    assert rows, f"{READ_TS}: ZONE_FLOOR is empty: {body!r}"
    return rows


def parse_target_majors(source: str) -> dict[str, int]:
    """``<ZONE>_TARGET.major`` as ``{zone: major}`` from the read.ts text."""
    return {
        m.group("zone").lower(): int(m.group("major"))
        for m in _TARGET.finditer(source)
    }


def floor_drift(app_floors: dict[str, int], app_majors: dict[str, int]) -> list[str]:
    """Every way the app's table disagrees with Python, one line each.

    A zone the app names must have a Python floor (``reader_floor``
    raises on an unknown zone); its floor minor must equal Python's;
    and its target major must equal Python's reader major, because a
    floor minor only means something under the right major.
    """
    findings: list[str] = []
    for zone, app_minor in app_floors.items():
        try:
            py_floor = reader_floor(zone)
        except ValueError as exc:
            findings.append(
                f"read.ts ZONE_FLOOR names {zone!r} ({zone}: {app_minor}) but "
                f"Python has no floor for it: {exc}"
            )
            continue
        if app_minor != py_floor.minor:
            findings.append(
                f"{zone}: read.ts ZONE_FLOOR.{zone} = {app_minor} but Python's "
                f"floor is {py_floor} (minor {py_floor.minor}); the two must "
                f"agree (ADR 0113 D7)"
            )
        app_major = app_majors.get(zone)
        py_major = reader_version(zone).major
        if app_major is None:
            findings.append(
                f"{zone}: read.ts has ZONE_FLOOR.{zone} but no "
                f"{zone.upper()}_TARGET giving its major"
            )
        elif app_major != py_major:
            findings.append(
                f"{zone}: read.ts {zone.upper()}_TARGET.major = {app_major} but "
                f"Python's {zone} reader is on major {py_major}"
            )
    return findings


@pytest.fixture(scope="module")
def read_ts() -> str:
    assert READ_TS.is_file(), f"{READ_TS} is missing: the app's reader moved?"
    return READ_TS.read_text(encoding="utf-8")


def test_app_floors_equal_python_floors(read_ts: str) -> None:
    floors = parse_zone_floors(read_ts)
    findings = floor_drift(floors, parse_target_majors(read_ts))
    assert findings == [], "\n".join(findings)


def test_app_floor_table_covers_the_zones_the_app_reads(read_ts: str) -> None:
    """The app reads these zones today; each must sit in its floor table.

    The set is the app's, not Python's: Python's ``_ZONE_KEY`` also holds
    results, which the app does not read yet (see the next test).
    """
    floors = parse_zone_floors(read_ts)
    assert {"neutral", "opensees", "geometry", "provenance"} <= set(floors), floors


def test_results_zone_is_absent_until_v4(read_ts: str) -> None:
    """Results is not in the app's table yet. It may be absent only while
    read.ts says so (``TODO(V4)``); once present, the drift test above
    compares it like every other zone."""
    floors = parse_zone_floors(read_ts)
    if RESULTS in floors:
        # Results joined the table: test_app_floors_equal_python_floors
        # now compares it like every other zone.
        assert floor_drift({RESULTS: floors[RESULTS]}, parse_target_majors(read_ts)) == []
        return
    assert re.search(r"TODO\(V4\).*results", read_ts), (
        f"read.ts ZONE_FLOOR has no {RESULTS!r} row and no TODO(V4) note "
        "saying why; add the row or the note"
    )


def test_comparison_catches_a_zone_once_added() -> None:
    """Self-test: a results row with the wrong minor is reported, naming
    both values; one with the right minor is not. So the day results
    joins the app's table, the drift test judges it."""
    py_results = reader_floor(RESULTS)
    wrong = py_results.minor + 1
    majors = {RESULTS: reader_version(RESULTS).major}
    findings = floor_drift({RESULTS: wrong}, majors)
    assert len(findings) == 1, findings
    assert f"ZONE_FLOOR.results = {wrong}" in findings[0]
    assert f"Python's floor is {py_results}" in findings[0]
    assert floor_drift({RESULTS: py_results.minor}, majors) == []


def test_comparison_rejects_a_zone_python_does_not_know() -> None:
    findings = floor_drift({"sequence": 0}, {"sequence": 1})
    assert len(findings) == 1 and "Python has no floor" in findings[0], findings


def test_comparison_rejects_a_major_drift() -> None:
    py = reader_floor("neutral")
    findings = floor_drift({"neutral": py.minor}, {"neutral": py.major + 1})
    assert len(findings) == 1 and "TARGET.major" in findings[0], findings


def test_parser_reads_the_committed_shape() -> None:
    """The regexes read the exact line shape read.ts uses, so a reformat
    fails here (loudly) rather than parsing nothing."""
    sample = (
        "export const NEUTRAL_TARGET = { major: 2, minor: 33 } as const;\n"
        "export const ZONE_FLOOR = { neutral: 10, opensees: 11 } as const;\n"
    )
    assert parse_zone_floors(sample) == {"neutral": 10, "opensees": 11}
    assert parse_target_majors(sample) == {"neutral": 2}
