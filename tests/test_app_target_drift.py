"""ADR 0113 INV-4 (app drift, targets): the app's zone targets equal main's writers.

The app (``apeGmshViewer/``) hard-codes the version of each zone it was
written against as ``<ZONE>_TARGET = { major, minor }`` in its reader
(``read.ts`` for neutral, opensees, geometry, provenance and sequence;
``results.ts`` for results). A file newer than a target opens with a
"newer than this app" banner, so a target that lags its Python writer
puts that banner on every file fresh from main (V4t, #1821).

This test parses the reader sources **as text** (nothing crosses the
ADR 0112 D4 wall into the app) and compares every target's major.minor
with the Python writer's version, :func:`reader_version`, which reads
the writer constants (``NEUTRAL_SCHEMA_VERSION``, the emitter's
``SCHEMA_VERSION``, ``RESULTS_SCHEMA_VERSION``, ...). It fails when a
target is behind its writer, when it is ahead, when a target has no
Python zone, and when a Python zone has no target and is not listed in
:data:`NOT_READ_BY_APP` with the reason. The failure names the constant,
its file and the h5-schema.md history entry to read before bumping it.

The patch is not compared: a patch is fix-only (ADR 0023) and the app's
version check ignores it.

``_ZONE_KEY`` is imported by name so a new Python zone joins this
comparison the day it is registered (and a rename fails the import).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from apeGmsh.opensees._internal.schema_version import (
    _ZONE_KEY,
    ASSEMBLY,
    GEOMETRY,
    NEUTRAL,
    OPENSEES,
    PROVENANCE,
    RESULTS,
    SEQUENCE,
    reader_version,
)

REPO = Path(__file__).resolve().parents[1]
READER = REPO / "apeGmshViewer" / "src" / "reader"
SCHEMA_DOC = "architecture/h5-schema.md"

#: Where each target lives, as a repo-relative path: the constant a
#: schema-bump PR edits.
EXPECTED_HOME: dict[str, str] = {
    NEUTRAL: "apeGmshViewer/src/reader/read.ts",
    OPENSEES: "apeGmshViewer/src/reader/read.ts",
    GEOMETRY: "apeGmshViewer/src/reader/read.ts",
    PROVENANCE: "apeGmshViewer/src/reader/read.ts",
    SEQUENCE: "apeGmshViewer/src/reader/read.ts",
    RESULTS: "apeGmshViewer/src/reader/results.ts",
}

#: Python zones the app does not read, each with the reason. A zone here
#: must have no target; a zone the app starts reading leaves this table.
NOT_READ_BY_APP: dict[str, str] = {
    ASSEMBLY: (
        "/assembly (ADR 0117 D5) is written only by Assembly.h5 archives; "
        "the app opens model.h5 / results.h5 and has no assembly reader"
    ),
}

_TARGET = re.compile(
    r"^export const (?P<zone>[A-Z_]+)_TARGET\s*=\s*\{\s*major:\s*(?P<major>\d+)"
    r"\s*,\s*minor:\s*(?P<minor>\d+)\s*\}\s*as const;",
    re.MULTILINE,
)


def const_of(zone: str) -> str:
    return f"{zone.upper()}_TARGET"


def parse_targets(sources: dict[str, str]) -> tuple[dict[str, tuple[str, int, int]], list[str]]:
    """``{zone: (file, major, minor)}`` from ``{repo-relative file: text}``.

    A target defined in two files is a finding (the second one could drift
    unseen), returned beside the table.
    """
    found: dict[str, tuple[str, int, int]] = {}
    findings: list[str] = []
    for path, text in sorted(sources.items()):
        for m in _TARGET.finditer(text):
            zone = m.group("zone").lower()
            if zone in found:
                findings.append(
                    f"`{const_of(zone)}` is defined in both `{found[zone][0]}` and "
                    f"`{path}`; keep one"
                )
                continue
            found[zone] = (path, int(m.group("major")), int(m.group("minor")))
    return found, findings


def target_drift(
    app: dict[str, tuple[str, int, int]],
    writers: dict[str, tuple[int, int, int]],
    not_read: dict[str, str] = NOT_READ_BY_APP,
    homes: dict[str, str] = EXPECTED_HOME,
) -> list[str]:
    """Every way the app's targets disagree with the Python writers, one line each."""
    findings: list[str] = []
    for zone, (path, major, minor) in sorted(app.items()):
        const = const_of(zone)
        if zone not in writers:
            findings.append(
                f"`{const}` in `{path}` has no Python counterpart: no zone {zone!r} in "
                f"schema_version._ZONE_KEY ({sorted(writers)}); remove it or register the zone"
            )
            continue
        if zone in not_read:
            findings.append(
                f"`{const}` in `{path}` targets {zone!r}, which NOT_READ_BY_APP says the app "
                f"does not read; drop that row of tests/test_app_target_drift.py"
            )
        home = homes.get(zone)
        if home is not None and home != path:
            findings.append(f"`{const}` lives in `{path}`; this test expects it in `{home}`")
        w_major, w_minor, w_patch = writers[zone]
        wanted = f"{{ major: {w_major}, minor: {w_minor} }}"
        if (major, minor) < (w_major, w_minor):
            findings.append(
                f"{zone}: `{const}` in `{path}` is {major}.{minor}, behind the Python "
                f"writer's {w_major}.{w_minor}.{w_patch}: bump `{const}` in `{path}` to "
                f"{wanted} after reading the {zone} history entry for "
                f"{w_major}.{w_minor}.0 in {SCHEMA_DOC} (read the new data, or say above "
                f"the constant why the app does not need it)"
            )
        elif (major, minor) > (w_major, w_minor):
            findings.append(
                f"{zone}: `{const}` in `{path}` is {major}.{minor}, ahead of the Python "
                f"writer's {w_major}.{w_minor}.{w_patch}: lower `{const}` in `{path}` to "
                f"{wanted}; the app cannot target a version main does not write"
            )
    for zone in sorted(writers):
        if zone in app or zone in not_read:
            continue
        home = homes.get(zone, "apeGmshViewer/src/reader/read.ts")
        w_major, w_minor, _ = writers[zone]
        findings.append(
            f"{zone}: the Python writer is at {w_major}.{w_minor} but the app has no "
            f"`{const_of(zone)}` (expected in `{home}`); add it after reading the "
            f"{zone} history in {SCHEMA_DOC}, or list {zone!r} in NOT_READ_BY_APP "
            f"(tests/test_app_target_drift.py) with the reason the app does not read it"
        )
    return findings


def _writer_versions() -> dict[str, tuple[int, int, int]]:
    out = {}
    for zone in _ZONE_KEY:
        v = reader_version(zone)
        out[zone] = (v.major, v.minor, v.patch)
    return out


@pytest.fixture(scope="module")
def sources() -> dict[str, str]:
    assert READER.is_dir(), f"{READER} is missing: the app's reader moved?"
    return {
        p.relative_to(REPO).as_posix(): p.read_text(encoding="utf-8")
        for p in sorted(READER.glob("*.ts"))
    }


def test_app_targets_equal_python_writers(sources: dict[str, str]) -> None:
    app, findings = parse_targets(sources)
    findings += target_drift(app, _writer_versions())
    assert findings == [], "\n".join(findings)


def test_every_expected_target_is_parsed(sources: dict[str, str]) -> None:
    """The parser reads each target the app has today, from its home file,
    so a reformat of a target line fails loudly rather than parsing nothing."""
    app, _ = parse_targets(sources)
    assert {z: app[z][0] for z in app} == EXPECTED_HOME


def test_not_read_zones_are_python_zones() -> None:
    assert set(NOT_READ_BY_APP) <= set(_ZONE_KEY), NOT_READ_BY_APP


# ---- self-tests: each mutation the card names is reported --------------

_W = {OPENSEES: (2, 28, 0), NEUTRAL: (2, 35, 0)}
_RT = "apeGmshViewer/src/reader/read.ts"


def test_a_lowered_target_names_the_constant_file_and_history_entry() -> None:
    app = {OPENSEES: (_RT, 2, 27), NEUTRAL: (_RT, 2, 35)}
    f = target_drift(app, _W, not_read={})
    assert len(f) == 1, f
    assert "bump `OPENSEES_TARGET` in `apeGmshViewer/src/reader/read.ts`" in f[0]
    assert "opensees history entry for 2.28.0 in architecture/h5-schema.md" in f[0]


def test_a_raised_target_fails() -> None:
    f = target_drift({OPENSEES: (_RT, 2, 29), NEUTRAL: (_RT, 2, 35)}, _W, not_read={})
    assert len(f) == 1 and "ahead of the Python writer's 2.28.0" in f[0], f
    assert "lower `OPENSEES_TARGET`" in f[0]


def test_another_major_fails_either_way() -> None:
    assert len(target_drift({OPENSEES: (_RT, 1, 99), NEUTRAL: (_RT, 2, 35)}, _W, not_read={})) == 1
    assert len(target_drift({OPENSEES: (_RT, 3, 0), NEUTRAL: (_RT, 2, 35)}, _W, not_read={})) == 1


def test_a_deleted_target_fails() -> None:
    f = target_drift({NEUTRAL: (_RT, 2, 35)}, _W, not_read={})
    assert len(f) == 1 and "the app has no `OPENSEES_TARGET`" in f[0], f


def test_a_target_with_no_python_zone_fails() -> None:
    f = target_drift({OPENSEES: (_RT, 2, 28), NEUTRAL: (_RT, 2, 35), "mystery": (_RT, 1, 0)}, _W, not_read={})
    assert len(f) == 1 and "`MYSTERY_TARGET`" in f[0] and "no Python counterpart" in f[0], f


def test_a_target_for_a_not_read_zone_fails() -> None:
    f = target_drift({OPENSEES: (_RT, 2, 28), NEUTRAL: (_RT, 2, 35)}, _W, not_read={NEUTRAL: "x"})
    assert len(f) == 1 and "NOT_READ_BY_APP" in f[0], f


def test_a_target_in_the_wrong_file_or_twice_fails() -> None:
    f = target_drift({OPENSEES: ("apeGmshViewer/src/reader/x.ts", 2, 28), NEUTRAL: (_RT, 2, 35)}, _W, not_read={})
    assert len(f) == 1 and "this test expects it in" in f[0], f
    line = "export const OPENSEES_TARGET = { major: 2, minor: 28 } as const;\n"
    _, dup = parse_targets({"a.ts": line, "b.ts": line})
    assert len(dup) == 1 and "defined in both" in dup[0], dup


def test_equal_targets_pass_and_the_patch_is_ignored() -> None:
    assert target_drift({OPENSEES: (_RT, 2, 28), NEUTRAL: (_RT, 2, 35)}, {OPENSEES: (2, 28, 3), NEUTRAL: (2, 35, 0)}, not_read={}) == []
