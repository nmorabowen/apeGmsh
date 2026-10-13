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

#: A declaration of a target, exported or not, in comment-free source.
_DECL = re.compile(r"\bconst\s+(?P<zone>[A-Z][A-Z0-9_]*)_TARGET\b")
#: The value that must follow it: an optional type annotation, then
#: ``{ major: X, minor: Y }`` (any whitespace, newlines and an optional
#: trailing comma), then ``as const`` / ``satisfies T`` / nothing.
_VALUE = re.compile(
    r"\s*(?::[^=]*)?=\s*\{\s*major\s*:\s*(?P<major>\d+)\s*,"
    r"\s*minor\s*:\s*(?P<minor>\d+)\s*,?\s*\}"
)


def strip_comments(text: str) -> str:
    """``text`` with every ``//`` and ``/* */`` comment blanked, string
    literals (``'``, ``"``, `````) kept, so a commented-out target is
    never read as the real one. Newlines inside a block comment are kept."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in "'\"`":
            j = i + 1
            while j < n and text[j] != c:
                if text[j] == "\\":
                    j += 1
                elif c != "`" and text[j] == "\n":
                    break  # an unterminated ' or " string ends at the line
                j += 1
            out.append(text[i:j + 1])
            i = j + 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join(ch for ch in text[i:j] if ch == "\n"))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def const_of(zone: str) -> str:
    return f"{zone.upper()}_TARGET"


def parse_targets(
    sources: dict[str, str],
) -> tuple[dict[str, tuple[str, int, int]], list[str], set[str]]:
    """``({zone: (file, major, minor)}, findings, declared zones)`` from
    ``{repo-relative file: text}``, comments stripped first.

    A target defined in two files, or declared in a shape this parser cannot
    read ("could not parse"), is a finding returned beside the table; the
    declared set lets :func:`target_drift` tell an unparsable target from a
    missing one.
    """
    found: dict[str, tuple[str, int, int]] = {}
    findings: list[str] = []
    declared: set[str] = set()
    seen: dict[str, str] = {}
    for path, text in sorted(sources.items()):
        code = strip_comments(text)
        for d in _DECL.finditer(code):
            zone = d.group("zone").lower()
            declared.add(zone)
            if zone in seen:
                findings.append(
                    f"`{const_of(zone)}` is defined in both `{seen[zone]}` and "
                    f"`{path}`; keep one"
                )
                continue
            seen[zone] = path
            m = _VALUE.match(code, d.end())
            if m is None:
                snippet = code[d.start():d.end() + 60].split(";")[0]
                findings.append(
                    f"could not parse `{const_of(zone)}` in `{path}` "
                    f"(`{' '.join(snippet.split())}`): write it as "
                    f"`export const {const_of(zone)} = {{ major: X, minor: Y }} as const;`"
                )
                continue
            found[zone] = (path, int(m.group("major")), int(m.group("minor")))
    return found, findings, declared


def target_drift(
    app: dict[str, tuple[str, int, int]],
    writers: dict[str, tuple[int, int, int]],
    not_read: dict[str, str] = NOT_READ_BY_APP,
    homes: dict[str, str] = EXPECTED_HOME,
    declared: frozenset[str] | set[str] = frozenset(),
) -> list[str]:
    """Every way the app's targets disagree with the Python writers, one line each.

    A zone in ``declared`` but not in ``app`` was reported by
    :func:`parse_targets` as unparsable, so it is not reported missing too.
    """
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
        if zone in app or zone in not_read or zone in declared:
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
    app, findings, declared = parse_targets(sources)
    findings += target_drift(app, _writer_versions(), declared=declared)
    assert findings == [], "\n".join(findings)


def test_every_expected_target_is_parsed(sources: dict[str, str]) -> None:
    """The parser reads each target the app has today, from its home file,
    so a reformat of a target line fails loudly rather than parsing nothing."""
    app, findings, _ = parse_targets(sources)
    assert findings == [], findings
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
    _, dup, _ = parse_targets({"a.ts": line, "b.ts": line})
    assert len(dup) == 1 and "defined in both" in dup[0], dup


def test_equal_targets_pass_and_the_patch_is_ignored() -> None:
    assert target_drift({OPENSEES: (_RT, 2, 28), NEUTRAL: (_RT, 2, 35)}, {OPENSEES: (2, 28, 3), NEUTRAL: (2, 35, 0)}, not_read={}) == []


def _drift_of(source: str) -> list[str]:
    """parse_targets then target_drift on one read.ts text, writers ``_W``."""
    app, findings, declared = parse_targets({_RT: source})
    return findings + target_drift(app, _W, not_read={}, declared=declared)


_NEUTRAL_LINE = "export const NEUTRAL_TARGET = { major: 2, minor: 35 } as const;\n"


def test_a_commented_decoy_does_not_hide_a_behind_target() -> None:
    """Review round 1 reproducer: a block-commented decoy at 28 and the real
    line at 27 with ``satisfies``. The decoy is not read; the real line is
    behind. A line-comment decoy is not read either."""
    src = (
        "/*\nexport const OPENSEES_TARGET = { major: 2, minor: 28 } as const;\n*/\n"
        "export const OPENSEES_TARGET = { major: 2, minor: 27 } satisfies Version;\n"
        + _NEUTRAL_LINE
    )
    f = _drift_of(src)
    assert len(f) == 1 and "is 2.27, behind" in f[0] and "bump `OPENSEES_TARGET`" in f[0], f
    line = (
        "// export const OPENSEES_TARGET = { major: 2, minor: 28 } as const;\n"
        "export const OPENSEES_TARGET = { major: 2, minor: 27 } as const; // was 28\n"
        + _NEUTRAL_LINE
    )
    f = _drift_of(line)
    assert len(f) == 1 and "is 2.27, behind" in f[0], f
    # A commented-out target alone is a missing target, not a present one.
    f = _drift_of("/* export const OPENSEES_TARGET = { major: 2, minor: 28 } as const; */\n" + _NEUTRAL_LINE)
    assert len(f) == 1 and "the app has no `OPENSEES_TARGET`" in f[0], f


def test_comment_markers_inside_strings_are_not_comments() -> None:
    src = 'const url = "http://x/*y"; ' + "export const OPENSEES_TARGET = { major: 2, minor: 28 } as const;\n" + _NEUTRAL_LINE
    assert _drift_of(src) == []


def test_a_multi_line_target_with_a_trailing_comma_parses() -> None:
    src = "export const OPENSEES_TARGET = {\n  major: 2,\n  minor: 27,\n} as const;\n" + _NEUTRAL_LINE
    f = _drift_of(src)
    assert len(f) == 1 and "is 2.27, behind" in f[0], f
    assert _drift_of(src.replace("minor: 27", "minor: 28")) == []
    typed = "export const OPENSEES_TARGET: Version = { major: 2, minor: 28 };\n" + _NEUTRAL_LINE
    assert _drift_of(typed) == []


def test_an_unparsable_target_says_could_not_parse_never_missing() -> None:
    for bad in (
        "export const OPENSEES_TARGET = { minor: 28, major: 2 } as const;\n",
        "export const OPENSEES_TARGET = version(2, 28);\n",
        "const OPENSEES_TARGET = BASE;\nexport { OPENSEES_TARGET };\n",
    ):
        f = _drift_of(bad + _NEUTRAL_LINE)
        assert len(f) == 1, (bad, f)
        assert f[0].startswith("could not parse `OPENSEES_TARGET` in `apeGmshViewer/src/reader/read.ts`"), f
        assert "missing" not in f[0] and "has no" not in f[0], f
