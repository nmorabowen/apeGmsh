"""#2274 (the viewer side of #2239 / PR #2273): the app's legacy simplex
Gauss table equals Python's.

``apeGmshViewer/src/reader/legacySimplex.ts`` maps the ``_natural_coords``
rows that native files written before #2238 hold (node-0-first) to the gmsh
rows the app extrapolates with, by exact table match, as
``NativeReader._gauss_natural_coords`` does. Its table,
``LEGACY_SIMPLEX_GP_ROWS``, is a TS copy of Python's ``_SIMPLEX_GP_ROWS``
(``src/apeGmsh/results/readers/_native.py``): class tag, class name, the
older rows and the current rows. This test parses legacySimplex.ts **as
text** (one entry per line), the way ``test_app_hash_232_drift.py`` does, and
compares every number bit for bit with Python's: a match is exact equality
on both sides, so a table off by one ulp would silently stop matching.

It imports nothing across the ADR 0112 D4 wall, and the app imports nothing
from here. Any difference fails.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from apeGmsh.results.readers._native import _SIMPLEX_GP_ROWS

ROOT = Path(__file__).resolve().parents[1]
LEGACY_TS = ROOT / "apeGmshViewer" / "src" / "reader" / "legacySimplex.ts"

_TABLE = re.compile(
    r"^export const LEGACY_SIMPLEX_GP_ROWS\b[^=]*=\s*\{\n(?P<body>.*?)^\};",
    re.MULTILINE | re.DOTALL,
)
_ENTRY = re.compile(
    r'^\s*(?P<tag>\d+):\s*\{\s*cls:\s*"(?P<cls>\w+)",\s*'
    r"legacy:\s*(?P<legacy>\[\[[^;]*?\]\]),\s*"
    r"current:\s*(?P<current>\[\[[^;]*?\]\])\s*\},\s*$",
)


def parse_ts_table(source: str) -> dict[int, tuple[str, np.ndarray, np.ndarray]]:
    """``LEGACY_SIMPLEX_GP_ROWS`` from the legacySimplex.ts text."""
    matches = list(_TABLE.finditer(source))
    assert len(matches) == 1, (
        f"{LEGACY_TS}: expected one `export const LEGACY_SIMPLEX_GP_ROWS ... = {{`"
        f" ending in a `}};` line, found {len(matches)}"
    )
    out: dict[int, tuple[str, np.ndarray, np.ndarray]] = {}
    for line in matches[0].group("body").splitlines():
        if not line.strip():
            continue
        m = _ENTRY.match(line)
        assert m is not None, (
            f"{LEGACY_TS}: {line.strip()[:80]!r} is not one "
            f'`<tag>: {{ cls: "<name>", legacy: [[...]], current: [[...]] }},` line'
        )
        tag = int(m.group("tag"))
        assert tag not in out, f"{LEGACY_TS}: class tag {tag} listed twice"
        out[tag] = (
            m.group("cls"),
            np.array(json.loads(m.group("legacy")), dtype=np.float64),
            np.array(json.loads(m.group("current")), dtype=np.float64),
        )
    return out


def test_the_app_table_is_pythons_value_for_value() -> None:
    ts = parse_ts_table(LEGACY_TS.read_text(encoding="utf-8"))
    assert sorted(ts) == sorted(_SIMPLEX_GP_ROWS), "the class tags differ"
    for tag, (cls, legacy, current) in _SIMPLEX_GP_ROWS.items():
        ts_cls, ts_legacy, ts_current = ts[tag]
        assert ts_cls == cls, f"class tag {tag}: app {ts_cls}, Python {cls}"
        np.testing.assert_array_equal(ts_legacy, legacy, err_msg=f"{cls}: older rows")
        np.testing.assert_array_equal(ts_current, current, err_msg=f"{cls}: current rows")


def test_the_parser_sees_a_one_ulp_change() -> None:
    """A mutation the parser must not round away: one digit off in one value."""
    source = LEGACY_TS.read_text(encoding="utf-8")
    assert "0.33333333333333337" in source
    mutated = source.replace("0.33333333333333337", "0.3333333333333333", 1)
    ts = parse_ts_table(mutated)
    assert any(
        not np.array_equal(ts[tag][2], current)
        for tag, (_cls, _legacy, current) in _SIMPLEX_GP_ROWS.items()
    )
