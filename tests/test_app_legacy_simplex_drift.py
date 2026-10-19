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

It also owns the app's oracle, ``apeGmshViewer/fixtures/legacy_simplex.oracle.json``:
Python's node values on the two legacy fixtures (``legacy_simplex_*.results.h5``,
PR #2273's files written at 31ba6a63c), read through ``NativeReader`` (with
#2273's mapping) and carried to the corners by
``extrapolate_gauss_slab_to_nodes`` on a one-element gmsh-coded stand-in, nodes
``1..nc`` at the fixture's corners. ``test/contour/extrapolation.test.ts``
checks the app against it. The test here recomputes it and compares; run this
file as a script to rewrite it::

    python tests/test_app_legacy_simplex_drift.py

It imports nothing across the ADR 0112 D4 wall, and the app imports nothing
from here. Any difference fails.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from apeGmsh.results._gauss_extrapolation import extrapolate_gauss_slab_to_nodes
from apeGmsh.results.readers._native import _SIMPLEX_GP_ROWS, NativeReader

ROOT = Path(__file__).resolve().parents[1]
LEGACY_TS = ROOT / "apeGmshViewer" / "src" / "reader" / "legacySimplex.ts"
FIXTURES = ROOT / "apeGmshViewer" / "fixtures"
ORACLE = FIXTURES / "legacy_simplex.oracle.json"

#: The geometry of every fixture element (#2273's generate.py).
CORNERS = {
    "tri": [[0.1, 0.2], [2.3, 0.4], [0.6, 1.8]],
    "tet": [[0.1, 0.2, 0.0], [2.3, 0.1, 0.4], [0.5, 1.7, 0.2], [0.3, 0.4, 1.9]],
}
#: fixture -> (stem, gmsh code, nodes per element, components).
FILES = {
    "legacy_simplex_capture_tet.results.h5": ("tet", 11, 10, ("strain_xx", "strain_yy", "strain_zz")),
    "legacy_simplex_transcoded_tri.results.h5": ("tri", 9, 6, ("strain_xx", "strain_yy")),
}

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


def _fem(stem: str, eid: int, code: int, npe: int) -> SimpleNamespace:
    """One gmsh-coded element with nodes 1..npe, its corners at CORNERS."""
    corners = np.asarray(CORNERS[stem])
    xyz = np.zeros((npe, 3))
    xyz[: corners.shape[0], : corners.shape[1]] = corners
    tags = np.arange(1, npe + 1, dtype=np.int64)

    class _Group:
        element_type = SimpleNamespace(code=code, dim=corners.shape[1], npe=npe, order=2, gmsh_name="gmsh")
        connectivity = tags[None, :]
        ids = np.array([eid])

        def __len__(self) -> int:
            return 1

    return SimpleNamespace(nodes=SimpleNamespace(ids=tags, coords=xyz), elements=[_Group()])


def python_oracle() -> dict[str, Any]:
    """fixture -> element id -> {node_ids, <component>: corner values}, every multi-point element."""
    out: dict[str, Any] = {}
    for name, (stem, code, npe, comps) in FILES.items():
        per: dict[str, Any] = {}
        with NativeReader(FIXTURES / name) as r:
            sid = r.stages()[0].id
            for comp in comps:
                eidx = np.asarray(r.read_gauss(sid, comp).element_index)
                for eid in np.unique(eidx):
                    if int((eidx == eid).sum()) < 2:
                        continue
                    one = r.read_gauss(sid, comp, element_ids=np.array([eid]))
                    node_ids, nodal = extrapolate_gauss_slab_to_nodes(one, _fem(stem, int(eid), code, npe))
                    entry = per.setdefault(str(int(eid)), {"node_ids": [int(n) for n in node_ids]})
                    entry[comp] = [float(v) for v in nodal[-1]]
        out[name] = per
    return out


def test_the_app_oracle_is_pythons_extrapolation() -> None:
    """The committed oracle equals what Python computes now, and is the closed
    form: each corner's own coordinate (the strain at a Gauss point is its
    position). Read through the older rows it was the corners rotated."""
    want = python_oracle()
    have = json.loads(ORACLE.read_text(encoding="utf-8"))
    assert sorted(have) == sorted(want)
    for name, per in want.items():
        stem = FILES[name][0]
        assert sorted(have[name]) == sorted(per) and len(per) == 2, name
        for eid, entry in per.items():
            assert have[name][eid]["node_ids"] == entry["node_ids"]
            for comp, axis in zip(FILES[name][3], range(3)):
                np.testing.assert_allclose(have[name][eid][comp], entry[comp], rtol=1e-12, atol=1e-14,
                                           err_msg=f"{name} {eid} {comp}")
                np.testing.assert_allclose(entry[comp], np.asarray(CORNERS[stem])[:, axis], rtol=0, atol=1e-10,
                                           err_msg=f"{name} {eid} {comp} vs the corners")


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


if __name__ == "__main__":
    ORACLE.write_text(json.dumps(python_oracle(), indent=1) + "\n", encoding="utf-8")
    print("wrote", ORACLE)
