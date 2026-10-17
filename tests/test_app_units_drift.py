"""Consistent unit systems in ``apeGmsh.units`` equal ``apeGmshViewer/src/state/units.ts``.

This test parses ``apeGmshViewer/src/state/units.ts`` as text (like
``test_app_ladruno_gate_drift.py``) and checks that ``UNIT_SYSTEM_IDS`` and
``UNIT_SYSTEMS`` (id, label, F, L, T, symbols) match ``apeGmsh.units`` bit-for-bit.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from apeGmsh.units import UNIT_SYSTEM_IDS, UNIT_SYSTEMS, UnitSystemSpec

UNITS_TS = (
    Path(__file__).resolve().parents[1]
    / "apeGmshViewer" / "src" / "state" / "units.ts"
)

_IDS_RE = re.compile(
    r"export const UNIT_SYSTEM_IDS\s*:\s*readonly\s+UnitSystemId\[\]\s*=\s*\[(?P<body>[^\]]+)\];",
    re.MULTILINE,
)

_SYSTEM_ENTRY_RE = re.compile(
    r"(?P<key>\w+)\s*:\s*\{\s*"
    r'id:\s*"(?P<id>[^"]+)",\s*'
    r'label:\s*"(?P<label>[^"]+)",\s*'
    r"F:\s*(?P<F>[-+0-9.eE]+),\s*"
    r"L:\s*(?P<L>[-+0-9.eE]+),\s*"
    r"T:\s*(?P<T>[-+0-9.eE]+),\s*"
    r"symbols:\s*\{\s*"
    r'force:\s*"(?P<force>[^"]+)",\s*'
    r'length:\s*"(?P<length>[^"]+)",\s*'
    r'time:\s*"(?P<time>[^"]+)",\s*'
    r'pressure:\s*"(?P<pressure>[^"]+)"\s*'
    r"\},?\s*\}",
    re.MULTILINE,
)


@pytest.fixture(scope="module")
def units_ts() -> str:
    assert UNITS_TS.is_file(), f"{UNITS_TS} is missing"
    return UNITS_TS.read_text(encoding="utf-8")


def test_app_unit_system_ids_equal_python(units_ts: str) -> None:
    m = _IDS_RE.search(units_ts)
    assert m is not None, f"{UNITS_TS}: could not find UNIT_SYSTEM_IDS array"
    ts_ids = tuple(re.findall(r'"([^"]+)"', m.group("body")))
    assert ts_ids == UNIT_SYSTEM_IDS


def test_app_unit_systems_equal_python(units_ts: str) -> None:
    ts_specs: dict[str, UnitSystemSpec] = {}
    for m in _SYSTEM_ENTRY_RE.finditer(units_ts):
        key = m.group("key")
        assert key == m.group("id"), f"Key {key!r} != id {m.group('id')!r} in {UNITS_TS}"
        ts_specs[key] = UnitSystemSpec(
            id=m.group("id"),  # type: ignore[arg-type]
            label=m.group("label"),
            F=float(m.group("F")),
            L=float(m.group("L")),
            T=float(m.group("T")),
            force_symbol=m.group("force"),
            length_symbol=m.group("length"),
            time_symbol=m.group("time"),
            pressure_symbol=m.group("pressure"),
        )
    assert set(ts_specs.keys()) == set(UNIT_SYSTEMS.keys())
    for uid, py_spec in UNIT_SYSTEMS.items():
        assert ts_specs[uid] == py_spec, f"Drift in {uid}: TS={ts_specs[uid]} vs PY={py_spec}"

