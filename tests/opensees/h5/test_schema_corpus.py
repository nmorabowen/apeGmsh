"""The schema corpus: real old files open through today's readers (ADR 0113 D8).

``tests/fixtures/schema_corpus/`` holds one ``model.h5`` per schema minor,
written by that minor's frozen writer checked out of git
(``scripts/build_schema_corpus.py``), with the semantic dump the era's
**own** reader recorded beside it. ``MANIFEST.json`` names every minor
from each zone's floor to its current minor, ``ok`` or ``gap``.

Invariants held here (ADR 0113):

* INV 5, corpus opens: every ``ok`` file opens through ``FEMData.from_h5``
  (and ``OpenSeesModel.from_h5`` when it has ``/opensees``), and today's
  dump equals the era's except for the fields the shim ledger names;
* INV 6, corpus complete: one entry per minor from floor to current, each
  ``ok`` with its commit, or a ``gap`` with its reason;
* INV 7, bytes unchanged: opening a file never changes its bytes;
* INV 10, tamper check kept: a tampered copy fails the snapshot_id check;
* INV 12, paper floor fails: a floor set to the current minor turns the
  corpus check red;
* D3, evidence-gated floors: a floor stands only if no runnable era at or
  above it is a gap. A zone whose floor the corpus cannot prove is listed in
  ``UNPROVEN_FLOORS`` as a strict xfail until the maintainer raises it.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from apeGmsh.mesh import _femdata_h5_io
from apeGmsh.mesh.FEMData import FEMData
from apeGmsh.opensees._internal.schema_version import (
    NEUTRAL,
    OPENSEES,
    SchemaVersion,
    SchemaVersionError,
    reader_floor,
    reader_version,
)
from apeGmsh.opensees.emitter import h5 as _h5_writer
from apeGmsh.opensees.emitter.h5_reader import (
    META_NDM_IS_SPATIAL_FROM,
    MalformedH5Error,
)
from apeGmsh.opensees.opensees_model import OpenSeesModel
from tests.fixtures.schema import NEUTRAL_CURRENT, OPENSEES_CURRENT
from tests.fixtures.schema_corpus._semantic_dump import dump_fem, dump_model

CORPUS = Path(__file__).resolve().parents[2] / "fixtures" / "schema_corpus"
MANIFEST: dict[str, Any] = json.loads((CORPUS / "MANIFEST.json").read_text(encoding="utf-8"))
ENTRIES: list[dict[str, Any]] = MANIFEST["entries"]
ZONES = {"neutral": NEUTRAL, "opensees": OPENSEES}
STAMP_KEYS = {"neutral": "neutral_schema_version", "opensees": "opensees_schema_version"}

#: The ops.model ndm every opensees-era generator declares
#: (``_era_generator.opensees_frame``): the closed-form answer for ``ndm``.
DECLARED_NDM = 3

#: Pre-2.26.1 SP loads read as one ``default`` case (ADR 0113 D4 ledger, Q5).
SP_PER_CASE_FROM = (2, 26, 1)

#: The shim ledger of ADR 0113 D4, as (neutral version the field is trusted
#: from, dump paths excluded from the era comparison below it).
SHIM_LEDGER: tuple[tuple[tuple[int, int, int], tuple[str, ...]], ...] = (
    (META_NDM_IS_SPATIAL_FROM, ("model.ndm",)),
    (SP_PER_CASE_FROM, ("fem.loads.sp_patterns", "model.fem.loads.sp_patterns")),
)

#: Zones whose floor constant the corpus cannot prove (ADR 0113 D3: the
#: floor must rise past the gap; that is a maintainer decision, recorded on
#: #1303). Strict xfail: raising the floor turns it XPASS, which fails until
#: the entry is removed here.
UNPROVEN_FLOORS: dict[str, str] = {
    "opensees": (
        "every opensees-2.11 writer stamped neutral 2.7.0, below the neutral "
        "floor 2.10.0, so no opensees-2.11 file opens; the floor must rise to 2.12"
    ),
}


def _v(s: str) -> tuple[int, int, int]:
    p = SchemaVersion.parse(s)
    return (p.major, p.minor, p.patch)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _with_files() -> list[dict[str, Any]]:
    return [e for e in ENTRIES if "files" in e]


def _openable() -> list[dict[str, Any]]:
    return [e for e in ENTRIES if e["status"] == "ok"]


def _id(e: dict[str, Any]) -> str:
    return f"{e['zone']}-{e['minor']}"


def _excluded(entry: dict[str, Any]) -> set[str]:
    neutral = _v(entry["stamps"]["neutral"])
    return {p for since, paths in SHIM_LEDGER if neutral < since for p in paths}


def _diff(era: Any, today: Any, path: str, skip: set[str]) -> list[str]:
    """Paths where ``today`` does not reproduce what ``era`` recorded.

    Every key the era's reader produced must come back with an equal value.
    A key only today's reader knows is an additive field the era had no name
    for, so it is not compared.
    """
    if path in skip:
        return []
    if isinstance(era, dict):
        if not isinstance(today, dict):
            return [f"{path}: era dict, today {type(today).__name__}"]
        out: list[str] = []
        for k, v in era.items():
            sub = f"{path}.{k}" if path else k
            if sub in skip:
                continue
            if k not in today:
                out.append(f"{sub}: missing today")
            else:
                out += _diff(v, today[k], sub, skip)
        return out
    if isinstance(era, list):
        if not isinstance(today, list) or len(era) != len(today):
            return [f"{path}: era {era!r} != today {today!r}"]
        return [d for i, (a, b) in enumerate(zip(era, today))
                for d in _diff(a, b, f"{path}[{i}]", skip)]
    return [] if era == today else [f"{path}: era {era!r} != today {today!r}"]


def check_entry(entry: dict[str, Any]) -> None:
    """Open one corpus file through today's readers and compare the dumps."""
    h5 = CORPUS / entry["files"]["h5"]["name"]
    era = json.loads((CORPUS / entry["files"]["dump"]["name"]).read_text(encoding="utf-8"))
    with h5py.File(h5, "r") as f:
        has_bridge = "opensees" in f
    today: dict[str, Any] = {"fem": dump_fem(FEMData.from_h5(str(h5)))}
    if has_bridge:
        model = OpenSeesModel.from_h5(str(h5))
        today["model"] = dump_model(model)
        assert model.ndm == DECLARED_NDM, (
            f"{h5.name}: today's reader resolves ndm={model.ndm}, "
            f"the generator declared {DECLARED_NDM}"
        )
    assert ("model" in era) == has_bridge
    diffs = _diff({k: era[k] for k in ("fem", "model") if k in era}, today, "",
                  _excluded(entry))
    assert not diffs, f"{h5.name}: today's reader departs from its era:\n" + "\n".join(diffs)


# ---------------------------------------------------------------------------
# INV 6 — one entry per minor from floor to current
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("zone", sorted(ZONES))
def test_corpus_covers_every_minor_from_floor_to_current(zone: str) -> None:
    floor, current = reader_floor(ZONES[zone]), reader_version(ZONES[zone])
    have = {e["minor"]: e for e in ENTRIES if e["zone"] == zone}
    want = [f"{floor.major}.{m}" for m in range(floor.minor, current.minor + 1)]
    assert sorted(have, key=lambda m: int(m.split(".")[1])) == want, (
        f"{zone}: the manifest must hold exactly the minors {want[0]}..{want[-1]}; "
        "rerun scripts/build_schema_corpus.py after a bump or a floor change"
    )
    for minor, e in have.items():
        if e["status"] == "ok":
            assert e.get("sha") and e.get("files"), f"{zone} {minor}: ok without a file"
            assert _v(e["stamps"][zone])[:2] == _v(f"{minor}.0")[:2]
        else:
            assert e["status"] == "gap" and e.get("reason"), f"{zone} {minor}: {e}"
            assert e.get("gap_kind") in {"unwritten", "unrunnable", "below_floor"}


def test_current_minor_files_match_the_test_fixture_constants() -> None:
    current = {e["zone"]: e for e in ENTRIES if e["status"] == "ok"}
    assert MANIFEST["current"]["neutral"] == NEUTRAL_CURRENT
    assert MANIFEST["current"]["opensees"] == OPENSEES_CURRENT
    assert current  # at least one openable file per zone is checked below
    for zone in ZONES:
        assert any(e["zone"] == zone for e in _openable()), f"no openable {zone} file"


# ---------------------------------------------------------------------------
# INV 5 + INV 7 — every ok file opens, matches its era, bytes unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", _openable(), ids=_id)
def test_corpus_file_opens_and_matches_its_era(entry: dict[str, Any]) -> None:
    h5 = CORPUS / entry["files"]["h5"]["name"]
    before = _sha(h5)
    assert before == entry["files"]["h5"]["sha256"], (
        f"{h5.name}: committed bytes differ from the manifest; corpus files are "
        "frozen, rebuild with scripts/build_schema_corpus.py"
    )
    check_entry(entry)
    assert _sha(h5) == before, f"{h5.name}: opening the file changed its bytes"


# ---------------------------------------------------------------------------
# A below-floor gap keeps its real file, and today's readers refuse it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entry", [e for e in _with_files() if e["status"] == "gap"], ids=_id,
)
def test_below_floor_gap_is_refused_by_the_floor(entry: dict[str, Any]) -> None:
    assert entry["gap_kind"] == "below_floor"
    low = [z for z, s in entry["stamps"].items()
           if z in ZONES and SchemaVersion.parse(s).minor < reader_floor(ZONES[z]).minor]
    assert low, f"{_id(entry)}: recorded below the floor but no stamp is"
    h5 = CORPUS / entry["files"]["h5"]["name"]
    before = _sha(h5)
    with pytest.raises(SchemaVersionError, match="supports"):
        check_entry(entry)
    assert _sha(h5) == before


# ---------------------------------------------------------------------------
# INV 10 — a tampered copy fails the snapshot_id check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", _openable(), ids=_id)
def test_tampered_copy_fails_snapshot_check(entry: dict[str, Any], tmp_path: Path) -> None:
    src = CORPUS / entry["files"]["h5"]["name"]
    copy = tmp_path / src.name
    shutil.copyfile(src, copy)
    with h5py.File(copy, "r+") as f:
        coords = np.asarray(f["nodes/coords"][...])
        coords[-1, 0] += 0.25
        f["nodes/coords"][...] = coords
    with pytest.raises(MalformedH5Error, match="snapshot_id mismatch"):
        FEMData.from_h5(str(copy))


# ---------------------------------------------------------------------------
# D3 — the floor constants are proven by the corpus
# ---------------------------------------------------------------------------


def _unproven_minors(zone: str) -> list[str]:
    floor = reader_floor(ZONES[zone])
    return [
        e["minor"] for e in ENTRIES
        if e["zone"] == zone and e["status"] == "gap"
        and e["gap_kind"] != "unwritten"
        and _v(f"{e['minor']}.0")[1] >= floor.minor
    ]


@pytest.mark.parametrize("zone", [
    pytest.param(z, marks=pytest.mark.xfail(strict=True, reason=UNPROVEN_FLOORS[z]))
    if z in UNPROVEN_FLOORS else z
    for z in sorted(ZONES)
])
def test_floor_is_proven_by_the_corpus(zone: str) -> None:
    """A floor stands only where no writable era at or above it is a gap.

    An ``unwritten`` minor (no main commit ever stamped it) holds no files in
    the wild and does not count against the floor.
    """
    gaps = _unproven_minors(zone)
    assert not gaps, (
        f"{zone} floor {reader_floor(ZONES[zone])} is not proven: gaps at {gaps}. "
        "ADR 0113 D3: the floor must rise past them (a maintainer decision)."
    )


# ---------------------------------------------------------------------------
# INV 12 — a paper floor turns the corpus check red
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("zone,module,const", [
    ("neutral", _femdata_h5_io, "NEUTRAL_SCHEMA_FLOOR"),
    ("opensees", _h5_writer, "SCHEMA_FLOOR"),
])
def test_paper_floor_fails_the_corpus(
    zone: str, module: Any, const: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = reader_version(ZONES[zone])
    monkeypatch.setattr(module, const, str(current))
    assert reader_floor(ZONES[zone]) == current
    refused = []
    for entry in _openable():
        if entry["zone"] != zone or _v(entry["stamps"][zone])[1] == current.minor:
            continue
        with pytest.raises(SchemaVersionError, match="supports"):
            check_entry(entry)
        refused.append(entry["minor"])
    assert refused, f"a paper {zone} floor at {current} refused no corpus file"
