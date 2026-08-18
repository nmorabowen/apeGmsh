"""ADR 0095 S5d — published schemas + goldens (INV-17)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from apeGmsh.studio import SelectionEnvelope, collect_status, write_envelope
from apeGmsh.studio._contract import (
    CONTRACT_VERSION,
    ContractError,
    check_contract_version,
    fixtures_dir,
    load_fixture,
    load_schema,
    schemas_dir,
    validate,
)
from apeGmsh.studio._envelope import PickEvidence
from apeGmsh.studio._highlight import write_highlight
from apeGmsh.studio._ledger import make_record
from apeGmsh.studio._names import write_names

_KINDS = (
    "selection", "names", "ledger", "ledger_pin", "highlight", "status", "host",
    "project", "busy", "assess",
)


@pytest.mark.parametrize("kind", _KINDS)
def test_fixture_validates_against_schema(kind: str) -> None:
    schema = load_schema(kind)
    assert schema.get("type") == "object"
    fixture = load_fixture(kind)
    validate(kind, fixture)


@pytest.mark.parametrize("kind", _KINDS)
def test_schema_and_fixture_files_exist(kind: str) -> None:
    assert (schemas_dir() / f"{kind}.schema.json").is_file()
    assert (fixtures_dir() / f"{kind}.json").is_file()


def test_unknown_contract_major_refused() -> None:
    with pytest.raises(ContractError, match="unsupported contract major"):
        check_contract_version("2.0.0")
    check_contract_version(CONTRACT_VERSION)
    check_contract_version("1.99.0")


def test_wrong_file_schema_int_refused() -> None:
    data = load_fixture("names")
    data = {**data, "schema": 99}
    with pytest.raises(ContractError, match="schema=99"):
        validate("names", data)


def test_status_carries_contract_version(tmp_path: Path) -> None:
    payload = collect_status(tmp_path)
    assert payload["contract_version"] == CONTRACT_VERSION
    validate("status", payload)


def test_live_writers_match_contract(tmp_path: Path) -> None:
    env = SelectionEnvelope(
        phase="model",
        substrate="model_brep",
        labels=("body",),
        physical_groups=("Body",),
        unnamed=(),
        evidence=(
            PickEvidence(
                substrate="model_brep",
                dim=3,
                key=1,
                labels=("body",),
                physical_groups=("Body",),
                bbox=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            ),
        ),
        bbox=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
    )
    sel_path = tmp_path / "selection.json"
    write_envelope(sel_path, env)
    validate("selection", json.loads(sel_path.read_text(encoding="utf-8")))

    names_path = tmp_path / "names.json"
    write_names(
        names_path,
        {
            "schema": 1,
            "phase": "model",
            "labels": ["body"],
            "physical_groups": ["Body"],
            "counts": {"entities": {"3": 1}, "elements": {"3": 0}},
            "entities": [],
        },
    )
    validate("names", json.loads(names_path.read_text(encoding="utf-8")))

    write_highlight(tmp_path / "highlight.json", names=["body"], dimtags=[(3, 1)])
    validate(
        "highlight",
        json.loads((tmp_path / "highlight.json").read_text(encoding="utf-8")),
    )

    rec = make_record(
        script=tmp_path / "box.py",
        phase="model",
        ok=True,
        hash="abc",
        root=tmp_path,
        cwd=tmp_path,
        ts="2026-08-16T05:00:00Z",
    )
    validate("ledger", rec)


# =====================================================================
# ADR 0098 S5b — runs.jsonl has TWO record kinds, both published
# =====================================================================

def _pin_a_session(tmp_path: Path) -> dict:
    """One REAL pin record, from the real writer, with a real snapshot."""
    from apeGmsh.results.session import Contour, ResultsSession, save_snapshot
    from apeGmsh.studio._bundle import results_pin

    (tmp_path / ".apegmsh").mkdir(exist_ok=True)
    session = ResultsSession()
    session.add_view().contour = Contour("stress_xx")
    snapshot = tmp_path / "out.h5.session.json"
    save_snapshot(session, snapshot)
    model = tmp_path / "model.h5"
    model.write_bytes(b"pretend h5")

    return results_pin(
        model_h5=model, session_snapshot=snapshot, root=tmp_path,
    )["pin"]


def test_live_pin_writer_matches_the_published_pin_schema(tmp_path: Path):
    """The drift guard the pin record never had.

    ``ledger_pin.schema.json`` is hand-written and ``results_pin`` is
    Python; nothing compared them until this test. It goes red if either
    side moves — which is the only reason a published contract is worth
    anything to a consumer that cannot import the FEM stack.
    """
    record = _pin_a_session(tmp_path)

    validate("ledger_pin", record)
    assert record["kind"] == "pin"
    assert set(record["session_snapshot"]) == {"path", "sha256", "size"}


def test_a_pin_line_is_refused_by_the_run_schema(tmp_path: Path):
    """Documents WHY there are two schemas rather than one.

    A pin line has no ``script`` / ``phase`` / ``ok``. Relaxing the run
    schema's ``required`` list to admit both would stop it catching a run
    line that lost its script — so a consumer discriminates on ``kind``
    instead, and this asserts the discrimination is real.
    """
    record = _pin_a_session(tmp_path)

    with pytest.raises(ContractError, match="missing required 'script'"):
        validate("ledger", record)


def test_a_run_line_is_refused_by_the_pin_schema(tmp_path: Path):
    """The other direction — or `ledger_pin` would just be a looser
    `ledger` and the split would buy nothing."""
    run = make_record(
        script=tmp_path / "box.py", phase="model", ok=True,
        root=tmp_path, cwd=tmp_path, ts="2026-08-16T05:00:00Z",
    )

    with pytest.raises(ContractError):
        validate("ledger_pin", run)


def test_the_pin_schema_pins_the_kind_and_not_merely_its_presence(
    tmp_path: Path,
):
    """A record carrying a DIFFERENT ``kind`` must be refused.

    The discriminating case, found by a surviving mutation: dropping the
    schema's ``const: "pin"`` broke nothing, because the only other
    record this file holds — a run line — has no ``kind`` at all and is
    already refused as *missing required*. So neither direction of the
    existing pair could see the const disappear. A third kind in
    ``runs.jsonl`` is the shape that can: without the const it would
    validate as a pin and be handed to a pin reader.
    """
    record = _pin_a_session(tmp_path)

    validate("ledger_pin", record)  # unchanged, it really is a pin

    for foreign in ("checkpoint", "run", "PIN", ""):
        with pytest.raises(ContractError):
            validate("ledger_pin", {**record, "kind": foreign})


def test_every_line_of_a_real_ledger_validates_under_one_of_the_two(
    tmp_path: Path,
):
    """The consumer's actual walk: open runs.jsonl, dispatch on ``kind``.

    Both kinds land in ONE file, so this is the loop a Workbench-style
    reader writes — and before S5b its pin branch had no schema to call.
    """
    from apeGmsh.studio._ledger import append_run, read_runs
    from apeGmsh.studio._paths import ledger_path

    _pin_a_session(tmp_path)
    append_run(ledger_path(tmp_path), make_record(
        script=tmp_path / "box.py", phase="model", ok=True,
        root=tmp_path, cwd=tmp_path, ts="2026-08-16T05:00:00Z",
    ))

    rows = read_runs(ledger_path(tmp_path))
    kinds = []
    for row in rows:
        kind = "ledger_pin" if row.get("kind") == "pin" else "ledger"
        validate(kind, row)
        kinds.append(kind)

    assert sorted(kinds) == ["ledger", "ledger_pin"]
