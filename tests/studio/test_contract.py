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
    "selection", "names", "ledger", "highlight", "status", "host", "project", "busy",
    "assess",
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
