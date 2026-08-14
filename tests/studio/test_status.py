"""ADR 0095 S2d — inspect .apegmsh/ without replay."""
from __future__ import annotations

import json
from pathlib import Path

from apeGmsh.studio import collect_status, format_status, write_envelope
from apeGmsh.studio._envelope import PickEvidence, SelectionEnvelope
from apeGmsh.studio._ledger import append_run, make_record
from apeGmsh.studio._status import has_studio_state
from apeGmsh.studio.__main__ import main


def test_empty_status(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    payload = collect_status(tmp_path)
    assert has_studio_state(payload) is False
    assert "no .apegmsh state" in format_status(payload)
    assert main(["--status"]) == 2


def test_status_from_names_and_ledger(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    names = {
        "schema": 1,
        "phase": "model",
        "labels": ["body"],
        "physical_groups": ["Body"],
        "counts": {"entities": {"3": 1}, "elements": {"3": 0}},
        "entities": [],
    }
    (tmp_path / ".apegmsh").mkdir()
    (tmp_path / ".apegmsh" / "names.json").write_text(
        json.dumps(names) + "\n", encoding="utf-8"
    )
    append_run(
        tmp_path / ".apegmsh" / "runs.jsonl",
        make_record(
            script=tmp_path / "box.py",
            phase="model",
            ok=True,
            hash="abc",
            stopped_at="g.mesh.generation.generate",
            session="studio_box",
            manifest=names,
            ts="2026-08-14T06:00:00Z",
        ),
    )
    write_envelope(
        tmp_path / ".apegmsh" / "selection.json",
        SelectionEnvelope(
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
                ),
            ),
        ),
    )
    payload = collect_status(tmp_path)
    assert payload["n_runs"] == 1
    assert payload["last_run"]["phase"] == "model"
    assert payload["names"]["labels"] == ["body"]
    assert payload["selection"]["labels"] == ("body",)
    text = format_status(payload)
    assert "phase=model" in text
    assert "Body" in text
    assert "labels=body" in text
    code = main(["--status"])
    assert code == 0
    out = capsys.readouterr().out
    assert "studio status" in out
    assert "Body" in out


def test_status_json_flag(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".apegmsh").mkdir()
    (tmp_path / ".apegmsh" / "names.json").write_text(
        json.dumps({"phase": "mesh", "labels": [], "physical_groups": ["Rock"]})
        + "\n",
        encoding="utf-8",
    )
    code = main(["--status", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["names"]["physical_groups"] == ["Rock"]


def test_main_without_script_or_status(capsys) -> None:
    assert main([]) == 2
    assert "--status" in capsys.readouterr().err
