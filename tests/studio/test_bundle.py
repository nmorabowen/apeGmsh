"""ADR 0095 S4c — ReportBundle, results_pin, emit_report(markdown)."""
from __future__ import annotations

import json
from pathlib import Path

from apeGmsh.studio.__main__ import main
from apeGmsh.studio._bundle import (
    BUNDLE_SCHEMA,
    collect_bundle,
    emit_report,
    results_pin,
)
from apeGmsh.studio._ledger import append_run, make_record
from apeGmsh.studio._status import collect_status


def _names(tmp_path: Path) -> None:
    (tmp_path / ".apegmsh").mkdir(exist_ok=True)
    (tmp_path / ".apegmsh" / "names.json").write_text(
        json.dumps({
            "schema": 1,
            "phase": "mesh",
            "labels": ["body"],
            "physical_groups": ["Body"],
            "counts": {"entities": {"3": 1}, "elements": {"3": 12}},
        })
        + "\n",
        encoding="utf-8",
    )


def test_pin_and_status_keep_last_run(tmp_path: Path) -> None:
    _names(tmp_path)
    append_run(
        tmp_path / ".apegmsh" / "runs.jsonl",
        make_record(
            script=tmp_path / "box.py",
            phase="mesh",
            ok=True,
            hash="abc",
            ts="2026-08-14T12:00:00Z",
        ),
    )
    model = tmp_path / "model.h5"
    model.write_bytes(b"h5")
    visor_dir = tmp_path / ".apegmsh" / "visors"
    visor_dir.mkdir(parents=True)
    png = visor_dir / "contour.png"
    png.write_bytes(b"png")
    pinned = results_pin(model, root=tmp_path)
    assert pinned["ok"] is True
    assert pinned["id"].startswith("pin-")
    payload = collect_status(tmp_path)
    assert payload["n_runs"] == 1
    assert payload["last_run"]["phase"] == "mesh"
    assert payload["n_pins"] == 1
    assert payload["last_pin"]["id"] == pinned["id"]
    assert payload["last_pin"]["model_h5"]["sha256"]
    assert str(png.resolve()) in payload["last_pin"]["visors"]


def test_pin_needs_a_file(tmp_path: Path) -> None:
    try:
        results_pin(root=tmp_path)
    except ValueError as exc:
        assert "model_h5" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_emit_markdown_promotes_visors(tmp_path: Path) -> None:
    _names(tmp_path)
    model = tmp_path / "model.h5"
    model.write_bytes(b"h5")
    visor_dir = tmp_path / ".apegmsh" / "visors"
    visor_dir.mkdir(parents=True)
    (visor_dir / "mesh.png").write_bytes(b"png")
    pin = results_pin(model, assess={"text": "Verdict: ok\n"}, root=tmp_path)
    out = emit_report(root=tmp_path)
    chapter = Path(out["path"])
    assert chapter == (tmp_path / "docs" / "studio-report.md").resolve()
    text = chapter.read_text(encoding="utf-8")
    assert "ReportBundle schema 1" in text
    assert pin["id"] in text
    assert "body" in text
    assert "Body" in text
    assert "Verdict: ok" in text
    assert "figures/" in text
    copied = tmp_path / "docs" / "figures" / pin["id"] / "mesh.png"
    assert copied.is_file()
    assert copied.read_bytes() == b"png"
    assert ".apegmsh" not in chapter.as_posix() or "visors" in text


def test_emit_refuses_apegmsh(tmp_path: Path) -> None:
    model = tmp_path / "model.h5"
    model.write_bytes(b"h5")
    results_pin(model, root=tmp_path)
    try:
        emit_report(output=tmp_path / ".apegmsh" / "report.md", root=tmp_path)
    except ValueError as exc:
        assert "INV-12" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_emit_html_not_shipped(tmp_path: Path) -> None:
    try:
        emit_report(format="html", root=tmp_path)
    except ValueError as exc:
        assert "not shipped" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_collect_bundle_schema(tmp_path: Path) -> None:
    _names(tmp_path)
    bundle = collect_bundle(root=tmp_path)
    assert bundle["schema"] == BUNDLE_SCHEMA
    assert bundle["names"]["labels"] == ["body"]


def test_pin_cli_json(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    model = tmp_path / "model.h5"
    model.write_bytes(b"h5")
    assert main(["--pin", "--model-h5", str(model), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["id"].startswith("pin-")


def test_emit_cli_html_exits_2(tmp_path: Path, capsys) -> None:
    assert main(["--emit-report", "--format", "html"]) == 2
    assert "not shipped" in capsys.readouterr().err
