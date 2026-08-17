"""ADR 0095 S4c — ReportBundle, results_pin, emit_report(markdown)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from apeGmsh.studio.__main__ import main
from apeGmsh.studio._assess_snapshot import write_snapshot
from apeGmsh.studio._bundle import (
    BUNDLE_SCHEMA,
    collect_bundle,
    emit_report,
    results_pin,
)
from apeGmsh.studio._ledger import append_run, make_record
from apeGmsh.studio._status import collect_status


def _assess_snapshot(tmp_path: Path, *, text: str = "Verdict: ok\n") -> None:
    write_snapshot(
        tmp_path,
        {
            "schema": 1,
            "ts": "2026-08-17T00:00:00Z",
            "targets": {"path": "model.h5", "model_h5": None},
            "findings": [],
            "skipped": [],
            "figures": [],
            "text": text,
        },
    )


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
    _assess_snapshot(tmp_path)
    pin = results_pin(model, root=tmp_path)
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
    assert out["archive"] == str(chapter)


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


def test_emit_html_print_skin(tmp_path: Path) -> None:
    _names(tmp_path)
    model = tmp_path / "model.h5"
    model.write_bytes(b"h5")
    visor_dir = tmp_path / ".apegmsh" / "visors"
    visor_dir.mkdir(parents=True)
    (visor_dir / "mesh.png").write_bytes(b"png")
    _assess_snapshot(tmp_path)
    pin = results_pin(model, root=tmp_path)
    out = emit_report(format="html", root=tmp_path)
    html_path = Path(out["path"])
    archive = Path(out["archive"])
    assert html_path == (tmp_path / "docs" / "studio-report.html").resolve()
    assert archive == (tmp_path / "docs" / "studio-report.md").resolve()
    assert html_path.is_file()
    assert archive.is_file()
    text = html_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in text
    assert pin["id"] in text
    assert "body" in text
    assert "Verdict: ok" in text
    assert "figures/" in text
    assert archive.is_file()
    assert "ReportBundle schema 1" in archive.read_text(encoding="utf-8")


def test_emit_html_escapes_labels(tmp_path: Path) -> None:
    (tmp_path / ".apegmsh").mkdir(exist_ok=True)
    (tmp_path / ".apegmsh" / "names.json").write_text(
        json.dumps({
            "schema": 1,
            "phase": "model",
            "labels": ["<script>alert(1)</script>"],
            "physical_groups": ["Body"],
        })
        + "\n",
        encoding="utf-8",
    )
    model = tmp_path / "model.h5"
    model.write_bytes(b"h5")
    results_pin(model, root=tmp_path)
    out = emit_report(format="html", root=tmp_path)
    text = Path(out["path"]).read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text


def test_emit_canvas_to_output(tmp_path: Path) -> None:
    _names(tmp_path)
    model = tmp_path / "model.h5"
    model.write_bytes(b"h5")
    visor_dir = tmp_path / ".apegmsh" / "visors"
    visor_dir.mkdir(parents=True)
    (visor_dir / "mesh.png").write_bytes(b"png")
    _assess_snapshot(tmp_path)
    pin = results_pin(model, root=tmp_path)
    dest = tmp_path / "review.canvas.tsx"
    out = emit_report(format="canvas", output=dest, root=tmp_path)
    canvas = Path(out["path"])
    archive = Path(out["archive"])
    assert canvas == dest.resolve()
    assert canvas.is_file()
    assert archive == (tmp_path / "docs" / "studio-report.md").resolve()
    text = canvas.read_text(encoding="utf-8")
    assert 'from "cursor/canvas"' in text
    assert "export default function StudioReport" in text
    assert pin["id"] in text
    assert "body" in text
    assert "Verdict: ok" in text
    ast_ok = "const BUNDLE = " in text
    assert ast_ok
    assert archive.is_file()


def test_emit_canvas_needs_output_without_ide_dir(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.delenv("CURSOR_CANVAS_DIR", raising=False)
    _names(tmp_path)
    model = tmp_path / "model.h5"
    model.write_bytes(b"h5")
    results_pin(model, root=tmp_path)
    try:
        emit_report(format="canvas", root=tmp_path)
    except ValueError as exc:
        assert "output=" in str(exc)
        assert "canvas" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_emit_canvas_uses_cursor_canvas_dir(
    tmp_path: Path, monkeypatch,
) -> None:
    _names(tmp_path)
    model = tmp_path / "model.h5"
    model.write_bytes(b"h5")
    results_pin(model, root=tmp_path)
    canvases = tmp_path / "ide-canvases"
    canvases.mkdir()
    monkeypatch.setenv("CURSOR_CANVAS_DIR", str(canvases))
    out = emit_report(format="canvas", root=tmp_path)
    assert Path(out["path"]) == (canvases / "studio-report.canvas.tsx").resolve()
    assert Path(out["path"]).is_file()


def test_emit_canvas_refuses_wrong_suffix(tmp_path: Path) -> None:
    _names(tmp_path)
    model = tmp_path / "model.h5"
    model.write_bytes(b"h5")
    results_pin(model, root=tmp_path)
    try:
        emit_report(
            format="canvas",
            output=tmp_path / "docs" / "studio-report.md",
            root=tmp_path,
        )
    except ValueError as exc:
        assert ".canvas.tsx" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_cursor_project_slug_matches_cursor_convention() -> None:
    from apeGmsh.studio._skins import cursor_project_slug

    if sys.platform == "win32":
        slug = cursor_project_slug(Path(r"C:\Users\nmb\Documents\Github\apeGmsh"))
        assert slug == "c-Users-nmb-Documents-Github-apeGmsh"
        wt = cursor_project_slug(Path(r"C:\Users\nmb\.cursor\worktrees\apeGmsh\i0b4"))
        assert wt == "c-Users-nmb-cursor-worktrees-apeGmsh-i0b4"
        return
    slug = cursor_project_slug(Path("/home/nmb/Documents/Github/apeGmsh"))
    assert slug == "-home-nmb-Documents-Github-apeGmsh"
    wt = cursor_project_slug(Path("/home/nmb/.cursor/worktrees/apeGmsh/i0b4"))
    assert wt == "-home-nmb-cursor-worktrees-apeGmsh-i0b4"


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


def test_emit_cli_html_writes_docs(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".apegmsh").mkdir()
    model = tmp_path / "model.h5"
    model.write_bytes(b"h5")
    assert main(["--pin", "--model-h5", str(model), "--json"]) == 0
    capsys.readouterr()
    assert main(["--emit-report", "--format", "html", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["format"] == "html"
    assert Path(payload["path"]).name == "studio-report.html"
    assert Path(payload["archive"]).name == "studio-report.md"
