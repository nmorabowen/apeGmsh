"""ADR 0095 S4a–S4c — MCP tool bodies (no SDK, no Qt, no Gmsh)."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

from apeGmsh.studio._envelope import PickEvidence, SelectionEnvelope
from apeGmsh.studio._mcp import (
    animate,
    assess,
    emit_report,
    get_selection,
    highlight,
    promote_selection,
    render,
    results_pin,
    run_until,
    status,
)
from apeGmsh.studio._status import has_studio_state

MCP_DIR = Path(__file__).resolve().parents[2] / "src" / "apeGmsh" / "studio"
_MCP_FILES = (
    MCP_DIR / "_mcp.py",
    MCP_DIR / "mcp.py",
    MCP_DIR / "_bundle.py",
    MCP_DIR / "_highlight.py",
)


def _forbidden(module: str) -> bool:
    if module == "gmsh" or module.startswith("gmsh."):
        return True
    if module == "apeGmsh.viewers" or module.startswith("apeGmsh.viewers."):
        return True
    return False


def test_mcp_modules_do_not_import_gmsh_or_viewers() -> None:
    leaks: list[str] = []
    for path in _MCP_FILES:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _forbidden(alias.name):
                        leaks.append(f"{path.name}:{node.lineno} {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    continue
                if _forbidden(module):
                    leaks.append(f"{path.name}:{node.lineno} {module}")
    assert not leaks, "S4a MCP must not import gmsh/viewers:\n" + "\n".join(
        leaks
    )


def test_status_empty(tmp_path: Path) -> None:
    payload = status(root=tmp_path)
    assert payload["empty"] is True
    assert has_studio_state(payload) is False


def test_status_and_get_selection(tmp_path: Path) -> None:
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
    from apeGmsh.studio import write_envelope

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
    payload = status(root=tmp_path)
    assert payload["empty"] is False
    assert payload["names"]["labels"] == ["body"]
    pick = get_selection(root=tmp_path)
    assert pick["present"] is True
    assert pick["labels"] == ["body"]
    assert pick["physical_groups"] == ["Body"]
    assert pick["phase"] == "model"


def test_get_selection_missing(tmp_path: Path) -> None:
    pick = get_selection(root=tmp_path)
    assert pick["present"] is False
    assert pick["envelope"] is None


def test_run_until_missing_script(tmp_path: Path) -> None:
    result = run_until("nope.py", root=tmp_path)
    assert result["ok"] is False
    assert result["returncode"] == 2
    assert "not found" in result["stderr"]


def test_run_until_bad_phase(tmp_path: Path) -> None:
    try:
        run_until("x.py", phase="cad", root=tmp_path)
    except ValueError as exc:
        assert "phase" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_run_until_subprocesses_cli(tmp_path: Path, monkeypatch) -> None:
    script = tmp_path / "box.py"
    script.write_text("pass\n", encoding="utf-8")
    recorded: dict = {}

    def fake_run(argv, **kwargs):
        recorded["argv"] = list(argv)
        recorded["cwd"] = kwargs.get("cwd")
        return SimpleNamespace(returncode=0, stdout="replay ok\n", stderr="")

    monkeypatch.setattr("apeGmsh.studio._mcp.subprocess.run", fake_run)
    result = run_until(script, phase="mesh", root=tmp_path)
    assert result["ok"] is True
    assert recorded["cwd"] == str(tmp_path)
    argv = recorded["argv"]
    assert argv[1:3] == ["-m", "apeGmsh.studio"]
    assert str(script.resolve()) in argv
    assert "--phase" in argv and "mesh" in argv
    assert "--no-viewer" in argv
    assert result["status"]["empty"] is True


def test_assess_subprocesses_cli(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "model.h5"
    src.write_bytes(b"x")
    recorded: dict = {}

    def fake_run(argv, **kwargs):
        recorded["argv"] = list(argv)
        recorded["cwd"] = kwargs.get("cwd")
        return SimpleNamespace(
            returncode=0,
            stdout='{"ok": true, "source": "fem", "findings": [], "text": "ok"}\n',
            stderr="",
        )

    monkeypatch.setattr("apeGmsh.studio._mcp.subprocess.run", fake_run)
    result = assess(src, figures=True, root=tmp_path)
    assert result["ok"] is True
    assert result["source"] == "fem"
    argv = recorded["argv"]
    assert argv[1:3] == ["-m", "apeGmsh.studio"]
    assert "--assess" in argv and str(src.resolve()) in argv
    assert "--json" in argv and "--figures" in argv


def test_assess_missing_file(tmp_path: Path) -> None:
    result = assess("nope.h5", root=tmp_path)
    assert result["ok"] is False
    assert result["returncode"] == 2


def test_render_subprocesses_viewers_cli(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "run.h5"
    src.write_bytes(b"x")
    recorded: dict = {}

    def fake_run(argv, **kwargs):
        recorded["argv"] = list(argv)
        png = tmp_path / ".apegmsh" / "visors" / "contour.png"
        return SimpleNamespace(returncode=0, stdout=f"{png}\n", stderr="")

    monkeypatch.setattr("apeGmsh.studio._mcp.subprocess.run", fake_run)
    result = render(src, view="contour", root=tmp_path)
    assert result["ok"] is True
    argv = recorded["argv"]
    assert argv[1:4] == ["-m", "apeGmsh.viewers", "render"]
    assert "--view" in argv and "contour" in argv
    assert "--no-viewer" not in argv
    assert result["written"]


def test_render_rejects_open_view(tmp_path: Path) -> None:
    src = tmp_path / "run.h5"
    src.write_bytes(b"x")
    try:
        render(src, view="poster", root=tmp_path)
    except ValueError as exc:
        assert "view" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_animate_subprocesses_cli(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "run.h5"
    src.write_bytes(b"x")
    recorded: dict = {}

    def fake_run(argv, **kwargs):
        recorded["argv"] = list(argv)
        out = tmp_path / ".apegmsh" / "visors" / "history.gif"
        return SimpleNamespace(returncode=0, stdout=f"{out}\n", stderr="")

    monkeypatch.setattr("apeGmsh.studio._mcp.subprocess.run", fake_run)
    result = animate(src, kind="history", root=tmp_path)
    assert result["ok"] is True
    argv = recorded["argv"]
    assert argv[1:3] == ["-m", "apeGmsh.studio"]
    assert "--animate" in argv and "--kind" in argv and "history" in argv
    assert "--output" in argv


def test_animate_yield_subprocesses_cli(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "run.h5"
    src.write_bytes(b"x")
    recorded: dict = {}

    def fake_run(argv, **kwargs):
        recorded["argv"] = list(argv)
        out = tmp_path / ".apegmsh" / "visors" / "yield.gif"
        return SimpleNamespace(returncode=0, stdout=f"{out}\n", stderr="")

    monkeypatch.setattr("apeGmsh.studio._mcp.subprocess.run", fake_run)
    result = animate(src, kind="yield", root=tmp_path)
    assert result["ok"] is True
    argv = recorded["argv"]
    assert "--kind" in argv and "yield" in argv
    assert "yield.gif" in " ".join(argv)
    assert "setup" not in argv


def test_animate_formation_not_shipped(tmp_path: Path) -> None:
    src = tmp_path / "run.h5"
    src.write_bytes(b"x")
    try:
        animate(src, kind="formation", root=tmp_path)
    except ValueError as exc:
        assert "not shipped" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_results_pin_and_emit_report(tmp_path: Path) -> None:
    model = tmp_path / "model.h5"
    model.write_bytes(b"h5")
    visors = tmp_path / ".apegmsh" / "visors"
    visors.mkdir(parents=True)
    (visors / "contour.png").write_bytes(b"png")
    pin = results_pin(model_h5=model, root=tmp_path)
    assert pin["ok"] is True
    out = emit_report(root=tmp_path)
    assert out["ok"] is True
    assert out["format"] == "markdown"
    assert Path(out["path"]).is_file()
    text = Path(out["path"]).read_text(encoding="utf-8")
    assert pin["id"] in text


def test_highlight_and_promote(tmp_path: Path) -> None:
    names = {
        "schema": 1,
        "phase": "model",
        "labels": ["front"],
        "physical_groups": ["Face"],
        "entities": [
            {
                "dim": 2,
                "tag": 5,
                "labels": ["front"],
                "physical_groups": ["Face"],
                "bbox": [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            },
            {
                "dim": 2,
                "tag": 7,
                "labels": [],
                "physical_groups": [],
                "bbox": [1.0, 0.0, 0.0, 2.0, 1.0, 0.0],
            },
        ],
    }
    (tmp_path / ".apegmsh").mkdir()
    (tmp_path / ".apegmsh" / "names.json").write_text(
        json.dumps(names) + "\n", encoding="utf-8"
    )
    existing = tmp_path / ".apegmsh" / "selection.json"
    existing.write_text('{"schema": 1, "phase": "model", "keep": true}\n')
    hit = highlight(["front"], root=tmp_path)
    assert hit["ok"] is True
    assert hit["missing"] == []
    assert hit["matched"] == [[2, 5]]
    assert (tmp_path / ".apegmsh" / "highlight.json").is_file()
    assert existing.read_text(encoding="utf-8").startswith('{"schema": 1')
    assert "keep" in existing.read_text(encoding="utf-8")

    from apeGmsh.studio import write_envelope
    from apeGmsh.studio._envelope import PickEvidence, SelectionEnvelope

    write_envelope(
        tmp_path / ".apegmsh" / "selection.json",
        SelectionEnvelope(
            phase="model",
            substrate="model_brep",
            labels=(),
            physical_groups=(),
            unnamed=(
                PickEvidence(
                    substrate="model_brep",
                    dim=2,
                    key=7,
                    bbox=(1.0, 0.0, 0.0, 2.0, 1.0, 0.0),
                ),
            ),
            evidence=(
                PickEvidence(
                    substrate="model_brep",
                    dim=2,
                    key=7,
                    bbox=(1.0, 0.0, 0.0, 2.0, 1.0, 0.0),
                ),
            ),
        ),
    )
    promo = promote_selection(root=tmp_path)
    assert promo["ok"] is True
    assert promo["applied"] is False
    assert len(promo["suggestions"]) == 1
    snippet = promo["suggestions"][0]["to_label"]
    ast.parse(snippet)
    compile(snippet, "<promo>", "exec")
    assert "select(None, dim=2)" in snippet
    assert ".in_box((" in snippet
    assert "in_box=" not in snippet
    assert ".to_label(" in snippet
    physical = promo["suggestions"][0]["to_physical"]
    ast.parse(physical)
    assert "g.labels.add" not in snippet
    assert "g.physical.add" not in physical
    assert ".py" in promo["note"]


def test_promote_without_bbox_does_not_author_tags(tmp_path: Path) -> None:
    from apeGmsh.studio import write_envelope
    from apeGmsh.studio._envelope import PickEvidence, SelectionEnvelope

    (tmp_path / ".apegmsh").mkdir()
    write_envelope(
        tmp_path / ".apegmsh" / "selection.json",
        SelectionEnvelope(
            phase="model",
            substrate="model_brep",
            labels=(),
            physical_groups=(),
            unnamed=(
                PickEvidence(substrate="model_brep", dim=2, key=7),
            ),
            evidence=(
                PickEvidence(substrate="model_brep", dim=2, key=7),
            ),
        ),
    )
    promo = promote_selection(root=tmp_path)
    sug = promo["suggestions"][0]
    assert sug["to_label"] is None
    assert sug["to_physical"] is None
    assert "INV-3" in sug["note"]


def test_highlight_does_not_write_envelope(tmp_path: Path) -> None:
    (tmp_path / ".apegmsh").mkdir()
    (tmp_path / ".apegmsh" / "names.json").write_text(
        json.dumps({"schema": 1, "phase": "model", "entities": []}) + "\n",
        encoding="utf-8",
    )
    hit = highlight(["front"], root=tmp_path)
    assert hit["ok"] is False
    assert not (tmp_path / ".apegmsh" / "selection.json").is_file()
    pick = get_selection(root=tmp_path)
    assert pick["present"] is False


def test_highlight_missing_name(tmp_path: Path) -> None:
    (tmp_path / ".apegmsh").mkdir()
    (tmp_path / ".apegmsh" / "names.json").write_text(
        json.dumps({"schema": 1, "phase": "model", "entities": []}) + "\n",
        encoding="utf-8",
    )
    hit = highlight(["nope"], root=tmp_path)
    assert hit["ok"] is False
    assert hit["missing"] == ["nope"]


def test_mcp_adapter_importable_without_sdk() -> None:
    import apeGmsh.studio.mcp as mcp_mod

    assert callable(mcp_mod.build_server)
    assert callable(mcp_mod.main)
