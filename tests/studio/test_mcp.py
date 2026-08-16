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
    lookup,
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
    MCP_DIR / "_mcp_log.py",
    MCP_DIR / "_bundle.py",
    MCP_DIR / "_skins.py",
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
    result = run_until("x.py", phase="cad", root=tmp_path)
    assert result["ok"] is False
    assert result["error"]["code"] == "INVALID_PHASE"


def test_run_until_bad_root(tmp_path: Path) -> None:
    missing = tmp_path / "no_such_project"
    result = run_until("x.py", phase="model", root=missing)
    assert result["ok"] is False
    assert result["error"]["code"] == "BAD_ROOT"
    assert "not a directory" in result["error"]["message"]


def test_promote_selection_torn_envelope(tmp_path: Path) -> None:
    habitat = tmp_path / ".apegmsh"
    habitat.mkdir()
    (habitat / "selection.json").write_text("{torn\n", encoding="utf-8")
    promo = promote_selection(root=tmp_path)
    assert promo["ok"] is False
    assert promo["error"]["code"] == "UNREADABLE"
    assert promo["present"] is False
    assert promo["suggestions"] == []


def test_run_until_timeout(tmp_path: Path, monkeypatch) -> None:
    import subprocess

    script = tmp_path / "box.py"
    script.write_text("pass\n", encoding="utf-8")

    def boom(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0] if args else "x",
            timeout=kwargs.get("timeout") or 1,
            output="",
            stderr=b"",
        )

    monkeypatch.setattr("apeGmsh.studio._mcp.subprocess.run", boom)
    result = run_until(script, phase="model", root=tmp_path)
    assert result["ok"] is False
    assert result["error"]["code"] == "TIMEOUT"


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
    assert "--root" in argv
    assert str(tmp_path.resolve()) in argv
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
    assert "--json" in argv
    assert "--no-viewer" not in argv
    assert result["written"]


def test_render_rejects_open_view(tmp_path: Path) -> None:
    src = tmp_path / "run.h5"
    src.write_bytes(b"x")
    result = render(src, view="poster", root=tmp_path)
    assert result["ok"] is False
    assert result["error"]["code"] == "INVALID_VIEW"


def test_animate_subprocesses_cli(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "run.h5"
    src.write_bytes(b"x")
    recorded: dict = {}

    def fake_run(argv, **kwargs):
        recorded["argv"] = list(argv)
        out = tmp_path / ".apegmsh" / "visors" / "history.gif"
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"ok": True, "written": [str(out)]}) + "\n",
            stderr="",
        )

    monkeypatch.setattr("apeGmsh.studio._mcp.subprocess.run", fake_run)
    result = animate(src, kind="history", root=tmp_path)
    assert result["ok"] is True
    argv = recorded["argv"]
    assert argv[1:3] == ["-m", "apeGmsh.studio"]
    assert "--animate" in argv and "--kind" in argv and "history" in argv
    assert "--output" in argv
    assert "--json" in argv


def test_animate_yield_subprocesses_cli(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "run.h5"
    src.write_bytes(b"x")
    recorded: dict = {}

    def fake_run(argv, **kwargs):
        recorded["argv"] = list(argv)
        out = tmp_path / ".apegmsh" / "visors" / "yield.gif"
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"ok": True, "written": [str(out)]}) + "\n",
            stderr="",
        )

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
    result = animate(src, kind="formation", root=tmp_path)
    assert result["ok"] is False
    assert result["error"]["code"] == "NOT_SHIPPED"


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

    html = emit_report(format="html", root=tmp_path)
    assert html["ok"] is True
    assert html["format"] == "html"
    assert Path(html["path"]).suffix == ".html"
    assert Path(html["archive"]).name == "studio-report.md"


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


def test_quiet_env_keeps_mcp_stdout_clean() -> None:
    """The office venv .pth banner is stdout; MCP JSON-RPC needs it off."""
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    env["LADRUNO_OPENSEES_QUIET"] = "1"
    env["APEGMSH_QUIET"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", "import apeGmsh.studio._mcp"],
        capture_output=True,
        env=env,
        check=True,
    )
    assert proc.stdout == b"", proc.stdout[:200]


_MCP_TOOLS = frozenset({
    "status",
    "get_selection",
    "lookup",
    "run_until",
    "assess",
    "render",
    "animate",
    "results_pin",
    "emit_report",
    "highlight",
    "promote_selection",
})
_MCP_FORBIDDEN = frozenset({
    "add_box",
    "generate",
    "fix_skill",
    "remember_steps",
    "agent_profile",
})


def _tool_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            func = dec.func if isinstance(dec, ast.Call) else dec
            if isinstance(func, ast.Attribute) and func.attr == "tool":
                names.add(node.name)
    return names


def test_mcp_tool_catalog_is_closed() -> None:
    names = _tool_names(MCP_DIR / "mcp.py")
    assert names == _MCP_TOOLS, sorted(names)
    assert not (names & _MCP_FORBIDDEN)


def test_mcp_lookup_add_box() -> None:
    payload = lookup("add_box")
    assert payload["ok"] is True
    assert payload["code"] == 0
    assert payload["kind"] == "hit"
    text = payload["text"]
    assert "g.model.geometry.add_box" in text
    assert "add_box(" in text
    assert text.count("\n") <= 20


def test_mcp_lookup_from_native() -> None:
    payload = lookup("from_native")
    assert payload["ok"] is True
    assert payload["kind"] == "hit"
    assert "Results.from_native" in payload["text"]
    assert payload["text"].count("\n") <= 20


def test_mcp_lookup_miss_does_not_grep_src() -> None:
    payload = lookup("definitely_not_an_apegmsh_symbol")
    assert payload["ok"] is False
    assert payload["code"] == 2
    assert payload["kind"] == "miss"
    assert "miss:" in payload["text"]
    assert "grep src" not in payload["text"].lower()


def test_mcp_lookup_select_is_ambiguous() -> None:
    payload = lookup("select")
    assert payload["ok"] is False
    assert payload["code"] == 2
    assert payload["kind"] == "ambiguous"
    text = payload["text"]
    assert "g.model.select" in text
    assert "select(" in text
    assert text.count("\n") <= 20
