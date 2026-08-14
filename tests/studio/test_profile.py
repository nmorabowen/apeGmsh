"""ADR 0096 S4 / S5 — sidecar profiler and observe-only promotion."""
from __future__ import annotations

import ast
import json
from pathlib import Path

from apeGmsh.studio._mcp_log import append_mcp_call, logged, make_call_record
from apeGmsh.studio._paths import mcp_calls_path
from apeGmsh.studio._profile import (
    MISS_PROMOTE_REPEATS,
    build_report,
    classify_tool,
    format_report,
    promote_report,
    skill_budget_violations,
    src_search_rate,
)
from apeGmsh.studio.profile import main as profile_main

STUDIO_DIR = Path(__file__).resolve().parents[2] / "src" / "apeGmsh" / "studio"
_PROFILE_FILES = (
    STUDIO_DIR / "_profile.py",
    STUDIO_DIR / "_mcp_log.py",
    STUDIO_DIR / "profile.py",
)


def _forbidden(module: str) -> bool:
    if module == "gmsh" or module.startswith("gmsh."):
        return True
    if module == "apeGmsh.viewers" or module.startswith("apeGmsh.viewers."):
        return True
    return False


def test_profile_modules_have_no_gmsh_or_viewers() -> None:
    leaks: list[str] = []
    for path in _PROFILE_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
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
    assert not leaks, leaks


def test_classify_tool_kinds() -> None:
    assert classify_tool("lookup", {"symbol": "add_box"}) == "index_lookup"
    assert classify_tool("status", {}) == "habitat_mcp"
    assert classify_tool(
        "CallMcpTool",
        {"server": "user-apegmsh-studio", "toolName": "lookup"},
    ) == "index_lookup"
    assert classify_tool(
        "CallMcpTool",
        {"server": "user-apegmsh-studio", "toolName": "assess"},
    ) == "habitat_mcp"
    assert classify_tool("Grep", {"path": "src/apeGmsh", "pattern": "add_box"}) == (
        "src_search"
    )
    assert classify_tool("Read", {"path": "src/apeGmsh/core/Model.py"}) == "src_read"
    assert classify_tool("Read", {"path": "skills/apegmsh/SKILL.md"}) == "skill_read"
    assert classify_tool("Read", {"path": "docs/guide.md"}) == "other"


def test_src_search_rate() -> None:
    assert src_search_rate({"src_search": 1, "index_lookup": 1}) == 0.5
    assert src_search_rate({}) == 0.0


def test_append_mcp_call_and_report(tmp_path: Path) -> None:
    append_mcp_call("lookup", {"symbol": "add_box"}, {"ok": True, "text": "x"}, root=tmp_path)
    append_mcp_call("status", {}, {"empty": True}, root=tmp_path)
    path = mcp_calls_path(tmp_path)
    report = build_report(mcp_path=path)
    assert report["counts"]["index_lookup"] == 1
    assert report["counts"]["habitat_mcp"] == 1
    assert report["events"] == 2
    rec = make_call_record("lookup", {"symbol": "a"}, {"ok": False})
    assert rec["ok"] is False
    assert rec["symbol"] == "a"
    assert "text" not in rec
    rec_miss = make_call_record(
        "lookup",
        {"symbol": "from_native"},
        {"ok": False, "code": 2, "kind": "miss", "text": "miss: 'from_native'\n"},
    )
    assert rec_miss["kind"] == "miss"
    assert rec_miss["code"] == 2
    assert "text" not in rec_miss


def test_logged_swallows_and_returns(tmp_path: Path) -> None:
    def fn(*, x: int) -> dict:
        return {"ok": True, "x": x}

    out = logged("status", fn, root=tmp_path, x=1)
    assert out == {"ok": True, "x": 1}
    lines = mcp_calls_path(tmp_path).read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(lines[0])["tool"] == "status"


def test_logged_lookup_records_kind_not_text(tmp_path: Path) -> None:
    from apeGmsh.studio._mcp import lookup as mcp_lookup

    logged("lookup", mcp_lookup, symbol="select", root=tmp_path)
    logged(
        "lookup", mcp_lookup,
        symbol="definitely_not_an_apegmsh_symbol", root=tmp_path,
    )
    lines = mcp_calls_path(tmp_path).read_text(encoding="utf-8").strip().splitlines()
    amb = json.loads(lines[0])
    miss = json.loads(lines[1])
    assert amb["kind"] == "ambiguous"
    assert amb["ok"] is False
    assert "text" not in amb
    assert miss["kind"] == "miss"
    assert miss["symbol"] == "definitely_not_an_apegmsh_symbol"
    report = promote_report(mcp_calls_path(tmp_path))
    assert report["eligible_misses"] == []
    assert report["observed_misses"] == [
        {"symbol": "definitely_not_an_apegmsh_symbol", "n": 1},
    ]


def test_transcript_classification(tmp_path: Path) -> None:
    events = [
        {
            "role": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Grep",
                        "input": {"path": "src/apeGmsh", "pattern": "add_box"},
                    }
                ]
            },
        },
        {
            "role": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "CallMcpTool",
                        "input": {
                            "server": "user-apegmsh-studio",
                            "toolName": "lookup",
                            "arguments": {"symbol": "add_box"},
                        },
                    }
                ]
            },
        },
    ]
    path = tmp_path / "t.jsonl"
    path.write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8",
    )
    report = build_report(transcript_path=path)
    assert report["counts"]["src_search"] == 1
    assert report["counts"]["index_lookup"] == 1
    assert report["src_search_rate"] == 0.5
    text = format_report(report)
    assert "src_search_rate:" in text


def test_profile_cli_json(tmp_path: Path, capsys) -> None:
    append_mcp_call("lookup", {"symbol": "add_box"}, {"ok": True}, root=tmp_path)
    assert profile_main(["--mcp", str(mcp_calls_path(tmp_path)), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["counts"]["index_lookup"] == 1


def test_skill_budget_current_tree() -> None:
    root = Path(__file__).resolve().parents[2]
    assert skill_budget_violations(root) == []


def test_profile_cli_skill_budget() -> None:
    root = Path(__file__).resolve().parents[2]
    assert profile_main(["--skill-budget", "--root", str(root)]) == 0


def test_promote_bar_requires_three_repeats(tmp_path: Path) -> None:
    assert MISS_PROMOTE_REPEATS == 3
    miss = {"ok": False, "code": 2, "kind": "miss", "text": "miss: 'from_native'\n"}
    for _ in range(3):
        append_mcp_call("lookup", {"symbol": "from_native"}, miss, root=tmp_path)
    append_mcp_call(
        "lookup",
        {"symbol": "Cluster"},
        {"ok": False, "code": 2, "kind": "miss", "text": "miss: 'Cluster'\n"},
        root=tmp_path,
    )
    append_mcp_call(
        "lookup", {"symbol": "add_box"}, {"ok": True, "code": 0, "kind": "hit"},
        root=tmp_path,
    )
    report = promote_report(mcp_calls_path(tmp_path))
    assert report["writes"] is False
    assert report["eligible_misses"] == [{"symbol": "from_native", "n": 3}]
    assert report["observed_misses"] == [{"symbol": "Cluster", "n": 1}]


def test_promote_bar_ignores_ambiguous(tmp_path: Path) -> None:
    ambiguous = {
        "ok": False,
        "code": 2,
        "kind": "ambiguous",
        "text": "ambiguous: 'select' matches 2 symbols:\n",
    }
    for _ in range(3):
        append_mcp_call("lookup", {"symbol": "select"}, ambiguous, root=tmp_path)
    report = promote_report(mcp_calls_path(tmp_path))
    assert report["eligible_misses"] == []
    assert report["observed_misses"] == []


def test_promote_cli_writes_nothing(tmp_path: Path, capsys) -> None:
    root = Path(__file__).resolve().parents[2]
    skill = root / "skills" / "apegmsh" / "SKILL.md"
    derived = root / ".claude" / "skills" / "apegmsh-helper" / "SKILL.md"
    before = skill.read_text(encoding="utf-8")
    derived_before = derived.read_text(encoding="utf-8") if derived.is_file() else None
    for _ in range(3):
        append_mcp_call(
            "lookup",
            {"symbol": "from_native"},
            {"ok": False, "code": 2, "kind": "miss", "text": "miss: 'from_native'\n"},
            root=tmp_path,
        )
    assert profile_main(
        ["--promote", "--mcp", str(mcp_calls_path(tmp_path)), "--json"]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["eligible_misses"][0]["symbol"] == "from_native"
    assert payload["writes"] is False
    assert skill.read_text(encoding="utf-8") == before
    if derived_before is not None:
        assert derived.read_text(encoding="utf-8") == derived_before
    assert "fix_skill" in payload["forbidden"]
    assert "write skills/apegmsh/ in-session" in payload["forbidden"]


def test_workflows_recommendation_is_non_normative() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "skills" / "apegmsh" / "references" / "workflows.md").read_text(
        encoding="utf-8",
    )
    assert "Recommendation — studio / ImageMage" in text
    assert "not the required path" in text
    assert "INV-14" in text
