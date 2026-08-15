"""ADR 0096 S2 — generated API index lookup (no src/ grep, no module dump)."""
from __future__ import annotations

import ast
from pathlib import Path

from apeGmsh.studio._index_build import INDEX_PATH, index_drift
from apeGmsh.studio._lookup import (
    _MAX_LINES,
    format_ambiguous,
    format_miss,
    lookup,
    match_symbols,
)
from apeGmsh.studio.lookup import main as lookup_main

STUDIO_DIR = Path(__file__).resolve().parents[2] / "src" / "apeGmsh" / "studio"
_LOOKUP_FILES = (
    STUDIO_DIR / "lookup.py",
    STUDIO_DIR / "_lookup.py",
    STUDIO_DIR / "_index_build.py",
)


def _forbidden(module: str) -> bool:
    if module == "gmsh" or module.startswith("gmsh."):
        return True
    if module == "apeGmsh.viewers" or module.startswith("apeGmsh.viewers."):
        return True
    return False


def test_lookup_modules_have_no_toplevel_gmsh_or_viewers() -> None:
    leaks: list[str] = []
    for path in _LOOKUP_FILES:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in tree.body:
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
    assert not leaks, "lookup must not import gmsh/viewers at import time:\n" + "\n".join(
        leaks
    )


def test_committed_index_exists() -> None:
    assert INDEX_PATH.is_file(), (
        f"API index missing: {INDEX_PATH}. "
        "Run `python -m apeGmsh.studio.lookup --build`."
    )


def test_add_box_hit() -> None:
    text, code = lookup("add_box")
    assert code == 0, text
    assert "g.model.geometry.add_box" in text
    assert "add_box(" in text
    assert "*," in text or "label:" in text
    assert "references/api-cheatsheet.md" in text
    assert "Add an axis-aligned box." in text
    assert "add_box(...)" not in text
    assert text.count("\n") <= _MAX_LINES


def test_get_fem_data_skill_pointer() -> None:
    text, code = lookup("get_fem_data")
    assert code == 0, text
    assert "g.mesh.queries.get_fem_data" in text
    assert "references/fem-broker.md" in text


def test_from_native_hit() -> None:
    text, code = lookup("from_native")
    assert code == 0, text
    assert "Results.from_native" in text
    assert "from_native(" in text
    assert "references/results.md" in text
    assert text.count("\n") <= _MAX_LINES


def test_part_ctor_hit() -> None:
    text, code = lookup("Part")
    assert code == 0, text
    assert text.startswith("Part\n")
    assert "Part(" in text
    assert "references/compose.md" in text
    assert "Initialize self" not in text
    assert text.count("\n") <= _MAX_LINES


def test_cluster_load_hit() -> None:
    text, code = lookup("Cluster.load")
    assert code == 0, text
    assert text.startswith("Cluster.load\n")
    assert "load(" in text
    assert "references/api-cheatsheet.md" in text


def test_in_box_fluent_hit() -> None:
    text, code = lookup("in_box")
    assert code == 0, text
    assert "g.model.select.in_box" in text
    assert "in_box(" in text
    assert text.count("\n") <= _MAX_LINES


def test_to_label_fluent_hit() -> None:
    text, code = lookup("to_label")
    assert code == 0, text
    assert "g.model.select.to_label" in text
    assert "to_label(" in text


def test_assembly_materialize_hit() -> None:
    text, code = lookup("materialize")
    assert code == 0, text
    assert "Assembly.materialize" in text
    assert "references/compose.md" in text


def test_four_node_tet_skill_pointer() -> None:
    text, code = lookup("FourNodeTetrahedron")
    assert code == 0, text
    assert "ops.element.FourNodeTetrahedron" in text
    assert "references/opensees-bridge.md" in text


def test_add_box_full_symbol() -> None:
    text, code = lookup("g.model.geometry.add_box")
    assert code == 0, text
    assert text.startswith("g.model.geometry.add_box\n")


def test_lookup_miss_does_not_grep_src() -> None:
    text, code = lookup("definitely_not_an_apegmsh_symbol")
    assert code == 2
    assert "miss:" in text
    assert "ADR 0096" in text
    assert "grep src" not in text.lower()


def test_ambiguous_is_bounded() -> None:
    hits = [f"g.fake.sym{i}" for i in range(12)]
    text = format_ambiguous("sym", hits)
    assert text.startswith("ambiguous:")
    assert text.count("\n") <= _MAX_LINES
    assert "+4 more" in text


def test_ambiguous_prints_signatures() -> None:
    hits = ["g.model.select", "g.mesh_selection.select"]
    entries = {
        "g.model.select": {"signature": "select(target=None)"},
        "g.mesh_selection.select": {"signature": "select(name: str)"},
    }
    text = format_ambiguous("select", hits, entries)
    assert "g.model.select" in text
    assert "select(target=None)" in text
    assert "g.mesh_selection.select" in text
    assert "select(name: str)" in text
    assert text.count("\n") <= _MAX_LINES


def test_select_ambiguous_live() -> None:
    text, code = lookup("select")
    assert code == 2, text
    assert "g.model.select" in text
    assert "g.mesh_selection.select" in text
    assert "select(" in text
    assert text.count("\n") <= _MAX_LINES


def test_miss_format_points_at_skill() -> None:
    text = format_miss("nope")
    assert "references/api-cheatsheet.md" in text
    assert "src/" in text


def test_match_suffix() -> None:
    entries = {
        "g.model.geometry.add_box": {},
        "ops.element.FourNodeTetrahedron": {},
    }
    assert match_symbols("add_box", entries) == ["g.model.geometry.add_box"]
    assert match_symbols("FourNodeTetrahedron", entries) == [
        "ops.element.FourNodeTetrahedron"
    ]


def test_cli_add_box(capsys) -> None:
    assert lookup_main(["add_box"]) == 0
    out = capsys.readouterr().out
    assert "g.model.geometry.add_box" in out
    assert "add_box(...)" not in out
    assert out.count("\n") <= _MAX_LINES


def test_index_drift_ignores_generated_stamp() -> None:
    committed = {
        "generated": "old",
        "entries": {
            "g.model.geometry.add_box": {
                "signature": "add_box(x)",
                "skill": "references/api-cheatsheet.md",
            }
        },
    }
    live = {
        "generated": "new",
        "entries": {
            "g.model.geometry.add_box": {
                "signature": "add_box(x)",
                "skill": "references/api-cheatsheet.md",
            }
        },
    }
    assert index_drift(committed, live) == []
    live["entries"]["g.model.geometry.add_box"]["signature"] = "add_box(x, y)"
    assert index_drift(committed, live)


def test_committed_index_matches_live_harvest() -> None:
    from apeGmsh.studio._index_build import committed_index_drift

    drift = committed_index_drift()
    assert not drift, "committed _api_index.json drifted from live harvest:\n" + "\n".join(
        drift
    )
