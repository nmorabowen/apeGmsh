"""ADR 0096 S2 — generated API index lookup (no src/ grep, no module dump)."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from apeGmsh.studio._index_build import INDEX_PATH, index_drift, skill_for
from apeGmsh.studio._lookup import (
    _MAX_LINES,
    format_ambiguous,
    format_miss,
    load_index,
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


def test_assembly_bridge_hit() -> None:
    text, code = lookup("Assembly.bridge")
    assert code == 0, text
    assert "Assembly.bridge" in text
    assert "references/compose.md" in text


def test_four_node_tet_skill_pointer() -> None:
    text, code = lookup("FourNodeTetrahedron")
    assert code == 0, text
    assert "ops.element.FourNodeTetrahedron" in text
    assert "references/opensees-bridge.md" in text


def test_eigen_result_to_native_hit() -> None:
    """SK1-S4 (#2008): a returned object's methods resolve by class name."""
    text, code = lookup("EigenResult.to_native")
    assert code == 0, text
    assert text.startswith("EigenResult.to_native\n")
    assert "to_native(" in text
    assert "references/opensees-bridge.md" in text
    assert "api-cheatsheet" not in text
    assert text.count("\n") <= _MAX_LINES


def test_eigen_result_results_hit() -> None:
    text, code = lookup("EigenResult.results")
    assert code == 0, text
    assert text.startswith("EigenResult.results\n")
    assert "results(" in text
    assert "references/opensees-bridge.md" in text


def test_bare_to_native_is_ambiguous_and_lists_eigen_result() -> None:
    text, code = lookup("to_native")
    assert code == 2, text
    assert text.startswith("ambiguous:")
    assert "EigenResult.to_native" in text
    assert text.count("\n") <= _MAX_LINES


def test_returned_types_are_returned_by_indexed_callables() -> None:
    """The allow-list follows real return types, read off the index."""
    entries = load_index()["entries"]
    returns = {
        "ops.eigen": "EigenResult",
        "ops.eigen_feast": "EigenResult",
        "ops.complex_eigen": "ComplexEigenResult",
        "g.mesh.queries.get_fem_data": "FEMData",
    }
    for symbol, cls_name in returns.items():
        sig = entries[symbol]["signature"]
        assert sig.split(") -> ", 1)[1].strip("'\"") == cls_name, sig


def test_returned_types_indexed_with_own_skill() -> None:
    entries = load_index()["entries"]
    for name in ("EigenResult", "ComplexEigenResult", "ParallelModalResult", "FEMData"):
        own = [k for k in entries if k.startswith(f"{name}.")]
        assert own, f"no index entry for returned type {name}"
        for key in own:
            assert entries[key]["skill"] != "references/api-cheatsheet.md", key
            assert entries[key]["skill"] == skill_for(key), key


@pytest.mark.parametrize(
    "symbol",
    [
        "FEMData.nodes.select",
        "FEMData.elements.select",
        "FEMData.inspect.summary",
        "FEMData.nodes.constraints.equal_dofs",
    ],
)
def test_femdata_sub_composites_hit_fem_broker(symbol: str) -> None:
    """SK2-S4b (#2062): FEMData.nodes/.elements/.inspect are indexed, nested walk on."""
    text, code = lookup(symbol)
    assert code == 0, text
    assert text.startswith(f"{symbol}\n")
    assert "references/fem-broker.md" in text
    assert "api-cheatsheet" not in text
    assert text.count("\n") <= _MAX_LINES


@pytest.mark.parametrize(
    "query, symbol",
    [
        ("g.constraints.summary", "g.constraints.summary"),
        ("g.loads.summary", "g.loads.summary"),
        ("constraints.summary", "g.constraints.summary"),
        ("loads.summary", "g.loads.summary"),
    ],
)
def test_exact_symbol_wins_over_suffix_hits(query: str, symbol: str) -> None:
    """#2063 review: FEMData.nodes.*.summary must not make the g. symbol ambiguous."""
    text, code = lookup(query)
    assert code == 0, text
    assert text.startswith(f"{symbol}\n")


@pytest.mark.parametrize("query", ["from_h5", "from_h5()", "g.from_h5"])
def test_bare_name_stays_ambiguous_despite_exact_g_symbol(query: str) -> None:
    """Exact-match precedence is for dotted queries only (#2063 review)."""
    text, code = lookup(query)
    assert code == 2, text
    assert text.startswith("ambiguous:")
    assert "FEMData.from_h5" in text


def test_session_walk_fails_closed_on_unimportable_composite(monkeypatch) -> None:
    """A composite that does not import raises; it is never silently dropped."""
    from apeGmsh._core import apeGmsh as Session
    from apeGmsh.studio._index_build import _walk_session

    monkeypatch.setattr(
        Session, "_COMPOSITES", (("bogus", ".no_such_module", "Nope", False),)
    )
    with pytest.raises(ModuleNotFoundError, match="no_such_module"):
        _walk_session({})


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


def test_index_drift_detects_signature_change() -> None:
    committed = {
        "entries": {
            "g.model.geometry.add_box": {
                "signature": "add_box(x)",
                "skill": "references/api-cheatsheet.md",
            }
        },
    }
    live = {
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


def test_index_build_is_byte_deterministic() -> None:
    from apeGmsh.studio._index_build import build_index, serialize_index

    first = serialize_index(build_index())
    second = serialize_index(build_index())
    assert first == second
    assert '"generated"' not in first
    assert first.endswith("\n")
    assert first == INDEX_PATH.read_bytes().decode("utf-8"), (
        "committed _api_index.json is not what the builder emits; "
        "run `python -m apeGmsh.studio.lookup --build`"
    )
