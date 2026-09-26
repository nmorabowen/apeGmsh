"""scripts/check_quirks.py: each rule flags its bug shape and passes its fixes.

One case per shape a rule must flag or pass, and one per hole a review
finds (a hole found is a case added). The scan over the checkout itself is
the last step of CI's `static-gates` job, not a test here.
"""
from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_quirks.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_quirks", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


quirks = _load()

FEMDATA = """\
class NodeComposite:
    def __init__(self, node_ids, node_coords, masses=None):
        pass

class ElementComposite:
    def __init__(self, groups, physical, contacts=None, embed_ties=None):
        pass
"""


def _write(root: Path, rel: str, source: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")


def _found(root: Path) -> list[str]:
    return [f"{f.rule}:{Path(f.path).name}:{f.line}" for f in quirks.scan(root)]


# --- adr-number: a second 0065 (#676 -> #677) --------------------------------

DECISIONS = "src/apeGmsh/opensees/architecture/decisions"


def _adrs(root: Path, names: list[str], indexed: list[str]) -> None:
    for name in names:
        _write(root, f"{DECISIONS}/{name}", "# ADR\n")
    rows = "\n".join(f"| [{n[:4]}]({n}) | t | Accepted |" for n in indexed)
    _write(root, f"{DECISIONS}/README.md", f"# ADRs\n\n{rows}\n")


def test_adr_number_flags_two_adrs_with_one_number(tmp_path: Path) -> None:
    names = ["0065-streaming-deck-emission.md", "0065-h5drm-drm-authoring.md"]
    _adrs(tmp_path, names, indexed=names)
    assert _found(tmp_path) == ["adr-number:decisions:0"]


def test_adr_number_flags_an_adr_the_index_does_not_link(tmp_path: Path) -> None:
    _adrs(tmp_path, ["0064-a.md", "0065-b.md"], indexed=["0064-a.md"])
    assert _found(tmp_path) == ["adr-number:README.md:0"]


def test_adr_number_passes_unique_indexed_adrs(tmp_path: Path) -> None:
    names = ["0064-a.md", "0065-b.md", "0066-c.md"]
    _adrs(tmp_path, names, indexed=names)
    assert _found(tmp_path) == []


def test_adr_number_is_silent_without_a_decisions_folder(tmp_path: Path) -> None:
    assert _found(tmp_path) == []


# --- schema-literal: the #642 and #738 shapes --------------------------------


def test_schema_literal_flags_the_constant_pinned_to_a_literal(tmp_path: Path) -> None:
    _write(tmp_path, "tests/mesh/test_a.py", """\
        from apeGmsh.x import NEUTRAL_SCHEMA_VERSION
        def test_bump():
            assert NEUTRAL_SCHEMA_VERSION == "2.16.0"
        """)
    assert _found(tmp_path) == ["schema-literal:test_a.py:3"]


def test_schema_literal_flags_a_read_back_pinned_to_a_literal(tmp_path: Path) -> None:
    _write(tmp_path, "tests/test_a.py", """\
        def test_inspect(info):
            assert info["neutral_schema_version"] == "2.12.0"
        """)
    assert _found(tmp_path) == ["schema-literal:test_a.py:2"]


def test_schema_literal_flags_either_operand_order_and_not_equal(tmp_path: Path) -> None:
    _write(tmp_path, "tests/test_a.py", """\
        def test_it(meta, m):
            assert "2.12.0" == meta.attrs["schema_version"]
            assert m.schema_version != "2.11.0"
        """)
    assert _found(tmp_path) == ["schema-literal:test_a.py:2", "schema-literal:test_a.py:3"]


def test_schema_literal_passes_the_fixture_constant(tmp_path: Path) -> None:
    _write(tmp_path, "tests/test_a.py", """\
        from tests.fixtures.schema import NEUTRAL_CURRENT
        def test_inspect(info):
            assert info["neutral_schema_version"] == NEUTRAL_CURRENT
        """)
    assert _found(tmp_path) == []


def test_schema_literal_passes_stamping_an_old_version(tmp_path: Path) -> None:
    _write(tmp_path, "tests/test_a.py", """\
        def test_window(f):
            f["meta"].attrs["schema_version"] = "2.6.0"
            f["meta"].attrs["neutral_schema_version"] = "3.0.0"
        """)
    assert _found(tmp_path) == []


def test_schema_literal_ignores_versions_that_are_not_schema_versions(tmp_path: Path) -> None:
    _write(tmp_path, "tests/test_a.py", """\
        def test_it(meta, CONTRACT_VERSION):
            assert CONTRACT_VERSION == "1.8.0"
            assert meta.attrs["apeGmsh_version"] == "0.99.0"
        """)
    assert _found(tmp_path) == []


def test_schema_literal_exempts_the_fixture_and_source(tmp_path: Path) -> None:
    _write(tmp_path, "tests/fixtures/schema.py", 'assert NEUTRAL_SCHEMA_VERSION == "2.33.0"\n')
    _write(tmp_path, "src/apeGmsh/mesh/_compose.py", 'ok = schema_version == "2.33.0"\n')
    assert _found(tmp_path) == []


# --- compose-streams: the #707 and #912/#913 shapes --------------------------


def test_compose_streams_flags_a_rebuild_that_drops_a_stream(tmp_path: Path) -> None:
    _write(tmp_path, "src/apeGmsh/mesh/FEMData.py", FEMDATA)
    _write(tmp_path, "src/apeGmsh/mesh/_compose.py", """\
        new = ElementComposite(groups=g, physical=p, contacts=c)
        """)
    found = quirks.scan(tmp_path)
    assert [f"{f.rule}:{f.line}" for f in found] == ["compose-streams:1"]
    assert "omits embed_ties" in found[0].message


def test_compose_streams_passes_a_rebuild_that_carries_everything(tmp_path: Path) -> None:
    _write(tmp_path, "src/apeGmsh/mesh/FEMData.py", FEMDATA)
    _write(tmp_path, "src/apeGmsh/mesh/_compose.py", """\
        new = ElementComposite(g, p, contacts=c, embed_ties=e)
        nodes = NodeComposite(ids, xyz, masses=m)
        """)
    assert _found(tmp_path) == []


def test_compose_streams_checks_the_h5_round_trip_and_node_composites(tmp_path: Path) -> None:
    _write(tmp_path, "src/apeGmsh/mesh/FEMData.py", FEMDATA)
    _write(tmp_path, "src/apeGmsh/mesh/_femdata_h5_io.py", """\
        from apeGmsh.mesh.FEMData import NodeComposite
        nodes = NodeComposite(node_ids=i, node_coords=x)
        """)
    assert _found(tmp_path) == ["compose-streams:_femdata_h5_io.py:2"]


def test_compose_streams_skips_a_call_it_cannot_read(tmp_path: Path) -> None:
    _write(tmp_path, "src/apeGmsh/mesh/FEMData.py", FEMDATA)
    _write(tmp_path, "src/apeGmsh/mesh/_compose.py", """\
        new = ElementComposite(**fields)
        more = ElementComposite(*args)
        """)
    assert _found(tmp_path) == []


def test_compose_streams_leaves_foreign_format_readers_alone(tmp_path: Path) -> None:
    _write(tmp_path, "src/apeGmsh/mesh/FEMData.py", FEMDATA)
    _write(tmp_path, "src/apeGmsh/mesh/_femdata_mpco_io.py", "e = ElementComposite(g, p)\n")
    assert _found(tmp_path) == []


def test_compose_streams_is_silent_without_the_class(tmp_path: Path) -> None:
    _write(tmp_path, "src/apeGmsh/mesh/_compose.py", "e = ElementComposite(g, p)\n")
    assert _found(tmp_path) == []


# --- waivers -------------------------------------------------------------------


def test_a_waiver_block_above_suppresses_one_site(tmp_path: Path) -> None:
    _write(tmp_path, "tests/test_a.py", """\
        def test_override(meta):
            # apegmsh-lint: schema-literal-ok the constructor override,
            # echoed back; never the current version.
            assert meta["schema_version"] == "1.2.3"
        """)
    assert _found(tmp_path) == []


def test_a_waiver_without_a_reason_waives_nothing(tmp_path: Path) -> None:
    _write(tmp_path, "tests/test_a.py", """\
        def test_it(m):
            # apegmsh-lint: schema-literal-ok
            assert m.schema_version == "2.1.0"
        """)
    assert _found(tmp_path) == ["waiver:test_a.py:2", "schema-literal:test_a.py:3"]


def test_a_waiver_that_waives_nothing_is_a_finding(tmp_path: Path) -> None:
    _write(tmp_path, "tests/test_a.py", """\
        # apegmsh-lint: schema-literal-ok left behind after a fix
        x = 1
        """)
    assert _found(tmp_path) == ["waiver:test_a.py:1"]


def test_adr_number_cannot_be_waived(tmp_path: Path) -> None:
    _write(tmp_path, "tests/test_a.py", "# apegmsh-lint: adr-number-ok because\nx = 1\n")
    assert _found(tmp_path) == ["waiver:test_a.py:1"]


def test_the_command_exits_nonzero_on_a_finding(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(tmp_path, "tests/test_a.py", 'def t(m):\n    assert m.schema_version == "2.1.0"\n')
    assert quirks.main(["--root", str(tmp_path)]) == 1
    assert "[schema-literal]" in capsys.readouterr().out
