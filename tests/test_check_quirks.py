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


# --- resolve-swallow: 3aecb417, and 06ccd266 -> 45340ac3 ---------------------

RESOLVER = "src/apeGmsh/_kernel/resolvers/_router.py"
FACTORY = "src/apeGmsh/mesh/_fem_factory.py"


def _handler(root: Path, caught: str, body: str, rel: str = RESOLVER) -> None:
    except_line = f"except {caught}:" if caught else "except:"
    _write(root, rel, f"def f(log, np):\n    try:\n        nodes = g()\n    {except_line}\n"
                      f"        {body}\n    return nodes\n")


def test_resolve_swallow_flags_the_chain_phase_router_shape(tmp_path: Path) -> None:
    _write(tmp_path, RESOLVER, """\
        def try_chain_phase_route(session, defn):
            try:
                new_fem = route_def_to_fem(session._fem, defn)
            except (KeyError, TypeError):
                return False
            return True
        """)
    assert _found(tmp_path) == ["resolve-swallow:_router.py:4"]


def test_resolve_swallow_flags_the_logged_fem_factory_shape(tmp_path: Path) -> None:
    _write(tmp_path, FACTORY, """\
        def build(session, log):
            try:
                session.constraints.resolve()
            except Exception as exc:
                log.warning("constraint resolve failed: %s", exc)
        """)
    assert _found(tmp_path) == ["resolve-swallow:_fem_factory.py:4"]


def test_resolve_swallow_flags_a_predicate_named_copy_of_the_incident(tmp_path: Path) -> None:
    # A name is not a contract: `is_routable` with the router's body is the incident.
    _write(tmp_path, RESOLVER, """\
        def is_routable(session, defn):
            try:
                route_def_to_fem(session._fem, defn)
                return True
            except (KeyError, TypeError):
                return False
        """)
    assert _found(tmp_path) == ["resolve-swallow:_router.py:5"]


def test_resolve_swallow_reaches_subpackages(tmp_path: Path) -> None:
    _handler(tmp_path, "KeyError", "pass", rel="src/apeGmsh/_kernel/resolvers/_sub/_y.py")
    assert _found(tmp_path) == ["resolve-swallow:_y.py:4"]


@pytest.mark.parametrize("caught", [
    "", "Exception", "KeyError", "(KeyError, TypeError)", "MortarTieError", "np.linalg.LinAlgError",
])
def test_resolve_swallow_flags_whatever_is_caught(tmp_path: Path, caught: str) -> None:
    # The resolvers' own errors subclass broad ones (MortarTieError(ValueError)).
    _handler(tmp_path, caught, "return None")
    assert _found(tmp_path) == ["resolve-swallow:_router.py:4"]


@pytest.mark.parametrize("body", [
    "pass", "...", "return", "return False", "return []", "return {}", "return set()",
    "return np.array([], dtype=int)", "return np.empty(0)", "log.warning('x')",
    "log.critical('x')", "warnings.warn('x')", "print('x')", "nodes = []",
    "nodes: list = []", "log.warning('x')\n        return set()",
    "if log:\n            log.warning('x')\n        else:\n            pass",
])
def test_resolve_swallow_flags_every_silent_body(tmp_path: Path, body: str) -> None:
    _handler(tmp_path, "KeyError", body)
    assert _found(tmp_path) == ["resolve-swallow:_router.py:4"]


def test_resolve_swallow_flags_contextlib_suppress(tmp_path: Path) -> None:
    _write(tmp_path, RESOLVER, "def f():\n    with contextlib.suppress(KeyError):\n        return g()\n")
    assert _found(tmp_path) == ["resolve-swallow:_router.py:2"]


@pytest.mark.parametrize("body", [
    "raise",                                                        # re-raise
    "raise ValueError('no such label') from None",                  # translate
    "if strict:\n            raise\n        return False",          # the 45340ac3 fix
    "if lenient:\n            log.warning('x')\n        else:\n            raise",
    "return self._default_nodes", "return True", "return 'fallback'", "return [0]",
    "self._misses += 1\n        return False",                      # unreadable: stay silent
])
def test_resolve_swallow_passes_a_handler_that_is_not_provably_silent(
    tmp_path: Path, body: str
) -> None:
    _handler(tmp_path, "KeyError", body)
    assert _found(tmp_path) == []


@pytest.mark.parametrize("rel", ["src/apeGmsh/core/_x.py", "src/apeGmsh/mesh/_compose.py"])
def test_resolve_swallow_ignores_code_outside_the_resolution_scope(tmp_path: Path, rel: str) -> None:
    # _compose.py is scanned (compose-streams) but is not resolution code.
    _handler(tmp_path, "KeyError", "return False", rel=rel)
    assert _found(tmp_path) == []


def test_resolve_swallow_waiver_on_the_except_line(tmp_path: Path) -> None:
    _write(tmp_path, RESOLVER, """\
        def has_target(self, target):
            try:
                self.nodes_for(target)
                return True
            except KeyError:  # apegmsh-lint: resolve-swallow-ok the predicate's contract
                return False
        """)
    assert _found(tmp_path) == []


def test_resolve_swallow_stale_waiver_is_a_finding(tmp_path: Path) -> None:
    _write(tmp_path, RESOLVER, """\
        def f():
            try:
                g()
            except KeyError:  # apegmsh-lint: resolve-swallow-ok was silent once
                raise
        """)
    assert _found(tmp_path) == ["waiver:_router.py:4"]


def test_resolve_swallow_scope_exists_in_this_checkout() -> None:
    # A moved or renamed path would turn the rule off silently; pin it here.
    for path in quirks.SWALLOW_SCOPE:
        target = quirks.REPO / path
        assert target.is_file() or any(target.rglob("*.py")), f"{path} moved: update SWALLOW_SCOPE"


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


# --- reading files: encodings must not blind or crash the scan ----------------

_SWALLOW = "def f():\n    try:\n        g()\n    except KeyError:\n        pass\n"


def _raw(root: Path, rel: str, data: bytes) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_a_file_with_a_bom_is_still_scanned(tmp_path: Path) -> None:
    # read_text("utf-8") kept the BOM, ast.parse refused it, and the file was skipped.
    _raw(tmp_path, RESOLVER, b"\xef\xbb\xbf" + _SWALLOW.encode())
    assert _found(tmp_path) == ["resolve-swallow:_router.py:4"]


def test_a_file_with_a_coding_cookie_is_still_scanned(tmp_path: Path) -> None:
    src = "# -*- coding: latin-1 -*-\n# caf\xe9\n" + _SWALLOW
    _raw(tmp_path, RESOLVER, src.encode("latin-1"))
    assert _found(tmp_path) == ["resolve-swallow:_router.py:6"]


def test_an_undecodable_file_is_skipped_not_fatal(tmp_path: Path) -> None:
    # Not valid Python (no cookie, not UTF-8): skip it like a SyntaxError, and
    # keep scanning the rest instead of aborting the whole run.
    _raw(tmp_path, RESOLVER, b"# caf\xe9\nx = 1\n")
    _write(tmp_path, FACTORY, _SWALLOW)
    assert _found(tmp_path) == ["resolve-swallow:_fem_factory.py:4"]
