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
        _write(root, f"{DECISIONS}/{name}", f"# ADR {name[:4]} — t\n\n**Status:** Accepted (2026-01-01)\n")
    rows = "\n".join(f"| [{n[:4]}]({n}) | t | Accepted |" for n in indexed)
    _write(root, f"{DECISIONS}/README.md", f"# ADRs\n\n| # | Title | Status |\n|---|---|---|\n{rows}\n")
    if indexed == names and len(set(n[:4] for n in names)) == len(names):
        (root / DECISIONS / "README.md").write_text(_adr_index().generate(root), encoding="utf-8")


def _adr_index():
    return quirks._adr_index()


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


def test_adr_index_flags_a_readme_stale_against_a_status_line(tmp_path: Path) -> None:
    names = ["0064-a.md", "0065-b.md"]
    _adrs(tmp_path, names, indexed=names)
    assert _found(tmp_path) == []
    adr = tmp_path / DECISIONS / "0065-b.md"
    adr.write_text(adr.read_text(encoding="utf-8").replace("Accepted", "Superseded by 0066"), encoding="utf-8")
    assert _found(tmp_path) == ["adr-number:README.md:0"]
    (tmp_path / DECISIONS / "README.md").write_text(_adr_index().generate(tmp_path), encoding="utf-8")
    assert _found(tmp_path) == []


def test_adr_index_flags_an_adr_whose_status_line_does_not_parse(tmp_path: Path) -> None:
    names = ["0064-a.md"]
    _adrs(tmp_path, names, indexed=names)
    (tmp_path / DECISIONS / "0064-a.md").write_text("# ADR 0064 — t\n", encoding="utf-8")
    assert _found(tmp_path) == ["adr-number:README.md:0"]


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


# --- openseespy-import: 9ffe6aa2, and the sites that kept the import ---------


def test_openseespy_import_flags_the_domain_capture_fallback(tmp_path: Path) -> None:
    # 9ffe6aa2 put the live emitter first and kept the import as the fallback.
    _write(tmp_path, "src/apeGmsh/results/capture/_domain.py", """\
        _live_emitter = None

        def _lazy_ops(self):
            live_ops = getattr(self._bridge, "_live_emitter", None)
            if live_ops is not None:
                return live_ops
            import openseespy.opensees as ops
            return ops
        """)
    assert _found(tmp_path) == ["openseespy-import:_domain.py:7"]


def test_openseespy_import_flags_the_example_that_runs_then_imports(tmp_path: Path) -> None:
    # arch-pushover: the bridge builds the model, then the loop drives another module.
    _write(tmp_path, "examples/shoebuckle_arch.py", """\
        def run_to_limit(ops, fem):
            ops.run(wipe=True)
            import openseespy.opensees as osi
            return osi.analyze(1)
        """)
    assert _found(tmp_path) == ["openseespy-import:shoebuckle_arch.py:3"]


@pytest.mark.parametrize("statement", [
    "import openseespy.opensees as ops_module",      # LiveMPCO, LiveRecorders
    "import openseespy.opensees as o",               # interop.solve
    "import openseespy",
    "import os, openseespy.opensees",
    "from openseespy import opensees",
    "from openseespy.opensees import getPID",
    "from openseespy.opensees \\\n        import getPID",
    "ops = importlib.import_module('openseespy.opensees')",
    "ops = __import__('openseespy.opensees')",
])
def test_openseespy_import_flags_every_spelling(tmp_path: Path, statement: str) -> None:
    _write(tmp_path, "src/apeGmsh/results/live/_mpco.py", f"def enter(self):\n    {statement}\n")
    assert _found(tmp_path) == ["openseespy-import:_mpco.py:2"]


def test_openseespy_import_exempts_the_resolver(tmp_path: Path) -> None:
    _write(tmp_path, "src/apeGmsh/opensees/emitter/live.py", """\
        def _stock():
            import openseespy.opensees as _ops
            return _ops
        """)
    assert _found(tmp_path) == []


def test_openseespy_import_passes_what_binds_something_else(tmp_path: Path) -> None:
    # The py emitter writes the deck's import as a string, a probe binds
    # nothing, and neither a relative module nor a wheel sharing the prefix is
    # openseespy. The deck string makes the file parse, so the AST decides.
    _write(tmp_path, "src/apeGmsh/opensees/emitter/py.py", '''\
        from .openseespy import shim
        import openseespylinux

        HEADER = ["import openseespy.opensees as ops", "ops.wipe()"]

        def available():
            """``import openseespy.opensees`` is only quoted here.

            >>> import openseespy.opensees as ops
            """
            return find_spec("openseespy") is not None
        ''')
    assert _found(tmp_path) == []


@pytest.mark.parametrize("rel", ["tests/test_a.py", "scripts/render.py", "src/sections/_rect.py"])
def test_openseespy_import_ignores_code_outside_the_scope(tmp_path: Path, rel: str) -> None:
    # Tests check py decks, which bind openseespy by design; `sections` is a
    # standalone package built into the caller's openseespy.
    _write(tmp_path, rel, "import openseespy.opensees as ops\n")
    assert _found(tmp_path) == []


def test_openseespy_import_waiver(tmp_path: Path) -> None:
    _write(tmp_path, "src/apeGmsh/studio/examples/raw/raw.py", """\
        # apegmsh-lint: openseespy-import-ok a hand-written openseespy walkthrough,
        # with no bridge domain to share.
        import openseespy.opensees as ops
        """)
    assert _found(tmp_path) == []


def test_openseespy_import_scope_exists_in_this_checkout() -> None:
    # A moved scope turns the rule off silently; a moved resolver flags itself.
    for path in (*quirks.IMPORT_SCOPE, quirks.RESOLVER):
        assert (quirks.REPO / path).exists(), f"{path} moved: update the openseespy-import scope"


# --- getattr-private / getattr-undefined: the capture outages (14445604, 7c7c1541) ---

BRIDGE = """\
class Bridge:
    def __init__(self):
        self._primitives = []
        self.tag = 1
"""


def test_getattr_private_flags_a_private_name_read_across_packages(tmp_path: Path) -> None:
    # results/capture/spec.py read bridge._primitives from another package.
    _write(tmp_path, "src/apeGmsh/opensees/bridge.py", BRIDGE)
    _write(tmp_path, "src/apeGmsh/results/spec.py", """\
        def prims(bridge):
            return getattr(bridge, "_primitives", ())
        """)
    assert _found(tmp_path) == ["getattr-private:spec.py:2"]


def test_getattr_private_flags_hasattr_too(tmp_path: Path) -> None:
    _write(tmp_path, "src/apeGmsh/opensees/bridge.py", BRIDGE)
    _write(tmp_path, "src/apeGmsh/results/spec.py", """\
        def has(bridge):
            return hasattr(bridge, "_primitives")
        """)
    assert _found(tmp_path) == ["getattr-private:spec.py:2"]


def test_getattr_private_passes_self_and_cls(tmp_path: Path) -> None:
    _write(tmp_path, "src/apeGmsh/opensees/bridge.py", BRIDGE)
    _write(tmp_path, "src/apeGmsh/results/spec.py", """\
        class Spec:
            _primitives = ()
            def a(self):
                return getattr(self, "_primitives", ())
            @classmethod
            def b(cls):
                return getattr(cls, "_primitives", ())
        """)
    assert _found(tmp_path) == []


def test_getattr_private_passes_a_name_defined_in_the_same_package(tmp_path: Path) -> None:
    _write(tmp_path, "src/apeGmsh/opensees/bridge.py", BRIDGE)
    _write(tmp_path, "src/apeGmsh/opensees/other.py", """\
        def prims(bridge):
            return getattr(bridge, "_primitives", ())
        """)
    assert _found(tmp_path) == []


def test_getattr_public_name_defined_elsewhere_passes(tmp_path: Path) -> None:
    _write(tmp_path, "src/apeGmsh/opensees/bridge.py", BRIDGE)
    _write(tmp_path, "src/apeGmsh/results/spec.py", """\
        def tag(bridge):
            return getattr(bridge, "tag", None)
        """)
    assert _found(tmp_path) == []


def test_getattr_undefined_flags_a_name_nothing_defines(tmp_path: Path) -> None:
    # 14445604 deleted _sec_tags; the getattr default kept the capture "working".
    _write(tmp_path, "src/apeGmsh/opensees/bridge.py", BRIDGE)
    _write(tmp_path, "src/apeGmsh/opensees/rec.py", """\
        def tags(self):
            return getattr(self._opensees, "_sec_tags", {})
        """)
    assert _found(tmp_path) == ["getattr-undefined:rec.py:2"]


def test_getattr_undefined_passes_every_way_to_define_a_name(tmp_path: Path) -> None:
    _write(tmp_path, "src/apeGmsh/opensees/defs.py", """\
        from dataclasses import dataclass
        from os import path as stat_path

        def fn_name(): ...
        class ClsName: ...
        module_level = 1
        setattr(object(), "via_setattr", 1)

        @dataclass
        class D:
            field_name: int = 0

        class S:
            __slots__ = ("slot_name",)
            def __init__(self):
                self.attr_store = 1
        """)
    names = ["fn_name", "ClsName", "module_level", "via_setattr", "field_name", "slot_name",
             "attr_store", "stat_path"]
    body = "".join(f"    getattr(x, {n!r}, None)\n" for n in names)
    _write(tmp_path, "src/apeGmsh/results/use.py", "def f(x):\n" + body)
    assert _found(tmp_path) == []


def test_getattr_ignores_dunders_dynamic_names_and_other_trees(tmp_path: Path) -> None:
    _write(tmp_path, "src/apeGmsh/results/use.py", """\
        def f(x, name):
            getattr(x, "__file__", None)
            getattr(x, name, None)
            getattr(x, "a" + "b")
        """)
    _write(tmp_path, "tests/test_use.py", """\
        def test_it(x):
            getattr(x, "nothing", 0)
        """)
    assert _found(tmp_path) == []


def test_getattr_waiver_suppresses_one_site(tmp_path: Path) -> None:
    _write(tmp_path, "src/apeGmsh/results/use.py", """\
        def f(x):
            # apegmsh-lint: getattr-undefined-ok a Qt widget, not an apeGmsh object
            return hasattr(x, "setText")
        """)
    assert _found(tmp_path) == []


BASELINE = "scripts/quirks_getattr_baseline.txt"


def test_getattr_baseline_holds_a_listed_site_and_rejects_a_new_one(tmp_path: Path) -> None:
    _write(tmp_path, "src/apeGmsh/results/use.py", """\
        def f(x):
            return hasattr(x, "setText"), hasattr(x, "setValue")
        """)
    _write(tmp_path, BASELINE, "# comment\nsrc/apeGmsh/results/use.py::setText  # a Qt widget\n")
    assert _found(tmp_path) == ["getattr-undefined:use.py:2"]  # only setValue


def test_getattr_baseline_line_matching_nothing_is_a_finding(tmp_path: Path) -> None:
    _write(tmp_path, "src/apeGmsh/results/use.py", "def f(x):\n    return 1\n")
    _write(tmp_path, BASELINE, "# comment\nsrc/apeGmsh/results/use.py::setText\n")
    assert _found(tmp_path) == ["getattr-baseline:quirks_getattr_baseline.txt:2"]


def test_the_checkout_getattr_baseline_is_a_ratchet_within_bounds() -> None:
    assert 0 < len(quirks._baseline_keys(quirks.REPO)) <= 250


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


# --- doc-path: the panel's 11% dead citations (#1192 P6, #1197 N1) ------------

ARCH = "src/apeGmsh/opensees/architecture"
GUIDE = ".claude/skills/apegmsh-bridge-feature/SKILL.md"
MODULE = "def emit_mp_constraints(b):\n    pass\n\nclass _StageBuilder:\n    def stage_open(self):\n        pass\n"


def _doc(root: Path, rel: str, *lines: str) -> None:
    _write(root, rel, "\n".join(lines) + "\n")


def _doc_paths(root: Path) -> list[str]:
    return [f"{Path(f.path).name}:{f.line}" for f in quirks.scan(root) if f.rule == "doc-path"]


def test_doc_path_passes_citations_that_resolve(tmp_path: Path) -> None:
    _write(tmp_path, "src/apeGmsh/opensees/_internal/build.py", MODULE)
    _write(tmp_path, "src/apeGmsh/mesh/FEMData.py", FEMDATA)
    _doc(tmp_path, f"{ARCH}/testing.md", "# t")
    _doc(tmp_path, f"{ARCH}/decisions/README.md", "# ADRs")
    _doc(tmp_path, "AGENTS.md",
         "See [testing.md](src/apeGmsh/opensees/architecture/testing.md) and `mesh/FEMData.py`.",
         "Lift `_internal/build.py::emit_mp_constraints`; `_internal/build.py::_StageBuilder.stage_open`",
         "and `src/apeGmsh/opensees/_internal/build.py:12` (a line is not checked).",
         "Not paths: `~/venv/x.py`, `C:\\venv\\x.py`, `ranks/rank<K>.yml`, `tests/**/*.py`,",
         "`https://x.org/a.md`, `{name}/fields.json`, and a bare `build.py`.")
    _doc(tmp_path, GUIDE, "`decisions/README.md`, `opensees/_internal/build.py`, [t](../../../AGENTS.md)")
    assert _doc_paths(tmp_path) == []


def test_doc_path_flags_a_path_that_does_not_resolve(tmp_path: Path) -> None:
    _doc(tmp_path, "AGENTS.md", "Read `scripts/nav.py` first.", "Then `viewers/ui/viewer_window.py`.")
    assert _doc_paths(tmp_path) == ["AGENTS.md:1", "AGENTS.md:2"]


def test_doc_path_flags_a_markdown_link_only_relative_to_the_doc(tmp_path: Path) -> None:
    # A renderer resolves a link from the doc's folder, never from the package roots.
    _doc(tmp_path, f"{ARCH}/testing.md", "# t")
    _doc(tmp_path, f"{ARCH}/h5-schema.md", "([README](../../../README.md)) and [t](testing.md)")
    _doc(tmp_path, "README.md", "# r")
    assert _doc_paths(tmp_path) == ["h5-schema.md:1"]


def test_doc_path_flags_a_symbol_the_file_does_not_define(tmp_path: Path) -> None:
    _write(tmp_path, "src/apeGmsh/opensees/emitter/h5.py", MODULE)
    _doc(tmp_path, f"{ARCH}/_DEFERRED.md", "lift touches `emitter/h5.py::_write_mp_constraints`,",
         "and `emitter/h5.py::emit_mp_constraints` (still there).")
    found = quirks.scan(tmp_path)
    assert [f"{f.rule}:{f.line}" for f in found] == ["doc-path:1"]
    assert "defines no `_write_mp_constraints`" in found[0].message


def test_doc_path_flags_a_symbol_into_a_file_that_does_not_parse(tmp_path: Path) -> None:
    _write(tmp_path, "src/apeGmsh/x.py", "def (:\n")
    _doc(tmp_path, "AGENTS.md", "`src/apeGmsh/x.py::f`")
    assert _doc_paths(tmp_path) == ["AGENTS.md:1"]


def test_doc_path_ignores_the_adrs_and_the_derived_skill_mirror(tmp_path: Path) -> None:
    _doc(tmp_path, f"{ARCH}/decisions/0001-x.md", "`src/apeGmsh/gone.py` was the plan.")
    _doc(tmp_path, ".claude/skills/apegmsh-helper/SKILL.md", "`src/apeGmsh/gone.py`")
    _doc(tmp_path, ".claude/skills/apegmsh-helper/references/x.md", "`src/apeGmsh/gone.py`")
    _doc(tmp_path, "internal_docs/plan_x.md", "`src/apeGmsh/gone.py`")
    assert _found(tmp_path) == []


def test_doc_path_cannot_be_waived(tmp_path: Path) -> None:
    _write(tmp_path, "tests/test_a.py", "# apegmsh-lint: doc-path-ok because\nx = 1\n")
    assert _found(tmp_path) == ["waiver:test_a.py:1"]


def test_doc_path_scope_exists_in_this_checkout() -> None:
    # A moved doc folder turns the rule off silently (N3 moves architecture/ out of src/).
    assert (quirks.REPO / quirks.AGENTS).is_file()
    assert any((quirks.REPO / quirks.SKILLS).glob("apegmsh-*/SKILL.md")), "the task guides moved"
    assert any((quirks.REPO / quirks.ARCHITECTURE).glob("*.md")), "architecture/ moved: update ARCHITECTURE"


def test_doc_path_skips_the_historical_plan_docs(tmp_path: Path) -> None:
    # May-2026 scope/plan docs cite the layout of their day; N3 deletes them.
    for name in ("phase-8-untangle.md", "phase-8.3b-scope.md", "mp-tag-tracking-scope.md", "plan_x.md"):
        _doc(tmp_path, f"{ARCH}/{name}", "`mesh/records/_kinds.py` moves.")
    _doc(tmp_path, f"{ARCH}/_DEFERRED.md", "`mesh/records/_kinds.py` moves.")
    assert _doc_paths(tmp_path) == ["_DEFERRED.md:1"]


# --- doc-path: the review of #1236 (suffix forms escaped; `::symbol` was loose) ---

SCOPED = '''\
import a.b
from x import y as Alias
try:
    import z
except ImportError:
    z = None
p, q = 1, 2

def test_geom_2d():
    some_local = 1
    return some_local

class Cls:
    CONST = 1
    def __init__(self):
        self.attr = 0
    def method(self):
        pass
'''


def test_doc_path_checks_the_file_under_every_suffix_form(tmp_path: Path) -> None:
    # `:10-20`, `::foo()` and `#L3` used to fail the regex, so the file went unchecked too.
    _doc(tmp_path, "AGENTS.md", "`scripts/gone.py:10-20`", "`scripts/gone.py::foo()`", "`scripts/gone.py#L3`",
         "`scripts/gone.py:7`", "`scripts/gone.py#L3-L9`", "`scripts/gone.py::a / b`")
    assert _doc_paths(tmp_path) == [f"AGENTS.md:{n}" for n in range(1, 7)]


def test_doc_path_reads_every_suffix_form_on_a_real_file(tmp_path: Path) -> None:
    _write(tmp_path, "x/real.py", SCOPED)
    _doc(tmp_path, "AGENTS.md", "`x/real.py:10-20` `x/real.py:7` `x/real.py#L3` `x/real.py#L3-L9`",
         "`x/real.py::Cls.method()` `x/real.py::Cls.method / Cls.attr` `x/real.py::test_geom_*`")
    assert _doc_paths(tmp_path) == []


def test_doc_path_flags_an_unreadable_suffix(tmp_path: Path) -> None:
    _write(tmp_path, "x/real.py", SCOPED)
    _doc(tmp_path, "AGENTS.md", "`x/real.py and friends`", "`x/real.py:abc`", "`x/real.py::Cls.*`")
    found = quirks.scan(tmp_path)
    assert [f.line for f in found] == [1, 2]
    assert "unreadable suffix ` and friends`" in found[0].message


def test_doc_path_resolves_a_symbol_in_its_scope(tmp_path: Path) -> None:
    _write(tmp_path, "x/real.py", SCOPED)
    _doc(tmp_path, "AGENTS.md",
         "`x/real.py::Alias` `x/real.py::a` `x/real.py::z` `x/real.py::q` `x/real.py::Cls.CONST`",
         "`x/real.py::Cls.attr` `x/real.py::Cls.__init__` `x/real.py::test_geom_2d`",
         "`x/real.py::some_local`",   # a function-local name is not a module name
         "`x/real.py::Nope.method`",  # no such class
         "`x/real.py::Cls.missing`",
         "`x/real.py::a.b.c`",        # deeper than Class.member: not read
         "`x/real.py::attr`")         # an instance attribute is not top-level
    found = quirks.scan(tmp_path)
    assert [f.line for f in found] == [3, 4, 5, 6, 7]
    assert "defines no `some_local` at the top level" in found[0].message
    assert "has no class `Nope`" in found[1].message
    assert "defines no `missing` in class Cls" in found[2].message
    assert "cannot be checked for `a.b.c`" in found[3].message
