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

DECISIONS = "architecture/decisions"


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


# --- bare-version-compare: the dead `_fv < (2, 7, 0)` branch (#1303 PR-6, ADR 0113 INV-8) ---

#: Verbatim from `src/apeGmsh/mesh/_femdata_h5_io.py` at 1691d389, the last commit before this rule.
PRE_CHANGE_FEMDATA_H5_IO = """\
        node_ndf = None
    else:
        # Tuple-compare the dataclass fields (SchemaVersion isn't
        # ordering-enabled, but its fields are comparable).
        _fv = (file_version.major, file_version.minor, file_version.patch)
        if _fv < (2, 7, 0):
            node_ndf = np.zeros(node_ids.shape, dtype=np.int8)
        else:
            node_ndf = None
"""
H5_IO = "src/apeGmsh/mesh/_femdata_h5_io.py"
H5_READER = "src/apeGmsh/opensees/emitter/h5_reader.py"


def test_bare_version_compare_flags_the_pre_change_femdata_h5_io_branch(tmp_path: Path) -> None:
    _write(tmp_path, H5_IO, "def load(file_version, node_ids, np):\n    if True:\n" + PRE_CHANGE_FEMDATA_H5_IO)
    assert _found(tmp_path) == ["bare-version-compare:_femdata_h5_io.py:8"]


def test_bare_version_compare_flags_the_real_pre_change_file(tmp_path: Path) -> None:
    import subprocess

    shown = subprocess.run(
        ["git", "-C", str(quirks.REPO), "show", f"1691d389:{H5_IO}"],
        capture_output=True, text=True, encoding="utf-8",
    )
    if shown.returncode != 0:
        pytest.skip("commit 1691d389 is not in this clone")
    _write(tmp_path, H5_IO, shown.stdout)
    found = [f for f in quirks.scan(tmp_path) if f.rule == "bare-version-compare"]
    assert [shown.stdout.splitlines()[f.line - 1].strip() for f in found] == ["if _fv < (2, 7, 0):"]


def _read_spatial_ndm() -> str:
    import ast

    source = (quirks.REPO / H5_READER).read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "read_spatial_ndm")
    return "from x import META_NDM_IS_SPATIAL_FROM, read_zone_version, NEUTRAL\n\n" + (
        ast.get_source_segment(source, node) or ""
    )


def test_bare_version_compare_passes_read_spatial_ndm_on_its_named_constant(tmp_path: Path) -> None:
    function = _read_spatial_ndm()
    assert ">= META_NDM_IS_SPATIAL_FROM" in function
    _write(tmp_path, H5_READER, function)
    assert _found(tmp_path) == []


def test_bare_version_compare_flags_read_spatial_ndm_with_the_constant_inlined(tmp_path: Path) -> None:
    function = _read_spatial_ndm().replace(">= META_NDM_IS_SPATIAL_FROM", ">= (2, 34, 0)")
    assert ">= (2, 34, 0)" in function
    _write(tmp_path, H5_READER, function)
    assert [f.rule for f in quirks.scan(tmp_path)] == ["bare-version-compare"]


@pytest.mark.parametrize(
    "compare",
    [
        "ver >= (2, 26, 1)",
        "(2, 7, 0) > ver",
        "(v.major, v.minor) < (2, 7)",
        "file_version >= SchemaVersion(2, 7, 0)",
        'file_version >= SchemaVersion.parse("2.7.0")',
        "schema_version < parse_version('2.7.0')",
    ],
)
def test_bare_version_compare_flags_every_literal_shape(tmp_path: Path, compare: str) -> None:
    _write(tmp_path, "src/apeGmsh/mesh/a.py", f"def f(ver, v, file_version, schema_version):\n    return {compare}\n")
    assert _found(tmp_path) == ["bare-version-compare:a.py:2"]


@pytest.mark.parametrize(
    "compare",
    [
        "ver >= SOME_FEATURE_FROM",
        "(v.major, v.minor, v.patch) >= ADR_FLOOR",
        "ver < module.COMPAT_FLOOR",
        "arr.shape == (3, 3)",
        "sys.version_info >= (3, 11)",
        "(a, b) == (1, 2)",
        "ver == other_ver",
    ],
)
def test_bare_version_compare_passes_named_constants_and_non_versions(tmp_path: Path, compare: str) -> None:
    _write(tmp_path, "src/apeGmsh/mesh/a.py", f"def f(ver, v, arr, a, b):\n    return {compare}\n")
    assert _found(tmp_path) == []


def test_bare_version_compare_is_scoped_to_the_package(tmp_path: Path) -> None:
    _write(tmp_path, "tests/test_a.py", "def test_it(ver):\n    assert ver < (2, 7, 0)\n")
    assert _found(tmp_path) == []


def test_bare_version_compare_waiver_suppresses_one_site(tmp_path: Path) -> None:
    _write(tmp_path, "src/apeGmsh/mesh/a.py", """\
        def f(ver):
            # apegmsh-lint: bare-version-compare-ok the third-party library's own version.
            return ver < (2, 7, 0)
        """)
    assert _found(tmp_path) == []


def test_the_checkout_has_no_bare_version_compare() -> None:
    assert [f for f in quirks.scan(quirks.REPO) if f.rule == "bare-version-compare"] == []


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

ARCH = "architecture"
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
         "See [testing.md](architecture/testing.md) and `mesh/FEMData.py`.",
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
    _doc(tmp_path, f"{ARCH}/h5-schema.md", "([README](README.md)) and [t](testing.md)")
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


# --- arch-path: no file at the old architecture folder (N3, #1197) -----------


def test_arch_path_flags_a_file_at_the_old_path(tmp_path: Path) -> None:
    old = "src/apeGmsh/opensees/" + "architecture/x.md"
    _write(tmp_path, old, "# x\n")
    found = [f for f in quirks.scan(tmp_path) if f.rule == "arch-path"]
    assert [f.path for f in found] == [old]
    assert "`architecture/` since N3 (#1197)" in found[0].message
    assert "doc tree" in found[0].message


def test_arch_path_passes_the_same_file_under_architecture(tmp_path: Path) -> None:
    _write(tmp_path, "architecture/x.md", "# x\n")
    assert [f for f in quirks.scan(tmp_path) if f.rule == "arch-path"] == []


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


# --- ratchet-baseline: a shrink-only list that can still grow (#1240) --------

REPO_ROOT = Path(__file__).resolve().parents[1]

PRE_FIX_EXCEPTIONS = '''\
"""Gate configuration. ``EXCEPTIONS`` is a ratchet: it may only shrink."""
EXCEPTIONS: dict[str, str] = {
    "apeGmsh.opensees.integration:Lobatto": "no family",
}
'''


def _real(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def test_ratchet_baseline_flags_the_pre_fix_exceptions_shape(tmp_path: Path) -> None:
    # The shape of tests/families.py when #1232 shipped it (0877f388, before the baseline).
    _write(tmp_path, "tests/families.py", PRE_FIX_EXCEPTIONS)
    assert _found(tmp_path) == ["ratchet-baseline:families.py:2"]


@pytest.mark.parametrize(
    "assign",
    [
        "EXTRAS_ONLY = frozenset({'a'})",
        "ALLOWLIST = ['a']",
        "GRANDFATHERED = ('a',)",
        "EXCEPTIONS = {'a'}",
        "EXCEPTIONS: set[str] = {'a'}",
    ],
)
def test_ratchet_baseline_flags_every_list_shape(tmp_path: Path, assign: str) -> None:
    _write(tmp_path, "tests/a.py", assign + "\n")
    assert [f.rule for f in quirks.scan(tmp_path)] == ["ratchet-baseline"]


@pytest.mark.parametrize("rel", ["tests/families.py", "tests/opensees/unit/test_element_capability_unknown.py"])
def test_ratchet_baseline_passes_the_fixed_files_from_main(tmp_path: Path, rel: str) -> None:
    _write(tmp_path, rel, _real(rel))
    assert [f for f in quirks.scan(tmp_path) if f.rule == "ratchet-baseline"] == []


def test_ratchet_baseline_passes_other_names_and_non_literals(tmp_path: Path) -> None:
    _write(tmp_path, "tests/a.py", "EXCEPTIONS_BASELINE = 3\nOTHER = {'a'}\nEXCEPTIONS = build()\n")
    assert _found(tmp_path) == []


def test_ratchet_baseline_is_scoped_to_tests(tmp_path: Path) -> None:
    _write(tmp_path, "src/apeGmsh/mesh/a.py", "EXCEPTIONS = {'a'}\n")
    assert _found(tmp_path) == []


def test_ratchet_baseline_waiver_suppresses_one_site(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/a.py",
        "# apegmsh-lint: ratchet-baseline-ok a fixed table, not a ratchet.\nALLOWLIST = {'a'}\n",
    )
    assert _found(tmp_path) == []


# --- qt-process-isolation: Qt + a thread in the shared process (#1242) -------

B6_BUG_COMMIT = "8269206d"
B6 = "tests/sections/test_builder_gui_b6.py"


def _b6_at(commit: str) -> str:
    import subprocess

    done = subprocess.run(
        ["git", "show", f"{commit}:{B6}"], cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8"
    )
    if done.returncode != 0:
        pytest.skip(f"{commit} is not in this clone")
    return done.stdout


def test_qt_process_isolation_flags_b6_at_the_commit_that_had_the_bug(tmp_path: Path) -> None:
    _write(tmp_path, B6, _b6_at(B6_BUG_COMMIT))
    assert [f.rule for f in quirks.scan(tmp_path)] == ["qt-process-isolation"]


def test_qt_process_isolation_passes_b6_on_main(tmp_path: Path) -> None:
    _write(tmp_path, B6, _real(B6))
    assert quirks.scan(tmp_path) == []


QT_THREAD = '''\
import threading
import pytest
{mark}
def test_it():
    qt = pytest.importorskip("qtpy.QtWidgets")
    threading.Thread(target=print).start()
'''


# Built by concatenation so this file never holds the literal qt marker that
# test_qt_lane_coverage scans for (#1241); the runtime strings are unchanged.
_QT = "pytest.mark." + "qt"


@pytest.mark.parametrize(
    ("mark", "flagged"),
    [
        ("", True),
        ("pytestmark = pytest.mark.slow", True),
        (f"pytestmark = {_QT}", False),
        ("pytestmark = [pytest.mark.subprocess]", False),
        (f"pytestmark: list = [pytest.mark.slow, {_QT}]", False),
    ],
)
def test_qt_process_isolation_needs_a_module_level_mark(tmp_path: Path, mark: str, flagged: bool) -> None:
    _write(tmp_path, "tests/a.py", QT_THREAD.format(mark=mark))
    assert [f.rule for f in quirks.scan(tmp_path)] == (["qt-process-isolation"] if flagged else [])


@pytest.mark.parametrize(
    ("source", "flagged"),
    [
        ("import threading\ndef test_it():\n    threading.Thread(target=print)\n", False),  # a thread, no Qt
        ("from qtpy import QtWidgets\ndef test_it():\n    pass\n", False),                  # Qt, no thread
        (
            "from apeGmsh.sections._properties import PropertiesController\n"
            "def test_it():\n    PropertiesController()\n",
            False,  # the _properties worker is a thread only: no Qt binding, not the #1242 class
        ),
    ],
)
def test_qt_process_isolation_needs_both_halves(tmp_path: Path, source: str, flagged: bool) -> None:
    _write(tmp_path, "tests/a.py", source)
    assert [f.rule for f in quirks.scan(tmp_path)] == (["qt-process-isolation"] if flagged else [])


def test_qt_process_isolation_is_scoped_to_tests(tmp_path: Path) -> None:
    _write(tmp_path, "src/apeGmsh/viewers/a.py", QT_THREAD.format(mark=""))
    assert _found(tmp_path) == []


def test_qt_process_isolation_passes_test_properties_from_main(tmp_path: Path) -> None:
    rel = "tests/sections/test_properties.py"
    _write(tmp_path, rel, _real(rel))
    assert quirks.scan(tmp_path) == []


# --- compose-streams: keyed by package glob, so a split keeps the rule -------


@pytest.mark.parametrize(
    "rel",
    [
        "src/apeGmsh/mesh/_compose/_rebuild.py",       # _compose.py became a package
        "src/apeGmsh/mesh/_femdata_h5_io/_nodes.py",   # the h5 I/O became a package
        "src/apeGmsh/mesh/_femdata_h5/_elements.py",   # or was split under a new name
    ],
)
def test_compose_streams_fires_at_a_post_split_path(tmp_path: Path, rel: str) -> None:
    _write(tmp_path, "src/apeGmsh/mesh/FEMData.py", FEMDATA)
    _write(tmp_path, rel, "new = ElementComposite(groups=g, physical=p)\n")
    assert [f.rule for f in quirks.scan(tmp_path)] == ["compose-streams"]


def test_compose_streams_passes_a_complete_rebuild_at_a_post_split_path(tmp_path: Path) -> None:
    _write(tmp_path, "src/apeGmsh/mesh/FEMData.py", FEMDATA)
    _write(tmp_path, "src/apeGmsh/mesh/_compose/_rebuild.py", "new = ElementComposite(g, p, contacts=c, embed_ties=e)\n")
    assert _found(tmp_path) == []


def test_compose_streams_glob_leaves_other_mesh_modules_alone(tmp_path: Path) -> None:
    _write(tmp_path, "src/apeGmsh/mesh/FEMData.py", FEMDATA)
    _write(tmp_path, "src/apeGmsh/mesh/_femdata_hash.py", "new = ElementComposite(groups=g, physical=p)\n")
    _write(tmp_path, "src/apeGmsh/mesh/_other.py", "e = ElementComposite(g, p)\n")
    assert _found(tmp_path) == []


# --- stale-patch-target: a string target no longer in src/ -------------------

PATCH_SRC = """\
    from other import reexported

    CONSTANT = 1

    def helper():
        pass

    class Base:
        inherited = 1

    class Widget:
        def method(self):
            self.attr = 1

    class Child(Base):
        pass
"""


def _patch_tree(tmp_path: Path, test_source: str) -> Path:
    _write(tmp_path, "src/apeGmsh/mesh/_mod.py", PATCH_SRC)
    _write(tmp_path, "src/apeGmsh/mesh/__init__.py", "from ._mod import helper\n")
    _write(tmp_path, "tests/test_a.py", test_source)
    return tmp_path


@pytest.mark.parametrize(
    "call",
    [
        'patch("apeGmsh.mesh._mod.gone")',                                  # the name was removed
        'mock.patch("apeGmsh.mesh._old.helper")',                           # the module moved
        'monkeypatch.setattr("apeGmsh.mesh._mod.gone", 1)',                 # same, via monkeypatch
        'patch("apeGmsh.mesh._mod.Widget.gone")',                           # a class member removed
        'patch("apeGmsh.nowhere.helper")',                                  # the package is gone
        'mocker.patch("apeGmsh.mesh.missing")',                             # not in the package __init__
    ],
)
def test_stale_patch_target_flags_a_target_that_does_not_resolve(tmp_path: Path, call: str) -> None:
    _patch_tree(tmp_path, f"def test_it(monkeypatch, mocker):\n    {call}\n")
    found = quirks.scan(tmp_path)
    assert [f.rule for f in found] == ["stale-patch-target"]
    assert call.split('"')[1] in found[0].message


@pytest.mark.parametrize(
    "call",
    [
        'patch("apeGmsh.mesh._mod.helper")',            # a def
        'patch("apeGmsh.mesh._mod.CONSTANT")',          # an assignment
        'patch("apeGmsh.mesh._mod.reexported")',        # an import re-export
        'patch("apeGmsh.mesh._mod.Widget")',            # a class
        'patch("apeGmsh.mesh._mod.Widget.method")',     # a method
        'patch("apeGmsh.mesh._mod.Widget.attr")',       # a self attribute
        'patch("apeGmsh.mesh._mod.Child.inherited")',   # inherited: a base the file does not show
        'patch("apeGmsh.mesh.helper")',                 # re-exported by the package __init__
        'patch("apeGmsh.mesh._mod")',                   # a module
        'patch("apeGmsh.mesh")',                        # a package
        'patch("os.path.join")',                        # not first-party
        'patch("gmsh.model.add")',                      # third party
        'monkeypatch.setattr("apeGmsh.mesh._mod.helper", 1)',
        'monkeypatch.setattr(obj, "gone", 1)',          # an object, not a string target
        'patch(target)',                                # a dynamic target
        'patch("apeGmsh.mesh._mod." + name)',           # computed
        'patch.object(mod, "gone")',                    # not a string target
    ],
)
def test_stale_patch_target_passes_a_target_that_resolves_or_is_unreadable(tmp_path: Path, call: str) -> None:
    _patch_tree(tmp_path, f"def test_it(monkeypatch, mocker, obj, target, name, mod):\n    {call}\n")
    assert _found(tmp_path) == []


def test_stale_patch_target_fires_at_a_post_split_path(tmp_path: Path) -> None:
    # _mod.py became a package whose __init__ does not re-export helper: the old string is stale.
    _write(tmp_path, "src/apeGmsh/mesh/_mod/__init__.py", "from ._impl import other\n")
    _write(tmp_path, "src/apeGmsh/mesh/_mod/_impl.py", "def helper():\n    pass\n\ndef other():\n    pass\n")
    _write(tmp_path, "tests/test_a.py", 'def test_it():\n    patch("apeGmsh.mesh._mod.helper")\n')
    assert [f.rule for f in quirks.scan(tmp_path)] == ["stale-patch-target"]
    _write(tmp_path, "tests/test_a.py", 'def test_it():\n    patch("apeGmsh.mesh._mod._impl.helper")\n')
    assert _found(tmp_path) == []


def test_stale_patch_target_stays_silent_when_a_module_can_supply_any_name(tmp_path: Path) -> None:
    _write(tmp_path, "src/apeGmsh/mesh/_mod.py", "from ._impl import *\n")
    _write(tmp_path, "src/apeGmsh/mesh/_lazy.py", "def __getattr__(name):\n    raise AttributeError(name)\n")
    _write(tmp_path, "tests/test_a.py", 'def test_it():\n    patch("apeGmsh.mesh._mod.x")\n    patch("apeGmsh.mesh._lazy.y")\n')
    assert _found(tmp_path) == []


def test_stale_patch_target_reads_a_multiline_with_block(tmp_path: Path) -> None:
    _patch_tree(tmp_path, """\
        def test_it():
            with patch(
                "apeGmsh.mesh._mod.gone",
                return_value=1,
            ):
                pass
        """)
    assert _found(tmp_path) == ["stale-patch-target:test_a.py:2"]


def test_stale_patch_target_is_scoped_to_tests(tmp_path: Path) -> None:
    _patch_tree(tmp_path, "x = 1\n")
    _write(tmp_path, "src/apeGmsh/mesh/_user.py", 'patch("apeGmsh.mesh._mod.gone")\n')
    assert _found(tmp_path) == []


def test_stale_patch_target_waiver_suppresses_one_site(tmp_path: Path) -> None:
    _patch_tree(tmp_path, """\
        def test_it():
            # apegmsh-lint: stale-patch-target-ok the module is generated at import time
            patch("apeGmsh.mesh._mod.gone")
            patch("apeGmsh.mesh._mod.gone2")
        """)
    assert _found(tmp_path) == ["stale-patch-target:test_a.py:4"]


def test_stale_patch_target_stale_waiver_is_a_finding(tmp_path: Path) -> None:
    _patch_tree(tmp_path, """\
        def test_it():
            # apegmsh-lint: stale-patch-target-ok was dynamic once
            patch("apeGmsh.mesh._mod.helper")
        """)
    assert _found(tmp_path) == ["waiver:test_a.py:2"]


def test_stale_patch_target_passes_every_target_on_main() -> None:
    found = [f for f in quirks.scan(quirks.REPO) if f.rule == "stale-patch-target"]
    assert found == []


# --- emitter-sniff: ADR 0114 D6, the two apesees.py sniffs (K1-5, #1462) ------

SNIFF_COMMIT = "11690cc7"
SNIFF_SITES = {
    "src/apeGmsh/opensees/apesees.py": {1121, 5101},
    "src/apeGmsh/opensees/_internal/compose.py": {1277},
}


def test_emitter_sniff_flags_the_sites_at_the_commit_that_had_them(tmp_path: Path) -> None:
    for rel in SNIFF_SITES:
        _write(tmp_path, rel, _file_at(SNIFF_COMMIT, rel))
    found: dict[str, set[int]] = {}
    for f in quirks.scan(tmp_path):
        if f.rule == "emitter-sniff":
            found.setdefault(f.path, set()).add(f.line)
    assert found == SNIFF_SITES


@pytest.mark.parametrize(
    "sniff",
    [
        'type(emitter).__name__ == "H5Emitter"',          # BuiltModel.emit
        '"H5Emitter" != type(emitter).__name__',
        'emitter.__class__.__name__ == "TclEmitter"',
        'type(e).__name__ in ("TclEmitter", "PyEmitter")',
        "isinstance(emitter, H5Emitter)",                  # _guard_mass_from_model
        "isinstance(emitter, h5.H5Emitter)",
        "isinstance(emitter, (int, LiveOpsEmitter))",
        "issubclass(type(emitter), RecordingEmitter)",
        "type(emitter) is H5Emitter",                       # review of #1616
        "type(emitter) == H5Emitter",
        "H5Emitter is not type(emitter)",
        "emitter.__class__ is H5Emitter",
        'type(emitter).__qualname__ == "H5Emitter"',
        'type(emitter).__name__.startswith("H5")',
        'emitter.__class__.__qualname__.endswith("Emitter")',
        '"H5" in type(emitter).__name__',
    ],
)
def test_emitter_sniff_flags_every_spelling(tmp_path: Path, sniff: str) -> None:
    _write(tmp_path, "src/apeGmsh/opensees/apesees.py", f"def archival(emitter):\n    return {sniff}\n")
    assert _found(tmp_path) == ["emitter-sniff:apesees.py:2"]


@pytest.mark.parametrize(
    "source",
    [
        "from apeGmsh.opensees.emitter.h5 import H5Emitter as _H5\n"
        "def f(e):\n    return isinstance(e, _H5)\n",
        "import apeGmsh.opensees.emitter.h5 as h5\n"
        "from apeGmsh.opensees.emitter.live import LiveOpsEmitter as Live\n"
        "def f(e):\n    return type(e) is Live\n",
        "def f(e):\n    match e:\n        case H5Emitter():\n            return True\n",
        "from apeGmsh.opensees.emitter.tcl import TclEmitter as T\n"
        "def f(e):\n    match e:\n        case T(lines=_):\n            return True\n",
    ],
    ids=["isinstance-alias", "type-is-alias", "match-case", "match-case-alias"],
)
def test_emitter_sniff_flags_aliases_and_match(tmp_path: Path, source: str) -> None:
    _write(tmp_path, "src/apeGmsh/opensees/apesees.py", source)
    found = _found(tmp_path)
    assert len(found) == 1 and found[0].startswith("emitter-sniff:apesees.py:")


def test_emitter_sniff_reaches_every_package_but_the_emitter_one(tmp_path: Path) -> None:
    body = "def live(emitter):\n    return isinstance(emitter, LiveOpsEmitter)\n"
    _write(tmp_path, "src/apeGmsh/opensees/_internal/compose.py", body)
    _write(tmp_path, "src/apeGmsh/results/capture/spec.py", body)
    _write(tmp_path, "src/apeGmsh/opensees/emitter/h5.py", body)
    _write(tmp_path, "src/apeGmsh/opensees/emitter/sub/helper.py", body)
    assert _found(tmp_path) == ["emitter-sniff:compose.py:2", "emitter-sniff:spec.py:2"]


@pytest.mark.parametrize(
    "line",
    [
        "isinstance(emitter, Primitive)",
        "isinstance(p, (Analysis, Recorder))",
        "type(emitter) is Frame",                          # not an emitter class
        'type(emitter).__name__',                          # read for a message, not asked about
        'f"got {type(emitter).__name__}"',
        'emitter.caps.archival',                           # the replacement
        'name = "H5Emitter"',
        "emitter.lines",
    ],
)
def test_emitter_sniff_passes_what_does_not_ask_for_the_class(tmp_path: Path, line: str) -> None:
    _write(tmp_path, "src/apeGmsh/opensees/apesees.py", f"def f(emitter, p, Frame):\n    return {line}\n")
    assert _found(tmp_path) == []


@pytest.mark.parametrize("rel", ["tests/opensees/test_x.py", "examples/demo.py", "scripts/tool.py"])
def test_emitter_sniff_ignores_code_outside_src(tmp_path: Path, rel: str) -> None:
    _write(tmp_path, rel, "def f(emitter):\n    return isinstance(emitter, H5Emitter)\n")
    assert _found(tmp_path) == []


def test_emitter_sniff_waiver(tmp_path: Path) -> None:
    _write(tmp_path, "src/apeGmsh/opensees/apesees.py", """\
        def f(emitter):
            # apegmsh-lint: emitter-sniff-ok a test double has no caps
            return isinstance(emitter, H5Emitter)
        """)
    assert _found(tmp_path) == []


def test_emitter_sniff_is_silent_in_this_checkout() -> None:
    assert [f for f in quirks.scan(quirks.REPO) if f.rule == "emitter-sniff"] == []


def test_emitter_sniff_scope_exists_in_this_checkout() -> None:
    assert (quirks.REPO / quirks.EMITTER_PACKAGE).is_dir(), "the emitter package moved: update EMITTER_PACKAGE"


# --- raw-meta-ndm: #1405, fabfa042 -------------------------------------------

NDM_BUG_COMMIT = "fabfa042"
NDM_SITES = {
    "src/apeGmsh/opensees/model_data.py": 486,
    "src/apeGmsh/results/capture/_domain.py": 652,
    "src/apeGmsh/opensees/emitter/h5_reader.py": 514,
}


def _file_at(commit: str, rel: str) -> str:
    import subprocess

    done = subprocess.run(
        ["git", "show", f"{commit}:{rel}"], cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8"
    )
    if done.returncode != 0:
        pytest.skip(f"{commit} is not in this clone")
    return done.stdout


def test_raw_meta_ndm_flags_the_three_raw_reads_at_the_commit_that_had_them(tmp_path: Path) -> None:
    for rel in NDM_SITES:
        _write(tmp_path, rel, _file_at(NDM_BUG_COMMIT, rel))
    found = {f.path: f.line for f in quirks.scan(tmp_path) if f.rule == "raw-meta-ndm"}
    assert found == NDM_SITES


def test_raw_meta_ndm_is_silent_on_main() -> None:
    assert [f for f in quirks.scan(quirks.REPO) if f.rule == "raw-meta-ndm"] == []


@pytest.mark.parametrize(
    "read",
    [
        'int(meta.attrs.get("ndm", 3))',
        'int(meta["ndm"])',
        'int(self.meta().get("ndm", 3) or 3)',
        'int(model.meta()["ndm"])',
        'int(attrs.get("ndm", 0))',
        'int(h5.meta.get("ndm"))',
    ],
)
def test_raw_meta_ndm_flags_a_raw_read(tmp_path: Path, read: str) -> None:
    _write(tmp_path, "src/apeGmsh/results/_x.py", f"def f(meta, attrs, self, model, h5):\n    return {read}\n")
    assert _found(tmp_path) == ["raw-meta-ndm:_x.py:2"]


@pytest.mark.parametrize(
    "source",
    [
        'def f(meta):\n    meta.attrs["ndm"] = 3\n',                                # a write
        'def f():\n    return {"ndm": 3}\n',                                        # a dict literal
        'def f(cfg):\n    return cfg.get("ndm", 3)\n',                             # not a meta receiver
        'def f(meta):\n    return meta.get("ndf", 3)\n',                           # another key
        'import json\ndef f(p):\n    meta = json.loads(p)\n    return meta.get("ndm", 3)\n',  # a JSON sidecar
        'def f(meta, h5_reader):\n    return h5_reader.read_spatial_ndm(meta, None, None)\n',
    ],
)
def test_raw_meta_ndm_passes_other_shapes(tmp_path: Path, source: str) -> None:
    _write(tmp_path, "src/apeGmsh/results/_x.py", source)
    assert _found(tmp_path) == []


def test_raw_meta_ndm_is_scoped_to_src(tmp_path: Path) -> None:
    _write(tmp_path, "tests/test_a.py", 'def test_it(meta):\n    assert meta["ndm"] == 3\n')
    assert _found(tmp_path) == []


def test_raw_meta_ndm_exempts_the_reader_and_its_helpers_only(tmp_path: Path) -> None:
    _write(tmp_path, "src/apeGmsh/opensees/emitter/h5_reader.py", """\
        def read_spatial_ndm(meta, f, coords):
            return int(meta.get("ndm", 0))

        def _trusted_meta_ndm(meta, why):
            return int(meta["ndm"])

        def _salvage_pre_spatial_ndm(stamp, f, coords):
            return int(f.attrs["meta"].get("ndm"))

        def other(meta):
            return int(meta["ndm"])
        """)
    assert _found(tmp_path) == ["raw-meta-ndm:h5_reader.py:11"]


def test_raw_meta_ndm_waiver_suppresses_one_site(tmp_path: Path) -> None:
    _write(tmp_path, "src/apeGmsh/results/_x.py", """\
        def f(meta):
            # apegmsh-lint: raw-meta-ndm-ok a bridge-only sidecar, never a neutral file
            a = meta["ndm"]
            return a, meta["ndm"]
        """)
    assert _found(tmp_path) == ["raw-meta-ndm:_x.py:4"]


# --- comment-provenance: added comments only, read from a diff ----------------


def _git(root: Path, *args: str) -> str:
    import subprocess

    done = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
        cwd=root, capture_output=True, text=True, encoding="utf-8",
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def _repo_with(tmp_path: Path, before: str, after: str, rel: str = "src/apeGmsh/_m.py") -> Path:
    _git(tmp_path, "init", "-q", "-b", "main")
    _write(tmp_path, rel, before)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    _git(tmp_path, "checkout", "-q", "-b", "topic")
    _write(tmp_path, rel, after)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "topic")
    return tmp_path


def _provenance(root: Path, base: str = "main") -> list[str]:
    return [f"{f.path}:{f.line}" for f in quirks.scan(root, base) if f.rule == "comment-provenance"]


@pytest.mark.parametrize(
    "comment",
    [
        "# fixed in #1234",
        "# see PR #99 for the reason",
        "# shipped in 2.34.0",
        "# shipped with the h5 reader",
        "# as of 2026-09 this is the default",
        "# as of 12 September 2026 only 3-D is read",
        "# as of Sept 2026 only 3-D is read",
        "# as of 2026 only 3-D is read",
        "# previously this returned None",
        "# it used to raise",
        "# Previously a dict",
    ],
)
def test_comment_provenance_flags_an_added_comment_with_history(tmp_path: Path, comment: str) -> None:
    root = _repo_with(tmp_path, "x = 1\n", f"x = 1\n{comment}\ny = 2\n")
    assert _provenance(root) == ["src/apeGmsh/_m.py:2"]


@pytest.mark.parametrize(
    "comment",
    [
        "# the node ids are dense and 1-based",
        "# issue the query once, then cache it",           # the word, not a number
        "# step #1 of the pipeline",                       # a single digit is a step, not a ticket
        "# as of the next call the cache is warm",         # no date
        "# a used tool and a previous one",
    ],
)
def test_comment_provenance_passes_a_comment_that_says_what_the_code_does(tmp_path: Path, comment: str) -> None:
    root = _repo_with(tmp_path, "x = 1\n", f"x = 1\n{comment}\ny = 2\n")
    assert _provenance(root) == []


def test_comment_provenance_never_flags_an_existing_comment(tmp_path: Path) -> None:
    before = "# fixed in #1234, previously a dict\nx = 1\n"
    root = _repo_with(tmp_path, before, before + "y = 2  # a clean note\n")
    assert _provenance(root) == []


def test_comment_provenance_flags_a_trailing_comment_and_reports_its_line(tmp_path: Path) -> None:
    root = _repo_with(tmp_path, "x = 1\n", "x = 1\ny = 2\nz = 3  # was #77\n")
    assert _provenance(root) == ["src/apeGmsh/_m.py:3"]


def test_comment_provenance_reads_a_string_as_code_not_a_comment(tmp_path: Path) -> None:
    root = _repo_with(tmp_path, "x = 1\n", 'x = 1\ny = "see #1234 previously"\n')
    assert _provenance(root) == []


def test_comment_provenance_is_scoped_to_src(tmp_path: Path) -> None:
    root = _repo_with(tmp_path, "x = 1\n", "x = 1\n# fixed in #1234\n", rel="tests/test_a.py")
    assert _provenance(root) == []


def test_comment_provenance_waiver_suppresses_one_site(tmp_path: Path) -> None:
    after = (
        "x = 1\n"
        "# apegmsh-lint: comment-provenance-ok the issue is the contract this guards\n"
        "# see #1234 for the reproducer\n"
        "y = 2  # apegmsh-lint: comment-provenance-ok quotes the upstream ticket #55\n"
        "z = 3  # was #77\n"
    )
    root = _repo_with(tmp_path, "x = 1\n", after)
    assert _provenance(root) == ["src/apeGmsh/_m.py:5"]
    assert [f for f in quirks.scan(root, "main") if f.rule == "waiver"] == []


def test_comment_provenance_waiver_needs_a_reason(tmp_path: Path) -> None:
    root = _repo_with(tmp_path, "x = 1\n", "x = 1\n# was #77  # apegmsh-lint: comment-provenance-ok\n")
    assert sorted(f.rule for f in quirks.scan(root, "main")) == ["comment-provenance", "waiver"]


def test_comment_provenance_is_silent_without_a_base(tmp_path: Path) -> None:
    _repo_with(tmp_path, "x = 1\n", "x = 1\n# fixed in #1234\n")
    assert _found(tmp_path) == []


def test_comment_provenance_fails_loudly_on_an_unknown_base(tmp_path: Path) -> None:
    root = _repo_with(tmp_path, "x = 1\n", "x = 2\n")
    with pytest.raises(SystemExit, match="merge-base"):
        quirks.scan(root, "no-such-ref")


def test_comment_provenance_runs_from_the_command_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _repo_with(tmp_path, "x = 1\n", "x = 1\n# fixed in #1234\n")
    assert quirks.main(["--root", str(root), "--base", "main"]) == 1
    assert "[comment-provenance]" in capsys.readouterr().out
    assert quirks.main(["--root", str(root), "--no-base"]) == 0


def test_added_lines_reads_a_zero_context_diff() -> None:
    diff = textwrap.dedent("""\
        diff --git a/src/a.py b/src/a.py
        --- a/src/a.py
        +++ b/src/a.py
        @@ -3 +3,2 @@ def f():
        -old
        +new
        +newer
        @@ -10,2 +11,0 @@
        -gone
        -gone
        diff --git a/src/b.py b/src/b.py
        --- a/src/b.py
        +++ /dev/null
        @@ -1 +0,0 @@
        -deleted
        """)
    assert quirks._added_lines(diff) == {"src/a.py": {3, 4}}


# --- comment-provenance: a comment moved verbatim is not new provenance -------


def _two_file_repo(tmp_path: Path, before: dict[str, str], after: dict[str, str]) -> Path:
    _git(tmp_path, "init", "-q", "-b", "main")
    for rel, text in before.items():
        _write(tmp_path, rel, text)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    _git(tmp_path, "checkout", "-q", "-b", "topic")
    for rel, text in after.items():
        _write(tmp_path, rel, text)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "topic")
    return tmp_path


def test_comment_provenance_passes_a_comment_moved_verbatim_between_files(tmp_path: Path) -> None:
    root = _two_file_repo(
        tmp_path,
        {"src/apeGmsh/_a.py": "def f():\n    # kept for the hub (ADR 0112 D3, #1378)\n    return 1\n",
         "src/apeGmsh/_b.py": "z = 0\n"},
        {"src/apeGmsh/_a.py": "def f():\n    return 1\n",
         "src/apeGmsh/_b.py": "z = 0\ndef f():\n    # kept for the hub (ADR 0112 D3, #1378)\n    return 1\n"},
    )
    assert _provenance(root) == []


def test_comment_provenance_flags_a_new_comment_when_another_is_deleted(tmp_path: Path) -> None:
    root = _two_file_repo(
        tmp_path,
        {"src/apeGmsh/_a.py": "# old note #1378\nx = 1\n"},
        {"src/apeGmsh/_a.py": "x = 1\n# brand new note #1500\n"},
    )
    assert _provenance(root) == ["src/apeGmsh/_a.py:2"]


def test_comment_provenance_moved_once_added_twice_is_flagged_once(tmp_path: Path) -> None:
    root = _two_file_repo(
        tmp_path,
        {"src/apeGmsh/_a.py": "# moved note #1378\nx = 1\n", "src/apeGmsh/_b.py": "z = 0\n"},
        {"src/apeGmsh/_a.py": "x = 1\n",
         "src/apeGmsh/_b.py": "z = 0\n# moved note #1378\ny = 1\n# moved note #1378\n"},
    )
    assert len(_provenance(root)) == 1


def test_main_without_base_defaults_to_main_and_flags_provenance(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _repo_with(tmp_path, "x = 1\n", "x = 1\n# fixed in #1234\n")
    assert quirks.main(["--root", str(root)]) == 1
    assert "comment-provenance" in capsys.readouterr().out


def test_main_no_base_skips_the_diff_rules_with_a_notice(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _repo_with(tmp_path, "x = 1\n", "x = 1\n# fixed in #1234\n")
    assert quirks.main(["--root", str(root), "--no-base"]) == 0
    assert "note: comment-provenance not run" in capsys.readouterr().out


def test_main_with_no_main_branch_prints_the_notice_and_does_not_crash(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _repo_with(tmp_path, "x = 1\n", "x = 1\n# fixed in #1234\n")
    _git(root, "branch", "-m", "main", "trunk")
    assert quirks.main(["--root", str(root)]) == 0
    assert "note: comment-provenance not run: no --base" in capsys.readouterr().out


def test_main_outside_a_repo_prints_the_notice(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert quirks.main(["--root", str(tmp_path)]) == 0
    assert "note: comment-provenance not run" in capsys.readouterr().out


def test_main_prefers_origin_main_over_local_main(tmp_path: Path) -> None:
    root = _repo_with(tmp_path, "x = 1\n", "x = 1\n# fixed in #1234\n")
    _git(root, "update-ref", "refs/remotes/origin/main", "main")
    assert quirks.default_base(root) == "origin/main"
    assert quirks.main(["--root", str(root)]) == 1


def test_main_explicit_base_is_unchanged(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _repo_with(tmp_path, "x = 1\n", "x = 1\n# fixed in #1234\n")
    assert quirks.main(["--root", str(root), "--base", "main"]) == 1
    assert "note:" not in capsys.readouterr().out
