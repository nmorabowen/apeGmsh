"""S0a — import-DAG eager/deferred polarity lock (the FP-1 tripwire).

The selection-unification work (``docs/plans/selection-unification.md``)
rests on a load-bearing invariant: the apeGmsh import graph survives a
*latent* ``core <-> mesh`` cycle only because the cross-package edges
have a specific eager/deferred polarity.  Reparenting a Selection onto
a ``core`` base, or any future refactor that adds a NEW eager
cross-package import among ``{core, mesh, viz, results}``, can flip a
deferred edge eager and crash ``import apeGmsh`` (red-team FP-1).

A static cycle detector cannot catch this — the ``core<->mesh`` cycle
already exists.  Only the *set of eager cross-package edges* matters.
This test snapshots that set at file granularity and fails on ANY
growth or change, with a readable diff.  It also asserts the specific
FP-1 mechanism is closed: ``core/__init__.py`` must not eagerly import
the new leaf modules.

If a change here is intentional, update ``BASELINE`` in the same
commit — that makes the import-graph change an explicit, reviewed
diff rather than a silent regression.
"""

from __future__ import annotations

import ast
import importlib
import pathlib

import apeGmsh

# ``_kernel`` + ``fem`` are in scope so the tripwire can police the
# selection-unification-v2 keystone edges *before* ``_kernel`` exists
# (plan HT5/HT10).  ``apeGmsh.fem`` is leaf-pure, but other packages
# eagerly import it; without ``fem`` here those edges were invisible
# and a future ``_kernel<->*`` / ``*<->fem`` cycle could green-wash.
PKGS = {"core", "mesh", "viz", "results", "_kernel", "fem"}

# Frozen snapshot of every EAGER (module-level, non-TYPE_CHECKING)
# cross-package import among PKGS.  (src_pkg, dst_pkg,
# file-relative-to-apeGmsh/).
#
# REFROZEN by selection-unification-v2 **P1-K** (the keystone
# relayer).  The relocation diff *is* the reviewed decision (see the
# module docstring): the pure data/algorithm layer moved into the
# root-leaf ``apeGmsh._kernel`` package, so ``core`` / ``mesh`` /
# ``results`` now import strictly DOWNWARD into ``_kernel`` and the
# latent ``core<->mesh`` cycle is **deleted**.  Classified delta vs the
# pre-P1-K baseline:
#   * REMOVED — all 3 ``('core','mesh',…)`` + all 5 ``('mesh','core',…)``
#     triples, and ``('mesh','fem','mesh/_mass_resolver.py')``
#     (the cycle and the old fem edge are gone).
#   * ADDED — only ``*→_kernel`` downward edges, plus the single
#     ``('_kernel','fem','_kernel/resolvers/_mass_resolver.py')`` (HT10:
#     ``_kernel`` MAY depend on leaf-pure ``apeGmsh.fem``).
# The 2 ``('results','fem',…)`` edges are unchanged (carried).  No
# ``core<->mesh`` triple survives — that is the P1-K invariant.
BASELINE = {
    ("_kernel", "fem", "_kernel/resolvers/_mass_resolver.py"),
    ("core", "_kernel", "core/ConstraintsComposite.py"),
    ("core", "_kernel", "core/Labels.py"),
    ("core", "_kernel", "core/LoadsComposite.py"),
    ("core", "_kernel", "core/MassesComposite.py"),
    ("core", "_kernel", "core/_selection.py"),
    ("core", "_kernel", "core/constraints/defs.py"),
    ("core", "_kernel", "core/loads/defs.py"),
    ("core", "_kernel", "core/masses/defs.py"),
    ("mesh", "_kernel", "mesh/FEMData.py"),
    ("mesh", "_kernel", "mesh/PhysicalGroups.py"),
    ("mesh", "_kernel", "mesh/__init__.py"),
    ("mesh", "_kernel", "mesh/_elem_chain.py"),
    ("mesh", "_kernel", "mesh/_element_types.py"),
    ("mesh", "_kernel", "mesh/_mesh_selection_chain.py"),
    ("mesh", "_kernel", "mesh/_node_chain.py"),
    ("results", "_kernel", "results/_result_chain.py"),
    ("results", "fem", "results/_gauss_extrapolation.py"),
    ("results", "fem", "results/_gauss_world_coords.py"),
}

_ROOT = pathlib.Path(apeGmsh.__file__).parent


def _module_name(f: pathlib.Path) -> str:
    rel = f.relative_to(_ROOT.parent).with_suffix("")
    return ".".join(rel.parts)


def _eager_import_nodes(tree: ast.Module) -> list:
    """Module-level imports, excluding ``if TYPE_CHECKING:`` blocks."""
    out: list = []
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            out.append(n)
        elif isinstance(n, ast.If):
            t = n.test
            is_tc = (
                (isinstance(t, ast.Name) and t.id == "TYPE_CHECKING")
                or (isinstance(t, ast.Attribute)
                    and t.attr == "TYPE_CHECKING")
            )
            if not is_tc:
                for s in ast.walk(n):
                    if isinstance(s, (ast.Import, ast.ImportFrom)) \
                            and s is not n:
                        out.append(s)
    return out


def _abs_module(node: ast.AST, containing_mod: str) -> str | None:
    """Resolve an import node to an absolute dotted module (handles
    relative ``from .. import``)."""
    if isinstance(node, ast.ImportFrom):
        if node.level:
            base = containing_mod.split(".")
            base = base[: len(base) - node.level]
            return ".".join(base + ([node.module] if node.module else []))
        return node.module
    return None


def _compute_edges() -> set[tuple[str, str, str]]:
    triples: set[tuple[str, str, str]] = set()
    for f in sorted(_ROOT.rglob("*.py")):
        rel = f.relative_to(_ROOT).parts
        src = rel[0] if len(rel) > 1 else "_root"
        if src not in PKGS:
            continue
        mod = _module_name(f)
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for n in _eager_import_nodes(tree):
            if isinstance(n, ast.ImportFrom):
                names = [a] if (a := _abs_module(n, mod)) else []
            else:
                names = [x.name for x in n.names]
            for a in names:
                if a and a.startswith("apeGmsh.") \
                        and len(a.split(".")) >= 2:
                    dst = a.split(".")[1]
                    if dst in PKGS and dst != src:
                        triples.add(
                            (src, dst,
                             str(f.relative_to(_ROOT)).replace("\\", "/"))
                        )
    return triples


def test_eager_cross_package_edges_frozen() -> None:
    current = _compute_edges()
    added = sorted(current - BASELINE)
    removed = sorted(BASELINE - current)
    assert not added and not removed, (
        "Eager cross-package import graph changed (FP-1 risk).\n"
        f"  ADDED (new eager edge — likely the regression): {added}\n"
        f"  REMOVED (baseline edge gone): {removed}\n"
        "If intentional, update BASELINE in this commit so the "
        "import-graph change is an explicit, reviewed diff."
    )


def test_core_init_does_not_import_selection_leaves() -> None:
    """The exact FP-1 mechanism: core/__init__.py must stay clear of
    the leaf selection modules, so a sibling leaf is reachable without
    pulling the eager core->mesh chain."""
    src = (_ROOT / "core" / "__init__.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported: set[str] = set()
    for n in tree.body:
        if isinstance(n, ast.ImportFrom) and n.module:
            imported.add(n.module.lstrip("."))
        elif isinstance(n, ast.Import):
            imported.update(a.name for a in n.names)
    for forbidden in ("_selection", "_chain", "_spatial", "_resolution"):
        assert not any(forbidden in m for m in imported), (
            f"core/__init__.py eagerly imports {forbidden!r}; this "
            f"reopens the FP-1 import-cycle risk."
        )


def test_spike_modules_present_and_safe() -> None:
    """S0a spike: the leaf chain + one point-family chainable + the
    deferred host hook import cleanly on the active path."""
    chain = importlib.import_module("apeGmsh._kernel.chain")
    nodec = importlib.import_module("apeGmsh.mesh._node_chain")
    femd = importlib.import_module("apeGmsh.mesh.FEMData")
    assert hasattr(chain, "SelectionChain")
    assert nodec.NodeChain.FAMILY == "point"
    assert callable(getattr(femd.NodeComposite, "select", None))
