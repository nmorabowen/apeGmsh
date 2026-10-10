"""Tag law (K1-3, #1361; K1-3d S6, #1458): emit and replay never mint a tag.

A tag is an archive fact; replay never allocates (K0 design, #1341). The
build-time tag plan (design #1445, ADR 0114 D4 amended) makes this true for
``BuiltModel.emit``: every tag is planned and frozen by ``plan_tags``, and
the emit hands the :class:`TagPlan` itself, which has no allocation API, to
every helper. Two locks hold it.

The replay lock: no module under ``emitter/``, and neither
``_internal/compose.py`` nor ``opensees_model.py``, may break the rules
below except at a waived ledger site.

The hub lock (S6): no ``BuiltModel`` method (the emit) except
``_tag_plan``, which runs the planner, no module-level function of
``apesees.py``, no function of the build modules outside the planners and
the three replay entry points, and nothing in ``recorder.py`` may break
them, with no waiver at all. ``max_plus_one`` is a replay-lock rule only:
the hubs compute ``max(...) + 1`` for things that are not tags (a DOF
range).

"The build" is a module *set* (S2-0, #1611): ``_internal/build.py`` today,
every module under ``_internal/build/`` once S2 splits it, whichever
exists (:func:`build_modules`). The minting helpers are derived over the
union of the set, following calls across its modules through
``from .x import name`` and ``from apeGmsh.opensees._internal.build[.x]
import name`` (:func:`derive_minting_helpers`), and the hub scope covers
every module of the set. Only those static ``ImportFrom`` forms are
followed: a build module imported as a module object, or star-imported,
fails the derivation outright rather than letting a helper that mints
through such an import go unseen, and so does a module-level alias or
dispatch table naming a build function.

The rules:

``allocate``
    reference a minting method: ``TagAllocator``'s ``allocate*`` and
    ``reserve_through``, or a recorder method that mints
    (:data:`RECORDER_MINT_METHODS`, derived from ``recorder.py``; empty
    since S6);
``helper``
    reference a tag-minting helper of ``_internal/build.py`` or
    ``_internal/tag_plan.py`` (:data:`MINTING_HELPERS`,
    :data:`TAG_PLAN_MINTING_HELPERS`, each derived from its module and
    locked, so a new minting helper must join its list);
``counters``
    touch ``TagAllocator._counters``;
``max_plus_one``
    compute a tag as ``max(...) + 1``;
``construct`` (hub lock)
    construct a ``TagAllocator``;
``allocator_param`` (hub lock)
    take a parameter annotated ``TagAllocator``.

The build's minting helpers are its planners (:data:`PLANNERS`, which
``tag_plan.plan_tags`` runs) and exactly three standalone replay entry
points (:data:`REPLAY_ENTRIES`), one per replay the deck archive cannot
feed yet: the step-8b reinforce ties, and the global and staged
initial-stress and absorbing parameter tags. ``compose.py`` reaches them
under the waivers of ``tag_law_ledger.txt``. That ledger is shrink-only,
each waiver is commented at its site, and each one is pinned by
``test_tag_law_replay_pins.py``. ``src/`` constructs a ``TagAllocator``
only in the bridge, the planner and those waived replay sites
(:data:`ALLOCATOR_SITES`).

The ``allocate`` and ``helper`` rules flag every *reference* to a minting
callable, not only a direct call, so a call through an assigned alias
(``a = tags.allocate``; ``pc = plan_contacts``), an aliased import, or a
callback argument is caught at the reference.

Known gap: ``max_plus_one`` sees the expression ``max(...) + 1`` only. A
tag computed through an intermediate name (``m = max(xs); m + 1``) is not
caught. The ``counters`` rule still catches the usual way of seeding an
allocator from such a value.

The lock reads source with ``ast`` and imports nothing from apeGmsh.
"""
from __future__ import annotations

import ast
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src"
_OPENSEES = _ROOT / "src" / "apeGmsh" / "opensees"
#: The two layouts of the build: one module today, a package after S2.
_BUILD = _OPENSEES / "_internal" / "build.py"
_BUILD_PKG = _OPENSEES / "_internal" / "build"
_APESEES = _OPENSEES / "apesees.py"
_RECORDER = _OPENSEES / "recorder.py"
_ALLOCATOR = _OPENSEES / "_internal" / "tag_allocator.py"
_LEDGER = Path(__file__).with_name("tag_law_ledger.txt")
_PINS = Path(__file__).with_name("test_tag_law_replay_pins.py")

#: ``N_WAIVED`` when the ledger opened (#1361). It may only go down.
_CEILING = 5

#: ``TagAllocator`` methods, split into the ones that mint (advance a
#: counter) and the ones that only read or reset. A new method fails
#: :func:`test_allocator_methods_are_classified` until it joins one.
MINT_METHODS = frozenset({
    "allocate", "allocate_block", "allocate_for", "reserve_through",
})
NON_MINT_METHODS = frozenset({
    "__init__", "last", "tag_for", "reset",
    "freeze", "fork", "frozen", "frozen_kinds", "_refuse",
})

#: Recorder methods that mint (``recorder.py``). Since S6 a recorder reads
#: its region tags from the plan and none mints. Derived from
#: ``recorder.py`` by :func:`test_recorder_mint_methods_are_derived`.
RECORDER_MINT_METHODS: frozenset[str] = frozenset()

#: Every module-level function of ``_internal/tag_plan.py`` that mints a
#: tag. Derived by :func:`derive_minting_helpers`; this literal is the lock
#: on that list, and the locked modules may reference none of them.
TAG_PLAN_MINTING_HELPERS = frozenset({"plan_regions", "plan_tags"})

#: The build's planners: the allocation loops ``tag_plan.plan_tags`` runs
#: once per emit mode, before the emit.
PLANNERS = frozenset({
    "allocate_element_tags",
    "plan_contacts",
    "plan_interface_tags",
    "plan_mp_elements",
    "plan_parameters",
    "plan_transform_specs",
    "reserve_fem_element_tags",
})

#: The build's standalone replay entry points: the only writers that
#: number tags from an allocator of their own, for the deck replay the
#: archive cannot feed yet. Each is named by a ledger waiver.
REPLAY_ENTRIES = frozenset({
    "replay_activate_absorbing",
    "replay_initial_stress_global",
    "replay_reinforce_ties",
})

#: Every module-level function of the build module set that mints a tag,
#: directly or through another one, in any module of the set. Derived by
#: :func:`derive_minting_helpers`; this literal is the lock on that list.
MINTING_HELPERS = PLANNERS | REPLAY_ENTRIES

#: Every ``TagAllocator(...)`` construction in ``src/apeGmsh`` outside
#: ``tag_allocator.py``, by ``(module under src/apeGmsh, enclosing
#: function)``: the bridge's registration allocator, the planner's, and the
#: waived replay sites of ``compose.py`` (the reinforce ties and the global
#: initial stress in ``_replay_into``; the staged parameters).
ALLOCATOR_SITES = Counter({
    ("opensees/apesees.py", "apeSees.__init__"): 1,
    ("opensees/_internal/tag_plan.py", "plan_tags"): 1,
    ("opensees/_internal/compose.py", "_replay_into"): 2,
    ("opensees/_internal/compose.py", "_replay_staged_into"): 1,
})


def _locked_modules() -> list[Path]:
    return [
        *sorted((_OPENSEES / "emitter").rglob("*.py")),
        _OPENSEES / "_internal" / "compose.py",
        _OPENSEES / "opensees_model.py",
        # ADR 0117 INV-3: the assembly rehydrates onto the bridge and
        # never mints; its tags come from the bridge's plan.
        *sorted((_OPENSEES.parent / "assembly").rglob("*.py")),
    ]


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


# ---------------------------------------------------------------------------
# The build module set
# ---------------------------------------------------------------------------


def build_modules(file: Path = _BUILD, pkg: Path = _BUILD_PKG) -> list[Path]:
    """The build's modules, by whichever layout exists: ``[build.py]``
    today, ``build/**/*.py`` once S2 splits it. Both at once, or neither,
    is a broken tree and fails here rather than locking half of it. The
    package is its ``__init__.py``: a ``build/`` folder holding only a
    ``__pycache__`` left by a checkout from after the split is not one."""
    is_file, is_pkg = file.is_file(), (pkg / "__init__.py").is_file()
    assert is_file != is_pkg, (
        f"the build is one module or one package, not {is_file=} {is_pkg=}")
    return [file] if is_file else sorted(pkg.rglob("*.py"))


def module_name(path: Path, root: Path) -> str:
    """The absolute dotted name of ``path`` under source root ``root``
    (``pkg/__init__.py`` is ``pkg``)."""
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def module_set(paths: Iterable[Path], root: Path) -> dict[str, ast.Module]:
    """Parse ``paths`` into ``{dotted name: tree}`` under ``root``."""
    return {module_name(p, root): _parse(p) for p in paths}


def _build_set() -> dict[str, ast.Module]:
    return module_set(build_modules(), _SRC)


#: The build modules as the hub lock keys them: paths under ``opensees/``.
_BUILD_RELS = frozenset(
    p.relative_to(_OPENSEES).as_posix() for p in build_modules())


def _is_allocator_mint_attr(node: ast.AST) -> bool:
    """``x.allocate*`` / ``x.reserve_through``: a ``TagAllocator`` mint."""
    return isinstance(node, ast.Attribute) and (
        node.attr in MINT_METHODS or node.attr.startswith("allocate"))


def _is_mint_attr(node: ast.AST) -> bool:
    """An allocator mint, or a recorder method that mints."""
    return _is_allocator_mint_attr(node) or (
        isinstance(node, ast.Attribute)
        and node.attr in RECORDER_MINT_METHODS)


# ---------------------------------------------------------------------------
# The helper list
# ---------------------------------------------------------------------------


_Func = tuple[str, str]  # (dotted module, top-level function name)


def _package_of(module: str, is_pkg: bool) -> str:
    """The package a module's relative imports resolve from. A package
    (an ``__init__``) is recognised by having a submodule in the set; a
    set built from ``rglob("*.py")`` contains one with every package."""
    return module if is_pkg else module.rpartition(".")[0]


def _resolve_import_from(
    node: ast.ImportFrom, package: str,
) -> str | None:
    """The absolute dotted module an ``ImportFrom`` names, or ``None`` for
    a relative import that climbs above the top package."""
    if node.level == 0:
        return node.module or ""
    base = package.split(".") if package else []
    climb = node.level - 1
    if climb > len(base):
        return None
    base = base[:len(base) - climb] if climb else base
    return ".".join([*base, node.module]) if node.module else ".".join(base)


def _import_aliases(
    nodes: Iterable[ast.AST], package: str, modules: Mapping[str, ast.Module],
    where: str,
) -> dict[str, _Func]:
    """``{local name: (module, name)}`` for every name imported from a
    module of the set, over the ``ImportFrom`` statements in ``nodes``.

    Fails on the import forms the derivation cannot follow when they
    reach into the set: a star import, or a module of the set bound as an
    object (``from . import x``, ``import ...build.x``), whose attribute
    calls would hide a minting helper.
    """
    out: dict[str, _Func] = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            hit = [a.name for a in node.names if a.name in modules]
            assert not hit, (
                f"{where}: imports build module(s) {hit} as objects; the "
                "tag-law lock follows only `from .x import name`")
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        source = _resolve_import_from(node, package)
        if source is None:
            continue
        for alias in node.names:
            if source in modules:
                assert alias.name != "*", (
                    f"{where}: star-imports build module {source!r}; the "
                    "tag-law lock follows only `from .x import name`")
                assert f"{source}.{alias.name}" not in modules, (
                    f"{where}: binds build module "
                    f"{source}.{alias.name!r} as an object; the tag-law "
                    "lock follows only `from .x import name`")
                out[alias.asname or alias.name] = (source, alias.name)
            elif f"{source}.{alias.name}" in modules:
                raise AssertionError(
                    f"{where}: binds build module "
                    f"{source}.{alias.name!r} as an object; the tag-law "
                    "lock follows only `from .x import name`")
    return out


def derive_minting_helpers(
    modules: Mapping[str, ast.Module] | ast.Module,
) -> frozenset[str]:
    """The module-level functions of a module set that mint, directly or
    through each other, across modules.

    ``modules`` maps absolute dotted names to trees (:func:`module_set`);
    a bare tree is a set of one. A call is followed when its callee is a
    top-level function of the same module, or a name imported from a
    module of the set at module level or inside the caller (an alias or a
    re-export through another module of the set included). Two modules
    defining one function name is refused: the scanner keys helpers by
    bare name.
    """
    if isinstance(modules, ast.Module):
        modules = {"<module>": modules}
    packages = {
        m: _package_of(m, any(o.startswith(f"{m}.") for o in modules))
        for m in modules
    }
    funcs: dict[_Func, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for m, tree in modules.items():
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs[(m, n.name)] = n
    by_name = Counter(name for _m, name in funcs)
    dupes = sorted(n for n, k in by_name.items() if k > 1)
    assert not dupes, (
        f"function name(s) defined in two build modules: {dupes}; the "
        "tag-law lock keys helpers by bare name")

    # Every node outside any function body (methods included): the
    # module-level statements, and class bodies.
    def outside_functions(tree: ast.Module) -> list[ast.AST]:
        inside = {
            id(k) for fn in ast.walk(tree)
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
            for k in ast.walk(fn) if k is not fn}
        return [n for n in ast.walk(tree) if id(n) not in inside]

    top_level = {m: outside_functions(t) for m, t in modules.items()}
    module_aliases = {
        m: _import_aliases(top_level[m], packages[m], modules, m)
        for m in modules
    }

    def target(ref: _Func, seen: frozenset[_Func] = frozenset()) -> _Func | None:
        """The function ``ref`` names, through re-exports."""
        if ref in funcs:
            return ref
        m, name = ref
        if ref in seen or m not in modules or name not in module_aliases[m]:
            return None
        return target(module_aliases[m][name], seen | {ref})

    # A module-level alias (``_f = plan_contacts``) or table
    # (``{"c": plan_contacts}``) would hide a mint from the call-following
    # below and from the scanner's bare-name rule: refuse it.
    for m in modules:
        for n in top_level[m]:
            if not (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)):
                continue
            hit = target(module_aliases[m].get(n.id, (m, n.id)))
            assert hit is None, (
                f"{m}:{n.lineno}: module-level reference to build function "
                f"{hit[1]!r} (an alias or a dispatch table); the tag-law lock "
                "follows calls only, so call it by name inside a function")

    minting: set[_Func] = set()
    calls: dict[_Func, set[_Func]] = {}
    for key, fn in funcs.items():
        m, _name = key
        aliases = {
            **module_aliases[m],
            **_import_aliases(ast.walk(fn), packages[m], modules, m),
        }
        calls[key] = set()
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            if _is_mint_attr(node.func):
                minting.add(key)
            if isinstance(node.func, ast.Name):
                ref = aliases.get(node.func.id, (m, node.func.id))
                callee = target(ref)
                if callee is not None:
                    calls[key].add(callee)
    grew = True
    while grew:
        grew = False
        for key, callees in calls.items():
            if key not in minting and callees & minting:
                minting.add(key)
                grew = True
    return frozenset(name for _m, name in minting)


# ---------------------------------------------------------------------------
# The scanner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    function: str  # enclosing qualname, "<module>" at top level
    rule: str
    symbol: str
    line: int

    def key(self) -> tuple[str, str, str]:
        return self.function, self.rule, self.symbol


def _is_max_call(node: ast.expr) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "max")


def _is_one(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value == 1


def _names_allocator(node: ast.expr | None) -> bool:
    """An annotation that names ``TagAllocator`` (bare, dotted or quoted)."""
    if node is None:
        return False
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return "TagAllocator" in node.value
    return any(
        (isinstance(n, ast.Name) and n.id == "TagAllocator")
        or (isinstance(n, ast.Attribute) and n.attr == "TagAllocator")
        for n in ast.walk(node))


class _Scanner(ast.NodeVisitor):
    def __init__(self, helpers: frozenset[str], *, hub: bool = False) -> None:
        self.helpers = helpers
        self.hub = hub
        self.aliases: dict[str, str] = {}
        self.stack: list[str] = []
        self.found: list[Violation] = []

    def _add(self, rule: str, symbol: str, node: ast.AST) -> None:
        where = ".".join(self.stack) or "<module>"
        self.found.append(Violation(where, rule, symbol, node.lineno))  # type: ignore[attr-defined]

    def _scoped(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = _scoped
    visit_AsyncFunctionDef = _scoped
    visit_ClassDef = _scoped

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name in self.helpers:
                self.aliases[alias.asname or alias.name] = alias.name
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            if node.id in self.aliases:
                self._add("helper", self.aliases[node.id], node)
            elif node.id in self.helpers:
                self._add("helper", node.id, node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == "_counters":
            self._add("counters", "_counters", node)
        elif _is_mint_attr(node):
            self._add("allocate", node.attr, node)
        elif node.attr in self.helpers:
            self._add("helper", node.attr, node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        if self.hub and (
            (isinstance(f, ast.Name) and f.id == "TagAllocator")
            or (isinstance(f, ast.Attribute) and f.attr == "TagAllocator")
        ):
            self._add("construct", "TagAllocator", node)
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        if self.hub and _names_allocator(node.annotation):
            self._add("allocator_param", node.arg, node)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Add) and (
            (_is_max_call(node.left) and _is_one(node.right))
            or (_is_max_call(node.right) and _is_one(node.left))
        ):
            self._add("max_plus_one", "max", node)
        self.generic_visit(node)


def scan(
    tree: ast.Module, helpers: frozenset[str] = MINTING_HELPERS,
    *, hub: bool = False,
) -> list[Violation]:
    scanner = _Scanner(helpers, hub=hub)
    scanner.visit(tree)
    return scanner.found


# ---------------------------------------------------------------------------
# The hub lock: the emit side of apesees.py, the build modules and recorder.py
# ---------------------------------------------------------------------------


def _in_hub_scope(
    module: str, function: str, module_funcs: frozenset[str] = frozenset(),
    *, build: frozenset[str] = _BUILD_RELS,
) -> bool:
    """Whether the hub lock covers ``function`` (a scanner qualname) of
    ``module``: every emit-side function, which must mint nothing.

    ``apesees.py``: every ``BuiltModel`` method except ``_tag_plan`` (it
    runs the planner) and every module-level function (``module_funcs``);
    the ``apeSees`` bridge registers primitive tags and is out of scope.
    A module of the build set (``build``, paths under ``opensees/``):
    every function but the planners and the replay entry points.
    ``recorder.py``: everything.
    """
    if function == "<module>":
        return False
    top, *rest = function.split(".")
    if module == "apesees.py":
        if top == "BuiltModel":
            return bool(rest) and rest[0] != "_tag_plan"
        return top in module_funcs
    if module in build:
        return top not in MINTING_HELPERS
    return module == "recorder.py"


def hub_violations(
    sources: dict[Path, ast.Module] | None = None,
) -> list[str]:
    """Every rule broken in the hub lock's scope, as readable strings."""
    out: list[str] = []
    for path in (_APESEES, *build_modules(), _RECORDER):
        tree = (sources or {}).get(path) or _parse(path)
        rel = path.relative_to(_OPENSEES).as_posix()
        funcs = frozenset(
            n.name for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        for v in scan(tree, MINTING_HELPERS | TAG_PLAN_MINTING_HELPERS,
                      hub=True):
            if v.rule != "max_plus_one" and _in_hub_scope(
                    rel, v.function, funcs):
                out.append(f"{rel}:{v.line} {v.function} {v.rule} {v.symbol}")
    return out


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Waiver:
    name: str
    module: str
    function: str
    rule: str
    symbol: str


def _read_ledger() -> tuple[int, list[Waiver]]:
    n_waived: int | None = None
    rows: list[Waiver] = []
    for raw in _LEDGER.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"N_WAIVED\s*=\s*(\d+)", line)
        if m:
            n_waived = int(m.group(1))
            continue
        row, _hash, retired_by = line.partition("#")
        assert "retired by" in retired_by, (
            f"ledger row names no retiring slice: {raw!r}")
        parts = row.split()
        assert len(parts) == 5, f"malformed ledger row: {raw!r}"
        rows.append(Waiver(*parts))
    assert n_waived is not None, "tag_law_ledger.txt lacks N_WAIVED"
    return n_waived, rows


def _violations_by_module(
    sources: dict[Path, ast.Module] | None = None,
) -> dict[str, list[Violation]]:
    out: dict[str, list[Violation]] = {}
    for path in _locked_modules():
        tree = (sources or {}).get(path) or _parse(path)
        rel = (path.relative_to(_OPENSEES) if path.is_relative_to(_OPENSEES)
               else Path("..") / path.relative_to(_OPENSEES.parent)).as_posix()
        out[rel] = scan(tree, MINTING_HELPERS | TAG_PLAN_MINTING_HELPERS)
    return out


def _unwaived(found: dict[str, list[Violation]], waivers: list[Waiver]) -> tuple[
        list[str], list[str]]:
    """``(violations no waiver covers, waivers no violation matches)``."""
    have = Counter(
        (mod, *v.key()) for mod, vs in found.items() for v in vs)
    want = Counter(
        (w.module, w.function, w.rule, w.symbol) for w in waivers)
    extra = []
    for key in sorted((have - want).elements()):
        lines = [
            v.line for v in found[key[0]] if (key[0], *v.key()) == key]
        extra.append(f"{key} at line(s) {lines}")
    stale = [str(k) for k in sorted((want - have).elements())]
    return extra, stale


# ---------------------------------------------------------------------------
# The locks
# ---------------------------------------------------------------------------


def test_allocator_methods_are_classified() -> None:
    tree = _parse(_ALLOCATOR)
    (cls,) = [
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "TagAllocator"]
    methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert methods == MINT_METHODS | NON_MINT_METHODS, (
        "classify the new TagAllocator method(s) as minting or not: "
        f"{sorted(methods ^ (MINT_METHODS | NON_MINT_METHODS))}"
    )


def test_recorder_mint_methods_are_derived() -> None:
    """``recorder.py``'s minting methods are exactly RECORDER_MINT_METHODS.

    A method mints if it calls an allocator mint, calls a tag-plan minting
    helper (``plan_regions``), or calls a minting method of its own class
    hierarchy (``self.m(...)``, ``FilterableRecorder.m(self, ...)``).
    """
    tree = _parse(_OPENSEES / "recorder.py")
    methods = [
        fn for cls in tree.body if isinstance(cls, ast.ClassDef)
        for fn in cls.body if isinstance(fn, ast.FunctionDef)
    ]

    def calls(fn: ast.FunctionDef) -> list[ast.expr]:
        return [n.func for n in ast.walk(fn) if isinstance(n, ast.Call)]

    minting = {
        fn.name for fn in methods
        if any(
            _is_allocator_mint_attr(f) or (
                isinstance(f, ast.Name) and f.id in TAG_PLAN_MINTING_HELPERS)
            for f in calls(fn))
    }
    grew = True
    while grew:
        grew = False
        for fn in methods:
            if fn.name not in minting and any(
                    isinstance(f, ast.Attribute) and f.attr in minting
                    for f in calls(fn)):
                minting.add(fn.name)
                grew = True
    assert minting == RECORDER_MINT_METHODS, (
        "recorder.py's minting methods changed; update RECORDER_MINT_METHODS "
        f"(new: {sorted(minting - RECORDER_MINT_METHODS)}, "
        f"gone: {sorted(RECORDER_MINT_METHODS - minting)})")


def test_tag_plan_minting_helper_list_is_derived() -> None:
    derived = derive_minting_helpers(
        _parse(_OPENSEES / "_internal" / "tag_plan.py"))
    assert derived == TAG_PLAN_MINTING_HELPERS, (
        "tag_plan.py's tag-minting helpers changed; update "
        f"TAG_PLAN_MINTING_HELPERS (new: "
        f"{sorted(derived - TAG_PLAN_MINTING_HELPERS)}, gone: "
        f"{sorted(TAG_PLAN_MINTING_HELPERS - derived)})"
    )


def test_minting_helper_list_is_derived_from_build() -> None:
    derived = derive_minting_helpers(_build_set())
    assert derived == MINTING_HELPERS, (
        "the build's tag-minting helpers changed; update MINTING_HELPERS "
        f"(new: {sorted(derived - MINTING_HELPERS)}, "
        f"gone: {sorted(MINTING_HELPERS - derived)})"
    )


def test_locked_modules_mint_only_under_a_ledger_waiver() -> None:
    _n, waivers = _read_ledger()
    extra, stale = _unwaived(_violations_by_module(), waivers)
    assert not extra, (
        "tag minting in a module that replays or serialises a model "
        "(K1-3 tag law): " + "; ".join(extra)
    )
    assert not stale, (
        "ledger rows that no longer match a site; remove them and lower "
        "N_WAIVED: " + "; ".join(stale)
    )


def test_ledger_only_shrinks() -> None:
    n_waived, waivers = _read_ledger()
    assert n_waived == len(waivers)
    assert n_waived <= _CEILING, "N_WAIVED may only go down"


def test_each_waiver_is_named_at_its_site() -> None:
    _n, waivers = _read_ledger()
    for name in sorted({w.name for w in waivers}):
        rows = [w for w in waivers if w.name == name]
        for module in {w.module for w in rows}:
            path = _OPENSEES / module
            text = path.read_text(encoding="utf-8")
            assert text.count(f"tag-law waiver {name} ") == 1, (
                f"{module} must name waiver {name!r} exactly once")
            tree = _parse(path)
            marker = next(
                i for i, ln in enumerate(text.splitlines(), 1)
                if f"tag-law waiver {name} " in ln)
            for w in rows:
                fn = next(
                    n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef)
                    and n.name == w.function.split(".")[-1])
                assert fn.lineno <= marker <= (fn.end_lineno or 0), (
                    f"waiver {name!r} comment is not inside {w.function}")


def test_each_waiver_is_pinned_by_a_test() -> None:
    _n, waivers = _read_ledger()
    pins = {
        n.name for n in _parse(_PINS).body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
    }
    for name in {w.name for w in waivers}:
        slug = name.replace("-", "_")
        assert any(slug in p for p in pins), (
            f"waiver {name!r} has no pin test in {_PINS.name}")


# ---------------------------------------------------------------------------
# Planted violations: the scanner catches each kind
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("snippet", "rule", "symbol"), [
    ("def f(tags):\n    return tags.allocate('element')\n",
     "allocate", "allocate"),
    ("def f(self):\n    return self._tags.allocate_block('element', 3)\n",
     "allocate", "allocate_block"),
    ("def f(tags):\n    tags.reserve_through('element', 9)\n",
     "allocate", "reserve_through"),
    ("def f(fem, es, tags):\n    plan_contacts(fem, es, tags)\n",
     "helper", "plan_contacts"),
    ("def f(fem, es, tags):\n    build.plan_contacts(fem, es, tags)\n",
     "helper", "plan_contacts"),
    ("from .build import plan_contacts as pc\n"
     "def f(fem, es, tags):\n    pc(fem, es, tags)\n",
     "helper", "plan_contacts"),
    ("def f(e, fem, tags):\n"
     "    from .build import replay_reinforce_ties\n"
     "    replay_reinforce_ties(e, fem, tags, name_to_tag={})\n",
     "helper", "replay_reinforce_ties"),
    ("def f(fem, es, tags):\n    pc = plan_contacts\n    pc(fem, es, tags)\n",
     "helper", "plan_contacts"),
    ("def f(tags):\n    a = tags.allocate\n    return a('element')\n",
     "allocate", "allocate"),
    ("def f(run, tags):\n    run(plan_contacts, tags)\n",
     "helper", "plan_contacts"),
    ("def f(t):\n    t._counters['element'] = 4\n",
     "counters", "_counters"),
    ("def f(t):\n    t._counters.update({'element': 4})\n",
     "counters", "_counters"),
    ("def f(xs):\n    return max(xs) + 1\n", "max_plus_one", "max"),
    ("def f(xs):\n    return 1 + max(x.tag for x in xs)\n",
     "max_plus_one", "max"),
])
def test_planted_violation_is_caught(snippet: str, rule: str, symbol: str) -> None:
    found = scan(ast.parse(snippet))
    assert [(v.function, v.rule, v.symbol) for v in found] == [
        ("f", rule, symbol)]


def test_clean_code_is_not_flagged() -> None:
    snippet = (
        "from .build import emit_initial_stress_addtoparameter\n"
        "def f(tags, xs, p, e):\n"
        "    n = max(xs)\n"
        "    t = tags.tag_for(p)\n"
        "    emit_initial_stress_addtoparameter(e)\n"
        "    return n + 2, t, tags.last('element')\n"
    )
    assert scan(ast.parse(snippet)) == []


@pytest.mark.parametrize("body", [
    "    from .build import plan_contacts\n"
    "    plan_contacts(fem, [], tags)\n",
    "    from .build import replay_activate_absorbing\n"
    "    replay_activate_absorbing([], emitter, fem, {}, tags)\n",
    "    from .tag_plan import plan_regions\n"
    "    plan_regions([], tags)\n",
])
def test_planted_violation_in_a_locked_module_fails_the_lock(body: str) -> None:
    """An unwaived minting reference appended to ``compose.py`` is reported."""
    path = _OPENSEES / "_internal" / "compose.py"
    planted = path.read_text(encoding="utf-8") + (
        "\n\ndef _planted(emitter, fem, tags, spec):\n" + body)
    _n, waivers = _read_ledger()
    extra, stale = _unwaived(
        _violations_by_module({path: ast.parse(planted)}), waivers)
    assert stale == []
    assert len(extra) == 1 and "_planted" in extra[0], extra


def test_planted_minting_helper_fails_the_list_lock() -> None:
    """A new build function that mints must join MINTING_HELPERS."""
    first = build_modules()[0]
    planted = first.read_text(encoding="utf-8") + (
        "\n\ndef emit_planted(emitter, tags):\n"
        "    return tags.allocate('element')\n"
        "\n\ndef emit_planted_caller(emitter, tags):\n"
        "    return emit_planted(emitter, tags)\n"
    )
    trees = _build_set()
    trees[module_name(first, _SRC)] = ast.parse(planted)
    derived = derive_minting_helpers(trees)
    assert derived - MINTING_HELPERS == {"emit_planted", "emit_planted_caller"}


# ---------------------------------------------------------------------------
# The hub lock (K1-3d S6)
# ---------------------------------------------------------------------------


def test_build_minting_helpers_are_planners_and_three_replay_entries() -> None:
    """The replay entries are exactly the ledger's ``helper`` symbols."""
    _n, waivers = _read_ledger()
    assert MINTING_HELPERS - PLANNERS == REPLAY_ENTRIES
    assert len(REPLAY_ENTRIES) == 3
    assert {w.symbol for w in waivers if w.rule == "helper"} == REPLAY_ENTRIES


def test_hub_emit_side_mints_nothing() -> None:
    """No emit-side function of the two hubs or of ``recorder.py`` mints,
    constructs an allocator, or takes one: the emit holds a ``TagPlan``."""
    found = hub_violations()
    assert not found, (
        "the emit side of apesees.py / build.py / recorder.py mints or "
        "holds an allocator (K1-3d S6 tag law): " + "; ".join(found))


def test_replay_entries_are_referenced_only_under_a_waiver() -> None:
    """Outside the build modules, only the ledger's waived functions of
    ``compose.py`` name a replay entry point, anywhere in ``src/apeGmsh``."""
    _n, waivers = _read_ledger()
    allowed = {(w.module, w.function) for w in waivers if w.rule == "helper"}
    seen: set[tuple[str, str]] = set()
    build = set(build_modules())
    for path in sorted(_OPENSEES.parent.rglob("*.py")):
        if path in build:
            continue
        for v in scan(_parse(path), REPLAY_ENTRIES):
            if v.rule != "helper":
                continue
            rel = path.relative_to(_OPENSEES).as_posix() if (
                _OPENSEES in path.parents) else path.as_posix()
            seen.add((rel, v.function))
    assert seen == allowed, (seen ^ allowed)


def test_allocator_constructions_are_the_known_sites() -> None:
    """``src/apeGmsh`` constructs a ``TagAllocator`` only in the bridge,
    the planner and the waived replay sites."""
    root = _OPENSEES.parent
    found: Counter[tuple[str, str]] = Counter()
    for path in sorted(root.rglob("*.py")):
        if path == _ALLOCATOR:
            continue
        for v in scan(_parse(path), frozenset(), hub=True):
            if v.rule == "construct":
                found[(path.relative_to(root).as_posix(), v.function)] += 1
    assert found == ALLOCATOR_SITES, (
        f"TagAllocator constructions changed: {dict(found)}")


#: One module of the build set, for the planted hub cases below.
_A_BUILD_REL = min(_BUILD_RELS)


@pytest.mark.parametrize(("module", "snippet"), [
    (_A_BUILD_REL,
     "\n\ndef emit_planted(emitter, fem, tag_plan):\n"
     "    from .tag_allocator import TagAllocator\n"
     "    return TagAllocator()\n"),
    (_A_BUILD_REL,
     "\n\ndef emit_planted(emitter, fem, tags: TagAllocator):\n"
     "    return None\n"),
    (_A_BUILD_REL,
     "\n\ndef emit_planted(emitter, entries, tag_plan):\n"
     "    return plan_mp_elements(entries, tag_plan)\n"),
    ("apesees.py",
     "\n\nclass BuiltModel:\n"
     "    def _emit_planted(self, tag_plan):\n"
     "        return tag_plan.allocator.allocate('element')\n"),
    ("recorder.py",
     "\n\ndef planted(fem, tags: 'TagAllocator | None'):\n"
     "    return None\n"),
])
def test_planted_mint_on_the_emit_side_fails_the_hub_lock(
        module: str, snippet: str) -> None:
    path = _OPENSEES / module
    planted = ast.parse(path.read_text(encoding="utf-8") + snippet)
    found = hub_violations({path: planted})
    assert found and all("planted" in f for f in found), found


def test_hub_scope_spares_the_planner_and_replay_entries() -> None:
    assert not _in_hub_scope("apesees.py", "BuiltModel._tag_plan")
    assert not _in_hub_scope("apesees.py", "apeSees.__init__")
    assert _in_hub_scope("apesees.py", "BuiltModel._emit_flat._inner")
    assert _in_hub_scope(
        "apesees.py", "_planned_element_specs",
        frozenset({"_planned_element_specs"}))
    for rel in sorted(_BUILD_RELS):
        assert not _in_hub_scope(rel, "plan_mp_elements")
        assert not _in_hub_scope(rel, "replay_reinforce_ties.standalone")
        assert _in_hub_scope(rel, "emit_reinforce_ties")
    assert _in_hub_scope("recorder.py", "MPCO.materialize")
    # A module outside the set is not in the build's scope, however named.
    assert not _in_hub_scope("_internal/build_helpers.py", "emit_x")
    assert not _in_hub_scope("_internal/build.py", "emit_x", build=frozenset())


# ---------------------------------------------------------------------------
# The module set (S2-0, #1611): the lock survives the split of build.py
# ---------------------------------------------------------------------------


def _cut(text: str, tree: ast.Module, name: str) -> tuple[str, str]:
    """``(text without top-level function name, that function's source)``."""
    (fn,) = [
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == name]
    lines = text.splitlines(keepends=True)
    start = (fn.decorator_list[0].lineno if fn.decorator_list else fn.lineno) - 1
    end = fn.end_lineno or 0
    return "".join(lines[:start] + lines[end:]), "".join(lines[start:end])


def _simulated_split(tmp_path: Path, planted: str = "") -> dict[str, ast.Module]:
    """``build.py`` as a two-module package ``pkg``: ``plan_contacts`` moves to
    ``pkg/contacts.py``, the rest stays in ``pkg/core.py`` and reaches it
    through ``from .contacts import plan_contacts``; ``planted`` is appended
    to ``core.py``."""
    text = _BUILD.read_text(encoding="utf-8")
    rest, moved = _cut(text, ast.parse(text), "plan_contacts")
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "from .contacts import plan_contacts\n", encoding="utf-8")
    (pkg / "contacts.py").write_text(
        "from ..tag_allocator import TagAllocator\n\n\n" + moved,
        encoding="utf-8")
    (pkg / "core.py").write_text(
        "from .contacts import plan_contacts\n" + rest + planted,
        encoding="utf-8")
    return module_set(sorted(pkg.rglob("*.py")), tmp_path)


def test_simulated_split_derives_the_same_helpers(tmp_path: Path) -> None:
    """Moving a planner into its own module changes nothing: the helper
    set is derived over the union of the set."""
    trees = _simulated_split(tmp_path)
    assert set(trees) == {"pkg", "pkg.contacts", "pkg.core"}
    assert derive_minting_helpers(trees) == MINTING_HELPERS
    # One module alone no longer sees it: the union is what locks.
    assert "plan_contacts" not in derive_minting_helpers(trees["pkg.core"])


@pytest.mark.parametrize("body", [
    # a module-level relative import
    "    return plan_contacts(fem, entries, tags)\n",
    # a function-local aliased relative import
    "    from .contacts import plan_contacts as pc\n"
    "    return pc(fem, entries, tags)\n",
    # an absolute import of the moved module
    "    from pkg.contacts import plan_contacts as pc\n"
    "    return pc(fem, entries, tags)\n",
    # a re-export through the package's __init__
    "    from pkg import plan_contacts as pc\n"
    "    return pc(fem, entries, tags)\n",
    "    from . import contacts\n"
    "    return contacts.plan_contacts(fem, entries, tags)\n",
])
def test_simulated_split_catches_a_mint_through_the_moved_module(
        tmp_path: Path, body: str) -> None:
    """A non-planner that mints only by calling the moved planner is still
    a minting helper, so it fails the list lock, and its reference is in
    the hub lock's scope. A module-object import of a build module is not
    followed and fails outright instead."""
    planted = "\n\ndef emit_planted_caller(emitter, fem, entries, tags):\n" + body
    trees = _simulated_split(tmp_path, planted)
    if "from . import contacts" in body:
        with pytest.raises(AssertionError, match="as an object"):
            derive_minting_helpers(trees)
        return
    derived = derive_minting_helpers(trees)
    assert derived - MINTING_HELPERS == {"emit_planted_caller"}
    build = frozenset({"pkg/core.py", "pkg/contacts.py", "pkg/__init__.py"})
    hits = [
        v for v in scan(trees["pkg.core"], MINTING_HELPERS, hub=True)
        if v.rule != "max_plus_one"
        and _in_hub_scope("pkg/core.py", v.function, build=build)]
    assert [(v.function, v.rule, v.symbol) for v in hits] == [
        ("emit_planted_caller", "helper", "plan_contacts")]


def test_simulated_split_refuses_a_star_import(tmp_path: Path) -> None:
    trees = _simulated_split(
        tmp_path, "\n\nfrom .contacts import *  # noqa: F403\n")
    with pytest.raises(AssertionError, match="star-imports"):
        derive_minting_helpers(trees)


@pytest.mark.parametrize("planted", [
    # an alias of the imported planner, called from a non-planner
    "\n\n_pc = plan_contacts\n\n\n"
    "def emit_planted_caller(fem, entries, tags):\n"
    "    return _pc(fem, entries, tags)\n",
    # a dispatch table holding it
    "\n\n_PLANTED_TABLE = {'contacts': plan_contacts}\n",
    # an alias of a same-module function
    "\n\n_planted = emit_contacts\n",
])
def test_simulated_split_refuses_a_module_level_function_reference(
        tmp_path: Path, planted: str) -> None:
    trees = _simulated_split(tmp_path, planted)
    with pytest.raises(AssertionError, match="module-level reference"):
        derive_minting_helpers(trees)


def test_build_layout_ignores_a_pycache_only_package_folder(
        tmp_path: Path) -> None:
    """A ``build/`` folder holding only ``__pycache__`` (left by a checkout
    from after the split) is not the package layout."""
    file = tmp_path / "build.py"
    file.write_text("", encoding="utf-8")
    (tmp_path / "build" / "__pycache__").mkdir(parents=True)
    (tmp_path / "build" / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"")
    assert build_modules(file, tmp_path / "build") == [file]
    (tmp_path / "build" / "__init__.py").write_text("", encoding="utf-8")
    with pytest.raises(AssertionError, match="one module or one package"):
        build_modules(file, tmp_path / "build")
    file.unlink()
    assert build_modules(file, tmp_path / "build") == [
        tmp_path / "build" / "__init__.py"]


def test_simulated_split_refuses_a_duplicate_function_name(
        tmp_path: Path) -> None:
    trees = _simulated_split(
        tmp_path, "\n\ndef plan_contacts(fem, entries, tags):\n    return 0\n")
    with pytest.raises(AssertionError, match="two build modules"):
        derive_minting_helpers(trees)
