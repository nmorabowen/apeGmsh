"""Tag law (K1-3, #1361): the emitters and the replay never mint a tag.

A tag is an archive fact; replay never allocates (K0 design, #1341). The
build-time tag plan that makes this true for ``BuiltModel.emit`` is the
design slice #1445. This lock holds the other side now. No module under
``emitter/``, and neither ``_internal/compose.py`` nor
``opensees_model.py``, may:

``allocate``
    reference a minting method: ``TagAllocator``'s ``allocate*`` and
    ``reserve_through``, or a recorder's ``materialize``, which allocates
    the recorder's region tags (:data:`RECORDER_MINT_METHODS`);
``helper``
    reference a tag-minting helper of ``_internal/build.py``
    (:data:`MINTING_HELPERS`, derived from ``build.py`` and locked, so a
    new minting helper must join the list);
``counters``
    touch ``TagAllocator._counters``;
``max_plus_one``
    compute a tag as ``max(...) + 1``.

Today's replay minting in ``compose.py`` (the step-8b reinforce ties and
the initial-stress and absorbing parameter tags) is waived by name in
``tag_law_ledger.txt``. That ledger is shrink-only, each waiver is commented
at its site, and each one is pinned by ``test_tag_law_replay_pins.py``.

The ``allocate`` and ``helper`` rules flag every *reference* to a minting
callable, not only a direct call, so a call through an assigned alias
(``a = tags.allocate``; ``ec = emit_contacts``), an aliased import, or a
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
from dataclasses import dataclass
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_OPENSEES = _ROOT / "src" / "apeGmsh" / "opensees"
_BUILD = _OPENSEES / "_internal" / "build.py"
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

#: Recorder methods that mint (``recorder.py``: ``materialize`` allocates
#: the filter/energy region tags). Checked against ``recorder.py`` by
#: :func:`test_recorder_mint_methods_are_derived`.
RECORDER_MINT_METHODS = frozenset({"materialize"})

#: Every module-level function of ``build.py`` that mints a tag, directly or
#: through another one. Derived by :func:`derive_minting_helpers`; this
#: literal is the lock on that list.
MINTING_HELPERS = frozenset({
    "_emit_kinematic_couplings",
    "_emit_one_interpolation",
    "_emit_rigid_body_elements",
    "_emit_surface_couplings",
    "_emit_surface_couplings_for_rank",
    "allocate_element_tags",
    "allocate_interface_tags",
    "emit_activate_absorbing",
    "emit_contact_planes",
    "emit_contacts",
    "emit_element_spec",
    "emit_embed_ties",
    "emit_initial_stress_global",
    "emit_interfaces",
    "emit_mp_constraints",
    "emit_mp_constraints_partitioned",
    "emit_rebar_elements",
    "emit_recorder_spec",
    "emit_reinforce_ties",
    "emit_stage_interfaces",
    "emit_stage_mp_constraints",
    "emit_stage_mp_constraints_partitioned",
    "emit_transform_specs",
    "emit_update_parameters",
    "plan_transform_specs",
    "reserve_fem_element_tags",
})


def _locked_modules() -> list[Path]:
    return [
        *sorted((_OPENSEES / "emitter").rglob("*.py")),
        _OPENSEES / "_internal" / "compose.py",
        _OPENSEES / "opensees_model.py",
    ]


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _is_allocator_mint_attr(node: ast.AST) -> bool:
    """``x.allocate*`` / ``x.reserve_through``: a ``TagAllocator`` mint."""
    return isinstance(node, ast.Attribute) and (
        node.attr in MINT_METHODS or node.attr.startswith("allocate"))


def _is_mint_attr(node: ast.AST) -> bool:
    """An allocator mint, or a recorder's ``x.materialize``."""
    return _is_allocator_mint_attr(node) or (
        isinstance(node, ast.Attribute)
        and node.attr in RECORDER_MINT_METHODS)


# ---------------------------------------------------------------------------
# The helper list
# ---------------------------------------------------------------------------


def derive_minting_helpers(tree: ast.Module) -> frozenset[str]:
    """``build.py`` functions that mint, directly or through each other."""
    funcs = {
        n.name: n for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    minting: set[str] = set()
    calls: dict[str, set[str]] = {}
    for name, fn in funcs.items():
        calls[name] = set()
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            if _is_mint_attr(node.func):
                minting.add(name)
            if isinstance(node.func, ast.Name) and node.func.id in funcs:
                calls[name].add(node.func.id)
    grew = True
    while grew:
        grew = False
        for name, callees in calls.items():
            if name not in minting and callees & minting:
                minting.add(name)
                grew = True
    return frozenset(minting)


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


class _Scanner(ast.NodeVisitor):
    def __init__(self, helpers: frozenset[str]) -> None:
        self.helpers = helpers
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

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Add) and (
            (_is_max_call(node.left) and _is_one(node.right))
            or (_is_max_call(node.right) and _is_one(node.left))
        ):
            self._add("max_plus_one", "max", node)
        self.generic_visit(node)


def scan(tree: ast.Module, helpers: frozenset[str] = MINTING_HELPERS) -> list[Violation]:
    scanner = _Scanner(helpers)
    scanner.visit(tree)
    return scanner.found


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
        parts = line.split()
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
        rel = path.relative_to(_OPENSEES).as_posix()
        out[rel] = scan(tree)
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
    """Every ``recorder.py`` method that allocates is in RECORDER_MINT_METHODS."""
    tree = _parse(_OPENSEES / "recorder.py")
    minting = {
        fn.name
        for cls in tree.body if isinstance(cls, ast.ClassDef)
        for fn in cls.body if isinstance(fn, ast.FunctionDef)
        if any(
            isinstance(n, ast.Call) and _is_allocator_mint_attr(n.func)
            for n in ast.walk(fn))
    }
    assert minting, "recorder.py no longer allocates; revisit the lock"
    assert minting <= RECORDER_MINT_METHODS, (
        "recorder.py methods mint but are not locked: "
        f"{sorted(minting - RECORDER_MINT_METHODS)}")


def test_minting_helper_list_is_derived_from_build() -> None:
    derived = derive_minting_helpers(_parse(_BUILD))
    assert derived == MINTING_HELPERS, (
        "build.py's tag-minting helpers changed; update MINTING_HELPERS "
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
    ("def f(e, fem, tags):\n    emit_contacts(e, fem, tags)\n",
     "helper", "emit_contacts"),
    ("def f(e, fem, tags):\n    build.emit_contacts(e, fem, tags)\n",
     "helper", "emit_contacts"),
    ("from .build import emit_contacts as ec\n"
     "def f(e, fem, tags):\n    ec(e, fem, tags)\n",
     "helper", "emit_contacts"),
    ("def f(spec, e, fem, tags):\n    spec.materialize(e, fem, tags)\n",
     "allocate", "materialize"),
    ("def f(spec, e, fem, tags):\n"
     "    from .build import emit_recorder_spec\n"
     "    emit_recorder_spec(spec, e, fem, tags)\n",
     "helper", "emit_recorder_spec"),
    ("def f(e, fem, tags):\n    ec = emit_contacts\n    ec(e, fem, tags)\n",
     "helper", "emit_contacts"),
    ("def f(tags):\n    a = tags.allocate\n    return a('element')\n",
     "allocate", "allocate"),
    ("def f(run, tags):\n    run(emit_contacts, tags)\n",
     "helper", "emit_contacts"),
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
    "    from .build import emit_contacts\n"
    "    emit_contacts(emitter, fem, tags)\n",
    # Fable review of #1447: region minting through a recorder.
    "    from .build import emit_recorder_spec\n"
    "    emit_recorder_spec(spec, emitter, fem, tags)\n",
    "    spec.materialize(emitter, fem, tags)\n",
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
    """A new ``build.py`` function that mints must join MINTING_HELPERS."""
    planted = _BUILD.read_text(encoding="utf-8") + (
        "\n\ndef emit_planted(emitter, tags):\n"
        "    return tags.allocate('element')\n"
        "\n\ndef emit_planted_caller(emitter, tags):\n"
        "    return emit_planted(emitter, tags)\n"
    )
    derived = derive_minting_helpers(ast.parse(planted))
    assert derived - MINTING_HELPERS == {"emit_planted", "emit_planted_caller"}
