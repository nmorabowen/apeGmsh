"""Structural invariant: every declaration goes through ``_declare``.

``_DeclarationsMixin._declare`` stores a def, routes it into a
chain-phase broker and bumps the session's FEMData cache counter as one
step (``src/apeGmsh/core/_declarations.py``).  The behavioural contract
is tested in ``test_fem_cache_invalidation.py``: a def declared after
the first ``get_fem_data()`` must appear in the next one.  That file
checks the verbs that exist *today*; this one checks the verbs added
*tomorrow*, by sweeping the composites for code that records a def
without going through the helper.

Why it exists: the bump used to be a line written beside each
``append``.  Five verbs added after the cache (``contact`` /
``contact_plane`` / ``interface``, ``g.reinforce``, ``g.embed``) never
got one, so ``get_fem_data()`` returned the old snapshot without them,
and ``ConstraintsComposite.clear()`` emptied one of its five lists.
Each was invisible until something swept for it.

Method: AST with transitive closure over the methods of the classes in
one module, as in ``test_chain_phase_guard_coverage.py``.  A call
``self.m(...)`` resolves to the same class, and ``self.<attr>.m(...)``
to another swept class of the module, which is how ``g.loads.point``
reaches ``LoadsComposite._add_def``.
"""
from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import pytest

import apeGmsh

SRC = Path(apeGmsh.__file__).parent

# label -> (module relative to the package, class name)
COMPOSITES: dict[str, tuple[str, str]] = {
    "g.constraints":     ("core/ConstraintsComposite.py", "ConstraintsComposite"),
    "g.reinforce":       ("core/ReinforcementsComposite.py", "ReinforcementsComposite"),
    "g.embed":           ("core/EmbedmentsComposite.py", "EmbedmentsComposite"),
    "g.rebar":           ("core/RebarComposite.py", "RebarComposite"),
    "g.loads":           ("core/LoadsComposite.py", "LoadsComposite"),
    "g.loads.point":     ("core/LoadsComposite.py", "_PointLoads"),
    "g.loads.surface":   ("core/LoadsComposite.py", "_SurfaceLoads"),
    "g.displacements":   ("core/DisplacementsComposite.py", "DisplacementsComposite"),
    "g.masses":          ("core/MassesComposite.py", "MassesComposite"),
    "g.decoupled_nodes": ("core/DecoupledNodesComposite.py", "DecoupledNodesComposite"),
}

# The sub-namespaces hold no stores; they forward to ``g.loads``.
NAMESPACES = {"g.loads.point", "g.loads.surface"}

# The mixin's own methods.  A composite that redefines one could store
# without bumping, so none may.
MIXIN_METHODS = ("_declare", "_clear_declarations", "_store_for",
                 "_invalidate_fem")

MUTATORS = {"append", "extend", "insert", "clear", "pop", "remove",
            "sort", "reverse"}

# The verbs the sweep must recognise as declarations.  The bug was in the
# first five.  If the classification ever stops seeing them, the gate
# below would pass by not looking.
KNOWN_VERBS = {
    ("g.constraints", "contact"),
    ("g.constraints", "contact_plane"),
    ("g.constraints", "interface"),
    ("g.reinforce", "reinforce"),
    ("g.embed", "embed"),
    ("g.reinforce", "__call__"),
    ("g.embed", "__call__"),
    ("g.constraints", "bc"),
    ("g.constraints", "equal_dof"),
    ("g.constraints", "mortar"),
    ("g.rebar", "place"),
    ("g.loads", "gravity"),
    ("g.loads.point", "force"),
    ("g.loads.surface", "pressure"),
    ("g.displacements", "point"),
    ("g.masses", "volume"),
    ("g.decoupled_nodes", "add"),
}


def _composite_class(label: str) -> type:
    rel, name = COMPOSITES[label]
    module = "apeGmsh." + rel[:-3].replace("/", ".")
    return getattr(importlib.import_module(module), name)


def _registries() -> dict[str, dict[str, tuple[type, ...]]]:
    """``_DECLARATION_STORES`` per store-owning composite ({} if absent)."""
    return {
        label: getattr(_composite_class(label), "_DECLARATION_STORES", {})
        for label in COMPOSITES if label not in NAMESPACES
    }


def _store_names() -> set[str]:
    return {attr for reg in _registries().values() for attr in reg}


def _declared_type_names() -> set[str]:
    return {t.__name__ for reg in _registries().values()
            for kinds in reg.values() for t in kinds}


def _is_store(attr: str, stores: set[str]) -> bool:
    # ``*_defs`` catches a store added without registering it.
    return attr in stores or attr.endswith("_defs")


def _is_declared_type(name: str, types: set[str]) -> bool:
    return name in types or re.fullmatch(r"[A-Z]\w*Def", name) is not None


def _public(name: str) -> bool:
    # ``g.reinforce(...)`` / ``g.embed(...)`` enter through ``__call__``.
    return not name.startswith("_") or name == "__call__"


def _swept_classes() -> dict[str, list[tuple[str, str]]]:
    """Module path -> ``[(label, class name), ...]``."""
    out: dict[str, list[tuple[str, str]]] = {}
    for label, (rel, name) in COMPOSITES.items():
        out.setdefault(rel, []).append((label, name))
    return out


def _methods(rel: str, wanted: list[tuple[str, str]]) -> dict:
    """``(label, method) -> FunctionDef`` for the wanted classes of ``rel``."""
    tree = ast.parse((SRC / rel).read_text(encoding="utf-8"))
    label_of = {name: label for label, name in wanted}
    out = {}
    found = set()
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name in label_of:
            found.add(cls.name)
            for fn in cls.body:
                if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[(label_of[cls.name], fn.name)] = fn
    missing = set(label_of) - found
    assert not missing, f"{rel}: class(es) moved or renamed: {sorted(missing)}"
    return out


def _classify() -> dict[tuple[str, str], tuple[bool, bool, int, str]]:
    """``(label, method) -> (constructs, declares, lineno, rel)``.

    ``constructs``: the method (transitively) builds a declaration type —
    a registered store type or any ``*Def``.  ``declares``: it
    (transitively) calls ``_declare``.
    """
    types = _declared_type_names()
    result = {}
    for rel, wanted in _swept_classes().items():
        methods = _methods(rel, wanted)
        by_name: dict[str, list[tuple[str, str]]] = {}
        for key in methods:
            by_name.setdefault(key[1], []).append(key)

        constructs, declares, callees = {}, {}, {}
        for key, fn in methods.items():
            c = d = False
            calls: set[tuple[str, str]] = set()
            for n in ast.walk(fn):
                if not isinstance(n, ast.Call):
                    continue
                f = n.func
                if isinstance(f, ast.Name) and _is_declared_type(f.id, types):
                    c = True
                if not isinstance(f, ast.Attribute):
                    continue
                if _is_declared_type(f.attr, types):
                    c = True
                if f.attr == "_declare":
                    d = True
                v = f.value
                if isinstance(v, ast.Name) and v.id == "self":
                    if (key[0], f.attr) in methods:
                        calls.add((key[0], f.attr))
                elif (isinstance(v, ast.Attribute)
                        and isinstance(v.value, ast.Name)
                        and v.value.id == "self"):
                    calls.update(k for k in by_name.get(f.attr, ())
                                 if k[0] != key[0])
            constructs[key], declares[key], callees[key] = c, d, calls

        for _ in range(len(methods) + 2):         # fixed point
            changed = False
            for key in methods:
                for s in callees[key]:
                    if constructs[s] and not constructs[key]:
                        constructs[key] = True
                        changed = True
                    if declares[s] and not declares[key]:
                        declares[key] = True
                        changed = True
            if not changed:
                break
        for key, fn in methods.items():
            result[key] = (constructs[key], declares[key], fn.lineno, rel)
    return result


def _store_mutations():
    """Yield ``(label, method, lineno, rel, what)`` for each direct write
    to a declaration store: a mutator call, ``+=``, item assignment or
    deletion anywhere, or rebinding outside ``__init__``."""
    stores = _store_names()

    def store_of(node: ast.expr) -> str | None:
        if isinstance(node, ast.Attribute) and _is_store(node.attr, stores):
            return node.attr
        return None

    for rel, wanted in _swept_classes().items():
        for (label, name), fn in _methods(rel, wanted).items():
            for n in ast.walk(fn):
                hits: list[str] = []
                if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                        and n.func.attr in MUTATORS):
                    s = store_of(n.func.value)
                    if s:
                        hits.append(f"{s}.{n.func.attr}()")
                elif isinstance(n, ast.AugAssign):
                    s = store_of(n.target)
                    if s:
                        hits.append(f"{s} {type(n.op).__name__}=")
                elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.Delete)):
                    targets = (n.targets if isinstance(n, (ast.Assign, ast.Delete))
                               else [n.target])
                    for t in targets:
                        for el in (t.elts if isinstance(t, ast.Tuple) else [t]):
                            if isinstance(el, ast.Subscript):
                                s = store_of(el.value)
                                if s:
                                    hits.append(f"{s}[...] write")
                            elif name != "__init__" or isinstance(n, ast.Delete):
                                s = store_of(el)
                                if s:
                                    hits.append(f"{s} rebound")
                for what in hits:
                    yield label, name, n.lineno, rel, what


# =====================================================================
# The gates
# =====================================================================

def test_no_direct_store_mutation() -> None:
    """Only the mixin writes a declaration store.

    A direct ``append`` is exactly the bug: the def is stored, nothing
    bumps the cache, and ``get_fem_data()`` keeps returning the snapshot
    taken before the def existed.  A direct ``clear()`` repeats it for
    removal, and empties only the lists its author remembered.
    """
    bad = sorted(set(_store_mutations()))
    if bad:
        lines = [f"  {label}.{name}  ({rel}:{line})  {what}"
                 for label, name, line, rel, what in bad]
        pytest.fail(
            f"{len(bad)} direct write(s) to a declaration store:\n"
            + "\n".join(lines)
            + "\n\nStore a def with `return self._declare(defn)`, and empty "
              "the stores with `self._clear_declarations()`. Both bump the "
              "FEMData cache. A new store goes in the class's "
              "_DECLARATION_STORES."
        )


def test_every_public_declaration_verb_goes_through_declare() -> None:
    """Every public method that builds a def reaches ``_declare``.

    Catches a verb that builds its def and then keeps it somewhere the
    mixin does not know about, or drops it.
    """
    bad = [
        (label, name, line, rel)
        for (label, name), (c, d, line, rel) in _classify().items()
        if _public(name) and c and not d
    ]
    if bad:
        lines = [f"  {label}.{name}  ({rel}:{line})"
                 for label, name, line, rel in sorted(bad)]
        pytest.fail(
            f"{len(bad)} declaration verb(s) never reach _declare:\n"
            + "\n".join(lines)
            + "\n\nEnd the verb with `return self._declare(defn)` (or route "
              "it through the composite's _add_def, which does)."
        )


def test_composites_share_the_mixin() -> None:
    """Every store-owning composite inherits ``_DeclarationsMixin``,
    registers at least one store, and redefines none of its methods."""
    from apeGmsh.core._declarations import _DeclarationsMixin

    problems = []
    for label in COMPOSITES:
        if label in NAMESPACES:
            continue
        cls = _composite_class(label)
        if not issubclass(cls, _DeclarationsMixin):
            problems.append(f"{label}: {cls.__name__} does not inherit "
                            f"_DeclarationsMixin")
            continue
        if not cls._DECLARATION_STORES:
            problems.append(f"{label}: empty _DECLARATION_STORES")
        overridden = [m for m in MIXIN_METHODS if m in vars(cls)]
        if overridden:
            problems.append(f"{label}: redefines {overridden}")
    assert not problems, "\n".join(problems)


def test_registered_stores_match_the_instance() -> None:
    """Each registered store is a list the constructor creates, and every
    ``*_defs`` list the constructor creates is registered.

    An unregistered store is invisible to ``_clear_declarations``, and a
    misspelt registration fails only at the first declaration of its
    kind.
    """
    problems = []
    for label, registry in _registries().items():
        inst = _composite_class(label)(None)
        for attr in registry:
            if not isinstance(getattr(inst, attr, None), list):
                problems.append(f"{label}: registered store {attr!r} is "
                                f"not a list on a new instance")
        unregistered = sorted(
            a for a, v in vars(inst).items()
            if a.endswith("_defs") and isinstance(v, list)
            and a not in registry)
        if unregistered:
            problems.append(f"{label}: unregistered store(s) {unregistered}")
    assert not problems, "\n".join(problems)


def test_undeclared_kind_is_refused() -> None:
    """``_declare`` refuses a def type no store accepts, instead of
    appending it to some other kind's list."""
    from apeGmsh.core.ReinforcementsComposite import ReinforcementsComposite
    from apeGmsh._kernel.defs.constraints import EmbedDef

    comp = ReinforcementsComposite(None)
    defn = EmbedDef(master_label="host", slave_label="nodes")
    with pytest.raises(TypeError, match="no declaration store for EmbedDef"):
        comp._declare(defn)
    assert comp.reinforce_defs == []


def test_sweep_actually_covers_the_verbs() -> None:
    """Guard against a silently empty sweep.

    If the AST walk broke, the gates above would pass by not looking.
    Pin the verbs that must classify as declarations, and a floor.
    """
    verbs = {key for key, (c, _d, _l, _r) in _classify().items()
             if _public(key[1]) and c}
    missing = KNOWN_VERBS - verbs
    assert not missing, (
        f"sweep no longer classifies these as declaration verbs: "
        f"{sorted(missing)}"
    )
    assert len(verbs) >= 35, f"sweep saw only {len(verbs)} verbs — walk broken?"
