"""Lock: the ``VERBS`` table matches the Emitter Protocol (ADR 0114).

Parses ``emitter/base.py`` and the five concrete emitters with ``ast``
and loads ``emitter/verbs.py`` as a standalone file, so nothing from the
apeGmsh package is imported. What it pins:

(a) ``base.py::Emitter`` has exactly ``EMITTER_METHOD_COUNT`` methods;
(b) their names are the ``via == "protocol"`` rows, and each row's
    ``returns`` is the method's return annotation;
(c) the K0-5 stage verbs are ``archive`` with scope ``stage`` or ``both``;
(d) the ``ledger`` rows are exactly the ones listed in
    ``verbs_ledger.txt``, whose ``N_LEDGER`` may only go down;
(e) each emitter defines every Protocol verb, and every other public
    name (method or class attribute) it defines is in ``SIDE_CHANNELS``;
(f) the ``h5`` column against ``H5Emitter``'s source (K1-2): a
    ``refuse`` row's method reaches ``self._refuse(`` (directly or
    through any ``self._helper`` it calls), a ``ledger`` row's method
    reaches ``self._ledger(``, no other row's method reaches either or
    raises ``NotImplementedError`` / ``H5RefusedVerb`` itself, and an
    ``archive`` row's method does more than discard its arguments;
(g) the command channel's callers (ADR 0114 D3): every ``x.command(...)``
    call under ``src/apeGmsh/opensees/`` passes a literal verb that is a
    ``via == "command"`` row, from a ``_emit`` method of a class (or from
    ``_internal/compose.py``, K2's replay). A non-literal verb, a token
    without such a row, or any other caller fails;
(h) the ``params_names`` ratchet (K1-7, ADR 0114 Q4): the sampled
    primitives whose store argv is not their dataclass fields, by the
    writer's own ``decl_argv_names``, are exactly the lines of
    ``params_names_ledger.txt``, whose ``N_LEDGER`` may only go down.

(a)-(g) read source, not behaviour: an archive body that stores the wrong
thing is K2's round-trip oracle. (h) is the one check that imports
apeGmsh, inside its own functions, since the argv exists only at emit.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_OPENSEES_DIR = _ROOT / "src" / "apeGmsh" / "opensees"
_EMITTER_DIR = _OPENSEES_DIR / "emitter"
_LEDGER_FILE = Path(__file__).with_name("verbs_ledger.txt")

#: The one module besides a primitive's ``_emit`` that may call
#: ``command()``: K2's replay (ADR 0114 D3).
_COMMAND_REPLAY_MODULE = _OPENSEES_DIR / "_internal" / "compose.py"

#: Emitter module stem -> its concrete class.
_EMITTERS = {
    "tcl": "TclEmitter",
    "py": "PyEmitter",
    "live": "LiveOpsEmitter",
    "h5": "H5Emitter",
    "recording": "RecordingEmitter",
}

#: K0-5 (#1341): the stage verbs ``/sequence`` reads.
_K0_5_VERBS = frozenset({
    "stage_open", "stage_close",
    "constraints", "numberer", "system", "test", "algorithm",
    "integrator", "analysis",
    "analyze", "pattern_open", "pattern_close", "recorder", "fix",
    "mass", "remove_sp", "remove_element", "domain_change", "set_time",
    "update_material_stage",
})


def _load_verbs() -> ModuleType:
    """Execute ``verbs.py`` alone, without importing the apeGmsh package."""
    name = "_apegmsh_verbs_lock_probe"
    spec = importlib.util.spec_from_file_location(
        name, _EMITTER_DIR / "verbs.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    saved = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if saved is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved
    return module


def _class_node(stem: str, cls: str) -> ast.ClassDef:
    tree = ast.parse((_EMITTER_DIR / f"{stem}.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return node
    raise AssertionError(f"{stem}.py defines no class {cls}")


def _methods(cls: ast.ClassDef) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [n for n in cls.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _assigned_names(cls: ast.ClassDef) -> set[str]:
    """Names bound by class-level assignments (``x = property(...)``,
    ``flag: bool = True``): public surface a ``def`` scan would miss."""
    names: set[str] = set()
    for node in cls.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Name):
                    names.add(sub.id)
    return names


def _read_ledger(path: Path = _LEDGER_FILE) -> tuple[int, frozenset[str]]:
    n_ledger: int | None = None
    names: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("N_LEDGER"):
            n_ledger = int(line.split("=", 1)[1])
            continue
        names.append(line)
    assert n_ledger is not None, f"{path.name} has no N_LEDGER line"
    assert len(names) == len(set(names)), f"{path.name} repeats a line"
    return n_ledger, frozenset(names)


VERBS_MOD = _load_verbs()
VERBS = VERBS_MOD.VERBS
PROTOCOL = _methods(_class_node("base", "Emitter"))
PROTOCOL_NAMES = [m.name for m in PROTOCOL]
PROTOCOL_ROWS = {k: v for k, v in VERBS.items() if v.via == "protocol"}
#: The Protocol's non-method names (ADR 0114 D6): ``caps`` is the
#: ``TargetCaps`` declaration every emitter carries. Not a side channel
#: and not a verb, so it is outside both ``SIDE_CHANNELS`` and the count.
PROTOCOL_ATTRS = frozenset({"caps"})


def test_verbs_module_imports_nothing_from_apegmsh() -> None:
    tree = ast.parse((_EMITTER_DIR / "verbs.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0, "verbs.py uses a relative import"
            assert not (node.module or "").startswith("apeGmsh")
        elif isinstance(node, ast.Import):
            assert not any(a.name.startswith("apeGmsh") for a in node.names)


def test_a_protocol_method_count_is_frozen() -> None:
    assert len(PROTOCOL_NAMES) == len(set(PROTOCOL_NAMES))
    bound = _assigned_names(_class_node("base", "Emitter"))
    assert bound == PROTOCOL_ATTRS, (
        f"base.py::Emitter binds {sorted(bound)} by assignment; the "
        f"Protocol holds methods plus exactly {sorted(PROTOCOL_ATTRS)} "
        "(ADR 0114 D6), and its method count is frozen"
    )
    assert len(PROTOCOL_NAMES) == VERBS_MOD.EMITTER_METHOD_COUNT, (
        f"base.py::Emitter has {len(PROTOCOL_NAMES)} methods; ADR 0114 "
        f"freezes it at {VERBS_MOD.EMITTER_METHOD_COUNT}. A new verb goes "
        "through command(), not a new Protocol method."
    )
    assert PROTOCOL_NAMES[-1] == "command", (
        f"base.py::Emitter ends with {PROTOCOL_NAMES[-1]!r}; ADR 0114 D2 "
        "makes command() the last method"
    )


def test_b_protocol_rows_match_the_protocol() -> None:
    assert set(PROTOCOL_ROWS) == set(PROTOCOL_NAMES), (
        f"missing rows: {sorted(set(PROTOCOL_NAMES) - set(PROTOCOL_ROWS))}; "
        f"rows with no method: {sorted(set(PROTOCOL_ROWS) - set(PROTOCOL_NAMES))}"
    )
    for method in PROTOCOL:
        annotation = "None" if method.returns is None else ast.unparse(method.returns)
        assert PROTOCOL_ROWS[method.name].returns == annotation, method.name


def test_b_rows_are_well_formed() -> None:
    for key, row in VERBS.items():
        assert key == row.verb
        assert row.via in {"protocol", "command"}, key
        assert row.family in VERBS_MOD.FAMILIES, key
        assert row.scope in {"global", "stage", "both"}, key
        assert row.h5 in {"archive", "refuse", "ledger"}, key
        assert row.seq_op in VERBS_MOD.SEQ_OPS, key
        assert row.seq_what in VERBS_MOD.SEQ_WHATS, key
        assert (row.seq_op == "") == (row.seq_what == ""), key
        assert isinstance(row.requires, frozenset), key
        if row.h5 == "archive":
            assert row.store, f"{key}: an archive row names its store"
        if row.h5 == "refuse":
            assert row.store == "", f"{key}: a refuse row writes nothing"


def test_c_k0_5_stage_verbs_archive() -> None:
    for verb in sorted(_K0_5_VERBS):
        row = VERBS[verb]
        assert row.h5 == "archive", f"K0-5: {verb} must archive, not {row.h5}"
        assert row.scope in {"stage", "both"}, f"K0-5: {verb} scope {row.scope}"


def test_d_ledger_only_shrinks() -> None:
    n_ledger, listed = _read_ledger()
    ledger = frozenset(k for k, v in VERBS.items() if v.h5 == "ledger")
    assert len(listed) == n_ledger, (
        f"verbs_ledger.txt lists {len(listed)} verbs but N_LEDGER = {n_ledger}"
    )
    assert len(ledger) <= n_ledger, (
        f"{len(ledger)} ledger rows exceed N_LEDGER = {n_ledger}; the "
        "ledger may only shrink"
    )
    assert ledger == listed, (
        f"new ledger rows: {sorted(ledger - listed)}; rows that left the "
        f"ledger (delete their lines and lower N_LEDGER): {sorted(listed - ledger)}"
    )


_FuncDef = ast.FunctionDef | ast.AsyncFunctionDef

#: ``archive`` rows whose H5 call writes nothing: their information
#: reaches the archive through the ``set_initial_stress_records`` /
#: ``set_stage_records`` side channels (ADR 0055). Each must stay a
#: trivial body; an entry that grows a real body is stale.
_ARCHIVED_BY_SIDE_CHANNEL = frozenset(
    {"addToParameter", "flip_element_stage", "step_hook_ramp"})


#: The exception classes an H5 refusal raises by name. ``H5RefusedVerb``
#: is the ``_refuse`` helper's class; ``test_f_refused_verb_is_a_not_implemented_error``
#: pins its base.
_REFUSAL_EXCEPTIONS = frozenset({"NotImplementedError", "H5RefusedVerb"})


def _self_calls(fn: _FuncDef) -> set[str]:
    """Attribute names of every ``self.<name>(...)`` call in ``fn``."""
    return {
        node.func.attr for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
    }


def _reaches(method: _FuncDef, helpers: dict[str, _FuncDef], target: str) -> bool:
    """True when ``method``, or any ``self._helper(...)`` it calls
    transitively, calls ``self.<target>(...)``. The walk stops at
    ``target`` itself, so a helper's own body is not inspected."""
    seen: set[str] = set()
    stack: list[_FuncDef] = [method]
    while stack:
        fn = stack.pop()
        calls = _self_calls(fn)
        if target in calls:
            return True
        for name in calls:
            if name in helpers and name not in seen:
                seen.add(name)
                stack.append(helpers[name])
    return False


def _refusal_names(tree: ast.Module) -> frozenset[str]:
    """``NotImplementedError``, ``H5RefusedVerb`` and every name bound to
    one of them anywhere in ``tree`` (``E = NotImplementedError``), so an
    aliased raise is still a refusal."""
    names = set(_REFUSAL_EXCEPTIONS)
    grew = True
    while grew:
        grew = False
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Name)
                    and node.value.id in names):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id not in names:
                        names.add(target.id)
                        grew = True
    return frozenset(names)


def _raises_refusal(
    method: _FuncDef, helpers: dict[str, _FuncDef], names: frozenset[str],
    *, skip: frozenset[str],
) -> bool:
    """True when ``method``, or any ``self._helper(...)`` it calls
    transitively (a visited set bounds the walk, not a depth), has
    ``raise <refusal>(...)`` for a name in ``names``. Helpers in ``skip``
    (``_refuse`` itself) are not followed: reaching ``_refuse`` is the
    sanctioned path, checked by :func:`_reaches`."""
    seen: set[str] = set()
    stack: list[_FuncDef] = [method]
    while stack:
        fn = stack.pop()
        for node in ast.walk(fn):
            if isinstance(node, ast.Raise) and node.exc is not None:
                exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
                if isinstance(exc, ast.Name) and exc.id in names:
                    return True
        for name in _self_calls(fn):
            if name in helpers and name not in skip and name not in seen:
                seen.add(name)
                stack.append(helpers[name])
    return False


def _is_trivial(stmt: ast.stmt) -> bool:
    """``del ...``, ``_ = ...``, ``pass``, ``return <const>`` or a bare
    constant (a docstring or ``...``): a statement that stores nothing."""
    if isinstance(stmt, (ast.Delete, ast.Pass)):
        return True
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
        return True
    if isinstance(stmt, ast.Return):
        value = stmt.value
        if value is None or isinstance(value, ast.Constant):
            return True
        return isinstance(value, (ast.List, ast.Dict, ast.Tuple)) and not (
            value.elts if isinstance(value, (ast.List, ast.Tuple)) else value.keys)
    if isinstance(stmt, ast.Assign):
        return all(isinstance(t, ast.Name) and t.id == "_" for t in stmt.targets)
    return False


def _h5_tree() -> ast.Module:
    return ast.parse((_EMITTER_DIR / "h5.py").read_text(encoding="utf-8"))


def _h5_methods() -> dict[str, _FuncDef]:
    return {m.name: m for m in _methods(_class_node("h5", "H5Emitter"))}


def test_f_h5_column_agrees_with_the_h5_emitter_source() -> None:
    """A ``refuse`` row's ``H5Emitter`` method reaches ``self._refuse(``, a
    ``ledger`` row's reaches ``self._ledger(``, and no method reaches the
    helper its row does not name or raises a refusal outside ``_refuse``
    (itself, through a helper within three hops, or under an alias)."""
    methods = _h5_methods()
    names = _refusal_names(_h5_tree())
    for helper in ("_refuse", "_ledger"):
        assert helper in methods, f"H5Emitter.{helper} is gone; every row body routes through it"
    for verb, row in PROTOCOL_ROWS.items():
        assert verb in methods, f"H5Emitter.{verb} is not a def; the lock cannot read it"
        method = methods[verb]
        refuses = _reaches(method, methods, "_refuse")
        ledgers = _reaches(method, methods, "_ledger")
        assert not _raises_refusal(method, methods, names, skip=frozenset({"_refuse"})), (
            f"H5Emitter.{verb} raises a refusal outside self._refuse (directly, "
            "through a helper, or under an alias); call self._refuse(verb, detail)"
        )
        assert refuses == (row.h5 == "refuse"), (
            f"{verb} is {row.h5!r} but H5Emitter.{verb} "
            f"{'reaches' if refuses else 'never reaches'} self._refuse("
        )
        assert ledgers == (row.h5 == "ledger"), (
            f"{verb} is {row.h5!r} but H5Emitter.{verb} "
            f"{'reaches' if ledgers else 'never reaches'} self._ledger("
        )


def test_f_refused_verb_is_a_not_implemented_error() -> None:
    """``H5RefusedVerb`` subclasses ``NotImplementedError``, so the sites
    that already catch the H5 deferral keep catching it."""
    tree = ast.parse((_EMITTER_DIR / "h5.py").read_text(encoding="utf-8"))
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    assert "H5RefusedVerb" in classes, "h5.py defines no H5RefusedVerb"
    bases = {ast.unparse(b) for b in classes["H5RefusedVerb"].bases}
    assert "NotImplementedError" in bases, bases


def test_archive_rows_have_a_body_that_stores() -> None:
    """An ``archive`` row's ``H5Emitter`` method does more than discard its
    arguments. It catches a row flipped to ``archive`` over a no-op body
    and an archive body emptied to ``del``; a body that still stores but
    stores the wrong thing is K2's round-trip oracle, not this lock."""
    methods = _h5_methods()
    for verb, row in PROTOCOL_ROWS.items():
        if row.h5 != "archive":
            continue
        trivial = all(_is_trivial(s) for s in methods[verb].body)
        if verb in _ARCHIVED_BY_SIDE_CHANNEL:
            assert trivial, (
                f"H5Emitter.{verb} now has a body; drop it from "
                "_ARCHIVED_BY_SIDE_CHANNEL"
            )
        else:
            assert not trivial, (
                f"{verb} is 'archive' but H5Emitter.{verb} only discards "
                "its arguments; mark it 'ledger' or make it store"
            )


@pytest.mark.parametrize("stem", sorted(_EMITTERS))
def test_e_emitters_define_protocol_and_declare_side_channels(stem: str) -> None:
    cls = _class_node(stem, _EMITTERS[stem])
    defined = {m.name for m in _methods(cls)} | _assigned_names(cls)
    missing = (set(PROTOCOL_NAMES) | PROTOCOL_ATTRS) - defined
    assert not missing, f"{_EMITTERS[stem]} lacks {sorted(missing)}"
    public_extra = {n for n in defined
                    if not n.startswith("_") and n not in PROTOCOL_NAMES
                    and n not in PROTOCOL_ATTRS}
    side = VERBS_MOD.SIDE_CHANNELS[stem]
    assert public_extra == side, (
        f"{_EMITTERS[stem]}: undeclared public names "
        f"{sorted(public_extra - side)}; stale SIDE_CHANNELS entries "
        f"{sorted(side - public_extra)}"
    )


def test_e_side_channels_cover_exactly_the_five_emitters() -> None:
    assert set(VERBS_MOD.SIDE_CHANNELS) == set(_EMITTERS)


# -- (g) the command channel's callers (ADR 0114 D3) -------------------------

COMMAND_VERBS = frozenset(k for k, v in VERBS.items() if v.via == "command")


def _primitive_classes(trees: list[ast.Module]) -> frozenset[str]:
    """Names of the classes whose base chain reaches ``Primitive``, resolved
    by base **name** across ``trees`` (a primitive's bases, ``Numberer``,
    ``Integrator``, ..., are imported from ``_internal/types.py``). Two
    classes sharing a name share their bases, which only widens the set."""
    bases: dict[str, set[str]] = {}
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases.setdefault(node.name, set()).update(
                    ast.unparse(b).split(".")[-1] for b in node.bases)
    primitives = {"Primitive"}
    grew = True
    while grew:
        grew = False
        for name, its_bases in bases.items():
            if name not in primitives and its_bases & primitives:
                primitives.add(name)
                grew = True
    return frozenset(primitives)


def _command_call_violations(
    tree: ast.AST, label: str, allowed: frozenset[str],
    primitives: frozenset[str], *, replay: bool,
) -> list[str]:
    """Every use of ``command`` in ``tree`` that breaks ADR 0114 D3.

    ``allowed`` is the set of ``via == "command"`` verbs, ``primitives``
    the class names that resolve to ``Primitive``; ``replay`` is True for
    ``_internal/compose.py``, where any function may call it. Elsewhere
    the caller must be a ``_emit`` method of a ``Primitive`` subclass, the
    verb a string literal with a row, and the call a direct
    ``<x>.command(...)``: ``getattr(<x>, "command")`` and a bound
    ``<x>.command`` that is not called on the spot (``cmd = e.command``)
    are flagged, since the scanner cannot follow them to their verb.
    """
    out: list[str] = []
    called_attrs = {
        id(node.func) for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    def visit(node: ast.AST, func: _FuncDef | None, cls: ast.ClassDef | None) -> None:
        if isinstance(node, ast.ClassDef):
            cls, func = node, None
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func = node
        where = f"{label}:{getattr(node, 'lineno', '?')}"
        if isinstance(node, ast.Attribute) and node.attr == "command" and id(node) not in called_attrs:
            out.append(f"{where}: <x>.command bound without being called; call it directly")
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "getattr" and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant) and node.args[1].value == "command"):
            out.append(f"{where}: command() reached through getattr(); call <x>.command(...) directly")
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "command"):
            caller = func.name if func is not None else "<module>"
            owner = cls.name if cls is not None else None
            if not replay and not (caller == "_emit" and owner in primitives):
                out.append(
                    f"{where}: command() called from "
                    f"{(owner + '.' if owner else '') + caller!r}, not a Primitive._emit"
                )
            verb = node.args[0] if node.args else None
            if not (isinstance(verb, ast.Constant) and isinstance(verb.value, str)):
                out.append(f"{where}: command() verb is not a string literal")
            elif verb.value not in allowed:
                out.append(f"{where}: command({verb.value!r}) has no via='command' VERBS row")
        for child in ast.iter_child_nodes(node):
            visit(child, func, cls)

    visit(tree, None, None)
    return out


def test_g_command_callers_pass_an_allowed_literal_from_a_primitive_emit() -> None:
    paths = sorted(_OPENSEES_DIR.rglob("*.py"))
    trees = [ast.parse(p.read_text(encoding="utf-8")) for p in paths]
    primitives = _primitive_classes(trees)
    assert {"Primitive", "Numberer", "Plain", "Integrator"} <= primitives, (
        "the Primitive hierarchy no longer resolves by base name; fix the resolver"
    )
    violations: list[str] = []
    for path, tree in zip(paths, trees):
        violations += _command_call_violations(
            tree, str(path.relative_to(_ROOT)), COMMAND_VERBS, primitives,
            replay=path == _COMMAND_REPLAY_MODULE,
        )
    assert not violations, "\n".join(violations)


_OK_CALL = """
class P(Primitive):
    def _emit(self, emitter, tag):
        emitter.command("probe", tag, 1.5)
"""
_OK_VIA_INTERMEDIATE_BASE = """
class Mid(Primitive): ...
class P(Mid):
    def _emit(self, emitter, tag):
        emitter.command("probe", tag)
"""
_NON_LITERAL = """
class P(Primitive):
    def _emit(self, emitter, tag):
        verb = "probe"
        emitter.command(verb, tag)
"""
_NO_ROW = """
class P(Primitive):
    def _emit(self, emitter, tag):
        emitter.command("fix", tag, 1)
"""
_WRONG_CALLER = """
def helper(emitter):
    emitter.command("probe", 1)
"""
_METHOD_NOT_EMIT = """
class P(Primitive):
    def emit_extra(self, emitter):
        self._emitter.command("probe", 1)
"""
_EMIT_ON_NON_PRIMITIVE = """
class Helper:
    def _emit(self, emitter, tag):
        emitter.command("probe", tag)
"""
_GETATTR_BYPASS = """
class P(Primitive):
    def _emit(self, emitter, tag):
        getattr(emitter, "command")("probe", tag)
"""
_ALIAS_BYPASS = """
class P(Primitive):
    def _emit(self, emitter, tag):
        cmd = emitter.command
        cmd("probe", tag)
"""


@pytest.mark.parametrize(
    ("source", "replay", "n_violations"),
    [
        (_OK_CALL, False, 0),
        (_OK_VIA_INTERMEDIATE_BASE, False, 0),
        (_NON_LITERAL, False, 1),
        (_NO_ROW, False, 1),
        (_WRONG_CALLER, False, 1),
        (_METHOD_NOT_EMIT, False, 1),
        (_EMIT_ON_NON_PRIMITIVE, False, 1),
        (_GETATTR_BYPASS, False, 1),
        (_ALIAS_BYPASS, False, 1),
        (_WRONG_CALLER, True, 0),
        (_NON_LITERAL, True, 1),
        (_GETATTR_BYPASS, True, 1),
    ],
    ids=["ok", "ok-intermediate-base", "non-literal", "no-row", "module-function",
         "other-method", "emit-on-non-primitive", "getattr-bypass", "alias-bypass",
         "replay-module", "replay-non-literal", "replay-getattr-bypass"],
)
def test_g_scanner_catches_each_planted_violation(
    source: str, replay: bool, n_violations: int,
) -> None:
    """The (g) scanner on planted sources: ``"probe"`` stands for a verb
    with a ``via='command'`` row, ``"fix"`` for one without; ``Primitive``
    resolves from the planted source itself."""
    tree = ast.parse(source)
    found = _command_call_violations(
        tree, "<planted>", frozenset({"probe"}), _primitive_classes([tree]),
        replay=replay)
    assert len(found) == n_violations, found


def test_f_detector_catches_the_review_plants() -> None:
    """Finding 1 of the K1-2 review: a refusal raised in a helper called
    from a verb, and a refusal raised under an alias, are both caught."""
    source = """
E = NotImplementedError
class H5Emitter:
    def fix(self, tag, *dofs):
        self._defer_fix()
    def _defer_fix(self):
        raise NotImplementedError("x")
    def mass(self, tag, *values):
        raise E("x")
    def node(self, tag):
        self._refuse("node", "ok path")
    def _refuse(self, verb, detail):
        raise H5RefusedVerb(verb, None, detail)
    def element(self, *args):
        self._h1()
    def _h1(self):
        self._h2()
    def _h2(self):
        self._h3()
    def _h3(self):
        self._h4()
    def _h4(self):
        self._h1()  # a cycle: the visited set must end the walk
        raise NotImplementedError("four hops deep")
"""
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    methods = {m.name: m for m in _methods(cls)}
    names = _refusal_names(tree)
    assert "E" in names
    skip = frozenset({"_refuse"})
    assert _raises_refusal(methods["fix"], methods, names, skip=skip)
    assert _raises_refusal(methods["mass"], methods, names, skip=skip)
    assert _raises_refusal(methods["element"], methods, names, skip=skip)
    assert not _raises_refusal(methods["node"], methods, names, skip=skip)


# ---------------------------------------------------------------------------
# (h) The params_names ratchet (K1-7, #1464): argv equals fields, over
#     every concrete primitive
# ---------------------------------------------------------------------------

#: The ledger: every concrete primitive of the registry that the archive
#: does NOT name, one line each, as ``<category> <kind>.<Class>``:
#:   ``unnamed``      its store row's argv is not its dataclass fields;
#:   ``uncheckable``  a flat-argv family, but no roster sample to emit
#:                    (``: <reason>`` follows);
#:   ``nostore``      a family whose rows carry no flat argv (elements,
#:                    patterns, recorders, transforms, the chain), so
#:                    ``params_names`` is ``""`` by rule.
#: ``N_LEDGER`` counts the ``unnamed`` + ``uncheckable`` lines and may
#: only go down; ``N_NOSTORE`` pins the ``nostore`` lines exactly. A
#: primitive that is neither named nor on a line fails the test.
_PARAMS_LEDGER_FILE = Path(__file__).with_name("params_names_ledger.txt")
_LEDGER_CATEGORIES = ("unnamed", "uncheckable", "nostore")


def _read_params_ledger() -> tuple[int, int, dict[str, str]]:
    """``(N_LEDGER, N_NOSTORE, {"<kind>.<Class>": category})``."""
    n_ledger: int | None = None
    n_nostore: int | None = None
    lines: dict[str, str] = {}
    for raw in _PARAMS_LEDGER_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("N_LEDGER"):
            n_ledger = int(line.split("=", 1)[1])
            continue
        if line.startswith("N_NOSTORE"):
            n_nostore = int(line.split("=", 1)[1])
            continue
        category, rest = line.split(" ", 1)
        assert category in _LEDGER_CATEGORIES, f"params_names_ledger.txt: {line!r}"
        name = rest.split(":", 1)[0].strip()
        assert name not in lines, f"params_names_ledger.txt repeats {name}"
        lines[name] = category
    assert n_ledger is not None and n_nostore is not None, (
        "params_names_ledger.txt needs N_LEDGER and N_NOSTORE lines")
    return n_ledger, n_nostore, lines


def _registry() -> dict[type, str]:
    """Every concrete primitive (a non-abstract dataclass below
    ``Primitive``, its ``_emit`` own or inherited) the
    ``apeGmsh.opensees`` package defines, with its allocator kind."""
    import dataclasses
    import importlib
    import inspect
    import pkgutil

    import apeGmsh.opensees as pkg
    from apeGmsh.opensees._internal.types import Primitive
    from apeGmsh.opensees.apesees import _KIND_BY_FAMILY

    for mod in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        importlib.import_module(mod.name)
    out: dict[type, str] = {}
    stack: list[type] = [Primitive]
    seen: set[type] = set()
    while stack:
        for sub in stack.pop().__subclasses__():
            if sub in seen:
                continue
            seen.add(sub)
            stack.append(sub)
            # ``@dataclass(slots=True)`` rebuilds the class; the pre-slots
            # original survives in ``__subclasses__`` through its methods'
            # ``__class__`` cells. The module's attribute is the live one.
            live = getattr(sys.modules[sub.__module__], sub.__qualname__, None)
            if live is not sub or not sub.__module__.startswith(pkg.__name__ + "."):
                continue  # a test's own fake primitive is not the registry
            if dataclasses.is_dataclass(sub) and not inspect.isabstract(sub):
                kind = next(k for base, k in _KIND_BY_FAMILY if issubclass(sub, base))
                out[sub] = kind
    return out


def _samples() -> dict[type, object]:
    """One instance per primitive the contract rosters can construct
    (their minimal instances), plus one of each integration rule and
    damping object, which have no roster of their own."""
    from apeGmsh.opensees import integration as integ
    from apeGmsh.opensees.damping import damping as damp
    from apeGmsh.opensees.section.beam import ElasticSection

    from .test_analysis_contract import ALL_ANALYSIS_COMPONENTS
    from .test_analysis_contract import _minimal as analysis
    from .test_element_beam_column_contract import ALL_BEAM_COLUMN_ELEMENTS
    from .test_element_beam_column_contract import _minimal as beam
    from .test_element_shell_contract import ALL_SHELL_ELEMENTS
    from .test_element_shell_contract import _make_minimal as shell
    from .test_element_solid_contract import ALL_SOLID_ELEMENTS
    from .test_element_solid_contract import _make_minimal as solid
    from .test_element_truss_contract import ALL_TRUSS_ELEMENTS
    from .test_element_truss_contract import _minimal as truss
    from .test_element_zero_length_contract import ALL_ZERO_LENGTH_ELEMENTS
    from .test_element_zero_length_contract import _minimal as zero
    from .test_nd_material_contract import ALL_ND, _instantiate
    from .test_pattern_contract import ALL_PATTERNS
    from .test_pattern_contract import _minimal_instance as pattern
    from .test_recorder_contract import ALL_RECORDERS
    from .test_recorder_contract import _minimal_instance as recorder
    from .test_section_contract import ALL_SECTIONS, _make_minimal
    from .test_time_series_contract import ALL_TIME_SERIES, _minimal_instance
    from .test_uniaxial_material_contract import ALL_UNIAXIAL, _minimal

    sec = ElasticSection(E=200e9, A=0.01, Iz=1e-4)
    rosters: list[tuple[list[type], object]] = [
        (ALL_UNIAXIAL, _minimal), (ALL_ND, _instantiate),
        (ALL_SECTIONS, _make_minimal), (ALL_TIME_SERIES, _minimal_instance),
        (ALL_BEAM_COLUMN_ELEMENTS, beam), (ALL_TRUSS_ELEMENTS, truss),
        (ALL_ZERO_LENGTH_ELEMENTS, zero), (ALL_SHELL_ELEMENTS, shell),
        (ALL_SOLID_ELEMENTS, solid), (ALL_PATTERNS, pattern),
        (ALL_RECORDERS, recorder), (ALL_ANALYSIS_COMPONENTS, analysis),
    ]
    out: dict[type, object] = {}
    for roster, make in rosters:
        out.update({cls: make(cls) for cls in roster})
    out.update({cls: cls(section=sec, n_ip=3) for cls in (
        integ.Lobatto, integ.Legendre, integ.NewtonCotes, integ.Radau,
        integ.Trapezoidal)})
    out.update({cls: cls(
        section_i=sec, lp_i=0.1, section_j=sec, lp_j=0.1, section_interior=sec,
    ) for cls in (integ.HingeRadau, integ.HingeRadauTwo, integ.HingeMidpoint,
                  integ.HingeEndpoint)})
    out[damp.Uniform] = damp.Uniform(zeta=0.05, freq1=1.0, freq2=10.0)
    out[damp.SecStif] = damp.SecStif(beta=0.01)
    out[damp.URD] = damp.URD(points=((1.0, 0.05), (10.0, 0.05)))
    out[damp.URDbeta] = damp.URDbeta(points=((1.0, 0.01), (10.0, 0.01)))
    return out


def _argv_is_fields(kind: str, prim: object) -> bool:
    """The writer's own rule (``decl_argv_names``) on one primitive: it is
    emitted into a ``RecordingEmitter`` with a stub resolver; its store
    row is the one ``kind`` call's args after the type token and the tag
    (a primitive that emits anything else, a ``Fiber`` section's block,
    has no flat row and is unnamed)."""
    from apeGmsh.opensees._internal.tag_resolution import set_tag_resolver
    from apeGmsh.opensees.emitter.h5 import decl_argv_names, encode_decl_params
    from apeGmsh.opensees.emitter.recording import RecordingEmitter

    rec = RecordingEmitter()
    tags: dict[int, int] = {}
    set_tag_resolver(rec, lambda p: tags.setdefault(id(p), 100 + len(tags)))
    prim._emit(rec, tag=1)  # type: ignore[attr-defined]
    params = encode_decl_params(prim, lambda p: f"k{id(p)}")
    calls = rec.calls
    if len(calls) != 1 or calls[0][0] != kind:
        return False
    return decl_argv_names(
        params, calls[0][1][2:], lambda k: tags.get(int(k[1:]))) is not None


def _classify_registry() -> tuple[dict[str, str], frozenset[str]]:
    """``({"<kind>.<Class>": category}, {named "<kind>.<Class>"})`` over
    every concrete primitive: ``nostore`` outside the flat-argv families,
    ``uncheckable`` without a sample, else named or ``unnamed``."""
    from apeGmsh.opensees.emitter.h5 import _ARGV_STORE_KINDS

    registry = _registry()
    samples = _samples()
    ledger: dict[str, str] = {}
    named: set[str] = set()
    for cls, kind in registry.items():
        name = f"{kind}.{cls.__name__}"
        if kind not in _ARGV_STORE_KINDS:
            ledger[name] = "nostore"
        elif cls not in samples:
            ledger[name] = "uncheckable"
        elif _argv_is_fields(kind, samples[cls]):
            named.add(name)
        else:
            ledger[name] = "unnamed"
    assert len(ledger) + len(named) == len(registry), "a kind.Class repeats"
    return ledger, frozenset(named)


def _check_params_ledger(
    n_ledger: int, n_nostore: int, listed: dict[str, str],
    computed: dict[str, str],
) -> None:
    shrink = {n for n, c in listed.items() if c != "nostore"}
    nostore = {n for n, c in listed.items() if c == "nostore"}
    assert len(shrink) == n_ledger, (
        f"params_names_ledger.txt lists {len(shrink)} unnamed + uncheckable "
        f"primitives but N_LEDGER = {n_ledger}")
    assert len(nostore) == n_nostore, (
        f"params_names_ledger.txt lists {len(nostore)} nostore primitives "
        f"but N_NOSTORE = {n_nostore}")
    computed_shrink = {n for n, c in computed.items() if c != "nostore"}
    assert len(computed_shrink) <= n_ledger, (
        f"{len(computed_shrink)} unnamed + uncheckable primitives exceed "
        f"N_LEDGER = {n_ledger}; the ledger may only shrink")
    assert computed == listed, (
        f"primitives neither named nor ledgered, or in another category: "
        f"{sorted(set(computed.items()) - set(listed.items()))}; lines "
        f"to delete (now named, or gone): "
        f"{sorted(set(listed.items()) - set(computed.items()))}")


def test_h_params_names_ledger_covers_every_primitive_and_only_shrinks() -> None:
    n_ledger, n_nostore, listed = _read_params_ledger()
    computed, named = _classify_registry()
    # The rule names something: a plain material's argv is its fields.
    assert "uniaxialMaterial.Steel01" in named
    assert not (named & set(listed)), "a named primitive is on the ledger"
    _check_params_ledger(n_ledger, n_nostore, listed, computed)


def test_h_a_grown_stale_or_incomplete_ledger_fails() -> None:
    n_ledger, n_nostore, listed = _read_params_ledger()
    computed, _named = _classify_registry()
    # A line for a primitive that is named (stale), with or without the
    # count raised; a new unlisted primitive in each category; and a
    # primitive listed under the wrong category.
    with pytest.raises(AssertionError):
        _check_params_ledger(
            n_ledger + 1, n_nostore,
            {**listed, "uniaxialMaterial.Steel01": "unnamed"}, computed)
    with pytest.raises(AssertionError):
        _check_params_ledger(
            n_ledger, n_nostore,
            {**listed, "uniaxialMaterial.Steel01": "unnamed"}, computed)
    for category in _LEDGER_CATEGORIES:
        with pytest.raises(AssertionError):
            _check_params_ledger(
                n_ledger, n_nostore, listed,
                {**computed, "uniaxialMaterial.New": category})
    name = next(n for n, c in listed.items() if c == "nostore")
    with pytest.raises(AssertionError):
        _check_params_ledger(
            n_ledger, n_nostore, listed, {**computed, name: "unnamed"})


def test_h_an_inheriting_subclass_outside_the_ledger_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A concrete primitive that only inherits ``_emit`` (the
    ``LadrunoRCConcrete`` shape) is in the registry; a new one that is
    neither named nor listed fails the ledger check."""
    import dataclasses

    from apeGmsh.opensees.material import uniaxial
    from apeGmsh.opensees.material.uniaxial import Steel01

    @dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
    class Steel01Plus(Steel01):  # inherits Steel01._emit; one extra field
        extra: float = 1.0

    Steel01Plus.__module__ = uniaxial.__name__
    Steel01Plus.__qualname__ = "Steel01Plus"
    monkeypatch.setattr(uniaxial, "Steel01Plus", Steel01Plus, raising=False)
    n_ledger, n_nostore, listed = _read_params_ledger()
    computed, named = _classify_registry()
    assert "uniaxialMaterial.Steel01Plus" in computed
    assert computed["uniaxialMaterial.Steel01Plus"] == "uncheckable"
    assert "uniaxialMaterial.Steel01Plus" not in listed
    with pytest.raises(AssertionError):
        _check_params_ledger(n_ledger, n_nostore, listed, computed)

