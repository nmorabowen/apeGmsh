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
    name (method or class attribute) it defines is in ``SIDE_CHANNELS``.

It also checks the ``h5`` column against ``H5Emitter``'s source: a
``refuse`` row's method raises ``NotImplementedError`` (directly or
through any ``self._helper`` it reaches) and no other row's method
does, and an ``archive`` row's method does more than discard its
arguments. It reads source, not behaviour: an archive body that stores
the wrong thing is K2's round-trip oracle.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_EMITTER_DIR = _ROOT / "src" / "apeGmsh" / "opensees" / "emitter"
_LEDGER_FILE = Path(__file__).with_name("verbs_ledger.txt")

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


def _read_ledger() -> tuple[int, frozenset[str]]:
    n_ledger: int | None = None
    names: list[str] = []
    for raw in _LEDGER_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("N_LEDGER"):
            n_ledger = int(line.split("=", 1)[1])
            continue
        names.append(line)
    assert n_ledger is not None, "verbs_ledger.txt has no N_LEDGER line"
    assert len(names) == len(set(names)), "verbs_ledger.txt repeats a verb"
    return n_ledger, frozenset(names)


VERBS_MOD = _load_verbs()
VERBS = VERBS_MOD.VERBS
PROTOCOL = _methods(_class_node("base", "Emitter"))
PROTOCOL_NAMES = [m.name for m in PROTOCOL]
PROTOCOL_ROWS = {k: v for k, v in VERBS.items() if v.via == "protocol"}


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
    assert not bound, (
        f"base.py::Emitter binds {sorted(bound)} by assignment; the "
        "Protocol holds methods only, and its count is frozen"
    )
    assert len(PROTOCOL_NAMES) == VERBS_MOD.EMITTER_METHOD_COUNT, (
        f"base.py::Emitter has {len(PROTOCOL_NAMES)} methods; ADR 0114 "
        f"freezes it at {VERBS_MOD.EMITTER_METHOD_COUNT}. A new verb goes "
        "through command(), not a new Protocol method."
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


def _raises_not_implemented(method: _FuncDef, helpers: dict[str, _FuncDef]) -> bool:
    """True when the body, or any ``self._helper(...)`` it reaches
    transitively, raises ``NotImplementedError`` (the H5 deferral refusal)."""
    seen: set[str] = set()
    stack: list[_FuncDef] = [method]
    while stack:
        fn = stack.pop()
        for node in ast.walk(fn):
            if isinstance(node, ast.Raise) and node.exc is not None:
                exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
                if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
                    return True
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"
                    and node.func.attr in helpers
                    and node.func.attr not in seen):
                seen.add(node.func.attr)
                stack.append(helpers[node.func.attr])
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


def _h5_methods() -> dict[str, _FuncDef]:
    return {m.name: m for m in _methods(_class_node("h5", "H5Emitter"))}


def test_h5_column_agrees_with_the_h5_emitter_source() -> None:
    """``refuse`` rows raise ``NotImplementedError`` in ``H5Emitter``;
    ``archive`` and ``ledger`` rows never do."""
    methods = _h5_methods()
    for verb, row in PROTOCOL_ROWS.items():
        assert verb in methods, f"H5Emitter.{verb} is not a def; the lock cannot read it"
        raises = _raises_not_implemented(methods[verb], methods)
        if row.h5 == "refuse":
            assert raises, f"{verb} is 'refuse' but H5Emitter.{verb} never refuses"
        else:
            assert not raises, f"{verb} is {row.h5!r} but H5Emitter.{verb} refuses"


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
    missing = set(PROTOCOL_NAMES) - defined
    assert not missing, f"{_EMITTERS[stem]} lacks {sorted(missing)}"
    public_extra = {n for n in defined
                    if not n.startswith("_") and n not in PROTOCOL_NAMES}
    side = VERBS_MOD.SIDE_CHANNELS[stem]
    assert public_extra == side, (
        f"{_EMITTERS[stem]}: undeclared public names "
        f"{sorted(public_extra - side)}; stale SIDE_CHANNELS entries "
        f"{sorted(side - public_extra)}"
    )


def test_e_side_channels_cover_exactly_the_five_emitters() -> None:
    assert set(VERBS_MOD.SIDE_CHANNELS) == set(_EMITTERS)
