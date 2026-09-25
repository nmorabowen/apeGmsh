"""Viewer recurrence guards: a fixed bug pattern must not survive at a sibling site.

The viewers' most repeated failure in git history is not a new bug. It is
the second copy of an old one: a fix repairs the site that was reported,
and the same pattern survives somewhere else until it bites on its own.
Every guard below comes from that sequence, with the real commits:

* **G-VTK-REMOVED** — viewer code calls no method that a supported VTK
  removed. ``vtk>=9.2`` has no upper bound. VTK 9.7 removed
  ``AddActor2D`` / ``RemoveActor2D``.
  - #1122 (3b820e6e) fixed the legend title in ``backends/pyvista_qt.py``.
  - #1123 (091e5a05), the same day, fixed ``backends/_pyvista_pick.py``,
    where rubber-band selection raised. "No test covers this path".
* **G-QAPP-ENV** — every ``QApplication(...)`` construction is preceded,
  in the same function, by ``prepare_qt_environment()``, or by a helper in
  the same file that calls it, such as ``ViewerWindow._lazy_qt()``. Qt reads ``QT_QPA_PLATFORM`` / ``QT_STYLE_OVERRIDE``
  only once, at construction.
  - #743 (bf3c62c6) added the guard at three entry points.
  - #782 (dc4344e1) found a fourth, ``results_viewer._ensure_qapplication``,
    where KDE Wayland still crashed "despite the guard existing".
* **G-GHOST-BIT** — the hidden-cell ghost bit is
  ``vtkDataSetAttributes.HIDDENCELL = 0x20``. ``0x01`` is DUPLICATECELL,
  which hides nothing.
  - #781 (5e23f5fd) fixed the backend's constant.
  - #878 (45db15df), three weeks later, found ``core/element_visibility.py``
    and ``core/results_pick.py`` still on ``0x01``. Hide, isolate and the
    dim filter had no visual effect: "two halves disagreed".
* **G-HASH** — no builtin ``hash()`` in viewer code outside ``__hash__``.
  It is randomized per process (``PYTHONHASHSEED``), so any colour or
  palette index derived from it changes between sessions. Use
  ``zlib.crc32``.
  - #374 (3b5cd921) fixed the Module colour mode and wrote down that the
    Physical Group mode "still uses hash()".
  - 184b5734 fixed that one.
  - A third site, ``ui/loads_tab.py::pattern_color``, which colours the
    viewport's load arrows, was never swept.

Pure ``ast`` walks over ALL of ``src/apeGmsh/viewers/**``, with no Qt, VTK
or GL needed, so they run in every lane. They are hard zero, with no
allowlist. A docstring or comment that names a pattern is not a hit. Each
collector takes the viewers root as an argument, so it can be run against
a ``git archive`` of a historical tree. The mutation acceptance on the
commits above is recorded in ``internal_docs/plan_agent_surface_viewers.md``.

Siblings: the ADR 0056 guards (``test_viewer_state_contract.py``, which
also holds G-ACTORS, the same story for ``Diagram._actors``) and the
ADR 0087 style guards (``test_viewer_style_contract.py``). The behavioural
tests of these incidents (``test_scalar_bar_title_warning.py``,
``test_qt_env.py``, ``test_element_hide_pixels.py``) need a Qt/VTK stack;
this file is their structural half.
"""
from __future__ import annotations

import ast
import functools
from collections.abc import Callable, Iterator
from pathlib import Path

VIEWERS_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "apeGmsh"
    / "viewers"
)

Hits = list[tuple[int, str]]


@functools.lru_cache(maxsize=None)
def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _hits_under(viewers_dir: Path, collect: Callable[[ast.AST], Hits]) -> list[str]:
    """Every hit of ``collect`` under ``viewers_dir`` as ``rel:line: what``."""
    found: list[str] = []
    for path in sorted(viewers_dir.rglob("*.py")):
        tree = _parse(path)
        rel = path.relative_to(viewers_dir).as_posix()
        found.extend(f"{rel}:{ln}: {what}" for ln, what in collect(tree))
    return found


def _int_literal(node: ast.expr) -> int | None:
    """The value of an int literal (``0x01``, ``~0x01``); ``None`` otherwise."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
        inner = _int_literal(node.operand)
        return None if inner is None else ~inner
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ):
        return node.value
    return None


# ── G-VTK-REMOVED ────────────────────────────────────────────────────
# method name -> (VTK release that removed it, what to call instead, incident)
REMOVED_VTK_API: dict[str, tuple[str, str, str]] = {
    "AddActor2D": ("9.7", "AddActor", "#1122 / #1123"),
    "RemoveActor2D": ("9.7", "RemoveActor", "#1122 / #1123"),
}


def removed_vtk_api(tree: ast.AST) -> Hits:
    """Every ``<expr>.<removed name>`` attribute, called or only referenced."""
    return [
        (node.lineno, node.attr)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in REMOVED_VTK_API
    ]


# ── G-QAPP-ENV ───────────────────────────────────────────────────────
_ENV_GUARD = "prepare_qt_environment"


def _own_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """Nodes of ``scope`` without descending into nested defs/classes."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            stack.extend(ast.iter_child_nodes(node))


def _called_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _env_guard_names(tree: ast.AST) -> set[str]:
    """``prepare_qt_environment`` plus every function defined in this file
    that (transitively) calls it — e.g. ``ViewerWindow._lazy_qt``.

    The helper counts as a guard only while it really calls the guard.
    Before #743, ``_lazy_qt`` existed but did not, and a name-based
    allowance would have hidden that site.
    """
    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    names = {_ENV_GUARD}
    grew = True
    while grew:
        grew = False
        for fn in funcs:
            if fn.name in names:
                continue
            if any(isinstance(n, ast.Call) and _called_name(n) in names for n in _own_nodes(fn)):
                names.add(fn.name)
                grew = True
    return names


def unguarded_qapplication(tree: ast.AST) -> Hits:
    """``QApplication(...)`` with no earlier env-guard call in the same scope.

    The scope is the innermost enclosing function, or the module body.
    ``QApplication.instance()`` is a lookup, not a construction, and is
    not matched.
    """
    hits: Hits = []
    if not any(
        isinstance(n, ast.Call) and _called_name(n) == "QApplication" for n in ast.walk(tree)
    ):
        return hits
    guard_names = _env_guard_names(tree)
    scopes = [tree] + [
        n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for scope in scopes:
        calls = [n for n in _own_nodes(scope) if isinstance(n, ast.Call)]
        guards = [c.lineno for c in calls if _called_name(c) in guard_names]
        for call in calls:
            if _called_name(call) == "QApplication" and not any(g < call.lineno for g in guards):
                hits.append((call.lineno, "QApplication(...) before prepare_qt_environment()"))
    return hits


# ── G-GHOST-BIT ──────────────────────────────────────────────────────
HIDDENCELL = 0x20  # vtkDataSetAttributes.HIDDENCELL; 0x01 is DUPLICATECELL
_BIT_OPS = (ast.BitAnd, ast.BitOr, ast.BitXor)


def wrong_ghost_bit(tree: ast.AST) -> Hits:
    """A hidden-cell constant, or a literal mask on a ghost array, that is not 0x20.

    Two shapes: ``*HIDDEN*CELL* = <int>`` with the wrong value (the #781 /
    #878 constants), and ``<...ghost...> & <int literal>`` with the wrong
    value (the #878 ``results_pick`` mask). A mask through a NAMED
    constant is not read, because the constant's own definition is what
    gets checked.
    """
    hits: Hits = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = _int_literal(node.value)
            for target in targets:
                name = target.id if isinstance(target, ast.Name) else (
                    target.attr if isinstance(target, ast.Attribute) else ""
                )
                upper = name.upper()
                if "HIDDEN" in upper and "CELL" in upper and value is not None and value != HIDDENCELL:
                    hits.append((node.lineno, f"{name} = {value:#04x} (HIDDENCELL is 0x20)"))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, _BIT_OPS):
            for lit, other in ((node.right, node.left), (node.left, node.right)):
                value = _int_literal(lit)
                if value is None or value in (HIDDENCELL, ~HIDDENCELL):
                    continue
                if "ghost" in ast.unparse(other).lower():
                    hits.append((node.lineno, f"ghost mask {ast.unparse(node)} (HIDDENCELL is 0x20)"))
    return hits


# ── G-HASH ───────────────────────────────────────────────────────────
def builtin_hash(tree: ast.AST) -> Hits:
    """Builtin ``hash(...)`` calls outside a ``__hash__`` method."""
    hits: Hits = []
    if not any(isinstance(n, ast.Name) and n.id == "hash" for n in ast.walk(tree)):
        return hits
    scopes = [tree] + [
        n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for scope in scopes:
        if getattr(scope, "name", None) == "__hash__":
            continue
        for node in _own_nodes(scope):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "hash":
                hits.append((node.lineno, f"{ast.unparse(node)} (randomized per process)"))
    return hits


GUARDS: dict[str, tuple[Callable[[ast.AST], Hits], str]] = {
    "G-VTK-REMOVED": (
        removed_vtk_api,
        "a VTK method a supported release removed; use the replacement in "
        "REMOVED_VTK_API (#1122 / #1123)",
    ),
    "G-QAPP-ENV": (
        unguarded_qapplication,
        "call prepare_qt_environment() (ui/_qt_env.py) earlier in the same "
        "function — Qt reads the platform/style env once, at construction (#743 / #782)",
    ),
    "G-GHOST-BIT": (
        wrong_ghost_bit,
        "the hidden-cell ghost bit is 0x20; import HIDDENCELL from "
        "core/element_visibility.py rather than spelling it (#781 / #878)",
    ),
    "G-HASH": (
        builtin_hash,
        "builtin hash() differs between processes; derive colours/indices "
        "from zlib.crc32 (#374 / 184b5734)",
    ),
}


def guard_hits(viewers_dir: Path) -> dict[str, list[str]]:
    """``{guard id: hits}`` over ``viewers_dir``, so the plan doc's mutation
    acceptance can run these same guards on historical trees."""
    return {gid: _hits_under(viewers_dir, collect) for gid, (collect, _) in GUARDS.items()}


def test_scope_covers_the_incident_files() -> None:
    # A scan that stops seeing the files the incidents lived in passes vacuously.
    scanned = {p.relative_to(VIEWERS_DIR).as_posix() for p in VIEWERS_DIR.rglob("*.py")}
    assert {
        "backends/pyvista_qt.py",
        "backends/_pyvista_pick.py",
        "results_viewer.py",
        "ui/viewer_window.py",
        "core/element_visibility.py",
        "core/results_pick.py",
        "core/color_mode_controller.py",
    } <= scanned


def _assert_clean(gid: str) -> None:
    collect, fix = GUARDS[gid]
    hits = _hits_under(VIEWERS_DIR, collect)
    assert not hits, (
        f"{gid} — {fix}. Fix EVERY site listed, not only the one that was "
        "reported; that is how each of these bugs came back:\n  " + "\n  ".join(hits)
    )


def test_g_vtk_removed() -> None:
    _assert_clean("G-VTK-REMOVED")


def test_g_qapp_env() -> None:
    _assert_clean("G-QAPP-ENV")


def test_g_ghost_bit() -> None:
    _assert_clean("G-GHOST-BIT")


def test_g_hash() -> None:
    _assert_clean("G-HASH")


# ── Self-test: each collector sees its incident shapes and nothing else ──
_FLAGGED: dict[str, list[tuple[str, str]]] = {
    "G-VTK-REMOVED": [
        ("pr1122_title", "def f(self, a):\n    self._plotter.renderer.AddActor2D(a)\n"),
        ("pr1123_guarded_remove", (
            "def stop(self):\n    try:\n        self._plotter.renderer.RemoveActor2D(self._band)\n"
            "    except Exception:\n        pass\n")),
        ("bound_method", "def f(ren, a):\n    add = ren.AddActor2D\n    add(a)\n"),
    ],
    "G-QAPP-ENV": [
        ("pr782_ensure", (
            "def _ensure_qapplication():\n    from qtpy import QtWidgets\n"
            "    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])\n")),
        ("guard_after_construction", (
            "def f():\n    app = QApplication([])\n    prepare_qt_environment()\n")),
        ("pr743_helper_without_guard", (
            "class W:\n    def _lazy_qt(self):\n        from qtpy import QtWidgets\n"
            "        return QtWidgets\n    def __init__(self):\n"
            "        QtWidgets = self._lazy_qt()\n        app = QtWidgets.QApplication([])\n")),
        ("guard_only_in_another_function", (
            "def g():\n    prepare_qt_environment()\n"
            "def f():\n    return QtWidgets.QApplication([])\n")),
    ],
    "G-GHOST-BIT": [
        ("pr781_backend_const", "_GHOST_HIDDEN_CELL = 0x01\n"),
        ("pr878_annotated_const", "HIDDENCELL: int = 0x01\n"),
        ("pr878_pick_mask", "def f(mask, ghosts):\n    return mask & ~(ghosts & 0x01)\n"),
    ],
    "G-HASH": [
        ("pr374_palette", "def idle(label, pal):\n    return pal[abs(hash(label)) % len(pal)]\n"),
        ("module_level", "SEED = hash('x')\n"),
    ],
}
_CLEAN: dict[str, list[tuple[str, str]]] = {
    "G-VTK-REMOVED": [
        ("fix_shape", "def f(self, a):\n    self._plotter.renderer.AddActor(a)\n"
                      "    self._plotter.renderer.RemoveActor(a)\n"),
        ("prose", 'def f():\n    """VTK 9.7 removed ``AddActor2D``."""\n    # was AddActor2D\n'),
        ("string", "MSG = 'AddActor2D is gone'\n"),
    ],
    "G-QAPP-ENV": [
        ("fix_shape", (
            "def _ensure_qapplication():\n    from .ui._qt_env import prepare_qt_environment\n"
            "    prepare_qt_environment()\n    from qtpy import QtWidgets\n"
            "    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])\n")),
        ("lazy_qt_shape", (
            "class W:\n    def _lazy_qt(self):\n        prepare_qt_environment()\n"
            "        from qtpy import QtWidgets\n        return QtWidgets\n"
            "    def __init__(self):\n        QtWidgets = self._lazy_qt()\n"
            "        app = QtWidgets.QApplication.instance()\n        if app is None:\n"
            "            app = QtWidgets.QApplication([])\n")),
        ("lookup_only", "def f():\n    return QtWidgets.QApplication.instance()\n"),
    ],
    "G-GHOST-BIT": [
        ("fix_const", "HIDDENCELL: int = 0x20\n_GHOST_HIDDEN_CELL = 0x20\n"),
        ("named_mask", "def f(ghosts):\n    return ghosts & HIDDENCELL\n"),
        ("literal_0x20", "def f(ghosts):\n    return ghosts | 0x20\n"),
        ("float_opacity", "GHOST_OPACITY = 0.3\nGHOST_DIM_OPACITY = 0.1\n"),
        ("not_a_ghost_mask", "def f(flags):\n    return flags & 0x01\n"),
    ],
    "G-HASH": [
        ("fix_shape", "import zlib\ndef idle(label, pal):\n"
                      "    return pal[zlib.crc32(label.encode()) % len(pal)]\n"),
        ("dunder_hash", "class K:\n    def __hash__(self):\n        return hash((self.a, self.b))\n"),
        ("prose", "def f():\n    # zlib.crc32 instead of Python's hash()\n    return 0\n"),
        ("method_named_hash", "def f(h):\n    return h.hash()\n"),
    ],
}


def test_collectors_flag_the_incident_shapes() -> None:
    missed = [
        f"{gid}:{name}"
        for gid, cases in _FLAGGED.items()
        for name, src in cases
        if not GUARDS[gid][0](ast.parse(src))
    ]
    assert not missed, f"collector went blind to: {missed}"


def test_collectors_pass_the_sanctioned_shapes() -> None:
    noisy = {
        f"{gid}:{name}": hits
        for gid, cases in _CLEAN.items()
        for name, src in cases
        if (hits := GUARDS[gid][0](ast.parse(src)))
    }
    assert not noisy, f"collector flagged sanctioned code: {noisy}"
