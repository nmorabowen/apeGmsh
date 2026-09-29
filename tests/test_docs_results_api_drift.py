"""Results-API docs drift lane — the calls a results page shows must exist.

On 2026-09-28 ``docs/how-to/choose-results-strategy.md``, the page that
tells a reader how to record a run, prescribed ``spec.capture(...)``,
``ops.tcl(..., recorders=spec)`` and ``ops.tcl(..., recorders=spec,
mpco=True)``. None of the three existed: ``capture`` left the recorder
spec when Phase 9 deleted the fluent ``Recorders`` helper, and
``apeSees.tcl`` takes neither ``recorders=`` nor ``mpco=``.
``results-mpco.md`` repeated them. A reader copying any of those cells hit
an ``AttributeError`` or a ``TypeError``, and nothing failed.

This lane resolves the Python that the curated results pages show against
the live code, in the source-scanning style of ``test_skill_docs_drift.py``:

* every ``apeGmsh`` import in a python fence must import;
* every attribute chain rooted at a name the page binds (an import, a
  constructor call, a call whose return annotation names a class, a
  ``with ... as`` target, a page-defined helper that returns one) must
  exist on the live class. Class attributes, dataclass fields, class
  annotations, ``self.<name> = ...`` in any method, and a session's
  ``_COMPOSITES`` table all count;
* every keyword passed to a resolved callable must be one its live
  signature accepts, unless it takes ``**kwargs``;
* inline code spans outside the fences (the decision grid is a table)
  that parse as a Python expression get the same checks, against the names
  the page's fences bound, plus the two receivers prose uses by convention
  (``ops``, ``spec``) when the fences never bind them.

Whatever the resolver cannot type (third-party modules, unannotated or
union returns, names a page never binds) is skipped, not guessed. The lane
can miss drift, but it never reports drift that is not there. The controls
at the bottom replay the incident and must fail.
"""
from __future__ import annotations

import ast
import dataclasses
import functools
import importlib
import inspect
import re
import sys
import textwrap
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"

# The results-API pages whose Python is resolved. Each must resolve at
# least ``_MIN_CHECKED`` names, so a page whose fences stop binding what
# they use cannot pass by checking nothing.
_PAGES = (
    "how-to/choose-results-strategy.md",
    "how-to/results-mpco.md",
    "how-to/read-results.md",
    "concepts/results.md",
    "examples/results-strategies.md",
)
_MIN_CHECKED = 3


# ── Markdown extraction ─────────────────────────────────────────────

def _python_fences(text: str) -> list[tuple[int, str]]:
    """``(first_content_lineno, dedented_body)`` for each python fence."""
    fences: list[tuple[int, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        opener = lines[i].strip()
        if opener.startswith("```") and opener[3:].strip() in ("python", "py"):
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            fences.append((i + 2, textwrap.dedent("\n".join(lines[i + 1:j]))))
            i = j + 1
        else:
            i += 1
    return fences


_SPAN = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")


def _inline_spans(text: str) -> list[tuple[int, str]]:
    """``(lineno, code)`` for each single-backtick span outside fences."""
    spans: list[tuple[int, str]] = []
    fenced = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.strip().startswith("```"):
            fenced = not fenced
        elif not fenced:
            spans += [(lineno, m.group(1)) for m in _SPAN.finditer(line)]
    return spans


# ── What a doc name is bound to ─────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class _T:
    """``kind`` is module / class / inst / union (one of several classes,
    see ``_CONVENTIONAL``) / func / page (a helper the page defines) /
    foreign (third-party, never judged)."""

    kind: str
    obj: object


_FOREIGN = _T("foreign", None)
_MISSING = object()
_IDENT = re.compile(r"[A-Za-z_][\w.]*")

# Inline file names (`model.h5`, `run.tcl`) parse as attribute access; a
# stem that happens to be a bound name must not be judged as one.
_FILE_SUFFIXES = frozenset({
    "h5", "hdf5", "mpco", "ladruno", "tcl", "py", "out", "xml", "json",
    "msh", "vtk", "vtu", "csv", "txt", "log",
})

# Prose names these receivers by convention without always binding them in
# a fence: `ops` is the bridge and `spec` is either recorder spec. The
# incident's grid wrote `spec.capture(...)` and `ops.tcl(..., recorders=)`
# on a page whose fences bound neither. Consulted for inline spans only,
# and only when the fences never bound the name; with two candidates, an
# attribute passes if either class has it.
_CONVENTIONAL: dict[str, tuple[tuple[str, str], ...]] = {
    "ops": (("apeGmsh.opensees", "apeSees"),),
    "spec": (
        ("apeGmsh.results.capture", "DomainCaptureSpec"),
        ("apeGmsh.results.spec", "ResolvedRecorderSpec"),
    ),
}


def _ours(obj: object) -> bool:
    return (getattr(obj, "__module__", None) or "").startswith("apeGmsh")


def _wrap(obj: object) -> _T | None:
    if inspect.ismodule(obj):
        return _T("module", obj) if obj.__name__.startswith("apeGmsh") else _FOREIGN
    if isinstance(obj, type):
        return _T("class", obj) if _ours(obj) else _FOREIGN
    if callable(obj):
        return _T("func", obj) if _ours(obj) else _FOREIGN
    return None


def _class_named(name: str, namespace: dict) -> type | None:
    """``name`` looked up in ``namespace``, else the one loaded ``apeGmsh``
    module that defines a class so named. Return annotations often name
    classes imported only under ``TYPE_CHECKING``."""
    head, _, rest = name.partition(".")
    obj = namespace.get(head)
    for part in filter(None, rest.split(".")):
        obj = getattr(obj, part, None)
    if isinstance(obj, type):
        return obj
    if rest:
        return None
    hits = {
        id(c): c
        for m in list(sys.modules.values())
        if (getattr(m, "__name__", None) or "").startswith("apeGmsh")
        for c in [vars(m).get(name)]
        if isinstance(c, type) and c.__module__ == m.__name__
    }
    return next(iter(hits.values())) if len(hits) == 1 else None


def _annotation_type(ann: object, namespace: dict) -> _T | None:
    if isinstance(ann, type):
        return _T("inst", ann) if _ours(ann) else None
    if not isinstance(ann, str):
        return None
    name = ann.strip().strip("'\"")      # PEP 563 may quote a quoted name
    if not _IDENT.fullmatch(name):
        return None                      # unions, generics: not guessed
    cls = _class_named(name, namespace)
    return _T("inst", cls) if cls is not None and _ours(cls) else None


def _returns(fn: object) -> _T | None:
    fn = inspect.unwrap(fn)  # type: ignore[arg-type]
    ann = getattr(fn, "__annotations__", {}).get("return")
    return _annotation_type(ann, getattr(fn, "__globals__", {}))


@functools.lru_cache(maxsize=None)
def _instance_attrs(cls: type) -> dict[str, _T | None]:
    """Attributes an instance gets outside the class body: class
    annotations, a session's ``_COMPOSITES`` table, ``self.<name> = ...``
    in any method, and dataclass fields."""
    attrs: dict[str, _T | None] = {}
    for klass in reversed(cls.__mro__):
        if not _ours(klass):
            continue
        module = sys.modules[klass.__module__]
        ns = vars(module)
        for name, ann in inspect.get_annotations(klass).items():
            attrs[name] = _annotation_type(ann, ns)
        for name, mod_path, cls_name, *_ in vars(klass).get("_COMPOSITES", ()):
            try:
                comp = getattr(
                    importlib.import_module(mod_path, module.__package__),
                    cls_name,
                )
            except (ImportError, AttributeError):
                comp = None
            attrs[name] = _T("inst", comp) if isinstance(comp, type) else None
        try:
            # Parsed for structure, not linted: a stray `\ ` in a src
            # docstring must not warn here, or vanish under -W error.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(textwrap.dedent(inspect.getsource(klass)))
        except (OSError, TypeError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign):
                targets, value = [node.target], node.value
            else:
                continue
            for tgt in targets:
                if (isinstance(tgt, ast.Attribute)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "self"):
                    t = None
                    if (isinstance(value, ast.Call)
                            and isinstance(value.func, ast.Name)):
                        c = _class_named(value.func.id, ns)
                        t = _T("inst", c) if c is not None and _ours(c) else None
                    attrs.setdefault(tgt.attr, t)
    if dataclasses.is_dataclass(cls):
        ns = vars(sys.modules[cls.__module__])
        for f in dataclasses.fields(cls):
            attrs.setdefault(f.name, _annotation_type(f.type, ns))
    return attrs


def _member(cls: type, attr: str) -> tuple[bool, _T | None]:
    """``(exists, type)`` of ``<instance of cls>.<attr>``, found statically."""
    raw = inspect.getattr_static(cls, attr, _MISSING)
    if raw is _MISSING:
        attrs = _instance_attrs(cls)
        if attr in attrs:
            return True, attrs[attr]
        # A ``__getattr__`` answers names no static scan can see.
        return any("__getattr__" in vars(k) for k in cls.__mro__), None
    if isinstance(raw, (classmethod, staticmethod)):
        raw = raw.__func__
    if isinstance(raw, property):
        return True, _returns(raw.fget) if raw.fget else None
    if isinstance(raw, functools.cached_property):
        return True, _returns(raw.func)
    if isinstance(raw, type):
        return True, _T("class", raw) if _ours(raw) else _FOREIGN
    if inspect.isfunction(raw):
        return True, _T("func", raw)
    return True, None


# ── The page walker ─────────────────────────────────────────────────

class _Page:
    """Walks one page's fences in order, binding names as a reader's
    session would, then its inline spans against those bindings."""

    def __init__(self, where: str) -> None:
        self.where = where
        self.env: dict[str, _T] = {}
        self.history: dict[str, set[tuple[str, int]]] = {}
        self.returns: dict[str, _T | None] = {}
        self.checked: set[str] = set()
        self.problems: list[str] = []
        self._line = 1
        self._fn: list[str] = []

    def _flag(self, node: object, msg: str) -> None:
        entry = f"  {self.where}:{self._line + getattr(node, 'lineno', 1) - 1}: {msg}"
        if entry not in self.problems:
            self.problems.append(entry)

    def _bind(self, name: str, t: _T | None) -> None:
        if t is None:
            self.env.pop(name, None)
            return
        self.env[name] = t
        self.history.setdefault(name, set()).add((t.kind, id(t.obj)))

    def _assign(self, target: ast.expr, t: _T | None) -> None:
        if isinstance(target, ast.Name):
            self._bind(target.id, t)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._assign(elt, None)
        elif isinstance(target, ast.Starred):
            self._assign(target.value, None)
        elif isinstance(target, (ast.Attribute, ast.Subscript)):
            self._eval(target.value)

    # -- entry points
    def fence(self, body: str, first_line: int) -> None:
        self._line = first_line
        try:
            tree = ast.parse(body)
        except SyntaxError as exc:
            self._flag(SimpleNamespace(lineno=exc.lineno or 1),
                       f"python fence does not parse: {exc.msg}")
            return
        for stmt in tree.body:
            self._stmt(stmt)

    def inline(self, span: str, line: int) -> None:
        try:
            expr = ast.parse(span, mode="eval").body
        except SyntaxError:
            return
        if not any(isinstance(n, (ast.Attribute, ast.Call)) for n in ast.walk(expr)):
            return
        if isinstance(expr, ast.Attribute) and expr.attr in _FILE_SUFFIXES:
            return
        # A name the fences bound to two different things is ambiguous in
        # prose; leave it unjudged. A prose `apeGmsh.x.y` is a module path
        # even where the fences bound `apeGmsh` to the session class.
        env = self.env
        self.env = {k: v for k, v in env.items() if len(self.history[k]) == 1}
        for name, candidates in _CONVENTIONAL.items():
            if name not in self.history:
                classes = tuple(
                    getattr(importlib.import_module(m), c) for m, c in candidates
                )
                self.env[name] = (_T("inst", classes[0]) if len(classes) == 1
                                  else _T("union", classes))
        self.env["apeGmsh"] = _T("module", importlib.import_module("apeGmsh"))
        self._line = line
        self._eval(expr)
        self.env = env

    # -- statements
    def _stmt(self, node: ast.stmt) -> None:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            self._import(node)
        elif isinstance(node, ast.Assign):
            t = self._eval(node.value)
            for target in node.targets:
                self._assign(target, t)
        elif isinstance(node, ast.AnnAssign):
            self._assign(node.target, self._eval(node.value))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                t = self._enter(self._eval(item.context_expr))
                if item.optional_vars is not None:
                    self._assign(item.optional_vars, t)
            for stmt in node.body:
                self._stmt(stmt)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            self._eval(node.iter)
            self._assign(node.target, None)
            for stmt in node.body + node.orelse:
                self._stmt(stmt)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._function(node)
        elif isinstance(node, ast.Return):
            t = self._eval(node.value)
            if self._fn:
                self.returns[self._fn[-1]] = t
        elif not isinstance(node, ast.ClassDef):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.stmt):
                    self._stmt(child)
                elif isinstance(child, ast.expr):
                    self._eval(child)
                elif isinstance(child, ast.excepthandler):
                    for stmt in child.body:
                        self._stmt(stmt)

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        a = node.args
        params = [p.arg for p in a.posonlyargs + a.args + a.kwonlyargs]
        params += [p.arg for p in (a.vararg, a.kwarg) if p is not None]
        shadowed = {p: self.env.pop(p) for p in params if p in self.env}
        self._fn.append(node.name)
        self.returns[node.name] = None
        for stmt in node.body:
            self._stmt(stmt)
        self._fn.pop()
        self.env.update(shadowed)
        self._bind(node.name, _T("page", node.name))

    def _import(self, node: ast.Import | ast.ImportFrom) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top != "apeGmsh":
                    self._bind(alias.asname or top, _FOREIGN)
                elif (mod := self._module(alias.name, node)) is not None:
                    bound = mod if alias.asname else sys.modules["apeGmsh"]
                    self._bind(alias.asname or top, _T("module", bound))
            return
        if node.level or (node.module or "").split(".")[0] != "apeGmsh":
            for alias in node.names:
                self._bind(alias.asname or alias.name, _FOREIGN)
            return
        mod = self._module(node.module, node)
        for alias in node.names:
            if alias.name != "*":
                t = None if mod is None else self._attr(_T("module", mod), alias.name, node)
                self._bind(alias.asname or alias.name, t)

    def _module(self, name: str, node: object):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if (exc.name or "").startswith("apeGmsh"):
                self._flag(node, f"no module `{exc.name}`")
            return None

    # -- expressions
    def _eval(self, node: ast.expr | None) -> _T | None:
        if node is None:
            return None
        if isinstance(node, ast.Name):
            return self.env.get(node.id)
        if isinstance(node, ast.Attribute):
            return self._attr(self._eval(node.value), node.attr, node)
        if isinstance(node, ast.Call):
            fn = self._eval(node.func)
            for arg in node.args:
                self._eval(arg)
            for kw in node.keywords:
                self._eval(kw.value)
            return self._call(fn, node)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            env = dict(self.env)
            for gen in node.generators:
                self._eval(gen.iter)
                self._assign(gen.target, None)
                for cond in gen.ifs:
                    self._eval(cond)
            parts = (node.key, node.value) if isinstance(node, ast.DictComp) else (node.elt,)
            for part in parts:
                self._eval(part)
            self.env = env
            return None
        if not isinstance(node, ast.Lambda):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr):
                    self._eval(child)
        return None

    def _attr(self, base: _T | None, attr: str, node: object) -> _T | None:
        if base is None or base.kind not in ("module", "class", "inst", "union"):
            return None
        if base.kind == "union":
            hits = [(c, t) for c in base.obj for ok, t in [_member(c, attr)] if ok]
            if not hits:
                names = " / ".join(f"`{c.__name__}`" for c in base.obj)
                self._flag(node, f"neither of {names} has attribute `{attr}`")
                return None
            if len(hits) > 1:
                return None
            self.checked.add(f"{hits[0][0].__name__}.{attr}")
            return hits[0][1]
        if base.kind == "module":
            mod = base.obj
            try:
                obj = getattr(mod, attr)
            except AttributeError:
                try:
                    obj = importlib.import_module(f"{mod.__name__}.{attr}")
                except ModuleNotFoundError as exc:
                    if (exc.name or "").startswith(mod.__name__):
                        self._flag(node, f"module `{mod.__name__}` has no `{attr}`")
                    return None
            self.checked.add(f"{mod.__name__}.{attr}")
            return _wrap(obj)
        cls = base.obj
        exists, t = _member(cls, attr)
        if not exists:
            self._flag(node, f"`{cls.__name__}` has no attribute `{attr}`")
            return None
        self.checked.add(f"{cls.__name__}.{attr}")
        return t

    def _call(self, fn: _T | None, node: ast.Call) -> _T | None:
        if fn is None or fn.kind not in ("class", "func", "page"):
            return None
        if fn.kind == "page":
            return self.returns.get(fn.obj)
        target = fn.obj
        try:
            params = inspect.signature(target).parameters
        except (TypeError, ValueError):
            params = None
        if params is not None and not any(
            p.kind is p.VAR_KEYWORD for p in params.values()
        ):
            for kw in node.keywords:
                p = params.get(kw.arg) if kw.arg else None
                if kw.arg and (p is None or p.kind in (p.POSITIONAL_ONLY, p.VAR_POSITIONAL)):
                    self._flag(node, f"`{target.__qualname__}()` takes no keyword `{kw.arg}=`")
        return _T("inst", target) if fn.kind == "class" else _returns(target)

    def _enter(self, t: _T | None) -> _T | None:
        """What ``with <t> as x`` binds ``x`` to: ``__enter__``'s annotated
        return, else the manager itself (the common ``return self``). A
        base class that annotates ``return self`` with its own name
        (``_SessionBase.__enter__``) must not erase the subclass."""
        if t is None or t.kind != "inst":
            return None
        raw = inspect.getattr_static(t.obj, "__enter__", None)
        if not inspect.isfunction(raw):
            return None
        entered = _returns(raw)
        if entered is None or issubclass(t.obj, entered.obj):  # type: ignore[arg-type]
            return t
        return entered


def _scan(text: str, where: str) -> _Page:
    page = _Page(where)
    for line, body in _python_fences(text):
        page.fence(body, line)
    for line, span in _inline_spans(text):
        page.inline(span, line)
    return page


# ── Corpus check ────────────────────────────────────────────────────

@pytest.mark.parametrize("rel", _PAGES)
def test_results_page_api_resolves(rel: str) -> None:
    path = DOCS / rel
    assert path.is_file(), f"{rel} not found — update _PAGES if it moved."
    page = _scan(path.read_text(encoding="utf-8"), rel)
    assert len(page.checked) >= _MIN_CHECKED, (
        f"{rel}: only {sorted(page.checked)} resolved — its fences no longer "
        "bind the names they use, so the lane is guarding (almost) nothing."
    )
    if page.problems:
        raise AssertionError(
            "Results-API docs drift — these calls do not exist on the live "
            "code (fix the PAGE; the code is the truth):\n"
            + "\n".join(page.problems)
        )


# ── Positive controls — the resolver must actually resolve ──────────

# The grid's cells as choose-results-strategy.md shipped them at 07f757e0
# (cell prose trimmed): its fences bound neither `spec` nor `ops`.
_INCIDENT_GRID = """\
| **Run: in-process** (notebook) | **`spec.capture(...)`** — the default. \
| **`spec.emit_recorders("out/")`** — classic recorders. \
| **`spec.emit_mpco("run.mpco")`** — MPCO recorder in-process. |
| **Run: export** (cluster / external) | — \
| **`ops.tcl(..., recorders=spec)`** / **`ops.py(...)`** — emit a deck. \
| **`ops.tcl(..., recorders=spec, mpco=True)`** — one `recorder mpco` line. |
"""

# The same phantoms as results-mpco.md shipped them: bound in a fence.
_INCIDENT_FENCE = """\
```python
from apeGmsh.opensees import apeSees
from apeGmsh.results.spec import ResolvedRecorderSpec

ops_bridge = apeSees(fem)
spec = ResolvedRecorderSpec(fem_snapshot_id="0")
ops_bridge.tcl("model.tcl", recorders=spec, mpco=True)
```

Use native domain capture (`spec.capture(...)`) instead.
"""


def test_control_incident_grid_is_flagged() -> None:
    problems = "\n".join(_scan(_INCIDENT_GRID, "incident.md").problems)
    for expected in (
        "neither of `DomainCaptureSpec` / `ResolvedRecorderSpec` has "
        "attribute `capture`",
        "`apeSees.tcl()` takes no keyword `recorders=`",
        "`apeSees.tcl()` takes no keyword `mpco=`",
    ):
        assert expected in problems, f"{expected!r} not flagged in:\n{problems}"
    # The cells that were right must stay quiet.
    assert "emit_recorders" not in problems and "emit_mpco" not in problems
    assert "apeSees.py" not in problems


def test_control_incident_fence_is_flagged() -> None:
    problems = "\n".join(_scan(_INCIDENT_FENCE, "incident.md").problems)
    for expected in (
        "`ResolvedRecorderSpec` has no attribute `capture`",
        "`apeSees.tcl()` takes no keyword `recorders=`",
        "`apeSees.tcl()` takes no keyword `mpco=`",
    ):
        assert expected in problems, f"{expected!r} not flagged in:\n{problems}"


def test_control_live_chains_resolve() -> None:
    page = _scan(textwrap.dedent("""\
        ```python
        from apeGmsh.opensees import apeSees
        from apeGmsh.results.capture import DomainCaptureSpec

        def build():
            ops = apeSees(fem)
            return ops

        ops = build()
        ops.recorder.MPCO(file="run.mpco", nodal_responses=("displacement",))
        ops.tcl("model.tcl", analyze_steps=10)
        spec = DomainCaptureSpec(opensees=ops)
        with ops.domain_capture(spec, path="run.h5") as cap:
            cap.capture_modes(2)
        ```
        """), "control.md")
    assert page.problems == [], page.problems
    # Across a page-defined helper, an attribute set in ``__init__``
    # (``self.recorder``), an annotated return (``-> DomainCapture``) and
    # a ``with ... as`` target.
    assert {
        "apeSees.recorder", "_RecorderNS.MPCO", "apeSees.tcl",
        "apeSees.domain_capture", "DomainCapture.capture_modes",
    } <= page.checked, sorted(page.checked)


def test_control_unknown_names_are_skipped_not_guessed() -> None:
    page = _scan(
        "```python\nimport numpy as np\nnp.not_a_numpy_name()\n"
        "fem.whatever.at_all(x=1)\n```\n\n`opspy.recorder('mpco', ...)`\n",
        "skip.md",
    )
    assert page.problems == [] and page.checked == set()
