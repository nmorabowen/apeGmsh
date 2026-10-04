"""Quirk lint: lessons this repo already paid for, held by a check.

A lesson becomes a rule here only when it names a pattern a machine can
see *and* it bit again after it was written down (AGENTS.md, "Build and
test"). Every other lesson stays a line in a task guide. Each rule below
names its lesson and the incidents that earned it. A rule that cannot read
a case stays silent: skip, never guess.

Waive one site of a Python rule with a comment on the flagged statement or
in the comment block directly above it:

    # apegmsh-lint: <rule>-ok <reason>

The reason is mandatory, and a waiver that no longer suppresses anything
is itself a finding. The repo-level rules `adr-number`, `arch-path` and `doc-path` have no
waiver: a collision is never right, and a dead citation is fixed in the doc.

    python scripts/check_quirks.py              # this checkout
    python scripts/check_quirks.py --root DIR   # another tree, e.g. a `git archive`

`getattr-private` and `getattr-undefined` also read a ratchet baseline
(`scripts/quirks_getattr_baseline.txt`, `path::name` per line): it may only
shrink, and a line matching no site is a finding.

Stdlib only, a few seconds. `tests/test_check_quirks.py` holds one case per
shape each rule must flag or pass; CI runs this scan as the last step of
`static-gates`.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import gc
import io
import re
import sys
import tokenize
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DECISIONS = Path("architecture/decisions")
FEMDATA = Path("src/apeGmsh/mesh/FEMData.py")

#: The rebuilds that must carry a whole model: compose, and the model.h5
#: round-trip. Foreign-format readers (MPCO, .ladruno) legitimately build
#: partial composites and are out of scope.
CARRY_ALL = (Path("src/apeGmsh/mesh/_compose.py"), Path("src/apeGmsh/mesh/_femdata_h5_io.py"))
COMPOSITES = ("ElementComposite", "NodeComposite")

#: The one file allowed to spell schema versions as literals.
SCHEMA_FIXTURE = Path("tests/fixtures/schema.py")
VERSION = re.compile(r"^\d+\.\d+\.\d+$")

#: The name-resolution code, where the contract is "fail loud". Elsewhere a
#: broad `except` is usually deliberate (gmsh raises bare `Exception`).
SWALLOW_SCOPE = (Path("src/apeGmsh/_kernel/resolvers"), Path("src/apeGmsh/mesh/_fem_factory.py"))
LOG_METHODS = {"debug", "info", "warning", "warn", "error", "exception", "log", "critical", "fatal"}
LOG_FUNCS = {"warn", "print"}
EMPTY_CTORS = {"set", "frozenset", "dict", "list", "tuple"}
EMPTY_ARRAYS = {"array", "asarray", "empty", "zeros"}

#: Code that runs in the bridge's process: the package, and the examples users
#: copy. Tests stay out: they check emitted py decks, which bind openseespy by design.
IMPORT_SCOPE = (Path("src/apeGmsh"), Path("examples"))
#: The backend resolver, the one module that may import openseespy by name.
RESOLVER = Path("src/apeGmsh/opensees/emitter/live.py")
DYNAMIC_IMPORTS = {"import_module", "__import__"}
#: Text that could hold such an import: a superset of what the AST check flags,
#: so a file without it is skipped unparsed. Prose that quotes the import in
#: backticks does not match; backslash-continued lines are joined first.
IMPORT_TEXT = re.compile(
    r"(?<![`\w.])import\s+[\w., ]*?\bopenseespy\b"
    r"|(?<![`\w.])from\s+openseespy\b[\w.]*\s+import\b"
    r"|\b(?:import_module|__import__)\(\s*[rbu]?['\"]openseespy\b"
)

#: The agent-facing docs whose citations must resolve: AGENTS.md, the task
#: guides (not the derived user-skill mirror) and the non-ADR architecture
#: docs. ADRs are append-only history and stay out.
AGENTS = Path("AGENTS.md")
SKILLS = Path(".claude/skills")
DERIVED_SKILL = "apegmsh-helper"
ARCHITECTURE = Path("architecture")
#: Where the architecture docs lived until N3 (#1197): nothing may be tracked there.
#: (Built from parts so a grep for the old path finds only stale citations.)
OLD_ARCHITECTURE = Path("src/apeGmsh/opensees") / "architecture"
#: Where a backticked path may be rooted, after the doc's own folder: the
#: repo, `src`, the package, the bridge and the architecture folder are the
#: shorthands the docs use (`mesh/FEMData.py`, `emitter/h5.py`, `decisions/README.md`).
#: A Markdown link resolves only against the doc's folder, as a renderer does.
DOC_BASES = (Path("."), Path("src"), Path("src/apeGmsh"), Path("src/apeGmsh/opensees"), ARCHITECTURE)
DOC_EXTENSIONS = r"(?:py|md|json|toml|yml|yaml)"
#: `x/y.py`, `x/y.md`, ... in backticks, with whatever follows the path up to the
#: closing backtick (SUFFIX reads it). A slash is required: a bare file name is
#: not a citation the rule can place.
CITATION = re.compile(r"`(?P<path>[^`\s]*/[^`\s]*?\." + DOC_EXTENSIONS + r")(?P<suffix>[^`]*)`")
#: What may follow a cited path: nothing, `:line`, `:first-last`, `#Lnn`, or
#: `::symbol` (optionally `symbol()`, a `Class.member`, a glob, or several
#: separated by ` / `). Every symbol must be defined in that file; anything
#: else is unreadable and a finding, so no citation escapes the check.
SYMBOL = r"[\w.*]+(?:\(\))?"
SUFFIX = re.compile(
    r"^(?::\d+(?:-\d+)?|#L\d+(?:-L?\d+)?|::(?P<symbols>" + SYMBOL + r"(?:\s*/\s*" + SYMBOL + r")*))?$"
)
MD_LINK = re.compile(r"\]\((?P<path>[^)#\s]+?\." + DOC_EXTENSIONS + r")(?:#[^)]*)?\)")
#: Not a repo path: a URL, a home or Windows path, a placeholder or a glob.
NOT_A_PATH = re.compile(r"^(?:https?:|~|[A-Za-z]:[\\/])|[*{}<>\[\]$]")

#: The package the getattr rules read and index, and the ratchet of sites that
#: predate them (one `path::name` per line, `#` comments; a stale line is a finding).
GETATTR_SCOPE = "src/apeGmsh/"
GETATTR_BASELINE = Path("scripts/quirks_getattr_baseline.txt")

WAIVER = re.compile(r"#\s*apegmsh-lint:\s*(?P<rule>[a-z-]+?)-ok\b(?P<reason>.*)$")

RULES: dict[str, str] = {
    "adr-number": (
        "two ADRs share a number, an ADR file has no row in decisions/README.md, or the README "
        "is stale against the ADRs' Status lines (it is generated by scripts/adr_index.py). "
        "List the directory on origin/main before numbering, and add the index row in "
        "the same PR: an unindexed ADR makes its number look free. Incidents: a second "
        "0065 (#676, fixed #677), renumbers 0072->0073 (#741) and 0074->0079 (#817)"
    ),
    "schema-literal": (
        "a schema version compared against a hard-coded literal goes stale at the next "
        "bump and turns main red. Import NEUTRAL_CURRENT / OPENSEES_CURRENT, the "
        "*_FLOOR constants (the oldest minor each reader opens, ADR 0113) or "
        "*_PRIOR_MINOR from tests/fixtures/schema.py. Incidents: stale at the "
        "2.12.0 and 2.13.0 bumps (fixed 60252205, #642), and at 2.16.0 (fixed #738)"
    ),
    "compose-streams": (
        "a rebuild that must carry the whole model omits a stream the composite "
        "accepts, so that stream is silently dropped. Pass every parameter of "
        "{cls}.__init__. Lesson: #707 (compose must carry every FEMData stream); "
        "recurred for embed_ties, contacts and contact_planes (fixed #912/#913)"
    ),
    "resolve-swallow": (
        "name resolution swallows an error: this handler only passes, continues, "
        "returns or assigns an empty value, or logs, so a missing target becomes a "
        "silent no-op. Raise (or re-raise on the strict path). Lesson: _fem_factory "
        "downgraded every resolve error to a warning (fixed 3aecb417); re-added nine "
        "days later in the chain-phase router (06ccd266), where a tie against a "
        "destroyed physical group was silently dropped (fixed 45340ac3)"
    ),
    "openseespy-import": (
        "imports openseespy by name. Beside a fork build (on PYTHONPATH, or loaded from "
        "APEGMSH_OPENSEES_BIN) that binds a second module with its own, empty domain, not "
        "the one the bridge built the model in. Take an explicit ops= first, else call "
        "apeGmsh.opensees.emitter.live.get_ops(). Lesson: DomainCapture sampled an empty "
        "domain (fixed 9ffe6aa2, which kept the import as its fallback); that fallback, "
        "LiveMPCO, LiveRecorders, interop.solve and the arch-pushover example still bound it"
    ),
    "doc-path": (
        "an agent-facing doc cites a path or symbol that does not exist, so the reader "
        "is sent to a file that moved or a name that was renamed and reads the doc as "
        "current anyway. Cite the current path (from the doc's folder, the repo root, "
        "src, src/apeGmsh, src/apeGmsh/opensees or the architecture folder), or drop the "
        "backticks from a reference that names nothing in this repo. Lesson: docs lag the "
        "source (AGENTS.md, 'What this repo is'); the 2026-09-28 panel found 11% of the "
        "paths these docs cite dead (#1192 P6, #1197 N1)"
    ),
    "arch-path": (
        "a file sits under the old architecture folder, src/apeGmsh/opensees/" "architecture/: "
        "architecture docs live in `architecture/` since N3 (#1197), so put the file there. "
        "Lesson: the doc tree (2.8 MB of Markdown) shipped inside the package. No waiver is "
        "offered: there is no legitimate file at the old path"
    ),
    "getattr-private": (
        "getattr/hasattr with a literal _private name on an object that is not self or cls, "
        "and no module of this top-level subpackage defines that name: it reaches across a "
        "package boundary into another package's internals and quietly falls back to the "
        "default when that package renames it. Use a public accessor with a contract test. "
        "Lesson: assessment 'Fail closed at seams'; results/capture/spec.py read "
        "bridge._primitives this way (98- and 138-day capture outages)"
    ),
    "getattr-undefined": (
        "getattr/hasattr with a literal attribute name that nothing under src/apeGmsh defines "
        "(no def, class, assignment, attribute store, import, setattr or __slots__ entry): the "
        "probe can only ever take its default, or the name lives on a foreign object and "
        "should be typed. Lesson: 14445604 deleted the legacy bridge's _sec_tags while "
        "getattr(self._opensees, '_sec_tags', {}) stayed, and the recorder capture "
        "silently lost its section tags for 138 days (fixed 7c7c1541); the node_ndf vestige "
        "is the same shape. Ratchet: scripts/quirks_getattr_baseline.txt"
    ),
}


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: [{self.rule}] {self.message}"


# --- arch-path ---------------------------------------------------------------


def check_arch_path(root: Path) -> list[Finding]:
    folder = root / OLD_ARCHITECTURE
    if not folder.is_dir():
        return []
    return [
        Finding(p.relative_to(root).as_posix(), 1, "arch-path", RULES["arch-path"])
        for p in sorted(folder.rglob("*")) if p.is_file()
    ]


# --- adr-number -------------------------------------------------------------


def _adr_index():
    """Load scripts/adr_index.py by path: this file may itself be loaded by path."""
    import importlib.util

    path = Path(__file__).resolve().with_name("adr_index.py")
    spec = importlib.util.spec_from_file_location("adr_index", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check_adr_numbers(root: Path) -> list[Finding]:
    folder = root / DECISIONS
    if not folder.is_dir():
        return []
    files = sorted(p.name for p in folder.glob("[0-9][0-9][0-9][0-9]-*.md"))
    rel = DECISIONS.as_posix()
    findings: list[Finding] = []
    counts = Counter(name[:4] for name in files)
    for number, count in sorted(counts.items()):
        if count > 1:
            twins = ", ".join(name for name in files if name.startswith(number))
            findings.append(
                Finding(rel, 0, "adr-number", f"{count} ADRs numbered {number}: {twins}. "
                        + RULES["adr-number"])
            )
    readme = folder / "README.md"
    if readme.is_file():
        index = readme.read_text(encoding="utf-8")
        unindexed = [name for name in files if f"({name})" not in index]
        for name in unindexed:
            findings.append(
                Finding(f"{rel}/README.md", 0, "adr-number",
                        f"no index row links {name}. " + RULES["adr-number"])
            )
        if not unindexed and not findings:
            # The README is generated from each ADR's title and Status line.
            adr_index = _adr_index()
            try:
                stale = adr_index.is_stale(root)
                detail = "stale: run `python scripts/adr_index.py`"
            except ValueError as err:
                stale, detail = True, f"cannot be generated ({err})"
            if stale:
                findings.append(
                    Finding(f"{rel}/README.md", 0, "adr-number",
                            f"the index is {detail}. " + RULES["adr-number"])
                )
    return findings


# --- doc-path ---------------------------------------------------------------


def _agent_docs(root: Path) -> list[Path]:
    docs = [root / AGENTS] if (root / AGENTS).is_file() else []
    docs += sorted(
        p for p in (root / SKILLS).glob("apegmsh-*/SKILL.md") if p.parent.name != DERIVED_SKILL
    )
    docs += sorted(  # not decisions/: ADRs are history
        p for p in (root / ARCHITECTURE).glob("*.md")
    )
    return docs


_SCOPE_BLOCKS = (ast.If, ast.Try, ast.With, ast.For, ast.While)
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

#: A module's top-level names, and each top-level class's members.
Names = tuple[set[str], dict[str, set[str]]]


def _scope_statements(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """The statements that bind names in one scope: its own, and those inside its
    `if`/`try`/`with`/loops (a guarded import or def still binds there). Never
    the bodies of functions or classes, which are scopes of their own."""
    for stmt in body:
        yield stmt
        if isinstance(stmt, _SCOPE_BLOCKS):
            nested = [*stmt.body, *getattr(stmt, "orelse", [])]
            if isinstance(stmt, ast.Try):
                nested += [*stmt.finalbody, *(s for h in stmt.handlers for s in h.body)]
            yield from _scope_statements(nested)


def _bound_names(stmt: ast.stmt) -> set[str]:
    """The names one statement binds in its scope: a def, a class, an assignment
    (plain, annotated or tuple-unpacked) or an import, aliases included."""
    if isinstance(stmt, _DEFS):
        return {stmt.name}
    if isinstance(stmt, ast.Assign):
        return {n.id for t in stmt.targets for n in _target_names(t)}
    if isinstance(stmt, ast.AnnAssign):
        return {n.id for n in _target_names(stmt.target)}
    if isinstance(stmt, ast.Import):
        return {a.asname or a.name.split(".")[0] for a in stmt.names}
    if isinstance(stmt, ast.ImportFrom):
        return {a.asname or a.name for a in stmt.names}
    return set()


def _target_names(target: ast.expr) -> list[ast.Name]:
    if isinstance(target, ast.Name):
        return [target]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [n for elt in target.elts for n in _target_names(elt)]
    return []


def _class_members(cls: ast.ClassDef) -> set[str]:
    """Methods, class-level assignments and nested classes, plus every
    `self.<member> = ...` inside the class's own methods."""
    members: set[str] = set()
    for stmt in _scope_statements(cls.body):
        members |= _bound_names(stmt)
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(stmt):
                targets = node.targets if isinstance(node, ast.Assign) else (
                    [node.target] if isinstance(node, ast.AnnAssign) else [])
                for target in targets:
                    if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                            and target.value.id == "self"):
                        members.add(target.attr)
    return members


def _module_names(path: Path, cache: dict[Path, Names | None]) -> Names | None:
    """A module's top-level names and its classes' members; None if it does not parse."""
    if path not in cache:
        source = _read_source(path)
        try:
            tree = ast.parse(source or "", filename=str(path)) if source is not None else None
        except SyntaxError:
            tree = None
        names: Names | None = None
        if tree is not None:
            top: set[str] = set()
            classes: dict[str, set[str]] = {}
            for stmt in _scope_statements(tree.body):
                top |= _bound_names(stmt)
                if isinstance(stmt, ast.ClassDef):
                    classes[stmt.name] = _class_members(stmt)
            names = (top, classes)
        cache[path] = names
    return cache[path]


def _symbol_missing(symbol: str, names: Names) -> str | None:
    """Why `symbol` is not defined by a module with `names`; None if it is.

    One part is a top-level name; `Class.member` is a member of a top-level
    class; a `*` is a glob over the candidates. Deeper paths are not read."""
    top, classes = names
    parts = symbol.removesuffix("()").split(".")
    if len(parts) == 1:
        candidates, where = top, "at the top level"
    elif len(parts) == 2 and parts[0] in classes:
        candidates, where = classes[parts[0]], f"in class {parts[0]}"
    elif len(parts) == 2:
        return f"has no class `{parts[0]}`"
    else:
        return f"cannot be checked for `{symbol}`: cite a top-level name or `Class.member`"
    if not fnmatch.filter(candidates, parts[-1]):
        return f"defines no `{parts[-1]}` {where}"
    return None


def _resolve_citation(cited: str, doc: Path, root: Path, link: bool) -> Path | None:
    bases = [doc.parent] if link else [doc.parent, *(root / base for base in DOC_BASES)]
    for base in bases:
        candidate = base / cited
        if candidate.is_file():
            return candidate
    return None


def check_doc_paths(root: Path) -> list[Finding]:
    """Every backticked repo path and relative Markdown link in the agent-facing
    docs resolves, its suffix is one SUFFIX reads, and each `::symbol` is a
    top-level name or `Class.member` of that file (AST).

    A path with no slash, a URL, a placeholder or a glob is not a citation and
    passes. A `.py` file a symbol points into that does not parse is a finding.
    """
    findings: list[Finding] = []
    cache: dict[Path, Names | None] = {}
    for doc in _agent_docs(root):
        rel = doc.relative_to(root).as_posix()
        for number, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            cited = [(m.group("path"), m.group("suffix"), False) for m in CITATION.finditer(line)]
            cited += [(m.group("path"), "", True) for m in MD_LINK.finditer(line)]
            for path, suffix, link in cited:
                if NOT_A_PATH.search(path):
                    continue
                problems: list[str] = []
                target = _resolve_citation(path, doc, root, link)
                read = SUFFIX.match(suffix)
                if target is None:
                    problems.append(f"`{path}` does not resolve")
                if read is None:
                    problems.append(f"`{path}{suffix}`: unreadable suffix `{suffix}`; cite "
                                    "`:line`, `:first-last`, `#Lnn` or `::symbol`")
                elif target is not None and read.group("symbols") and target.suffix == ".py":
                    names = _module_names(target, cache)
                    for symbol in re.split(r"\s*/\s*", read.group("symbols")):
                        if names is None:
                            problems.append(f"`{path}` does not parse, so `::{symbol}` "
                                            "cannot be checked")
                        elif (why := _symbol_missing(symbol, names)) is not None:
                            problems.append(f"`{path}` {why}")
                findings += [Finding(rel, number, "doc-path", f"{why}. " + RULES["doc-path"])
                             for why in problems]
    return findings


# --- Python rules -----------------------------------------------------------


def _comments(text: str) -> dict[int, str]:
    found: dict[int, str] = {}
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.COMMENT:
                found[token.start[0]] = token.string
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return found


_COMPOUND = (
    ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.If, ast.For, ast.AsyncFor,
    ast.While, ast.With, ast.AsyncWith, ast.Try, ast.Match,
)


def _statement_spans(tree: ast.AST) -> list[tuple[int, int]]:
    return [
        (node.lineno, node.end_lineno or node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, ast.stmt) and not isinstance(node, _COMPOUND)
    ]


def _waivable_lines(line: int, spans: list[tuple[int, int]], lines: list[str]) -> range:
    """The innermost simple statement holding `line`, plus the comments above it."""
    holding = [span for span in spans if span[0] <= line <= span[1]]
    first, last = min(holding, key=lambda s: s[1] - s[0]) if holding else (line, line)
    while first > 1 and lines[first - 2].lstrip().startswith("#"):
        first -= 1
    return range(first, last + 1)


def _is_version(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and VERSION.match(node.value) is not None
    )


def check_schema_literal(tree: ast.AST, rel: str, root: Path) -> Iterator[tuple[int, str]]:
    """`<something naming schema_version> ==/!= "N.N.N"`, in tests only.

    Stamping an old or wrong version into a fixture (`attrs[...] = "2.6.0"`)
    is how the floor and major-refusal tests work, so assignments pass;
    only a comparison asserts what the *current* version is.
    """
    if not rel.startswith("tests/") or rel == SCHEMA_FIXTURE.as_posix():
        return
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        sides = [node.left, *node.comparators]
        for index, op in enumerate(node.ops):
            if not isinstance(op, (ast.Eq, ast.NotEq)):
                continue
            left, right = sides[index], sides[index + 1]
            for literal, other in ((left, right), (right, left)):
                if _is_version(literal) and "schema_version" in ast.unparse(other).lower():
                    yield node.lineno, RULES["schema-literal"]
                    break


def _init_params(root: Path) -> dict[str, list[str]]:
    """Each composite's `__init__` parameters, read from FEMData.py."""
    path = root / FEMDATA
    if not path.is_file():
        return {}
    params: dict[str, list[str]] = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ClassDef) and node.name in COMPOSITES:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    args = item.args
                    params[node.name] = [a.arg for a in args.args[1:] + args.kwonlyargs]
    return params


def check_compose_streams(tree: ast.AST, rel: str, root: Path) -> Iterator[tuple[int, str]]:
    if rel not in {path.as_posix() for path in CARRY_ALL}:
        return
    params = _init_params(root)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name not in params:
            continue
        if any(keyword.arg is None for keyword in node.keywords) or any(
            isinstance(arg, ast.Starred) for arg in node.args
        ):
            continue  # **kwargs / *args: unreadable, stay silent
        given = set(params[name][: len(node.args)]) | {k.arg for k in node.keywords}
        missing = [param for param in params[name] if param not in given]
        if missing:
            yield node.lineno, (
                f"{name}(...) omits {', '.join(missing)}. "
                + RULES["compose-streams"].format(cls=name)
            )


def _in_swallow_scope(rel: str) -> bool:
    return any(rel == p.as_posix() or rel.startswith(p.as_posix() + "/") for p in SWALLOW_SCOPE)


def _empty(node: ast.expr | None) -> bool:
    """None, a falsy literal, an empty container, or an empty-array call."""
    if node is None:
        return True
    if isinstance(node, ast.Constant):
        return not node.value
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return not node.elts
    if isinstance(node, ast.Dict):
        return not node.keys
    if isinstance(node, ast.Call):
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name in EMPTY_CTORS and isinstance(node.func, ast.Name):
            return not node.args and not node.keywords
        if name in EMPTY_ARRAYS and node.args:
            return _empty(node.args[0])
    return False


def _is_log_call(stmt: ast.stmt) -> bool:
    if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
        return False
    func = stmt.value.func
    if isinstance(func, ast.Attribute):
        return func.attr in LOG_METHODS
    return isinstance(func, ast.Name) and func.id in LOG_FUNCS


def _silent(body: list[ast.stmt]) -> bool:
    """Every statement provably swallows. Anything else (a raise, a real value,
    another call or assignment) is not provably silent, so the rule stays quiet."""
    for stmt in body:
        if isinstance(stmt, (ast.Pass, ast.Continue, ast.Break)):
            continue
        if isinstance(stmt, ast.Return) and _empty(stmt.value):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue
        if _is_log_call(stmt):
            continue
        if isinstance(stmt, ast.If) and _silent(stmt.body) and _silent(stmt.orelse):
            continue
        if (isinstance(stmt, ast.Assign) and _empty(stmt.value)
                and all(isinstance(t, ast.Name) for t in stmt.targets)):
            continue
        if (isinstance(stmt, ast.AnnAssign) and stmt.value is not None
                and _empty(stmt.value) and isinstance(stmt.target, ast.Name)):
            continue
        return False
    return True


def check_resolve_swallow(tree: ast.AST, rel: str, root: Path) -> Iterator[tuple[int, str]]:
    """Any handler, whatever it catches, whose body only swallows; any `suppress(...)`.

    No exemption by exception type or function name: the resolvers' own
    errors subclass broad ones, and a predicate-named copy of the router
    incident is still the incident. Legitimate sites carry a waiver.
    """
    if not _in_swallow_scope(rel):
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and _silent(node.body):
            yield node.lineno, RULES["resolve-swallow"]
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                ctx = item.context_expr
                if isinstance(ctx, ast.Call) and (
                    getattr(ctx.func, "id", None) or getattr(ctx.func, "attr", None)
                ) == "suppress":
                    yield node.lineno, RULES["resolve-swallow"]


def _in_import_scope(rel: str) -> bool:
    return any(rel.startswith(p.as_posix() + "/") for p in IMPORT_SCOPE)


def _imported_modules(node: ast.AST) -> list[str]:
    """The absolute modules an import statement, or a literal `import_module(...)`, binds."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        return [node.module] if node.level == 0 and node.module else []
    if (
        isinstance(node, ast.Call)
        and (getattr(node.func, "id", None) or getattr(node.func, "attr", None)) in DYNAMIC_IMPORTS
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        return [node.args[0].value]
    return []


def check_openseespy_import(tree: ast.AST, rel: str, root: Path) -> Iterator[tuple[int, str]]:
    """Any import of `openseespy` or a submodule in scope, outside the resolver.

    A string that only spells the import (a docstring, a line of an emitted py
    deck) binds nothing and passes, as does a `find_spec("openseespy")` probe.
    """
    if not _in_import_scope(rel) or rel == RESOLVER.as_posix():
        return
    for node in ast.walk(tree):
        if any(m == "openseespy" or m.startswith("openseespy.") for m in _imported_modules(node)):
            yield node.lineno, RULES["openseespy-import"]


# --- getattr-private / getattr-undefined --------------------------------------

_GETATTR_CALLS = {"getattr": (2, 3), "hasattr": (2, 2)}
_getattr_cache: dict[Path, dict[str, set[str]]] = {}
_baseline_seen: dict[Path, set[str]] = {}
_baseline_cache: dict[Path, dict[str, int]] = {}
_probe_cache: dict[tuple[Path, str], list[ast.Call]] = {}
_tree_cache: dict[tuple[Path, str], ast.AST] = {}
_sites_cache: dict[tuple[Path, str], list[tuple[str, int, str]]] = {}


def _subpackage(rel: str) -> str:
    """Top-level apeGmsh subpackage of a path under src/apeGmsh/ ('.' for a top-level module)."""
    parts = rel.removeprefix(GETATTR_SCOPE).split("/")
    return parts[0] if len(parts) > 1 else "."


def _literal_attr(node: ast.Call, arity: tuple[int, int]) -> str | None:
    if not arity[0] <= len(node.args) <= arity[1] or node.keywords:
        return None
    name = node.args[1]
    return name.value if isinstance(name, ast.Constant) and isinstance(name.value, str) else None


def _defined_names(tree: ast.AST, probes: list[ast.Call]) -> Iterator[str]:
    """Every attribute name this module can give an object: def, class, assignment,
    attribute store, import alias, setattr literal, __slots__ entry. One walk also
    gathers the module's getattr/hasattr calls into `probes`."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.name
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            yield node.id
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            yield node.attr
        elif isinstance(node, ast.alias):
            yield (node.asname or node.name).split(".")[0]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "setattr":
                name = _literal_attr(node, (3, 3))
                if name is not None:
                    yield name
            elif node.func.id in _GETATTR_CALLS:
                probes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__slots__" for t in node.targets
        ):
            for const in ast.walk(node.value):
                if isinstance(const, ast.Constant) and isinstance(const.value, str):
                    yield const.value


def _getattr_index(root: Path) -> dict[str, set[str]]:
    """name -> the top-level subpackages that define it, over all of src/apeGmsh."""
    if root not in _getattr_cache:
        index: dict[str, set[str]] = {}
        base = root / GETATTR_SCOPE
        for path in sorted(base.rglob("*.py")) if base.is_dir() else []:
            text = _read_source(path)
            if text is None:
                continue
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            rel = path.relative_to(root).as_posix()
            probes: list[ast.Call] = []
            for name in _defined_names(tree, probes):
                index.setdefault(name, set()).add(_subpackage(rel))
            _probe_cache[(root, rel)] = probes
            _tree_cache[(root, rel)] = tree  # scan_file reuses it, the parse is the cost
        _getattr_cache[root] = index
    return _getattr_cache[root]


def _baseline_keys(root: Path) -> dict[str, int]:
    """`path::name` -> its line in the baseline file."""
    if root in _baseline_cache:
        return _baseline_cache[root]
    path = root / GETATTR_BASELINE
    if not path.is_file():
        return _baseline_cache.setdefault(root, {})
    keys: dict[str, int] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        key = raw.split("#", 1)[0].strip()
        if key:
            keys.setdefault(key, number)
    return _baseline_cache.setdefault(root, keys)


def _getattr_sites(tree: ast.AST, rel: str, root: Path, rule: str) -> Iterator[tuple[int, str]]:
    if not rel.startswith(GETATTR_SCOPE):
        return
    cached = _sites_cache.get((root, rel))
    if cached is None:  # both rules read the probes the index pass gathered
        _getattr_index(root)
        cached = _sites_cache[(root, rel)] = list(_all_getattr_sites(tree, rel, root))
    for found, line, message in cached:
        if found == rule:
            yield line, message


def _all_getattr_sites(tree: ast.AST, rel: str, root: Path) -> Iterator[tuple[str, int, str]]:
    index, package, baseline = _getattr_index(root), _subpackage(rel), _baseline_keys(root)
    for node in _probe_cache.get((root, rel), []):
        assert isinstance(node.func, ast.Name)
        name = _literal_attr(node, _GETATTR_CALLS[node.func.id])
        if name is None or (name.startswith("__") and name.endswith("__")):
            continue  # dunders belong to Python, not to apeGmsh
        receiver = node.args[0]
        if name not in index:
            found = "getattr-undefined"
            message = f"{node.func.id}(..., {name!r}): nothing in src/apeGmsh defines {name!r}"
        elif (
            name.startswith("_")
            and not (isinstance(receiver, ast.Name) and receiver.id in {"self", "cls"})
            and package not in index[name]
        ):
            found = "getattr-private"
            message = (
                f"{node.func.id}({ast.unparse(receiver)}, {name!r}): private name defined only in "
                f"{', '.join(sorted(index[name]))}, read from {package}"
            )
        else:
            continue
        key = f"{rel}::{name}"
        if key in baseline:
            _baseline_seen.setdefault(root, set()).add(key)
            continue
        yield found, node.lineno, f"{message} [baseline key {key}]"


def check_getattr_private(tree: ast.AST, rel: str, root: Path) -> Iterator[tuple[int, str]]:
    yield from _getattr_sites(tree, rel, root, "getattr-private")


def check_getattr_undefined(tree: ast.AST, rel: str, root: Path) -> Iterator[tuple[int, str]]:
    yield from _getattr_sites(tree, rel, root, "getattr-undefined")


def _stale_baseline(root: Path) -> list[Finding]:
    """A baseline line no site matches any more: the ratchet only tightens, so delete it."""
    seen = _baseline_seen.get(root, set())
    return [
        Finding(GETATTR_BASELINE.as_posix(), number, "getattr-baseline",
                f"{key} matches no getattr/hasattr site any more; delete the line")
        for key, number in _baseline_keys(root).items() if key not in seen
    ]


PYTHON_RULES: dict[str, Callable[[ast.AST, str, Path], Iterator[tuple[int, str]]]] = {
    "schema-literal": check_schema_literal,
    "compose-streams": check_compose_streams,
    "resolve-swallow": check_resolve_swallow,
    "openseespy-import": check_openseespy_import,
    "getattr-private": check_getattr_private,
    "getattr-undefined": check_getattr_undefined,
}


def _read_source(path: Path) -> str | None:
    """Decode a source file the way Python does (BOM, coding cookie); None if it can't be."""
    data = path.read_bytes()
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
        return data.decode(encoding)
    except (SyntaxError, UnicodeDecodeError, LookupError):
        return None


def _may_apply(rel: str, lowered: str) -> bool:
    """Whether a rule could flag anything in this file, judged on its text alone."""
    if rel.startswith("tests/"):
        return "schema_version" in lowered
    if rel in {path.as_posix() for path in CARRY_ALL} or _in_swallow_scope(rel):
        return True
    if rel.startswith(GETATTR_SCOPE) and ("getattr(" in lowered or "hasattr(" in lowered):
        return True
    return "openseespy" in lowered and IMPORT_TEXT.search(re.sub(r"\\\r?\n", " ", lowered)) is not None


def scan_file(path: Path, rel: str, root: Path) -> list[Finding]:
    text = _read_source(path)
    if text is None:
        return []  # not valid Python source; like a SyntaxError, ruff and pytest will say so
    lowered = text.lower()
    if "apegmsh-lint" not in lowered and not _may_apply(rel, lowered):
        return []  # nothing a rule reads here; skipping the parse keeps the scan fast
    tree = _tree_cache.pop((root, rel), None)
    if tree is None:
        try:
            tree = ast.parse(text, filename=rel)
        except SyntaxError:
            return []  # ruff and pytest will say so; this lint only reads code
    lines = text.splitlines()
    spans = _statement_spans(tree)

    findings: list[Finding] = []
    waivers: dict[int, str] = {}
    for line, comment in (_comments(text) if "apegmsh-lint" in lowered else {}).items():
        match = WAIVER.search(comment)
        if match is None:
            continue
        rule, reason = match.group("rule"), match.group("reason").strip(" —-:")
        if rule not in PYTHON_RULES:
            findings.append(Finding(rel, line, "waiver", f"no waivable rule {rule!r}"))
        elif not reason:
            findings.append(Finding(rel, line, "waiver", f"`{rule}-ok` needs a reason"))
        else:
            waivers[line] = rule

    used: set[int] = set()
    for rule, check in PYTHON_RULES.items():
        for line, message in check(tree, rel, root):
            hit = next(
                (at for at in _waivable_lines(line, spans, lines) if waivers.get(at) == rule),
                None,
            )
            if hit is None:
                findings.append(Finding(rel, line, rule, message))
            else:
                used.add(hit)

    for line, rule in waivers.items():
        if line not in used:
            findings.append(
                Finding(rel, line, "waiver", f"`{rule}-ok` waives nothing here any more; delete it")
            )
    return findings


def _python_files(root: Path) -> list[Path]:
    found = [root / path for path in CARRY_ALL if (root / path).is_file()]
    for path in SWALLOW_SCOPE:
        target = root / path
        found += [target] if target.is_file() else sorted(target.rglob("*.py")) if target.is_dir() else []
    tests = root / "tests"
    if tests.is_dir():
        found += sorted(tests.rglob("*.py"))
    for path in IMPORT_SCOPE:
        target = root / path
        found += sorted(target.rglob("*.py")) if target.is_dir() else []
    return list(dict.fromkeys(found))  # src/apeGmsh holds the narrow scopes too: scan each once


def scan(root: Path) -> list[Finding]:
    was_enabled = gc.isenabled()
    gc.disable()  # the getattr index holds every parsed tree; collection passes over them cost more than they free
    try:
        return _scan(root)
    finally:
        if was_enabled:
            gc.enable()


def _scan(root: Path) -> list[Finding]:
    _getattr_cache.pop(root, None)
    _baseline_seen.pop(root, None)
    _baseline_cache.pop(root, None)
    for cache in (_sites_cache, _probe_cache, _tree_cache):
        for key in [k for k in cache if k[0] == root]:
            del cache[key]
    findings = check_adr_numbers(root) + check_doc_paths(root) + check_arch_path(root)
    for path in _python_files(root):
        findings.extend(scan_file(path, path.relative_to(root).as_posix(), root))
    findings.extend(_stale_baseline(root))
    _tree_cache.clear()
    _probe_cache.clear()
    return sorted(findings, key=lambda f: (f.path, f.line, f.rule))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO, help="tree to scan")
    args = parser.parse_args(argv)
    findings = scan(args.root.resolve())
    for finding in findings:
        print(finding)
    if findings:
        print(f"\n{len(findings)} quirk finding(s). Fix the code, or waive one Python site "
              "with `# apegmsh-lint: <rule>-ok <reason>` if the lesson does not apply there.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
