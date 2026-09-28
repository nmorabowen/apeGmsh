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
is itself a finding. `adr-number` has no waiver: a collision is never right.

    python scripts/check_quirks.py              # this checkout
    python scripts/check_quirks.py --root DIR   # another tree, e.g. a `git archive`

Stdlib only, a few seconds. `tests/test_check_quirks.py` holds one case per
shape each rule must flag or pass; CI runs this scan as the last step of
`static-gates`.
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import sys
import tokenize
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DECISIONS = Path("src/apeGmsh/opensees/architecture/decisions")
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

WAIVER = re.compile(r"#\s*apegmsh-lint:\s*(?P<rule>[a-z-]+?)-ok\b(?P<reason>.*)$")

RULES: dict[str, str] = {
    "adr-number": (
        "two ADRs share a number, or an ADR file has no row in decisions/README.md. "
        "List the directory on origin/main before numbering, and add the index row in "
        "the same PR: an unindexed ADR makes its number look free. Incidents: a second "
        "0065 (#676, fixed #677), renumbers 0072->0073 (#741) and 0074->0079 (#817)"
    ),
    "schema-literal": (
        "a schema version compared against a hard-coded literal goes stale at the next "
        "bump and turns main red. Import NEUTRAL_CURRENT / OPENSEES_CURRENT (or the "
        "*_PRIOR_MINOR pair) from tests/fixtures/schema.py. Incidents: stale at the "
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
}


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: [{self.rule}] {self.message}"


# --- adr-number -------------------------------------------------------------


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
        for name in files:
            if f"({name})" not in index:
                findings.append(
                    Finding(f"{rel}/README.md", 0, "adr-number",
                            f"no index row links {name}. " + RULES["adr-number"])
                )
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
    is how the window and major-refusal tests work, so assignments pass;
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


PYTHON_RULES: dict[str, Callable[[ast.AST, str, Path], Iterator[tuple[int, str]]]] = {
    "schema-literal": check_schema_literal,
    "compose-streams": check_compose_streams,
    "resolve-swallow": check_resolve_swallow,
    "openseespy-import": check_openseespy_import,
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
    return "openseespy" in lowered and IMPORT_TEXT.search(re.sub(r"\\\r?\n", " ", lowered)) is not None


def scan_file(path: Path, rel: str, root: Path) -> list[Finding]:
    text = _read_source(path)
    if text is None:
        return []  # not valid Python source; like a SyntaxError, ruff and pytest will say so
    lowered = text.lower()
    if "apegmsh-lint" not in lowered and not _may_apply(rel, lowered):
        return []  # nothing a rule reads here; skipping the parse keeps the scan fast
    try:
        tree = ast.parse(text, filename=rel)
    except SyntaxError:
        return []  # ruff and pytest will say so; this lint only reads code
    lines = text.splitlines()
    spans = _statement_spans(tree)

    findings: list[Finding] = []
    waivers: dict[int, str] = {}
    for line, comment in _comments(text).items():
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
    findings = check_adr_numbers(root)
    for path in _python_files(root):
        findings.extend(scan_file(path, path.relative_to(root).as_posix(), root))
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
