#!/usr/bin/env python
"""Advisory lint for agent-written apeGmsh model scripts ("Script English").

Checks the machine-checkable rules of ``references/model-scripts.md``:

    S1  imports at the top: no import after the first real statement
    S2  flat and local: no ``def`` / ``main()``; no unused imports or locals
    S3  no absolute output paths (a drive letter or a leading ``/``)
    V1  labels, not tags: geometry results kept as handles, ``.tags[...]`` /
        ``.ids[...]``, integer tags passed to bridge calls
    V3  find by name: ``in_box``, ``np.isclose`` on coordinates, iterating
        ``fem.nodes.coords``
    V4  named numbers: inline float literals in call arguments, a data constant
        with no trailing comment, several assignments on one line
    V5  units by a named divisor: ``* 1000``, ``/ 1e3`` and the like
    T1  one action per statement: arithmetic inside an f-string, nested
        comprehensions, overlong lines
    T2  one OpenSees handle: raw openseespy imported as ``ops_raw`` only, each
        ``ops_raw.`` use marked ``# raw-ops: <why>``
    T4  assert success and the check: ``analyze`` result asserted, at least one
        ``assert``, no "Expected ... <number>" claims written before the run

It is advisory: the skill runs it on agent output, and it is not a CI gate on user
scripts. M1 (fidelity), V2 (role naming), V6 (symbol choice), T3 (hand-check
quality) and whether a comment is true still need a reviewer.

Usage:
    python lint_model_script.py [--rules S1,V5] FILE [FILE ...]

Each finding prints ``path:line: <RULE> <message>``. Exit code 0 with no findings,
1 with findings. Stdlib only (``ast`` and ``tokenize``); ``pyflakes`` is optional and
is used for the S2 unused-name sub-check, which is skipped with a note when missing.
Nothing is imported or run from the script under test.
"""
from __future__ import annotations

import argparse
import ast
import io
import re
import sys
import tokenize

ALL_RULES = ("S1", "S2", "S3", "V1", "V3", "V4", "V5", "T1", "T2", "T4")

MAX_LINE = 110                      # T1: longest allowed code line
GEOMETRY_ADDERS = re.compile(r"^add_")
GEOMETRY_RECEIVER = re.compile(r"geo", re.IGNORECASE)
TAG_KEYWORDS = {
    "tag", "tags", "node", "nodes", "ele", "eles", "element", "elements",
    "master", "slave", "retained", "constrained", "ids",
}
TAG_POSITIONAL_METHODS = {"fix", "mass", "load"}      # ops.fix(1, ...) and friends
V4_EXEMPT_ROOTS = {
    "math", "np", "numpy", "plt", "ax", "axes", "axs", "fig", "matplotlib", "mpl",
    "range", "print", "round", "int", "float", "abs", "min", "max", "sum", "len",
    "enumerate", "zip", "divmod", "pow", "str", "format", "sorted", "isinstance",
}
V4_EXEMPT_KEYWORDS = {
    "ndm", "ndf", "dofs", "dim", "order", "n_nodes", "n_increments", "steps",
    "max_iter", "figsize", "dpi", "linewidth", "lw", "fontsize", "alpha",
    "markersize", "ms", "s", "n", "nmodes", "num_modes",
}
V4_FREE_FLOATS = {0.0, 1.0, -1.0}
V4_INT_RANGE = range(-1, 7)         # DOF numbers, ndm/ndf, dim, element order
V5_FACTORS = {1000, 0.001}
ABS_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|/[\w.~-]+)")
EXPECTED_CLAIM = re.compile(r"\bExpected\b[^\n]*\d")


_NOTED: list[bool] = []         # the pyflakes note prints once per run


class Finding:
    def __init__(self, line: int, rule: str, message: str) -> None:
        self.line, self.rule, self.message = line, rule, message


def _root_name(node: ast.AST) -> str | None:
    """Leftmost name of an attribute / call / subscript chain."""
    while True:
        if isinstance(node, ast.Attribute):
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        elif isinstance(node, ast.Subscript):
            node = node.value
        else:
            break
    return node.id if isinstance(node, ast.Name) else None


def _chain_names(node: ast.AST) -> list[str]:
    """Every attribute and name along a chain, outermost first."""
    names: list[str] = []
    while True:
        if isinstance(node, ast.Attribute):
            names.append(node.attr)
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        elif isinstance(node, ast.Subscript):
            node = node.value
        else:
            break
    if isinstance(node, ast.Name):
        names.append(node.id)
    return names


def _number(node: ast.AST) -> float | None:
    """Value of a numeric literal, including a unary minus; else None."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _number(node.operand)
        if inner is None:
            return None
        return -inner if isinstance(node.op, ast.USub) else inner
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return node.value
    return None


def _is_docstring(parent: ast.AST, node: ast.AST) -> bool:
    body = getattr(parent, "body", None)
    return (
        isinstance(parent, (ast.Module, ast.FunctionDef, ast.ClassDef))
        and bool(body) and body[0] is node
        and isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
    )


class Source:
    """One parsed script plus the token-level facts the rules need."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.lines = text.splitlines()
        self.tree = ast.parse(text)
        self.comments: dict[int, str] = {}
        self.string_lines: set[int] = set()     # lines inside multi-line strings
        toks = tokenize.generate_tokens(io.StringIO(text).readline)
        for tok in toks:
            if tok.type == tokenize.COMMENT:
                self.comments[tok.start[0]] = tok.string
            elif tok.type == tokenize.STRING and tok.end[0] > tok.start[0]:
                self.string_lines.update(range(tok.start[0], tok.end[0] + 1))
        self.docstrings: set[int] = set()       # id() of docstring Constant nodes
        for parent in ast.walk(self.tree):
            for child in getattr(parent, "body", []) if isinstance(
                    getattr(parent, "body", None), list) else []:
                if _is_docstring(parent, child):
                    self.docstrings.add(id(child.value))


# ---------------------------------------------------------------- rules


def rule_s1(src: Source) -> list[Finding]:
    out: list[Finding] = []
    seen_statement = False
    for stmt in src.tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            if seen_statement:
                out.append(Finding(stmt.lineno, "S1", "import after the first statement; move it to the top"))
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue                                    # docstring
        if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                and _chain_names(stmt.value.func)[:1] == ["use"]
                and _root_name(stmt.value.func) in ("matplotlib", "mpl")):
            continue                                    # matplotlib.use(...) is a header
        seen_statement = True
    top_level = {id(s) for s in src.tree.body}
    for node in ast.walk(src.tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and id(node) not in top_level:
            out.append(Finding(node.lineno, "S1", "import inside a block; move it to the top"))
    return out


def rule_s2(src: Source) -> list[Finding]:
    out: list[Finding] = []
    for node in ast.walk(src.tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(Finding(node.lineno, "S2", f"function '{node.name}' defined; a model script is flat"))
        elif isinstance(node, ast.ClassDef):
            out.append(Finding(node.lineno, "S2", f"class '{node.name}' defined; a model script is flat"))
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "main"):
            out.append(Finding(node.lineno, "S2", "main() call; a model script is flat"))
    try:
        from pyflakes.checker import Checker
    except ImportError:
        if _NOTED:
            return out
        _NOTED.append(True)
        print("note: pyflakes is not installed; skipping the S2 unused-name sub-check",
              file=sys.stderr)
        return out
    for msg in Checker(src.tree, "<script>").messages:
        if type(msg).__name__ in ("UnusedImport", "UnusedVariable"):
            out.append(Finding(msg.lineno, "S2", msg.message % msg.message_args))
    return out


def rule_s3(src: Source) -> list[Finding]:
    out: list[Finding] = []
    for node in ast.walk(src.tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in src.docstrings
                and not any(ch.isspace() for ch in node.value)
                and ABS_PATH.match(node.value)):
            out.append(Finding(node.lineno, "S3", f"absolute path {node.value!r}; write beside the script"))
    return out


def rule_v1(src: Source) -> list[Finding]:
    out: list[Finding] = []
    discarded = {id(s.value) for s in ast.walk(src.tree)
                 if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)}
    for node in ast.walk(src.tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names = _chain_names(node.func.value)
            if (GEOMETRY_ADDERS.match(node.func.attr)
                    and any(GEOMETRY_RECEIVER.search(n) for n in names)
                    and id(node) not in discarded):
                out.append(Finding(node.lineno, "V1",
                                   f"{node.func.attr}() result kept as a handle; give it a label and refer by label"))
            if _root_name(node.func) == "ops":
                for kw in node.keywords:
                    if kw.arg in TAG_KEYWORDS:
                        out.append(Finding(node.lineno, "V1",
                                           f"'{kw.arg}=' in a bridge call; select by group or label"))
                if node.func.attr in TAG_POSITIONAL_METHODS:
                    for arg in node.args:
                        if _number(arg) is not None or isinstance(arg, (ast.List, ast.Tuple)):
                            out.append(Finding(node.lineno, "V1",
                                               "positional tag in a bridge call; select by group or label"))
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute)
                and node.value.attr in ("tags", "ids")):
            out.append(Finding(node.lineno, "V1", f".{node.value.attr}[...] indexes a tag; use a label"))
    return out


def rule_v3(src: Source) -> list[Finding]:
    out: list[Finding] = []
    for node in ast.walk(src.tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "in_box":
                out.append(Finding(node.lineno, "V3", "in_box() selects by coordinates; find by name"))
            elif node.func.attr == "isclose" and _root_name(node.func) in ("np", "numpy"):
                out.append(Finding(node.lineno, "V3", "np.isclose() on coordinates; find by name"))
        iters: list[ast.AST] = []
        if isinstance(node, (ast.For, ast.AsyncFor)):
            iters.append(node.iter)
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            iters.extend(gen.iter for gen in node.generators)
        for it in iters:
            for sub in ast.walk(it):
                if (isinstance(sub, ast.Attribute) and sub.attr == "coords"
                        and isinstance(sub.value, ast.Attribute) and sub.value.attr == "nodes"):
                    out.append(Finding(node.lineno, "V3", "iterating nodes.coords; find by name"))
                    break
    return out


def _v4_literal_call_args(node: ast.Call) -> list[tuple[ast.AST, str | None]]:
    """Numeric literals among the arguments, with the keyword they sit under."""
    found: list[tuple[ast.AST, str | None]] = []

    def collect(value: ast.AST, keyword: str | None) -> None:
        if _number(value) is not None:
            found.append((value, keyword))
        elif isinstance(value, (ast.Tuple, ast.List)):
            for element in value.elts:
                collect(element, keyword)

    for arg in node.args:
        collect(arg, None)
    for kw in node.keywords:
        collect(kw.value, kw.arg)
    return found


def rule_v4(src: Source) -> list[Finding]:
    out: list[Finding] = []
    for node in ast.walk(src.tree):
        if isinstance(node, ast.Call) and _root_name(node.func) not in V4_EXEMPT_ROOTS:
            for value, keyword in _v4_literal_call_args(node):
                number = _number(value)
                if keyword in V4_EXEMPT_KEYWORDS:
                    continue
                if isinstance(number, int) and number in V4_INT_RANGE:
                    continue
                if isinstance(number, float) and number in V4_FREE_FLOATS:
                    continue
                where = f"'{keyword}='" if keyword else "an argument"
                out.append(Finding(value.lineno, "V4",
                                   f"literal {number!r} in {where}; name it in the data section with its unit"))
    for stmt in ast.walk(src.tree):
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id.isupper():
            has_float = any(isinstance(n, ast.Constant) and isinstance(n.value, float)
                            for n in ast.walk(stmt.value))
            documented = stmt.lineno in src.comments or (
                stmt.lineno - 1 in src.comments
                and src.lines[stmt.lineno - 2].lstrip().startswith("#"))
            if has_float and not documented:
                out.append(Finding(stmt.lineno, "V4",
                                   f"{stmt.targets[0].id} has no trailing comment giving its unit and meaning"))
    for parent in ast.walk(src.tree):
        body = getattr(parent, "body", None)
        if isinstance(body, list):
            for first, second in zip(body, body[1:]):
                if first.lineno == second.lineno and isinstance(first, ast.Assign):
                    out.append(Finding(first.lineno, "V4", "several assignments on one line"))
    return out


def rule_v5(src: Source) -> list[Finding]:
    out: list[Finding] = []
    for node in ast.walk(src.tree):
        if not isinstance(node, ast.BinOp):
            continue
        right = _number(node.right)
        left = _number(node.left)
        if isinstance(node.op, ast.Mult) and (right in V5_FACTORS or left in V5_FACTORS):
            out.append(Finding(node.lineno, "V5", "unit conversion by a literal; divide by a named unit"))
        elif isinstance(node.op, ast.Div) and right in V5_FACTORS:
            out.append(Finding(node.lineno, "V5", "unit conversion by a literal; divide by a named unit"))
    return out


def _is_named_unit_division(node: ast.AST) -> bool:
    """``x / UNIT`` where UNIT is an upper-case name: the one allowed f-string arithmetic."""
    return (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
            and isinstance(node.right, ast.Name) and node.right.id.isupper()
            and not any(isinstance(n, ast.BinOp) for n in ast.walk(node.left)))


def rule_t1(src: Source) -> list[Finding]:
    out: list[Finding] = []
    for node in ast.walk(src.tree):
        if isinstance(node, ast.FormattedValue):
            if any(isinstance(n, ast.BinOp) for n in ast.walk(node.value)) \
                    and not _is_named_unit_division(node.value):
                out.append(Finding(node.value.lineno, "T1", "arithmetic inside an f-string; compute it on its own line"))
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            inner = [n for n in ast.walk(node) if n is not node and isinstance(
                n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))]
            if inner or len(node.generators) > 1:
                out.append(Finding(node.lineno, "T1", "nested comprehension; use a loop"))
    for number, line in enumerate(src.lines, start=1):
        if len(line) > MAX_LINE and number not in src.string_lines:
            out.append(Finding(number, "T1", f"line is {len(line)} characters (limit {MAX_LINE}); split it"))
    return out


def rule_t2(src: Source) -> list[Finding]:
    out: list[Finding] = []
    for node in ast.walk(src.tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in ("openseespy", "opensees") and alias.asname != "ops_raw":
                    out.append(Finding(node.lineno, "T2", "raw openseespy must be imported 'as ops_raw'"))
        elif isinstance(node, ast.ImportFrom) and node.module \
                and node.module.split(".")[0] in ("openseespy", "opensees"):
            out.append(Finding(node.lineno, "T2", "import raw openseespy as 'import openseespy.opensees as ops_raw'"))
        elif isinstance(node, ast.Call) and _root_name(node.func) == "ops_raw":
            first, last = node.lineno, getattr(node, "end_lineno", node.lineno)
            lines = list(range(first - 1, last + 1))
            if not any("raw-ops:" in src.comments.get(n, "") for n in lines):
                out.append(Finding(node.lineno, "T2", "ops_raw call without a '# raw-ops: <why>' comment"))
    return out


def rule_t4(src: Source) -> list[Finding]:
    out: list[Finding] = []
    asserted: set[str] = set()
    for node in ast.walk(src.tree):
        if isinstance(node, ast.Assert):
            asserted.update(n.id for n in ast.walk(node.test) if isinstance(n, ast.Name))
    has_assert = any(isinstance(n, ast.Assert) for n in ast.walk(src.tree))
    if not has_assert:
        out.append(Finding(1, "T4", "no assert: assert the run succeeded and that each check agrees"))
    for node in ast.walk(src.tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) \
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "analyze" \
                and _root_name(node.value.func) == "ops":
            out.append(Finding(node.lineno, "T4", "analyze() result discarded; assign it and assert it"))
        elif (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "analyze"
                and _root_name(node.value.func) == "ops"):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if names and not names & asserted:
                out.append(Finding(node.lineno, "T4", "analyze() status never asserted"))
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for match in EXPECTED_CLAIM.finditer(node.value):
                line = node.lineno + node.value.count("\n", 0, match.start())
                out.append(Finding(line, "T4", "'Expected ... <number>' written before the run; report only run numbers"))
    return out


RULES = {
    "S1": rule_s1, "S2": rule_s2, "S3": rule_s3, "V1": rule_v1, "V3": rule_v3,
    "V4": rule_v4, "V5": rule_v5, "T1": rule_t1, "T2": rule_t2, "T4": rule_t4,
}


def lint_text(text: str, rules: tuple[str, ...] = ALL_RULES) -> list[Finding]:
    """Findings for one script's source, sorted by line then rule."""
    src = Source(text)
    found: list[Finding] = []
    for rule in rules:
        found.extend(RULES[rule](src))
    found.sort(key=lambda f: (f.line, f.rule, f.message))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="model scripts to check")
    parser.add_argument("--rules", default=",".join(ALL_RULES),
                        help="comma-separated rule ids to run (default: all)")
    args = parser.parse_args(argv)
    rules = tuple(r.strip().upper() for r in args.rules.split(",") if r.strip())
    unknown = [r for r in rules if r not in RULES]
    if unknown:
        parser.error(f"unknown rule(s): {', '.join(unknown)}; choose from {', '.join(ALL_RULES)}")

    total = 0
    for path in args.files:
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
            findings = lint_text(text, rules)
        except (OSError, SyntaxError, tokenize.TokenError) as exc:
            print(f"{path}:1: ERROR cannot check this file: {exc}")
            total += 1
            continue
        for finding in findings:
            print(f"{path}:{finding.line}: {finding.rule} {finding.message}")
        total += len(findings)
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
