"""Skill-docs drift lane — quoted API facts must match the live code.

A 2026-08-12 review found seven stale lines across ``skills/apegmsh/``
claiming ``results.viewer()`` defaults to ``blocking=True`` while the
live signature (``src/apeGmsh/results/Results.py``) had moved to
``blocking=None`` auto-detect — the docs were never re-read after the
signature changed, and nothing failed. This lane machine-checks the two
claim classes that review exposed, in the established source-scanning
style of ``test_viewers_pure_h5_consumer.py`` /
``tests/viewers/test_viewer_state_contract.py`` (structural scan +
explicit registry + positive controls):

* **D-SIG** — every ``def name(self|cls, ...)`` quoted inside a python
  code fence is parsed and compared against ``inspect.signature`` of
  the live object it documents: parameter names, order, kinds
  (keyword-only vs positional), and default values.
* **D-DEFAULT** — every claim of the form "default … ``blocking=X``"
  (or "``blocking=X`` … default", prose or fence) is compared against
  the live default of the registered parameter.

Scope is deliberately narrow — signature quotes and default-value
claims only; a general prose-claim checker is out of scope.

The registries are the coupling points, run with the same two-way
ratchet discipline as the AST guards' allowlists:

* A quoted API signature (first parameter ``self``/``cls``) whose name
  is not in ``_SIG_TARGETS`` FAILS and asks to be registered — docs
  cannot quote an unchecked signature.
* A ``_SIG_TARGETS`` entry whose name is no longer quoted anywhere
  FAILS and asks to be pruned — the registry cannot rot.
* A default-value claim for a registered keyword in a file with no
  ``_DEFAULT_CLAIM_TARGETS`` binding FAILS and asks for one. The
  binding is per FILE because two viewers share the ``blocking``
  keyword with different live defaults: ``results.viewer`` auto-detects
  (``None``) while the sections inspector still blocks (``True``).

Only the canonical skill (``skills/apegmsh/``) is scanned — the derived
mirror ``.claude/skills/apegmsh-helper/`` is byte-identical by
``scripts/sync_skill.py --check``, which already gates in CI.

Known ambiguity, unresolved because nothing hits it yet: both registries
key coarsely — ``_SIG_TARGETS`` by bare ``def`` name across the whole
corpus, ``_DEFAULT_CLAIM_TARGETS`` by file. Two different objects
sharing a method name (``results.viewer`` and ``sec.viewer``) can
therefore only be told apart when their quotes live in different files
and their claims in different files. Today that holds: only
``references/results.md`` quotes ``def viewer(...)``, and
``SKILL.md``'s single ``sec.viewer`` mention makes no default claim.
The moment it stops holding, re-key both registries by
``(file, name)`` rather than stretching a binding to cover two owners —
a mis-bound entry checks the docs against the wrong object and reads
green while drifting.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKILL_DIR = REPO / "skills" / "apegmsh"

_RESULTS_VIEWER = ("apeGmsh.results.Results", "Results.viewer")

# ── D-SIG registry — doc-quoted def name -> (module, qualname) ──────
#
# Maps each API signature quoted in the skill docs to the live object
# whose ``inspect.signature`` is the truth. ``compose`` binds to
# ``Compose.compose`` (not the ``apeGmsh.compose`` session facade,
# which is ``(source, *, label, **kwargs)`` and forwards): the doc
# quotes the full keyword surface, and the forwarding target is where
# that surface — and its defaults — actually live.
_SIG_TARGETS: dict[str, tuple[str, str]] = {
    "viewer": _RESULTS_VIEWER,
    "show_web": ("apeGmsh.results.Results", "Results.show_web"),
    "serve_web": ("apeGmsh.results.Results", "Results.serve_web"),
    "compose": ("apeGmsh.mesh._compose", "Compose.compose"),
    "from_h5": ("apeGmsh", "apeGmsh.from_h5"),
}

# ── D-DEFAULT registry — file -> {keyword: (module, qualname)} ──────
_DEFAULT_CLAIM_KEYWORDS = ("blocking",)
_DEFAULT_CLAIM_TARGETS: dict[str, dict[str, tuple[str, str]]] = {
    "SKILL.md": {"blocking": _RESULTS_VIEWER},
    "references/results.md": {"blocking": _RESULTS_VIEWER},
    "references/workflows.md": {"blocking": _RESULTS_VIEWER},
    # The sections inspector has NOT adopted the notebook auto-detect;
    # its docs legitimately claim a different default.
    "references/section-properties.md": {
        "blocking": ("apeGmsh.sections._analysis", "SectionProperties.viewer"),
    },
}


def _doc_files() -> list[Path]:
    files = [SKILL_DIR / "SKILL.md"]
    files += sorted((SKILL_DIR / "references").glob("*.md"))
    return files


def _resolve(module: str, qualname: str):
    """Import ``module`` and walk ``qualname`` to the underlying function.

    ``inspect.getattr_static`` on the last hop keeps the raw
    ``classmethod``/``staticmethod`` wrapper so the doc-quoted
    ``cls``/``self`` first parameter stays visible in the signature.
    """
    obj = importlib.import_module(module)
    parts = qualname.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    obj = inspect.getattr_static(obj, parts[-1])
    if isinstance(obj, (classmethod, staticmethod)):
        obj = obj.__func__
    return obj


# ── Fence + signature extraction ────────────────────────────────────

def _python_fences(text: str) -> list[tuple[int, str]]:
    """``(first_content_lineno, body)`` for each \\```python fence."""
    fences: list[tuple[int, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("```") and stripped[3:].strip() in ("python", "py"):
            start = i + 1
            j = start
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            fences.append((start + 1, "\n".join(lines[start:j])))
            i = j + 1
        else:
            i += 1
    return fences


def _balanced_paren(text: str, open_idx: int) -> str | None:
    """Content between ``text[open_idx] == '('`` and its matching ')'."""
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1:i]
    return None


_NO_DEFAULT = object()


def _parse_doc_params(params_src: str) -> tuple[list, bool]:
    """Parse a quoted parameter list into ``[(name, kind, default_node)]``.

    A trailing ``...`` marks the quote as deliberately abbreviated
    (``exhaustive=False``): quoted params are then checked as a subset
    instead of the full list. Raises ``SyntaxError`` on an unparseable
    quote — the caller reports it as a failure.
    """
    src = params_src.strip()
    stripped = re.sub(r",?\s*\.\.\.\s*$", "", src)
    exhaustive = stripped == src
    fn = ast.parse(f"def _doc_sig({stripped}): pass").body[0]
    assert isinstance(fn, ast.FunctionDef)
    a = fn.args
    P = inspect.Parameter
    params: list[tuple[str, object, object]] = []
    pos = a.posonlyargs + a.args
    pos_defaults = [_NO_DEFAULT] * (len(pos) - len(a.defaults)) + list(a.defaults)
    for arg, dflt in zip(pos, pos_defaults):
        kind = (P.POSITIONAL_ONLY if arg in a.posonlyargs
                else P.POSITIONAL_OR_KEYWORD)
        params.append((arg.arg, kind, dflt))
    if a.vararg:
        params.append((a.vararg.arg, P.VAR_POSITIONAL, _NO_DEFAULT))
    for arg, dflt in zip(a.kwonlyargs, a.kw_defaults):
        params.append((arg.arg, P.KEYWORD_ONLY,
                       _NO_DEFAULT if dflt is None else dflt))
    if a.kwarg:
        params.append((a.kwarg.arg, P.VAR_KEYWORD, _NO_DEFAULT))
    return params, exhaustive


def _default_matches(node: ast.expr, real_default: object) -> tuple[bool, str]:
    """Compare a doc default (AST) with the live default object."""
    src = ast.unparse(node)
    try:
        doc_val = ast.literal_eval(node)
    except ValueError:
        return src == repr(real_default), src
    if isinstance(doc_val, bool) != isinstance(real_default, bool):
        return False, src
    return doc_val == real_default, src


def _compare_doc_sig(params_src: str, live_func) -> list[str]:
    """Problems between a quoted parameter list and the live signature."""
    try:
        doc_params, exhaustive = _parse_doc_params(params_src)
    except SyntaxError as exc:
        return [f"quoted signature does not parse as Python: {exc.msg}"]
    real = inspect.signature(live_func)
    real_params = list(real.parameters.values())
    real_by_name = {p.name: p for p in real_params}
    problems: list[str] = []
    for name, kind, default_node in doc_params:
        rp = real_by_name.get(name)
        if rp is None:
            problems.append(
                f"parameter `{name}` does not exist in the live signature "
                f"{real}"
            )
            continue
        if kind != rp.kind:
            problems.append(
                f"`{name}` kind drift: doc quotes {kind.description}, "
                f"live is {rp.kind.description}"
            )
        if default_node is _NO_DEFAULT:
            if rp.default is not inspect.Parameter.empty:
                problems.append(
                    f"`{name}` is quoted as required but the live "
                    f"parameter defaults to {rp.default!r}"
                )
        elif rp.default is inspect.Parameter.empty:
            problems.append(
                f"doc gives `{name}` a default but the live parameter "
                "is required"
            )
        else:
            ok, doc_src = _default_matches(default_node, rp.default)
            if not ok:
                problems.append(
                    f"`{name}` default drift: doc quotes {doc_src}, "
                    f"live is {rp.default!r}"
                )
    if exhaustive:
        doc_names = [name for name, _, _ in doc_params]
        real_names = [p.name for p in real_params]
        if doc_names != real_names:
            problems.append(
                f"parameter list drift: doc quotes {doc_names}, live is "
                f"{real_names} (append `...` to the quote if the "
                "abbreviation is deliberate)"
            )
    return problems


def _quoted_api_signatures(
    body: str,
) -> list[tuple[int, str, str]]:
    """``(line_offset_in_body, name, params_src)`` for each API quote.

    A quote is an API signature when its first parameter is ``self`` or
    ``cls``; other ``def``s in fences are example code and are skipped.
    """
    quotes: list[tuple[int, str, str]] = []
    for m in re.finditer(r"^[ \t]*def[ \t]+(\w+)[ \t]*\(", body, re.M):
        params_src = _balanced_paren(body, m.end() - 1)
        if params_src is None:
            continue
        first = params_src.strip().split(",", 1)[0].strip()
        if first not in ("self", "cls"):
            continue
        quotes.append((body[:m.start()].count("\n"), m.group(1), params_src))
    return quotes


# ── D-DEFAULT claim extraction ──────────────────────────────────────

def _claim_pattern(kw: str) -> re.Pattern[str]:
    # A claim couples the word "default(s)" with a `kw=value` token,
    # in either order, with nothing sentence-ending in between. The
    # 30-char gap keeps "use `blocking=False` … the default is …"
    # style sentences (two different facts) from fusing into one claim.
    val = r"[^\s`,)*]+"
    gap = r"[^.!?;:]{0,30}?"
    return re.compile(
        rf"\bdefaults?\b{gap}`?{kw}=(?P<pre>{val})"
        rf"|\b{kw}=(?P<post>{val}){gap}\bdefaults?\b",
        re.IGNORECASE,
    )


def _scan_default_claims(text: str, kw: str) -> list[tuple[int, str]]:
    """``(lineno, claimed_value_src)`` for every default-claim of ``kw``."""
    claims: list[tuple[int, str]] = []
    for m in _claim_pattern(kw).finditer(text):
        lineno = text[:m.start()].count("\n") + 1
        claims.append((lineno, m.group("pre") or m.group("post")))
    return claims


# ── Corpus checks ───────────────────────────────────────────────────

def test_docs_corpus_found() -> None:
    files = _doc_files()
    assert (SKILL_DIR / "SKILL.md").is_file(), (
        f"skills/apegmsh/SKILL.md not found under {SKILL_DIR} — update "
        "the path constants if the canonical skill moved."
    )
    assert len(files) >= 5, "references/*.md corpus unexpectedly small"
    n_fences = sum(
        len(_python_fences(p.read_text(encoding="utf-8"))) for p in files
    )
    assert n_fences >= 5, (
        "python-fence extractor found almost nothing — extractor or "
        "doc-format drift, the lane is no longer scanning anything."
    )


def test_quoted_signatures_match_live_code() -> None:
    failures: list[str] = []
    seen: set[str] = set()
    for path in _doc_files():
        rel = path.relative_to(SKILL_DIR).as_posix()
        for fence_line, body in _python_fences(path.read_text(encoding="utf-8")):
            for offset, name, params_src in _quoted_api_signatures(body):
                lineno = fence_line + offset
                if name not in _SIG_TARGETS:
                    failures.append(
                        f"  {rel}:{lineno}: quoted API signature `def "
                        f"{name}(...)` has no _SIG_TARGETS entry — "
                        "register the live object it documents."
                    )
                    continue
                seen.add(name)
                live = _resolve(*_SIG_TARGETS[name])
                for problem in _compare_doc_sig(params_src, live):
                    failures.append(f"  {rel}:{lineno}: def {name}: {problem}")
    assert seen, (
        "No quoted API signatures found at all — the extractor no "
        "longer matches the docs' fence format."
    )
    if failures:
        raise AssertionError(
            "D-SIG drift — signatures quoted in skills/apegmsh/ no "
            "longer match the live code (fix the DOC, or the registry "
            "if the object moved):\n" + "\n".join(failures)
        )


def test_signature_registry_has_no_stale_entries() -> None:
    seen: set[str] = set()
    for path in _doc_files():
        for _, body in _python_fences(path.read_text(encoding="utf-8")):
            seen.update(name for _, name, _ in _quoted_api_signatures(body))
    stale = set(_SIG_TARGETS) - seen
    assert not stale, (
        f"_SIG_TARGETS entries no longer quoted anywhere in the docs: "
        f"{sorted(stale)} — prune them (the registry must not rot)."
    )


def test_default_value_claims_match_live_code() -> None:
    failures: list[str] = []
    n_claims = 0
    for path in _doc_files():
        rel = path.relative_to(SKILL_DIR).as_posix()
        text = path.read_text(encoding="utf-8")
        bindings = _DEFAULT_CLAIM_TARGETS.get(rel, {})
        for kw in _DEFAULT_CLAIM_KEYWORDS:
            claims = _scan_default_claims(text, kw)
            if not claims:
                continue
            if kw not in bindings:
                for lineno, claimed in claims:
                    failures.append(
                        f"  {rel}:{lineno}: default-claim `{kw}={claimed}` "
                        f"in a file with no _DEFAULT_CLAIM_TARGETS binding "
                        f"for `{kw}` — bind the file to the object whose "
                        "default it describes."
                    )
                continue
            live = _resolve(*bindings[kw])
            real_default = inspect.signature(live).parameters[kw].default
            for lineno, claimed in claims:
                n_claims += 1
                try:
                    claimed_val = ast.literal_eval(claimed)
                except (ValueError, SyntaxError):
                    failures.append(
                        f"  {rel}:{lineno}: unparseable default-claim "
                        f"`{kw}={claimed}`"
                    )
                    continue
                if (claimed_val != real_default
                        or isinstance(claimed_val, bool)
                        != isinstance(real_default, bool)):
                    failures.append(
                        f"  {rel}:{lineno}: docs claim the default is "
                        f"`{kw}={claimed}` but the live default of "
                        f"{bindings[kw][1]} is {real_default!r}"
                    )
    assert n_claims >= 1, (
        "No default-value claims matched at all — the claim pattern no "
        "longer fits the docs' phrasing, the lane is guarding nothing."
    )
    if failures:
        raise AssertionError(
            "D-DEFAULT drift — default-value claims in skills/apegmsh/ "
            "no longer match the live code:\n" + "\n".join(failures)
        )


# ── Positive controls — the detectors must actually detect ──────────

def _control_auto(self, *, blocking=None, title=None):  # pragma: no cover
    """Stand-in for the auto-detect signature (default ``None``)."""


def test_control_fence_extractor_finds_fences() -> None:
    md = "prose\n```python\nx = 1\n```\n\n```text\nnot python\n```\n"
    fences = _python_fences(md)
    assert fences == [(3, "x = 1")]


def test_control_stale_signature_default_is_flagged() -> None:
    problems = _compare_doc_sig("self, *, blocking=True, title=None",
                                _control_auto)
    assert any("blocking" in p and "default drift" in p for p in problems), (
        f"planted blocking=True default not flagged: {problems}"
    )


def test_control_current_signature_passes() -> None:
    assert _compare_doc_sig("self, *, blocking=None, title=None",
                            _control_auto) == []


def test_control_abbreviated_signature_subset() -> None:
    assert _compare_doc_sig("self, *, blocking=None, ...",
                            _control_auto) == []
    problems = _compare_doc_sig("self, *, blocking=None", _control_auto)
    assert any("parameter list drift" in p for p in problems)


def test_control_stale_default_claim_is_found() -> None:
    for text in (
        "The **default `blocking=True`** runs the loop in-process.",
        "# results.viewer()  -> blocking=True default CRASHES the kernel",
        "`results.viewer(blocking=True)` is the\nDEFAULT and crashes.",
    ):
        claims = _scan_default_claims(text, "blocking")
        assert [c for _, c in claims] == ["True"], (text, claims)


def test_control_non_claims_are_not_matched() -> None:
    # Explicitly passing a value is advice, not a default claim — the
    # sentence boundary between the two facts must keep them apart.
    text = ("In notebooks use `results.viewer(blocking=False)`. "
            "Sessions restore by default.")
    assert _scan_default_claims(text, "blocking") == []
