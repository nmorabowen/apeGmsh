"""The removed-symbols registry: every entry is absent live, and every
negative claim in the skill names an entry.

``skills/apegmsh/scripts/removed_symbols.json`` lists each apeGmsh symbol
the skill says is removed or not shipped. Lint rule R1 flags those symbols
in a model script. Before the registry, ``gotchas.md`` said ``gauss.select``
was not shipped while ``GaussResultsComposite`` inherits it, and nothing
failed (#1197).

This lane runs in the suite with no marker. It checks that:

* **each entry is absent live.** "Absent" means the parent resolves and
  only the last name is missing, through the shared static resolver
  (``tests/_doc_resolver.py``) the fence gate uses. A typo in the parent
  is not proof of absence, and a symbol that resolves fails;
* **each negative claim in the skill names only entries.** A negative is
  a clause that says ``isn't / aren't / not (yet) shipped``,
  ``was / were / has been / have been removed|deleted``, ``is / are gone``,
  ``no longer`` or ``does not exist`` (``**`` emphasis ignored) and carries
  a backticked symbol. The clause is the parenthesis holding the phrase, or
  else its dash-separated part of the sentence (the whole sentence when that
  part opens with a pronoun such as "it"). Every backticked symbol in the
  clause must be an entry, an entry's member, or ``PROSE_NEGATIVE_BASELINE``
  (shrink-only, files this slice does not own). Files the fence gate defers
  to another chain are left out, as there.

The controls at the bottom are fixtures: each check must fail on the
defect it exists for.
"""
from __future__ import annotations

import importlib
import importlib.util
import re
import sys
from pathlib import Path

import pytest

from tests.test_skill_fences import DEFERRED, SKILL, _PROBLEM, _seeded_page

REPO = Path(__file__).resolve().parent.parent
REGISTRY = SKILL / "scripts" / "removed_symbols.json"
MIRROR = REPO / ".claude" / "skills" / "apegmsh-helper" / "scripts" / "removed_symbols.json"
LINT = SKILL / "scripts" / "lint_model_script.py"

_spec = importlib.util.spec_from_file_location("lint_model_script_registry", LINT)
lint = importlib.util.module_from_spec(_spec)
sys.modules["lint_model_script_registry"] = lint
_spec.loader.exec_module(lint)


def _entries() -> list[dict]:
    return lint.load_registry(REGISTRY)


# ── Absent live ─────────────────────────────────────────────────────

def probes(entry: dict) -> list[str]:
    """One resolver fence per spelling of an attribute entry, in the skill's
    vocabulary (``g`` / ``fem`` / ``ops`` / ``results`` are seeded)."""
    symbol = entry["symbol"]
    if "from" in entry:
        first = f"from {entry['from']} import {symbol.split('.')[0]}\n{symbol}"
    elif "import" in entry:
        first = f"import {entry['import']}\n{symbol}"
    else:
        first = symbol
    return [first, *entry.get("also", [])]


def why_present(fence: str) -> str | None:
    """``None`` when the fence's last name is absent and everything before it
    resolves; otherwise why that is not proven."""
    leaf = fence.splitlines()[-1].rsplit(".", 1)[-1]
    page = _seeded_page("removed_symbols.json")
    page.fence(fence, 1)
    problems = [m["msg"] if (m := _PROBLEM.match(p)) else p.strip() for p in page.problems]
    absent = re.compile(rf"(has no attribute|^module `[\w.]+` has no) `{re.escape(leaf)}`$")
    if len(problems) == 1 and absent.search(problems[0]):
        return None
    if problems:
        return f"its parent does not resolve: {problems}"
    if any(c.endswith(f".{leaf}") for c in page.checked):
        return "it resolves live"
    return "the resolver cannot type its parent, so absence is not proven"


def why_member_present(entry: dict) -> str | None:
    module, name = entry["member_of"].split(":")
    container = getattr(importlib.import_module(module), name, None)
    if not isinstance(container, (tuple, list, set, frozenset)) or not all(
            isinstance(v, str) for v in container):
        return f"`{entry['member_of']}` is not a container of names"
    return f"`{entry['member_of']}` holds it" if entry["symbol"] in container else None


@pytest.mark.parametrize(
    "fence",
    [f for e in _entries() if e["kind"] == "attribute" for f in probes(e)],
    ids=lambda f: f.splitlines()[-1],
)
def test_attribute_entry_is_absent_live(fence: str) -> None:
    reason = why_present(fence)
    assert reason is None, (
        f"`{fence.splitlines()[-1]}` is in removed_symbols.json, but {reason}. "
        "Drop the entry and correct the prose that calls it removed."
    )


@pytest.mark.parametrize("entry", [e for e in _entries() if e["kind"] == "member"],
                         ids=lambda e: e["symbol"])
def test_member_entry_is_absent_live(entry: dict) -> None:
    reason = why_member_present(entry)
    assert reason is None, f"`{entry['symbol']}` is in removed_symbols.json, but {reason}"


def test_registry_symbols_are_unique_and_the_mirror_matches() -> None:
    symbols = [s for e in _entries() for s in (e["symbol"], *e.get("also", []))]
    assert len(symbols) == len(set(symbols)), sorted(s for s in symbols if symbols.count(s) > 1)
    # Text, not bytes: a Windows checkout may hold either line ending.
    assert MIRROR.read_text(encoding="utf-8") == REGISTRY.read_text(encoding="utf-8"), (
        "run `python scripts/sync_skill.py`"
    )


# ── Negative claims in the prose ────────────────────────────────────

#: The phrases that make a sentence a negative claim (``**`` emphasis ignored).
NEGATIVE = re.compile(
    r"\bnot\s+(?:yet\s+)?shipped|n't\s+(?:yet\s+)?shipped"
    r"|\b(?:was|were|has\s+been|have\s+been)\s+(?:removed|deleted)"
    r"|\b(?:is|are)\s+gone\b|\bno\s+longer\b|\bdoes\s+not\s+exist",
    re.IGNORECASE,
)
_SPAN = re.compile(r"`([^`]+)`")
_LEAD = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
_BOUNDARY = re.compile(r"[.!?](?=\s)")
_DASH = re.compile(r"\s(?:—|–|--)\s")
_PRONOUN = re.compile(r"\s*(?:it|they|both|this|these)\b", re.IGNORECASE)
_NEW_BLOCK = re.compile(r"^(?:#|\||[-*+]\s|\d+\.\s)")

#: Negative claims in files this slice does not own whose clause names a
#: token that is not an apeGmsh symbol, as ``"<file> | <token>"``. Shrink-only:
#: the owning slice rewords the sentence and deletes the line.
PROSE_NEGATIVE_BASELINE: frozenset[str] = frozenset({
    # "... plus `emit_mass`) is the planned seam — it does not exist on `main`":
    # `main` names the default branch, not a symbol. interop.md is not SK1-S2's.
    "references/interop.md | main",
})


def prose_blocks(text: str) -> list[tuple[int, str]]:
    """``(first_lineno, joined_text)`` of each prose block outside code fences:
    a paragraph, list item, heading or table row, blockquote markers dropped."""
    blocks: list[tuple[int, str]] = []
    current: list[str] = []
    start = 0
    in_fence = False

    def close() -> None:
        if current:
            blocks.append((start, " ".join(current)))
            current.clear()

    for n, raw in enumerate(text.splitlines(), start=1):
        line = re.sub(r"^\s*(?:>\s?)+", "", raw).strip()
        if line.startswith("```"):
            close()
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not line or _NEW_BLOCK.match(line):
            close()
        if line:
            if not current:
                start = n
            current.append(line)
            if line.startswith(("#", "|")):        # a heading or a table row stands alone
                close()
    close()
    return blocks


def _clause(block: str, spans: list[tuple[int, int, str]], at: int) -> tuple[int, int]:
    """The negated clause around position ``at``: the innermost parenthesis
    holding it, else its dash-separated part of the sentence (parentheses
    included). Punctuation inside backticks never splits."""
    inside = lambda pos: any(a <= pos < b for a, b, _ in spans)  # noqa: E731
    stops = [m.end() for m in _BOUNDARY.finditer(block) if not inside(m.start())]
    lo = max([s for s in stops if s <= at], default=0)
    hi = min([s for s in stops if s > at], default=len(block))
    opened: list[int] = []
    for i in range(lo, hi):
        if inside(i):
            continue
        if block[i] == "(":
            opened.append(i)
        elif block[i] == ")" and opened:
            o = opened.pop()
            if o < at < i:
                return o + 1, i                # the innermost pair closes first
    dashes = [m for m in _DASH.finditer(block, lo, hi) if not inside(m.start())]
    part_lo = max([m.end() for m in dashes if m.end() <= at], default=lo)
    part_hi = min([m.start() for m in dashes if m.start() >= at], default=hi)
    if _PRONOUN.match(block, part_lo):
        return lo, hi                      # "— it does not exist": the subject is earlier
    return part_lo, part_hi


def negatives(text: str) -> list[tuple[int, list[str]]]:
    """``(lineno, symbols)`` for each negative claim: every backticked symbol
    in its clause. A clause that names no symbol makes no claim."""
    found: list[tuple[int, list[str]]] = []
    for lineno, block in prose_blocks(text):
        block = block.replace("**", "")
        spans = [(m.start(), m.end(), m.group(1)) for m in _SPAN.finditer(block)]
        for m in NEGATIVE.finditer(block):
            if any(a <= m.start() < b for a, b, _ in spans):
                continue
            lo, hi = _clause(block, spans, m.start())
            symbols = [lead.group(0) for a, b, code in spans
                       if lo <= a and b <= hi and (lead := _LEAD.match(code))]
            if symbols:
                found.append((lineno, symbols))
    return found


def registered(symbol: str, entries: list[dict]) -> bool:
    for entry in entries:
        for spelling in (entry["symbol"], *entry.get("also", [])):
            if symbol == spelling or (entry["kind"] == "attribute"
                                      and symbol.startswith(spelling + ".")):
                return True
    return False


def unregistered_negatives(text: str, rel: str, entries: list[dict]) -> list[str]:
    """``"<rel> | <symbol>"`` for each symbol a negative claim names that the
    registry does not hold."""
    return [f"{rel} | {s}" for _, symbols in negatives(text) for s in symbols
            if not registered(s, entries)]


def test_every_skill_negative_names_a_registered_symbol() -> None:
    entries = _entries()
    found: list[str] = []
    checked = 0
    for path in sorted(SKILL.rglob("*.md")):
        if path.name in DEFERRED:
            continue
        text = path.read_text(encoding="utf-8")
        checked += len(negatives(text))
        found += unregistered_negatives(text, path.relative_to(SKILL).as_posix(), entries)
    new = sorted(set(found) - PROSE_NEGATIVE_BASELINE)
    stale = sorted(PROSE_NEGATIVE_BASELINE - set(found))
    assert not new, (
        "A negative claim (removed / deleted / gone / not shipped / no longer / does not "
        "exist) names a symbol the registry does not hold. If the symbol is really gone, "
        "add it to skills/apegmsh/scripts/removed_symbols.json (the tests above then prove "
        "it absent); otherwise correct the prose:\n  " + "\n  ".join(new)
    )
    assert not stale, f"delete these from PROSE_NEGATIVE_BASELINE (shrink-only): {stale}"
    assert checked >= 10, f"only {checked} negatives found: the scan has stopped matching"


# ── Controls: each check flags the defect it exists for ─────────────

def test_control_absent_present_and_typo() -> None:
    assert why_present("g.opensees") is None
    assert why_present("fem.nodes.get_ids") is None
    # A symbol that exists must not pass as removed.
    assert why_present("g.model.select") == "it resolves live"
    assert why_present("from apeGmsh.assembly import Assembly\nAssembly.couple") == "it resolves live"
    # A typo in the parent is not proof that the leaf is gone.
    assert why_present("g.modle.selection").startswith("its parent does not resolve")
    assert why_present("from apeGmsh.assembli import Assembly\nAssembly.add") is not None


def test_control_member_entry() -> None:
    entry = {"symbol": "path", "member_of": "apeGmsh.results.session:PLOT_KINDS"}
    assert why_member_present(entry) is None
    assert why_member_present({**entry, "symbol": "xy"}) is not None
    assert why_member_present({**entry, "member_of": "apeGmsh.results.session:PLOT_KINDZ"}) is not None


def test_control_a_registry_entry_that_resolves_fails() -> None:
    # The gauss.select gotcha as a registry entry: present live, so it must fail.
    bad = {"kind": "attribute", "symbol": "results.elements.gauss.select", "use": "-", "why": "-"}
    assert [why_present(f) for f in probes(bad)] == ["it resolves live"]


def test_control_prose_negatives() -> None:
    entries = _entries()
    text = (
        "The `gauss.select` verb is not shipped yet.\n\n"           # the #1197 defect
        "> OpenSees is **not** a session composite (`g.opensees` was\n"
        "> removed) — it is the bridge `apeSees(fem)`.\n\n"
        "- The v1 `g.compose` and `Assembly.add / couple(part_a,\n"
        "  part_b) / materialize` were removed.\n"
        "- `g.modle` was removed. `g.opensees` is fine.\n\n"
        "No longer a problem: `fem.nodes.get_ids` and `x.y`.\n\n"
        "Gmsh no longer runs here.\n\n"                              # no symbol: not a claim
        "```python\nx = 1  # `g.foo` was removed, inside a fence\n```\n"
    )
    assert negatives(text) == [
        (1, ["gauss.select"]),
        (3, ["g.opensees"]),                      # the parenthesis, not `apeSees(fem)`
        (6, ["g.compose", "Assembly.add"]),
        (8, ["g.modle"]),                         # its own sentence, not `g.opensees`
        (10, ["fem.nodes.get_ids", "x.y"]),
    ]
    assert unregistered_negatives(text, "x.md", entries) == [
        "x.md | gauss.select", "x.md | g.modle", "x.md | x.y",
    ]


def test_control_every_symbol_in_the_clause_is_checked() -> None:
    # The two repros from the review of 0a74fcd8: a registered symbol beside an
    # unregistered one must not let the sentence pass.
    entries = _entries()
    for text in ("`fem.nodes.frobnicate` and `g.opensees` were removed.\n",
                 "`g.opensees` was removed; `fem.nodes.frobnicate` too.\n"):
        assert unregistered_negatives(text, "x.md", entries) == ["x.md | fem.nodes.frobnicate"], text
    # A pronoun-led clause takes its subject from the rest of the sentence.
    text = "The `x.planned` step is the seam — it does not exist on `main`.\n"
    assert negatives(text) == [(1, ["x.planned", "main"])]


@pytest.mark.parametrize("phrase", [
    "is not shipped", "isn't shipped", "aren't yet shipped", "are not shipped",
    "was removed", "were removed", "has been removed", "have been removed",
    "was deleted", "was **DELETED**", "has been deleted", "is gone", "are **gone**",
    "no longer exists", "does not exist", "does **not** exist",
])
def test_control_negative_phrases(phrase: str) -> None:
    assert negatives(f"The `g.frobnicate` verb {phrase} here.\n") == [(1, ["g.frobnicate"])]


def test_control_positive_phrases_make_no_claim() -> None:
    for text in ("`g.masses` is shipped.\n", "`fem.nodes.select` replaced the old accessors.\n",
                 "The gone-away `x.y` file.\n", "`Results.viewer` takes no `model_h5=`.\n"):
        assert negatives(text) == [], text
