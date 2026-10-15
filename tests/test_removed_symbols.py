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
* **each negative claim in the skill names an entry.** A negative is a
  sentence that says ``not (yet) shipped``, ``was``/``were removed``,
  ``no longer`` or ``does not exist`` (``**`` emphasis ignored) and
  carries a backticked symbol. The symbol nearest before the phrase in
  that sentence (or else the first after it) must be an entry or a member
  of one. Files the fence gate defers to another chain are left out, as
  there.

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

NEGATIVE = re.compile(
    r"not\s+(?:yet\s+)?shipped|(?:was|were)\s+removed|no\s+longer|does\s+not\s+exist",
    re.IGNORECASE,
)
_SPAN = re.compile(r"`([^`]+)`")
_LEAD = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
_BOUNDARY = re.compile(r"[.!?](?=\s)")
_NEW_BLOCK = re.compile(r"^(?:#|\||[-*+]\s|\d+\.\s)")


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


def negatives(text: str) -> list[tuple[int, str]]:
    """``(lineno, symbol)`` for each negative claim: the backticked symbol
    nearest before the phrase in its sentence, or else the first after it."""
    found: list[tuple[int, str]] = []
    for lineno, block in prose_blocks(text):
        block = block.replace("**", "")
        spans = [(m.start(), m.end(), m.group(1)) for m in _SPAN.finditer(block)]
        inside = lambda pos: any(a <= pos < b for a, b, _ in spans)  # noqa: E731
        stops = [m.end() for m in _BOUNDARY.finditer(block) if not inside(m.start())]
        for m in NEGATIVE.finditer(block):
            if inside(m.start()):
                continue
            lo = max([s for s in stops if s <= m.start()], default=0)
            hi = min([s for s in stops if s > m.end()], default=len(block))
            symbols = [(a, lead.group(0)) for a, b, code in spans
                       if lo <= a and b <= hi and (lead := _LEAD.match(code))]
            before = [s for a, s in symbols if a < m.start()]
            after = [s for a, s in symbols if a > m.start()]
            if before or after:
                found.append((lineno, before[-1] if before else after[0]))
    return found


def registered(symbol: str, entries: list[dict]) -> bool:
    for entry in entries:
        for spelling in (entry["symbol"], *entry.get("also", [])):
            if symbol == spelling or (entry["kind"] == "attribute"
                                      and symbol.startswith(spelling + ".")):
                return True
    return False


def unregistered_negatives(text: str, rel: str, entries: list[dict]) -> list[str]:
    return [f"{rel}:{n} | prose-negative | `{s}` is called removed or not shipped but "
            "is not in removed_symbols.json"
            for n, s in negatives(text) if not registered(s, entries)]


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
    assert not found, (
        "A negative claim names a symbol the registry does not hold. If the symbol is "
        "really gone, add it to skills/apegmsh/scripts/removed_symbols.json (the test "
        "above then proves it absent); otherwise correct the prose:\n  " + "\n  ".join(found)
    )
    assert checked >= 5, f"only {checked} negatives found: the scan has stopped matching"


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
        (1, "gauss.select"), (3, "g.opensees"), (6, "Assembly.add"),
        (8, "g.modle"), (10, "fem.nodes.get_ids"),
    ]
    assert [f.split(" | ")[0] for f in unregistered_negatives(text, "x.md", entries)] == [
        "x.md:1", "x.md:8",
    ]
