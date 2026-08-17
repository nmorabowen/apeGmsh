"""Capability-map drift lane — the published fork list must match the code.

``docs/concepts/backend-capabilities.md`` tells a user which primitives
need the Ladruno fork build. That list is only useful while it is true,
and the failure mode is silent: someone adds a fork element to
``_FORK_ONLY_ELEMENTS``, the gate starts refusing it on stock, and the
docs still imply it runs anywhere.

Two token lists in the page are therefore machine-checked against the
live gate sets, in the two-way ratchet style of
``tests/test_skill_docs_drift.py``:

* a token in the runtime set but missing from the page FAILS and asks to
  be documented;
* a token on the page but absent from the runtime set FAILS and asks to
  be pruned.

Scope is deliberately the two token lists that are (a) enumerable in code
and (b) long enough that hand-maintenance rots. The prose sections —
materials, recorders, solvers, the constraint verbs — are not scanned:
they name commands, not entries in a frozenset, so there is nothing to
compare them against. Adding a fork *material* still needs a manual doc
edit.

The markers are HTML comments so they render as nothing:

    <!-- capability-map:elements -->
    ...tokens...
    <!-- /capability-map:elements -->
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from apeGmsh.opensees.emitter.live import (
    _FORK_ONLY_ELEMENTS,
    _FORK_ONLY_INTEGRATORS,
)

_DOC = (
    Path(__file__).resolve().parents[1]
    / "docs" / "concepts" / "backend-capabilities.md"
)

#: section marker -> the live set it must reproduce.
_SECTIONS = {
    "elements": _FORK_ONLY_ELEMENTS,
    "integrators": _FORK_ONLY_INTEGRATORS,
}


def _documented(section: str) -> set[str]:
    """Backtick-quoted tokens inside one ``capability-map`` block."""
    text = _DOC.read_text(encoding="utf-8")
    block = re.search(
        rf"<!-- capability-map:{section} -->(.*?)<!-- /capability-map:{section} -->",
        text,
        re.S,
    )
    assert block is not None, (
        f"{_DOC.name} has no <!-- capability-map:{section} --> block. "
        "The drift lane keys off those markers; restore them (or drop "
        "this section from _SECTIONS if the page really lost it)."
    )
    return set(re.findall(r"`([A-Za-z0-9_]+)`", block.group(1)))


def test_doc_exists() -> None:
    """A positive control: the whole lane is vacuous if the page moves."""
    assert _DOC.is_file(), (
        f"{_DOC} is missing. If the capability map moved, update _DOC — "
        "do not delete this lane; a stale fork list is a silent trap for "
        "anyone on a stock build."
    )


@pytest.mark.parametrize("section", sorted(_SECTIONS))
def test_capability_map_matches_runtime(section: str) -> None:
    live = set(_SECTIONS[section])
    doc = _documented(section)

    missing = live - doc
    extra = doc - live
    assert not missing, (
        f"{sorted(missing)} are gated as fork-only in the code but are not "
        f"listed in the '{section}' block of {_DOC.name}. A user on stock "
        "openseespy will hit the gate with no warning from the docs — add "
        "them to the page."
    )
    assert not extra, (
        f"{sorted(extra)} are listed as fork-only in the '{section}' block "
        f"of {_DOC.name} but are not in the live gate set. Either the gate "
        "lost them (a silent-acceptance regression) or they became "
        "available on stock and the page should drop them."
    )


def test_sections_are_non_trivial() -> None:
    """Guard the regex itself: an empty parse would pass both directions
    only if the live set were also empty, but a marker typo that yields
    an empty *live* lookup would go unnoticed."""
    for section, live in _SECTIONS.items():
        assert live, f"_SECTIONS[{section!r}] is empty — wrong import?"
        assert _documented(section), (
            f"the '{section}' block parsed to zero tokens; the page must "
            "list them as `backticked` names for the lane to see them."
        )
