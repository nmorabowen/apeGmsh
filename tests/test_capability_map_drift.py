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


# --------------------------------------------------------------------------
# `all` completeness — the extra that silently lost `mcp`.
# --------------------------------------------------------------------------
#: Extras deliberately NOT folded into ``all``, with the reason. Keyed so
#: the test fails loudly when a new extra appears and nobody decided which
#: side it belongs on — the exact way ``mcp`` slipped through for months.
_ALL_EXCLUSIONS = {
    "all": "the aggregate itself",
    "docs": "build-only, never a user runtime dep",
    "section-oracle": "dev-only oracle for the section analyzer (ADR 0078)",
    "vtk": "empty marker extra",
    "animation": "imageio-ffmpeg is a large binary payload",
    "partition-pymetis": "no PyPI Windows wheel; conda-forge only",
    "partition-networkx": "inert without nxmetis, which is git-install only",
}


def _extras() -> dict[str, list[str]]:
    import tomllib

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]


def test_all_extra_covers_every_included_extra() -> None:
    """Every extra not explicitly excluded must be a subset of ``all``."""
    extras = _extras()
    allset = set(extras["all"])
    for name, deps in sorted(extras.items()):
        if name in _ALL_EXCLUSIONS:
            continue
        missing = set(deps) - allset
        assert not missing, (
            f"extra '{name}' is not covered by [all]: {sorted(missing)} "
            "missing. `pip install apeGmsh[all]` is the line the README "
            "and every tutorial hand out, so a gap here fails at feature "
            "use rather than at install. Either add the requirements to "
            "'all' or register the extra in _ALL_EXCLUSIONS with a reason."
        )


def test_all_exclusions_registry_is_current() -> None:
    """The exclusion registry cannot rot in either direction."""
    extras = _extras()
    stale = set(_ALL_EXCLUSIONS) - set(extras)
    assert not stale, (
        f"_ALL_EXCLUSIONS names extras that no longer exist: {sorted(stale)}."
    )
    unclassified = set(extras) - set(_ALL_EXCLUSIONS)
    covered = {
        n for n in unclassified if not (set(extras[n]) - set(extras["all"]))
    }
    assert unclassified == covered, (
        f"new extra(s) {sorted(unclassified - covered)} are neither covered "
        "by [all] nor registered in _ALL_EXCLUSIONS — decide which."
    )
