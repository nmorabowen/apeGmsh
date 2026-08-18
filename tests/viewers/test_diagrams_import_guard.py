"""ADR 0098 S6b — who may still import ``apeGmsh.viewers.diagrams``.

S6 is **de-publication, not deletion**. The old Geometry / Composition /
Diagram ontology is no longer a public surface, but four hatches keep
importing it privately and are expected to:

* ``viewers/render.py`` — its closed view tokens;
* ``viewers/web_viewer.py`` — ``show_web`` holds the live
  ``ResultsDirector``, which is what ADR 0098 Amendment 2 rests the six
  slotless kinds on (see ``test_diagram_hatch_survival.py``);
* ``studio/_verbs.py`` — ``_yield_setup``, the 0095 INV-11 hatch;
* ``viewers/results_viewer.py`` — still constructed offscreen by
  ``Results.export_animation`` (also INV-11).

**Be honest about what this guard proves.** The allowlist is 27 modules
and eleven of them are alive *only* because the headless export reuses
the full Qt window (``_realize_headless`` hides the docks rather than
skipping their construction). So this is **not** "the surface is small".
It is a **ratchet**: a new importer must be argued for, and an entry
that stops importing must be pruned. Same two-way discipline as
``test_skill_docs_drift``'s signature registry — a registry that can
only grow rots.

Relative imports are resolved, deliberately. Inside ``viewers/`` almost
every real hit is ``from ..diagrams import X``; the sibling guard
``tests/assess/test_import_guard.py`` skips relative imports because
nothing in ``assess/`` can reach viewers relatively, and a guard here
that copied it would see almost nothing and pass forever.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
PKG = "apeGmsh.viewers.diagrams"

#: Modules allowed to import the retired ontology. Ratcheted BOTH ways.
#: Grouped by why they are here, because "why" is the thing that rots.
_ALLOWED: frozenset[str] = frozenset({
    # -- the four hatches --------------------------------------------
    "apeGmsh.viewers.render",
    "apeGmsh.viewers.web_viewer",
    "apeGmsh.studio._verbs",
    "apeGmsh.viewers.results_viewer",
    # -- the NEW session window (reuses per-kind emit code in place) --
    "apeGmsh.viewers.session._realize",
    "apeGmsh.viewers.session._specs",
    # -- model / mesh viewers: not retired at all, Dispatcher only ----
    "apeGmsh.viewers.mesh_viewer",
    "apeGmsh.viewers.model_viewer",
    # -- shared core -------------------------------------------------
    "apeGmsh.viewers.core._legend",           # live via the SESSION window
    "apeGmsh.viewers.core.visibility",        # live via mesh/model viewers
    "apeGmsh.viewers.core.overlay_visibility",
    "apeGmsh.viewers.core.element_visibility",  # transitive (director)
    "apeGmsh.viewers.core._clip_planes",        # transitive (registry)
    "apeGmsh.viewers.core.opacity_controller",  # transitive, fragile
    # -- alive ONLY because the headless export reuses the fat window;
    #    slimming export_animation kills this whole block at once -----
    "apeGmsh.viewers.animation",
    "apeGmsh.viewers.overlays.local_axes_overlay",
    "apeGmsh.viewers.overlays.probe_overlay",
    "apeGmsh.viewers.ui._diagram_settings_tab",
    "apeGmsh.viewers.ui._geometry_settings_panel",
    "apeGmsh.viewers.ui._outline_tree",
    "apeGmsh.viewers.ui._pick_readout_hud",
    "apeGmsh.viewers.ui._time_scrubber",
    "apeGmsh.viewers.ui._isochrone_panel",     # via make_side_panel only
    "apeGmsh.viewers.ui._section_panel",       # via make_side_panel only
    "apeGmsh.viewers.ui._thickness_panel",     # via make_side_panel only
    # -- DEAD, deletion deferred out of S6b --------------------------
    #    Reachable from nothing; removing them needs surgery inside the
    #    hatch-live results_viewer.py, which buys no behaviour. Delete
    #    with the export_animation slimming, and prune these two lines.
    "apeGmsh.viewers._session_apply",
    "apeGmsh.viewers.ui._time_history",
})

#: NOT allowlisted on purpose — ``apeGmsh.results.session.*`` is
#: forbidden outright by ``tests/results/session/test_session_pure.py``.
#: Listing it here would leave two guards contradicting each other, and
#: the stricter one should win.


# ---------------------------------------------------------------------
# The predicate (unit-testable without touching the tree)
# ---------------------------------------------------------------------

def is_allowed(module: str) -> bool:
    """Exact membership. Never a prefix test.

    ``startswith`` would admit a future ``apeGmsh.viewers.render_pack``
    on the strength of ``apeGmsh.viewers.render``; a package-keyed rule
    would admit any new ``viewers.session.*`` or ``viewers.ui.*``
    sibling. Both are near-misses this file asserts against below.
    """
    return module in _ALLOWED


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve(module: str, level: int, base: str) -> str:
    """Absolute target of a possibly-relative import."""
    if not level:
        return module or ""
    pkg = base.split(".")
    anchor = pkg[: len(pkg) - (level - 1)] if level > 1 else pkg
    return ".".join([*anchor, module]) if module else ".".join(anchor)


def imports_diagrams(source: str, base: str) -> bool:
    """Whether ``source`` really imports the retired package.

    A docstring that merely NAMES ``viewers.diagrams`` is not an import
    — ``results/session/_snapshot.py`` mentions it twice in prose and a
    scan that matched text would have wrongly flagged a module another
    guard forbids from importing it at all.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        targets: list[str] = []
        if isinstance(node, ast.Import):
            targets = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            targets = [_resolve(node.module or "", node.level or 0, base)]
        for target in targets:
            if target == PKG or target.startswith(PKG + "."):
                return True
    return False


def _src_modules() -> list[tuple[str, Path]]:
    out = []
    for path in sorted(SRC.rglob("*.py")):
        name = _module_name(path)
        if name.startswith(PKG):
            continue          # the package may import itself
        out.append((name, path))
    return out


def _actual_importers() -> set[str]:
    found = set()
    for name, path in _src_modules():
        base = name if path.name == "__init__.py" else name.rsplit(".", 1)[0]
        if imports_diagrams(path.read_text(encoding="utf-8"), base):
            found.add(name)
    return found


# ---------------------------------------------------------------------
# The gate, both directions
# ---------------------------------------------------------------------

def test_src_tree_is_present() -> None:
    """A moved package must fail loudly, not pass on zero files."""
    assert SRC.is_dir(), SRC
    assert (SRC / "apeGmsh" / "viewers" / "diagrams").is_dir()


def test_no_unlisted_module_imports_the_retired_ontology() -> None:
    leaks = sorted(_actual_importers() - _ALLOWED)

    assert not leaks, (
        "These modules import apeGmsh.viewers.diagrams but are not on "
        "the S6b allowlist:\n  " + "\n  ".join(leaks) + "\n\n"
        "ADR 0098 de-published that ontology. Either use the session "
        "surface (apeGmsh.results.session / apeGmsh.viewers.session), "
        "or argue the new hatch and add it above with its reason."
    )


def test_every_allowlist_entry_still_imports() -> None:
    """The other half of the ratchet — the registry cannot rot."""
    stale = sorted(_ALLOWED - _actual_importers())

    assert not stale, (
        "These are allowlisted but no longer import "
        "apeGmsh.viewers.diagrams:\n  " + "\n  ".join(stale) + "\n\n"
        "Prune them. An allowlist that only ever grows stops being "
        "evidence of anything."
    )


# ---------------------------------------------------------------------
# Near-misses — the discriminating cases
# ---------------------------------------------------------------------
#
# A refusal asserted only against input that was already refused for
# another reason proves nothing; that gap survived the first mutation
# pass in S5b AND S5c. So every probe below is a module that a
# PLAUSIBLE-but-wrong guard would admit, not one that is obviously
# absent.

@pytest.mark.parametrize("module", [
    # Unescaped-dot regex `apeGmsh.viewers.session\..*`: the `.` after
    # `viewers` eats the `_`. Also the "deleted module must not come
    # back" probe if the deferred deletion ever lands.
    "apeGmsh.viewers._session_apply_v2",
    # Sits BETWEEN two allowlisted siblings — catches a package-keyed
    # rule. Real file, currently clean of diagrams.
    "apeGmsh.viewers.session._reconciler",
    # Kills naive startswith: `render_pack` is already a real exported
    # symbol of render.py, so this is a plausible future split.
    "apeGmsh.viewers.render_pack",
    # Wildcard probes for `viewers.ui.*` / `viewers.core.*` rules.
    "apeGmsh.viewers.ui._details_panel",
    "apeGmsh.viewers.core.scope_controller",
    # The stricter guard's territory — must never be admitted here.
    "apeGmsh.results.session._snapshot",
])
def test_near_miss_modules_are_refused(module: str) -> None:
    assert not is_allowed(module)


@pytest.mark.parametrize("module", [
    "apeGmsh.viewers.session._realize",
    "apeGmsh.viewers.core._legend",
    "apeGmsh.viewers.render",
])
def test_allowlisted_modules_are_admitted(module: str) -> None:
    """The positive half. Without it the guard goes green by refusing
    everything, which is the cheapest way to fake a passing gate."""
    assert is_allowed(module)


# ---------------------------------------------------------------------
# Positive controls for the AST walk itself
# ---------------------------------------------------------------------

@pytest.mark.parametrize("source, base", [
    ("import apeGmsh.viewers.diagrams\n", "apeGmsh.viewers.x"),
    ("from apeGmsh.viewers.diagrams import Contour\n", "apeGmsh.viewers.x"),
    ("from apeGmsh.viewers.diagrams._director import D\n", "apeGmsh.x"),
    ("from ..diagrams import Contour\n", "apeGmsh.viewers.ui"),
    ("from .diagrams._kinds import kind_ids\n", "apeGmsh.viewers"),
])
def test_walk_catches_every_import_shape(source: str, base: str) -> None:
    assert imports_diagrams(source, base)


@pytest.mark.parametrize("source, base", [
    # Prose, not an import — the exact shape that would have wrongly
    # flagged results/session/_snapshot.py.
    ('"""Mentions apeGmsh.viewers.diagrams in a docstring."""\n', "a.b"),
    ('x = {"diagrams": 1}\nif "diagrams" in x: pass\n', "a.b"),
    # A DIFFERENT package that merely shares a prefix.
    ("from apeGmsh.viewers.diagrams_v2 import X\n", "apeGmsh.viewers"),
    # A relative import that resolves somewhere else entirely.
    ("from ..session import realize_pane\n", "apeGmsh.viewers.ui"),
])
def test_walk_does_not_flag_non_imports(source: str, base: str) -> None:
    assert not imports_diagrams(source, base)
