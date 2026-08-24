"""ADR 0098 A6 G1 — nothing the old window could do goes missing quietly.

S6a made ``session().show()`` the only window a user gets. The old
``ResultsViewer`` installs five interactor-level capabilities; the
session pane ported two. Navigation and picking came across; the
legend gesture, the clip gizmo and the scope gizmo did not — and
nothing said so. They were found one at a time, months later, by a
person opening a model and noticing a gesture no longer worked.

That is the failure this guard exists to prevent, and it is a
COMPLETENESS failure, not a testing one: there was no artifact anywhere
listing what the old window could do, so nobody could tell what the new
one was missing.

So: every ``install_*`` capability in ``results_viewer.py`` must be
declared here with a status. Adding one to the old viewer without
declaring it fails this test, and so does declaring one that no longer
exists. The registry is the inventory the migration never had.

This guard checks that each capability is *accounted for*, which is a
source-level claim. Whether the session counterpart actually works is
the job of the tests named beside it.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src" / "apeGmsh" / "viewers"
OLD_VIEWER = _SRC / "results_viewer.py"
SESSION_PKG = _SRC / "session"

#: status -> meaning
#:   "ported"   — the session path has a counterpart; ``where`` names it
#:   "retired"  — deliberately not carried over; ``why`` says so
#:   "missing"  — known gap, not yet closed; ``why`` names the owner
#:
#: A "missing" entry is a promise, not an excuse: it keeps the gap in
#: the inventory so it cannot be rediscovered as a surprise.
CAPABILITIES: "dict[str, dict[str, str]]" = {
    "install_navigation": {
        "status": "ported",
        "where": "session/_pane.py::apply_pane_navigation",
        "tested_by": "tests/viewers/test_pane_selection.py",
    },
    "install_results_pick": {
        "status": "ported",
        "where": "session/_pick.py::PanePick",
        "tested_by": "tests/viewers/test_pane_selection.py",
    },
    "install_legend_interactor": {
        "status": "ported",
        "where": "session/_legend_bind.py::PaneLegendBinding",
        "tested_by": "tests/viewers/test_legend_binding.py",
        "note": "ADR 0098 Amendment 5 — needed placement as session "
                "state, not just the install.",
    },
    "install_clip_gizmo_interactor": {
        "status": "missing",
        "why": "ADR 0098 A6.6 R1 — the clip-plane gizmo is unreachable "
               "in the session window. NOT wiring: A5.5 said it was and "
               "A6.6 retracted that. The session emits no gizmo actors "
               "at all, so there is nothing for an interactor to grab; "
               "realize must draw them and something must own them. "
               "TWO of the three pieces have landed: ViewClipController "
               "(the controller) and the A7 reclip fast path, which "
               "fixed the drag-frame cost this work MEASURED -- clips "
               "were in the STRUCTURE half of _pane_signature, so one "
               "drag frame was a full realize at 240.8 ms; now 18.2 ms, "
               "at the orbit floor. What is still owed is the actors "
               "and the binding: the renderer must be added through "
               "LedgerBackend.inner (or _teardown sweeps it) and the "
               "interactor re-seated after every legend re-install, or "
               "it silently wins presses over the colour bar. See "
               "internal_docs/design/adr0098_r1_gizmo_brief.md.",
    },
    "install_scope_gizmo_interactor": {
        "status": "missing",
        "why": "ADR 0098 A6.6 R1 — same gap as the clip gizmo, and the "
               "same retraction applies: no scope gizmo actors are "
               "emitted on the session path either.",
    },
}

_VALID_STATUS = {"ported", "retired", "missing"}


def _installed_capabilities(path: Path) -> "set[str]":
    """Every ``install_*`` FUNCTION called in ``path``.

    Method calls (``self._install_legend_interactor()``) are the old
    viewer's own private wrappers, not capabilities — the capability is
    the free function the wrapper calls. Matching bare ``Name`` calls
    only is what keeps the registry about the seam rather than about
    one file's helper layout.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id.startswith("install_"):
            found.add(func.id)
    return found


def test_every_old_viewer_capability_is_declared():
    """The guard. A new ``install_*`` in the old viewer, or one deleted
    from it, must move the registry in the same commit."""
    found = _installed_capabilities(OLD_VIEWER)
    declared = set(CAPABILITIES)

    undeclared = sorted(found - declared)
    assert not undeclared, (
        f"{OLD_VIEWER.name} installs {undeclared} with no entry in "
        f"CAPABILITIES. Declare each one 'ported' (naming the session "
        f"counterpart), 'retired' (saying why), or 'missing' (naming "
        f"the owner) — an undeclared capability is exactly how the "
        f"legend, clip and scope gizmos were lost."
    )

    stale = sorted(declared - found)
    assert not stale, (
        f"CAPABILITIES declares {stale}, which {OLD_VIEWER.name} no "
        f"longer installs. Drop the entry — a stale inventory is worse "
        f"than none, because it reads as coverage."
    )


@pytest.mark.parametrize("name", sorted(CAPABILITIES))
def test_each_declaration_is_well_formed(name):
    entry = CAPABILITIES[name]
    status = entry.get("status")
    assert status in _VALID_STATUS, (
        f"{name}: status {status!r} is not one of {sorted(_VALID_STATUS)}."
    )
    if status == "ported":
        assert entry.get("where"), f"{name}: 'ported' needs a 'where'."
        assert entry.get("tested_by"), (
            f"{name}: 'ported' needs a 'tested_by' — 'the code exists' "
            f"is the claim that let three gestures ship unreachable."
        )
    else:
        assert entry.get("why"), f"{name}: {status!r} needs a 'why'."


@pytest.mark.parametrize(
    "name", sorted(
        n for n, e in CAPABILITIES.items() if e["status"] == "ported"
    ),
)
def test_a_ported_capability_names_a_module_that_exists(name):
    """'ported' must point at real code. Catches a counterpart that was
    renamed or removed, which would otherwise leave the inventory
    claiming coverage that evaporated."""
    where = CAPABILITIES[name]["where"]
    module = where.split("::")[0]
    assert module.startswith("session/"), (
        f"{name}: 'where' should name a session-path module, got "
        f"{where!r}."
    )
    path = SESSION_PKG / module.split("/", 1)[1]
    assert path.is_file(), f"{name}: {where!r} names no such file."
    if "::" in where:
        symbol = where.split("::", 1)[1]
        assert symbol in path.read_text(encoding="utf-8"), (
            f"{name}: {path.name} does not define {symbol!r}."
        )


def test_the_known_gaps_are_exactly_the_two_gizmos():
    """A change-detector on purpose. Closing R1 must delete these two
    entries, and this assertion is what makes that impossible to
    forget; a NEW 'missing' entry must be a deliberate edit here."""
    missing = sorted(
        n for n, e in CAPABILITIES.items() if e["status"] == "missing"
    )
    assert missing == [
        "install_clip_gizmo_interactor",
        "install_scope_gizmo_interactor",
    ]
