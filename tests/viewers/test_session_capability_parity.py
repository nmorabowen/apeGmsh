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
        "status": "ported",
        "where": "session/_clip_bind.py::PaneClipBinding",
        "tested_by": "tests/viewers/test_clip_binding.py",
        "note": "ADR 0098 Amendment 7 — and A5.5 was WRONG that this "
                "was wiring; A6.6 retracted it and the retraction was "
                "right. FOUR parts, none of which works alone: "
                "ViewClipController (the five-member contract over "
                "MeshView.clips), realize's reference_bounds, this "
                "binding (built on the RAW backend — the reconciler's "
                "LedgerBackend sweeps everything added through it — "
                "and re-seated after every legend re-bind so the "
                "colour bar keeps winning an overlap), and the "
                "reclip fast path, without which one drag frame cost "
                "a full realize at 240.8 ms (now 18.2). Reach is the "
                "inspector's Section planes section: a plane only "
                "Python could create would still be A6.1's defect.",
    },
    "install_scope_gizmo_interactor": {
        "status": "retired",
        "why": "ADR 0098 A7 Q5 — NOT the clip gizmo's twin, which is "
               "why it took a different disposition. The scope GIZMO "
               "drives a spatial axis-aligned BBox per geometry "
               "(core/scope_controller.py) and hides cells by writing "
               "vtkGhostType through ElementVisibility's LAYER_SCOPE; "
               "the session's Scope (§3) is a composition AXIS plus "
               "names — physical groups, materials, element types. "
               "Same word, unrelated concept. The session IR carries "
               "no spatial box and no active-geometry notion, so "
               "there is nothing to adapt: a port means new §3 IR, a "
               "snapshot field, a resolve-time filter and an "
               "inspector context — a §3/§4 widening argued at the "
               "ADR bar, not a slice. A2.2 is the precedent for "
               "recording a disposition rather than deleting working "
               "code: the old viewer keeps the gesture.",
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


def test_no_capability_is_still_missing():
    """R1 closed the last gap — the inventory is now complete.

    A change-detector on purpose, and it has already earned its keep
    twice: it failed when the clip gizmo flipped to ported, which is
    exactly the moment a stale "known gaps" list would otherwise have
    gone quietly wrong.

    The two entries left this list by different routes, and the
    difference is the point. The clip gizmo was BUILT (A7). The scope
    gizmo was RETIRED (A7 Q5) — a disposition is a legitimate way to
    close a gap, which is why 'retired' must carry a ``why`` a reader
    can disagree with rather than a status a reader must accept.

    A new 'missing' entry is now a deliberate edit here, and it should
    be: it means the session window lost something it used to do.
    """
    missing = sorted(
        n for n, e in CAPABILITIES.items() if e["status"] == "missing"
    )
    assert missing == [], (
        f"{len(missing)} capability/ies are unaccounted for: {missing}. "
        f"Either port one, or retire it with a reason."
    )
