"""PR6 — :func:`apeGmsh.viewers.mesh_viewer._needs_fem_for_overlays`.

The gate that decides whether mesh.viewer's ``show()`` should call
``parent.mesh.queries.get_fem_data()`` (~2.7 s on a 600 k-node mesh).

Pre-fix: gate checked composite *existence*
(``getattr(parent, 'loads', None) is not None``).  An empty
LoadsComposite the user instantiated but never populated tripped the
gate and cost the full broker build for zero overlay output.
Post-fix: gate checks ``*_defs`` population — same gate the tab
builders apply later.

These tests exercise the helper in isolation against duck-typed
stubs; the integration path (mesh_viewer.show) is exercised by the
existing mesh_viewer smoke suite.
"""
from __future__ import annotations

from types import SimpleNamespace

from apeGmsh.viewers.mesh_viewer import _needs_fem_for_overlays


def _parent(*, loads=None, masses=None, constraints=None):
    """Build a duck-typed parent stub.  Pass `None` to omit a composite
    (matching ``getattr(parent, 'loads', None) is None``); pass a
    SimpleNamespace with the relevant `*_defs` list to simulate a
    populated/empty composite."""
    return SimpleNamespace(
        loads=loads, masses=masses, constraints=constraints,
    )


# =====================================================================
# False — nothing to render
# =====================================================================


def test_mesh_only_parent_returns_false():
    """No composites at all → mesh-only model; skip the broker build."""
    parent = _parent()  # all three None
    assert _needs_fem_for_overlays(parent) is False


def test_all_empty_composites_returns_false():
    """The PR6 regression case: composites exist but are empty.
    Pre-fix this returned True and cost ~2.7 s for nothing."""
    parent = _parent(
        loads=SimpleNamespace(load_defs=[]),
        masses=SimpleNamespace(mass_defs=[]),
        constraints=SimpleNamespace(constraint_defs=[]),
    )
    assert _needs_fem_for_overlays(parent) is False


def test_composite_missing_defs_attribute_returns_false():
    """Defensive: a composite-shaped stub without the `*_defs`
    attribute returns False, not crashes."""
    parent = _parent(
        loads=SimpleNamespace(),
        masses=SimpleNamespace(),
        constraints=SimpleNamespace(),
    )
    assert _needs_fem_for_overlays(parent) is False


# =====================================================================
# True — at least one populated composite
# =====================================================================


def test_populated_loads_returns_true():
    parent = _parent(loads=SimpleNamespace(load_defs=["L1"]))
    assert _needs_fem_for_overlays(parent) is True


def test_populated_masses_returns_true():
    parent = _parent(masses=SimpleNamespace(mass_defs=["M1"]))
    assert _needs_fem_for_overlays(parent) is True


def test_populated_constraints_returns_true():
    parent = _parent(constraints=SimpleNamespace(constraint_defs=["C1"]))
    assert _needs_fem_for_overlays(parent) is True


def test_one_populated_one_empty_returns_true():
    """Short-circuits on first populated composite — empty siblings
    don't matter."""
    parent = _parent(
        loads=SimpleNamespace(load_defs=[]),
        masses=SimpleNamespace(mass_defs=["M1"]),
        constraints=SimpleNamespace(constraint_defs=[]),
    )
    assert _needs_fem_for_overlays(parent) is True
