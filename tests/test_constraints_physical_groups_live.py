"""Build-phase regression: constraint verbs accept physical-group names
in a live, Parts-less model (the reported ``add_box`` + ``.to_physical``
workflow).

Before this fix ``g.constraints.tie("pgA", "pgB")`` on a PG-only model
raised ``KeyError: Part label 'pgA' not found in g.parts. Available:
[]`` — the constraint DSL was unusable without building Parts.  These
tests drive a real gmsh session and assert the tie / kinematic-coupling
resolve into records through the physical-group fallback in
``_resolve_faces`` / ``_resolve_nodes``.
"""
from __future__ import annotations

import pytest

from apeGmsh._core import apeGmsh


def test_tie_by_physical_group_resolves_in_live_model() -> None:
    """Two stacked, independently-meshed boxes tied by PG name."""
    with apeGmsh(model_name="pg_tie_build") as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="lower")
        g.model.geometry.add_box(0, 0, 1.02, 1, 1, 1, label="upper")
        g.model.sync()

        # Tag the two interface faces as physical groups (no Parts).
        g.model.select(dim=2).on_plane(
            (0, 0, 1.0), (0, 0, 1), tol=1e-3).to_physical("A_top")
        g.model.select(dim=2).on_plane(
            (0, 0, 1.02), (0, 0, 1), tol=1e-3).to_physical("B_bot")

        # Different sizes ⇒ non-matching interface meshes.
        g.mesh.sizing.set_size("lower", 0.5)
        g.mesh.sizing.set_size("upper", 0.34)
        g.mesh.generation.generate(3)

        g.constraints.tie("A_top", "B_bot", dofs=[1, 2, 3],
                          tolerance=0.2, stiffness=1e12)
        fem = g.mesh.queries.get_fem_data(dim=3)

    recs = list(fem.elements.constraints.interpolations())
    assert len(recs) > 0, "tie by PG name resolved to zero records"


def test_kinematic_coupling_by_physical_group_resolves() -> None:
    """RBE2 by PG name — exercises the node-side PG fallback."""
    with apeGmsh(model_name="pg_kc_build") as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="blk")
        g.model.sync()
        g.model.select(dim=2).on_plane(
            (0, 0, 1.0), (0, 0, 1), tol=1e-3).to_physical("top")
        g.model.select(dim=2).on_plane(
            (0, 0, 0.0), (0, 0, 1), tol=1e-3).to_physical("bot")
        g.mesh.sizing.set_size("blk", 0.5)
        g.mesh.generation.generate(3)

        g.constraints.kinematic_coupling(
            "top", "bot", master_point=(0.5, 0.5, 1.0), dofs=[1, 2, 3])
        fem = g.mesh.queries.get_fem_data(dim=3)

    recs = list(fem.nodes.constraints)
    assert len(recs) == 1
    assert recs[0].slave_nodes, "kinematic coupling bound no slave nodes"


def test_unknown_label_error_names_all_namespaces() -> None:
    """The rewritten KeyError names part / physical group / label
    instead of the old 'Available: []'."""
    with apeGmsh(model_name="pg_err") as g:
        with pytest.raises(KeyError, match="physical group"):
            g.constraints.tie("does_not_exist_a", "does_not_exist_b")
