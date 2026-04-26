"""
Tests -- Cross-composite target resolution convention.

Guarantees the two rules documented in
``examples/EOS Examples/05_labels_and_pgs.py`` :

1. When the same name is registered as both a **label** and a
   **physical group**, ``target="..."`` resolves to the **label**
   entity on both the ``FEMData`` side (``fem.nodes.get`` /
   ``fem.elements.get``) and the ``g.loads`` / ``g.constraints`` side
   (``LoadsComposite._resolve_target``).

2. ``target=[(dim, tag)]`` is interpreted as a **raw Gmsh geometry
   DimTag list** on both sides, not as a physical-group tag lookup.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------
# Scenario: a name registered in BOTH namespaces on different entities
# ------------------------------------------------------------------
#
# Two points exist.  The point ``p_lbl`` is tagged with the label
# "foo"; the point ``p_pg`` is tagged with a physical group also
# named "foo".  Asking for ``target="foo"`` must return the *label*
# entity (p_lbl), not the PG entity (p_pg).


def _build_collision_scene(g, *, lc: float = 1.0):
    """Return (p_lbl, p_pg, beam_line) with a label/PG name collision."""
    p_lbl = g.model.geometry.add_point(0.0, 0.0, 0.0, lc=lc)
    p_pg  = g.model.geometry.add_point(1.0, 0.0, 0.0, lc=lc)
    p_tip = g.model.geometry.add_point(2.0, 0.0, 0.0, lc=lc)

    beam_line_L = g.model.geometry.add_line(p_lbl, p_pg)
    beam_line_R = g.model.geometry.add_line(p_pg,  p_tip)

    g.model.sync()

    # Same name registered in both tiers on DIFFERENT entities.
    g.labels.add(0, [p_lbl], "foo")
    g.physical.add(0, [p_pg],  name="foo")
    # Needed so the mesh actually contains the points as 1-D elements.
    g.physical.add(1, [beam_line_L, beam_line_R], name="beam")

    g.mesh.generation.generate(1)
    return p_lbl, p_pg


# ------------------------------------------------------------------
# FEMData side
# ------------------------------------------------------------------


def test_fem_nodes_target_prefers_label(g):
    """``fem.nodes.get(target="foo")`` returns the LABEL's node."""
    p_lbl, p_pg = _build_collision_scene(g)
    fem = g.mesh.queries.get_fem_data()

    via_label  = sorted(int(n) for n in fem.nodes.get(label="foo").ids)
    via_pg     = sorted(int(n) for n in fem.nodes.get(pg="foo").ids)
    via_target = sorted(int(n) for n in fem.nodes.get(target="foo").ids)

    assert via_label != via_pg, (
        "Test scenario broken: label and PG resolved to the same "
        "node set, so collision precedence cannot be observed."
    )
    assert via_target == via_label, (
        f"target='foo' resolved to {via_target}, expected label-match "
        f"{via_label} (PG was {via_pg})"
    )


def test_fem_nodes_target_raw_dimtag(g):
    """``target=[(dim, tag)]`` is a raw geometry DimTag, not a PG tag."""
    p_lbl, _ = _build_collision_scene(g)
    fem = g.mesh.queries.get_fem_data()

    via_dimtag = sorted(
        int(n) for n in fem.nodes.get(target=[(0, p_lbl)]).ids)
    via_label  = sorted(int(n) for n in fem.nodes.get(label="foo").ids)

    assert via_dimtag == via_label, (
        f"target=[(0, {p_lbl})] should resolve to the mesh nodes of "
        f"geometry point {p_lbl} (same as label='foo'), "
        f"got {via_dimtag} vs {via_label}"
    )


# ------------------------------------------------------------------
# LoadsComposite side
# ------------------------------------------------------------------


def test_loads_resolve_target_prefers_label(g):
    """``g.loads._resolve_target('foo')`` returns the LABEL's DimTag."""
    p_lbl, p_pg = _build_collision_scene(g)

    dts = g.loads._resolve_target("foo", source="auto")
    # Must be the geometry (dim, tag) of the LABEL'd point.
    assert dts == [(0, int(p_lbl))], (
        f"_resolve_target('foo') returned {dts}, expected "
        f"label entity [(0, {p_lbl})]"
    )


def test_loads_resolve_target_raw_dimtag_passthrough(g):
    """``_resolve_target([(0, t)])`` returns the list as-is."""
    p_lbl, _ = _build_collision_scene(g)

    dts = g.loads._resolve_target([(0, int(p_lbl))], source="auto")
    assert dts == [(0, int(p_lbl))]
