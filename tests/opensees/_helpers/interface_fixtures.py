"""Shared Gmsh fixture for the ADR 0093 S10 acceptance battery.

The two-squares strip is trivial enough that every interface test file
re-declares it (the repo's standing convention). The quarter-annulus
tunnel is not, and the battery needs the *same* one on both lanes — the
deck-level file asserts its per-pair frames, the engine file runs it —
so it lives here rather than in two places that could drift apart.

Users: ``tests/opensees/integration/test_interface_acceptance_battery.py``
and ``tests/opensees/subprocess/test_interface_acceptance_engine.py``.
"""
from __future__ import annotations

import gmsh


def tunnel_geometry(g, *, n_arc: int = 7, n_rad: int = 3) -> None:
    """Quarter-annulus tunnel: rock ``r∈[1,2]`` AROUND the opening with
    a liner ``r∈[0.85,1]`` inside it, node-for-node coincident on
    ``r=1``.

    The master (``"face"``) is the rock's INNER rim — a **concave**
    master with the material outside it, so outward-of-material points
    radially *inward*, toward the liner. That is the Cerro Lindo shape
    and the curvature mirror of a convex arc: a sign rule that leaned
    on curvature sense or winding would pass on a convex arc and fail
    here (the kernel-level twin is
    ``test_tunnel_master_normals_point_into_the_opening``).

    Physical groups: surfaces ``rock`` / ``liner``; curves ``face``
    (master, r=1 on the rock), ``wire`` (slave, r=1 on the liner),
    ``ground`` (the rock's outer rim, r=2) and ``intrados`` (the
    liner's inner rim, r=0.85).

    The arcs' centre point is created through **raw OCC** and deleted
    once the arcs exist. Created through ``g.model.geometry.add_point``
    it would be a registered, "user-intentional" entity that the orphan
    sweep keeps, and Gmsh would mesh it as a standalone vertex: a node
    with two DOFs, no element, and no support — a singular stiffness
    matrix at the first solve. (Measured before the fix: ``WARNING:
    numeric analysis returns 1 -- Umfpackgenlinsolver::solve``.) An OCC
    circular-arc edge does not carry its centre in its topology, so the
    removal is safe.
    """
    G = g.model.geometry
    o = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)
    gmsh.model.occ.synchronize()

    a1, a2 = G.add_point(1.0, 0.0, 0.0), G.add_point(0.0, 1.0, 0.0)
    b1, b2 = G.add_point(2.0, 0.0, 0.0), G.add_point(0.0, 2.0, 0.0)
    arc_in = G.add_arc(a1, o, a2)
    arc_out = G.add_arc(b2, o, b1)
    rock = G.add_plane_surface(G.add_curve_loop(
        [arc_in, G.add_line(a2, b2), arc_out, G.add_line(b1, a1)]))

    # The liner's OWN points on r=1 — never fragmented against the
    # rock's, so the two rims carry coincident-but-distinct nodes.
    d1, d2 = G.add_point(1.0, 0.0, 0.0), G.add_point(0.0, 1.0, 0.0)
    c1, c2 = G.add_point(0.85, 0.0, 0.0), G.add_point(0.0, 0.85, 0.0)
    arc_o_l = G.add_arc(d1, o, d2)
    arc_i_l = G.add_arc(c2, o, c1)
    liner = G.add_plane_surface(G.add_curve_loop(
        [arc_o_l, G.add_line(d2, c2), arc_i_l, G.add_line(c1, d1)]))

    gmsh.model.occ.remove([(0, o)])
    gmsh.model.occ.synchronize()

    st = g.mesh.structured
    arcs = (arc_in, arc_out, arc_o_l, arc_i_l)
    for c in arcs:
        st.set_transfinite_curve(c, n_arc)
    for _dim, tag in gmsh.model.getEntities(1):
        if tag not in arcs:
            st.set_transfinite_curve(tag, n_rad)
    st.set_transfinite_surface(rock)
    st.set_transfinite_surface(liner)
    st.set_recombine([(2, rock), (2, liner)])
    g.mesh.generation.generate(2)

    g.physical.add(2, [rock], name="rock")
    g.physical.add(2, [liner], name="liner")
    g.physical.add(1, [arc_in], name="face")
    g.physical.add(1, [arc_o_l], name="wire")
    g.physical.add(1, [arc_out], name="ground")
    g.physical.add(1, [arc_i_l], name="intrados")
