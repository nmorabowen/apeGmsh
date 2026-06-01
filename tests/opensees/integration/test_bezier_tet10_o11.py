"""O11 locking test — Gmsh ``tet10`` order == ``BezierTet10`` control-point order.

The Bézier plan's #1 correctness risk (O11): a wrong mid-edge node order
yields a valid-looking but **wrong** stiffness. ``BezierTet10`` reuses
``TenNodeTetrahedron``'s ``node_reorder={11: identity}`` on the argument
that the Gmsh second-order tetrahedron (MSH element type 11) edge order is
byte-identical to the element's expected control-point order.

This test **locks** that identity by meshing a straight-sided solid to
``tet10`` and asserting each of the 6 mid-edge nodes (connectivity
positions 5-10) sits at the midpoint of its corner-pair edge under the
order::

    N5=mid(1,2)  N6=mid(2,3)  N7=mid(1,3)  N8=mid(1,4)  N9=mid(3,4)  N10=mid(2,4)

A pass confirms node ordering **and** straight-sidedness simultaneously
(the same check that caught the Tri6 sibling at 1.78e-15). A fail on a
specific mid-edge position points straight at the offending edge.

Mesh-only — needs gmsh, **no fork build** (the element is never run here;
only apeGmsh's emitted connectivity order is under test). Vertex winding /
Jacobian sign is a non-issue by construction: ``BezierTet10.cpp`` integrates
with ``fabs(detJ)``, so a flipped tet cannot corrupt the stiffness — the
midpoint-only check is sufficient (see the plan's O11 note).
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh import apeGmsh

# Gmsh tet10 (MSH type 11) mid-edge order, as 0-based corner-index pairs.
# Connectivity columns 4..9 must be the midpoints of these edges.
_TET10_EDGES = [(0, 1), (1, 2), (0, 2), (0, 3), (2, 3), (1, 3)]


def test_tet10_midedge_nodes_at_corner_pair_midpoints() -> None:
    g = apeGmsh(model_name="o11")
    g.begin()
    g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, label="body")
    g.physical.add_volume("body", name="Body")
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(dim=3)
    g.mesh.generation.set_order(2)  # quadratic → tet10 (MSH type 11)
    fem = g.mesh.queries.get_fem_data(dim=3)

    conn = fem.elements.connectivity  # (E, 10) node IDs, single type
    assert conn.shape[1] == 10, f"expected tet10, got {conn.shape[1]} nodes/elem"

    ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    coords = np.asarray(fem.nodes.coords, dtype=np.float64)
    id_to_coord = {int(i): coords[k] for k, i in enumerate(ids)}

    max_dev = 0.0
    for row in conn:
        x = [id_to_coord[int(n)] for n in row]
        for mid_idx, (a, b) in enumerate(_TET10_EDGES, start=4):
            edge_len = float(np.linalg.norm(x[a] - x[b]))
            assert edge_len > 0.0
            expected = 0.5 * (x[a] + x[b])
            dev = float(np.linalg.norm(x[mid_idx] - expected)) / edge_len
            max_dev = max(max_dev, dev)

    # Machine-precision agreement locks the identity ordering verdict.
    assert max_dev < 1e-12, (
        f"tet10 mid-edge nodes deviate from corner-pair midpoints by "
        f"{max_dev:.2e} (rel) — the Gmsh tet10 order does NOT match the "
        f"BezierTet10 control-point order; the node_reorder identity is wrong."
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
