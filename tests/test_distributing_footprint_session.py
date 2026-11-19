"""``distributing_coupling(weighting="footprint")`` through a live gmsh
session: a column-section footprint on a meshed slab face whose edges do
not follow it. The resolved RBE3 weights are the footprint's overlap with
each face, so they sum to the footprint area and their moment about the
footprint centroid is zero (the load of a uniform pressure on it)."""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh import apeGmsh


def _footprint(cx: float, cy: float, hx: float, hy: float, z: float):
    return [(cx - hx, cy - hy, z), (cx + hx, cy - hy, z),
            (cx + hx, cy + hy, z), (cx - hx, cy + hy, z)]


def test_footprint_rbe3_on_a_meshed_slab():
    cx, cy, hx, hy = 1.37, 0.91, 0.30, 0.20
    with apeGmsh(model_name="rbe3_footprint", verbose=False) as g:
        g.model.geometry.add_rectangle(0.0, 0.0, 0.0, 3.0, 2.0, label="slab")
        g.physical.add_surface("slab", name="Slab")
        ref = g.decouple_node(coords=(cx, cy, 0.0), label="col")
        g.constraints.distributing_coupling(
            ref, "Slab", weighting="footprint",
            footprint=_footprint(cx, cy, hx, hy, 0.0), name="col_rbe3")
        g.mesh.sizing.set_global_size(0.25)
        g.mesh.generation.generate(dim=2)
        fem = g.mesh.queries.get_fem_data(dim=2)
    recs = list(fem.elements.constraints.interpolations())
    assert len(recs) == 1
    rec = recs[0]
    assert rec.slave_node == ref.tag
    w = np.asarray(rec.weights)
    assert w.sum() == pytest.approx(4 * hx * hy, rel=1e-9)
    pos = {int(t): i for i, t in enumerate(fem.nodes.ids)}
    xy = np.array([fem.nodes.coords[pos[int(t)]][:2] for t in rec.master_nodes])
    # a node's share sits at the node, not at the overlap's centroid, so the
    # moment is zero only to the mesh size; it is small against the area
    m = (w[:, None] * (xy - (cx, cy))).sum(axis=0) / w.sum()
    assert np.abs(m).max() < 0.25
    # every independent touches the footprint's faces: none lies far off it
    assert np.all(np.abs(xy[:, 0] - cx) <= hx + 0.5)
    assert np.all(np.abs(xy[:, 1] - cy) <= hy + 0.5)


def test_footprint_needs_the_polygon():
    with apeGmsh(model_name="rbe3_footprint_missing", verbose=False) as g:
        with pytest.raises(ValueError, match="needs footprint"):
            g.constraints.distributing_coupling("A", "B", weighting="footprint")
