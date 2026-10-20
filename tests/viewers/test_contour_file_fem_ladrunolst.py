"""A LadrunoLST discrete contour on the file's own FEMData paints its corners.

The fixture ``tests/fixtures/simplex_gp/tri.ladruno`` is one fork solve under
``u_i = x_i^2 / 2`` (``tests/results/test_gauss_simplex_coords.py``, SOLVE):
a quadratic element's recorded strain ``eps_xx`` at a Gauss point is that
point's ``x``. Extrapolated to the corners of the 6-node LadrunoLST triangle,
it must give each corner its own ``x``. Two seams meet here. The FEMData read
from the file must call the class a 2-D triangle, so the substrate draws a
3-corner cell rather than a 6-node wedge (#2303). The extrapolation must give
it 3 corner values from its Gauss positions (#2240). When the two disagree,
the shattered scatter skipped the cell, and it painted 0.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from apeGmsh.results import Results
from apeGmsh.viewers.diagrams import (
    ContourDiagram,
    ContourStyle,
    DiagramSpec,
    SlabSelector,
)
from apeGmsh.viewers.scene.fem_scene import build_fem_scene

_TRI = (Path(__file__).resolve().parents[1]
        / "fixtures" / "simplex_gp" / "tri.ladruno")
_LST = 4      # the LadrunoLST element in the fixture


def test_ladrunolst_discrete_contour_paints_its_corner_x(backend) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")    # reader provenance, not under test
        results = Results.from_ladruno(_TRI)
    with results:
        fem = results.fem
        scene = build_fem_scene(fem)
        d = ContourDiagram(
            DiagramSpec(
                kind="contour",
                selector=SlabSelector(component="strain_xx", ids=(_LST,)),
                style=ContourStyle(topology="gauss", averaging="discrete"),
            ),
            results,
        )
        d.attach(backend, fem, scene)
        assert d._effective_topology == "gauss_node_discrete"
        points = np.asarray(d._points.coords, dtype=np.float64)
        painted = np.asarray(d._scalar_values, dtype=np.float64)

    # Each shattered point painted its own x, and the cell is a triangle.
    # The substrate's points are float32, so the match is to float32.
    np.testing.assert_allclose(painted, points[:, 0], rtol=0, atol=1e-6)
    assert points.shape[0] == 3, f"cell has {points.shape[0]} points"
