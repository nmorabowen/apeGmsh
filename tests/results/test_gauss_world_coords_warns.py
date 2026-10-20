"""WarnGaussCoordsApproximate — degraded GP reconstructions are loud.

``compute_global_coords_from_arrays`` has two locked, reasonable
degradations: the centroid+bbox approximation for element types with
no shape-function coverage, and NaN rows for elements it cannot resolve
against the FEMData (#2317; they were parked at the origin, a plausible
coordinate). Both used to be silent — a
mis-placed marker cloud indistinguishable from a correct one (the
ADR 0056 INV-6 bug class). These tests pin the aggregated warning
and that the supported paths stay silent.
"""
from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh.results._gauss_world_coords import (
    WarnGaussCoordsApproximate,
    _world_via_bbox,
    compute_global_coords_from_arrays,
)


class _FakeGroup:
    def __init__(self, element_type, eids, conn) -> None:
        self.element_type = element_type
        self.ids = np.asarray(eids, dtype=np.int64)
        self.connectivity = np.asarray(conn, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.ids)


def _fake_fem(*, code, dim, npe, node_ids, coords, eids, conn,
              gmsh_name="", order=1):
    group = _FakeGroup(
        SimpleNamespace(
            code=code, dim=dim, npe=npe, order=order, gmsh_name=gmsh_name,
        ),
        eids, conn,
    )
    return SimpleNamespace(
        nodes=SimpleNamespace(
            ids=np.asarray(node_ids, dtype=np.int64),
            coords=np.asarray(coords, dtype=np.float64),
        ),
        elements=[group],
    )


# A 5-node pyramid (Gmsh code 7): not in the shape-function catalog
# (which covers lines/tris/quads/tets/hexes/wedges incl. P2) and
# (dim=3, npe=5) has no linear-counterpart mapping, so the
# reconstruction takes the bbox approximation.
_PYRAMID_NODES = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0.5, 0.5, 2.0],
], dtype=np.float64)


def test_bbox_fallback_warns_with_count_and_type_code():
    fem = _fake_fem(
        code=7, dim=3, npe=5,
        node_ids=[1, 2, 3, 4, 5], coords=_PYRAMID_NODES,
        eids=[10], conn=[[1, 2, 3, 4, 5]],
    )
    nat = np.array([[0.2, 0.2, 0.1], [0.1, 0.1, 0.5]])
    with pytest.warns(
        WarnGaussCoordsApproximate,
        match=r"2 GP\(s\).*centroid\+bbox.*\[7\]",
    ):
        out = compute_global_coords_from_arrays(
            np.array([10, 10]), nat, fem,
        )
    # The fallback values themselves are unchanged by the warning.
    expected = np.array([
        _world_via_bbox(nat[k], _PYRAMID_NODES) for k in range(2)
    ])
    np.testing.assert_allclose(out, expected)


def test_unresolved_element_warns_and_names_eids():
    fem = _fake_fem(
        code=3, dim=2, npe=4,
        node_ids=[1, 2, 3, 4],
        coords=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        eids=[10], conn=[[1, 2, 3, 4]],
    )
    with pytest.warns(
        WarnGaussCoordsApproximate, match=r"NaN \(not placed\).*\[99\]",
    ):
        out = compute_global_coords_from_arrays(
            np.array([99]), np.array([[0.0, 0.0]]), fem,
        )
    assert out.shape == (1, 3)
    assert np.isnan(out).all()


def test_supported_types_stay_silent():
    fem = _fake_fem(
        code=3, dim=2, npe=4,
        node_ids=[1, 2, 3, 4],
        coords=[[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]],
        eids=[10], conn=[[1, 2, 3, 4]],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", WarnGaussCoordsApproximate)
        out = compute_global_coords_from_arrays(
            np.array([10]), np.array([[0.0, 0.0]]), fem,
        )
    # Bilinear centre of the unit-2 quad.
    np.testing.assert_allclose(out, [[1.0, 1.0, 0.0]])
