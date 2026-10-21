"""Unplaced Gauss points are NaN for every cause, never the origin (#2317).

The origin is a plausible coordinate: a click near it would select an
unplaced element, a box over it would include one, and a marker would be
drawn there. NaN is not a coordinate, so a consumer that masks finite rows
cannot mistake it for one. Each cause is named by the one aggregated
:class:`WarnGaussCoordsApproximate`.

Oracle: the bilinear quad map is exact, so a placed row is the closed-form
image of its natural coordinates; every unplaced row is all-NaN.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh.results._gauss_world_coords import (
    WarnGaussCoordsApproximate,
    compute_global_coords_from_arrays,
)


class _FakeGroup:
    def __init__(self, element_type, eids, conn) -> None:
        self.element_type = element_type
        self.ids = np.asarray(eids, dtype=np.int64)
        self.connectivity = np.asarray(conn, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.ids)


_QUAD = SimpleNamespace(code=3, dim=2, npe=4, order=1, gmsh_name="Quadrangle 4")
# A unit-2 square at an offset, so no placed GP of it can sit at the origin.
_NODE_IDS = [1, 2, 3, 4]
_COORDS = [[1, 1, 0], [3, 1, 0], [3, 3, 0], [1, 3, 0]]


def _fem(*, node_ids=_NODE_IDS, coords=_COORDS, eids=(10,),
         conn=((1, 2, 3, 4),)):
    return SimpleNamespace(
        nodes=SimpleNamespace(
            ids=np.asarray(node_ids, dtype=np.int64),
            coords=np.asarray(coords, dtype=np.float64).reshape(-1, 3),
        ),
        elements=[_FakeGroup(_QUAD, list(eids), [list(c) for c in conn])],
    )


def _quad_world(nat):
    """Closed form of the bilinear map of the offset unit-2 square."""
    nat = np.asarray(nat, dtype=np.float64)
    return np.column_stack([2.0 + nat[:, 0], 2.0 + nat[:, 1],
                            np.zeros(len(nat))])


def test_missing_element_is_nan_beside_placed_rows():
    """A missing element's rows are NaN; the placed element's rows are its
    exact images, in slab order."""
    eids = np.array([10, 99, 10, 99])
    nat = np.array([[-0.5, 0.5], [0.0, 0.0], [0.5, -0.5], [0.1, 0.2]])
    with pytest.warns(WarnGaussCoordsApproximate, match=r"\[99\]"):
        out = compute_global_coords_from_arrays(eids, nat, _fem())
    np.testing.assert_allclose(out[[0, 2]], _quad_world(nat[[0, 2]]))
    assert np.isnan(out[[1, 3]]).all()


def test_unknown_node_is_nan():
    """Connectivity naming a node the FEMData lacks: NaN, named by eid."""
    fem = _fem(conn=((1, 2, 3, 7),))
    with pytest.warns(WarnGaussCoordsApproximate, match=r"unknown nodes.*\[10\]"):
        out = compute_global_coords_from_arrays(
            np.array([10]), np.array([[0.0, 0.0]]), fem,
        )
    assert np.isnan(out).all()


@pytest.mark.parametrize("bad", [-2, -1, 0])
def test_out_of_range_node_id_is_nan_not_a_wrapped_index(bad):
    """A negative or absent id must not index the lookup from its end and
    borrow a real node's coordinates."""
    fem = _fem(conn=((1, 2, 3, bad),))
    with pytest.warns(WarnGaussCoordsApproximate, match=r"\[10\]"):
        out = compute_global_coords_from_arrays(
            np.array([10]), np.array([[0.0, 0.0]]), fem,
        )
    assert np.isnan(out).all()


def test_fem_without_nodes_is_nan_and_warns():
    """No nodes at all used to return the origin silently."""
    fem = _fem(node_ids=[], coords=np.zeros((0, 3)))
    with pytest.warns(WarnGaussCoordsApproximate, match=r"\[10\]"):
        out = compute_global_coords_from_arrays(
            np.array([10, 10]), np.array([[0.0, 0.0], [0.5, 0.5]]), fem,
        )
    assert out.shape == (2, 3)
    assert np.isnan(out).all()


def test_no_natural_coords_is_nan():
    """The #2315 cause stays NaN beside the others."""
    with pytest.warns(WarnGaussCoordsApproximate, match=r"no natural"):
        out = compute_global_coords_from_arrays(
            np.array([10, 10]), np.array([[0.0, 0.0], [np.nan, np.nan]]),
            _fem(),
        )
    np.testing.assert_allclose(out[0], [2.0, 2.0, 0.0])
    assert np.isnan(out[1]).all()


def test_no_unplaced_row_is_finite():
    """Every row is either a closed-form placement or all-NaN: no row is
    partly finite, and none sits at the origin."""
    eids = np.array([10, 99, 10, 98, 10])
    nat = np.array([[0.0, 0.0], [0.0, 0.0], [np.nan, np.nan],
                    [0.0, 0.0], [1.0, 1.0]])
    with pytest.warns(WarnGaussCoordsApproximate):
        out = compute_global_coords_from_arrays(eids, nat, _fem())
    finite = np.isfinite(out).all(axis=1)
    assert (finite | np.isnan(out).all(axis=1)).all()
    np.testing.assert_array_equal(finite, [True, False, False, False, True])
    assert not (np.abs(out[finite]).sum(axis=1) == 0.0).any()
    np.testing.assert_allclose(out[finite], _quad_world(nat[finite]))


# --- #2327 review: three pre-existing silent-wrong paths -----------------


def test_negative_fem_node_id_does_not_wrap_onto_another_node():
    """A negative id in ``fem.nodes.ids`` used to be written at the end of
    the lookup, so connectivity naming node 3 read the -2 node's coords."""
    fem = _fem(
        node_ids=[1, 2, 3, -2],
        coords=[[1, 1, 0], [3, 1, 0], [3, 3, 0], [100, 100, 100]],
        conn=((1, 2, 3, 3),),
    )
    with pytest.warns(
        WarnGaussCoordsApproximate, match=r"negative.*node ids: \[-2\]",
    ):
        out = compute_global_coords_from_arrays(
            np.array([10]), np.array([[0.0, 0.0]]), fem,
        )
    # Degenerate quad 1-2-3-3 at its centre: the mean of its four rows.
    np.testing.assert_allclose(out, [[2.5, 2.0, 0.0]])


def test_override_length_mismatch_raises_by_name():
    """It used to fall back silently to the undeformed coords."""
    with pytest.raises(ValueError, match=r"node_coords_override has 3 row"):
        compute_global_coords_from_arrays(
            np.array([10]), np.array([[0.0, 0.0]]), _fem(),
            node_coords_override=np.zeros((3, 3)),
        )


def test_nan_node_coordinate_makes_the_whole_row_nan_and_warns():
    """A NaN z on one node used to give a row with finite x, y, NaN z and
    no warning."""
    coords = np.array(_COORDS, dtype=np.float64)
    coords[2, 2] = np.nan
    with pytest.warns(
        WarnGaussCoordsApproximate, match=r"non-finite node.*\[10\]",
    ):
        out = compute_global_coords_from_arrays(
            np.array([10]), np.array([[0.0, 0.0]]), _fem(coords=coords),
        )
    assert np.isnan(out).all()
