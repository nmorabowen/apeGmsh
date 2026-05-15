"""Tests for Selection.parallel_to() and Selection.normal_along()."""
import math

import gmsh
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# parallel_to — curves only (dim=1)
# ---------------------------------------------------------------------------

def test_parallel_to_axis_alias_z(g):
    """parallel_to('z') keeps the 4 vertical edges of a box."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 2, label="box")
    edges = g.model.queries.select("box", dim=1)
    assert len(edges) == 12

    z_edges = edges.parallel_to("z")
    assert len(z_edges) == 4
    for dt in z_edges:
        _, _, zmin, _, _, zmax = g.model.queries.bounding_box(dt)
        assert abs((zmax - zmin) - 2.0) < 1e-5


def test_parallel_to_axis_alias_x_and_y(g):
    """parallel_to('x') and ('y') each keep their 4 edges."""
    g.model.geometry.add_box(0, 0, 0, 3, 2, 1, label="box")
    edges = g.model.queries.select("box", dim=1)
    assert len(edges.parallel_to("x")) == 4
    assert len(edges.parallel_to("y")) == 4
    assert len(edges.parallel_to("z")) == 4


def test_parallel_to_vector(g):
    """A vector direction filters the same as the matching axis alias."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    edges = g.model.queries.select("box", dim=1)
    by_alias  = edges.parallel_to("z")
    by_vector = edges.parallel_to((0.0, 0.0, 1.0))
    assert sorted(by_alias) == sorted(by_vector)


def test_parallel_to_anti_parallel_is_parallel(g):
    """A negative-direction vector matches the same edges as positive."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    edges = g.model.queries.select("box", dim=1)
    pos = edges.parallel_to((0, 0, 1))
    neg = edges.parallel_to((0, 0, -1))
    assert sorted(pos) == sorted(neg)


def test_parallel_to_unnormalized_vector(g):
    """Direction vectors do not need to be unit length."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    edges = g.model.queries.select("box", dim=1)
    a = edges.parallel_to((0, 0, 1))
    b = edges.parallel_to((0, 0, 17.3))
    assert sorted(a) == sorted(b)


def test_parallel_to_angle_tol(g):
    """A diagonal edge passes a wide tolerance and fails a tight one."""
    p1 = g.model.geometry.add_point(0, 0, 0)
    p2 = g.model.geometry.add_point(1, 0, 1)        # 45° from x-axis in xz-plane
    g.model.geometry.add_line(p1, p2, label="diag")
    edge = g.model.queries.select("diag", dim=1)

    assert len(edge.parallel_to("x", angle_tol=46.0)) == 1
    assert len(edge.parallel_to("x", angle_tol=44.0)) == 0


def test_parallel_to_returns_new_selection(g):
    """parallel_to returns a new Selection; original is unchanged."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    edges = g.model.queries.select("box", dim=1)
    z = edges.parallel_to("z")
    assert len(edges) == 12          # parent untouched
    assert len(z) == 4
    assert z is not edges


def test_parallel_to_chains_with_select(g):
    """Chains with .select() — bbox predicates compose with direction."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    edges = g.model.queries.select("box", dim=1)
    # Z-edges on the x=0 wall — should be 2 of the 4 vertical edges.
    result = edges.parallel_to("z").select(on={"x": 0})
    assert len(result) == 2


def test_parallel_to_wrong_dim_raises(g):
    """Calling parallel_to on dim=3 entities raises ValueError."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    vols = g.model.queries.select_all_volumes()
    with pytest.raises(ValueError, match=r"parallel_to.*dim=1"):
        vols.parallel_to("z")


def test_parallel_to_mixed_dim_raises(g):
    """Mixed-dim selection (select_all) raises with educational message."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    everything = g.model.queries.select_all()
    with pytest.raises(ValueError, match=r"parallel_to"):
        everything.parallel_to("z")


def test_parallel_to_unknown_axis_raises(g):
    """Unknown axis alias raises ValueError."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    edges = g.model.queries.select("box", dim=1)
    with pytest.raises(ValueError, match=r"Unknown axis"):
        edges.parallel_to("w")


def test_parallel_to_zero_vector_raises(g):
    """Zero direction vector raises ValueError."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    edges = g.model.queries.select("box", dim=1)
    with pytest.raises(ValueError, match=r"zero magnitude"):
        edges.parallel_to((0, 0, 0))


# ---------------------------------------------------------------------------
# normal_along — surfaces only (dim=2)
# ---------------------------------------------------------------------------

def test_normal_along_z_returns_top_and_bottom(g):
    """A unit cube has 2 faces whose normal is along z."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    faces = g.model.queries.select("box", dim=2)
    assert len(faces) == 6
    assert len(faces.normal_along("z")) == 2


def test_normal_along_each_axis_returns_two(g):
    """Each axis matches the two faces whose normal is along it."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    faces = g.model.queries.select("box", dim=2)
    assert len(faces.normal_along("x")) == 2
    assert len(faces.normal_along("y")) == 2
    assert len(faces.normal_along("z")) == 2


def test_normal_along_vector(g):
    """Vector direction matches the same faces as the axis alias."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    faces = g.model.queries.select("box", dim=2)
    a = faces.normal_along("z")
    b = faces.normal_along((0, 0, 1))
    assert sorted(a) == sorted(b)


def test_normal_along_wrong_dim_raises(g):
    """Calling normal_along on dim=1 entities raises ValueError."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    edges = g.model.queries.select("box", dim=1)
    with pytest.raises(ValueError, match=r"normal_along.*dim=2"):
        edges.normal_along("z")


def test_normal_along_chains_with_select(g):
    """normal_along + on= chains to find specific faces."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    faces = g.model.queries.select("box", dim=2)
    bottom = faces.normal_along("z").select(on={"z": 0})
    assert len(bottom) == 1
