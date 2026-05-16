"""Tests for g.mesh.options — global Gmsh mesher option wrappers."""
import gmsh
import pytest


# ---------------------------------------------------------------------------
# set_subdivision_algorithm
# ---------------------------------------------------------------------------

def test_subdivision_algorithm_string(g):
    g.mesh.options.set_subdivision_algorithm("all_hex")
    assert int(gmsh.option.getNumber("Mesh.SubdivisionAlgorithm")) == 2


def test_subdivision_algorithm_int(g):
    g.mesh.options.set_subdivision_algorithm(2)
    assert int(gmsh.option.getNumber("Mesh.SubdivisionAlgorithm")) == 2


def test_subdivision_algorithm_invalid_string_raises(g):
    with pytest.raises(ValueError, match="unknown value 'turbo'"):
        g.mesh.options.set_subdivision_algorithm("turbo")


def test_subdivision_algorithm_round_trip(g):
    g.mesh.options.set_subdivision_algorithm("all_hex")
    assert g.mesh.options.get_subdivision_algorithm() == "all_hex"


# ---------------------------------------------------------------------------
# set_smoothing  (note: kwarg form, disambiguates from per-entity smoothing)
# ---------------------------------------------------------------------------

def test_smoothing(g):
    g.mesh.options.set_smoothing(iterations=5)
    assert int(gmsh.option.getNumber("Mesh.Smoothing")) == 5
    assert g.mesh.options.get_smoothing() == 5


# ---------------------------------------------------------------------------
# set_element_order
# ---------------------------------------------------------------------------

def test_element_order(g):
    g.mesh.options.set_element_order(2)
    assert int(gmsh.option.getNumber("Mesh.ElementOrder")) == 2
    assert g.mesh.options.get_element_order() == 2


# ---------------------------------------------------------------------------
# set_algorithm_2d / 3d
# ---------------------------------------------------------------------------

def test_algorithm_2d_string(g):
    g.mesh.options.set_algorithm_2d("frontal_quads")
    assert int(gmsh.option.getNumber("Mesh.Algorithm")) == 8
    assert g.mesh.options.get_algorithm_2d() == "frontal_quads"


def test_algorithm_2d_invalid_raises(g):
    with pytest.raises(ValueError, match="unknown value 'fast'"):
        g.mesh.options.set_algorithm_2d("fast")


def test_algorithm_3d_string(g):
    g.mesh.options.set_algorithm_3d("hxt")
    assert int(gmsh.option.getNumber("Mesh.Algorithm3D")) == 10
    assert g.mesh.options.get_algorithm_3d() == "hxt"


def test_algorithm_3d_int(g):
    g.mesh.options.set_algorithm_3d(10)
    assert g.mesh.options.get_algorithm_3d() == "hxt"


# ---------------------------------------------------------------------------
# set_recombination_algorithm
# ---------------------------------------------------------------------------

def test_recombination_algorithm(g):
    g.mesh.options.set_recombination_algorithm("blossom_full")
    assert int(gmsh.option.getNumber("Mesh.RecombinationAlgorithm")) == 3
    assert g.mesh.options.get_recombination_algorithm() == "blossom_full"


# ---------------------------------------------------------------------------
# Chaining
# ---------------------------------------------------------------------------

def test_chaining_returns_self(g):
    """Each setter returns self for fluent chaining."""
    result = (g.mesh.options
                .set_subdivision_algorithm("all_hex")
                .set_smoothing(iterations=3)
                .set_element_order(1)
                .set_algorithm_3d("hxt"))
    assert result is g.mesh.options


# ---------------------------------------------------------------------------
# Unknown int code → getter returns raw int (not a string)
# ---------------------------------------------------------------------------

def test_getter_returns_int_when_no_enum_match(g):
    """If gmsh has a value not in our enum table, getter returns the int."""
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 99)
    assert g.mesh.options.get_subdivision_algorithm() == 99


# ---------------------------------------------------------------------------
# Does not collide with per-entity g.mesh.structured.set_smoothing
# ---------------------------------------------------------------------------

def test_global_vs_per_entity_smoothing_dont_collide(g):
    """Both methods coexist — different namespaces, different effects."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    # Per-entity (on a curve)
    edges = g.model.queries.select("box", dim=1)
    g.mesh.structured.set_smoothing(edges.tags()[0], 5, dim=1)
    # Global
    g.mesh.options.set_smoothing(iterations=3)
    assert int(gmsh.option.getNumber("Mesh.Smoothing")) == 3
