"""Introspection on ``Recorders`` — categories, components, shorthands.

These tests cover the static introspection methods that let users
discover what they can declare without leaving the REPL or the docs
site.
"""
from __future__ import annotations

import pytest

from apeGmsh.solvers.Recorders import Recorders


# =====================================================================
# categories()
# =====================================================================

def test_categories_returns_seven() -> None:
    cats = Recorders.categories()
    assert isinstance(cats, tuple)
    assert set(cats) == {
        "nodes", "elements", "line_stations", "gauss",
        "fibers", "layers", "modal",
    }


# =====================================================================
# components_for()
# =====================================================================

def test_components_for_nodes_includes_kinematics() -> None:
    comps = Recorders.components_for("nodes")
    assert "displacement_x" in comps
    assert "displacement_y" in comps
    assert "displacement_z" in comps
    assert "rotation_x" in comps
    assert "velocity_x" in comps
    assert "acceleration_x" in comps


def test_components_for_nodes_includes_forces_and_reactions() -> None:
    comps = Recorders.components_for("nodes")
    assert "force_x" in comps
    assert "moment_z" in comps
    assert "reaction_force_x" in comps
    assert "reaction_moment_z" in comps
    assert "pore_pressure" in comps


def test_components_for_gauss_has_stress_and_strain() -> None:
    comps = Recorders.components_for("gauss")
    assert "stress_xx" in comps
    assert "stress_xy" in comps
    assert "strain_yy" in comps
    assert "von_mises_stress" in comps


def test_components_for_line_stations_has_diagrams() -> None:
    comps = Recorders.components_for("line_stations")
    assert set(comps) == {
        "axial_force", "shear_y", "shear_z", "torsion",
        "bending_moment_y", "bending_moment_z",
    }


def test_components_for_fibers_has_root_canonicals() -> None:
    comps = Recorders.components_for("fibers")
    assert "fiber_stress" in comps
    assert "fiber_strain" in comps


def test_components_for_modal_is_empty() -> None:
    assert Recorders.components_for("modal") == ()


def test_components_for_unknown_raises() -> None:
    with pytest.raises(KeyError, match="Unknown category"):
        Recorders.components_for("not-a-category")


def test_components_for_returns_sorted() -> None:
    comps = Recorders.components_for("gauss")
    assert list(comps) == sorted(comps)


# =====================================================================
# shorthands_for()
# =====================================================================

def test_shorthands_for_nodes_includes_displacement() -> None:
    sh = Recorders.shorthands_for("nodes")
    assert "displacement" in sh
    assert sh["displacement"] == (
        "displacement_x", "displacement_y", "displacement_z",
    )


def test_shorthands_for_nodes_includes_reaction_megashorthand() -> None:
    sh = Recorders.shorthands_for("nodes")
    assert "reaction" in sh
    # Forces + moments
    assert "reaction_force_x" in sh["reaction"]
    assert "reaction_moment_z" in sh["reaction"]


def test_shorthands_for_gauss_includes_stress_strain() -> None:
    sh = Recorders.shorthands_for("gauss")
    assert "stress" in sh
    assert "strain" in sh
    assert sh["stress"][:3] == ("stress_xx", "stress_yy", "stress_zz")


def test_shorthands_for_line_stations_includes_section_force() -> None:
    sh = Recorders.shorthands_for("line_stations")
    assert "section_force" in sh
    assert "axial_force" in sh["section_force"]


def test_shorthands_for_modal_is_empty() -> None:
    assert Recorders.shorthands_for("modal") == {}


def test_shorthands_for_unknown_returns_empty() -> None:
    assert Recorders.shorthands_for("nope") == {}


# =====================================================================
# Cross-check: shorthand expansions are subsets of the category's
# canonical components (i.e. the introspection is self-consistent).
# =====================================================================

def test_shorthands_expand_to_valid_components_for_each_category() -> None:
    for category in Recorders.categories():
        if category == "modal":
            continue
        canonicals = set(Recorders.components_for(category))
        for shorthand, expansion in Recorders.shorthands_for(category).items():
            for component in expansion:
                assert component in canonicals, (
                    f"shorthand {shorthand!r} for category {category!r} "
                    f"expands to {component!r}, which is not a valid "
                    f"component for that category"
                )
