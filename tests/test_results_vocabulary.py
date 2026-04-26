"""Phase 0 — canonical vocabulary checks."""
from __future__ import annotations

import pytest

from apeGmsh.results._vocabulary import (
    ALL_CANONICAL,
    ALL_SHORTHANDS,
    DERIVED_SCALARS,
    FIBER,
    LINE_DIAGRAMS,
    MATERIAL_STATE,
    NODAL_FORCES,
    NODAL_KINEMATICS,
    PER_ELEMENT_NODAL_FORCES,
    STRAIN,
    STRESS,
    is_canonical,
    is_shorthand,
)


def test_no_overlap_between_canonical_and_shorthand() -> None:
    """A name is exactly one of: canonical, shorthand, or unknown."""
    assert ALL_CANONICAL.isdisjoint(ALL_SHORTHANDS)


def test_known_canonical_names() -> None:
    for name in [
        "displacement_x", "displacement_y", "displacement_z",
        "rotation_x", "velocity_z", "acceleration_y",
        "force_x", "moment_z",
        "reaction_force_x", "reaction_moment_z",
        "pore_pressure",
        "axial_force", "shear_y", "torsion", "bending_moment_y",
        "stress_xx", "stress_xy", "strain_zz", "strain_xz",
        "von_mises_stress", "principal_stress_2",
        "fiber_stress", "fiber_strain",
        "damage",
    ]:
        assert is_canonical(name), f"{name!r} should be canonical"


def test_state_variable_pattern_is_canonical() -> None:
    assert is_canonical("state_variable_0")
    assert is_canonical("state_variable_42")
    # Non-integer suffix is rejected.
    assert not is_canonical("state_variable_x")
    assert not is_canonical("state_variable_")


def test_unknown_names_are_not_canonical() -> None:
    assert not is_canonical("displacement")  # shorthand, not canonical
    assert not is_canonical("disp")
    assert not is_canonical("DISPLACEMENT")  # MPCO style is not canonical
    assert not is_canonical("displacement_w")
    assert not is_canonical("")


def test_known_shorthands() -> None:
    for name in [
        "displacement", "rotation", "velocity", "acceleration",
        "angular_velocity", "angular_acceleration",
        "force", "moment", "reaction",
        "stress", "strain",
    ]:
        assert is_shorthand(name), f"{name!r} should be shorthand"


def test_categories_are_disjoint() -> None:
    """Each canonical name belongs to exactly one category."""
    cats = [
        ("kinematics", NODAL_KINEMATICS),
        ("nodal_forces", NODAL_FORCES),
        ("per_element_forces", PER_ELEMENT_NODAL_FORCES),
        ("line_diagrams", LINE_DIAGRAMS),
        ("stress", STRESS),
        ("strain", STRAIN),
        ("derived", DERIVED_SCALARS),
        ("fiber", FIBER),
        ("material_state", MATERIAL_STATE),
    ]
    seen: dict[str, str] = {}
    for cat_name, items in cats:
        for n in items:
            if n in seen:
                pytest.fail(
                    f"{n!r} appears in both {seen[n]!r} and {cat_name!r}"
                )
            seen[n] = cat_name


def test_stress_strain_have_six_components() -> None:
    assert len(STRESS) == 6
    assert len(STRAIN) == 6
    # Every component starts with the right prefix.
    assert all(s.startswith("stress_") for s in STRESS)
    assert all(s.startswith("strain_") for s in STRAIN)


def test_recorder_token_coverage() -> None:
    """Every OpenSees recorder token has a canonical mapping target.

    This is a smoke-level check. The actual token translation tables
    live in the readers (Phase 3 for MPCO; Phase 6 for txt/xml). Here
    we just confirm that the apeGmsh-side names exist for the common
    tokens called out by the ``opensees-expert`` skill.
    """
    expected = [
        # Node recorder tokens
        "displacement_x", "displacement_y", "displacement_z",
        "rotation_x", "rotation_y", "rotation_z",
        "velocity_x", "acceleration_x",
        "displacement_increment_x",
        "reaction_force_x", "reaction_moment_x",
        "pore_pressure",
        # Element recorder tokens (continuum)
        "stress_xx", "strain_xx",
        # Element recorder tokens (beam)
        "axial_force", "moment_y", "torsion",
        # Element recorder tokens (per-element-node)
        "nodal_resisting_force_x", "nodal_resisting_force_local_x",
        # Fiber
        "fiber_stress", "fiber_strain",
    ]
    for name in expected:
        assert is_canonical(name), f"missing canonical name: {name!r}"
