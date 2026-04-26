"""Phase 3 — STKO ↔ canonical name translation."""
from __future__ import annotations

import pytest

from apeGmsh.results.readers._mpco_translation import (
    all_known_mpco_result_names,
    canonical_node_component,
    canonical_to_mpco_lookup,
    component_axis,
    has_canonical_mapping,
)


# =====================================================================
# Component axis extraction
# =====================================================================

def test_component_axis_latin() -> None:
    assert component_axis("Ux") == "x"
    assert component_axis("Uy") == "y"
    assert component_axis("Uz") == "z"
    assert component_axis("Rx") == "x"
    assert component_axis("Vy") == "y"
    assert component_axis("Az") == "z"
    assert component_axis("RFx") == "x"
    assert component_axis("RMy") == "y"
    assert component_axis("RVz") == "z"
    assert component_axis("RAx") == "x"


def test_component_axis_case_insensitive() -> None:
    assert component_axis("UX") == "x"
    assert component_axis("uX") == "x"


def test_component_axis_returns_none_for_scalar() -> None:
    # Pressure component label is just "p"
    assert component_axis("p") is None
    assert component_axis("P") is None
    assert component_axis("") is None


def test_component_axis_unicode() -> None:
    """Greek-prefixed variants from the docs (ω, α) still end in x/y/z."""
    assert component_axis("ωx") == "x"
    assert component_axis("αy") == "y"


# =====================================================================
# Full translation
# =====================================================================

@pytest.mark.parametrize("mpco_name,comp,expected", [
    ("DISPLACEMENT", "Ux", "displacement_x"),
    ("DISPLACEMENT", "Uy", "displacement_y"),
    ("DISPLACEMENT", "Uz", "displacement_z"),
    ("ROTATION", "Rx", "rotation_x"),
    ("ROTATION", "Ry", "rotation_y"),
    ("ROTATION", "Rz", "rotation_z"),
    ("VELOCITY", "Vx", "velocity_x"),
    ("ANGULAR_VELOCITY", "RVx", "angular_velocity_x"),
    ("ACCELERATION", "Ax", "acceleration_x"),
    ("ANGULAR_ACCELERATION", "RAx", "angular_acceleration_x"),
    ("REACTION_FORCE", "RFx", "reaction_force_x"),
    ("REACTION_MOMENT", "RMz", "reaction_moment_z"),
    ("REACTION_FORCE_INCLUDING_INERTIA", "RFy", "reaction_force_y"),
    ("RAYLEIGH_FORCE", "RFx", "reaction_force_x"),
    ("RAYLEIGH_MOMENT", "RMz", "reaction_moment_z"),
    ("UNBALANCED_FORCE", "Fx", "force_x"),
    ("UNBALANCED_MOMENT", "Mz", "moment_z"),
    ("PRESSURE", "p", "pore_pressure"),
])
def test_canonical_node_component(
    mpco_name: str, comp: str, expected: str,
) -> None:
    assert canonical_node_component(mpco_name, comp) == expected


def test_unknown_mpco_name_returns_none() -> None:
    assert canonical_node_component("NOT_A_RESULT", "Ux") is None


def test_unknown_component_returns_none() -> None:
    # 'Q' isn't a known axis suffix
    assert canonical_node_component("DISPLACEMENT", "Q") is None


# =====================================================================
# Reverse lookup
# =====================================================================

def test_canonical_to_mpco_lookup() -> None:
    assert canonical_to_mpco_lookup("displacement_x") == ("DISPLACEMENT", "x")
    assert canonical_to_mpco_lookup("rotation_z") == ("ROTATION", "z")
    assert canonical_to_mpco_lookup("velocity_y") == ("VELOCITY", "y")


def test_canonical_to_mpco_pressure() -> None:
    assert canonical_to_mpco_lookup("pore_pressure") == ("PRESSURE", "")


def test_canonical_to_mpco_unknown() -> None:
    assert canonical_to_mpco_lookup("not_a_name") is None


# =====================================================================
# Coverage
# =====================================================================

def test_known_mpco_names_includes_all_real_world_groups() -> None:
    """Sanity: every result-group name we saw in real MPCO files is mapped."""
    real_world = [
        "DISPLACEMENT", "ROTATION", "VELOCITY", "ANGULAR_VELOCITY",
        "ACCELERATION", "ANGULAR_ACCELERATION",
        "REACTION_FORCE", "REACTION_MOMENT",
        "REACTION_FORCE_INCLUDING_INERTIA",
        "REACTION_MOMENT_INCLUDING_INERTIA",
        "RAYLEIGH_FORCE", "RAYLEIGH_MOMENT",
        "UNBALANCED_FORCE", "UNBALANCED_MOMENT",
        "UNBALANCED_FORCE_INCLUDING_INERTIA",
        "UNBALANCED_MOMENT_INCLUDING_INERTIA",
        "PRESSURE",
    ]
    known = set(all_known_mpco_result_names())
    missing = [n for n in real_world if n not in known]
    assert not missing, f"Unmapped MPCO result names: {missing}"


def test_has_canonical_mapping() -> None:
    assert has_canonical_mapping("DISPLACEMENT")
    assert not has_canonical_mapping("MODES_OF_VIBRATION(U)")  # special case
