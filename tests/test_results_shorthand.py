"""Phase 0 — shorthand expansion with ndm/ndf clipping."""
from __future__ import annotations

import pytest

from apeGmsh.results._vocabulary import expand_many, expand_shorthand


# =====================================================================
# Pass-through of canonical names
# =====================================================================

def test_canonical_pass_through() -> None:
    assert expand_shorthand("displacement_x") == ("displacement_x",)
    assert expand_shorthand("stress_xy") == ("stress_xy",)
    assert expand_shorthand("damage") == ("damage",)


def test_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown component"):
        expand_shorthand("not_a_real_name")
    with pytest.raises(ValueError):
        expand_shorthand("DISPLACEMENT")  # caps style is not canonical


# =====================================================================
# Translational shorthands clip to ndm
# =====================================================================

def test_displacement_3d() -> None:
    assert expand_shorthand("displacement", ndm=3, ndf=6) == (
        "displacement_x", "displacement_y", "displacement_z",
    )


def test_displacement_2d() -> None:
    # 2D, regardless of ndf, gives only x and y.
    assert expand_shorthand("displacement", ndm=2, ndf=3) == (
        "displacement_x", "displacement_y",
    )
    assert expand_shorthand("displacement", ndm=2, ndf=2) == (
        "displacement_x", "displacement_y",
    )


def test_displacement_1d() -> None:
    assert expand_shorthand("displacement", ndm=1, ndf=1) == (
        "displacement_x",
    )


def test_rotation_excluded_from_displacement() -> None:
    """`displacement` is translations only, never rotations."""
    out = expand_shorthand("displacement", ndm=3, ndf=6)
    assert "rotation_x" not in out
    assert "rotation_y" not in out
    assert "rotation_z" not in out


# =====================================================================
# Rotational shorthands need rotational DOFs
# =====================================================================

def test_rotation_3d_full_dof() -> None:
    assert expand_shorthand("rotation", ndm=3, ndf=6) == (
        "rotation_x", "rotation_y", "rotation_z",
    )


def test_rotation_3d_translational_only_ndf_returns_empty() -> None:
    # ndf=3 in 3D = no rotational DOFs.
    assert expand_shorthand("rotation", ndm=3, ndf=3) == ()


def test_rotation_2d_returns_z_only() -> None:
    # 2D rotation is about the out-of-plane (z) axis.
    assert expand_shorthand("rotation", ndm=2, ndf=3) == ("rotation_z",)


def test_rotation_2d_no_rotational_dof() -> None:
    assert expand_shorthand("rotation", ndm=2, ndf=2) == ()


def test_rotation_1d_returns_empty() -> None:
    assert expand_shorthand("rotation", ndm=1, ndf=1) == ()


def test_angular_velocity_clipping() -> None:
    assert expand_shorthand("angular_velocity", ndm=3, ndf=6) == (
        "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
    )
    assert expand_shorthand("angular_velocity", ndm=2, ndf=3) == (
        "angular_velocity_z",
    )
    assert expand_shorthand("angular_velocity", ndm=2, ndf=2) == ()


# =====================================================================
# Reaction — combines translational forces + rotational moments
# =====================================================================

def test_reaction_3d_full_dof() -> None:
    assert expand_shorthand("reaction", ndm=3, ndf=6) == (
        "reaction_force_x", "reaction_force_y", "reaction_force_z",
        "reaction_moment_x", "reaction_moment_y", "reaction_moment_z",
    )


def test_reaction_3d_no_rotational_dof() -> None:
    # ndf=3 → forces only.
    assert expand_shorthand("reaction", ndm=3, ndf=3) == (
        "reaction_force_x", "reaction_force_y", "reaction_force_z",
    )


def test_reaction_2d_with_moment() -> None:
    assert expand_shorthand("reaction", ndm=2, ndf=3) == (
        "reaction_force_x", "reaction_force_y",
        "reaction_moment_z",
    )


def test_reaction_2d_no_moment() -> None:
    assert expand_shorthand("reaction", ndm=2, ndf=2) == (
        "reaction_force_x", "reaction_force_y",
    )


# =====================================================================
# Tensor shorthands (stress, strain) — 6 in 3D, 3 in 2D, 1 in 1D
# =====================================================================

def test_stress_3d() -> None:
    assert expand_shorthand("stress", ndm=3, ndf=6) == (
        "stress_xx", "stress_yy", "stress_zz",
        "stress_xy", "stress_yz", "stress_xz",
    )


def test_stress_2d_plane() -> None:
    # Plane: xx, yy, xy.
    assert expand_shorthand("stress", ndm=2, ndf=3) == (
        "stress_xx", "stress_yy", "stress_xy",
    )


def test_stress_1d_axial_only() -> None:
    assert expand_shorthand("stress", ndm=1, ndf=1) == ("stress_xx",)


def test_strain_clipping_matches_stress() -> None:
    # Strain follows the same clipping rule as stress.
    assert expand_shorthand("strain", ndm=2, ndf=3) == (
        "strain_xx", "strain_yy", "strain_xy",
    )


# =====================================================================
# expand_many — dedup + order preservation
# =====================================================================

def test_expand_many_deduplicates() -> None:
    out = expand_many(
        ["displacement", "displacement_x", "displacement"],
        ndm=3, ndf=6,
    )
    assert out == ("displacement_x", "displacement_y", "displacement_z")


def test_expand_many_preserves_order() -> None:
    out = expand_many(
        ["rotation", "displacement"], ndm=3, ndf=6,
    )
    assert out == (
        "rotation_x", "rotation_y", "rotation_z",
        "displacement_x", "displacement_y", "displacement_z",
    )


def test_expand_many_mixed_kinematics() -> None:
    """Translations + rotations together — one entry each."""
    out = expand_many(["displacement", "rotation"], ndm=3, ndf=6)
    assert out == (
        "displacement_x", "displacement_y", "displacement_z",
        "rotation_x", "rotation_y", "rotation_z",
    )


# =====================================================================
# Bad ndm
# =====================================================================

def test_invalid_ndm() -> None:
    with pytest.raises(ValueError):
        expand_shorthand("displacement", ndm=4, ndf=6)
    with pytest.raises(ValueError):
        expand_shorthand("displacement", ndm=0, ndf=6)
