"""B5 regression: typo'd load/mass targets fail before meshing runs."""
from __future__ import annotations

from unittest.mock import patch

import pytest


def _make_box(g, label: str = "block"):
    with g.parts.part(label):
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)


def test_typo_load_target_fails_before_mesh(g):
    _make_box(g, "block")
    g.loads.surface.pressure("blokc", magnitude=1.0)  # typo — should not match

    with patch("gmsh.model.mesh.generate") as mocked:
        with pytest.raises(KeyError, match="blokc"):
            g.mesh.generation.generate(3)
        mocked.assert_not_called()


def test_typo_mass_target_fails_before_mesh(g):
    _make_box(g, "block")
    g.masses.point("blokc", mass=10.0)  # typo

    with patch("gmsh.model.mesh.generate") as mocked:
        with pytest.raises(KeyError, match="blokc"):
            g.mesh.generation.generate(3)
        mocked.assert_not_called()


def test_valid_load_target_passes_validation(g):
    _make_box(g, "block")
    g.loads.surface.pressure("block", magnitude=1.0)

    g.mesh.generation.generate(3)
    assert g.loads.load_defs[0].target == "block"


def test_raw_dimtag_target_skips_validation(g):
    _make_box(g, "block")
    g.loads.surface.pressure([(2, 1)], magnitude=1.0)

    g.mesh.generation.generate(3)
