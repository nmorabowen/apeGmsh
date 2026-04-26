"""Phase 4 — element capability annotations on ``_ElemSpec``.

Spot-checks that each entry in ``_ELEM_REGISTRY`` carries sensible
``has_*`` flags. These flags drive resolve-time / emission-time
validation (e.g. requesting GP results on a truss should fail).
"""
from __future__ import annotations

import pytest

from apeGmsh.solvers._element_specs import _ELEM_REGISTRY


# =====================================================================
# Continuum solids → has_gauss
# =====================================================================

@pytest.mark.parametrize("name", [
    "FourNodeTetrahedron", "TenNodeTetrahedron",
    "stdBrick", "bbarBrick", "SSPbrick",
    "quad", "tri31", "SSPquad",
])
def test_continuum_has_gauss(name: str) -> None:
    spec = _ELEM_REGISTRY[name]
    assert spec.has_gauss
    assert not spec.has_fibers
    assert not spec.has_layers
    assert not spec.has_line_stations


# =====================================================================
# Shells → has_gauss + has_layers
# =====================================================================

@pytest.mark.parametrize("name", [
    "ShellMITC3", "ShellMITC4", "ShellDKGQ", "ASDShellQ4",
])
def test_shells_have_gauss_and_layers(name: str) -> None:
    spec = _ELEM_REGISTRY[name]
    assert spec.has_gauss
    assert spec.has_layers
    assert not spec.has_line_stations


# =====================================================================
# Beams → has_line_stations
# =====================================================================

@pytest.mark.parametrize("name", [
    "elasticBeamColumn", "ElasticTimoshenkoBeam",
])
def test_beams_have_line_stations(name: str) -> None:
    spec = _ELEM_REGISTRY[name]
    assert spec.has_line_stations
    # Elastic beams don't have GPs (they're integrated analytically).
    assert not spec.has_gauss


# =====================================================================
# Trusses — none of the element-level capabilities
# =====================================================================

@pytest.mark.parametrize("name", ["truss", "corotTruss"])
def test_trusses_have_no_capabilities(name: str) -> None:
    spec = _ELEM_REGISTRY[name]
    assert not spec.has_gauss
    assert not spec.has_fibers
    assert not spec.has_layers
    assert not spec.has_line_stations


# =====================================================================
# supports() helper
# =====================================================================

def test_supports_returns_correct_flag() -> None:
    tet = _ELEM_REGISTRY["FourNodeTetrahedron"]
    assert tet.supports("gauss")
    assert not tet.supports("fibers")
    assert not tet.supports("layers")
    assert not tet.supports("line_stations")
    # nodes / elements always supported
    assert tet.supports("nodes")
    assert tet.supports("elements")


def test_supports_unknown_category() -> None:
    tet = _ELEM_REGISTRY["FourNodeTetrahedron"]
    assert not tet.supports("not_a_category")


# =====================================================================
# Coverage: every entry carries at least one consistent capability
# =====================================================================

def test_all_registry_entries_have_capability_fields() -> None:
    """Defensive check that no future entry forgets the capability fields."""
    for name, spec in _ELEM_REGISTRY.items():
        for attr in ("has_gauss", "has_fibers", "has_layers",
                      "has_line_stations"):
            assert hasattr(spec, attr), (
                f"Element {name!r} is missing capability flag {attr!r}."
            )
            assert isinstance(getattr(spec, attr), bool), (
                f"{name}.{attr} must be bool."
            )
