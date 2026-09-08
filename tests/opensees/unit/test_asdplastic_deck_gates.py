"""ADR 0105 D4 — the swallowing-host gate for ``ASDPlasticMaterial3D``.

The fork's fail-loud contract (ADR-94) only reaches the analysis through
an element that ACTS on the material's return code.  ``stdBrick`` was
measured to discard it (20/20 "successes" on a deck ``LadrunoBrick``
refuses 0/20 — fork B2), so a ``strict_convergence`` material on it is
silently inert.  The gate warns once per deck and is keyed on the
registry's measured ``propagates_material_refusal`` flag, never on an
element name.
"""
from __future__ import annotations

import warnings

import pytest

from apeGmsh.opensees._element_capabilities import (
    _ELEM_REGISTRY,
    element_propagates_material_refusal,
)
from apeGmsh.opensees._internal.build import (
    ASDPlasticHostWarning,
    validate_asdplastic_host,
)
from apeGmsh.opensees.element.solid import (
    LadrunoBrick,
    TenNodeTetrahedron,
    stdBrick,
)
from apeGmsh.opensees.material.nd import (
    ElasticIsotropic,
    MohrCoulombSoil,
    PlaneStrain,
)


def _mc():
    return MohrCoulombSoil(c=100.0, phi=30.0, psi=0.0, E=1e6, nu=0.3)


def _silent(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ASDPlasticHostWarning)
        fn(*a, **kw)


class TestCapabilityFlag:
    def test_measured_hosts_are_marked(self) -> None:
        assert element_propagates_material_refusal("LadrunoBrick") is True
        assert element_propagates_material_refusal("TenNodeTetrahedron") is True
        assert element_propagates_material_refusal("stdBrick") is False

    def test_unmeasured_is_none_not_false(self) -> None:
        """An unmeasured element must not be reported as swallowing."""
        assert element_propagates_material_refusal("NoSuchElement") is None
        assert element_propagates_material_refusal("FourNodeQuad") is None

    def test_every_registry_entry_carries_a_tri_state_flag(self) -> None:
        for name, spec in _ELEM_REGISTRY.items():
            assert spec.propagates_material_refusal in (True, False, None), name


class TestHostGate:
    def test_std_brick_warns(self) -> None:
        with pytest.warns(ASDPlasticHostWarning, match="fork ADR-94 B2"):
            validate_asdplastic_host([stdBrick(pg="rock", material=_mc())])

    def test_the_warning_names_the_host_the_group_and_the_fix(self) -> None:
        with pytest.warns(ASDPlasticHostWarning) as rec:
            validate_asdplastic_host([stdBrick(pg="rock", material=_mc())])
        msg = str(rec[0].message)
        assert "stdBrick (pg rock)" in msg
        assert "LadrunoBrick or TenNodeTetrahedron" in msg
        assert "strict_convergence" in msg

    def test_ladruno_brick_is_silent(self) -> None:
        _silent(validate_asdplastic_host, [LadrunoBrick(pg="rock", material=_mc())])

    def test_ten_node_tetrahedron_is_silent(self) -> None:
        _silent(
            validate_asdplastic_host,
            [TenNodeTetrahedron(pg="rock", material=_mc())],
        )

    def test_a_non_asdp_material_on_std_brick_is_silent(self) -> None:
        _silent(
            validate_asdplastic_host,
            [stdBrick(pg="rock", material=ElasticIsotropic(E=1e6, nu=0.3))],
        )

    def test_reaches_through_a_wrapper(self) -> None:
        """The material is found through ``PlaneStrain(base=...)`` — the
        wrapper graph, not just the element's direct ``material``."""
        with pytest.warns(ASDPlasticHostWarning):
            validate_asdplastic_host(
                [stdBrick(pg="rock", material=PlaneStrain(base=_mc()))]
            )

    def test_warns_once_per_deck(self) -> None:
        with pytest.warns(ASDPlasticHostWarning) as rec:
            validate_asdplastic_host([
                stdBrick(pg="rock", material=_mc()),
                stdBrick(pg="soil", material=_mc()),
                LadrunoBrick(pg="core", material=_mc()),
            ])
        assert len(rec) == 1
        msg = str(rec[0].message)
        assert "stdBrick (pg rock, soil)" in msg
        assert "LadrunoBrick (pg" not in msg

    def test_empty_deck_is_silent(self) -> None:
        _silent(validate_asdplastic_host, [])
