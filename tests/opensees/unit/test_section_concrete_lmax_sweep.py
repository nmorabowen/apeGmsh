"""The bind-time snap-back sweep reaches concrete nested in a shell section (ADR 0044).

A crack-band concrete inside ``spec.section`` (a ``LayeredShell`` layer, a
``PlateFiber``, a ``LadrunoShellModifier`` inner) latches the HOST element's
``getCharacteristicLength`` (fork 23d937bc4): the minimum node distance for
every shell, halved on ``ASDShellQ4`` while its EAS membrane is on. The
oracles are hand measures of a square shell, never the code's own output.
"""
from __future__ import annotations

import dataclasses
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh.opensees._internal.build.validators.materials import (
    sweep_asdconcrete_element_size,
)
from apeGmsh.opensees.element.shell import ASDShellQ4, ASDShellT3, ShellMITC4
from apeGmsh.opensees.material.nd import (
    ASDConcrete3D,
    ASDRegularizationWarning,
    LadrunoConcrete3D,
    PlateFiber,
)
from apeGmsh.opensees.section.plate import (
    LadrunoShellModifier,
    LayeredShell,
    ShellLayer,
)

# l_max = 2 E Gf / ft^2 = 2 * 30000 * 0.1 / 9 = 666.67 mm.
_E, _FC, _FT, _GF = 30_000.0, 30.0, 3.0, 0.1
_LMAX = 2.0 * _E * _GF / _FT**2
_QUAD = [(1, (1, 2, 3, 4))]
_TRI = [(1, (1, 2, 3))]


def _square_fem(a: float):
    """FEM stub: one flat a x a shell; min node distance a (the diagonal is longer)."""
    coords = np.array([(0, 0, 0), (a, 0, 0), (a, a, 0), (0, a, 0)], dtype=float)
    nodes = SimpleNamespace(index=lambda nid: nid - 1, coords=coords)
    return SimpleNamespace(nodes=nodes)


def _asd() -> ASDConcrete3D:
    return ASDConcrete3D.from_fc(E=_E, v=0.2, fc=_FC, ft=_FT, Gf=_GF, lch_ref=50.0)


def _layered(mat) -> LayeredShell:
    return LayeredShell(layers=tuple(
        ShellLayer(material=PlateFiber(material=mat), thickness=50.0) for _ in range(5)
    ))


def _silent(spec, elements, fem) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sweep_asdconcrete_element_size(spec, elements, fem)


def test_ceiling_is_the_hand_value() -> None:
    assert _asd().l_max() == pytest.approx(_LMAX)


class TestASDShellQ4:
    def test_eas_halves_the_length(self) -> None:
        # a = 1.2 l_max: the base rule (a) is over, but EAS halves it to 0.6 l_max.
        spec = ASDShellQ4(pg="slab", section=_layered(_asd()))
        _silent(spec, _QUAD, _square_fem(1.2 * _LMAX))

    def test_no_eas_keeps_the_full_length(self) -> None:
        spec = ASDShellQ4(pg="slab", section=_layered(_asd()), no_eas=True)
        with pytest.warns(
            ASDRegularizationWarning,
            match=r"section of ASDShellQ4 in PG 'slab': ASDConcrete: 1/1 elements .*ratio 1\.20",
        ):
            sweep_asdconcrete_element_size(spec, _QUAD, _square_fem(1.2 * _LMAX))

    def test_eas_warns_once_past_twice_the_ceiling(self) -> None:
        # a = 2.4 l_max -> a / 2 = 1.2 l_max: over even after the halving.
        spec = ASDShellQ4(pg="slab", section=_layered(_asd()))
        with pytest.warns(ASDRegularizationWarning, match=r"ratio 1\.20") as rec:
            sweep_asdconcrete_element_size(spec, _QUAD, _square_fem(2.4 * _LMAX))
        assert len(rec) == 1


class TestBaseRuleShells:
    def test_mitc4_uses_the_min_node_distance(self) -> None:
        spec = ShellMITC4(pg="slab", section=_layered(_asd()))
        with pytest.warns(ASDRegularizationWarning, match=r"ratio 1\.20"):
            sweep_asdconcrete_element_size(spec, _QUAD, _square_fem(1.2 * _LMAX))

    def test_mitc4_silent_within_the_ceiling(self) -> None:
        spec = ShellMITC4(pg="slab", section=_layered(_asd()))
        _silent(spec, _QUAD, _square_fem(0.9 * _LMAX))

    def test_asdshellt3_does_not_halve(self) -> None:
        # legs a: min node distance a (the hypotenuse is a sqrt 2).
        spec = ASDShellT3(pg="slab", section=_layered(_asd()))
        with pytest.warns(ASDRegularizationWarning, match=r"ratio 1\.20"):
            sweep_asdconcrete_element_size(spec, _TRI, _square_fem(1.2 * _LMAX))


class TestGraphAndLadruno:
    def test_modifier_inner_section_is_reached(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the nonlinear-inner advisory
            sec = LadrunoShellModifier(inner=_layered(_asd()))
        spec = ShellMITC4(pg="slab", section=sec)
        with pytest.warns(ASDRegularizationWarning, match="section of ShellMITC4"):
            sweep_asdconcrete_element_size(spec, _QUAD, _square_fem(1.2 * _LMAX))

    def test_ladruno_fixed_lch_is_checked_whatever_the_mesh(self) -> None:
        mat = LadrunoConcrete3D(E=_E, nu=0.2, fc=_FC, ft=_FT, Gf=_GF, lch=1.0e4)
        spec = ASDShellQ4(pg="slab", section=_layered(mat))
        with pytest.warns(ASDRegularizationWarning, match="LadrunoConcrete3D: the fixed lch=10000"):
            sweep_asdconcrete_element_size(spec, _QUAD, _square_fem(1.0))

    def test_ladruno_auto_regularize_uses_the_host_length(self) -> None:
        mat = LadrunoConcrete3D(E=_E, nu=0.2, fc=_FC, ft=_FT, Gf=_GF, auto_regularize=True)
        ceiling = min(mat.l_max())
        spec = ASDShellQ4(pg="slab", section=_layered(mat))
        _silent(spec, _QUAD, _square_fem(1.5 * ceiling))  # EAS: 0.75 ceiling
        with pytest.warns(ASDRegularizationWarning, match=r"LadrunoConcrete3D: 1/1 .*ratio 1\.50"):
            sweep_asdconcrete_element_size(spec, _QUAD, _square_fem(3.0 * ceiling))

    def test_raw_asd_without_provenance_is_skipped(self) -> None:
        raw = dataclasses.replace(_asd(), Gf=None)
        assert raw.l_max() is None
        _silent(ShellMITC4(pg="slab", section=_layered(raw)), _QUAD, _square_fem(1.0e6))
