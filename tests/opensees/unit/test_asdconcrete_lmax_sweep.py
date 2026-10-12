"""Unit tests for the bind-time ASDConcrete l_max sweep (ADR 0044).

`sweep_asdconcrete_element_size` runs in the element-emit loop and emits a
single aggregated `ASDRegularizationWarning` when solid / 2-node-member
elements whose `spec.material` is an ASDConcrete material exceed the
crack-band snapback ceiling `l_max = 2 E Gf / ft^2`. Minimal cut: only a
direct `spec.material` is inspected (section-nested ASDConcrete is skipped).
"""
from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh.opensees._internal.build.validators.materials import (
    sweep_asdconcrete_element_size,
)
from apeGmsh.opensees.material.nd import (
    ASDConcrete3D,
    ASDRegularizationWarning,
    ElasticIsotropic,
)
from apeGmsh.opensees.material import _asdconcrete_laws as laws


_E, _FC, _FT, _GF = 30_000.0, 30.0, 3.0, 0.1
_LMAX = laws.l_max(_E, _GF, _FT)   # 2*E*Gf/ft^2 = 666.67 mm


def _cube_fem(edge: float):
    """FEM stub: 8 nodes of a cube of the given edge (min inter-node = edge)."""
    e = float(edge)
    coords = np.array([
        (0, 0, 0), (e, 0, 0), (e, e, 0), (0, e, 0),
        (0, 0, e), (e, 0, e), (e, e, e), (0, e, e),
    ], dtype=float)
    nodes = SimpleNamespace(index=lambda nid: nid - 1, coords=coords)
    return SimpleNamespace(nodes=nodes)


_CUBE_ELEMENTS = [(1, (1, 2, 3, 4, 5, 6, 7, 8))]


def _conc(**kw) -> ASDConcrete3D:
    return ASDConcrete3D.from_fc(E=_E, v=0.2, fc=_FC, ft=_FT, Gf=_GF,
                                 lch_ref=50.0, **kw)


class TestLmaxSweep:
    def test_warns_when_over_ceiling(self) -> None:
        spec = SimpleNamespace(material=_conc(), pg="block")
        fem = _cube_fem(edge=_LMAX * 1.5)   # edge > l_max
        with pytest.warns(ASDRegularizationWarning, match="snapback ceiling"):
            sweep_asdconcrete_element_size(spec, _CUBE_ELEMENTS, fem)

    def test_silent_within_ceiling(self) -> None:
        spec = SimpleNamespace(material=_conc(), pg="block")
        fem = _cube_fem(edge=_LMAX * 0.5)   # edge < l_max
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sweep_asdconcrete_element_size(spec, _CUBE_ELEMENTS, fem)

    def test_single_aggregated_warning_for_many_elements(self) -> None:
        spec = SimpleNamespace(material=_conc(), pg="block")
        fem = _cube_fem(edge=_LMAX * 2.0)
        elements = [(i, (1, 2, 3, 4, 5, 6, 7, 8)) for i in range(1, 6)]
        with pytest.warns(ASDRegularizationWarning) as rec:
            sweep_asdconcrete_element_size(spec, elements, fem)
        assert len(rec) == 1                       # one aggregated warning
        assert "5/5 elements" in str(rec[0].message)
        assert "PG 'block'" in str(rec[0].message)

    def test_skips_non_asdconcrete_material(self) -> None:
        spec = SimpleNamespace(material=ElasticIsotropic(E=_E, nu=0.2), pg="b")
        fem = _cube_fem(edge=_LMAX * 5.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sweep_asdconcrete_element_size(spec, _CUBE_ELEMENTS, fem)

    def test_skips_raw_curve_without_provenance(self) -> None:
        # Raw-constructed ASDConcrete3D has no ft/Gf -> l_max() is None -> skip.
        Te, Ts, Td = laws.make_tension(_E, _FT, _GF, 50.0)
        Ce, Cs, Cd = laws.make_compression(_E, _FC,
                                           laws.ceb_fip_Gc(_FC, _FT, _GF), 50.0)
        raw = ASDConcrete3D(E=_E, v=0.2, lch_ref=50.0,
                            Te=tuple(Te), Ts=tuple(Ts), Td=tuple(Td),
                            Ce=tuple(Ce), Cs=tuple(Cs), Cd=tuple(Cd))
        assert raw.l_max() is None
        spec = SimpleNamespace(material=raw, pg="b")
        fem = _cube_fem(edge=_LMAX * 5.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sweep_asdconcrete_element_size(spec, _CUBE_ELEMENTS, fem)

    def test_skips_spec_without_material(self) -> None:
        spec = SimpleNamespace(pg="b")   # e.g. a section-based element
        fem = _cube_fem(edge=_LMAX * 5.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sweep_asdconcrete_element_size(spec, _CUBE_ELEMENTS, fem)
