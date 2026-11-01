"""The bind-time snap-back sweep covers the Ladruno concretes (ADR 0044).

`sweep_asdconcrete_element_size` warns once per PG when the length the fork
regularizes a crack-band concrete with exceeds that material's snap-back
ceiling. For `LadrunoConcrete3D` that length is each element's own under
`auto_regularize`, else the fixed `lch`; an `_LadrunoRC` checks only under
`auto_regularization`. The element's length mirrors the fork's per-class
`getCharacteristicLength` override (fork 23d937bc4), so the oracles below are
hand measures of the element, never the code's own output.
"""
from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh.opensees._internal.build.validators.materials import (
    _element_lch,
    sweep_asdconcrete_element_size,
)
from apeGmsh.opensees.material.nd import (
    ASDRegularizationWarning,
    LadrunoConcrete3D,
    LadrunoRCConcrete,
)

# l_max_t = (20/21) E Gf / ft^2 = (20/21) * 30000 * 0.1 / 9 = 317.46 mm;
# l_max_c = 2 E Gc / (fc^2 (1 + Hp)) = 2 * 30000 * 25 / (900 * 1.5) = 1111 mm.
_E, _FC, _FT, _GF = 30_000.0, 30.0, 3.0, 0.1
_LMAX_T = (20.0 / 21.0) * _E * _GF / _FT**2


def _box_fem(a: float, b: float, c: float):
    """FEM stub: one brick of edges (a, b, c) in OpenSees node order."""
    coords = np.array([
        (0, 0, 0), (a, 0, 0), (a, b, 0), (0, b, 0),
        (0, 0, c), (a, 0, c), (a, b, c), (0, b, c),
    ], dtype=float)
    nodes = SimpleNamespace(index=lambda nid: nid - 1, coords=coords)
    return SimpleNamespace(nodes=nodes)


_BRICK = [(1, (1, 2, 3, 4, 5, 6, 7, 8))]


def _spec(cls_name: str, material: object, pg: str = "wall") -> object:
    """A spec whose class name is the element the fork would build."""
    return type(cls_name, (), {"material": material, "pg": pg})()


def _ladruno(**kw) -> LadrunoConcrete3D:
    return LadrunoConcrete3D(E=_E, nu=0.2, fc=_FC, ft=_FT, Gf=_GF, **kw)


def test_ceiling_is_the_hand_value() -> None:
    l_t, l_c = _ladruno().l_max()
    assert l_t == pytest.approx(_LMAX_T)
    assert l_t < l_c


class TestElementLength:
    """Each length is the fork override's formula, worked by hand."""

    def test_ladruno_brick_is_cbrt_volume(self) -> None:
        # 2 x 1 x 1 box: V = 2, so cbrt(V) = 1.2599; min node distance is 1.
        x = _box_fem(2.0, 1.0, 1.0).nodes.coords
        assert _element_lch("LadrunoBrick", x) == pytest.approx(2.0 ** (1 / 3))
        assert _element_lch("stdBrick", x) == pytest.approx(1.0)

    def test_bezier_tet10_is_cbrt_six_volume(self) -> None:
        # unit right tet: V = 1/6, so cbrt(6V) = 1 (corners only are read).
        x = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], dtype=float)
        assert _element_lch("BezierTet10", x) == pytest.approx(1.0)

    def test_ladruno_quad_is_sqrt_area(self) -> None:
        x = np.array([(0, 0, 0), (4, 0, 0), (4, 1, 0), (0, 1, 0)], dtype=float)
        assert _element_lch("LadrunoQuad", x) == pytest.approx(2.0)

    def test_triangles_are_sqrt_two_area(self) -> None:
        # legs 2 and 1: A = 1, so sqrt(2A) = sqrt(2)
        x = np.array([(0, 0, 0), (2, 0, 0), (0, 1, 0)], dtype=float)
        for cls in ("LadrunoCST", "LadrunoLST", "BezierTri6"):
            assert _element_lch(cls, x) == pytest.approx(np.sqrt(2.0))

    def test_degenerate_measure_falls_back_to_base(self) -> None:
        # collinear triangle: A = 0 -> min node distance, as the fork does
        x = np.array([(0, 0, 0), (1, 0, 0), (3, 0, 0)], dtype=float)
        assert _element_lch("LadrunoCST", x) == pytest.approx(1.0)


class TestLadrunoConcrete3DSweep:
    def test_auto_regularize_warns_on_the_cbrt_volume_length(self) -> None:
        # min edge 0.9 l_max (the base rule would pass it), but
        # cbrt(V) = cbrt(2 * 0.9^3) l_max = 1.134 l_max: over the ceiling.
        h = 0.9 * _LMAX_T
        spec = _spec("LadrunoBrick", _ladruno(auto_regularize=True))
        with pytest.warns(ASDRegularizationWarning, match="LadrunoConcrete3D: 1/1 elements in PG 'wall'"):
            sweep_asdconcrete_element_size(spec, _BRICK, _box_fem(2 * h, h, h))

    def test_auto_regularize_silent_within_the_ceiling(self) -> None:
        h = 0.5 * _LMAX_T
        spec = _spec("LadrunoBrick", _ladruno(auto_regularize=True))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sweep_asdconcrete_element_size(spec, _BRICK, _box_fem(h, h, h))

    def test_fixed_lch_is_checked_whatever_the_mesh(self) -> None:
        # no auto_regularize: every element uses lch, so a tiny mesh still warns
        spec = _spec("LadrunoBrick", _ladruno(lch=2 * _LMAX_T))
        with pytest.warns(ASDRegularizationWarning, match="fixed lch="):
            sweep_asdconcrete_element_size(spec, _BRICK, _box_fem(1.0, 1.0, 1.0))

    def test_fixed_lch_within_the_ceiling_is_silent_on_a_coarse_mesh(self) -> None:
        spec = _spec("LadrunoBrick", _ladruno(lch=50.0))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            big = 10 * _LMAX_T
            sweep_asdconcrete_element_size(spec, _BRICK, _box_fem(big, big, big))


class TestLadrunoRCSweep:
    def _rc(self, **kw) -> LadrunoRCConcrete:
        return LadrunoRCConcrete.from_fc(E=_E, nu=0.2, fc=_FC, lch_ref=50.0, **kw)

    def test_warns_over_the_table_ceiling(self) -> None:
        rc = self._rc()
        ceiling = min(rc.l_max())
        h = 2.0 * ceiling
        with pytest.warns(ASDRegularizationWarning, match="LadrunoRCConcrete: 1/1 elements"):
            sweep_asdconcrete_element_size(_spec("LadrunoBrick", rc), _BRICK, _box_fem(h, h, h))

    def test_without_auto_regularization_nothing_depends_on_the_mesh(self) -> None:
        rc = self._rc(regularize=False)
        assert rc.auto_regularization is None
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            h = 1.0e6
            sweep_asdconcrete_element_size(_spec("LadrunoBrick", rc), _BRICK, _box_fem(h, h, h))
