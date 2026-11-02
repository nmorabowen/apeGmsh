"""The bind-time snap-back sweep reaches concrete nested in a shell section (ADR 0044).

A crack-band concrete inside ``spec.section`` (a ``LayeredShell`` layer, a
``PlateFiber``, a ``LadrunoShellModifier`` inner) reads the active element's
``getCharacteristicLength`` (fork 23d937bc4). On ``ASDShellQ4`` and
``ASDShellT3``, whose ``update()`` drives the sections, that is the host: the
minimum node distance, halved on ``ASDShellQ4`` while its EAS membrane is on.
On ``ShellMITC4``, ``ShellDKGQ`` and ``ShellNLDKGQ`` it is another element
(#2680), so there is no length to compare and the sweep says so. The oracles
are hand measures of the shell, never the code's own output.
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
from apeGmsh.opensees.element.shell import (
    ASDShellQ4,
    ASDShellT3,
    ShellDKGQ,
    ShellMITC4,
    ShellNLDKGQ,
)
from apeGmsh.opensees.material.nd import (
    ASDConcrete3D,
    ASDRegularizationWarning,
    ElasticIsotropic,
    LadrunoConcrete3D,
    LadrunoRCConcrete,
    PlateFiber,
)
from apeGmsh.opensees.material.uniaxial import ASDConcrete1D
from apeGmsh.opensees.section.plate import (
    LadrunoShellModifier,
    LayeredShell,
    ShellLayer,
)

# l_max = 2 E Gf / ft^2 = 2 * 30000 * 0.1 / 9 = 666.67 mm.
_E, _FC, _FT, _GF = 30_000.0, 30.0, 3.0, 0.1
_LMAX = 2.0 * _E * _GF / _FT**2
# LadrunoConcrete3D: l_max_t = (20/21) E Gf / ft^2 = 317.46 mm (below l_max_c).
_LMAX_LADRUNO = (20.0 / 21.0) * _E * _GF / _FT**2
_QUAD = [(1, (1, 2, 3, 4))]
_TRI = [(1, (1, 2, 3))]
_LATCHING = [ShellMITC4, ShellDKGQ, ShellNLDKGQ]


def _rect_fem(a: float, b: float):
    """FEM stub: one flat shell, first edge a, second edge b; min node distance min(a, b)."""
    coords = np.array([(0, 0, 0), (a, 0, 0), (a, b, 0), (0, b, 0)], dtype=float)
    nodes = SimpleNamespace(index=lambda nid: nid - 1, coords=coords)
    return SimpleNamespace(nodes=nodes)


def _square_fem(a: float):
    return _rect_fem(a, a)


def _strip_fem(sizes: list[tuple[float, float]]):
    """``(fem, elements)``: disjoint a x b quads, one per ``sizes`` entry, in order."""
    coords: list[tuple[float, float, float]] = []
    elements = []
    for i, (a, b) in enumerate(sizes):
        y = 1.0e5 * i
        first = len(coords) + 1
        coords += [(0, y, 0), (a, y, 0), (a, y + b, 0), (0, y + b, 0)]
        elements.append((i + 1, (first, first + 1, first + 2, first + 3)))
    nodes = SimpleNamespace(index=lambda nid: nid - 1, coords=np.array(coords, dtype=float))
    return SimpleNamespace(nodes=nodes), elements


def _asd(**kw) -> ASDConcrete3D:
    return ASDConcrete3D.from_fc(E=_E, v=0.2, fc=_FC, ft=_FT, Gf=_GF, lch_ref=50.0, **kw)


def _ladruno(**kw) -> LadrunoConcrete3D:
    return LadrunoConcrete3D(E=_E, nu=0.2, fc=_FC, ft=_FT, Gf=_GF, **kw)


def _rc(**kw) -> LadrunoRCConcrete:
    return LadrunoRCConcrete.from_fc(E=_E, nu=0.2, fc=_FC, lch_ref=50.0, **kw)


def _layered(*mats) -> LayeredShell:
    """A section of ``mats``, padded to the 5 layers with the last one (OpenSees needs 3)."""
    mats = mats + (mats[-1],) * (5 - len(mats))
    return LayeredShell(layers=tuple(
        ShellLayer(material=PlateFiber(material=m), thickness=50.0) for m in mats
    ))


def _silent(spec, elements, fem) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sweep_asdconcrete_element_size(spec, elements, fem)


def test_ceiling_is_the_hand_value() -> None:
    assert _asd().l_max() == pytest.approx(_LMAX)
    assert min(_ladruno().l_max()) == pytest.approx(_LMAX_LADRUNO)


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

    def test_non_square_reads_the_minimum_edge(self) -> None:
        # 3 x 1.2 l_max, no EAS: the minimum edge is the second one, ratio 1.20.
        spec = ASDShellQ4(pg="slab", section=_layered(_asd()), no_eas=True)
        with pytest.warns(ASDRegularizationWarning, match=r"ratio 1\.20"):
            sweep_asdconcrete_element_size(spec, _QUAD, _rect_fem(3 * _LMAX, 1.2 * _LMAX))

    def test_non_square_within_the_ceiling_on_its_short_edge(self) -> None:
        # 3 x 0.9 l_max, no EAS: the long first edge does not count.
        spec = ASDShellQ4(pg="slab", section=_layered(_asd()), no_eas=True)
        _silent(spec, _QUAD, _rect_fem(3 * _LMAX, 0.9 * _LMAX))


class TestASDShellT3:
    def test_does_not_halve(self) -> None:
        # legs a: min node distance a (the hypotenuse is a sqrt 2).
        spec = ASDShellT3(pg="slab", section=_layered(_asd()))
        with pytest.warns(ASDRegularizationWarning, match=r"ratio 1\.20"):
            sweep_asdconcrete_element_size(spec, _TRI, _square_fem(1.2 * _LMAX))

    def test_silent_within_the_ceiling(self) -> None:
        spec = ASDShellT3(pg="slab", section=_layered(_asd()))
        _silent(spec, _TRI, _square_fem(0.9 * _LMAX))

    def test_non_square_reads_the_minimum_edge(self) -> None:
        # first edge 2 l_max, second 0.9 l_max: the short edge governs.
        spec = ASDShellT3(pg="slab", section=_layered(_asd()))
        _silent(spec, _TRI, _rect_fem(2 * _LMAX, 0.9 * _LMAX))


class TestManyElements:
    def test_counts_the_over_and_reports_the_worst(self) -> None:
        # minimum edges 1.2, 0.5, 1.5, 1.3 l_max: 3 of 4 over, worst 1.5 l_max = 1000.
        fem, elements = _strip_fem([
            (1.2 * _LMAX, 4 * _LMAX),
            (4 * _LMAX, 0.5 * _LMAX),
            (4 * _LMAX, 1.5 * _LMAX),
            (1.3 * _LMAX, 1.3 * _LMAX),
        ])
        spec = ASDShellQ4(pg="slab", section=_layered(_asd()), no_eas=True)
        with pytest.warns(ASDRegularizationWarning) as rec:
            sweep_asdconcrete_element_size(spec, elements, fem)
        assert len(rec) == 1
        msg = str(rec[0].message)
        assert "ASDConcrete: 3/4 elements" in msg
        assert "worst lch=1000," in msg
        assert "ratio 1.50" in msg

    def test_all_within_is_silent(self) -> None:
        fem, elements = _strip_fem([(0.9 * _LMAX, 4 * _LMAX), (4 * _LMAX, 0.5 * _LMAX)])
        spec = ASDShellQ4(pg="slab", section=_layered(_asd()), no_eas=True)
        _silent(spec, elements, fem)


class TestManyConcretes:
    @pytest.mark.parametrize("swap", [False, True])
    def test_every_concrete_over_is_listed_once(self, swap: bool) -> None:
        # a = 800: ASD ratio 800 / 666.67 = 1.20, Ladruno 800 / 317.46 = 2.52.
        mats = [_asd(), _ladruno(auto_regularize=True)]
        spec = ASDShellT3(pg="slab", section=_layered(*(mats[::-1] if swap else mats)))
        with pytest.warns(ASDRegularizationWarning) as rec:
            sweep_asdconcrete_element_size(spec, _TRI, _square_fem(800.0))
        assert len(rec) == 1
        msg = str(rec[0].message)
        assert "ASDConcrete: 1/1" in msg and "ratio 1.20" in msg
        assert "LadrunoConcrete3D: 1/1" in msg and "ratio 2.52" in msg

    @pytest.mark.parametrize("swap", [False, True])
    def test_only_the_concrete_over_is_listed(self, swap: bool) -> None:
        # a = 400: within ASD's 666.67, over Ladruno's 317.46 (ratio 1.26).
        mats = [_asd(), _ladruno(auto_regularize=True)]
        spec = ASDShellT3(pg="slab", section=_layered(*(mats[::-1] if swap else mats)))
        with pytest.warns(ASDRegularizationWarning, match=r"ratio 1\.26") as rec:
            sweep_asdconcrete_element_size(spec, _TRI, _square_fem(400.0))
        assert "ASDConcrete:" not in str(rec[0].message)


class TestLatchingHosts:
    """No length to compare: the fork reads another element's (#2680)."""

    @pytest.mark.parametrize("host", _LATCHING)
    @pytest.mark.parametrize("mat", [
        pytest.param(_asd(), id="asd"),
        pytest.param(_ladruno(auto_regularize=True), id="ladruno"),
        pytest.param(_rc(), id="rc"),
    ])
    def test_auto_regularized_warns_whatever_the_size(self, host, mat) -> None:
        # 1 mm is far inside every ceiling: the warning is not about size.
        spec = host(pg="slab", section=_layered(mat))
        with pytest.warns(ASDRegularizationWarning) as rec:
            sweep_asdconcrete_element_size(spec, _QUAD, _square_fem(1.0))
        assert len(rec) == 1
        msg = str(rec[0].message)
        assert f"section of {host.__name__} in PG 'slab'" in msg
        assert type(mat).__name__ in msg
        assert "latches the characteristic length" in msg
        assert "#2680" in msg
        assert "ratio" not in msg

    @pytest.mark.parametrize("host", _LATCHING)
    def test_no_ceiling_ratio_over_the_ceiling_either(self, host) -> None:
        spec = host(pg="slab", section=_layered(_asd()))
        with pytest.warns(ASDRegularizationWarning, match="latches") as rec:
            sweep_asdconcrete_element_size(spec, _QUAD, _square_fem(5 * _LMAX))
        assert len(rec) == 1
        assert "ratio" not in str(rec[0].message)

    def test_once_per_pg_for_many_elements_and_concretes(self) -> None:
        fem, elements = _strip_fem([(10.0, 10.0), (20.0, 20.0), (30.0, 30.0)])
        section = _layered(_asd(), _ladruno(auto_regularize=True))
        spec = ShellMITC4(pg="slab", section=section)
        with pytest.warns(ASDRegularizationWarning) as rec:
            sweep_asdconcrete_element_size(spec, elements, fem)
        assert len(rec) == 1
        msg = str(rec[0].message)
        assert "ASDConcrete3D" in msg and "LadrunoConcrete3D" in msg

    def test_no_elements_on_this_rank_stays_silent(self) -> None:
        spec = ShellMITC4(pg="slab", section=_layered(_asd()))
        _silent(spec, [], _square_fem(1.0))

    def test_raw_asd_without_provenance_still_latches(self) -> None:
        # No ceiling to check, but the fork still rescales by the latched length.
        raw = dataclasses.replace(_asd(), Gf=None)
        spec = ShellMITC4(pg="slab", section=_layered(raw))
        with pytest.warns(ASDRegularizationWarning, match="latches"):
            sweep_asdconcrete_element_size(spec, _QUAD, _square_fem(1.0))

    @pytest.mark.parametrize("host", _LATCHING)
    def test_without_auto_regularization_nothing_reads_the_host(self, host) -> None:
        asd = dataclasses.replace(_asd(), auto_regularize=False)
        rc = _rc(regularize=False)
        spec = host(pg="slab", section=_layered(asd, rc))
        _silent(spec, _QUAD, _square_fem(5 * _LMAX))

    @pytest.mark.parametrize("host", _LATCHING)
    def test_elastic_section_is_silent(self, host) -> None:
        spec = host(pg="slab", section=_layered(ElasticIsotropic(E=_E, nu=0.2)))
        _silent(spec, _QUAD, _square_fem(5 * _LMAX))

    @pytest.mark.parametrize("host", _LATCHING)
    def test_fixed_lch_keeps_its_check(self, host) -> None:
        # 2 x 317.46 = 634.92 > the ceiling, on a 1 mm mesh; it reads no element.
        mat = _ladruno(lch=2 * _LMAX_LADRUNO)
        spec = host(pg="slab", section=_layered(mat))
        with pytest.warns(ASDRegularizationWarning, match=r"the fixed lch=634\.9.*ratio 2\.00") as rec:
            sweep_asdconcrete_element_size(spec, _QUAD, _square_fem(1.0))
        assert len(rec) == 1
        assert "latches" not in str(rec[0].message)

    @pytest.mark.parametrize("host", _LATCHING)
    def test_fixed_lch_within_the_ceiling_is_silent(self, host) -> None:
        spec = host(pg="slab", section=_layered(_ladruno(lch=50.0)))
        _silent(spec, _QUAD, _square_fem(5 * _LMAX))

    def test_mixed_section_gets_both_warnings(self) -> None:
        section = _layered(_asd(), _ladruno(lch=2 * _LMAX_LADRUNO))
        spec = ShellMITC4(pg="slab", section=section)
        with pytest.warns(ASDRegularizationWarning) as rec:
            sweep_asdconcrete_element_size(spec, _QUAD, _square_fem(1.0))
        msgs = [str(r.message) for r in rec]
        assert len(msgs) == 2
        assert sum("latches" in m for m in msgs) == 1
        assert sum("the fixed lch=" in m for m in msgs) == 1

    def test_explicit_rc_lch_does_not_latch(self) -> None:
        ceiling = min(_rc().l_max())
        rc = dataclasses.replace(_rc(), lch=0.5 * ceiling)
        _silent(ShellMITC4(pg="slab", section=_layered(rc)), _QUAD, _square_fem(1.0))


class TestUnmirroredHost:
    def _host(self, section):
        return type("ZeroLengthSection", (), {"section": section, "pg": "z"})()

    def test_per_element_concrete_is_skipped(self) -> None:
        # The base rule would read 5 l_max here; this host is not mirrored.
        _silent(self._host(_layered(_asd())), _QUAD, _square_fem(5 * _LMAX))

    def test_fixed_lch_is_still_checked(self) -> None:
        spec = self._host(_layered(_ladruno(lch=2 * _LMAX_LADRUNO)))
        with pytest.warns(ASDRegularizationWarning, match="the fixed lch="):
            sweep_asdconcrete_element_size(spec, _QUAD, _square_fem(1.0))


class TestASDWithoutAutoRegularize:
    """Without the flag the fork never rescales the curves (ASDConcrete3DMaterial.cpp:1689)."""

    def test_section_path_is_silent_past_the_ceiling(self) -> None:
        asd = dataclasses.replace(_asd(), auto_regularize=False)
        spec = ASDShellT3(pg="slab", section=_layered(asd))
        _silent(spec, _TRI, _square_fem(2 * _LMAX))

    @pytest.mark.parametrize("make", [
        pytest.param(_asd, id="3d"),
        pytest.param(
            lambda: ASDConcrete1D.from_fc(E=_E, fc=_FC, ft=_FT, Gf=_GF, lch_ref=50.0), id="1d"),
    ])
    def test_direct_path_follows_the_flag(self, make) -> None:
        mat = make()
        fem = _square_fem(2 * _LMAX)
        with pytest.warns(ASDRegularizationWarning, match=r"ratio 2\.00"):
            sweep_asdconcrete_element_size(
                SimpleNamespace(material=mat, pg="b"), _QUAD, fem)
        off = dataclasses.replace(mat, auto_regularize=False)
        _silent(SimpleNamespace(material=off, pg="b"), _QUAD, fem)


class TestRCExplicitLch:
    """``P.lch > 0`` wins in the fork (LadrunoRCConcrete.cpp:336)."""

    def _ceiling(self) -> float:
        return min(_rc().l_max())

    def test_section_checks_the_lch_not_the_mesh(self) -> None:
        rc = dataclasses.replace(_rc(), lch=2 * self._ceiling())
        spec = ASDShellT3(pg="slab", section=_layered(rc))
        with pytest.warns(ASDRegularizationWarning, match=r"the fixed lch=.*ratio 2\.00"):
            sweep_asdconcrete_element_size(spec, _TRI, _square_fem(1.0))

    def test_section_within_the_ceiling_ignores_a_coarse_mesh(self) -> None:
        rc = dataclasses.replace(_rc(), lch=0.5 * self._ceiling())
        spec = ASDShellT3(pg="slab", section=_layered(rc))
        _silent(spec, _TRI, _square_fem(10 * self._ceiling()))

    def test_direct_checks_the_lch_not_the_mesh(self) -> None:
        over = dataclasses.replace(_rc(), lch=2 * self._ceiling())
        with pytest.warns(ASDRegularizationWarning, match=r"the fixed lch=.*ratio 2\.00"):
            sweep_asdconcrete_element_size(
                SimpleNamespace(material=over, pg="b"), _QUAD, _square_fem(1.0))
        within = dataclasses.replace(_rc(), lch=0.5 * self._ceiling())
        _silent(SimpleNamespace(material=within, pg="b"), _QUAD,
                _square_fem(10 * self._ceiling()))

    def test_lch_zero_keeps_the_element_length(self) -> None:
        rc = _rc()
        assert rc.lch == 0.0
        spec = ASDShellT3(pg="slab", section=_layered(rc))
        with pytest.warns(ASDRegularizationWarning, match=r"1/1 elements.*ratio 2\.00"):
            sweep_asdconcrete_element_size(spec, _TRI, _square_fem(2 * self._ceiling()))

    def test_without_auto_regularization_the_lch_is_not_checked(self) -> None:
        rc = dataclasses.replace(_rc(regularize=False), lch=1.0e6)
        _silent(ASDShellT3(pg="slab", section=_layered(rc)), _TRI, _square_fem(1.0))


class TestDegenerateEnergy:
    """A zero ``ft`` or ``Gf`` has no ceiling; emit must go on."""

    @pytest.mark.parametrize("kw", [{"Gf": 0.0}, {"ft": 0.0}], ids=["Gf0", "ft0"])
    def test_section_path_skips(self, kw) -> None:
        mat = dataclasses.replace(_asd(), **kw)
        spec = ASDShellT3(pg="slab", section=_layered(mat))
        _silent(spec, _TRI, _square_fem(1000.0))

    @pytest.mark.parametrize("kw", [{"Gf": 0.0}, {"ft": 0.0}], ids=["Gf0", "ft0"])
    def test_direct_path_skips(self, kw) -> None:
        for make in (_asd, lambda: ASDConcrete1D.from_fc(
                E=_E, fc=_FC, ft=_FT, Gf=_GF, lch_ref=50.0)):
            mat = dataclasses.replace(make(), **kw)
            _silent(SimpleNamespace(material=mat, pg="b"), _QUAD, _square_fem(1000.0))

    def test_a_sibling_concrete_is_still_checked(self) -> None:
        dead = dataclasses.replace(_asd(), Gf=0.0)
        spec = ASDShellT3(pg="slab", section=_layered(dead, _asd()))
        with pytest.warns(ASDRegularizationWarning, match=r"ratio 1\.20"):
            sweep_asdconcrete_element_size(spec, _TRI, _square_fem(800.0))


class TestGraphAndLadruno:
    def test_modifier_inner_section_is_reached(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the nonlinear-inner advisory
            sec = LadrunoShellModifier(inner=_layered(_asd()))
        spec = ASDShellT3(pg="slab", section=sec)
        with pytest.warns(ASDRegularizationWarning, match=r"section of ASDShellT3.*ratio 1\.20"):
            sweep_asdconcrete_element_size(spec, _TRI, _square_fem(1.2 * _LMAX))

    def test_modifier_inner_section_latches(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sec = LadrunoShellModifier(inner=_layered(_asd()))
        spec = ShellMITC4(pg="slab", section=sec)
        with pytest.warns(ASDRegularizationWarning, match="latches"):
            sweep_asdconcrete_element_size(spec, _QUAD, _square_fem(1.0))

    def test_ladruno_fixed_lch_is_checked_whatever_the_mesh(self) -> None:
        mat = _ladruno(lch=1.0e4)
        spec = ASDShellQ4(pg="slab", section=_layered(mat))
        with pytest.warns(ASDRegularizationWarning, match="LadrunoConcrete3D: the fixed lch=10000"):
            sweep_asdconcrete_element_size(spec, _QUAD, _square_fem(1.0))

    def test_ladruno_auto_regularize_uses_the_host_length(self) -> None:
        mat = _ladruno(auto_regularize=True)
        spec = ASDShellQ4(pg="slab", section=_layered(mat))
        _silent(spec, _QUAD, _square_fem(1.5 * _LMAX_LADRUNO))  # EAS: 0.75 ceiling
        with pytest.warns(ASDRegularizationWarning, match=r"LadrunoConcrete3D: 1/1 .*ratio 1\.50"):
            sweep_asdconcrete_element_size(spec, _QUAD, _square_fem(3.0 * _LMAX_LADRUNO))

    def test_raw_asd_without_provenance_is_skipped(self) -> None:
        raw = dataclasses.replace(_asd(), Gf=None)
        assert raw.l_max() is None
        _silent(ASDShellT3(pg="slab", section=_layered(raw)), _TRI, _square_fem(1.0e6))
