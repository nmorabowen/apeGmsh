"""Unit tests for ``FrameModifiers`` and the ``modifiers=`` frame route.

The emitted args must be ``base x modifier`` for every field, in 2-D and
3-D, on ``elasticBeamColumn`` / ``ElasticTimoshenkoBeam`` (scalars) and on
``forceBeamColumn`` / ``dispBeamColumn`` (modified ``ElasticSection`` and
integration copies, deduplicated, the declared section untouched). Every
modifier the element cannot apply is refused. The live oracles are in
``tests/opensees/live/test_frame_modifiers_live.py``.
"""
from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest

from apeGmsh.opensees import FrameModifiers, apeSees
from apeGmsh.opensees.element.beam_column import (
    ElasticTimoshenkoBeam,
    dispBeamColumn,
    elasticBeamColumn,
    forceBeamColumn,
)
from apeGmsh.opensees.element.frame_modifiers import emitted_integration
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.integration import HingeRadau, Lobatto
from apeGmsh.opensees.section.beam import ElasticSection
from apeGmsh.opensees.transform import Linear

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
    make_two_node_beam,
)

E, G = 2.0e11, 8.0e10
A, IZ, IY, J = 0.01, 2.0e-4, 1.0e-4, 5.0e-5
AVY, AVZ = 0.008, 0.006
MASS = 50.0

#: Distinct, non-representable-as-power-of-2 factors so every product is
#: checked against the right base.
M = FrameModifiers(A=0.81, Avy=0.62, Avz=0.43, J=0.24, Iy=0.35, Iz=0.76,
                   mass=0.57)


def _bridge(ndm: int = 3, fem: Any = None) -> apeSees:
    ops = apeSees(cast(Any, fem if fem is not None else make_two_node_beam(ndm=ndm)))
    ops.model(ndm=ndm, ndf=6 if ndm == 3 else 3)
    return ops


def _calls(ops: apeSees, verb: str) -> list[tuple[Any, ...]]:
    rec = RecordingEmitter()
    ops.build().emit(rec)
    return [c[1] for c in rec.calls if c[0] == verb]


def _two_pg_fem() -> FEMStub:
    """Two columns, each its own element PG (``ColA``, ``ColB``, ``ColC``)."""
    return FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4, 5, 6],
            coords=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
                    (1.0, 0.0, 0.0), (1.0, 0.0, 1.0),
                    (2.0, 0.0, 0.0), (2.0, 0.0, 1.0)],
            node_pgs={"Base": [1, 3, 5], "Top": [2, 4, 6]},
        ),
        elements=_ElementsStub(elem_pgs={
            "ColA": _ElementGroupView(ids=(1,), connectivity=((1, 2),)),
            "ColB": _ElementGroupView(ids=(2,), connectivity=((3, 4),)),
            "ColC": _ElementGroupView(ids=(3,), connectivity=((5, 6),)),
        }),
    )


# ===========================================================================
# The value object
# ===========================================================================

class TestFrameModifiersValue:
    def test_defaults_are_identity(self) -> None:
        m = FrameModifiers()
        assert m.is_identity
        assert m.non_unit() == {}

    def test_etabs_maps_2_to_y_and_3_to_z(self) -> None:
        m = FrameModifiers.etabs(A=0.1, As2=0.2, As3=0.3, J=0.4, I22=0.5,
                                 I33=0.6, mass=0.7, weight=0.8)
        assert m == FrameModifiers(A=0.1, Avy=0.2, Avz=0.3, J=0.4, Iy=0.5,
                                   Iz=0.6, mass=0.7, weight=0.8)

    @pytest.mark.parametrize("name", ["A", "Avy", "Avz", "J", "Iy", "Iz"])
    @pytest.mark.parametrize("bad", [0.0, -0.5])
    def test_stiffness_must_be_positive(self, name: str, bad: float) -> None:
        with pytest.raises(ValueError, match=f"stiffness modifier {name}"):
            FrameModifiers(**{name: bad})

    @pytest.mark.parametrize("name", ["mass", "weight"])
    def test_mass_and_weight_may_be_zero_not_negative(self, name: str) -> None:
        assert getattr(FrameModifiers(**{name: 0.0}), name) == 0.0
        with pytest.raises(ValueError, match=">= 0"):
            FrameModifiers(**{name: -0.1})

    @pytest.mark.parametrize("bad", [math.nan, math.inf])
    def test_non_finite_refused(self, bad: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            FrameModifiers(Iz=bad)

    def test_bool_refused(self) -> None:
        with pytest.raises(TypeError, match="number"):
            FrameModifiers(Iz=True)


class TestEtabsAxisClaim:
    """OpenSees ``geomTransf``: local y = vecxz x x, local z = x x y. With
    ``vecxz`` along ETABS 3, z = ETABS 3 and y = ETABS 2 (both triads
    right-handed with a common first axis)."""

    @staticmethod
    def _opensees_axes(x: np.ndarray, vecxz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y = np.cross(vecxz, x)
        y /= np.linalg.norm(y)
        z = np.cross(x, y)
        return y, z

    @pytest.mark.parametrize("e1, e2", [
        ((0, 0, 1), (1, 0, 0)),        # vertical column: 2 = +X
        ((1, 0, 0), (0, 0, 1)),        # beam along +X: 2 = +Z
        ((0, 1, 0), (0, 0, 1)),        # beam along +Y: 2 = +Z
        ((0.6, 0.8, 0), (0, 0, 1)),    # skew beam
    ])
    def test_vecxz_along_etabs_3_gives_y_2_and_z_3(self, e1: Any, e2: Any) -> None:
        x = np.asarray(e1, dtype=float)
        e2v = np.asarray(e2, dtype=float)
        e3 = np.cross(x, e2v)          # ETABS 3 = 1 x 2 (right-handed)
        y, z = self._opensees_axes(x, e3)
        np.testing.assert_allclose(y, e2v, atol=1e-15)
        np.testing.assert_allclose(z, e3, atol=1e-15)

    def test_importer_vertical_vecxz_puts_i33_on_etabs_2(self) -> None:
        """The ETABS importer's vertical-member ``vecxz=(1, 0, 0)`` is
        ETABS 2 for a default column (1 = +Z, 2 = +X, 3 = +Y), not 3:
        OpenSees local z is then ETABS 2, so ``etabs(I33=...)``, which
        scales ``Iz``, lands on the minor axis. Pins the warning in the
        ``FrameModifiers.etabs`` docstring; when the importer moves its
        ``vecxz`` onto ETABS 3, this test and that note change together."""
        from apeGmsh.interop.etabs_import import _VECXZ

        assert _VECXZ["v"] == (1.0, 0.0, 0.0)
        assert FrameModifiers.etabs(I33=0.5).non_unit() == {"Iz": 0.5}
        x = np.array([0.0, 0.0, 1.0])
        e2 = np.array([1.0, 0.0, 0.0])
        e3 = np.cross(x, e2)
        y, z = self._opensees_axes(x, np.asarray(_VECXZ["v"], dtype=float))
        np.testing.assert_allclose(z, e2, atol=1e-15)      # Iz is about 2
        np.testing.assert_allclose(y, -e3, atol=1e-15)


# ===========================================================================
# Scalar routes: elasticBeamColumn, ElasticTimoshenkoBeam
# ===========================================================================

class TestElasticBeamColumnModifiers:
    def test_3d_every_field_scaled(self) -> None:
        ops = _bridge()
        m = FrameModifiers(A=M.A, J=M.J, Iy=M.Iy, Iz=M.Iz, mass=M.mass)
        ops.element.elasticBeamColumn(
            pg="Cols", transf=ops.geomTransf.Linear(vecxz=(0.0, 1.0, 0.0)),
            A=A, E=E, Iz=IZ, Iy=IY, G=G, J=J, mass=MASS, modifiers=m,
        )
        (args,) = _calls(ops, "element")
        # elasticBeamColumn tag i j A E G J Iy Iz transf -mass m
        assert args[4:10] == (A * m.A, E, G, J * m.J, IY * m.Iy, IZ * m.Iz)
        assert args[-2:] == ("-mass", MASS * m.mass)

    def test_2d_every_field_scaled(self) -> None:
        ops = _bridge(ndm=2)
        m = FrameModifiers(A=M.A, Iz=M.Iz, mass=M.mass)
        ops.element.elasticBeamColumn(
            pg="Cols", transf=ops.geomTransf.Linear(), A=A, E=E, Iz=IZ,
            mass=MASS, modifiers=m,
        )
        (args,) = _calls(ops, "element")
        assert args[4:7] == (A * m.A, E, IZ * m.Iz)
        assert args[-2:] == ("-mass", MASS * m.mass)

    def test_identity_emits_the_declared_args(self) -> None:
        def run(mods: FrameModifiers | None) -> list[tuple[Any, ...]]:
            ops = _bridge()
            ops.element.elasticBeamColumn(
                pg="Cols", transf=ops.geomTransf.Linear(vecxz=(0.0, 1.0, 0.0)),
                A=A, E=E, Iz=IZ, Iy=IY, G=G, J=J, mass=MASS, modifiers=mods,
            )
            return _calls(ops, "element")
        assert run(FrameModifiers()) == run(None)

    def test_declared_fields_unchanged(self) -> None:
        e = elasticBeamColumn(pg="Cols", transf=Linear(vecxz=(0.0, 1.0, 0.0)),
                              A=A, E=E, Iz=IZ, Iy=IY, G=G, J=J,
                              modifiers=FrameModifiers(A=0.5))
        assert e.A == A and e.modifiers == FrameModifiers(A=0.5)

    @pytest.mark.parametrize("name", ["Avy", "Avz"])
    def test_shear_modifier_refused(self, name: str) -> None:
        with pytest.raises(ValueError, match="Euler-Bernoulli"):
            elasticBeamColumn(pg="Cols", transf=Linear(vecxz=(0.0, 1.0, 0.0)),
                              A=A, E=E, Iz=IZ, Iy=IY, G=G, J=J,
                              modifiers=FrameModifiers(**{name: 0.5}))

    @pytest.mark.parametrize("name", ["Iy", "J"])
    def test_2d_refuses_out_of_plane(self, name: str) -> None:
        with pytest.raises(ValueError, match="2-D element"):
            elasticBeamColumn(pg="Cols", transf=Linear(), A=A, E=E, Iz=IZ,
                              modifiers=FrameModifiers(**{name: 0.5}))

    def test_weight_refused(self) -> None:
        with pytest.raises(ValueError, match="no frame self-weight"):
            elasticBeamColumn(pg="Cols", transf=Linear(), A=A, E=E, Iz=IZ,
                              modifiers=FrameModifiers(weight=1.1))

    def test_mass_without_element_mass_refused(self) -> None:
        with pytest.raises(ValueError, match="without mass="):
            elasticBeamColumn(pg="Cols", transf=Linear(), A=A, E=E, Iz=IZ,
                              modifiers=FrameModifiers(mass=0.5))

    def test_not_a_frame_modifiers_refused(self) -> None:
        with pytest.raises(TypeError, match="FrameModifiers"):
            elasticBeamColumn(pg="Cols", transf=Linear(), A=A, E=E, Iz=IZ,
                              modifiers=cast(Any, {"A": 0.5}))


class TestElasticTimoshenkoModifiers:
    def test_3d_every_field_scaled(self) -> None:
        ops = _bridge()
        ops.element.ElasticTimoshenkoBeam(
            pg="Cols", transf=ops.geomTransf.Linear(vecxz=(0.0, 1.0, 0.0)),
            E=E, G=G, A=A, Iz=IZ, Avy=AVY, Iy=IY, J=J, Avz=AVZ, mass=MASS,
            modifiers=M,
        )
        (args,) = _calls(ops, "element")
        # tag i j E G A J Iy Iz Avy Avz transf -mass m
        assert args[4:12] == (E, G, A * M.A, J * M.J, IY * M.Iy, IZ * M.Iz,
                              AVY * M.Avy, AVZ * M.Avz)
        assert args[-2:] == ("-mass", MASS * M.mass)

    def test_2d_every_field_scaled(self) -> None:
        ops = _bridge(ndm=2)
        m = FrameModifiers(A=M.A, Avy=M.Avy, Iz=M.Iz, mass=M.mass)
        ops.element.ElasticTimoshenkoBeam(
            pg="Cols", transf=ops.geomTransf.Linear(), E=E, G=G, A=A, Iz=IZ,
            Avy=AVY, mass=MASS, modifiers=m,
        )
        (args,) = _calls(ops, "element")
        assert args[4:9] == (E, G, A * m.A, IZ * m.Iz, AVY * m.Avy)
        assert args[-2:] == ("-mass", MASS * m.mass)

    @pytest.mark.parametrize("name", ["Iy", "J", "Avz"])
    def test_2d_refuses_out_of_plane(self, name: str) -> None:
        with pytest.raises(ValueError, match="2-D element"):
            ElasticTimoshenkoBeam(pg="Cols", transf=Linear(), E=E, G=G, A=A,
                                  Iz=IZ, Avy=AVY,
                                  modifiers=FrameModifiers(**{name: 0.5}))

    def test_weight_refused(self) -> None:
        with pytest.raises(ValueError, match="no frame self-weight"):
            ElasticTimoshenkoBeam(pg="Cols", transf=Linear(), E=E, G=G, A=A,
                                  Iz=IZ, Avy=AVY,
                                  modifiers=FrameModifiers(weight=0.0))


# ===========================================================================
# Section route: forceBeamColumn / dispBeamColumn over ElasticSection
# ===========================================================================

_SEC_M = FrameModifiers(A=M.A, J=M.J, Iy=M.Iy, Iz=M.Iz, mass=M.mass)


class TestSectionRouteModifiers:
    @pytest.mark.parametrize("family", ["forceBeamColumn", "dispBeamColumn"])
    def test_3d_modified_copy_emitted(self, family: str) -> None:
        ops = _bridge()
        sec = ops.section.Elastic(E=E, A=A, Iz=IZ, Iy=IY, G=G, J=J)
        snapshot = ElasticSection(E=E, A=A, Iz=IZ, Iy=IY, G=G, J=J)
        integ = ops.beamIntegration.Lobatto(section=sec, n_ip=3)
        getattr(ops.element, family)(
            pg="Cols", transf=ops.geomTransf.Linear(vecxz=(0.0, 1.0, 0.0)),
            integration=integ, mass=MASS, modifiers=_SEC_M,
        )
        secs = _calls(ops, "section")
        integs = _calls(ops, "beamIntegration")
        (elem,) = _calls(ops, "element")
        # Declared section first, untouched; the copy second.
        assert secs[0] == ("Elastic", 1, E, A, IZ, IY, G, J)
        assert secs[1] == ("Elastic", 2, E, A * _SEC_M.A, IZ * _SEC_M.Iz,
                           IY * _SEC_M.Iy, G, J * _SEC_M.J)
        assert integs == [("Lobatto", 1, 1, 3), ("Lobatto", 2, 2, 3)]
        assert elem[5] == 2                      # references the copy
        assert elem[-2:] == ("-mass", MASS * _SEC_M.mass)
        assert sec == snapshot                   # shared section not mutated

    def test_2d_modified_copy_emitted(self) -> None:
        ops = _bridge(ndm=2)
        m = FrameModifiers(A=M.A, Iz=M.Iz)
        sec = ops.section.Elastic(E=E, A=A, Iz=IZ)
        ops.element.forceBeamColumn(
            pg="Cols", transf=ops.geomTransf.Linear(),
            integration=ops.beamIntegration.Lobatto(section=sec, n_ip=3),
            modifiers=m,
        )
        secs = _calls(ops, "section")
        assert secs[1] == ("Elastic", 2, E, A * m.A, IZ * m.Iz)

    def test_shear_section_alpha_rescaled(self) -> None:
        """Shear area alpha*A: alpha' = alpha * m_Av / m_A, so the shear
        area scales by m_Av only."""
        ops = _bridge()
        ay, az = AVY / A, AVZ / A
        sec = ops.section.Elastic(E=E, A=A, Iz=IZ, Iy=IY, G=G, J=J,
                                  alphaY=ay, alphaZ=az)
        m = FrameModifiers(A=M.A, Avy=M.Avy, Avz=M.Avz, J=M.J, Iy=M.Iy,
                           Iz=M.Iz)
        ops.element.forceBeamColumn(
            pg="Cols", transf=ops.geomTransf.Linear(vecxz=(0.0, 1.0, 0.0)),
            integration=ops.beamIntegration.Lobatto(section=sec, n_ip=3),
            modifiers=m,
        )
        copy = _calls(ops, "section")[1]
        a_new, ay_new, az_new = copy[3], copy[8], copy[9]
        assert a_new == A * m.A
        assert ay_new * a_new == pytest.approx(AVY * m.Avy, rel=1e-15)
        assert az_new * a_new == pytest.approx(AVZ * m.Avz, rel=1e-15)

    def test_shear_section_A_modifier_alone_keeps_shear_area(self) -> None:
        ops = _bridge()
        sec = ops.section.Elastic(E=E, A=A, Iz=IZ, Iy=IY, G=G, J=J,
                                  alphaY=0.8, alphaZ=0.9)
        ops.element.forceBeamColumn(
            pg="Cols", transf=ops.geomTransf.Linear(vecxz=(0.0, 1.0, 0.0)),
            integration=ops.beamIntegration.Lobatto(section=sec, n_ip=3),
            modifiers=FrameModifiers(A=0.5),
        )
        copy = _calls(ops, "section")[1]
        assert copy[3] * copy[8] == pytest.approx(A * 0.8, rel=1e-15)
        assert copy[3] * copy[9] == pytest.approx(A * 0.9, rel=1e-15)

    def test_shared_section_deduplicated_per_modifier_set(self) -> None:
        """One section, three groups: two with the same modifiers share one
        copy, the third gets its own; the declared section is emitted once
        and never changed."""
        ops = _bridge(fem=_two_pg_fem())
        sec = ops.section.Elastic(E=E, A=A, Iz=IZ, Iy=IY, G=G, J=J)
        integ = ops.beamIntegration.Lobatto(section=sec, n_ip=3)
        t = ops.geomTransf.Linear(vecxz=(0.0, 1.0, 0.0))
        cracked = FrameModifiers(Iy=0.35, Iz=0.35)
        ops.element.forceBeamColumn(pg="ColA", transf=t, integration=integ,
                                    modifiers=cracked)
        ops.element.forceBeamColumn(pg="ColB", transf=t, integration=integ,
                                    modifiers=FrameModifiers(Iy=0.35, Iz=0.35))
        ops.element.dispBeamColumn(pg="ColC", transf=t, integration=integ,
                                   modifiers=FrameModifiers(Iy=0.7, Iz=0.7))
        secs = _calls(ops, "section")
        assert len(secs) == 3                   # declared + 2 distinct copies
        assert secs[0][2:] == (E, A, IZ, IY, G, J)
        assert sorted(s[4] for s in secs[1:]) == [IZ * 0.35, IZ * 0.7]
        elems = _calls(ops, "element")
        integ_tags = [e[5] for e in elems]
        assert integ_tags[0] == integ_tags[1] != integ_tags[2]
        assert 1 not in integ_tags              # never the declared rule

    def test_hinge_rule_every_section_copied(self) -> None:
        ops = _bridge()
        s_end = ops.section.Elastic(E=E, A=A, Iz=IZ, Iy=IY, G=G, J=J)
        s_mid = ops.section.Elastic(E=E, A=A, Iz=3 * IZ, Iy=3 * IY, G=G, J=J)
        rule = ops.beamIntegration.HingeRadau(
            section_i=s_end, lp_i=0.1, section_j=s_end, lp_j=0.1,
            section_interior=s_mid,
        )
        ops.element.forceBeamColumn(
            pg="Cols", transf=ops.geomTransf.Linear(vecxz=(0.0, 1.0, 0.0)),
            integration=rule, modifiers=FrameModifiers(Iz=0.5),
        )
        secs = _calls(ops, "section")
        assert len(secs) == 4                   # 2 declared + 2 copies
        izs = sorted(s[4] for s in secs)
        assert izs == sorted([IZ, 3 * IZ, IZ * 0.5, 3 * IZ * 0.5])
        (hinge_copy,) = [c for c in _calls(ops, "beamIntegration")
                         if c[1] == 2]
        copy_tags = {s[1] for s in secs if s[4] in (IZ * 0.5, 3 * IZ * 0.5)}
        assert {hinge_copy[2], hinge_copy[4], hinge_copy[6]} == copy_tags

    def test_identity_makes_no_copy(self) -> None:
        ops = _bridge()
        sec = ops.section.Elastic(E=E, A=A, Iz=IZ, Iy=IY, G=G, J=J)
        ops.element.forceBeamColumn(
            pg="Cols", transf=ops.geomTransf.Linear(vecxz=(0.0, 1.0, 0.0)),
            integration=ops.beamIntegration.Lobatto(section=sec, n_ip=3),
            modifiers=FrameModifiers(),
        )
        assert len(_calls(ops, "section")) == 1

    def test_standalone_spec_registered_p11(self) -> None:
        ops = _bridge()
        sec = ops.section.Elastic(E=E, A=A, Iz=IZ, Iy=IY, G=G, J=J)
        integ = ops.beamIntegration.Lobatto(section=sec, n_ip=3)
        t = ops.geomTransf.Linear(vecxz=(0.0, 1.0, 0.0))
        spec = forceBeamColumn(pg="Cols", transf=t, integration=integ,
                               modifiers=FrameModifiers(Iz=0.5))
        ops.register(spec)
        ops.register(spec)                      # idempotent
        secs = _calls(ops, "section")
        assert [s[4] for s in secs] == [IZ, IZ * 0.5]

    @staticmethod
    def _cracked_column(ops: apeSees) -> forceBeamColumn:
        sec = ops.section.Elastic(E=E, A=A, Iz=IZ, Iy=IY, G=G, J=J)
        return ops.element.forceBeamColumn(
            pg="Cols", transf=ops.geomTransf.Linear(vecxz=(0.0, 1.0, 0.0)),
            integration=ops.beamIntegration.Lobatto(section=sec, n_ip=3),
            modifiers=FrameModifiers(Iz=0.5),
        )

    def test_two_sessions_do_not_share_copies(self) -> None:
        """Equal sections and modifiers in two bridges: each bridge makes,
        registers and emits its own copies (the memo is per bridge, not
        process-wide)."""
        o1, o2 = _bridge(), _bridge()
        s1, s2 = self._cracked_column(o1), self._cracked_column(o2)
        i1, i2 = emitted_integration(s1), emitted_integration(s2)
        assert i1 == i2 and i1 is not i2
        assert i1.section is not i2.section     # type: ignore[attr-defined]
        assert o1.tag_for(i1) is not None and o2.tag_for(i2) is not None
        assert o1.tag_for(i2) is None and o2.tag_for(i1) is None
        assert o1._frame_modifier_copies is not o2._frame_modifier_copies
        for ops in (o1, o2):
            secs = _calls(ops, "section")
            assert [s[4] for s in secs] == [IZ, IZ * 0.5]
            (elem,) = _calls(ops, "element")
            assert elem[5] == 2                 # its own copy of the rule

    def test_spec_shared_by_two_sessions_keeps_its_copy(self) -> None:
        """One standalone spec registered on two bridges keeps the copy it
        was bound to first (its tag is looked up by identity), and both
        bridges register and emit that copy."""
        o1, o2 = _bridge(), _bridge()
        sec = ElasticSection(E=E, A=A, Iz=IZ, Iy=IY, G=G, J=J)
        integ = Lobatto(section=sec, n_ip=3)
        transf = Linear(vecxz=(0.0, 1.0, 0.0))
        spec = forceBeamColumn(pg="Cols", transf=transf, integration=integ,
                               modifiers=FrameModifiers(Iz=0.5))
        for ops in (o1, o2):
            for prim in (sec, integ, transf, spec):
                ops.register(prim)
        bound = emitted_integration(spec)
        assert o1.tag_for(bound) is not None and o2.tag_for(bound) is not None
        for ops in (o1, o2):
            assert [s[4] for s in _calls(ops, "section")] == [IZ, IZ * 0.5]

    def test_non_elastic_section_refused(self) -> None:
        ops = _bridge()
        mat = ops.uniaxialMaterial.ElasticMaterial(E=E)
        fiber = ops.section.Aggregator(materials_by_dof={"P": mat, "Mz": mat})
        with pytest.raises(TypeError, match="only an ElasticSection"):
            ops.element.forceBeamColumn(
                pg="Cols", transf=ops.geomTransf.Linear(vecxz=(0.0, 1.0, 0.0)),
                integration=ops.beamIntegration.Lobatto(section=fiber, n_ip=3),
                modifiers=FrameModifiers(Iz=0.5),
            )

    @pytest.mark.parametrize("name", ["Avy", "Avz"])
    def test_shear_modifier_needs_shear_section(self, name: str) -> None:
        sec = ElasticSection(E=E, A=A, Iz=IZ, Iy=IY, G=G, J=J)
        with pytest.raises(ValueError, match="no shear flexibility"):
            forceBeamColumn(pg="Cols", transf=Linear(vecxz=(0.0, 1.0, 0.0)),
                            integration=Lobatto(section=sec, n_ip=3),
                            modifiers=FrameModifiers(**{name: 0.5}))

    @pytest.mark.parametrize("name", ["Iy", "J", "Avz"])
    def test_2d_section_refuses_out_of_plane(self, name: str) -> None:
        sec = ElasticSection(E=E, A=A, Iz=IZ, G=G, alphaY=0.8)
        with pytest.raises(ValueError, match="2-D ElasticSection"):
            dispBeamColumn(pg="Cols", transf=Linear(),
                           integration=Lobatto(section=sec, n_ip=3),
                           modifiers=FrameModifiers(**{name: 0.5}))

    def test_weight_and_massless_refused(self) -> None:
        sec = ElasticSection(E=E, A=A, Iz=IZ)
        rule = Lobatto(section=sec, n_ip=3)
        with pytest.raises(ValueError, match="no frame self-weight"):
            forceBeamColumn(pg="Cols", transf=Linear(), integration=rule,
                            modifiers=FrameModifiers(weight=2.0))
        with pytest.raises(ValueError, match="without mass="):
            forceBeamColumn(pg="Cols", transf=Linear(), integration=rule,
                            modifiers=FrameModifiers(mass=2.0))

    def test_hinge_rule_with_any_non_elastic_section_refused(self) -> None:
        ops = _bridge()
        mat = ops.uniaxialMaterial.ElasticMaterial(E=E)
        agg = ops.section.Aggregator(materials_by_dof={"P": mat, "Mz": mat})
        mid = ElasticSection(E=E, A=A, Iz=IZ, Iy=IY, G=G, J=J)
        rule = HingeRadau(section_i=agg, lp_i=0.1, section_j=agg, lp_j=0.1,
                          section_interior=mid)
        with pytest.raises(TypeError, match="Aggregator"):
            forceBeamColumn(pg="Cols", transf=Linear(vecxz=(0.0, 1.0, 0.0)),
                            integration=rule, modifiers=FrameModifiers(Iz=0.5))
