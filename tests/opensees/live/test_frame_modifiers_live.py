"""Live oracles for frame property modifiers (``FrameModifiers``).

A unit-length cantilever column along +Z (``make_two_node_beam``), its base
fixed, is loaded at the tip by an axial force, a force along each ETABS
transverse axis and a torque, all at once (the responses are uncoupled).
``vecxz=(0, 1, 0)`` puts OpenSees local z on ETABS local 3 (+Y for a
vertical ETABS column) and so local y on ETABS 2 (+X): the ETABS-named
modifiers then have to land as

- ``u_z = P L / (E A m_A)``
- ``u_x = P L^3 / (3 E I33 m_I33) [+ P L / (G As2 m_As2)]``  (force along 2)
- ``u_y = P L^3 / (3 E I22 m_I22) [+ P L / (G As3 m_As3)]``  (force along 3)
- ``theta_z = T L / (G J m_J)``

for each frame route: ``elasticBeamColumn``, ``ElasticTimoshenkoBeam``,
``forceBeamColumn`` / ``dispBeamColumn`` over an ``ElasticSection`` (the
force-based one also over a shear ``ElasticSection``). The mass modifier
is checked on a two-DOF modal problem (tip ``u_x`` and ``u_z`` free, tip
rotations held): with the lumped ``-mass`` the tip carries ``m L / 2``, so
``omega^2 = (E A m_A / L) / (m m_mass L / 2)`` (axial) and
``omega^2 = (12 E I33 m_I33 / L^3) / (m m_mass L / 2)`` (guided bending).

Stock elements only: these run on stock openseespy (``live-stock``).
"""
from __future__ import annotations

from typing import Any, cast

import pytest

from apeGmsh.opensees import FrameModifiers, apeSees

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
    make_two_node_beam,
)

pytestmark = pytest.mark.live

L = 1.0
E, G = 2.0e11, 8.0e10
A, IZ, IY, J = 0.01, 2.0e-4, 1.0e-4, 5.0e-5      # IZ = I33, IY = I22
AVY, AVZ = 0.008, 0.006                           # As2, As3
P, T = 1.0e4, 2.0e3
M_LEN = 50.0                                      # mass per unit length

MODS = FrameModifiers.etabs(
    A=0.8, As2=0.6, As3=0.45, J=0.2, I22=0.35, I33=0.7,
)
NO_SHEAR = FrameModifiers.etabs(A=0.8, J=0.2, I22=0.35, I33=0.7)
WITH_MASS = FrameModifiers.etabs(A=0.8, J=0.2, I22=0.35, I33=0.7, mass=0.5)

RTOL = 1e-10


@pytest.fixture(autouse=True)
def _backend() -> None:
    pytest.importorskip("openseespy.opensees")


def _bridge(ndm: int = 3) -> apeSees:
    ops = apeSees(cast(Any, make_two_node_beam(ndm=ndm)))
    ops.model(ndm=ndm, ndf=6 if ndm == 3 else 3)
    return ops


def _transf(ops: apeSees) -> Any:
    # vecxz along ETABS 3 (+Y) for a vertical ETABS column.
    return ops.geomTransf.Linear(vecxz=(0.0, 1.0, 0.0))


def _static(ops: apeSees, forces: tuple[float, ...], ndf: int) -> Any:
    ops.fix(pg="Base", dofs=(1,) * ndf)
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=ops.nodes.one(pg="Top"), forces=forces)
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-12, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    assert ops.analyze(steps=1) == 0
    return ops.nodes.one(pg="Top")


def _expected_3d(m: FrameModifiers, shear: bool) -> dict[str, float]:
    ux = P * L**3 / (3 * E * IZ * m.Iz)
    uy = P * L**3 / (3 * E * IY * m.Iy)
    if shear:
        ux += P * L / (G * AVY * m.Avy)
        uy += P * L / (G * AVZ * m.Avz)
    return {
        "displacement_x": ux,
        "displacement_y": uy,
        "displacement_z": P * L / (E * A * m.A),
        "rotation_z": T * L / (G * J * m.J),
    }


def _check_3d(ops: apeSees, m: FrameModifiers, *, shear: bool) -> None:
    top = _static(ops, (P, P, P, 0.0, 0.0, T), ndf=6)
    for comp, expected in _expected_3d(m, shear).items():
        assert top.read(comp) == pytest.approx(expected, rel=RTOL), comp


# ---------------------------------------------------------------------------
# Static: each frame route, 3-D
# ---------------------------------------------------------------------------

def test_elastic_beam_column_3d() -> None:
    ops = _bridge()
    ops.element.elasticBeamColumn(
        pg="Cols", transf=_transf(ops), A=A, E=E, Iz=IZ, Iy=IY, G=G, J=J,
        modifiers=NO_SHEAR,
    )
    _check_3d(ops, NO_SHEAR, shear=False)


def test_elastic_timoshenko_beam_3d() -> None:
    ops = _bridge()
    ops.element.ElasticTimoshenkoBeam(
        pg="Cols", transf=_transf(ops), E=E, G=G, A=A, Iz=IZ, Avy=AVY,
        Iy=IY, J=J, Avz=AVZ, modifiers=MODS,
    )
    _check_3d(ops, MODS, shear=True)


@pytest.mark.parametrize("family", ["forceBeamColumn", "dispBeamColumn"])
def test_beam_with_elastic_section_3d(family: str) -> None:
    ops = _bridge()
    sec = ops.section.Elastic(E=E, A=A, Iz=IZ, Iy=IY, G=G, J=J)
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=3)
    getattr(ops.element, family)(
        pg="Cols", transf=_transf(ops), integration=integ,
        modifiers=NO_SHEAR,
    )
    _check_3d(ops, NO_SHEAR, shear=False)


def test_force_beam_column_shear_section_3d() -> None:
    """A shear ``ElasticSection`` (alpha_y = As2/A, alpha_z = As3/A): the
    copy rescales alpha so the A modifier stays off the shear areas."""
    ops = _bridge()
    sec = ops.section.Elastic(
        E=E, A=A, Iz=IZ, Iy=IY, G=G, J=J, alphaY=AVY / A, alphaZ=AVZ / A,
    )
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=3)
    ops.element.forceBeamColumn(
        pg="Cols", transf=_transf(ops), integration=integ, modifiers=MODS,
    )
    _check_3d(ops, MODS, shear=True)


def test_etabs_beam_axes_along_x() -> None:
    """A beam along +X: ETABS 1 = +X, 2 = +Z, 3 = 1 x 2 = -Y, so
    ``vecxz=(0, -1, 0)``. A force along +Z (ETABS 2) bends about 3 (I33),
    one along +Y (against ETABS 3) about 2 (I22)."""
    fem = FEMStub(
        nodes=_NodesStub(ids=[1, 2], coords=[(0.0, 0.0, 0.0), (L, 0.0, 0.0)],
                         node_pgs={"Base": [1], "Top": [2]}),
        elements=_ElementsStub(elem_pgs={"Beam": _ElementGroupView(
            ids=(1,), connectivity=((1, 2),))}),
    )
    ops = apeSees(cast(Any, fem))
    ops.model(ndm=3, ndf=6)
    ops.element.elasticBeamColumn(
        pg="Beam", transf=ops.geomTransf.Linear(vecxz=(0.0, -1.0, 0.0)),
        A=A, E=E, Iz=IZ, Iy=IY, G=G, J=J, modifiers=NO_SHEAR,
    )
    top = _static(ops, (0.0, P, P, 0.0, 0.0, 0.0), ndf=6)
    assert top.read("displacement_z") == pytest.approx(
        P * L**3 / (3 * E * IZ * NO_SHEAR.Iz), rel=RTOL)
    assert top.read("displacement_y") == pytest.approx(
        P * L**3 / (3 * E * IY * NO_SHEAR.Iy), rel=RTOL)


def test_importer_vecxz_puts_i33_on_the_minor_axis() -> None:
    """``vecxz=(1, 0, 0)``, the ETABS importer's choice for a vertical
    member, is ETABS 2 for a default column (1 = +Z, 2 = +X, 3 = +Y), so
    OpenSees local z = +X = ETABS 2. ``etabs(I33=0.5)`` then doubles the
    tip displacement along ETABS 3 (+Y: bending about ETABS 2, the I22
    axis) and leaves the one along ETABS 2 (+X) alone. The convention the
    ``FrameModifiers.etabs`` docstring warns about."""
    ops = _bridge()
    ops.element.elasticBeamColumn(
        pg="Cols", transf=ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0)),
        A=A, E=E, Iz=IZ, Iy=IY, G=G, J=J,
        modifiers=FrameModifiers.etabs(I33=0.5),
    )
    top = _static(ops, (P, P, 0.0, 0.0, 0.0, 0.0), ndf=6)
    assert top.read("displacement_y") == pytest.approx(
        P * L**3 / (3 * E * IZ * 0.5), rel=RTOL)
    assert top.read("displacement_x") == pytest.approx(
        P * L**3 / (3 * E * IY), rel=RTOL)


def test_gross_section_reference_3d() -> None:
    """The oracle itself, without modifiers (guards a wrong closed form)."""
    ops = _bridge()
    ops.element.elasticBeamColumn(
        pg="Cols", transf=_transf(ops), A=A, E=E, Iz=IZ, Iy=IY, G=G, J=J,
    )
    _check_3d(ops, FrameModifiers(), shear=False)


# ---------------------------------------------------------------------------
# Static: 2-D (column along +Y, bending about local z)
# ---------------------------------------------------------------------------

M2 = FrameModifiers(A=0.8, Avy=0.6, Iz=0.35)


def test_elastic_beam_column_2d() -> None:
    ops = _bridge(ndm=2)
    m = FrameModifiers(A=0.8, Iz=0.35)
    ops.element.elasticBeamColumn(
        pg="Cols", transf=ops.geomTransf.Linear(), A=A, E=E, Iz=IZ,
        modifiers=m,
    )
    top = _static(ops, (P, P, 0.0), ndf=3)
    assert top.read("displacement_x") == pytest.approx(
        P * L**3 / (3 * E * IZ * m.Iz), rel=RTOL)
    assert top.read("displacement_y") == pytest.approx(
        P * L / (E * A * m.A), rel=RTOL)


def test_elastic_timoshenko_beam_2d() -> None:
    ops = _bridge(ndm=2)
    ops.element.ElasticTimoshenkoBeam(
        pg="Cols", transf=ops.geomTransf.Linear(), E=E, G=G, A=A, Iz=IZ,
        Avy=AVY, modifiers=M2,
    )
    top = _static(ops, (P, P, 0.0), ndf=3)
    assert top.read("displacement_x") == pytest.approx(
        P * L**3 / (3 * E * IZ * M2.Iz) + P * L / (G * AVY * M2.Avy),
        rel=RTOL)
    assert top.read("displacement_y") == pytest.approx(
        P * L / (E * A * M2.A), rel=RTOL)


# ---------------------------------------------------------------------------
# Modal: the mass modifier
# ---------------------------------------------------------------------------

def _omega_expected(m: FrameModifiers) -> list[float]:
    tip_mass = M_LEN * m.mass * L / 2.0
    w_axial = (E * A * m.A / L / tip_mass) ** 0.5
    w_bend = (12.0 * E * IZ * m.Iz / L**3 / tip_mass) ** 0.5
    return sorted([w_axial, w_bend])


@pytest.mark.parametrize("family", ["elasticBeamColumn", "forceBeamColumn"])
@pytest.mark.parametrize("mods", [FrameModifiers(), WITH_MASS],
                         ids=["gross", "modified"])
def test_modal_mass_modifier(family: str, mods: FrameModifiers) -> None:
    ops = _bridge()
    t = _transf(ops)
    if family == "elasticBeamColumn":
        ops.element.elasticBeamColumn(
            pg="Cols", transf=t, A=A, E=E, Iz=IZ, Iy=IY, G=G, J=J,
            mass=M_LEN, modifiers=mods,
        )
    else:
        sec = ops.section.Elastic(E=E, A=A, Iz=IZ, Iy=IY, G=G, J=J)
        ops.element.forceBeamColumn(
            pg="Cols", transf=t, mass=M_LEN, modifiers=mods,
            integration=ops.beamIntegration.Lobatto(section=sec, n_ip=3),
        )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    # Tip: u_x and u_z free, u_y and the rotations held.
    ops.fix(pg="Top", dofs=(0, 1, 0, 1, 1, 1))
    result = ops.eigen(num_modes=2, solver="-fullGenLapack")
    got = sorted(float(w) for w in result.omega[:2])
    assert got == pytest.approx(_omega_expected(mods), rel=RTOL)
