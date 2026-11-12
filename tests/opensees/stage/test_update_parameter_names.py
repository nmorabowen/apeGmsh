"""#2377: ``s.update_parameter`` refuses a name no targeted object takes.

OpenSees answers an ``addToParameter`` whose name no object identifies
with a printed ``Parameter::addComponent <pid> -- no objects were able to
identify parameter`` and carries on, so the update is silently lost.  The
check reads ``stage/_parameter_names.py`` (each entry cited to its C++
``setParameter``), under the owner ruling (b) + (c) of 2026-11-11:

* a catalogued element class that neither takes the name nor forwards it
  is refused;
* a forwarding class is refused only when every material it carries is
  catalogued and none takes the name;
* an uncatalogued class warns once per class and the update emits.

The oracle is the C++ bodies: ``FourNodeQuad::setParameter`` takes
``pressure`` and forwards the rest to its materials, and
``ElasticIsotropicMaterial::setParameter`` takes ``E`` / ``nu`` / ``v`` /
``rho``, so ``xPerm`` (a ``LadrunoUP`` name) reaches nothing.
"""
from __future__ import annotations

import contextlib
import warnings
from typing import Iterator

import pytest

from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.stage._parameter_names import (
    ELEMENT_PARAMETER_NAMES,
    ELEMENT_PARAMETER_NAMES_BY_NDM,
    MATERIAL_PARAMETER_NAMES,
    UncataloguedParameterWarning,
)

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


def _fem() -> FEMStub:
    """Two quads in ``soil`` (eids 1, 2), two bars in ``bars`` (3, 4)."""
    return FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4, 5, 6],
            coords=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0),
                    (0.0, 1.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0)],
            node_pgs={"Left": [1, 4]},
        ),
        elements=_ElementsStub(elem_pgs={
            "soil": _ElementGroupView(
                ids=(1, 2), connectivity=((1, 2, 3, 4), (2, 5, 6, 3))),
            "bars": _ElementGroupView(
                ids=(3, 4), connectivity=((1, 5), (4, 6))),
        }),
    )


def _close(ops: apeSees, s) -> None:
    s.analysis(
        test=ops.test.NormUnbalance(tol=1e-4, max_iter=50),
        algorithm=ops.algorithm.Newton(),
        integrator=ops.integrator.LoadControl(dlam=0.1),
        constraints=ops.constraints.Plain(),
        numberer=ops.numberer.RCM(),
        system=ops.system.UmfPack(),
        analysis=ops.analysis.Static(),
    )
    s.run(n_increments=1)


def _quad(ndf: int = 2) -> tuple[apeSees, object]:
    ops = apeSees(_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=ndf)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="soil", thickness=1.0, material=mat)
    return ops, mat


@contextlib.contextmanager
def _silent() -> Iterator[None]:
    """An uncatalogued warning inside raises: the catalogued cases are
    decided, not warned about."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UncataloguedParameterWarning)
        yield


# ---------------------------------------------------------------------------
# The reproducer: xPerm on FourNodeQuad over ElasticIsotropic
# ---------------------------------------------------------------------------


def test_xperm_on_fournodequad_over_elastic_isotropic_is_refused_by_name(
) -> None:
    ops, _mat = _quad()
    with ops.stage(name="perm") as s:
        with pytest.raises(BridgeError) as err:
            s.update_parameter("xPerm", 1e-5, pg="soil")
        _close(ops, s)
    msg = str(err.value)
    assert "'xPerm'" in msg
    assert "FourNodeQuad takes pressure" in msg
    assert "ElasticIsotropic takes E, nu, rho, v" in msg
    assert "FourNodeQuad.cpp::FourNodeQuad::setParameter" in msg
    # Refused before the stage recorded anything on the bridge.
    assert ops._stage_records[0].update_parameter_records == ()
    lines = TclEmitter()
    ops.build().emit(lines)
    assert not any("xPerm" in ln for ln in lines.lines())


def test_forwarded_material_names_still_emit() -> None:
    """``E`` / ``nu`` reach ElasticIsotropic through the quad's forwarding,
    with or without the material tag (the class does not read it)."""
    ops, mat = _quad()
    with _silent(), ops.stage(name="soften") as s:
        s.update_parameter("E", 2e6, pg="soil")
        s.update_parameter("nu", 0.25, pg="soil", material=mat)
        s.update_parameter("pressure", 1.0, elements=[2])
        _close(ops, s)
    names = [r.name for r in ops._stage_records[0].update_parameter_records]
    assert names == ["E", "nu", "pressure"]


# ---------------------------------------------------------------------------
# Truss and elastic beam
# ---------------------------------------------------------------------------


def _truss() -> apeSees:
    ops = apeSees(_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    steel = ops.uniaxialMaterial.ElasticMaterial(E=2e5)
    ops.element.Truss(pg="bars", A=0.01, material=steel)
    return ops


def test_truss_takes_a_and_forwards_e_to_its_material() -> None:
    ops = _truss()
    with _silent(), ops.stage(name="s") as s:
        s.update_parameter("A", 0.02, pg="bars")
        s.update_parameter("E", 2e6, elements=[4])
        _close(ops, s)
    assert len(ops._stage_records[0].update_parameter_records) == 2


def test_truss_refuses_a_name_neither_it_nor_elastic_material_takes() -> None:
    ops = _truss()
    with ops.stage(name="s") as s:
        with pytest.raises(BridgeError, match=r"Truss takes A, rho.*"
                           r"ElasticMaterial takes E, Eneg, Epos, eta"):
            s.update_parameter("nu", 0.3, pg="bars")
        _close(ops, s)


def _beam() -> apeSees:
    ops = apeSees(_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=3)
    transf = ops.geomTransf.Linear()
    ops.element.elasticBeamColumn(
        pg="bars", transf=transf, A=0.01, E=2e11, Iz=1e-4)
    return ops


def test_elastic_beam_takes_e() -> None:
    ops = _beam()
    with _silent(), ops.stage(name="s") as s:
        s.update_parameter("E", 2.1e11, pg="bars")
        _close(ops, s)
    assert len(ops._stage_records[0].update_parameter_records) == 1


def test_elastic_beam_forwards_nothing() -> None:
    ops = _beam()
    with ops.stage(name="s") as s:
        with pytest.raises(BridgeError,
                           match="forwards nothing to a material"):
            s.update_parameter("Epos", 1.0, pg="bars")
        _close(ops, s)


def _beam3d() -> apeSees:
    ops = apeSees(_fem(), default_orientation=None)
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    ops.element.elasticBeamColumn(
        pg="bars", transf=transf, A=0.01, E=2e11, Iz=1e-4, Iy=2e-4,
        G=8e10, J=3e-4)
    return ops


def test_2d_beam_refuses_iz() -> None:
    """ElasticBeam2d takes ``I``, not ``Iz`` (its setParameter): the
    union of the 2-D and 3-D names let ``Iz`` emit on a 2-D beam."""
    ops = _beam()
    with ops.stage(name="s") as s:
        with pytest.raises(BridgeError,
                           match=r"elasticBeamColumn takes A, E, I, release, "
                                 r"rho .*ElasticBeam2d"):
            s.update_parameter("Iz", 2e-4, pg="bars")
        s.update_parameter("I", 2e-4, pg="bars")
        _close(ops, s)


def test_3d_beam_refuses_rho_and_takes_iz() -> None:
    """ElasticBeam3d takes ``Iz`` / ``Iy`` / ``G`` / ``J``, and no ``rho``."""
    ops = _beam3d()
    with ops.stage(name="s") as s:
        with pytest.raises(BridgeError, match=r"ElasticBeam3d"):
            s.update_parameter("rho", 1.0, pg="bars")
        with pytest.raises(BridgeError, match=r"ElasticBeam3d"):
            s.update_parameter("I", 2e-4, pg="bars")
        with _silent():
            s.update_parameter("Iz", 2e-4, pg="bars")
        _close(ops, s)
    deck = TclEmitter()
    ops.build().emit(deck)
    lines = deck.lines()
    assert sum(" Iz" in ln and "addToParameter" in ln for ln in lines) == 2
    assert not any(" rho" in ln and "addToParameter" in ln for ln in lines)


# ---------------------------------------------------------------------------
# Seams
# ---------------------------------------------------------------------------


def test_two_classes_are_checked_each_and_one_refusal_refuses() -> None:
    """elements= spanning a quad (eid 1) and a bar (eid 3): ``A`` is a
    Truss name the quad's ElasticIsotropic does not take."""
    ops = apeSees(_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    soil = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="soil", thickness=1.0, material=soil)
    steel = ops.uniaxialMaterial.ElasticMaterial(E=2e5)
    ops.element.Truss(pg="bars", A=0.01, material=steel)
    with ops.stage(name="s") as s:
        with pytest.raises(BridgeError, match="FourNodeQuad takes") as err:
            s.update_parameter("A", 0.02, elements=[1, 3])
        assert "Truss takes" not in str(err.value)
        s.update_parameter("E", 2e6, elements=[1, 3])  # both take E
        _close(ops, s)


def test_an_element_registered_after_the_call_is_refused_at_close() -> None:
    ops = apeSees(_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    with pytest.raises(BridgeError, match="'xPerm'"):
        with ops.stage(name="late") as s:
            s.update_parameter("xPerm", 1e-5, pg="soil")
            ops.element.FourNodeQuad(pg="soil", thickness=1.0, material=mat)
            _close(ops, s)
    assert ops._stage_records == []


def test_tag_checked_material_needs_its_tag() -> None:
    """PM4Sand matches only when argv[1] is its own tag
    (``material/nd_soil.SET_PARAMETER_NAMES``): without material= the
    update reaches nothing."""
    ops = apeSees(_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    sand = ops.nDMaterial.PM4Sand(Dr=0.55, G0=677.0, hpo=0.4, rho=1.7)
    ops.element.FourNodeQuad(pg="soil", thickness=1.0, material=sand)
    with ops.stage(name="s") as s:
        with pytest.raises(BridgeError, match="only with its own tag"):
            s.update_parameter("PostShake", 1.0, pg="soil")
        s.update_parameter("PostShake", 1.0, pg="soil", material=sand)
        _close(ops, s)


def test_an_unregistered_element_id_is_left_to_the_emit() -> None:
    """elements= naming no registered element is not decided here: the
    emit refuses it, as before."""
    ops, _mat = _quad()
    with ops.stage(name="s") as s:
        s.update_parameter("xPerm", 1.0, elements=[999])
        _close(ops, s)
    with pytest.raises(BridgeError, match="not registered with any Element"):
        ops.build().emit(TclEmitter())


# ---------------------------------------------------------------------------
# Uncatalogued: warn once per class, emit
# ---------------------------------------------------------------------------


def test_uncatalogued_element_warns_once_and_emits() -> None:
    ops = apeSees(_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.LadrunoQuad(pg="soil", thickness=1.0, material=mat)
    assert "LadrunoQuad" not in ELEMENT_PARAMETER_NAMES
    with pytest.warns(UncataloguedParameterWarning,
                      match="LadrunoQuad is not in") as rec:
        with ops.stage(name="s") as s:
            s.update_parameter("xPerm", 1.0, pg="soil")
            s.update_parameter("yPerm", 1.0, pg="soil")
            _close(ops, s)
    assert len([w for w in rec
                if issubclass(w.category, UncataloguedParameterWarning)]) == 1
    deck = TclEmitter()
    ops.build().emit(deck)
    assert sum("xPerm" in ln for ln in deck.lines()) == 2


def test_a_warning_from_the_stage_close_points_at_the_user() -> None:
    """An element registered after the call is warned about at the stage
    close; the warning names this file (the ``with``), not the builder."""
    ops = apeSees(_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    with pytest.warns(UncataloguedParameterWarning,
                      match="LadrunoQuad") as rec:
        with ops.stage(name="late") as s:
            s.update_parameter("xPerm", 1.0, pg="soil")
            ops.element.LadrunoQuad(pg="soil", thickness=1.0, material=mat)
            _close(ops, s)
    (w,) = [w for w in rec
            if issubclass(w.category, UncataloguedParameterWarning)]
    assert w.filename == __file__


def test_a_warning_from_the_call_points_at_the_user() -> None:
    ops = apeSees(_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.LadrunoQuad(pg="soil", thickness=1.0, material=mat)
    with pytest.warns(UncataloguedParameterWarning) as rec:
        with ops.stage(name="s") as s:
            s.update_parameter("xPerm", 1.0, pg="soil")
            _close(ops, s)
    (w,) = [w for w in rec
            if issubclass(w.category, UncataloguedParameterWarning)]
    assert w.filename == __file__


def test_uncatalogued_material_behind_a_forwarding_element_warns() -> None:
    ops = apeSees(_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.J2Plasticity(
        K=1e6, G=5e5, sig0=100.0, sigInf=200.0, delta=1.0, H=0.0)
    ops.element.FourNodeQuad(pg="soil", thickness=1.0, material=mat)
    assert "J2Plasticity" not in MATERIAL_PARAMETER_NAMES
    with pytest.warns(UncataloguedParameterWarning, match="J2Plasticity"):
        with ops.stage(name="s") as s:
            s.update_parameter("K", 2e6, pg="soil")
            _close(ops, s)
    assert len(ops._stage_records[0].update_parameter_records) == 1


# ---------------------------------------------------------------------------
# The catalogue cites its source
# ---------------------------------------------------------------------------


def test_every_entry_cites_a_setparameter() -> None:
    for entry in (*ELEMENT_PARAMETER_NAMES.values(),
                  *(e for by_ndm in ELEMENT_PARAMETER_NAMES_BY_NDM.values()
                    for e in by_ndm.values()),
                  *MATERIAL_PARAMETER_NAMES.values()):
        assert "::setParameter" in entry.source
