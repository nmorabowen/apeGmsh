"""#2460: two ``s.update_parameter`` stage-flag routes that reach nothing.

* ``updateMaterialStage`` through ``updateParameter`` goes through
  ``Parameter::update(double)``, which sets only ``theDouble``, while every
  class that takes the name reads ``Information::theInt`` for it
  (``ManzariDafalias.cpp``, ``SAniSandMS.cpp``, ``PM4Sand.cpp``,
  ``PM4Silt.cpp``, the stress-density models; ``LadrunoSANISAND`` hands id 1
  to its base).  ``theInt`` is never set, so the flip the bridge counted
  depends on uninitialised memory.  Refused by name at stage close.
* ``materialState`` through a ``FourNodeQuad`` host never reaches the
  material: ``FourNodeQuad::setParameter`` reads any ``material...`` name as
  its own ``material <point> ...`` address.  Refused unless every targeted
  element that carries the named material is of a
  :data:`MATERIAL_STATE_ELEMENTS` class.

The unit tests pin both refusals; the live test shows the FourNodeQuad route
leaves the answer elastic while the forwarding host flips it.
"""
from __future__ import annotations

import warnings
from typing import Any

import pytest

from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.stage._builder import (
    CLASS_WIDE_STAGE_FLAG,
    MATERIAL_STATE_CLASSES,
    MATERIAL_STATE_ELEMENTS,
    STAGED_MATERIAL_CLASSES,
    TAG_BLIND_STAGE_PARAMETER,
)

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)

_MD = dict(
    G0=125.0, nu=0.05, e_init=0.8, Mc=1.25, c=0.712, lambda_c=0.019,
    e0=0.934, ksi=0.7, P_atm=100.0, m=0.01, h0=7.05, Ch=0.968, nb=1.1,
    A0=0.704, nd=3.5, z_max=4.0, cz=600.0, rho=1.42,
)
_MS = dict(
    G0=125.0, nu=0.05, e_init=0.8, Mc=1.25, c=0.712, lambda_c=0.019,
    e0=0.934, ksi=0.7, P_atm=100.0, m=0.01, h0=7.05, Ch=0.968, nb=1.1,
    A0=0.704, nd=3.5, zeta=0.0005, mu0=260.0, beta=1.0, rho=1.6,
)
_KW = {"ManzariDafalias": _MD, "LadrunoSANISAND": _MD, "SAniSandMS": _MS}


def _fem() -> FEMStub:
    """A 6-node triangle ``T`` (eid 1), and two DISCONNECTED unit quads,
    ``Q`` (eid 2) and ``R`` (eid 3)."""
    tri = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.0), (0.5, 0.5),
           (0.0, 0.5)]
    quad = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    coords = [(x, y, 0.0) for x, y in tri]
    coords += [(x + dx, y, 0.0) for dx in (3.0, 5.0) for x, y in quad]
    return FEMStub(
        nodes=_NodesStub(
            ids=list(range(1, 15)), coords=coords,
            node_pgs={"T": [1, 2, 3, 4, 5, 6], "Q": [7, 8, 9, 10],
                      "R": [11, 12, 13, 14],
                      "Base": [1, 2, 4, 7, 8, 11, 12]},
        ),
        elements=_ElementsStub(
            elem_pgs={
                "T": _ElementGroupView(
                    ids=(1,), connectivity=((1, 2, 3, 4, 5, 6),)),
                "Q": _ElementGroupView(
                    ids=(2,), connectivity=((7, 8, 9, 10),)),
                "R": _ElementGroupView(
                    ids=(3,), connectivity=((11, 12, 13, 14),)),
            },
        ),
    )


def _chain(ops: apeSees) -> dict[str, object]:
    return {
        "test": ops.test.NormUnbalance(tol=1e-6, max_iter=50),
        "algorithm": ops.algorithm.Newton(),
        "integrator": ops.integrator.LoadControl(dlam=0.1),
        "constraints": ops.constraints.Plain(),
        "numberer": ops.numberer.RCM(),
        "system": ops.system.UmfPack(),
        "analysis": ops.analysis.Static(),
    }


def _bridge(cls: str, pg: str) -> tuple[apeSees, Any]:
    """One material of ``cls`` on one element: SixNodeTri on ``T``,
    FourNodeQuad on ``Q``."""
    ops = apeSees(_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mat = getattr(ops.nDMaterial, cls)(name="sand", **_KW[cls])
    if pg == "T":
        ops.element.SixNodeTri(pg="T", thickness=1.0, material=mat)
    else:
        ops.element.FourNodeQuad(pg=pg, thickness=1.0, material=mat)
    ops.fix(pg="Base", dofs=(1, 1))
    return ops, mat


def _flip(ops: apeSees, name: str, value: float, *, material: Any,
          pg: "str | None" = None,
          elements: "list[int] | None" = None) -> None:
    with ops.stage(name="push") as s:
        s.update_parameter(name, value, pg=pg, elements=elements,
                           material=material)
        s.analysis(**_chain(ops))
        s.run(n_increments=1)


_STAGED = sorted(_KW)


def test_every_staged_class_is_covered() -> None:
    """A class that joins STAGED_MATERIAL_CLASSES (PM4Sand and PM4Silt with
    #2401) joins these refusal tests with its constructor keywords in
    ``_KW``."""
    assert set(_KW) == STAGED_MATERIAL_CLASSES


# ---------------------------------------------------------------------------
# updateMaterialStage through updateParameter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", _STAGED)
@pytest.mark.parametrize("value", [0.0, 1.0])
def test_the_updatematerialstage_parameter_refuses_by_name(
    cls: str, value: float,
) -> None:
    """On a host that forwards it, on one element: no other rule fires."""
    ops, mat = _bridge(cls, "T")
    with pytest.raises(BridgeError) as exc:
        _flip(ops, "updateMaterialStage", value, material=mat, pg="T")
    msg = str(exc.value)
    assert "s.update_parameter('updateMaterialStage'" in msg
    assert "theInt" in msg and "#2460" in msg
    assert not ops._stage_records


def test_the_updatematerialstage_parameter_refuses_without_a_material() -> None:
    ops, _mat = _bridge("ManzariDafalias", "T")
    with pytest.raises(BridgeError, match="#2460"):
        _flip(ops, "updateMaterialStage", 1.0, material=None, pg="T")


@pytest.mark.parametrize("cls", _STAGED)
def test_the_materialstate_parameter_passes_on_a_forwarding_host(
    cls: str,
) -> None:
    ops, mat = _bridge(cls, "T")
    _flip(ops, "materialState", 1.0, material=mat, pg="T")
    assert len(ops._stage_records) == 1


def test_the_materialstate_classes_table() -> None:
    """Read from the nD sources: every class with a materialState branch."""
    assert MATERIAL_STATE_CLASSES == {
        "ManzariDafalias", "LadrunoSANISAND", "SAniSandMS", "PM4Sand",
        "PM4Silt", "J2CyclicBoundingSurface", "DruckerPrager",
        "BoundingCamClay", "StressDensityModel", "stressDensity",
    }
    # Every class-wide group answers materialState (the stage-close count
    # relies on it); the tag-blind classes take it too.
    assert set(CLASS_WIDE_STAGE_FLAG) <= MATERIAL_STATE_CLASSES
    assert {c for c, n in TAG_BLIND_STAGE_PARAMETER.items()
            if "materialState" in n} <= MATERIAL_STATE_CLASSES


def test_materialstate_on_a_class_without_it_refuses_by_name() -> None:
    """ElasticIsotropic has no materialState branch: nothing takes it."""
    ops, _mat = _bridge("ManzariDafalias", "T")
    rock = ops.nDMaterial.ElasticIsotropic(name="rock", E=1e5, nu=0.3)
    with pytest.raises(BridgeError) as exc:
        with ops.stage(name="push") as s:
            s.update_parameter("materialState", 1.0, pg="T", material=rock)
    msg = str(exc.value)
    assert "ElasticIsotropic's setParameter has no materialState" in msg
    assert "#2460" in msg and "no stage flag" in msg
    assert "s.update_material_stage" not in msg


def test_the_updatematerialstage_refusal_steers_to_materialstate_only_where_it_acts(
) -> None:
    ops, mat = _bridge("ManzariDafalias", "T")
    with pytest.raises(BridgeError) as exc:
        _flip(ops, "updateMaterialStage", 1.0, material=mat, pg="T")
    assert "s.update_parameter('materialState'" in str(exc.value)


# ---------------------------------------------------------------------------
# materialState through a host that intercepts it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", _STAGED)
@pytest.mark.parametrize("value", [0.0, 1.0])
def test_materialstate_on_four_node_quad_refuses_by_name(
    cls: str, value: float,
) -> None:
    ops, mat = _bridge(cls, "Q")
    with pytest.raises(BridgeError) as exc:
        _flip(ops, "materialState", value, material=mat, pg="Q")
    msg = str(exc.value)
    assert "targets FourNodeQuad elements" in msg and "#2460" in msg
    assert "#2457" not in msg
    for host in sorted(MATERIAL_STATE_ELEMENTS):
        assert host in msg
    assert not ops._stage_records


def test_materialstate_on_a_ladruno_quad_cites_the_pin_remeasure() -> None:
    ops = apeSees(_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mat = ops.nDMaterial.ManzariDafalias(name="sand", **_MD)
    ops.element.LadrunoQuad(pg="Q", material=mat, thickness=1.0)
    ops.fix(pg="Base", dofs=(1, 1))
    with pytest.raises(BridgeError) as exc:
        _flip(ops, "materialState", 1.0, material=mat, pg="Q")
    msg = str(exc.value)
    assert "targets LadrunoQuad elements" in msg
    assert "79e062367" in msg and "#2457" in msg


def test_a_four_node_quad_that_does_not_carry_the_material_passes() -> None:
    """The quad of another material is offered the update and drops it;
    nothing that should flip is lost."""
    ops, mat = _bridge("ManzariDafalias", "T")
    other = ops.nDMaterial.ElasticIsotropic(name="rock", E=1e5, nu=0.3)
    ops.element.FourNodeQuad(pg="R", thickness=1.0, material=other)
    _flip(ops, "materialState", 1.0, material=mat, elements=[1, 3])
    assert len(ops._stage_records) == 1


def test_a_non_stage_parameter_on_four_node_quad_is_not_checked() -> None:
    ops, mat = _bridge("ManzariDafalias", "Q")
    _flip(ops, "poissonRatio", 0.3, material=mat, pg="Q")
    assert len(ops._stage_records) == 1


# ---------------------------------------------------------------------------
# Live: FourNodeQuad drops materialState, SSPquad hands it on
# ---------------------------------------------------------------------------

_ARGS = {
    "ManzariDafalias": [125.0, 0.05, 0.8, 1.25, 0.712, 0.019, 0.934, 0.7,
                        100.0, 0.01, 7.05, 0.968, 1.1, 0.704, 3.5, 4.0,
                        600.0, 1.42],
    "SAniSandMS": [125.0, 0.05, 0.8, 1.25, 0.712, 0.019, 0.934, 0.7, 100.0,
                   0.01, 7.05, 0.968, 1.1, 0.704, 3.5, 0.0005, 260.0, 1.0,
                   1.6],
}


def _run(cls: str, element: str, route: str, dev: float) -> "tuple[int, float]":
    """One unit quad (``quad`` or ``SSPquad``) of ``cls`` (tag 1): stage 0,
    hydrostatic 100, then ``route`` ("none", "command" =
    ``updateMaterialStage -stage 1``, or "materialState" through
    ``parameter``/``addToParameter``/``updateParameter``), then +dev on
    sigma_y.  Returns (analyze code, top-corner uy)."""
    from apeGmsh.opensees.emitter.live import get_ops

    ops = get_ops()
    ops.wipe()
    try:
        ops.model("basic", "-ndm", 2, "-ndf", 2)
        for n, (x, y) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1)), 1):
            ops.node(n, float(x), float(y))
            ops.fix(n, int(x == 0), int(y == 0))
        ops.nDMaterial(cls, 1, *_ARGS[cls])
        if element == "SSPquad":
            ops.element("SSPquad", 1, 1, 2, 3, 4, 1, "PlaneStrain", 1.0)
        else:
            ops.element("quad", 1, 1, 2, 3, 4, 1.0, "PlaneStrain", 1)
        # The class-wide static survives wipe(): start every run elastic.
        ops.updateMaterialStage("-material", 1, "-stage", 0)

        def analyze() -> int:
            ops.constraints("Plain")
            ops.numberer("Plain")
            ops.system("FullGeneral")
            ops.test("NormDispIncr", 1e-10, 100)
            ops.algorithm("Newton")
            ops.integrator("LoadControl", 0.1)
            ops.analysis("Static")
            code = int(ops.analyze(10))
            ops.wipeAnalysis()
            return code

        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        for n in (2, 3):
            ops.load(n, -50.0, 0.0)
        for n in (3, 4):
            ops.load(n, 0.0, -50.0)
        code = analyze()
        ops.loadConst("-time", 0.0)
        if route == "command":
            ops.updateMaterialStage("-material", 1, "-stage", 1)
        elif route == "materialState":
            ops.parameter(1)
            ops.addToParameter(1, "element", 1, "materialState", 1)
            ops.updateParameter(1, 1.0)
            ops.remove("parameter", 1)
        ops.timeSeries("Linear", 2)
        ops.pattern("Plain", 2, 2)
        for n in (3, 4):
            ops.load(n, 0.0, -dev / 2.0)
        code = code or analyze()
        return code, float(ops.nodeDisp(3, 2))
    finally:
        ops.wipe()


@pytest.mark.live
@pytest.mark.parametrize(("cls", "dev"), [
    ("ManzariDafalias", 40.0), ("SAniSandMS", 8.0)])
def test_live_four_node_quad_drops_materialstate(cls: str, dev: float) -> None:
    """Stock and fork: through ``quad`` the materialState route answers as
    if no flip happened; through ``SSPquad`` it answers as the
    ``updateMaterialStage`` command does.  The bridge refuses the first."""
    ops, mat = _bridge(cls, "Q")
    with pytest.raises(BridgeError, match="targets FourNodeQuad"):
        _flip(ops, "materialState", 1.0, material=mat, pg="Q")
    runs = {
        (element, route): _run(cls, element, route, dev)
        for element in ("quad", "SSPquad")
        for route in ("none", "command", "materialState")
    }
    assert {code for code, _uy in runs.values()} == {0}
    uy = {key: value for key, (_code, value) in runs.items()}
    for element in ("quad", "SSPquad"):
        assert abs(uy[element, "command"] - uy[element, "none"]) > 1e-2 * abs(
            uy[element, "none"])
    assert uy["quad", "materialState"] == pytest.approx(
        uy["quad", "none"], rel=1e-12)
    assert uy["SSPquad", "materialState"] == pytest.approx(
        uy["SSPquad", "command"], rel=1e-12)
