"""#2458: a stage flip must not reach a material that ignores the tag it names.

``DruckerPrager``, ``BoundingCamClay``, ``StressDensityModel`` and
``stressDensity`` take ``materialState`` (the last two also
``updateMaterialStage``) in ``setParameter`` without comparing ``argv[1]``
with their own tag.  So ``s.update_parameter("materialState", v,
material=sand)`` over elements that also carry one of them switches it too.
The unit tests pin the stage-close refusal (``TAG_BLIND_STAGE_PARAMETER`` in
``stage/_builder.py``) and that the bridge's own ``-ele`` sweep (#2459) never
lists such an element.  The live test runs the refused selection on OpenSees
and shows it is exactly the one that switches the neighbour.
"""
from __future__ import annotations

import warnings
from typing import Any

import pytest

from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees._internal.build.stage_emit import (
    MATERIAL_STATE_WRAPPERS,
    SWEPT_MATERIAL_CLASSES,
)
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.stage._builder import (
    CLASS_WIDE_STAGE_FLAG,
    STAGE_FLAG_PARAMETERS,
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
_DP = dict(K=1e5, G=5e4, sigmaY=10.0, rho=0.0, rhoBar=0.0, Kinf=10.0,
           Ko=10.0, delta1=0.0, delta2=0.0, H=1000.0, theta=1.0)

_TRI6 = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.0), (0.5, 0.5),
         (0.0, 0.5)]


def _fem() -> FEMStub:
    """Three DISCONNECTED 6-node triangles: ``A`` (eid 1), ``B`` (eid 2)
    and ``C`` (eid 3)."""
    coords = [(x + dx, y, 0.0) for dx in (0.0, 3.0, 6.0) for x, y in _TRI6]
    pgs = {}
    for k, pg in enumerate("ABC"):
        first = 6 * k + 1
        pgs[pg] = _ElementGroupView(
            ids=(k + 1,), connectivity=(tuple(range(first, first + 6)),))
    return FEMStub(
        nodes=_NodesStub(
            ids=list(range(1, 19)), coords=coords,
            node_pgs={**{pg: list(v.connectivity[0]) for pg, v in pgs.items()},
                      "Base": [1, 2, 4, 7, 8, 10, 13, 14, 16]},
        ),
        elements=_ElementsStub(elem_pgs=pgs),
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


def _bridge(sand_pgs: "tuple[str, ...]" = ("A",)) -> tuple[apeSees, Any, Any]:
    """The #2446 reviewer's model: a sand on SixNodeTri ``sand_pgs``, a
    DruckerPrager clay on SixNodeTri ``C``."""
    ops = apeSees(_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sand = ops.nDMaterial.ManzariDafalias(name="sand", **_MD)
    clay = ops.nDMaterial.DruckerPrager(name="clay", **_DP)
    for pg in sand_pgs:
        ops.element.SixNodeTri(pg=pg, thickness=1.0, material=sand)
    ops.element.SixNodeTri(pg="C", thickness=1.0, material=clay)
    ops.fix(pg="Base", dofs=(1, 1))
    return ops, sand, clay


def _flip(ops: apeSees, *, elements: "list[int]", material: Any,
          name: str = "materialState") -> None:
    with ops.stage(name="push") as s:
        s.update_parameter(name, 1.0, elements=elements, material=material)
        s.analysis(**_chain(ops))
        s.run(n_increments=1)


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def test_table_lists_the_tag_blind_classes() -> None:
    assert TAG_BLIND_STAGE_PARAMETER == {
        "DruckerPrager": {"materialState"},
        "BoundingCamClay": {"materialState"},
        "StressDensityModel": {"materialState", "updateMaterialStage"},
        "stressDensity": {"materialState", "updateMaterialStage"},
    }
    for names in TAG_BLIND_STAGE_PARAMETER.values():
        assert names <= STAGE_FLAG_PARAMETERS
    # None of them is a staged class the bridge flips on purpose.
    assert not set(TAG_BLIND_STAGE_PARAMETER) & (
        STAGED_MATERIAL_CLASSES | set(CLASS_WIDE_STAGE_FLAG))


def test_the_sweep_cannot_reach_a_tag_blind_class() -> None:
    """#2459's sweep lists an element only when its one material resolves,
    through MATERIAL_STATE_WRAPPERS alone, to a SWEPT_MATERIAL_CLASSES tag;
    no tag-blind class is either, so no carrier can hold one."""
    assert not set(TAG_BLIND_STAGE_PARAMETER) & (
        set(SWEPT_MATERIAL_CLASSES) | set(MATERIAL_STATE_WRAPPERS))


def test_the_sweep_skips_a_drucker_prager_neighbour() -> None:
    """The bridge's own flip of a two-element sand beside the clay lists
    the sand's elements only, and the stage close passes."""
    ops, sand, _clay = _bridge(("A", "B"))
    with ops.stage(name="push") as s:
        s.update_material_stage(materials=[sand], stage=1)
        s.analysis(**_chain(ops))
        s.run(n_increments=1)
    tcl = TclEmitter()
    ops.build().emit(tcl)
    lines = tcl.lines()
    tri = {ln.split()[2]: ln.split()[-1] for ln in lines
           if ln.startswith("element tri6n")}
    sweep = [ln for ln in lines if ln.startswith("setParameter")]
    assert len(sweep) == 1
    listed = sweep[0].split("-ele ")[1].split(" materialState")[0].split()
    sand_tag = str(ops.tag_for(sand))
    assert sorted(listed) == sorted(e for e, m in tri.items() if m == sand_tag)
    assert len(listed) == 2 and len(tri) == 3


# ---------------------------------------------------------------------------
# The stage-close refusal
# ---------------------------------------------------------------------------


def test_a_flip_over_the_neighbour_refuses_by_name() -> None:
    ops, sand, clay = _bridge()
    with pytest.raises(BridgeError) as exc:
        _flip(ops, elements=[1, 3], material=sand)
    msg = str(exc.value)
    assert f"DruckerPrager (nDMaterial tag {ops.tag_for(clay)})" in msg
    assert f"names nDMaterial tag {ops.tag_for(sand)}" in msg
    assert "#2458" in msg
    assert not ops._stage_records


def test_a_flip_of_the_carriers_only_passes() -> None:
    ops, sand, _clay = _bridge()
    _flip(ops, elements=[1], material=sand)
    assert len(ops._stage_records) == 1


def test_a_flip_naming_the_tag_blind_material_itself_passes() -> None:
    ops, _sand, clay = _bridge()
    _flip(ops, elements=[3], material=clay)
    assert len(ops._stage_records) == 1


def test_a_flip_naming_no_material_refuses() -> None:
    ops, _sand, clay = _bridge()
    with pytest.raises(BridgeError) as exc:
        _flip(ops, elements=[3], material=None)
    msg = str(exc.value)
    assert "names no material" in msg
    assert f"DruckerPrager (nDMaterial tag {ops.tag_for(clay)})" in msg


def test_a_non_stage_parameter_is_not_checked() -> None:
    ops, sand, _clay = _bridge()
    _flip(ops, elements=[1, 3], material=sand, name="poissonRatio")
    assert len(ops._stage_records) == 1


def test_update_material_stage_beside_a_tag_blind_updatematerialstage_class(
) -> None:
    """``MaterialStageParameter::setDomain`` stops at the first element that
    takes ``updateMaterialStage``; a stress-density element takes it
    whatever the tag.  No bridge primitive carries that class yet, so a
    stand-in of that class name pins the refusal for the one that will."""
    from apeGmsh.opensees.material.nd import ElasticIsotropic

    stand_in = type("StressDensityModel", (ElasticIsotropic,), {})
    ops, sand, _clay = _bridge()
    sdm = ops.register(stand_in(E=1e5, nu=0.3))
    ops.element.SixNodeTri(pg="B", thickness=1.0, material=sdm)
    with pytest.raises(BridgeError) as exc:
        with ops.stage(name="push") as s:
            s.update_material_stage(materials=[sand], stage=1)
            s.analysis(**_chain(ops))
            s.run(n_increments=1)
    msg = str(exc.value)
    assert "SixNodeTri on pg 'B' (StressDensityModel)" in msg
    assert "#2458" in msg


# ---------------------------------------------------------------------------
# Live: the refused selection is the one that switches the neighbour
# ---------------------------------------------------------------------------

#: ``(class, args, shear amplitude)`` of the tag-blind classes that run on
#: openseespy from a zero stress.  ``BoundingCamClay`` cannot (NaN on the
#: fork, an access violation on stock 3.8.0) and ``stressDensity`` is
#: registered for Tcl only, so those two rest on their ``setParameter``
#: source (``TAG_BLIND_STAGE_PARAMETER``).
_NEIGHBOURS = [
    ("DruckerPrager",
     [1e5, 5e4, 10.0, 0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 1000.0, 1.0], 1.0),
    ("StressDensityModel",
     [2.0, 0.65, 600.0, 0.6, 0.3, 0.58, 0.023, 230.0, 65.0, 79.0, 16.0,
      4.0, 0.22, 0.0, 0.0055, 0.607, 98.1], 0.05),
]
_MD_ARGS = [125.0, 0.05, 0.8, 1.25, 0.712, 0.019, 0.934, 0.7, 100.0, 0.01,
            7.05, 0.968, 1.1, 0.704, 3.5, 4.0, 600.0, 1.42]


def _run_neighbour(cls: str, args: "list[float]", amp: float, *,
                   named: "int | None") -> "tuple[int, float]":
    """One tri6n of ``cls`` (tag 2) in simple shear to t=1; then, when
    ``named`` is given, ``materialState 1`` on that element naming tag
    ``named``; then on to t=2.  Returns (analyze code, the third component
    of its first ``stresses`` row)."""
    from apeGmsh.opensees.emitter.live import get_ops

    ops = get_ops()
    ops.wipe()
    try:
        ops.model("basic", "-ndm", 2, "-ndf", 2)
        ops.nDMaterial("ManzariDafalias", 1, *_MD_ARGS)  # the "sand" tag
        ops.nDMaterial(cls, 2, *args)
        ops.timeSeries("Path", 1, "-time", 0.0, 1.0, 2.0,
                       "-values", 0.0, 1.0, 2.0)
        ops.pattern("Plain", 1, 1)
        for n, (x, y) in enumerate(_TRI6, 1):
            ops.node(n, x, y)
            ops.sp(n, 1, amp * (1e-2 * y - 2e-3 * x))
            ops.sp(n, 2, amp * (-2e-3 * y))
        ops.element("tri6n", 1, 1, 2, 3, 4, 5, 6, 1.0, "PlaneStrain", 2)
        ops.constraints("Penalty", 1.0e12, 1.0e12)
        ops.numberer("Plain")
        ops.system("FullGeneral")
        ops.test("NormUnbalance", 1e-8, 20, 0)
        ops.algorithm("Linear")
        ops.integrator("LoadControl", 0.05)
        ops.analysis("Static")
        code = int(ops.analyze(20))
        if named is not None:
            # What s.update_parameter("materialState", 1.0, material=<tag
            # named>) emits for an element of its targets.
            ops.parameter(1)
            ops.addToParameter(1, "element", 1, "materialState", named)
            ops.updateParameter(1, 1.0)
            ops.remove("parameter", 1)
        code = code or int(ops.analyze(20))
        return code, float(ops.eleResponse(1, "stresses")[2])
    finally:
        ops.wipe()


@pytest.mark.live
@pytest.mark.parametrize(("cls", "args", "amp"), _NEIGHBOURS,
                         ids=[c for c, _a, _amp in _NEIGHBOURS])
def test_live_a_flip_naming_another_tag_switches_the_neighbour(
    cls: str, args: "list[float]", amp: float,
) -> None:
    """Stock and fork: ``materialState`` naming tag 1 switches the tag-2
    neighbour, and the bridge refuses exactly that selection."""
    ops, sand, _clay = _bridge()
    with pytest.raises(BridgeError, match="#2458"):
        _flip(ops, elements=[1, 3], material=sand)
    code_n, tau_n = _run_neighbour(cls, args, amp, named=None)
    code_s, tau_s = _run_neighbour(cls, args, amp, named=1)
    assert (code_n, code_s) == (0, 0)
    assert abs(tau_s - tau_n) > 1e-2 * abs(tau_n)
    if cls == "DruckerPrager":
        # mElastFlag=1 is elastic: the plastic leg's capped stress
        # (-18.3) becomes an elastic one (-1115.4) on both builds.
        assert abs(tau_s) > 10.0 * abs(tau_n)
