"""#2439, #2440: a stage flip that names part of a class-wide stage-flag group
refuses.

``ManzariDafalias::mElastFlag`` (inherited by ``LadrunoSANISAND``) and
``SAniSandMS::mElastFlag`` are C++ statics: ``updateMaterialStage -material A``
switches every material of the group, and an unnamed one skips its own
``Elastic2Plastic()``.  The unit tests pin the emit-time refusal
(``validate_class_wide_stage_flags`` in ``_internal/build/stage_validate.py``,
over ``CLASS_WIDE_STAGE_FLAG`` in ``stage/_builder.py``).  It sees the final
model, so a material registered after the stage block counts (#2440), and it
models ``s.remove_element``: a material whose elements are all removed need
not be named.  The live tests run the refused selection on OpenSees and show
it is exactly the one that switches the material it did not name.
"""
from __future__ import annotations

import warnings
from typing import Any

import pytest

from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees._internal.build.stage_emit import (
    FORK_BUILD_HAS_MD_STATIC_FIX,
    MD_STATIC_FIX_COMMIT,
    MD_STATIC_RESET_GUARD,
)
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees._target import OpenSeesTarget
from apeGmsh.opensees.stage._builder import (
    CLASS_WIDE_STAGE_FLAG,
    FIRST_ELEMENT_ONLY_PLASTIC_INIT,
    FORK_RESETS_STAGE_STATIC,
    STAGE_FLAG_PARAMETERS,
    STAGED_MATERIAL_CLASSES,
)

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)

#: Toyoura sand (Dafalias & Manzari 2004), the probe's parameters.
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
_PM4SAND = dict(Dr=0.55, G0=677.0, hpo=0.4, rho=1.7)
_PM4SILT = dict(Su=0.0, Su_rate=0.25, G0=500.0, hpo=0.1, rho=1.7)
_KW = {"ManzariDafalias": _MD, "LadrunoSANISAND": _MD, "SAniSandMS": _MS,
       "PM4Sand": _PM4SAND, "PM4Silt": _PM4SILT}


def _fem() -> FEMStub:
    """Two DISCONNECTED unit quads, ``A`` (eid 1) and ``B`` (eid 2)."""
    return FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4, 5, 6, 7, 8],
            coords=[
                (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0),
                (3.0, 1.0, 0.0), (2.0, 1.0, 0.0),
            ],
            node_pgs={"A": [1, 2, 3, 4], "B": [5, 6, 7, 8],
                      "Base": [1, 2, 5, 6]},
        ),
        elements=_ElementsStub(
            elem_pgs={
                "A": _ElementGroupView(ids=(1,), connectivity=((1, 2, 3, 4),)),
                "B": _ElementGroupView(ids=(2,), connectivity=((5, 6, 7, 8),)),
            },
        ),
    )


def _tri6_fem() -> FEMStub:
    """Two DISCONNECTED 6-node triangles, ``A`` (eid 1) and ``B`` (eid 2):
    SixNodeTri hands ``materialState`` on to its material (#2460)."""
    tri = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.0), (0.5, 0.5),
           (0.0, 0.5)]
    coords = [(x + dx, y, 0.0) for dx in (0.0, 3.0) for x, y in tri]
    return FEMStub(
        nodes=_NodesStub(
            ids=list(range(1, 13)), coords=coords,
            node_pgs={"A": [1, 2, 3, 4, 5, 6], "B": [7, 8, 9, 10, 11, 12],
                      "Base": [1, 2, 4, 7, 8, 10]},
        ),
        elements=_ElementsStub(
            elem_pgs={
                "A": _ElementGroupView(
                    ids=(1,), connectivity=((1, 2, 3, 4, 5, 6),)),
                "B": _ElementGroupView(
                    ids=(2,), connectivity=((7, 8, 9, 10, 11, 12),)),
            },
        ),
    )


def _chain(ops: apeSees) -> dict[str, object]:
    return {
        "test": ops.test.NormDispIncr(tol=1e-8, max_iter=50),
        "algorithm": ops.algorithm.Newton(),
        "integrator": ops.integrator.LoadControl(dlam=0.1),
        "constraints": ops.constraints.Plain(),
        "numberer": ops.numberer.RCM(),
        "system": ops.system.UmfPack(),
        "analysis": ops.analysis.Static(),
    }


def _bridge(
    cls_a: str, cls_b: str, host: str = "FourNodeQuad",
) -> tuple[apeSees, Any, Any]:
    fem = _tri6_fem() if host == "SixNodeTri" else _fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # SAniSandMS's default scheme warns
        mat_a = getattr(ops.nDMaterial, cls_a)(name="mat_a", **_KW[cls_a])
        mat_b = getattr(ops.nDMaterial, cls_b)(name="mat_b", **_KW[cls_b])
    getattr(ops.element, host)(pg="A", thickness=1.0, material=mat_a)
    getattr(ops.element, host)(pg="B", thickness=1.0, material=mat_b)
    ops.fix(pg="Base", dofs=(0, 1))
    return ops, mat_a, mat_b


def _stage(ops: apeSees, flips: "list[tuple[list[Any], int]]", *,
           name: str = "flip", remove: "str | None" = None) -> None:
    with ops.stage(name=name) as s:
        if remove is not None:
            s.remove_element(pg=remove)
        for mats, value in flips:
            s.update_material_stage(materials=mats, stage=value)
        s.analysis(**_chain(ops))
        s.run(n_increments=1)


def _emit(ops: apeSees) -> RecordingEmitter:
    """The emit is where the check runs, beside validator V7."""
    rec = RecordingEmitter()
    ops.build().emit(rec)
    return rec


#: Every class-wide pair: two of one class, and the two classes that share
#: ManzariDafalias's static.
_GROUP_PAIRS = [
    ("ManzariDafalias", "ManzariDafalias"),
    ("SAniSandMS", "SAniSandMS"),
    ("LadrunoSANISAND", "LadrunoSANISAND"),
    ("ManzariDafalias", "LadrunoSANISAND"),
    ("LadrunoSANISAND", "ManzariDafalias"),
    ("PM4Sand", "PM4Sand"),
    ("PM4Silt", "PM4Silt"),
]


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def test_table_groups_the_shared_statics() -> None:
    assert CLASS_WIDE_STAGE_FLAG == {
        "ManzariDafalias": "ManzariDafalias::mElastFlag",
        "LadrunoSANISAND": "ManzariDafalias::mElastFlag",
        "SAniSandMS": "SAniSandMS::mElastFlag",
        "PM4Sand": "PM4Sand::me2p",      # its own static, not PM4Silt's
        "PM4Silt": "PM4Silt::me2p",
    }
    # A class-wide class must be stageable at all, or the check is dead.
    assert set(CLASS_WIDE_STAGE_FLAG) <= STAGED_MATERIAL_CLASSES


# ---------------------------------------------------------------------------
# Refusal: a strict subset of one group
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("cls_a", "cls_b"), _GROUP_PAIRS)
@pytest.mark.parametrize("value", [0, 1])
def test_subset_flip_refuses_and_names_the_unnamed(
    cls_a: str, cls_b: str, value: int,
) -> None:
    ops, mat_a, mat_b = _bridge(cls_a, cls_b)
    tag_b = ops.tag_for(mat_b)
    _stage(ops, [([mat_a], value)])
    with pytest.raises(BridgeError) as exc:
        _emit(ops)
    msg = str(exc.value)
    assert f"{cls_b} (nDMaterial tag {tag_b})" in msg
    assert CLASS_WIDE_STAGE_FLAG[cls_a] in msg
    assert "class-wide" in msg


@pytest.mark.parametrize(("cls_a", "cls_b"), _GROUP_PAIRS)
def test_full_group_flip_passes(cls_a: str, cls_b: str) -> None:
    ops, mat_a, mat_b = _bridge(cls_a, cls_b)
    _stage(ops, [([mat_a, mat_b], 1)])
    (rec,) = ops._stage_records[0].update_material_stage_records
    assert rec.mat_tags == (ops.tag_for(mat_a), ops.tag_for(mat_b))
    _emit(ops)


def test_full_group_over_two_calls_passes() -> None:
    """The lines emit back to back, so one call per material is the same."""
    ops, mat_a, mat_b = _bridge("ManzariDafalias", "LadrunoSANISAND")
    _stage(ops, [([mat_a], 1), ([mat_b], 1)])
    _emit(ops)


def test_one_group_two_values_refuses() -> None:
    ops, mat_a, mat_b = _bridge("ManzariDafalias", "ManzariDafalias")
    _stage(ops, [([mat_a], 1), ([mat_b], 0)])
    with pytest.raises(BridgeError, match=r"different stages \[0, 1\]"):
        _emit(ops)


def test_separate_groups_do_not_count_together() -> None:
    """ManzariDafalias and SAniSandMS have separate statics: flipping all of
    one group and none of the other is complete."""
    ops, mat_a, mat_b = _bridge("ManzariDafalias", "SAniSandMS")
    _stage(ops, [([mat_a], 1)])
    _emit(ops)


def test_pm4sand_and_pm4silt_are_separate_groups() -> None:
    """``PM4Sand::me2p`` and ``PM4Silt::me2p`` are two statics (PM4Silt does
    not derive from PM4Sand): flipping one class alone is complete."""
    ops, mat_a, mat_b = _bridge("PM4Sand", "PM4Silt")
    _stage(ops, [([mat_a], 1)])
    assert len(ops._stage_records) == 1


def test_an_unflipped_stage_is_not_checked() -> None:
    ops, _mat_a, _mat_b = _bridge("ManzariDafalias", "ManzariDafalias")
    _stage(ops, [])
    _emit(ops)


# ---------------------------------------------------------------------------
# #2440: the final model, and s.remove_element
# ---------------------------------------------------------------------------


def _late_bridge() -> tuple[apeSees, Any, Any]:
    """``mat_b`` and its element on ``B`` are registered AFTER the stage
    block that flips ``mat_a`` alone."""
    ops = apeSees(_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mat_a = ops.nDMaterial.ManzariDafalias(name="mat_a", **_MD)
    ops.element.FourNodeQuad(pg="A", thickness=1.0, material=mat_a)
    ops.fix(pg="Base", dofs=(0, 1))
    _stage(ops, [([mat_a], 1)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mat_b = ops.nDMaterial.ManzariDafalias(name="mat_b", **_MD)
    ops.element.FourNodeQuad(pg="B", thickness=1.0, material=mat_b)
    return ops, mat_a, mat_b


def test_a_material_registered_after_the_stage_counts() -> None:
    """The blind spot of a stage-close check: ``mat_b`` did not exist when
    the stage closed, yet its element is global, so it is live at the flip
    line and the flip would switch it."""
    ops, _mat_a, mat_b = _late_bridge()
    with pytest.raises(BridgeError) as exc:
        _emit(ops)
    assert f"ManzariDafalias (nDMaterial tag {ops.tag_for(mat_b)})" in str(
        exc.value)


@pytest.mark.parametrize("target", ["tcl", "py", "h5"])
def test_the_late_material_refuses_on_every_target(
    target: str, tmp_path: Any,
) -> None:
    ops, _mat_a, _mat_b = _late_bridge()
    with pytest.raises(BridgeError, match="class-wide"):
        getattr(ops, target)(str(tmp_path / f"deck.{target}"))


def test_a_material_removed_in_the_same_stage_need_not_be_named() -> None:
    """The removals emit before the flips, so the only element of ``B`` is
    out of the domain at the flip line: ``mat_b`` is not switched.  A
    stage-close check that did not model the removal refused this."""
    ops, mat_a, _mat_b = _bridge("ManzariDafalias", "ManzariDafalias")
    _stage(ops, [([mat_a], 1)], remove="B")
    rec = _emit(ops)
    assert ("update_material_stage", (ops.tag_for(mat_a), 1), {}) in rec.calls


def test_a_material_a_later_stage_activates_need_not_be_named() -> None:
    """``B`` is registered before the flip stage closes, but the stage after
    it activates ``B``, so ``B`` is out of the domain at the flip line.  Only
    the final model shows that, which is why no check runs at stage close:
    one there would refuse this correct model."""
    ops = apeSees(_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mat_a = ops.nDMaterial.ManzariDafalias(name="mat_a", **_MD)
        mat_b = ops.nDMaterial.ManzariDafalias(name="mat_b", **_MD)
    ops.element.FourNodeQuad(pg="A", thickness=1.0, material=mat_a)
    ops.element.FourNodeQuad(pg="B", thickness=1.0, material=mat_b)
    ops.fix(nodes=[1, 2], dofs=(0, 1))
    _stage(ops, [([mat_a], 1)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with ops.stage(name="install") as s:
            s.activate(pgs=["B"])
            s.fix(nodes=[5, 6], dofs=(0, 1))
            s.analysis(**_chain(ops))
            s.run(n_increments=1)
    _emit(ops)


def test_a_material_removed_in_an_earlier_stage_need_not_be_named() -> None:
    ops, mat_a, _mat_b = _bridge("ManzariDafalias", "ManzariDafalias")
    _stage(ops, [], name="excavate", remove="B")
    _stage(ops, [([mat_a], 1)])
    _emit(ops)


def test_a_flip_before_the_removal_still_counts_the_removed_material() -> None:
    ops, mat_a, mat_b = _bridge("ManzariDafalias", "ManzariDafalias")
    _stage(ops, [([mat_a], 1)])
    _stage(ops, [], name="excavate", remove="B")
    with pytest.raises(BridgeError) as exc:
        _emit(ops)
    assert f"(nDMaterial tag {ops.tag_for(mat_b)})" in str(exc.value)


def test_a_flip_whose_materials_are_all_removed_refuses() -> None:
    """Every element of the named material is gone: the line reaches no
    element and switches nothing (``no effect``), which V7 does not see."""
    ops, mat_a, _mat_b = _bridge("ManzariDafalias", "SAniSandMS")
    _stage(ops, [([mat_a], 1)], remove="A")
    with pytest.raises(BridgeError, match="has been removed"):
        _emit(ops)


# ---------------------------------------------------------------------------
# Live: the refused selection is the one that switches the unnamed material
# ---------------------------------------------------------------------------

_ARGS = {
    "ManzariDafalias": [125.0, 0.05, 0.8, 1.25, 0.712, 0.019, 0.934, 0.7,
                        100.0, 0.01, 7.05, 0.968, 1.1, 0.704, 3.5, 4.0,
                        600.0, 1.42],
    "SAniSandMS": [125.0, 0.05, 0.8, 1.25, 0.712, 0.019, 0.934, 0.7, 100.0,
                   0.01, 7.05, 0.968, 1.1, 0.704, 3.5, 0.0005, 260.0, 1.0,
                   1.6],
}
_ARGS["LadrunoSANISAND"] = _ARGS["ManzariDafalias"]


def _run(cls_a: str, cls_b: str, named: "tuple[int, ...]", *,
         start: int, flip: int, dev: float) -> "tuple[int, float, float]":
    """Two disconnected plane-strain quads, tags 1 (``cls_a``) and 2
    (``cls_b``).  Both start at stage ``start``; hydrostatic 100; then
    ``updateMaterialStage -stage flip`` on the tags in ``named``; then
    ``+dev`` on sigma_y.  Returns (analyze code, quad 2's top-corner ux, uy).
    """
    from apeGmsh.opensees.emitter.live import get_ops

    ops = get_ops()
    ops.wipe()
    try:
        ops.model("basic", "-ndm", 2, "-ndf", 2)
        quads = {1: (0.0, (1, 2, 3, 4)), 2: (2.0, (5, 6, 7, 8))}
        for x0, nn in quads.values():
            for n, (dx, y) in zip(nn, ((0, 0), (1, 0), (1, 1), (0, 1))):
                ops.node(n, x0 + dx, float(y))
                ops.fix(n, int(dx == 0), int(y == 0))
        ops.nDMaterial(cls_a, 1, *_ARGS[cls_a])
        ops.nDMaterial(cls_b, 2, *_ARGS[cls_b])
        for e, (_x0, nn) in quads.items():
            ops.element("quad", e, *nn, 1.0, "PlaneStrain", e)
        for t in (1, 2):
            ops.updateMaterialStage("-material", t, "-stage", start)

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
        for _x0, nn in quads.values():
            for n in (nn[1], nn[2]):
                ops.load(n, -50.0, 0.0)
            for n in (nn[2], nn[3]):
                ops.load(n, 0.0, -50.0)
        code = analyze()
        ops.loadConst("-time", 0.0)
        for t in named:
            ops.updateMaterialStage("-material", t, "-stage", flip)
        ops.timeSeries("Linear", 2)
        ops.pattern("Plain", 2, 2)
        for _x0, nn in quads.values():
            for n in (nn[2], nn[3]):
                ops.load(n, 0.0, -dev / 2.0)
        code = code or analyze()
        return code, float(ops.nodeDisp(7, 1)), float(ops.nodeDisp(7, 2))
    finally:
        ops.wipe()


def _assert_bridge_refuses(cls_a: str, cls_b: str, flip: int) -> None:
    ops, mat_a, _mat_b = _bridge(cls_a, cls_b)
    _stage(ops, [([mat_a], flip)])
    with pytest.raises(BridgeError, match="class-wide"):
        _emit(ops)


@pytest.mark.live
@pytest.mark.parametrize(("cls", "dev"), [
    ("ManzariDafalias", 40.0), ("SAniSandMS", 8.0)])
def test_live_flip_to_plastic_switches_the_unnamed_material(
    cls: str, dev: float,
) -> None:
    """Stock and fork: elastic start, then ``-stage 1`` on tag 1 only.  Quad 2
    leaves the elastic answer it keeps when nothing is flipped."""
    pytest.importorskip("openseespy.opensees")
    _assert_bridge_refuses(cls, cls, 1)
    code_n, ux_n, uy_n = _run(cls, cls, (), start=0, flip=1, dev=dev)
    code_1, ux_1, uy_1 = _run(cls, cls, (1,), start=0, flip=1, dev=dev)
    code_b, ux_b, uy_b = _run(cls, cls, (1, 2), start=0, flip=1, dev=dev)
    assert (code_n, code_1, code_b) == (0, 0, 0)
    # Quad 2 is not named, yet its answer moved by far more than round-off.
    assert abs(uy_1 - uy_n) > 1e-2 * abs(uy_n)
    if cls == "ManzariDafalias":
        # ...and it skipped Elastic2Plastic: it matches neither run.
        assert abs(uy_1 - uy_b) > 1e-2 * abs(uy_b)


@pytest.mark.live
@pytest.mark.ladruno_fork
@pytest.mark.parametrize(("cls_a", "cls_b"), [
    ("ManzariDafalias", "LadrunoSANISAND"),
    ("LadrunoSANISAND", "ManzariDafalias")])
def test_live_manzari_flip_switches_ladruno_sanisand(
    cls_a: str, cls_b: str,
) -> None:
    """Fork: both plastic, then ``-stage 0`` on tag 1 only.  Quad 2, of the
    other class, goes elastic exactly as if it had been named: one static."""
    pytest.importorskip("openseespy.opensees")
    _assert_bridge_refuses(cls_a, cls_b, 0)
    code_1, ux_1, uy_1 = _run(cls_a, cls_b, (1,), start=1, flip=0, dev=8.0)
    code_b, ux_b, uy_b = _run(cls_a, cls_b, (1, 2), start=1, flip=0, dev=8.0)
    code_n, _ux_n, uy_n = _run(cls_a, cls_b, (), start=1, flip=0, dev=8.0)
    assert (code_1, code_b) == (0, 0)
    assert (ux_1, uy_1) == pytest.approx((ux_b, uy_b), rel=1e-10, abs=0.0)
    # Left plastic, quad 2 cannot carry the leg the elastic one carries.
    assert code_n != 0 or abs(uy_n - uy_1) > 1e-2 * abs(uy_1)


# ---------------------------------------------------------------------------
# The update_parameter route flips the same static
# ---------------------------------------------------------------------------


def test_stage_flag_parameter_names() -> None:
    assert STAGE_FLAG_PARAMETERS == {"updateMaterialStage", "materialState"}


@pytest.mark.parametrize(("cls_a", "cls_b"), _GROUP_PAIRS)
@pytest.mark.parametrize("name", ["materialState"])
def test_parameter_route_subset_refuses(
    cls_a: str, cls_b: str, name: str,
) -> None:
    # #2460: on a forwarding host; the stage close refuses the
    # updateMaterialStage name (test_stage_parameter_routes.py).
    ops, mat_a, mat_b = _bridge(cls_a, cls_b, host="SixNodeTri")
    with ops.stage(name="flip") as s:
        s.update_parameter(name, 1.0, pg="A", material=mat_a)
        s.analysis(**_chain(ops))
        s.run(n_increments=1)
    with pytest.raises(BridgeError) as exc:
        _emit(ops)
    assert f"{cls_b} (nDMaterial tag {ops.tag_for(mat_b)})" in str(exc.value)


def test_parameter_route_full_group_passes() -> None:
    # #2460: on a SixNodeTri host, which forwards materialState.
    ops, mat_a, mat_b = _bridge(
        "ManzariDafalias", "ManzariDafalias", host="SixNodeTri")
    with ops.stage(name="flip") as s:
        s.update_parameter("materialState", 1.0, pg="A", material=mat_a)
        s.update_parameter("materialState", 1.0, pg="B", material=mat_b)
        s.analysis(**_chain(ops))
        s.run(n_increments=1)
    _emit(ops)


def test_parameter_route_full_group_on_four_node_quad_refuses() -> None:
    """#2460: FourNodeQuad reads materialState as its own per-point
    address, so the full-group flip that passes on SixNodeTri is lost."""
    ops, mat_a, mat_b = _bridge("ManzariDafalias", "ManzariDafalias")
    with pytest.raises(BridgeError) as exc:
        with ops.stage(name="flip") as s:
            s.update_parameter("materialState", 1.0, pg="A", material=mat_a)
            s.update_parameter("materialState", 1.0, pg="B", material=mat_b)
            s.analysis(**_chain(ops))
            s.run(n_increments=1)
    msg = str(exc.value)
    assert "targets FourNodeQuad elements" in msg and "#2460" in msg
    assert not ops._stage_records


@pytest.mark.parametrize(("value", "ok"), [(1.0, True), (0.0, False)])
def test_the_two_routes_count_together(value: float, ok: bool) -> None:
    ops, mat_a, mat_b = _bridge(
        "ManzariDafalias", "LadrunoSANISAND", host="SixNodeTri")

    def build() -> None:
        with ops.stage(name="flip") as s:
            s.update_material_stage(materials=[mat_a], stage=1)
            s.update_parameter("materialState", value, pg="B", material=mat_b)
            s.analysis(**_chain(ops))
            s.run(n_increments=1)

    build()
    if ok:
        _emit(ops)
    else:
        with pytest.raises(BridgeError, match=r"different stages \[0, 1\]"):
            _emit(ops)


def test_a_non_stage_parameter_is_not_a_flip() -> None:
    ops, mat_a, _mat_b = _bridge("ManzariDafalias", "ManzariDafalias")
    with ops.stage(name="soften") as s:
        s.update_parameter("poissonRatio", 0.3, pg="A", material=mat_a)
        s.analysis(**_chain(ops))
        s.run(n_increments=1)
    _emit(ops)


def _analyze(ops: Any) -> int:
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


def _run_param(cls: str, *, named_param: bool, dev: float) -> "tuple[int, float]":
    """Elastic start, hydrostatic 100, then (optionally) ``materialState 1``
    on element 1 only through ``parameter``/``addToParameter``, then +dev on
    sigma_y.  Returns (analyze code, quad 2's uy)."""
    from apeGmsh.opensees.emitter.live import get_ops

    ops = get_ops()
    ops.wipe()
    try:
        ops.model("basic", "-ndm", 2, "-ndf", 2)
        quads = {1: (0.0, (1, 2, 3, 4)), 2: (2.0, (5, 6, 7, 8))}
        for x0, nn in quads.values():
            for n, (dx, y) in zip(nn, ((0, 0), (1, 0), (1, 1), (0, 1))):
                ops.node(n, x0 + dx, float(y))
                ops.fix(n, int(dx == 0), int(y == 0))
        for t in (1, 2):
            ops.nDMaterial(cls, t, *_ARGS[cls])
        # SSPquad forwards every unmatched argv to its material; ``quad``
        # reads ``materialState`` as its own ``material <gp> ...`` form.
        for e, (_x0, nn) in quads.items():
            ops.element("SSPquad", e, *nn, e, "PlaneStrain", 1.0)
        for t in (1, 2):
            ops.updateMaterialStage("-material", t, "-stage", 0)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        for _x0, nn in quads.values():
            for n in (nn[1], nn[2]):
                ops.load(n, -50.0, 0.0)
            for n in (nn[2], nn[3]):
                ops.load(n, 0.0, -50.0)
        code = _analyze(ops)
        ops.loadConst("-time", 0.0)
        if named_param:
            ops.parameter(1)
            ops.addToParameter(1, "element", 1, "materialState", 1)
            ops.updateParameter(1, 1.0)
            ops.remove("parameter", 1)
        ops.timeSeries("Linear", 2)
        ops.pattern("Plain", 2, 2)
        for _x0, nn in quads.values():
            for n in (nn[2], nn[3]):
                ops.load(n, 0.0, -dev / 2.0)
        code = code or _analyze(ops)
        return code, float(ops.nodeDisp(7, 2))
    finally:
        ops.wipe()


@pytest.mark.live
@pytest.mark.parametrize(("cls", "dev"), [
    ("ManzariDafalias", 40.0), ("SAniSandMS", 8.0)])
def test_live_parameter_route_switches_the_unnamed_material(
    cls: str, dev: float,
) -> None:
    """Stock and fork: ``materialState 1`` on material 1 moves material 2,
    and the bridge refuses exactly that selection."""
    pytest.importorskip("openseespy.opensees")
    # #2460: a forwarding host, as the raw deck's SSPquad.
    ops, mat_a, _mat_b = _bridge(cls, cls, host="SixNodeTri")
    with ops.stage(name="flip") as s:
        s.update_parameter("materialState", 1.0, pg="A", material=mat_a)
        s.analysis(**_chain(ops))
        s.run(n_increments=1)
    with pytest.raises(BridgeError, match="class-wide"):
        _emit(ops)
    code_n, uy_n = _run_param(cls, named_param=False, dev=dev)
    code_p, uy_p = _run_param(cls, named_param=True, dev=dev)
    assert (code_n, code_p) == (0, 0)
    assert abs(uy_p - uy_n) > 1e-2 * abs(uy_n)


# ---------------------------------------------------------------------------
# Fork: a later activation resets the class-wide static (fork issue #2444)
# ---------------------------------------------------------------------------


def _staged_bridge(cls: str, mode: "str | None") -> tuple[apeSees, Any]:
    """Two materials of ``cls``, one element each: pg A is global and pg B
    is activated by a later stage (one element per material keeps the
    plastic flip clear of the #2445 refusal)."""
    target = None if mode is None else OpenSeesTarget(mode=mode)  # type: ignore[arg-type]
    ops = apeSees(_fem(), default_orientation=None, opensees=target)
    ops.model(ndm=2, ndf=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mat_a = getattr(ops.nDMaterial, cls)(name="mat_a", **_KW[cls])
        mat_b = getattr(ops.nDMaterial, cls)(name="mat_b", **_KW[cls])
    ops.element.FourNodeQuad(pg="A", thickness=1.0, material=mat_a)
    ops.element.FourNodeQuad(pg="B", thickness=1.0, material=mat_b)
    ops.fix(pg="Base", dofs=(0, 1))
    return ops, [mat_a, mat_b]


def _flip_then_activate(
    ops: apeSees, mats: Any, *, first: int = 1, reflip: bool = False,
) -> None:
    with ops.stage(name="plastic") as s:
        s.update_material_stage(materials=mats, stage=first)
        s.analysis(**_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="install") as s:
        s.activate(pgs=["B"])
        if reflip:
            s.update_material_stage(materials=mats, stage=first)
        s.analysis(**_chain(ops))
        s.run(n_increments=1)


def test_fork_reset_table() -> None:
    assert FORK_RESETS_STAGE_STATIC == {"ManzariDafalias", "LadrunoSANISAND"}
    assert FORK_RESETS_STAGE_STATIC <= set(CLASS_WIDE_STAGE_FLAG)


@pytest.mark.parametrize("cls", sorted(FORK_RESETS_STAGE_STATIC))
@pytest.mark.parametrize("mode", ["fork", "stock", "auto", None])
def test_the_stage_close_no_longer_refuses_the_reset(
    cls: str, mode: "str | None",
) -> None:
    """#2516: no target refuses the pattern at stage close.  A staged model
    runs only as a deck, so the deck checks the build it runs on
    (``tests/opensees/integration/test_deck_md_static_guard.py``)."""
    ops, mat = _staged_bridge(cls, mode)
    _flip_then_activate(ops, mat)
    assert len(ops._stage_records) == 2


def _run_reset(cls: str, *, add: bool) -> "tuple[int, float]":
    """Quad 1: elastic hydrostatic 100, ``-stage 1``, optionally a new
    quad 2 of the same material, then +4 on sigma_y.  Returns quad 1's uy."""
    from apeGmsh.opensees.emitter.live import get_ops

    ops = get_ops()
    ops.wipe()
    try:
        ops.model("basic", "-ndm", 2, "-ndf", 2)
        ops.nDMaterial(cls, 1, *_ARGS[cls])
        nn = (1, 2, 3, 4)
        for n, (x, y) in zip(nn, ((0, 0), (1, 0), (1, 1), (0, 1))):
            ops.node(n, float(x), float(y))
            ops.fix(n, int(x == 0), int(y == 0))
        ops.element("quad", 1, *nn, 1.0, "PlaneStrain", 1)
        ops.updateMaterialStage("-material", 1, "-stage", 0)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        for n in (2, 3):
            ops.load(n, -50.0, 0.0)
        for n in (3, 4):
            ops.load(n, 0.0, -50.0)
        code = _analyze(ops)
        ops.loadConst("-time", 0.0)
        ops.updateMaterialStage("-material", 1, "-stage", 1)
        if add:
            for n, (x, y) in zip((5, 6, 7, 8),
                                 ((2, 0), (3, 0), (3, 1), (2, 1))):
                ops.node(n, float(x), float(y))
                ops.fix(n, 1, 1)
            ops.element("quad", 2, 5, 6, 7, 8, 1.0, "PlaneStrain", 1)
        ops.timeSeries("Linear", 2)
        ops.pattern("Plain", 2, 2)
        for n in (3, 4):
            ops.load(n, 0.0, -2.0)
        code = code or _analyze(ops)
        return code, float(ops.nodeDisp(3, 2))
    finally:
        ops.wipe()


#: The fork commit that stops the constructor reset, and the fork builds
#: measured with :func:`_run_reset` (does the build carry it?).  One
#: table, in ``stage_emit``: the deck guard accepts its ``True`` rows.
_MD_STATIC_FIX = MD_STATIC_FIX_COMMIT
_FORK_BUILD_HAS_MD_STATIC_FIX = FORK_BUILD_HAS_MD_STATIC_FIX


def _reset_expected(cls: str) -> bool:
    """Whether the bound build resets ``cls``'s static on a new element:
    a fork build without :data:`_MD_STATIC_FIX`, for the classes of
    ``FORK_RESETS_STAGE_STATIC`` only.  An unmeasured fork build skips by
    name rather than guess."""
    from apeGmsh.opensees.emitter.live import get_backend_info

    info = get_backend_info()
    if info.kind != "fork" or cls not in FORK_RESETS_STAGE_STATIC:
        # Stock never resets; neither does a class whose constructor
        # keeps the flag (SAniSandMS), on any fork build.
        return False
    if info.build not in _FORK_BUILD_HAS_MD_STATIC_FIX:
        pytest.skip(
            f"fork build {info.build!r} is not measured for the #2444 "
            f"reset: add it to _FORK_BUILD_HAS_MD_STATIC_FIX (does it "
            f"descend from {_MD_STATIC_FIX[:9]}?)")
    fixed = _FORK_BUILD_HAS_MD_STATIC_FIX[info.build]
    return not fixed


def test_the_fork_pin_is_measured_for_the_reset() -> None:
    """A FORK_PIN bump re-measures the reset: the live-fork lane runs
    only the pinned build, so it must be in the table."""
    from pathlib import Path

    pin = Path(__file__).resolve().parents[3] / "FORK_PIN"
    shas = [line.split("=", 1)[1] for line in
            pin.read_text(encoding="utf-8").splitlines()
            if line.startswith("sha=")]
    assert len(shas) == 1, shas
    assert shas[0] in _FORK_BUILD_HAS_MD_STATIC_FIX, (
        "FORK_PIN moved: measure the #2444 reset on the new build "
        "(test_live_reset_follows_the_fork_build) and add it")
    # The pin carries the fix, so the deck guard accepts it (#2516).
    assert _FORK_BUILD_HAS_MD_STATIC_FIX[shas[0]] is True
    assert shas[0] in MD_STATIC_RESET_GUARD.builds
    assert set(MD_STATIC_RESET_GUARD.builds) == {
        sha for sha, fixed in _FORK_BUILD_HAS_MD_STATIC_FIX.items() if fixed}


@pytest.mark.live
@pytest.mark.parametrize("cls", ["ManzariDafalias", "SAniSandMS"])
def test_live_reset_happens_exactly_where_the_table_says(cls: str) -> None:
    """A new element after ``-stage 1`` resets the static on a fork build
    older than the fix, for the FORK_RESETS_STAGE_STATIC classes, and
    nowhere else (stock: never; a fork build with the fix: never)."""
    pytest.importorskip("openseespy.opensees")
    expected = _reset_expected(cls)
    code_f, uy_flipped = _run_reset(cls, add=False)
    code_a, uy_added = _run_reset(cls, add=True)
    assert (code_f, code_a) == (0, 0)
    reset = abs(uy_added - uy_flipped) > 1e-3 * abs(uy_flipped)
    assert reset == expected


@pytest.mark.live
@pytest.mark.ladruno_fork
@pytest.mark.parametrize("cls", ["ManzariDafalias", "LadrunoSANISAND"])
def test_live_reset_follows_the_fork_build(cls: str) -> None:
    """On the fork lane: reset on a fork build older than the fix
    (79e062367), kept at FORK_PIN (the #2444 plastic answer).
    LadrunoSANISAND inherits ManzariDafalias's static.  The engine comes
    from the resolver, not ``import openseespy``: the live-fork lane
    installs the fork as ``opensees`` and has no openseespy, so an
    ``importorskip`` would skip this test on the one lane it is for."""
    from apeGmsh.opensees.emitter.live import get_backend_info

    assert get_backend_info().kind == "fork"
    expected = _reset_expected(cls)
    code_f, uy_flipped = _run_reset(cls, add=False)
    code_a, uy_added = _run_reset(cls, add=True)
    assert (code_f, code_a) == (0, 0)
    reset = abs(uy_added - uy_flipped) > 1e-3 * abs(uy_flipped)
    assert reset == expected, (uy_flipped, uy_added)


# ---------------------------------------------------------------------------
# #2445: a plastic flip initialises only the first element of a material
# ---------------------------------------------------------------------------


def _one_material_bridge(
    cls: str, pgs: "tuple[str, ...]", host: str = "FourNodeQuad",
) -> tuple[apeSees, Any]:
    """One material of ``cls`` on the elements of ``pgs`` (one each)."""
    fem = _tri6_fem() if host == "SixNodeTri" else _fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mat = getattr(ops.nDMaterial, cls)(name="mat", **_KW[cls])
    for pg in pgs:
        getattr(ops.element, host)(pg=pg, thickness=1.0, material=mat)
    ops.fix(pg="Base", dofs=(0, 1))
    return ops, mat


#: #2460: the parameter route passes only on a host that forwards
#: materialState; FourNodeQuad drops it (refused below).
_PASS_HOST = {"stage": "FourNodeQuad", "parameter": "SixNodeTri"}


def _flip_one(ops: apeSees, mat: Any, *, route: str, value: int) -> None:
    with ops.stage(name="push") as s:
        if route == "stage":
            s.update_material_stage(materials=[mat], stage=value)
        else:
            for pg in ("A", "B"):
                s.update_parameter("materialState", float(value), pg=pg,
                                   material=mat)
        s.analysis(**_chain(ops))
        s.run(n_increments=1)


def test_first_element_only_table() -> None:
    assert FIRST_ELEMENT_ONLY_PLASTIC_INIT == {
        "ManzariDafalias", "LadrunoSANISAND"}


@pytest.mark.parametrize("cls", sorted(FIRST_ELEMENT_ONLY_PLASTIC_INIT))
@pytest.mark.parametrize("route", ["stage", "parameter"])
def test_plastic_flip_of_a_two_element_material_refuses_on_four_node_quad(
    cls: str, route: str,
) -> None:
    """#2446 lifted the count rule: a multi-element plastic flip passes on a
    class the flip's element sweep reaches
    (``tests/opensees/integration/test_md_stage_route_elements.py``), and
    FourNodeQuad, which drops it, is still refused, by name."""
    ops, mat = _one_material_bridge(cls, ("A", "B"))
    with pytest.raises(BridgeError) as exc:
        _flip_one(ops, mat, route=route, value=1)
    msg = str(exc.value)
    assert "#2445" in msg and "Elastic2Plastic" in msg
    assert "FourNodeQuad carries it" in msg
    assert f"nDMaterial tag {ops.tag_for(mat)} (2 elements)" in msg
    assert not ops._stage_records


@pytest.mark.parametrize("cls", sorted(FIRST_ELEMENT_ONLY_PLASTIC_INIT))
@pytest.mark.parametrize("route", ["stage", "parameter"])
def test_plastic_flip_of_a_one_element_material_passes(
    cls: str, route: str,
) -> None:
    ops, mat = _one_material_bridge(cls, ("A",), _PASS_HOST[route])
    with ops.stage(name="push") as s:
        if route == "stage":
            s.update_material_stage(materials=[mat], stage=1)
        else:
            s.update_parameter("materialState", 1.0, pg="A", material=mat)
        s.analysis(**_chain(ops))
        s.run(n_increments=1)
    assert len(ops._stage_records) == 1


@pytest.mark.parametrize("cls", sorted(FIRST_ELEMENT_ONLY_PLASTIC_INIT))
@pytest.mark.parametrize("route", ["stage", "parameter"])
def test_elastic_flip_of_a_two_element_material_passes(
    cls: str, route: str,
) -> None:
    ops, mat = _one_material_bridge(cls, ("A", "B"), _PASS_HOST[route])
    _flip_one(ops, mat, route=route, value=0)
    assert len(ops._stage_records) == 1


@pytest.mark.parametrize("cls", sorted(FIRST_ELEMENT_ONLY_PLASTIC_INIT))
@pytest.mark.parametrize(("pgs", "value"), [(("A",), 1), (("A", "B"), 0)])
def test_parameter_route_on_four_node_quad_refuses(
    cls: str, pgs: "tuple[str, ...]", value: int,
) -> None:
    """#2460: the one-element plastic and two-element elastic flips that
    pass on SixNodeTri never reach the material on FourNodeQuad."""
    ops, mat = _one_material_bridge(cls, pgs)
    with pytest.raises(BridgeError) as exc:
        with ops.stage(name="push") as s:
            for pg in pgs:
                s.update_parameter("materialState", float(value), pg=pg,
                                   material=mat)
            s.analysis(**_chain(ops))
            s.run(n_increments=1)
    msg = str(exc.value)
    assert "targets FourNodeQuad elements" in msg and "#2460" in msg
    assert not ops._stage_records


def test_a_flag_only_class_on_two_elements_passes() -> None:
    ops, mat = _one_material_bridge("SAniSandMS", ("A", "B"))
    _flip_one(ops, mat, route="stage", value=1)
    assert len(ops._stage_records) == 1


def _run_shared(cls: str, *, shared: bool, dev: float) -> "tuple[int, float]":
    """Two quads, elastic start, hydrostatic 100, ``-stage 1``, +dev sigma_y.
    ``shared``: ONE material (tag 1) on both quads, one flip line.  Else the
    per-element reference: tag 1 and tag 2, both flipped.  Quad 2's uy."""
    from apeGmsh.opensees.emitter.live import get_ops

    ops = get_ops()
    ops.wipe()
    try:
        ops.model("basic", "-ndm", 2, "-ndf", 2)
        quads = {1: (0.0, (1, 2, 3, 4)), 2: (2.0, (5, 6, 7, 8))}
        for x0, nn in quads.values():
            for n, (dx, y) in zip(nn, ((0, 0), (1, 0), (1, 1), (0, 1))):
                ops.node(n, x0 + dx, float(y))
                ops.fix(n, int(dx == 0), int(y == 0))
        tags = (1,) if shared else (1, 2)
        for t in tags:
            ops.nDMaterial(cls, t, *_ARGS[cls])
        for e, (_x0, nn) in quads.items():
            ops.element("quad", e, *nn, 1.0, "PlaneStrain",
                        1 if shared else e)
        for t in tags:
            ops.updateMaterialStage("-material", t, "-stage", 0)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        for _x0, nn in quads.values():
            for n in (nn[1], nn[2]):
                ops.load(n, -50.0, 0.0)
            for n in (nn[2], nn[3]):
                ops.load(n, 0.0, -50.0)
        code = _analyze(ops)
        ops.loadConst("-time", 0.0)
        for t in tags:
            ops.updateMaterialStage("-material", t, "-stage", 1)
        ops.timeSeries("Linear", 2)
        ops.pattern("Plain", 2, 2)
        for _x0, nn in quads.values():
            for n in (nn[2], nn[3]):
                ops.load(n, 0.0, -dev / 2.0)
        code = code or _analyze(ops)
        return code, float(ops.nodeDisp(7, 2))
    finally:
        ops.wipe()


def _assert_first_element_only(cls: str, dev: float) -> None:
    ops, mat = _one_material_bridge(cls, ("A", "B"))
    with pytest.raises(BridgeError, match="#2445"):
        _flip_one(ops, mat, route="stage", value=1)
    code_s, uy_shared = _run_shared(cls, shared=True, dev=dev)
    code_r, uy_ref = _run_shared(cls, shared=False, dev=dev)
    assert code_r == 0
    # The second element of the shared material is not initialised: it is
    # off the reference, or (LadrunoSANISAND) cannot carry the leg at all.
    assert code_s != 0 or abs(uy_shared - uy_ref) > 1e-2 * abs(uy_ref)
    if cls == "ManzariDafalias":
        assert code_s == 0, "the measured ManzariDafalias case converges"


@pytest.mark.live
def test_live_shared_manzari_flip_misses_the_second_element() -> None:
    """Stock and fork: one material on two quads, one ``-stage 1`` line;
    quad 2 is off the per-element reference, the case the bridge refuses."""
    pytest.importorskip("openseespy.opensees")
    _assert_first_element_only("ManzariDafalias", 40.0)


@pytest.mark.live
@pytest.mark.ladruno_fork
def test_live_shared_ladruno_sanisand_flip_misses_the_second_element() -> None:
    pytest.importorskip("openseespy.opensees")
    _assert_first_element_only("LadrunoSANISAND", 4.0)
