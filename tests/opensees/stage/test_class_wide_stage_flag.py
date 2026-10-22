"""#2439: a stage flip that names part of a class-wide stage-flag group refuses.

``ManzariDafalias::mElastFlag`` (inherited by ``LadrunoSANISAND``) and
``SAniSandMS::mElastFlag`` are C++ statics: ``updateMaterialStage -material A``
switches every material of the group, and an unnamed one skips its own
``Elastic2Plastic()``.  The unit tests pin the stage-close refusal
(``CLASS_WIDE_STAGE_FLAG`` in ``stage/_builder.py``); the live tests run the
refused selection on OpenSees and show it is exactly the one that switches
the material it did not name.
"""
from __future__ import annotations

import warnings
from typing import Any

import pytest

from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.stage._builder import (
    CLASS_WIDE_STAGE_FLAG,
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
_KW = {"ManzariDafalias": _MD, "LadrunoSANISAND": _MD, "SAniSandMS": _MS}


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


def _bridge(cls_a: str, cls_b: str) -> tuple[apeSees, Any, Any]:
    ops = apeSees(_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # SAniSandMS's default scheme warns
        mat_a = getattr(ops.nDMaterial, cls_a)(name="mat_a", **_KW[cls_a])
        mat_b = getattr(ops.nDMaterial, cls_b)(name="mat_b", **_KW[cls_b])
    ops.element.FourNodeQuad(pg="A", thickness=1.0, material=mat_a)
    ops.element.FourNodeQuad(pg="B", thickness=1.0, material=mat_b)
    ops.fix(pg="Base", dofs=(0, 1))
    return ops, mat_a, mat_b


def _stage(ops: apeSees, flips: "list[tuple[list[Any], int]]") -> None:
    with ops.stage(name="flip") as s:
        for mats, value in flips:
            s.update_material_stage(materials=mats, stage=value)
        s.analysis(**_chain(ops))
        s.run(n_increments=1)


#: Every class-wide pair: two of one class, and the two classes that share
#: ManzariDafalias's static.
_GROUP_PAIRS = [
    ("ManzariDafalias", "ManzariDafalias"),
    ("SAniSandMS", "SAniSandMS"),
    ("LadrunoSANISAND", "LadrunoSANISAND"),
    ("ManzariDafalias", "LadrunoSANISAND"),
    ("LadrunoSANISAND", "ManzariDafalias"),
]


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def test_table_groups_the_shared_statics() -> None:
    assert CLASS_WIDE_STAGE_FLAG == {
        "ManzariDafalias": "ManzariDafalias::mElastFlag",
        "LadrunoSANISAND": "ManzariDafalias::mElastFlag",
        "SAniSandMS": "SAniSandMS::mElastFlag",
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
    with pytest.raises(BridgeError) as exc:
        _stage(ops, [([mat_a], value)])
    msg = str(exc.value)
    assert f"{cls_b} (nDMaterial tag {tag_b})" in msg
    assert CLASS_WIDE_STAGE_FLAG[cls_a] in msg
    assert "class-wide" in msg
    assert not ops._stage_records, "a refused stage must not be recorded"


@pytest.mark.parametrize(("cls_a", "cls_b"), _GROUP_PAIRS)
def test_full_group_flip_passes(cls_a: str, cls_b: str) -> None:
    ops, mat_a, mat_b = _bridge(cls_a, cls_b)
    _stage(ops, [([mat_a, mat_b], 1)])
    (rec,) = ops._stage_records[0].update_material_stage_records
    assert rec.mat_tags == (ops.tag_for(mat_a), ops.tag_for(mat_b))


def test_full_group_over_two_calls_passes() -> None:
    """The lines emit back to back, so one call per material is the same."""
    ops, mat_a, mat_b = _bridge("ManzariDafalias", "LadrunoSANISAND")
    _stage(ops, [([mat_a], 1), ([mat_b], 1)])
    assert len(ops._stage_records) == 1


def test_one_group_two_values_refuses() -> None:
    ops, mat_a, mat_b = _bridge("ManzariDafalias", "ManzariDafalias")
    with pytest.raises(BridgeError, match=r"different stages \[0, 1\]"):
        _stage(ops, [([mat_a], 1), ([mat_b], 0)])


def test_separate_groups_do_not_count_together() -> None:
    """ManzariDafalias and SAniSandMS have separate statics: flipping all of
    one group and none of the other is complete."""
    ops, mat_a, mat_b = _bridge("ManzariDafalias", "SAniSandMS")
    _stage(ops, [([mat_a], 1)])
    assert len(ops._stage_records) == 1


def test_an_unflipped_stage_is_not_checked() -> None:
    ops, _mat_a, _mat_b = _bridge("ManzariDafalias", "ManzariDafalias")
    _stage(ops, [])
    assert len(ops._stage_records) == 1


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
    with pytest.raises(BridgeError, match="class-wide"):
        _stage(ops, [([mat_a], flip)])


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
