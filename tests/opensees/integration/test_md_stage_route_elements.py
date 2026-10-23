"""#2446: which element classes the plastic stage flip reaches.

The bridge emits a ``ManzariDafalias`` / ``LadrunoSANISAND`` flip to a plastic
stage, on a material two or more elements carry, as a ``materialState``
update of exactly those elements (``setParameter -val 1 -ele <carriers>
materialState <tag>``) followed by ``updateMaterialStage``
(``stage_emit.MaterialStageCarriers``).  The update initialises an element
only when the element hands ``materialState`` to its materials, so a
multi-element plastic flip passes only on a :data:`MATERIAL_STATE_ELEMENTS`
class and is refused, by class name, on any other.

* The unit tests pin the emitted lines and the stage-close rules.
* The live census runs two elements of each class on one material through a
  prescribed homogeneous strain path (every DOF driven, so any element runs
  the same way) and compares the emitted flip with the per-element reference
  (each element its own material and its own ``updateMaterialStage``).
"""
from __future__ import annotations

import io
import warnings
from typing import Any

import pytest

from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees._internal.build.stage_emit import (
    MATERIAL_STATE_ELEMENT_TOKENS,
    MATERIAL_STATE_WRAPPERS,
    SWEPT_MATERIAL_CLASSES,
    MaterialStageCarriers,
)
from apeGmsh.opensees._target import OpenSeesTarget
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.stage._builder import (
    FIRST_ELEMENT_ONLY_PLASTIC_INIT,
    MATERIAL_STATE_ELEMENTS,
)

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)
from tests.opensees.integration.test_md_stage_ladruno_at_pin import (
    _ELEMENT_CASES as _AT_PIN_CASES,
)
from tests.opensees.integration.test_md_stage_all_elements import (
    MD_ARGS,
    Deck,
    emitted_flip_lines,
)

_MD = dict(
    G0=125.0, nu=0.05, e_init=0.8, Mc=1.25, c=0.712, lambda_c=0.019,
    e0=0.934, ksi=0.7, P_atm=100.0, m=0.01, h0=7.05, Ch=0.968, nb=1.1,
    A0=0.704, nd=3.5, z_max=4.0, cz=600.0, rho=1.42,
)

#: The token each :data:`MATERIAL_STATE_ELEMENTS` class emits.
_TOKEN_OF = {
    "SixNodeTri": "tri6n", "FourNodeTetrahedron": "FourNodeTetrahedron",
    "SSPquad": "SSPquad", "SSPbrick": "SSPbrick", "BezierTri6": "BezierTri6",
    "BezierTet10": "BezierTet10", "LadrunoUP": "LadrunoUP",
    "LadrunoQuad": "LadrunoQuad", "LadrunoCST": "LadrunoCST",
    "LadrunoLST": "LadrunoLST", "LadrunoBrick": "LadrunoBrick",
    "LadrunoBrick20": "LadrunoBrick20",
}

_TRI6_ARGV = (1, 2, 3, 4, 5, 6, 1.0, "PlaneStrain")


# ---------------------------------------------------------------------------
# The emitted lines
# ---------------------------------------------------------------------------


def _carriers(mat: str = "ManzariDafalias") -> MaterialStageCarriers:
    """Tag 7 (``mat``) on tri6n 1 and 2, DruckerPrager tag 8 on tri6n 3."""
    c = MaterialStageCarriers()
    c.note_material(mat, 7)
    c.note_material("DruckerPrager", 8)
    c.note_element("tri6n", 1, (*_TRI6_ARGV, 7))
    c.note_element("tri6n", 2, (*_TRI6_ARGV, 7))
    c.note_element("tri6n", 3, (*_TRI6_ARGV, 8))
    return c


def _ums(tag: int, stage: int) -> tuple[str, tuple[Any, ...]]:
    return ("updateMaterialStage", ("-material", tag, "-stage", stage))


def test_the_flip_names_the_carriers_only() -> None:
    assert _carriers().flip_commands(7, 1) == (
        ("setParameter", ("-val", 1.0, "-ele", 1, 2, "materialState", "7")),
        _ums(7, 1),
    )


@pytest.mark.parametrize("cls", sorted(SWEPT_MATERIAL_CLASSES))
def test_both_swept_classes_are_updated(cls: str) -> None:
    assert _carriers(cls).flip_commands(7, 1)[0][0] == "setParameter"


def test_what_keeps_the_single_line() -> None:
    c = _carriers()
    assert c.flip_commands(7, 0) == (_ums(7, 0),), "an elastic flip"
    assert c.flip_commands(8, 1) == (_ums(8, 1),), "a class not swept"
    for mat in ("PressureDependMultiYield", "PM4Sand", "SAniSandMS"):
        other = MaterialStageCarriers()
        other.note_material(mat, 9)
        other.note_element("tri6n", 1, (*_TRI6_ARGV, 9))
        other.note_element("tri6n", 2, (*_TRI6_ARGV, 9))
        assert other.flip_commands(9, 1) == (_ums(9, 1),), mat
    c.note_removed(2)
    assert c.flip_commands(7, 1) == (_ums(7, 1),), "one carrier left"


def test_an_untracked_class_is_not_listed() -> None:
    """A FourNodeQuad carrier is not tracked (the stage close refuses a
    multi-element plastic flip on it), so the list holds the rest."""
    c = _carriers()
    c.note_element("quad", 4, (1, 2, 3, 4, 1.0, "PlaneStrain", 7))
    assert c.flip_commands(7, 1)[0][1][3:5] == (1, 2)


def test_a_rank_lists_its_own_and_the_global_carriers() -> None:
    c = MaterialStageCarriers()
    c.note_material("ManzariDafalias", 7)
    c.note_element("tri6n", 1, (*_TRI6_ARGV, 7))
    for rank, tags in ((0, (2, 3)), (1, (4, 5))):
        c.partition_open(rank)
        for t in tags:
            c.note_element("tri6n", t, (*_TRI6_ARGV, 7))
        c.partition_close()
    c.partition_open(1)
    assert c.flip_commands(7, 1)[0][1][3:6] == (1, 4, 5)
    c.partition_close()
    assert c.flip_commands(7, 1)[0][1][3:8] == (1, 2, 3, 4, 5)


@pytest.mark.parametrize(("token", "argv", "tag"), [
    ("tri6n", (*_TRI6_ARGV, 7), 7),
    ("BezierTri6", (*_TRI6_ARGV, 7, "-bbar"), 7),
    ("FourNodeTetrahedron", (1, 2, 3, 4, 7), 7),
    ("BezierTet10", (*range(1, 11), 7, "-rho", 1.0), 7),
    ("SSPquad", (1, 2, 3, 4, 7, "PlaneStrain", 1.0), 7),
    ("SSPbrick", (*range(1, 9), 7, 0.0, 0.0, -9.81), 7),
    ("LadrunoUP", (1, 2, 3, 4, 7, "-Kf", 2.2e6), 7),
    ("LadrunoUP", (*range(1, 11), 7, "-thick", 1.0, "-Kf", 2.2e6), 7),
    ("LadrunoQuad", (1, 2, 3, 4, 7, "-formulation", "ssp", "-thick", 1.0), 7),
    ("LadrunoCST", (1, 2, 3, 7, "-thick", 1.0), 7),
    ("LadrunoLST", (*range(1, 7), 7, "-thick", 1.0), 7),
    ("LadrunoBrick", (*range(1, 9), 7, "-formulation", "eas"), 7),
    ("LadrunoBrick20", (*range(1, 21), 7, "-formulation", "uri"), 7),
])
def test_the_carried_tag_is_read_from_the_grammar(
    token: str, argv: tuple[Any, ...], tag: int,
) -> None:
    c = MaterialStageCarriers()
    c.note_material("ManzariDafalias", tag)
    c.note_element(token, 1, argv)
    c.note_element(token, 2, argv)
    assert c.flip_commands(tag, 1)[0][1][3:5] == (1, 2)


def test_a_grammar_without_the_tag_fails_loud() -> None:
    c = MaterialStageCarriers()
    with pytest.raises(BridgeError, match="no nDMaterial tag"):
        c.note_element("tri6n", 1, (1, 2, 3, 4, 5, 6, 1.0, "PlaneStrain"))


def test_the_py_lines_pass_the_tag_as_a_string() -> None:
    """``setParameter`` reads its trailing argv as strings: an int tag is
    'Invalid String Input!' on openseespy (measured, stock and fork)."""
    em = PyEmitter()
    em.nDMaterial("ManzariDafalias", 7, *MD_ARGS)
    em.element("tri6n", 1, *_TRI6_ARGV, 7)
    em.element("tri6n", 2, *_TRI6_ARGV, 7)
    assert emitted_flip_lines(7, emitter=em) == [
        "ops.setParameter('-val', 1.0, '-ele', 1, 2, 'materialState', '7')",
        "ops.updateMaterialStage('-material', 7, '-stage', 1)",
    ]
    assert emitted_flip_lines(7, 0, emitter=em) == [
        "ops.updateMaterialStage('-material', 7, '-stage', 0)"]


def test_the_tables() -> None:
    assert FIRST_ELEMENT_ONLY_PLASTIC_INIT == {
        "ManzariDafalias", "LadrunoSANISAND"}
    assert MATERIAL_STATE_ELEMENTS == {
        "SixNodeTri", "FourNodeTetrahedron", "SSPquad", "SSPbrick",
        "BezierTri6", "BezierTet10", "LadrunoUP",
        "LadrunoQuad", "LadrunoCST", "LadrunoLST", "LadrunoBrick",
        "LadrunoBrick20"}
    assert FIRST_ELEMENT_ONLY_PLASTIC_INIT is SWEPT_MATERIAL_CLASSES
    # Every class of the table has a census row on each build it runs on:
    # here, or (the Ladruno solids, every formulation) in the at-pin file.
    assert MATERIAL_STATE_ELEMENTS == (
        {cls for cls, _builds in _FORWARDING} | {"LadrunoUP"}
        | {row[0] for row in _AT_PIN_CASES.values()})


def _emitted_token(cls: str) -> str:
    """The ``element`` token the bridge primitive ``cls`` emits, read from
    its ``_emit`` source; a class with no primitive (``SSPquad``,
    ``SSPbrick``) emits its own name."""
    import inspect
    import re

    from apeGmsh.opensees import element as element_pkg

    prim = getattr(element_pkg, cls, None)
    if prim is None:
        return cls
    found = set(re.findall(r'emitter\.element\(\s*"(\w+)"',
                           inspect.getsource(prim._emit)))
    assert len(found) == 1, (cls, found)
    return found.pop()


def test_the_stage_close_and_the_emit_side_list_the_same_classes() -> None:
    """Drift guard (#2457): the stage close passes a multi-element plastic
    flip on a :data:`MATERIAL_STATE_ELEMENTS` class, and only the emit
    side's :data:`MATERIAL_STATE_ELEMENT_TOKENS` makes the deck list its
    carriers.  A class on the first table without a token would pass the
    stage close and emit the bare first-element-only flip; a token without
    its class would never be reached.  Both directions fail here, with the
    token read from each primitive's own ``_emit``."""
    tokens = {cls: _emitted_token(cls) for cls in MATERIAL_STATE_ELEMENTS}
    no_token = sorted(c for c, t in tokens.items()
                      if t not in MATERIAL_STATE_ELEMENT_TOKENS)
    assert not no_token, f"allowlisted, but the emit side drops: {no_token}"
    no_class = sorted(set(MATERIAL_STATE_ELEMENT_TOKENS) - set(tokens.values()))
    assert not no_class, f"emit-side tokens with no allowlisted class: {no_class}"
    assert tokens == _TOKEN_OF


def _tri6_fem() -> FEMStub:
    """Two DISCONNECTED 6-node triangles: ``A`` (eid 1) and ``B`` (eid 2),
    plus a quad ``Q`` (eid 3) for the mixed-carrier case."""
    tri = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.0), (0.5, 0.5),
           (0.0, 0.5)]
    coords = [(x, y, 0.0) for x, y in tri]
    coords += [(x + 3.0, y, 0.0) for x, y in tri]
    coords += [(6.0, 0.0, 0.0), (7.0, 0.0, 0.0), (7.0, 1.0, 0.0),
               (6.0, 1.0, 0.0)]
    return FEMStub(
        nodes=_NodesStub(
            ids=list(range(1, 17)),
            coords=coords,
            node_pgs={"A": [1, 2, 3, 4, 5, 6], "B": [7, 8, 9, 10, 11, 12],
                      "Q": [13, 14, 15, 16], "Base": [1, 2, 7, 8, 13, 14]},
        ),
        elements=_ElementsStub(
            elem_pgs={
                "A": _ElementGroupView(ids=(1,),
                                       connectivity=((1, 2, 3, 4, 5, 6),)),
                "B": _ElementGroupView(
                    ids=(2,), connectivity=((7, 8, 9, 10, 11, 12),)),
                "Q": _ElementGroupView(ids=(3,),
                                       connectivity=((13, 14, 15, 16),)),
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


def _bridge(
    cls: str = "ManzariDafalias", *, pgs: "tuple[str, ...]" = ("A", "B"),
    quad: bool = False, mode: "str | None" = None,
) -> tuple[apeSees, Any]:
    target = None if mode is None else OpenSeesTarget(mode=mode)  # type: ignore[arg-type]
    ops = apeSees(_tri6_fem(), default_orientation=None, opensees=target)
    ops.model(ndm=2, ndf=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mat = getattr(ops.nDMaterial, cls)(name="sand", **_MD)
    for pg in pgs:
        ops.element.SixNodeTri(pg=pg, thickness=1.0, material=mat)
    if quad:
        ops.element.FourNodeQuad(pg="Q", thickness=1.0, material=mat)
    ops.fix(pg="Base", dofs=(0, 1))
    return ops, mat


def _flip(ops: apeSees, mat: Any, *, route: str = "stage", value: int = 1,
          pgs: "tuple[str, ...]" = ("A", "B")) -> None:
    with ops.stage(name="push") as s:
        if route == "stage":
            s.update_material_stage(materials=[mat], stage=value)
        else:
            for pg in pgs:
                s.update_parameter(route, float(value), pg=pg, material=mat)
        s.analysis(**_chain(ops))
        s.run(n_increments=1)


@pytest.mark.parametrize("cls", sorted(FIRST_ELEMENT_ONLY_PLASTIC_INIT))
def test_the_decks_sweep_then_flip(cls: str) -> None:
    ops, mat = _bridge(cls)
    _flip(ops, mat)
    tag = ops.tag_for(mat)
    tcl = TclEmitter()
    ops.build().emit(tcl)
    lines = tcl.lines()
    eles = [ln.split()[2] for ln in lines if ln.startswith("element tri6n")]
    assert len(eles) == 2
    sweep = (f"setParameter -val 1.0 -ele {' '.join(eles)} materialState "
             f"{tag}")
    ums = f"updateMaterialStage -material {tag} -stage 1"
    assert lines.index(sweep) + 1 == lines.index(ums)
    py = PyEmitter()
    ops.build().emit(py)
    buf = io.StringIO()
    py.write_to(buf)
    text = buf.getvalue()
    assert (
        f"ops.setParameter('-val', 1.0, '-ele', {', '.join(eles)}, "
        f"'materialState', '{tag}')\n"
        f"ops.updateMaterialStage('-material', {tag}, '-stage', 1)"
    ) in text


def test_a_neighbour_of_another_material_is_not_listed() -> None:
    """The reviewer's #2446 model: the sand on SixNodeTri A, DruckerPrager
    (which takes materialState whatever its tag) on B.  The sand has one
    carrier, so the deck keeps the bare updateMaterialStage line."""
    ops = apeSees(_tri6_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sand = ops.nDMaterial.ManzariDafalias(name="sand", **_MD)
    clay = ops.nDMaterial.DruckerPrager(
        K=1e5, G=5e4, sigmaY=10.0, rho=0.0, rhoBar=0.0, Kinf=10.0, Ko=10.0,
        delta1=0.0, delta2=0.0, H=1000.0, theta=1.0, name="clay")
    ops.element.SixNodeTri(pg="A", thickness=1.0, material=sand)
    ops.element.SixNodeTri(pg="B", thickness=1.0, material=clay)
    ops.fix(pg="Base", dofs=(1, 1))
    _flip(ops, sand)
    tcl = TclEmitter()
    ops.build().emit(tcl)
    assert not [ln for ln in tcl.lines() if ln.startswith("setParameter")]
    assert (f"updateMaterialStage -material {ops.tag_for(sand)} -stage 1"
            in tcl.lines())


def test_an_elastic_flip_is_one_line() -> None:
    ops, mat = _bridge()
    _flip(ops, mat, value=0)
    tcl = TclEmitter()
    ops.build().emit(tcl)
    assert not [ln for ln in tcl.lines() if ln.startswith("setParameter")]
    assert f"updateMaterialStage -material {ops.tag_for(mat)} -stage 0" in (
        tcl.lines())


# ---------------------------------------------------------------------------
# The stage-close rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", sorted(FIRST_ELEMENT_ONLY_PLASTIC_INIT))
@pytest.mark.parametrize("route", ["stage", "materialState"])
def test_a_forwarding_class_on_two_elements_passes(cls: str, route: str) -> None:
    ops, mat = _bridge(cls)
    _flip(ops, mat, route=route)
    assert len(ops._stage_records) == 1


@pytest.mark.parametrize("route", ["stage", "materialState"])
def test_a_non_forwarding_carrier_is_refused_by_name(route: str) -> None:
    ops, mat = _bridge(quad=True)
    with pytest.raises(BridgeError) as exc:
        _flip(ops, mat, route=route, pgs=("A", "B", "Q"))
    msg = str(exc.value)
    assert "FourNodeQuad carries it" in msg and "#2445" in msg
    assert "SixNodeTri carries" not in msg
    assert f"nDMaterial tag {ops.tag_for(mat)} (3 elements)" in msg
    assert not ops._stage_records


@pytest.mark.parametrize("route", ["stage", "materialState"])
def test_a_ladruno_quad_carrier_passes_and_is_listed(route: str) -> None:
    """Measured to forward at FORK_PIN (#2457, live-fork run 39874892907):
    the stage close passes a SixNodeTri + LadrunoQuad carrier pair, and the
    deck lists both elements in the materialState update."""
    ops, mat = _bridge(pgs=("A",))
    ops.element.LadrunoQuad(pg="Q", material=mat, thickness=1.0)
    _flip(ops, mat, route=route, pgs=("A", "Q"))
    tcl = TclEmitter()
    ops.build().emit(tcl)
    lines = tcl.lines()
    eles = [ln.split()[2] for ln in lines
            if ln.startswith(("element tri6n", "element LadrunoQuad"))]
    assert len(eles) == 2
    tag = ops.tag_for(mat)
    if route == "stage":
        sweep = (f"setParameter -val 1.0 -ele {' '.join(eles)} "
                 f"materialState {tag}")
        assert lines.index(sweep) + 1 == lines.index(
            f"updateMaterialStage -material {tag} -stage 1")
    else:
        # The user's own update: each target element joins a parameter.
        reached = {ln.split()[3] for ln in lines
                   if ln.startswith("addToParameter")
                   and ln.endswith(f"materialState {tag}")}
        assert reached == set(eles), lines


def test_a_materialstate_route_must_cover_every_element() -> None:
    ops, mat = _bridge()
    with pytest.raises(BridgeError) as exc:
        _flip(ops, mat, route="materialState", pgs=("A",))
    msg = str(exc.value)
    assert "leaves 1 of those elements out of its targets" in msg
    assert "#2445" in msg


def test_a_partial_materialstate_route_beside_the_stage_flip_passes() -> None:
    """``s.update_material_stage`` of the same material sweeps them all."""
    ops, mat = _bridge()
    with ops.stage(name="push") as s:
        s.update_material_stage(materials=[mat], stage=1)
        s.update_parameter("materialState", 1.0, pg="A", material=mat)
        s.analysis(**_chain(ops))
        s.run(n_increments=1)
    assert len(ops._stage_records) == 1


def test_the_updatematerialstage_parameter_is_refused() -> None:
    ops, mat = _bridge()
    with pytest.raises(BridgeError, match="theInt"):
        _flip(ops, mat, route="updateMaterialStage")


def test_a_one_element_material_passes_on_any_class() -> None:
    ops, mat = _bridge(pgs=(), quad=True)
    _flip(ops, mat)
    assert len(ops._stage_records) == 1


def _flip_then_activate(ops: apeSees, mat: Any, *, reflip: bool) -> None:
    with ops.stage(name="plastic") as s:
        s.update_material_stage(materials=[mat], stage=1)
        s.analysis(**_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="install") as s:
        s.activate(pgs=["B"])
        if reflip:
            s.update_material_stage(materials=[mat], stage=1)
        s.analysis(**_chain(ops))
        s.run(n_increments=1)


@pytest.mark.parametrize("mode", ["stock", "auto", None])
def test_activating_an_element_after_the_plastic_flip_refuses(
    mode: "str | None",
) -> None:
    ops, mat = _bridge(mode=mode)
    with pytest.raises(BridgeError) as exc:
        _flip_then_activate(ops, mat, reflip=False)
    msg = str(exc.value)
    assert "SixNodeTri on pg 'B'" in msg and "#2445" in msg
    assert len(ops._stage_records) == 1


def test_reflipping_in_the_activating_stage_passes() -> None:
    ops, mat = _bridge(mode="stock")
    _flip_then_activate(ops, mat, reflip=True)
    assert len(ops._stage_records) == 2


# ---------------------------------------------------------------------------
# The archive and its replay (R1b)
# ---------------------------------------------------------------------------


def r1b_tri6_flip_bridge() -> apeSees:
    """One ``ManzariDafalias`` on two disconnected ``SixNodeTri`` (a real
    ``FEMData``, so the archive replays): an elastic stage, then a stage
    that flips the material to plastic.  An R1b corpus model
    (``tests/opensees/golden/test_roundtrip_oracle.py``)."""
    import numpy as np

    from apeGmsh.mesh._element_types import ElementGroup, make_type_info
    from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
    from apeGmsh.mesh.FEMData import (
        ElementComposite,
        FEMData,
        MeshInfo,
        NodeComposite,
    )

    tri = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.0), (0.5, 0.5),
           (0.0, 0.5)]
    xy = [(x, y) for x, y in tri] + [(x + 3.0, y) for x, y in tri]
    ids = np.arange(1, 13, dtype=np.int64)
    coords = np.array([(x, y, 0.0) for x, y in xy], dtype=np.float64)
    info6 = make_type_info(code=9, gmsh_name="Triangle 6", dim=2, order=2,
                           npe=6, count=2)
    group = ElementGroup(
        element_type=info6, ids=np.array([1, 2], dtype=np.int64),
        connectivity=np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
                              dtype=np.int64),
    )

    def pg(name: str, nodes: "list[int]", eles: "list[int]") -> dict:
        sel = np.array(nodes, dtype=np.int64)
        return {"name": name, "node_ids": sel,
                "node_coords": coords[[i - 1 for i in nodes]],
                "element_ids": np.array(eles, dtype=np.int64)}

    groups = {
        (2, 301): pg("A", [1, 2, 3, 4, 5, 6], [1]),
        (2, 302): pg("B", [7, 8, 9, 10, 11, 12], [2]),
        (0, 303): pg("Base", [1, 2, 4, 7, 8, 10], []),
    }
    fem = FEMData(
        nodes=NodeComposite(node_ids=ids, node_coords=coords,
                            physical=PhysicalGroupSet(groups),
                            labels=LabelSet({})),
        elements=ElementComposite(groups={9: group},
                                  physical=PhysicalGroupSet(groups),
                                  labels=LabelSet({})),
        info=MeshInfo(n_nodes=12, n_elems=2, bandwidth=6, types=[info6]),
        model_name="tri6_flip",
    )
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    sand = ops.nDMaterial.ManzariDafalias(name="sand", **_MD)
    ops.element.SixNodeTri(pg="A", thickness=1.0, material=sand)
    ops.element.SixNodeTri(pg="B", thickness=1.0, material=sand)
    ops.fix(pg="Base", dofs=(1, 1))
    with ops.stage(name="gravity") as s:
        s.update_material_stage(materials=[sand], stage=0)
        s.analysis(**_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="push") as s:
        s.update_material_stage(materials=[sand], stage=1)
        s.analysis(**_chain(ops))
        s.run(n_increments=1)
    return ops


def test_the_replayed_deck_sweeps_too(tmp_path: Any) -> None:
    """The archive keeps the flip as ``(tag, stage)`` rows, as before
    #2446 (so ``model_hash`` does not move), and the replay renders the
    same two lines through the same emitter verb."""
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    ops = r1b_tri6_flip_bridge()
    tag = ops.tag_for(ops._names["sand"])
    deck = tmp_path / "bridge.tcl"
    ops.tcl(str(deck), flat=True)
    archive = tmp_path / "model.h5"
    ops.h5(str(archive))
    model = OpenSeesModel.from_h5(str(archive))
    _gravity, push = model.stages()
    assert push.update_material_stages == ((tag, 1),)
    replayed = model.build("tcl", flat=True)
    text = deck.read_text(encoding="utf-8")
    eles = [ln.split()[2] for ln in text.splitlines()
            if ln.startswith("element tri6n")]
    sweep = (f"setParameter -val 1.0 -ele {' '.join(eles)} materialState "
             f"{tag}")
    assert len(eles) == 2
    assert text.count(sweep) == 1
    assert replayed.count(sweep) == 1


def _deck_tri6_run(tmp_path: Any, *, shared: bool,
                   route: str = "stage") -> list[float]:
    """The bridge's own openseespy deck, run on the resolved backend: two
    SixNodeTri, hydrostatic 100 kPa elastic, a plastic flip, then 40 kPa
    more vertical.  ``shared``: one material on both (the swept flip);
    else each its own (one element per flip, the per-element reference).
    """
    ops = apeSees(_tri6_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mats = [ops.nDMaterial.ManzariDafalias(name=f"sand_{i}", **_MD)
            for i in range(1 if shared else 2)]
    ops.element.SixNodeTri(pg="A", thickness=1.0, material=mats[0])
    ops.element.SixNodeTri(pg="B", thickness=1.0, material=mats[-1])
    # Nodes 13-16 (the stub's quad, unused here) carry no element.
    ops.fix(nodes=[1, 7, 13, 14, 15, 16], dofs=(1, 1))
    ops.fix(nodes=[2, 4, 8, 10], dofs=(0, 1))
    ops.fix(nodes=[3, 6, 9, 12], dofs=(1, 0))

    def chain() -> dict[str, object]:
        return {
            "test": ops.test.NormUnbalance(tol=1e-8, max_iter=100),
            "algorithm": ops.algorithm.Newton(),
            "integrator": ops.integrator.LoadControl(dlam=0.05),
            "constraints": ops.constraints.Transformation(),
            "numberer": ops.numberer.RCM(),
            "system": ops.system.UmfPack(),
            "analysis": ops.analysis.Static(),
        }

    # Consistent loads of a quadratic edge: 1/6, 4/6, 1/6.
    edge = ((2, 1 / 6), (3, 1 / 6), (5, 4 / 6))
    with ops.stage(name="gravity") as s:
        s.update_material_stage(materials=mats, stage=0)
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            for b in (0, 6):
                for n, w in edge:
                    p.load(node=b + n, forces=(-100.0 * w, -100.0 * w))
        s.analysis(**chain())
        s.run(n_increments=20)
    with ops.stage(name="push") as s:
        if route == "stage":
            s.update_material_stage(materials=mats, stage=1)
        else:
            for pg in ("A", "B"):
                s.update_parameter("materialState", 1.0, pg=pg,
                                   material=mats[0])
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            for b in (0, 6):
                for n, w in edge:
                    p.load(node=b + n, forces=(0.0, -40.0 * w))
        s.analysis(**chain())
        s.run(n_increments=20)
    path = tmp_path / f"{'shared' if shared else 'reference'}_{route}.py"
    live = _run_bridge_deck(ops, path)
    try:
        return [d for n in sorted(live.getNodeTags())
                for d in live.nodeDisp(n)]
    finally:
        live.wipe()


def _run_bridge_deck(ops: apeSees, path: Any) -> Any:
    """Write the bridge's openseespy deck and run it on the resolved
    backend; returns that backend with the analysed domain."""
    from apeGmsh.opensees.emitter.live import get_ops

    ops.py(str(path))
    src = path.read_text(encoding="utf-8")
    header = "import openseespy.opensees as ops\n"
    assert src.count(header) == 1
    live = get_ops()
    try:
        exec(compile(src.replace(header, ""), str(path), "exec"),  # noqa: S102
             {"__name__": "deck", "ops": live})
    except BaseException:
        live.wipe()
        raise
    return live


def _deck_dp_neighbour_run(tmp_path: Any, *, stage: int) -> list[float]:
    """The reviewer's #2446 model as a bridge deck: ManzariDafalias on
    SixNodeTri A under load, DruckerPrager (which accepts materialState
    whatever its tag) on SixNodeTri B, driven in plastic shear by
    prescribed displacements in the stage that flips the sand to
    ``stage``.  Returns B's stresses: B is fully prescribed, so the sand
    cannot reach it except through a flip that names it."""
    ops = apeSees(_tri6_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sand = ops.nDMaterial.ManzariDafalias(name="sand", **_MD)
    clay = ops.nDMaterial.DruckerPrager(
        K=1e5, G=5e4, sigmaY=10.0, rho=0.0, rhoBar=0.0, Kinf=10.0, Ko=10.0,
        delta1=0.0, delta2=0.0, H=1000.0, theta=1.0, name="clay")
    ops.element.SixNodeTri(pg="A", thickness=1.0, material=sand)
    ops.element.SixNodeTri(pg="B", thickness=1.0, material=clay)
    ops.fix(nodes=[1, 7, 8, 10, 13, 14, 15, 16], dofs=(1, 1))
    ops.fix(nodes=[2, 4], dofs=(0, 1))
    ops.fix(nodes=[3, 6], dofs=(1, 0))

    def chain() -> dict[str, object]:
        return {
            "test": ops.test.NormUnbalance(tol=1e-8, max_iter=100),
            "algorithm": ops.algorithm.Newton(),
            "integrator": ops.integrator.LoadControl(dlam=0.01),
            "constraints": ops.constraints.Transformation(),
            "numberer": ops.numberer.RCM(),
            "system": ops.system.UmfPack(),
            "analysis": ops.analysis.Static(),
        }

    edge = ((2, 1 / 6), (3, 1 / 6), (5, 4 / 6))
    with ops.stage(name="gravity") as s:
        s.update_material_stage(materials=[sand], stage=0)
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            for n, w in edge:
                p.load(node=n, forces=(-100.0 * w, -100.0 * w))
        s.analysis(**chain())
        s.run(n_increments=100)
    with ops.stage(name="push") as s:
        s.update_material_stage(materials=[sand], stage=stage)
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            for n, w in edge:
                p.load(node=n, forces=(0.0, -40.0 * w))
            for n, y in ((9, 1.0), (11, 0.5), (12, 0.5)):
                p.sp(node=n, dof=1, value=0.02 * y)
                p.sp(node=n, dof=2, value=-0.002 * y)
        s.analysis(**chain())
        s.run(n_increments=100)
    live = _run_bridge_deck(ops, tmp_path / f"dp_{stage}.py")
    try:
        (b,) = [e for e in live.getEleTags() if 9 in live.eleNodes(e)]
        return list(live.eleResponse(b, "stresses"))
    finally:
        live.wipe()


@pytest.mark.live
def test_a_druckerprager_neighbour_is_not_switched(tmp_path: Any) -> None:
    """The reviewer's model: ManzariDafalias on SixNodeTri A, DruckerPrager
    on SixNodeTri B, the sand flipped plastic.  DruckerPrager accepts
    materialState without checking the tag, so a flip that reached B
    switched it to elastic (fd836151d: shear 12.36 became 1000.0).  B must
    answer exactly as when the sand stays elastic (stock and fork)."""
    pytest.importorskip("openseespy.opensees")
    flipped = _deck_dp_neighbour_run(tmp_path, stage=1)
    elastic = _deck_dp_neighbour_run(tmp_path, stage=0)
    scale = max(abs(v) for v in elastic)
    assert scale > 1.0
    assert _diff(flipped, elastic) <= 1e-10 * scale


@pytest.mark.live
def test_the_bridge_deck_initialises_every_element(tmp_path: Any) -> None:
    """End to end: the bridge's deck with one shared material matches the
    deck that gives each element its own (stock and fork)."""
    pytest.importorskip("openseespy.opensees")
    ref = _deck_tri6_run(tmp_path, shared=False)
    swept = _deck_tri6_run(tmp_path, shared=True)
    scale = max(abs(v) for v in ref)
    assert scale > 1e-4
    assert _diff(swept, ref) <= 1e-10 * max(1.0, scale)


@pytest.mark.live
def test_the_materialstate_parameter_route_initialises_every_element(
    tmp_path: Any,
) -> None:
    """``s.update_parameter("materialState", 1, material=m)`` over every
    element carrying ``m`` passes the stage close (#2442 refused it) and its
    per-element ``addToParameter`` deck matches the reference."""
    pytest.importorskip("openseespy.opensees")
    ref = _deck_tri6_run(tmp_path, shared=False)
    routed = _deck_tri6_run(tmp_path, shared=True, route="materialState")
    assert _diff(routed, ref) <= 1e-10 * max(1.0, max(abs(v) for v in ref))


# ---------------------------------------------------------------------------
# The live census
# ---------------------------------------------------------------------------

_TRI3 = [(0, 0), (1, 0), (0, 1)]
_TRI6 = _TRI3 + [(0.5, 0), (0.5, 0.5), (0, 0.5)]
_QUAD4 = [(0, 0), (1, 0), (1, 1), (0, 1)]
_TET4 = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]


def _mid(pts: "list[tuple[int, ...]]", a: int, b: int) -> tuple[float, ...]:
    return tuple((x + y) / 2 for x, y in zip(pts[a], pts[b]))


_TET10 = _TET4 + [_mid(_TET4, a, b)
                  for a, b in ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))]
_HEX8 = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]

_PS = "PlaneStrain"
#: apeGmsh class -> (Tcl token, ndm, node coordinates, argv after the tag).
_CLASSES: dict[str, Any] = {
    "SixNodeTri": ("tri6n", 2, _TRI6, lambda n, m: (*n, 1.0, _PS, m)),
    "BezierTri6": ("BezierTri6", 2, _TRI6, lambda n, m: (*n, 1.0, _PS, m)),
    "FourNodeTetrahedron": ("FourNodeTetrahedron", 3, _TET4,
                            lambda n, m: (*n, m)),
    "BezierTet10": ("BezierTet10", 3, _TET10, lambda n, m: (*n, m)),
    # No bridge primitive emits these two yet (element/solid.py defers
    # them); the census drives the raw tokens the future primitive emits.
    "SSPquad": ("SSPquad", 2, _QUAD4, lambda n, m: (*n, m, _PS, 1.0)),
    "SSPbrick": ("SSPbrick", 3, _HEX8, lambda n, m: (*n, m)),
    "FourNodeQuad": ("quad", 2, _QUAD4, lambda n, m: (*n, 1.0, _PS, m)),
    "Tri31": ("tri31", 2, _TRI3, lambda n, m: (*n, 1.0, _PS, m)),
    "stdBrick": ("stdBrick", 3, _HEX8, lambda n, m: (*n, m)),
}

#: The table's classes, and whether their census needs the fork.
_FORWARDING = [
    ("SixNodeTri", "both"), ("FourNodeTetrahedron", "both"),
    ("SSPquad", "both"), ("SSPbrick", "both"),
    ("BezierTri6", "fork"), ("BezierTet10", "fork"),
]
#: Upstream classes that drop the sweep (on stock and at FORK_PIN alike).
_DROPPING = ["FourNodeQuad", "Tri31", "stdBrick"]


def _census(cls: str, how: str, mat: str = "ManzariDafalias",
            wrapper: "tuple[str, tuple[Any, ...]] | None" = None,
            ) -> list[float]:
    """Two elements of ``cls``, every DOF on a homogeneous strain path to
    t=1 elastic, flip, then on to t=2; returns their stresses.
    ``wrapper``: ``(token, argv after the wrapped tag)``; each element then
    carries that wrapper (tag ``10 + t``) over the material (tag ``t``)."""
    # The resolved backend, not ``openseespy``: the live-fork lane installs
    # the fork as ``opensees`` with no openseespy, where an importorskip of
    # it skipped every fork row (#2457).
    from apeGmsh.opensees.emitter.live import get_ops

    try:
        backend = get_ops()
    except ImportError as exc:
        pytest.skip(f"no OpenSees backend resolves: {exc}")
    ops = Deck(backend)
    token, ndm, coords, argv = _CLASSES[cls]
    eps = (-1e-2, -1.5e-2, 0.0)[:ndm]
    ops.wipe()
    try:
        ops.model("basic", "-ndm", ndm, "-ndf", ndm)
        tags = [1, 2] if how == "reference" else [1, 1]
        for t in sorted(set(tags)):
            ops.nDMaterial(mat, t, *MD_ARGS)
            if wrapper is not None:
                ops.nDMaterial(wrapper[0], 10 + t, t, *wrapper[1])
        ops.timeSeries("Path", 1, "-time", 0.0, 1.0, 2.0,
                       "-values", 0.0, 1.0, 2.0)
        ops.pattern("Plain", 1, 1)
        for q, t in enumerate(tags):
            nodes = []
            for i, c in enumerate(coords):
                n = 100 * q + i + 1
                xyz = [float(v) for v in c]
                ops.node(n, xyz[0] + 3.0 * q, *xyz[1:])
                nodes.append(n)
                for d in range(ndm):
                    ops.sp(n, d + 1, eps[d] * xyz[d])
            carried = t if wrapper is None else 10 + t
            ops.element(token, q + 1, *argv(nodes, carried))
        for t in sorted(set(tags)):
            ops.updateMaterialStage("-material", t, "-stage", 0)
        ops.constraints("Penalty", 1.0e12, 1.0e12)
        ops.numberer("Plain")
        ops.system("FullGeneral")
        ops.test("NormUnbalance", 1e-8, 20, 0)
        ops.algorithm("Linear")
        ops.integrator("LoadControl", 0.05)
        ops.analysis("Static")
        assert ops.analyze(20) == 0
        if how == "emitted":
            ops.emitted_flip(1)
        else:
            for t in sorted(set(tags)):
                ops.updateMaterialStage("-material", t, "-stage", 1)
        assert ops.analyze(20) == 0
        return [
            s for e in (1, 2)
            for s in (ops.eleResponse(e, "stresses")
                      or ops.eleResponse(e, "stress"))
        ]
    finally:
        ops.wipe()


def _forwarding_params() -> list[Any]:
    out = []
    for cls, builds in _FORWARDING:
        marks = [pytest.mark.live]
        if builds == "fork":
            marks.append(pytest.mark.ladruno_fork)
        out.append(pytest.param(cls, marks=marks, id=cls))
    return out


def _diff(a: "list[float]", b: "list[float]") -> float:
    assert len(a) == len(b)
    return max(abs(x - y) for x, y in zip(a, b))


@pytest.mark.parametrize("cls", _forwarding_params())
def test_census_the_sweep_initialises_every_element(cls: str) -> None:
    ref = _census(cls, "reference")
    emitted = _census(cls, "emitted")
    first = _census(cls, "first")
    scale = max(1.0, max(abs(v) for v in ref))
    assert _diff(emitted, ref) <= 1e-10 * scale
    assert _diff(first, ref) > 1e-8 * scale, "the case must tell them apart"


@pytest.mark.live
@pytest.mark.parametrize("cls", _DROPPING)
def test_census_a_refused_class_drops_the_sweep(cls: str) -> None:
    ref = _census(cls, "reference")
    emitted = _census(cls, "emitted")
    first = _census(cls, "first")
    assert _diff(emitted, first) == 0.0
    assert _diff(emitted, ref) > 0.0
    assert cls not in MATERIAL_STATE_ELEMENTS


# ---------------------------------------------------------------------------
# A wrapped material (StagedStrain, PlaneStrain, ...)
# ---------------------------------------------------------------------------

#: Each :data:`MATERIAL_STATE_WRAPPERS` token over ManzariDafalias, as
#: ``(token, argv after the wrapped tag, element class, build)``: the
#: census element it runs on, and whether it needs the fork.
_WRAPPERS = [
    ("PlaneStrain", (), "SixNodeTri", "both"),
    ("StagedStrain", ("-noInit",), "FourNodeTetrahedron", "fork"),
]


def test_every_wrapper_has_a_census_row() -> None:
    assert {w for w, *_rest in _WRAPPERS} == set(MATERIAL_STATE_WRAPPERS)


def _wrapper_params() -> list[Any]:
    out = []
    for token, argv, cls, builds in _WRAPPERS:
        marks = [pytest.mark.live]
        if builds == "fork":
            marks.append(pytest.mark.ladruno_fork)
        out.append(pytest.param(token, argv, cls, marks=marks, id=token))
    return out


@pytest.mark.parametrize(("token", "argv", "cls"), _wrapper_params())
def test_census_a_wrapped_material_is_initialised_on_every_element(
    token: str, argv: tuple[Any, ...], cls: str,
) -> None:
    """The element carries the wrapper's tag, and the update still names
    it: it reaches the wrapped ManzariDafalias of every element, as the
    per-element reference does."""
    ref = _census(cls, "reference", wrapper=(token, argv))
    emitted = _census(cls, "emitted", wrapper=(token, argv))
    first = _census(cls, "first", wrapper=(token, argv))
    scale = max(1.0, max(abs(v) for v in ref))
    assert _diff(emitted, ref) <= 1e-10 * scale
    assert _diff(first, ref) > 1e-8 * scale, "the case must tell them apart"


def test_the_wrapper_table() -> None:
    assert MATERIAL_STATE_WRAPPERS == {"PlaneStrain": 0, "StagedStrain": 0}


def test_a_wrapper_resolves_to_the_material_it_wraps() -> None:
    c = MaterialStageCarriers()
    c.note_material("ManzariDafalias", 7, MD_ARGS)
    c.note_material("StagedStrain", 17, (7, "-noInit"))
    c.note_material("PlaneStrain", 27, (17,))
    c.note_element("tri6n", 1, (*_TRI6_ARGV, 17))
    c.note_element("tri6n", 2, (*_TRI6_ARGV, 27))
    assert c.flip_commands(7, 1)[0][1][3:5] == (1, 2)


def test_an_unknown_wrapper_is_not_resolved() -> None:
    """PlateFiber is not a MATERIAL_STATE_WRAPPERS wrapper: its elements are
    not listed (the stage close refuses such a flip by name)."""
    c = MaterialStageCarriers()
    c.note_material("ManzariDafalias", 7, MD_ARGS)
    c.note_material("PlateFiber", 17, (7,))
    c.note_element("tri6n", 1, (*_TRI6_ARGV, 17))
    c.note_element("tri6n", 2, (*_TRI6_ARGV, 17))
    assert c.flip_commands(7, 1) == (_ums(7, 1),)


def test_a_wrapper_grammar_without_the_tag_fails_loud() -> None:
    with pytest.raises(BridgeError, match="no wrapped material tag"):
        MaterialStageCarriers().note_material("StagedStrain", 17, ("-noInit",))


def _wrapped_bridge(wrap: str) -> tuple[apeSees, Any]:
    ops = apeSees(_tri6_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sand = ops.nDMaterial.ManzariDafalias(name="sand", **_MD)
    if wrap == "PlaneStrain":
        outer = ops.nDMaterial.PlaneStrain(base=sand)
    elif wrap == "LogStrain":
        outer = ops.nDMaterial.LogStrain(inner=sand)
    else:
        outer = ops.nDMaterial.PlateFiber(material=sand)
    for pg in ("A", "B"):
        ops.element.SixNodeTri(pg=pg, thickness=1.0, material=outer)
    ops.fix(pg="Base", dofs=(1, 1))
    return ops, sand


def test_the_stage_close_and_the_deck_agree_on_a_wrapper() -> None:
    """A forwarding wrapper passes the stage close, and the deck names
    both elements; the two read one table (MATERIAL_STATE_WRAPPERS)."""
    ops, sand = _wrapped_bridge("PlaneStrain")
    _flip(ops, sand)
    tcl = TclEmitter()
    ops.build().emit(tcl)
    eles = [ln.split()[2] for ln in tcl.lines()
            if ln.startswith("element tri6n")]
    assert (f"setParameter -val 1.0 -ele {' '.join(eles)} materialState "
            f"{ops.tag_for(sand)}") in tcl.lines()


@pytest.mark.parametrize("wrap", ["PlateFiber", "LogStrain"])
def test_an_unproven_wrapper_is_refused_by_name(wrap: str) -> None:
    """PlateFiber is not known to forward; LogStrain forwards in the fork
    source, but over ManzariDafalias it ends the process on the measured
    build, so it stays refused until measured."""
    ops, sand = _wrapped_bridge(wrap)
    with pytest.raises(BridgeError) as exc:
        _flip(ops, sand)
    msg = str(exc.value)
    assert f"through {wrap}" in msg and "#2445" in msg


# ---------------------------------------------------------------------------
# The live emitter (no carrier sweep): it refuses the flip (#2457)
# ---------------------------------------------------------------------------


class _RecordingOps:
    """A stub backend: records every call, drives no domain."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def __getattr__(self, name: str) -> Any:
        def call(*args: Any) -> None:
            self.calls.append((name, args))
        return call


def _live() -> tuple[Any, _RecordingOps]:
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    le = LiveOpsEmitter.__new__(LiveOpsEmitter)
    ops = _RecordingOps()
    le._ops = ops  # type: ignore[assignment]
    le._in_partition = False
    return le, ops


def _live_flips(ops: _RecordingOps) -> list[tuple[Any, ...]]:
    return [a for name, a in ops.calls if name == "updateMaterialStage"]


@pytest.mark.parametrize("cls", sorted(FIRST_ELEMENT_ONLY_PLASTIC_INIT))
def test_the_live_emitter_refuses_a_swept_flip_on_two_carriers(cls: str) -> None:
    le, ops = _live()
    le.nDMaterial(cls, 7, *MD_ARGS)
    le.element("tri6n", 1, *_TRI6_ARGV, 7)
    le.element("tri6n", 2, *_TRI6_ARGV, 7)
    with pytest.raises(BridgeError) as exc:
        le.update_material_stage(7, 1)
    msg = str(exc.value)
    assert "#2445" in msg and "bare updateMaterialStage" in msg
    assert "2 elements may carry" in msg
    assert not _live_flips(ops)


def test_the_live_emitter_passes_what_needs_no_sweep() -> None:
    le, ops = _live()
    le.nDMaterial("ManzariDafalias", 7, *MD_ARGS)
    le.nDMaterial("PM4Sand", 8, 0.5, 0.0, 400.0, 1.0)
    le.element("tri6n", 1, *_TRI6_ARGV, 7)
    le.element("tri6n", 2, *_TRI6_ARGV, 7)
    le.element("tri6n", 3, *_TRI6_ARGV, 8)
    le.element("tri6n", 4, *_TRI6_ARGV, 8)
    le.update_material_stage(7, 0)  # an elastic flip
    le.update_material_stage(8, 1)  # a class that is not swept
    le.remove_element(2)
    le.update_material_stage(7, 1)  # one carrier left
    assert _live_flips(ops) == [
        ("-material", 7, "-stage", 0), ("-material", 8, "-stage", 1),
        ("-material", 7, "-stage", 1)]


def test_the_live_emitter_counts_carriers_without_a_grammar() -> None:
    """A FourNodeQuad (no MATERIAL_STATE_ELEMENT_TOKENS entry) and an element
    on a wrapper outside MATERIAL_STATE_WRAPPERS still count."""
    le, _ops = _live()
    le.nDMaterial("ManzariDafalias", 7, *MD_ARGS)
    le.nDMaterial("PlateFiber", 17, 7)
    le.element("quad", 1, 11, 12, 13, 14, 1.0, "PlaneStrain", 7)
    le.element("tri6n", 2, 21, 22, 23, 24, 25, 26, 1.0, "PlaneStrain", 17)
    with pytest.raises(BridgeError, match="#2445"):
        le.update_material_stage(7, 1)


def test_the_live_emitter_follows_a_shell_section() -> None:
    """The review's model: MD -> PlateFiber -> LayeredShell -> two
    ShellMITC4 elements on the section."""
    le, ops = _live()
    le.nDMaterial("ManzariDafalias", 7, *MD_ARGS)
    le.nDMaterial("PlateFiber", 17, 7)
    le.section("LayeredShell", 30, 3, 17, 0.1, 17, 0.1, 17, 0.1)
    le.element("ShellMITC4", 1, 101, 102, 103, 104, 30)
    le.element("ShellMITC4", 2, 201, 202, 203, 204, 30)
    with pytest.raises(BridgeError, match="2 elements may carry"):
        le.update_material_stage(7, 1)
    assert not _live_flips(ops)


@pytest.mark.parametrize("member", ["fiber", "patch", "layer"])
def test_the_live_emitter_follows_a_fiber_section(member: str) -> None:
    """MD named by a fiber-section member, the section by a beam
    integration, the integration by two beam elements."""
    le, _ops = _live()
    le.nDMaterial("ManzariDafalias", 7, *MD_ARGS)
    le.section_open("NDFiber", 40)
    if member == "fiber":
        le.fiber(0.0, 0.0, 0.01, 7)
    elif member == "patch":
        le.patch("rect", 7, 2, 2, -0.1, -0.1, 0.1, 0.1)
    else:
        le.layer("straight", 7, 3, 0.01, -0.1, 0.0, 0.1, 0.0)
    le.section_close()
    le.beamIntegration("Lobatto", 50, 40, 3)
    le.element("dispBeamColumn", 1, 101, 102, 60, 50)
    le.element("dispBeamColumn", 2, 201, 202, 60, 50)
    with pytest.raises(BridgeError, match="2 elements may carry"):
        le.update_material_stage(7, 1)


def test_the_live_emitter_reads_numpy_integer_tags() -> None:
    np = pytest.importorskip("numpy")
    le, _ops = _live()
    le.nDMaterial("ManzariDafalias", np.int64(7), *MD_ARGS)
    le.element("tri6n", 1, *_TRI6_ARGV, np.int64(7))
    le.element("tri6n", 2, *_TRI6_ARGV, np.int32(7))
    with pytest.raises(BridgeError, match="2 elements may carry"):
        le.update_material_stage(7, 1)


def _shell_fem() -> FEMStub:
    """Two disconnected quads ``S`` (eids 1, 2), base nodes fixed."""
    coords = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
              (3, 0, 0), (4, 0, 0), (4, 1, 0), (3, 1, 0)]
    return FEMStub(
        nodes=_NodesStub(
            ids=list(range(1, 9)), coords=coords,
            node_pgs={"S": list(range(1, 9)), "Base": [1, 2, 5, 6]}),
        elements=_ElementsStub(elem_pgs={"S": _ElementGroupView(
            ids=(1, 2), connectivity=((1, 2, 3, 4), (5, 6, 7, 8)))}),
    )


def test_a_shell_section_carrier_is_refused_at_the_stage_close() -> None:
    """The deck path for the same model: ShellMITC4 is not a
    MATERIAL_STATE_ELEMENTS class, so the stage close refuses the flip by
    name instead of emitting a sweep that misses the shells."""
    from apeGmsh.opensees.section.plate import ShellLayer

    ops = apeSees(_shell_fem(), default_orientation=None)
    ops.model(ndm=3, ndf=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sand = ops.nDMaterial.ManzariDafalias(name="sand", **_MD)
    plate = ops.nDMaterial.PlateFiber(material=sand)
    sec = ops.section.LayeredShell(layers=tuple(
        ShellLayer(material=plate, thickness=0.1) for _ in range(3)))
    ops.element.ShellMITC4(pg="S", section=sec)
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    with pytest.raises(BridgeError) as exc:
        _flip(ops, sand, pgs=("S",))
    msg = str(exc.value)
    assert "ShellMITC4 carries it" in msg and "#2445" in msg
    assert f"nDMaterial tag {ops.tag_for(sand)} (2 elements)" in msg
    assert not ops._stage_records
