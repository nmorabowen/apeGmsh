"""ADR 0115 D4-D6 (R2-a, #1651): ``label=`` as a node selector,
``ops.nodes.one``, the same-tier homogeneous fix merge, and the duplicate
fix / mass validator (V2) on every build, staged or not.

The model is a real meshed plate (a "footing"), 4 x 2 quads, whose left
and bottom edges carry both a geometry label and a physical group over
the same nodes, so ``label=`` and ``pg=`` can be compared byte for byte
(Invariant 6). The two edges share the corner node at the origin: the
shared corner every merge test overlaps on.
"""
from __future__ import annotations

from typing import Any

import gmsh
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees._internal.build import BridgeError, FixRecord
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.node import Node, NodeSet, _resolve_label

from tests.opensees.unit.test_unconsumed_model_definition import (
    _bc,
    _calls,
    _emit_silent,
    _model,
    _ops,
)


# ---------------------------------------------------------------------------
# The footing
# ---------------------------------------------------------------------------


def build_footing_fem() -> Any:
    """A 2 x 1 plate, 4 x 2 quads: labels ``left`` / ``bottom`` and PGs
    ``Left`` / ``Bottom`` on the same edges, a PG ``Right`` with no label
    of that name, and a label ``corner`` on the origin point."""
    with apeGmsh(model_name="r2a_footing", verbose=False) as g:
        g.model.geometry.add_rectangle(0, 0, 0, 2.0, 1.0, label="plate")
        g.physical.add_surface("plate", name="Plate")
        surf = gmsh.model.getEntities(2)[0][1]
        edges: dict[str, int] = {}
        for _d, c in gmsh.model.getBoundary([(2, surf)], oriented=False):
            x0, y0, _z0, x1, _y1, _z1 = gmsh.model.getBoundingBox(1, abs(c))
            horizontal = abs(x1 - x0) > 1e-3
            gmsh.model.mesh.setTransfiniteCurve(abs(c), 5 if horizontal else 3)
            if not horizontal:
                edges["left" if x0 < 1e-6 else "right"] = abs(c)
            elif y0 < 1e-6:
                edges["bottom"] = abs(c)
        gmsh.model.mesh.setTransfiniteSurface(surf)
        gmsh.model.mesh.setRecombine(2, surf)
        origin = next(
            t for _d, t in gmsh.model.getEntities(0)
            if max(abs(v) for v in gmsh.model.getValue(0, t, [])) < 1e-9
        )
        g.labels.add(1, [edges["left"]], "left")
        g.physical.add(1, [edges["left"]], name="Left")
        g.labels.add(1, [edges["bottom"]], "bottom")
        g.physical.add(1, [edges["bottom"]], name="Bottom")
        g.physical.add(1, [edges["right"]], name="Right")
        g.labels.add(0, [origin], "corner")
        g.mesh.generation.generate(dim=2)
        return g.mesh.queries.get_fem_data(dim=2)


@pytest.fixture(scope="module")
def fem() -> Any:
    return build_footing_fem()


def footing(fem: Any) -> apeSees:
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Plate", thickness=1.0, material=mat)
    return ops


def chain(ops: apeSees) -> dict[str, object]:
    return {
        "test":        ops.test.NormDispIncr(tol=1e-8, max_iter=10),
        "algorithm":   ops.algorithm.Newton(),
        "integrator":  ops.integrator.LoadControl(dlam=1.0),
        "constraints": ops.constraints.Plain(),
        "numberer":    ops.numberer.RCM(),
        "system":      ops.system.BandGeneral(),
        "analysis":    ops.analysis.Static(),
    }


def _deck(ops: apeSees) -> list[str]:
    emitter = TclEmitter()
    ops.build().emit(emitter)
    return emitter.lines()


def _fix_lines(lines: list[str]) -> list[str]:
    return [ln for ln in lines if ln.startswith("fix ")]


def _sp_pairs(lines: list[str]) -> list[tuple[int, int]]:
    """Every ``(node, DOF)`` a ``fix`` line constrains, repeats kept."""
    out: list[tuple[int, int]] = []
    for ln in _fix_lines(lines):
        _verb, node, *flags = ln.split()
        out += [(int(node), d) for d, f in enumerate(flags, 1) if int(f)]
    return out


def corner_of(fem: Any) -> int:
    return int(fem.nodes.select(label="corner").ids[0])


# ---------------------------------------------------------------------------
# D4: label= emits what pg= emits over the same nodes (Invariant 6)
# ---------------------------------------------------------------------------


def test_label_and_pg_select_the_same_nodes(fem: Any) -> None:
    # The premise of every byte comparison below.
    assert list(fem.nodes.select(label="left").ids) == list(
        fem.nodes.select(pg="Left").ids)
    assert list(fem.nodes.select(label="bottom").ids) == list(
        fem.nodes.select(pg="Bottom").ids)


def test_fix_mass_and_node_recorder_by_label_emit_the_pg_bytes(fem: Any) -> None:
    by_label, by_pg = footing(fem), footing(fem)
    by_label.fix(label="left", dofs=(1, 1))
    by_pg.fix(pg="Left", dofs=(1, 1))
    by_label.mass(label="bottom", values=(2.0, 2.0))
    by_pg.mass(pg="Bottom", values=(2.0, 2.0))
    by_label.recorder.Node(file="d.out", response="disp", label="bottom",
                           dofs=(1, 2))
    by_pg.recorder.Node(file="d.out", response="disp", pg="Bottom",
                        dofs=(1, 2))
    deck = _deck(by_label)
    assert deck == _deck(by_pg)
    assert len(_fix_lines(deck)) == 3


def test_stage_fix_and_mass_by_label_emit_the_pg_bytes(fem: Any) -> None:
    decks = []
    for sel in ({"label": "left"}, {"pg": "Left"}):
        ops = footing(fem)
        with ops.stage(name="s1") as s:
            s.fix(dofs=(1, 1), **sel)
            s.mass(values=(3.0, 3.0), **sel)
            s.analysis(**chain(ops))
            s.run(n_increments=1)
        decks.append(_deck(ops))
    assert decks[0] == decks[1]


def test_a_label_list_unions_its_labels_in_first_seen_order(fem: Any) -> None:
    ops = footing(fem)
    ops.fix(label=["left", "bottom"], dofs=(1, 1))
    want = dict.fromkeys(
        [*fem.nodes.select(label="left").ids,
         *fem.nodes.select(label="bottom").ids])
    assert ops._fix_records[0] == FixRecord(
        pg=None, nodes=tuple(int(n) for n in want), dofs=(1, 1))


# ---------------------------------------------------------------------------
# D4: refusals, all before anything registers
# ---------------------------------------------------------------------------


_BAD_SELECTORS: list[tuple[dict[str, Any], type[Exception], str]] = [
    ({"label": "nope"}, BridgeError, r"no label 'nope'.*Labels that exist: "
                                     r"\['bottom', 'corner', 'left', 'plate'\]"),
    # a PG named Right exists; a label never falls back to it
    ({"label": "Right"}, BridgeError, r"never falls back to a physical group"),
    ({"label": ""}, ValueError, r"non-blank string"),
    ({"label": []}, ValueError, r"label= is empty"),
    ({"label": ["left", 3]}, ValueError, r"non-blank string"),
    ({"label": 7}, ValueError, r"string or a list of strings"),
    ({"label": "left", "pg": "Left"}, ValueError, r"exactly one of"),
]


@pytest.mark.parametrize(("sel", "exc", "match"), _BAD_SELECTORS)
def test_a_bad_selector_is_refused_before_anything_registers(
    fem: Any, sel: dict[str, Any], exc: type[Exception], match: str,
) -> None:
    ops = footing(fem)
    n_prims = len(ops._primitives)
    with pytest.raises(exc, match=match):
        ops.fix(dofs=(1, 1), name="f", **sel)
    with pytest.raises(exc, match=match):
        ops.mass(values=(1.0, 1.0), name="m", **sel)
    with pytest.raises(exc, match=match):
        ops.recorder.Node(file="d.out", response="disp", dofs=(1,),
                          name="r", **sel)
    assert len(ops._primitives) == n_prims  # no recorder registered
    with ops.stage(name="s1") as s:
        with pytest.raises(exc, match=match):
            s.fix(dofs=(1, 1), name="sf", **sel)
        with pytest.raises(exc, match=match):
            s.mass(values=(1.0, 1.0), name="sm", **sel)
        s.analysis(**chain(ops))
        s.run(n_increments=1)
    assert ops._fix_records == [] and ops._mass_records == []
    assert ops._stage_records[0].fix_records == ()
    assert ops._stage_records[0].mass_records == ()
    # No declaration key was consumed: every name is still free.
    ops.fix(pg="Left", dofs=(1, 1), name="f")
    ops.mass(pg="Left", values=(1.0, 1.0), name="m")
    ops.recorder.Node(file="d.out", response="disp", pg="Left", dofs=(1,),
                      name="r")
    ops.fix(pg="Right", dofs=(1, 1), name="sf")
    ops.mass(pg="Right", values=(1.0, 1.0), name="sm")


def test_fix_and_mass_need_a_selector(fem: Any) -> None:
    ops = footing(fem)
    want = r"exactly one of pg= or nodes= \(or label=\)"
    with pytest.raises(ValueError, match=want):
        ops.fix(dofs=(1, 1))
    with pytest.raises(ValueError, match=want):
        ops.mass(values=(1.0, 1.0))


def test_a_label_with_no_nodes_is_refused() -> None:
    class _Labels:
        def __contains__(self, name: object) -> bool:
            return name == "empty"

        def names(self) -> list[str]:
            return ["empty"]

    class _Sel:
        ids: list[int] = []

    class _Nodes:
        labels = _Labels()

        def select(self, *, label: object) -> _Sel:
            return _Sel()

    class _Fem:
        nodes = _Nodes()

    with pytest.raises(BridgeError, match=r"label='empty' selects 0 nodes"):
        _resolve_label(_Fem(), "empty", where="here")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# D2 / D5: ops.nodes.get(label=) is a set; ops.nodes.one is a Node
# ---------------------------------------------------------------------------


def test_nodes_get_by_label_is_always_a_set(fem: Any) -> None:
    ops = footing(fem)
    got = ops.nodes.get(label="left")
    assert isinstance(got, NodeSet)
    assert got.tags == tuple(int(n) for n in fem.nodes.select(label="left").ids)
    one = ops.nodes.get(label="corner")
    assert isinstance(one, NodeSet) and one.tags == (corner_of(fem),)
    with pytest.raises(ValueError, match="at most one of tag=, pg= or label="):
        ops.nodes.get(pg="Left", label="left")


def test_nodes_one_returns_the_single_node(fem: Any) -> None:
    ops = footing(fem)
    node = ops.nodes.one(label="corner")
    assert isinstance(node, Node) and node.tag == corner_of(fem)
    assert node.coords == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("sel", [{"pg": "Left"}, {"label": "left"}])
def test_nodes_one_on_three_nodes_names_the_count(
    fem: Any, sel: dict[str, str],
) -> None:
    ops = footing(fem)
    with pytest.raises(BridgeError, match="selects 3 nodes, not exactly one"):
        ops.nodes.one(**sel)


def test_nodes_one_on_zero_nodes_names_the_count() -> None:
    from tests.opensees.fixtures.fem_stub import (
        FEMStub, _ElementsStub, _NodesStub,
    )
    stub = FEMStub(
        nodes=_NodesStub(ids=[1], coords=[(0.0, 0.0, 0.0)],
                         node_pgs={"Empty": []}),
        elements=_ElementsStub(elem_pgs={}),
    )
    ops = apeSees(stub)  # type: ignore[arg-type]
    with pytest.raises(BridgeError, match="selects 0 nodes, not exactly one"):
        ops.nodes.one(pg="Empty")


def test_nodes_one_needs_exactly_one_selector(fem: Any) -> None:
    ops = footing(fem)
    with pytest.raises(ValueError, match="exactly one of pg= or label="):
        ops.nodes.one()
    with pytest.raises(ValueError, match="exactly one of pg= or label="):
        ops.nodes.one(pg="Left", label="left")


# ---------------------------------------------------------------------------
# D6: identical homogeneous fixes merge within a tier (Invariant 8)
# ---------------------------------------------------------------------------


def test_unstaged_overlapping_fixes_merge_to_one_sp_per_dof(fem: Any) -> None:
    """The shared corner is fixed by both edges. With the merge reverted
    this raises: V2 now runs on an unstaged emit too."""
    ops = footing(fem)
    ops.fix(label="left", dofs=(1, 1))
    ops.fix(pg="Bottom", dofs=(1, 1))
    lines = _deck(ops)
    pairs = _sp_pairs(lines)
    assert len(pairs) == len(set(pairs))
    corner = corner_of(fem)
    assert [ln for ln in _fix_lines(lines) if ln.split()[1] == str(corner)] \
        == [f"fix {corner} 1 1"]
    edge_nodes = {int(n) for n in fem.nodes.select(label="left").ids} | {
        int(n) for n in fem.nodes.select(label="bottom").ids}
    assert set(pairs) == {(n, d) for n in edge_nodes for d in (1, 2)}


def test_a_later_fix_emits_only_the_dofs_it_adds(fem: Any) -> None:
    ops = footing(fem)
    ops.fix(label="left", dofs=(1, 0))
    ops.fix(label="bottom", dofs=(1, 1))
    corner = corner_of(fem)
    mine = [ln for ln in _fix_lines(_deck(ops)) if ln.split()[1] == str(corner)]
    assert mine == [f"fix {corner} 1 0", f"fix {corner} 0 1"]


def test_a_record_without_overlap_is_kept_as_is(fem: Any) -> None:
    """Invariant 7: no overlap, no rewrite (the same record objects)."""
    ops = footing(fem)
    ops.fix(pg="Left", dofs=(1, 1))
    ops.fix(pg="Right", dofs=(1, 1))
    bm = ops.build()
    assert len(bm.fix_records) == 2
    assert all(a is b for a, b in zip(bm.fix_records, ops._fix_records))


def test_overlapping_fixes_within_one_stage_merge(fem: Any) -> None:
    ops = footing(fem)
    with ops.stage(name="s1") as s:
        s.fix(label="left", dofs=(1, 1))
        s.fix(label="bottom", dofs=(1, 1))
        s.analysis(**chain(ops))
        s.run(n_increments=1)
    pairs = _sp_pairs(_deck(ops))
    assert len(pairs) == len(set(pairs)) and (corner_of(fem), 1) in pairs


def test_fix_from_model_merges_with_an_explicit_fix() -> None:
    """Q1: the snapshot's supports and an explicit fix share the global
    tier, so the overlapping DOF emits once."""
    ops = _ops(_model(sp=_bc([1, 2], (1, 1, 1))))
    ops.fix(nodes=[1, 7], dofs=(1, 0, 0))
    ops.fix_from_model()
    ops.mass_from_model()
    calls = _calls(_emit_silent(ops), "fix")
    assert sorted(calls) == [(1, 0, 1, 1), (1, 1, 0, 0), (2, 1, 1, 1),
                             (7, 1, 0, 0)]


# ---------------------------------------------------------------------------
# D6: what still raises
# ---------------------------------------------------------------------------


def test_cross_tier_fix_overlap_still_raises(fem: Any) -> None:
    ops = footing(fem)
    ops.fix(label="left", dofs=(1, 1))
    with ops.stage(name="s1") as s:
        s.fix(label="bottom", dofs=(1, 1))
        s.analysis(**chain(ops))
        s.run(n_increments=1)
    with pytest.raises(BridgeError, match=r"(?s)Duplicate fix / mass.*'global pool' "
                                          r"AND in \"stage 's1'\""):
        _deck(ops)


def test_fix_against_a_stage_support_still_raises(fem: Any) -> None:
    ops = footing(fem)
    with ops.stage(name="s1") as s:
        s.fix(label="left", dofs=(1, 1))
        s.support(nodes=[corner_of(fem)], dofs=(1, 0))
        s.analysis(**chain(ops))
        s.run(n_increments=1)
    with pytest.raises(BridgeError, match=r"support on node"):
        _deck(ops)


def test_unstaged_mass_overlap_raises_at_emit(fem: Any) -> None:
    """V2 now runs on an unstaged build: a repeated mass is refused
    instead of silently overwritten by setMass; overwrite=True opts out."""
    ops = footing(fem)
    ops.mass(label="left", values=(1.0, 1.0))
    ops.mass(pg="Bottom", values=(2.0, 2.0))
    with pytest.raises(BridgeError, match=r"mass on node \d+ declared in "
                                          r"'global pool' AND in 'global pool'"):
        _deck(ops)
    ok = footing(fem)
    ok.mass(label="left", values=(1.0, 1.0))
    ok.mass(pg="Bottom", values=(2.0, 2.0), overwrite=True)
    _deck(ok)


def test_unstaged_duplicate_fix_the_merge_never_saw_raises(fem: Any) -> None:
    """A BuiltModel whose records skipped ``apeSees.build``'s merge is
    still validated on the unstaged path."""
    corner = corner_of(fem)
    ops = footing(fem)
    ops.fix(nodes=[corner], dofs=(1, 1))
    bm = ops.build()
    dup = FixRecord(pg=None, nodes=(corner,), dofs=(0, 1))
    object.__setattr__(bm, "fix_records", (*bm.fix_records, dup))
    with pytest.raises(BridgeError, match=rf"fix on node {corner} DOF 2"):
        bm.emit(TclEmitter())
