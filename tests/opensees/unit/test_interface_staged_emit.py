"""Stage-bound ``g.constraints.interface()`` — ``s.interface(name=)``
claim + stage-block emit (ADR 0093 S7 / INV-6).

The liner-install pattern: the interface is declared at apeGmsh time
like any other, but CLAIMED by a stage so its whole per-pair unit
(phantom → equalDOF → two materials → zeroLength) is written inside
that stage's block, on the ground the previous stages equilibrated —
never at ``t = 0``.

What is pinned here:

* the base pass SKIPS a claimed record and the stage block carries the
  full unit, in order, after the stage's activated topology and before
  the stage's ``domain_change`` / analysis chain;
* **tag continuity** — the stage's interface elements draw from the same
  allocator as the base pass, with no collision, and two emits of the
  same model are byte-identical;
* a fully-claimed **mixed-ndf** model still gets ``constraints
  Transformation``: the nested phantom-bridge ``equalDOF`` is a real
  ``MP_Constraint`` and the handler line is emitted up front in the
  analysis chain, before the stage that owns the equalDOF runs;
* the claim registry's fail-loud edges (unknown name, double claim, no
  interfaces at all) and the two refusals this slice owns —
  partitioned + staged (S9) and the staged H5 archive (S7 follow-up).
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh._kernel.records._constraints import (
    InterfaceRecord,
    NodePairRecord,
    NormalLaw,
    TangentialLaw,
)
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh.opensees._internal.build import BridgeError
from apeGmsh.opensees.apesees import apeSees

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)

# Exact binary values so the golden lines carry no float noise.
ENT_LAW = NormalLaw(kind="ent", k_per_area=1.0e6)
EPP_LAW = TangentialLaw(kind="epp", k_per_area=1.0e5, tau_b=250.0)
ORIENT = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def _iface(
    master: int, slave: int, *,
    name: str,
    a_trib: float = 0.25,
    phantom: int | None = None,
    coords=(1.0, 0.5, 0.0),
) -> InterfaceRecord:
    equal_dofs = []
    if phantom is not None:
        equal_dofs = [NodePairRecord(
            kind=ConstraintKind.EQUAL_DOF, name=name,
            master_node=slave, slave_node=phantom, dofs=[1, 2],
        )]
    return InterfaceRecord(
        kind=ConstraintKind.INTERFACE,
        name=name,
        master_node=master,
        slave_node=slave,
        backing_element=1,
        orient=ORIENT,
        a_trib=a_trib,
        normal_law=ENT_LAW,
        tangential_law=EPP_LAW,
        phantom_node=phantom,
        phantom_coords=(np.asarray(coords, dtype=float)
                        if phantom is not None else None),
        phantom_ndf=2 if phantom is not None else None,
        equal_dof_records=equal_dofs,
    )


def _fem(interfaces, *, beam_slave: bool = False) -> FEMStub:
    """Two unit quads sharing nothing: ``Rock`` (nodes 1-4) is the
    continuum master side, ``Liner`` (nodes 5-8) the slave side.  With
    ``beam_slave`` the slave side is a 3-dof line PG instead, which is
    what makes the pair mixed-ndf."""
    liner = (
        # A SEPARATE 3-dof line PG (no node shared with the quad — a
        # shared node would be the shell-on-solid ndf trap, not a
        # mixed-ndf interface).
        _ElementGroupView(ids=(2,), connectivity=((5, 6),))
        if beam_slave
        else _ElementGroupView(ids=(2,), connectivity=((3, 5, 6, 4),))
    )
    fem = FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4, 5, 6],
            coords=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (2.0, 1.0, 0.0),
                (2.0, 0.0, 0.0),
            ],
            node_pgs={"Base": [1, 2]},
        ),
        elements=_ElementsStub(
            elem_pgs={
                "Rock": _ElementGroupView(
                    ids=(1,), connectivity=((1, 2, 3, 4),),
                ),
                "Liner": liner,
            },
        ),
    )
    # The interface side-list is duck-typed on the elements composite
    # (the neutral-zone / fem-like contract) — the stub carries none by
    # default, so install one here.
    fem.elements.interfaces = list(interfaces)
    return fem


def _chain(ops, handler: str = "Plain"):
    return {
        "test":        ops.test.NormDispIncr(tol=1e-4, max_iter=50),
        "algorithm":   ops.algorithm.Newton(),
        "integrator":  ops.integrator.LoadControl(dlam=0.1),
        "constraints": getattr(ops.constraints, handler)(),
        "numberer":    ops.numberer.RCM(),
        "system":      ops.system.UmfPack(),
        "analysis":    ops.analysis.Static(),
    }


def _ops(fem, *, beam_slave: bool = False):
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    if beam_slave:
        ops.element.elasticBeamColumn(
            pg="Liner", transf=ops.geomTransf.Linear(),
            A=0.02, E=2e8, Iz=1e-4,
        )
    else:
        ops.element.FourNodeQuad(pg="Liner", thickness=1.0, material=mat)
    return ops


def _deck(ops, tmp_path, name="deck.tcl") -> "list[str]":
    path = tmp_path / name
    ops.tcl(str(path))
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]


def _stage_block(lines: "list[str]", name: str) -> "list[str]":
    """The lines between this stage's open banner and its close."""
    start = lines.index(f"# === Stage: {name} ===")
    rest = lines[start + 1:]
    for i, ln in enumerate(rest):
        if ln.startswith("# === Stage: ") or ln.startswith("loadConst "):
            return rest[:i]
    return rest


def _staged(
    fem, *,
    beam_slave: bool = False,
    claim: str | None = "RockLiner",
    handler: str = "Plain",
):
    """Two stages: ``ground`` (bare, the equilibration pass) then
    ``install``, which activates the liner PG and claims the interface."""
    ops = _ops(fem, beam_slave=beam_slave)
    with ops.stage(name="ground") as s:
        s.analysis(**_chain(ops, handler))
        s.run(n_increments=1, dt=1.0)
    with ops.stage(name="install") as s:
        s.activate(pgs=["Liner"])
        if claim is not None:
            s.interface(name=claim)
        s.analysis(**_chain(ops, handler))
        s.run(n_increments=1, dt=1.0)
    return ops


# ======================================================================
# Claim registry
# ======================================================================
def test_claim_populates_the_stage_pool_and_the_parallel_id_set():
    recs = [_iface(3, 5, name="RockLiner"), _iface(4, 6, name="RockLiner")]
    ops = _staged(_fem(recs))

    stage = ops._stage_records[1]
    assert stage.stage_interface_records == tuple(recs)
    assert {id(r) for r in recs} == ops._stage_claimed_interface_ids
    # The MP claim set is a SEPARATE pool — an interface claim must not
    # register there (nor vice versa), or the two could cross-release.
    assert ops._stage_claimed_constraint_ids == set()
    assert stage.stage_constraint_records == ()
    # The records stay on the broker: every predicate that must still
    # count them (handler auto-emit, partitioned refusal) reads there.
    assert list(ops._fem.elements.interfaces) == recs


def test_claim_returns_every_pair_of_the_named_interface():
    recs = [_iface(3, 5, name="RockLiner"), _iface(4, 6, name="RockLiner")]
    ops = _ops(_fem(recs))
    with ops.stage(name="install") as s:
        claimed = s.interface(name="RockLiner")
        s.analysis(**_chain(ops))
        s.run(n_increments=1, dt=1.0)
    assert claimed == tuple(recs)


def test_unknown_name_refuses_and_lists_the_available_names():
    recs = [_iface(3, 5, name="RockLiner"), _iface(4, 6, name="Other")]
    ops = _ops(_fem(recs))
    with pytest.raises(ValueError) as exc:
        with ops.stage(name="install") as s:
            s.interface(name="typo")
            s.analysis(**_chain(ops))
            s.run(n_increments=1, dt=1.0)
    msg = str(exc.value)
    assert "no resolved interface records found with name='typo'" in msg
    assert "'Other', 'RockLiner'" in msg


def test_claiming_when_the_model_has_no_interfaces_refuses():
    ops = _ops(_fem([]))
    with pytest.raises(ValueError, match="no g.constraints.interface"):
        with ops.stage(name="install") as s:
            s.interface(name="RockLiner")
            s.analysis(**_chain(ops))
            s.run(n_increments=1, dt=1.0)


def test_empty_name_refuses():
    ops = _ops(_fem([_iface(3, 5, name="RockLiner")]))
    with pytest.raises(ValueError, match="name= must be non-empty"):
        with ops.stage(name="install") as s:
            s.interface(name="")
            s.analysis(**_chain(ops))
            s.run(n_increments=1, dt=1.0)


def test_double_claim_across_stages_refuses():
    ops = _ops(_fem([_iface(3, 5, name="RockLiner")]))
    with ops.stage(name="first") as s:
        s.interface(name="RockLiner")
        s.analysis(**_chain(ops))
        s.run(n_increments=1, dt=1.0)
    with pytest.raises(ValueError, match="already claimed by another stage"):
        with ops.stage(name="second") as s:
            s.interface(name="RockLiner")
            s.analysis(**_chain(ops))
            s.run(n_increments=1, dt=1.0)


# ======================================================================
# Staged emit — equal ndf
# ======================================================================
def test_equal_ndf_claimed_interface_emits_only_inside_the_stage(tmp_path):
    recs = [_iface(3, 5, name="RockLiner"), _iface(4, 6, name="RockLiner")]
    lines = _deck(_staged(_fem(recs)), tmp_path)

    zl = [ln for ln in lines if ln.startswith("element zeroLength ")]
    assert len(zl) == 2                      # emitted exactly ONCE per pair

    block = _stage_block(lines, "install")
    assert [ln for ln in block if ln.startswith("element zeroLength ")] == zl
    # …and nothing interface-shaped in the base pass.
    base = lines[:lines.index("# === Stage: ground ===")]
    assert not [ln for ln in base if ln.startswith("element zeroLength ")]
    assert not [ln for ln in base
                if ln.startswith("uniaxialMaterial ENT ")]


def test_equal_ndf_stage_block_order_topology_then_unit_then_analysis(
    tmp_path,
):
    recs = [_iface(3, 5, name="RockLiner")]
    block = _stage_block(_deck(_staged(_fem(recs)), tmp_path), "install")

    i_liner = max(
        i for i, ln in enumerate(block) if ln.startswith("element quad ")
    )
    i_name = block.index("# RockLiner")
    i_zl = next(
        i for i, ln in enumerate(block)
        if ln.startswith("element zeroLength ")
    )
    i_barrier = block.index("domainChange")
    i_chain = block.index("constraints Plain")
    i_analyze = next(
        i for i, ln in enumerate(block) if "analyze 1 1.0" in ln
    )
    # activated topology → the interface unit → domainChange → chain →
    # analyze.  The barrier is the hard one: the zeroLength (and, for a
    # mixed pair, its phantom) must be in the Domain before the ranks
    # rebuild their DOF maps.
    assert i_liner < i_name < i_zl < i_barrier < i_chain < i_analyze
    # The unit itself: two materials then the element, contiguous.
    assert block[i_name + 1].startswith("uniaxialMaterial ENT ")
    assert block[i_name + 2].startswith("uniaxialMaterial ElasticPP ")
    assert block[i_name + 3].startswith("element zeroLength ")


def test_equal_ndf_needs_no_constraint_handler(tmp_path):
    lines = _deck(_staged(_fem([_iface(3, 5, name="RockLiner")])), tmp_path)
    # Each stage declares its own chain; the only handler lines are the
    # stages' own ``constraints Plain``.  No auto-emitted Transformation.
    assert "constraints Transformation" not in lines


# ======================================================================
# Staged emit — mixed ndf (the phantom bridge)
# ======================================================================
def _mixed_rec():
    return _iface(3, 5, name="RockLiner", phantom=101, coords=(1.0, 1.0, 0.0))


def test_mixed_ndf_full_unit_in_the_stage_block_golden(tmp_path):
    rec = _mixed_rec()
    ops = _staged(
        _fem([rec], beam_slave=True), beam_slave=True,
        handler="Transformation",
    )
    block = _stage_block(_deck(ops, tmp_path), "install")
    i = block.index("# RockLiner")
    # The byte-golden per-pair unit: phantom (ndf=2, the record's OWN,
    # not the MP helper's hardcoded 6) → the bridge equalDOF → the two
    # tributary-scaled uniaxials → the oriented zeroLength.  iNode is
    # the real continuum master, jNode the phantom (INV-1).
    assert block[i:i + 6] == [
        "# RockLiner",
        "node 101 1.0 1.0 0.0 -ndf 2",
        "equalDOF 5 101 1 2",
        "uniaxialMaterial ENT 1 250000.0",
        "uniaxialMaterial ElasticPP 2 25000.0 0.0025",
        "element zeroLength 5 3 101 -mat 1 2 -dir 1 2 "
        "-orient 1.0 0.0 0.0 0.0 1.0 0.0",
    ]


def test_claimed_records_still_count_for_the_handler_predicates():
    """The predicates read ``fem.elements.interfaces``, which a claim
    does NOT empty — so a fully-claimed mixed-ndf model is still an
    MP-carrying model as far as the handler logic is concerned."""
    from apeGmsh.opensees.apesees import (
        _fem_has_handler_requiring_mp,
        _fem_has_interface_equal_dofs,
        _fem_has_mp_constraints,
    )

    rec = _mixed_rec()
    fem = _fem([rec], beam_slave=True)
    ops = _staged(fem, beam_slave=True)
    assert ops._stage_claimed_interface_ids == {id(rec)}   # fully claimed
    assert _fem_has_interface_equal_dofs(fem)
    assert _fem_has_mp_constraints(fem)
    assert _fem_has_handler_requiring_mp(fem)


def test_fully_claimed_mixed_ndf_model_still_gets_transformation(tmp_path):
    """A fully-claimed mixed-ndf model still reaches the deck with an
    MP-capable handler: the stage that installs the interface declares
    ``Transformation``, and it is emitted BEFORE the phantom bridge the
    stage mints.  Nothing about the claim makes the model look MP-free
    (see the predicate test above)."""
    ops = _staged(
        _fem([_mixed_rec()], beam_slave=True), beam_slave=True,
        handler="Transformation",
    )
    lines = _deck(ops, tmp_path)
    block = _stage_block(lines, "install")
    assert "constraints Transformation" in lines
    assert lines.count("constraints Transformation") == 2   # one per stage
    assert block.index("constraints Transformation") > block.index(
        "equalDOF 5 101 1 2")


def test_fully_claimed_mixed_ndf_with_plain_still_warns(tmp_path):
    """The other half of the same guarantee: declaring ``Plain`` against
    a fully-claimed mixed-ndf interface still trips the MP warning — the
    claim must not silence the footgun detector."""
    ops = _staged(_fem([_mixed_rec()], beam_slave=True), beam_slave=True)
    with pytest.warns(UserWarning, match="MP constraints present but Plain"):
        _deck(ops, tmp_path)


# ======================================================================
# Tag continuity + determinism
# ======================================================================
def test_stage_interface_tags_continue_the_base_allocator(tmp_path):
    recs = [_iface(3, 5, name="RockLiner"), _iface(4, 6, name="RockLiner")]
    lines = _deck(_staged(_fem(recs)), tmp_path)

    ele_tags = [
        int(ln.split()[2]) for ln in lines if ln.startswith("element ")
    ]
    assert len(ele_tags) == len(set(ele_tags))        # no collision
    zl_tags = [
        int(ln.split()[2]) for ln in lines
        if ln.startswith("element zeroLength ")
    ]
    mesh_tags = [
        int(ln.split()[2]) for ln in lines if ln.startswith("element quad ")
    ]
    assert min(zl_tags) > max(mesh_tags)              # continues, not restarts
    mat_tags = [
        int(ln.split()[2]) for ln in lines
        if ln.startswith("uniaxialMaterial ")
    ]
    assert len(mat_tags) == len(set(mat_tags)) == 4   # 2 per pair


def test_two_emits_are_byte_identical(tmp_path):
    recs = [_iface(3, 5, name="RockLiner"), _iface(4, 6, name="RockLiner")]
    first = _deck(_staged(_fem(recs)), tmp_path, "a.tcl")
    recs2 = [_iface(3, 5, name="RockLiner"), _iface(4, 6, name="RockLiner")]
    second = _deck(_staged(_fem(recs2)), tmp_path, "b.tcl")
    assert first == second


# ======================================================================
# Partial claim — the unclaimed interface stays in the base pass
# ======================================================================
def test_unclaimed_interface_still_emits_in_the_base_pass(tmp_path):
    claimed = _iface(3, 5, name="RockLiner")
    # The loose record's endpoints are GLOBAL nodes (3 and 4 sit on the
    # always-on Rock quad) — an unclaimed record referencing a
    # stage-bound node is refused since ADR 0093 S8 (see
    # test_unclaimed_interface_on_stage_bound_node_refuses).
    loose = _iface(3, 4, name="Bedrock", a_trib=0.5)
    lines = _deck(_staged(_fem([claimed, loose])), tmp_path)

    base = lines[:lines.index("# === Stage: ground ===")]
    block = _stage_block(lines, "install")
    assert base.count("# Bedrock") == 1
    assert "# RockLiner" not in base
    assert block.count("# RockLiner") == 1
    assert "# Bedrock" not in block
    assert len([ln for ln in lines
                if ln.startswith("element zeroLength ")]) == 2


# ======================================================================
# Refusals owned by this slice
# ======================================================================
def test_partitioned_plus_staged_refuses_naming_s9(tmp_path):
    recs = [_iface(3, 5, name="RockLiner")]
    fem = _fem(recs)
    fem.set_partitions([(0, [1, 2, 3, 4], [1]), (1, [3, 4, 5, 6], [2])])
    ops = _staged(fem)
    with pytest.raises(BridgeError) as exc:
        ops.tcl(str(tmp_path / "deck.tcl"))
    msg = str(exc.value)
    # The staged case gets the message naming ITS slice, not the broader
    # S8 side-list refusal it is checked ahead of.
    assert "ADR 0093 S9" in msg
    assert "s.interface(name=...) in stage(s) 'install'" in msg


def test_unclaimed_interface_on_stage_bound_node_refuses(tmp_path):
    # ADR 0093 S8 hardening: node 5 only comes online inside the
    # 'install' stage (it belongs exclusively to the activated Liner
    # PG), so an UNCLAIMED record referencing it would emit a base-pass
    # zeroLength against a node declared only inside the stage block —
    # a parse-time crash. Refused with the s.interface(name=) advice.
    ops = _staged(_fem([_iface(3, 5, name="RockLiner")]), claim=None)
    with pytest.raises(BridgeError) as exc:
        ops.tcl(str(tmp_path / "deck.tcl"))
    msg = str(exc.value)
    assert "UNCLAIMED" in msg
    assert "s.interface(name=...)" in msg
    assert "interface #1 ('RockLiner'): slave node 5 is owned by " \
        "stage 'install'" in msg
    # The claim is exactly the advertised fix — same model, claimed,
    # emits (the S7 lane).
    ok = _staged(_fem([_iface(3, 5, name="RockLiner")]))
    ok.tcl(str(tmp_path / "ok.tcl"))


def test_unclaimed_interface_on_stage_bound_node_refuses_partitioned(
    tmp_path,
):
    # Same hole on the partitioned path (the base per-rank interface
    # pass, step 7c-bis, emits before any stage block).
    fem = _fem([_iface(3, 5, name="RockLiner")])
    fem.set_partitions([(0, [1, 2, 3, 4], [1]), (1, [3, 4, 5, 6], [2])])
    ops = _staged(fem, claim=None)
    with pytest.raises(BridgeError, match="UNCLAIMED"):
        ops.tcl(str(tmp_path / "deck.tcl"))


def test_staged_h5_archive_refuses_a_claimed_interface(tmp_path):
    recs = [_iface(3, 5, name="RockLiner")]
    ops = _staged(_fem(recs))
    with pytest.raises(BridgeError) as exc:
        ops.h5(str(tmp_path / "model.h5"))
    msg = str(exc.value)
    assert "ADR 0093 S7" in msg
    assert "'RockLiner'" in msg


def test_unclaimed_interface_does_not_block_the_staged_h5_archive(tmp_path):
    """The refusal is scoped to the CLAIM, not to interfaces at all — an
    ordinary staged model whose interface emits in the base pass still
    archives. (Global-node endpoints: since ADR 0093 S8 an unclaimed
    record on a stage-bound node refuses on every emit path, the h5
    archive included.)"""
    ops = _staged(_fem([_iface(3, 4, name="RockLiner")]), claim=None)
    ops.h5(str(tmp_path / "model.h5"))
    assert (tmp_path / "model.h5").exists()
