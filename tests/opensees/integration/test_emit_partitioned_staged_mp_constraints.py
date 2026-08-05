"""STAGED × PARTITIONED × MP-CONSTRAINTS — the intersection that had no
coverage until it produced three defects in a day.

Fourteen tests combined partitioned + constraints, three combined staged +
constraints, and exactly one combined all three.  Both defects pinned here
live only in the triple:

**Defect 1 — a stage-claimed MP constraint was silently dropped.**
``_emit_stages_partitioned``'s per-rank content gate did not count
``stage.stage_constraint_records``, so a rank whose only stage-bound
content was a constraint it participates in was skipped before
``emit_stage_mp_constraints_partitioned`` ever ran.  Measured 2026-07-28
on ``make_two_column_frame_partitioned`` + a cross-rank ``equal_dof``:

===============================  ======================
stage 2 contains                 ``equalDOF`` lines
===============================  ======================
the tie ONLY                     **0** — silently dropped
the tie + an unrelated ``s.fix`` 2 (correct)
===============================  ======================

The claim itself was fine (``stage_constraint_records`` had length 1) —
purely the emit gate.  It suppressed **all ten** claimable constraint
kinds equally, which is why the coverage here spans four of them rather
than just the one the bug was found with.

**Defect 2 — ghost SP state was never synchronised with its owner across
stage boundaries** (ADR 0027 INV-2, amended 2026-07-28).  A ghost is a
foreign node a rank declares because a constraint reaches across the
partition; its DOFs must be constrained there exactly when its owner has
them constrained, or ``numberer ParallelPlain`` disagrees.  Three
distinct ways that broke, all measured 2026-07-28:

* **backward** — ghost first declared in stage N, owner fixed it in
  stage N-1: ghost got the global ``ops.fix`` tier and stage N's own
  ``s.fix``, never stage N-1's.  Confirmed numerically under
  ``mpiexec -n 2`` against the fork ``OpenSeesMP``:
  ``MumpsParallelSolver … Error -10 … Matrix is Singular Numerically``,
  ``analyze failed, returned: -3``.
* **forward fix** — owner applies ``s.fix`` in a stage AFTER the ghost
  was declared.  The owner-side per-rank filter is keyed on
  ``rank_owned``, which a ghost is by definition not in.  Same
  singularity, later.
* **forward remove** — owner ``s.remove_sp``\\ s a DOF the ghost still
  carries.  This one is the dangerous direction: the ghost ends up MORE
  constrained than its owner, which is not singular at all — the model
  just silently answers stiffer.

Reducing the stream to a net fixity vector cannot express the third
case (``fix`` is additive per flagged DOF and never releases), so the
ghost replays the owner's **ordered** SP command stream instead.
"""
from __future__ import annotations

import math
import re
import warnings
from typing import cast

import numpy as np
import pytest

from apeGmsh._kernel.records._constraints import (
    InterpolationRecord,
    NodePairRecord,
    NodeToSurfaceRecord,
    SurfaceCouplingRecord,
)
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh.opensees._internal.build import (
    AUTO_STIFFNESS_ALPHA,
    BridgeError,
)
from apeGmsh.opensees.apesees import apeSees

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
    make_axial_chain_partitioned,
    make_two_column_frame_partitioned,
)

_RANK_OPEN_RE = re.compile(r"^\s*if \{\[getPID\] == (\d+)\} \{\s*$")
_STAGE_RE = re.compile(r"^# === Stage: (.+) ===\s*$")


def _stage_rank_lines(deck_text: str) -> "dict[tuple[int, int], list[str]]":
    """Split a partitioned STAGED Tcl deck into ``{(stage_idx, rank): lines}``.

    ``stage_idx`` is -1 for the pre-stage (global) scope.  Only lines
    inside a rank bracket are returned, ``.strip()``-ed — the 4-space
    bracket indent is emit formatting, not content (ADR 0027 INV-1).
    """
    out: "dict[tuple[int, int], list[str]]" = {}
    stage_idx = -1
    rank: "int | None" = None
    depth = 0
    for raw in deck_text.splitlines():
        if rank is None:
            if _STAGE_RE.match(raw):
                stage_idx += 1
                continue
            m = _RANK_OPEN_RE.match(raw)
            if m:
                rank = int(m.group(1))
                depth = 1
                out.setdefault((stage_idx, rank), [])
            continue
        depth += raw.count("{") - raw.count("}")
        if depth <= 0:
            rank = None
            continue
        if raw.strip():
            out[(stage_idx, rank)].append(raw.strip())
    return out


def _mp_chain(ops):
    return {
        "test": ops.test.NormDispIncr(tol=1e-4, max_iter=10),
        "algorithm": ops.algorithm.Newton(),
        "integrator": ops.integrator.LoadControl(dlam=1.0),
        "constraints": ops.constraints.Transformation(),
        "numberer": ops.numberer.ParallelPlain(),
        "system": ops.system.Mumps(),
        "analysis": ops.analysis.Static(),
    }


def _emit(ops, tmp_path, name="deck.tcl") -> str:
    deck = tmp_path / name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops.tcl(str(deck))
    return deck.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _frame_with_cross_rank_tie() -> FEMStub:
    """``make_two_column_frame_partitioned`` (nodes 1,2 → rank 0;
    3,4 → rank 1) plus an ``equal_dof`` tying node 1 to node 4 — the
    only cross-rank coupling the fixture's two disjoint columns admit."""
    fem = make_two_column_frame_partitioned()
    fem.add_node_constraints([
        NodePairRecord(
            kind=ConstraintKind.EQUAL_DOF,
            master_node=1, slave_node=4, dofs=[1, 2, 3], name="x_tie",
        ),
    ])
    return fem


def _frame_ops(fem: FEMStub) -> apeSees:
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    return ops


def _quad_split_fem() -> FEMStub:
    """Quad host (nodes 1-4, element 1) on rank 0; nodes 5, 6 on rank 1.

    Every surface-coupling kind below binds a rank-1 node into the
    rank-0 host element, so the coupling is genuinely cross-partition
    and the host rank must ghost-declare its slaves.
    """
    fem = FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4, 5, 6],
            coords=[
                (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
                (0.5, 0.5, 0.0), (2.0, 2.0, 0.0),
            ],
            node_pgs={"Base": [1, 2], "Emb": [5]},
        ),
        elements=_ElementsStub(
            elem_pgs={"Rock": _ElementGroupView(
                ids=(1,), connectivity=((1, 2, 3, 4),))},
        ),
    )
    fem.set_partitions([(0, [1, 2, 3, 4], [1]), (1, [5, 6], [])])
    return fem


def _quad_ops(fem: FEMStub, *, global_fix: bool = True) -> apeSees:
    ops = apeSees(cast("object", fem), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    if global_fix:
        ops.fix(pg="Base", dofs=(1, 1))
    return ops


def _two_stages(ops, claim) -> None:
    """Stage 1 empty, stage 2 carrying ONLY the constraint claim — the
    exact shape defect 1 dropped."""
    with ops.stage(name="s1") as s:
        s.analysis(**_mp_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="s2") as s:
        claim(s)
        s.analysis(**_mp_chain(ops))
        s.run(n_increments=1)


# ===========================================================================
# Defect 1 — a constraint-only stage must still emit
# ===========================================================================


def test_constraint_only_stage_emits_equal_dof(tmp_path) -> None:
    """The measured case: stage 2's only content is a cross-rank
    ``s.equal_dof``.  Before the gate fix this emitted **zero**
    ``equalDOF`` lines — the tie the user asked for simply was not in
    the deck, with no warning.  ADR 0027 replicate-on-both puts it on
    each of the two owning ranks."""
    ops = _frame_ops(_frame_with_cross_rank_tie())
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    _two_stages(ops, lambda s: s.equal_dof(name="x_tie"))
    blocks = _stage_rank_lines(_emit(ops, tmp_path))

    for rank in (0, 1):
        assert "equalDOF 1 4 1 2 3" in blocks.get((1, rank), []), (
            f"stage 2 rank {rank} lost the stage-claimed tie: "
            f"{blocks.get((1, rank))}"
        )


def test_constraint_only_stage_matches_stage_with_an_unrelated_fix(
    tmp_path,
) -> None:
    """The tie must not depend on unrelated stage content to survive.

    This pins the *shape* of defect 1 rather than its symptom: adding an
    ``s.fix`` on a completely different PG used to be what made the tie
    appear, because the fix was what opened the rank bracket."""
    def build(with_fix: bool, name: str) -> "dict[tuple[int, int], list[str]]":
        ops = _frame_ops(_frame_with_cross_rank_tie())
        ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
        with ops.stage(name="s1") as s:
            s.analysis(**_mp_chain(ops))
            s.run(n_increments=1)
        with ops.stage(name="s2") as s:
            if with_fix:
                s.fix(pg="Top", dofs=(0, 0, 1, 0, 0, 0))
            s.equal_dof(name="x_tie")
            s.analysis(**_mp_chain(ops))
            s.run(n_increments=1)
        return _stage_rank_lines(_emit(ops, tmp_path, name))

    tie_only = build(False, "a.tcl")
    tie_plus = build(True, "b.tcl")
    for rank in (0, 1):
        eq_only = [ln for ln in tie_only[(1, rank)] if ln.startswith("equalDOF")]
        eq_plus = [ln for ln in tie_plus[(1, rank)] if ln.startswith("equalDOF")]
        assert eq_only == eq_plus == ["equalDOF 1 4 1 2 3"], (
            f"rank {rank}: tie-only {eq_only} vs tie+fix {eq_plus}"
        )


def test_constraint_only_stage_emits_embedded(tmp_path) -> None:
    """``s.embedded`` routes through ``plan.embedded_records`` — a branch
    of ``emit_stage_mp_constraints_partitioned`` that no partitioned-
    staged test reached (12 flat-staged tests, 0 partitioned).  The
    ASDEmbeddedNodeElement emits on the single host-element-owning rank
    with the constrained node ghost-declared there."""
    fem = _quad_split_fem()
    fem.add_surface_constraints([InterpolationRecord(
        kind="embedded", name="emb", slave_node=5,
        master_nodes=[1, 2, 3], weights=None, dofs=[1, 2],
    )])
    ops = _quad_ops(fem)
    _two_stages(ops, lambda s: s.embedded(name="emb"))
    blocks = _stage_rank_lines(_emit(ops, tmp_path))

    rank0 = blocks.get((1, 0), [])
    assert any(
        ln.startswith("element ASDEmbeddedNodeElement") and " 5 1 2 3 " in ln
        for ln in rank0
    ), f"stage 2 rank 0 lost the stage-claimed embed: {rank0}"
    # The constrained node lives on rank 1 — rank 0 must ghost-declare it
    # BEFORE the element that references it (ADR 0027 INV-2 ordering).
    ghost = rank0.index("node 5 0.5 0.5 0.0")
    ele = next(
        i for i, ln in enumerate(rank0)
        if ln.startswith("element ASDEmbeddedNodeElement")
    )
    assert ghost < ele, rank0
    # Single canonical host rank — never replicated onto rank 1.
    assert not any(
        ln.startswith("element ASDEmbeddedNodeElement")
        for ln in blocks.get((1, 1), [])
    ), blocks.get((1, 1))


def test_constraint_only_stage_emits_node_to_surface_phantom(
    tmp_path,
) -> None:
    """``s.node_to_surface`` invents a phantom node.  ADR 0027 INV-3 pins
    phantom tag + coordinate identity across ranks; staging adds a second
    dimension nothing checked."""
    fem = _quad_split_fem()
    fem.add_node_constraints([_n2s_record()])
    ops = _quad_ops(fem)
    _two_stages(ops, lambda s: s.node_to_surface(name="bind"))
    blocks = _stage_rank_lines(_emit(ops, tmp_path))

    decl = {}
    for rank in (0, 1):
        lines = blocks.get((1, rank), [])
        phantom = [ln for ln in lines if ln.startswith("node 999 ")]
        assert len(phantom) == 1, (
            f"stage 2 rank {rank} phantom declarations: {phantom}"
        )
        decl[rank] = phantom[0]
        # ...and it precedes the first constraint that references it.
        first_ref = next(
            i for i, ln in enumerate(lines)
            if ln.startswith(("rigidLink ", "equalDOF "))
        )
        assert lines.index(phantom[0]) < first_ref, lines
    assert decl[0] == decl[1], (
        f"phantom identity diverged across ranks under staging: {decl}"
    )


def test_staged_partitioned_phantom_carries_no_fix(tmp_path) -> None:
    """Phantoms are the branch deliberately EXCLUDED from the ghost-BC
    rule — bridge-invented tags with no owner and no user BCs, so there
    is nothing to replicate and inventing a ``fix`` would over-constrain
    the coupling they exist to express.  That exclusion was unverified in
    the staged case, and defect 2 widened exactly the code path it sits
    in."""
    fem = _quad_split_fem()
    fem.add_node_constraints([_n2s_record()])
    ops = _quad_ops(fem)
    _two_stages(ops, lambda s: s.node_to_surface(name="bind"))
    blocks = _stage_rank_lines(_emit(ops, tmp_path))

    for (stage_idx, rank), lines in blocks.items():
        assert not [ln for ln in lines if ln.startswith("fix 999")], (
            f"stage {stage_idx} rank {rank} invented a fix for the "
            f"phantom: {lines}"
        )
    # The real ghosts on the same block DO carry theirs — otherwise this
    # test would pass on a build that fixes nothing at all.
    rank1 = blocks[(1, 1)]
    assert "fix 1 1 1" in rank1 and "fix 2 1 1" in rank1, rank1


def test_constraint_only_stage_emits_tied_contact(tmp_path) -> None:
    """``s.tied_contact`` is NOT refused under partitioned emit.

    Worth stating because the two are easy to conflate: the fail-loud
    guard in ``_emit_partitioned`` covers ``g.constraints.contact`` /
    ``contact_plane`` — the fork's serial-only contactSurface subsystem.
    ``tied_contact`` is a different feature, a ``SurfaceCouplingRecord``
    of interpolation ties, which lowers to the same parallel-safe
    ASDEmbeddedNodeElement rows as ``embedded`` and routes through the
    canonical-host-rank rule.  So the correct behaviour here is to emit,
    not to raise — and with zero tests, a silent DROP would have looked
    exactly like a deliberate refusal."""
    fem = _quad_split_fem()
    slaves = [
        InterpolationRecord(
            kind=ConstraintKind.TIED_CONTACT, name="iface", slave_node=sn,
            master_nodes=[1, 2, 3], weights=None, dofs=[1, 2],
        )
        for sn in (5, 6)
    ]
    fem.add_surface_constraints([SurfaceCouplingRecord(
        kind=ConstraintKind.TIED_CONTACT, name="iface",
        slave_records=slaves, master_nodes=[1, 2, 3, 4],
        slave_nodes=[5, 6], dofs=[1, 2],
    )])
    ops = _quad_ops(fem)
    _two_stages(ops, lambda s: s.tied_contact(name="iface"))
    blocks = _stage_rank_lines(_emit(ops, tmp_path))

    rank0 = blocks.get((1, 0), [])
    tie_lines = [
        ln for ln in rank0 if ln.startswith("element ASDEmbeddedNodeElement")
    ]
    assert len(tie_lines) == 2, (
        f"both tied-contact slaves must emit on the host rank; got {rank0}"
    )
    # Both slaves live on rank 1 → ghost-declared on the host rank.
    assert "node 5 0.5 0.5 0.0" in rank0 and "node 6 2.0 2.0 0.0" in rank0


# ===========================================================================
# stiffness="auto" through the PARTITIONED emitters
# ===========================================================================
#
# `"auto"` is the default on TieDef/TiedContactDef/EmbeddedDef, and it is
# resolved at emit from the host material — which means the emitter needs a
# `stiffness_resolver` threaded all the way down. Both partitioned entry
# points (flat and stage-bound) take one, and if it fails to arrive the
# record raises BridgeError rather than emitting a wrong number.
#
# Nothing covered that combination before: this file's other records take
# `InterpolationRecord.stiffness`'s numeric 1e18 default (the `"auto"`
# default lives one layer up, on the Def), `test_auto_tie_stiffness.py`
# never partitions, and `test_emit_partitioned_embedded.py` deliberately
# passes an explicit stiffness because its fixture declares no materials.
# So the resolver could have been dropped from either partitioned call site
# with every test still green.

#: The quad host is ElasticIsotropic(E=1e6) and the masters below span
#: (0,0)-(1,1), so L_char = sqrt(2) and K = ALPHA * E * L_char.
_AUTO_K = AUTO_STIFFNESS_ALPHA * 1.0e6 * math.sqrt(2.0)


def _quad_fem_with_host_node_pg() -> FEMStub:
    """:func:`_quad_split_fem` plus a *node* PG for the host group.

    The auto resolver maps a declared element spec's ``pg`` to nodes via
    ``fem.nodes.select(pg=)``; a real FEMData resolves a group on both
    composites, so the stub needs the node side too or no E is found.
    """
    fem = FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4, 5, 6],
            coords=[
                (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
                (0.5, 0.5, 0.0), (2.0, 2.0, 0.0),
            ],
            node_pgs={"Base": [1, 2], "Emb": [5], "Rock": [1, 2, 3, 4]},
        ),
        elements=_ElementsStub(
            elem_pgs={"Rock": _ElementGroupView(
                ids=(1,), connectivity=((1, 2, 3, 4),))},
        ),
    )
    fem.set_partitions([(0, [1, 2, 3, 4], [1]), (1, [5, 6], [])])
    return fem


def _auto_tie_coupling() -> SurfaceCouplingRecord:
    slaves = [
        InterpolationRecord(
            kind=ConstraintKind.TIED_CONTACT, name="iface", slave_node=sn,
            master_nodes=[1, 2, 3], weights=None, dofs=[1, 2],
            stiffness="auto",
        )
        for sn in (5, 6)
    ]
    return SurfaceCouplingRecord(
        kind=ConstraintKind.TIED_CONTACT, name="iface",
        slave_records=slaves, master_nodes=[1, 2, 3, 4],
        slave_nodes=[5, 6], dofs=[1, 2],
    )


def _tie_k_values(lines) -> "list[float]":
    return [
        float(ln.split("-K")[1].split()[0])
        for ln in lines
        if ln.startswith("element ASDEmbeddedNodeElement") and "-K" in ln
    ]


def test_auto_stiffness_resolves_through_staged_partitioned_emit(
    tmp_path,
) -> None:
    """A stage-claimed ``"auto"`` tie emits the resolved K, not 1e18."""
    fem = _quad_fem_with_host_node_pg()
    fem.add_surface_constraints([_auto_tie_coupling()])
    ops = _quad_ops(fem)
    _two_stages(ops, lambda s: s.tied_contact(name="iface"))
    rank0 = _stage_rank_lines(_emit(ops, tmp_path)).get((1, 0), [])

    ks = _tie_k_values(rank0)
    assert len(ks) == 2, f"both slaves must emit on the host rank; got {rank0}"
    for k in ks:
        assert k == pytest.approx(_AUTO_K, rel=1e-12), (
            "stage-bound partitioned emit lost the auto-stiffness resolver"
        )
        assert k != 1.0e18


def test_auto_stiffness_resolves_through_flat_partitioned_emit(
    tmp_path,
) -> None:
    """The same for an UNCLAIMED tie, which takes the flat partitioned
    emitter rather than the stage-bound one — a separate call site, and
    so a separately droppable resolver."""
    fem = _quad_fem_with_host_node_pg()
    fem.add_surface_constraints([_auto_tie_coupling()])
    ops = _quad_ops(fem)
    with ops.stage(name="s1") as s:            # stage present, no claim
        s.analysis(**_mp_chain(ops))
        s.run(n_increments=1)

    ks = _tie_k_values(
        ln.strip() for ln in _emit(ops, tmp_path).splitlines()
    )
    assert len(ks) == 2, "the unclaimed tie must still emit both slaves"
    for k in ks:
        assert k == pytest.approx(_AUTO_K, rel=1e-12), (
            "flat partitioned emit lost the auto-stiffness resolver"
        )


def test_auto_stiffness_without_host_material_fails_loud_partitioned(
    tmp_path,
) -> None:
    """No E-carrying material under the masters ⇒ named BridgeError.

    The point is that it *reaches* the resolver: a dropped
    ``stiffness_resolver=`` would raise the "cannot see the declared
    materials" message instead, so the two failure modes are
    distinguishable.
    """
    fem = _quad_split_fem()                    # no "Rock" NODE pg
    fem.add_surface_constraints([_auto_tie_coupling()])
    ops = _quad_ops(fem)
    _two_stages(ops, lambda s: s.tied_contact(name="iface"))

    with pytest.raises(BridgeError, match="no declared element with an E"):
        _emit(ops, tmp_path)


def _n2s_record() -> NodeToSurfaceRecord:
    return NodeToSurfaceRecord(
        master_node=5, slave_nodes=[1, 2, 3, 4],
        phantom_nodes=[999],
        phantom_coords=np.array([[0.5, 0.5, 0.0]]),
        rigid_link_records=[NodePairRecord(
            kind=ConstraintKind.RIGID_BEAM, master_node=5, slave_node=999,
            dofs=[1, 2, 3, 4, 5, 6], name="n2s_link")],
        equal_dof_records=[NodePairRecord(
            kind=ConstraintKind.EQUAL_DOF, master_node=999, slave_node=sl,
            dofs=[1, 2], name=f"n2s_ed_{sl}") for sl in (1, 2, 3, 4)],
        kind=ConstraintKind.NODE_TO_SURFACE, dofs=[1, 2], name="bind",
    )


def test_non_participating_rank_keeps_its_bracket_closed(tmp_path) -> None:
    """ADR 0034 / Phase SSI-2.D empty-bracket skip survives the gate fix.

    Naively adding ``stage_constraint_records`` to the content gate opens
    an empty ``if {[getPID] == K} { }`` on every rank that does not touch
    the constraint (``emit_stage_mp_constraints_partitioned`` early-
    returns on a rank with no share), which is a ``SyntaxError`` on the
    Py emitter.  Three ranks, a tie spanning only two of them."""
    fem = make_axial_chain_partitioned(n_mass=6, n_parts=3)
    # rank 0 → nodes 1,2,3 · rank 1 → 3,4,5 · rank 2 → 5,6,7
    fem.add_node_constraints([NodePairRecord(
        kind=ConstraintKind.EQUAL_DOF,
        master_node=1, slave_node=4, dofs=[2], name="x_tie",
    )])
    ops = apeSees(cast("object", fem))
    ops.model(ndm=2, ndf=2)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=100.0)
    ops.element.Truss(pg="Chain", A=1.0, material=mat)
    ops.fix(pg="Base", dofs=(1, 1))
    _two_stages(ops, lambda s: s.equal_dof(name="x_tie"))
    blocks = _stage_rank_lines(_emit(ops, tmp_path))

    assert (1, 0) in blocks and (1, 1) in blocks
    assert (1, 2) not in blocks, (
        "rank 2 owns no node the tie touches — it must not open a "
        f"bracket in stage 2; got {blocks.get((1, 2))}"
    )


# ===========================================================================
# Defect 2 — ghost SP state tracks its owner across stage boundaries
# ===========================================================================


def _staged_bc_ops(fem: FEMStub, stage1, stage2) -> apeSees:
    ops = _frame_ops(fem)
    with ops.stage(name="s1") as s:
        stage1(s)
        s.analysis(**_mp_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="s2") as s:
        stage2(s)
        s.analysis(**_mp_chain(ops))
        s.run(n_increments=1)
    return ops


def test_ghost_declared_in_stage_n_carries_the_prior_stage_fix(
    tmp_path,
) -> None:
    """BACKWARD half.  Node 1 is fixed by stage 1 (stage-bound — there is
    no global ``ops.fix`` here).  Stage 2 claims the cross-rank tie, so
    rank 1 ghost-declares node 1 there and must replay stage 1's fix.

    Without it rank 1 holds node 1's six DOFs free, massless and
    stiffness-less while rank 0 has them fixed; measured under
    ``mpiexec -n 2``: ``Error -10 … Matrix is Singular Numerically``,
    ``analyze failed, returned: -3``, no harvestable artifact.  This
    scenario was unreachable before defect 1 was fixed — a stage whose
    only content is the constraint emitted nothing at all."""
    ops = _staged_bc_ops(
        _frame_with_cross_rank_tie(),
        lambda s: s.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1)),
        lambda s: s.equal_dof(name="x_tie"),
    )
    blocks = _stage_rank_lines(_emit(ops, tmp_path))

    rank1 = blocks[(1, 1)]
    ghost = rank1.index("node 1 0.0 0.0 0.0")
    assert rank1[ghost + 1] == "fix 1 1 1 1 1 1 1", (
        "the ghost must replay its owner's stage-1 fix immediately after "
        f"its node line; got {rank1[ghost:ghost + 3]}"
    )
    assert rank1.index("equalDOF 1 4 1 2 3") > ghost
    # Node 4 was never fixed — its ghost on rank 0 must stay unfixed, or
    # this test would pass on a build that blanket-fixes every ghost.
    assert not any(ln.startswith("fix 4 ") for ln in blocks[(1, 0)])


def test_ghost_tracks_a_fix_its_owner_applies_in_a_later_stage(
    tmp_path,
) -> None:
    """FORWARD half, fix direction.  The ghost is declared in stage 1;
    the owner fixes it in stage 2.  The owner-side per-rank filter is
    ``nid in rank_owned`` and a ghost is by definition not owned by the
    declaring rank, so stage 2's ``s.fix`` lands only on rank 0 unless
    the ghost is mirrored — same singularity as the backward half, one
    stage later."""
    ops = _staged_bc_ops(
        _frame_with_cross_rank_tie(),
        lambda s: s.equal_dof(name="x_tie"),
        lambda s: s.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1)),
    )
    blocks = _stage_rank_lines(_emit(ops, tmp_path))

    assert "fix 1 1 1 1 1 1 1" in blocks[(1, 1)], (
        "rank 1 holds a ghost for node 1 and must mirror the owner's "
        f"stage-2 fix; got {blocks[(1, 1)]}"
    )
    # Rank 1 still fixes its OWN base node exactly once, and does not
    # sprout a fix for node 4 (which it owns and nobody fixed).
    assert blocks[(1, 1)].count("fix 3 1 1 1 1 1 1") == 1
    assert not any(ln.startswith("fix 4 ") for ln in blocks[(1, 1)])


def test_ghost_tracks_a_remove_sp_its_owner_applies_in_a_later_stage(
    tmp_path,
) -> None:
    """FORWARD half, REMOVE direction — the silent one.

    A missing ``fix`` leaves a free DOF and the solve goes singular
    loudly.  A missed ``remove sp`` is the opposite: the ghost stays MORE
    constrained than its owner, the matrix is perfectly well conditioned,
    and the model just answers stiffer than the one that was asked for.
    That is the whole reason the ghost replays an ORDERED op stream
    rather than a net fixity vector — ``fix`` is additive per flagged DOF
    and never releases, so a net vector cannot represent "fixed in the
    global tier, freed in stage 2"."""
    fem = _frame_with_cross_rank_tie()
    ops = _frame_ops(fem)
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))     # global tier
    with ops.stage(name="s1") as s:
        s.equal_dof(name="x_tie")                    # ghost declared here
        s.analysis(**_mp_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="s2") as s:
        s.remove_sp(pg="Base", dofs=(1, 2, 3))       # owner releases
        s.analysis(**_mp_chain(ops))
        s.run(n_increments=1)
    blocks = _stage_rank_lines(_emit(ops, tmp_path))

    rank1 = blocks[(1, 1)]
    for dof in (1, 2, 3):
        assert f"remove sp 1 {dof}" in rank1, (
            "rank 1's ghost of node 1 still carries the global fix; "
            f"without the mirrored release it is over-constrained: {rank1}"
        )
    # Rank 0 (the owner) releases its own copy exactly once per DOF.
    for dof in (1, 2, 3):
        assert blocks[(1, 0)].count(f"remove sp 1 {dof}") == 1


def test_ghost_declared_after_a_fix_then_release_replays_both_in_order(
    tmp_path,
) -> None:
    """The ordering case that a net-fixity-vector design cannot express.

    Stage 1 fixes node 1, stage 2 releases DOFs 1-3, stage 3 claims the
    tie.  The ghost declared in stage 3 must replay ``fix`` THEN
    ``remove sp`` — collapsing the stream either way loses which came
    last, and the two orders describe different structures."""
    ops = _frame_ops(_frame_with_cross_rank_tie())
    with ops.stage(name="s1") as s:
        s.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
        s.analysis(**_mp_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="s2") as s:
        s.remove_sp(pg="Base", dofs=(1, 2, 3))
        s.analysis(**_mp_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="s3") as s:
        s.equal_dof(name="x_tie")
        s.analysis(**_mp_chain(ops))
        s.run(n_increments=1)
    blocks = _stage_rank_lines(_emit(ops, tmp_path))

    rank1 = blocks[(2, 1)]
    ghost = rank1.index("node 1 0.0 0.0 0.0")
    assert rank1[ghost + 1:ghost + 5] == [
        "fix 1 1 1 1 1 1 1",
        "remove sp 1 1",
        "remove sp 1 2",
        "remove sp 1 3",
    ], f"ordered replay broken: {rank1[ghost:ghost + 6]}"


def test_ghost_declared_in_a_stage_is_not_also_mirrored_there(
    tmp_path,
) -> None:
    """A ghost declared in stage N already replays that stage's own SP
    delta as part of its declaration.  Mirroring it again in the same
    block would double-emit — harmless for ``fix`` (OpenSees warns and
    carries on, so it would ship easily) but a genuine error for
    ``remove sp``, which would then release a constraint that no longer
    exists."""
    ops = _staged_bc_ops(
        _frame_with_cross_rank_tie(),
        lambda s: None,
        lambda s: (
            s.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1)),
            s.equal_dof(name="x_tie"),
        ),
    )
    rank1 = _stage_rank_lines(_emit(ops, tmp_path))[(1, 1)]
    assert rank1.count("fix 1 1 1 1 1 1 1") == 1, (
        f"ghost node 1's fix emitted twice on rank 1: {rank1}"
    )


def test_unconstrained_ghost_stays_unconstrained_across_stages(
    tmp_path,
) -> None:
    """The negative control for the whole mechanism.  Node 4 carries no
    BC in any tier, so its ghost on rank 0 must never sprout one — in the
    declaring stage or any later one.  A blanket "keep ghosts fixed"
    would over-constrain rather than under-constrain, and every positive
    test above would still pass."""
    ops = _staged_bc_ops(
        _frame_with_cross_rank_tie(),
        lambda s: s.equal_dof(name="x_tie"),
        lambda s: s.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1)),
    )
    blocks = _stage_rank_lines(_emit(ops, tmp_path))
    for (stage_idx, rank), lines in blocks.items():
        assert not [ln for ln in lines if ln.startswith("fix 4 ")], (
            f"stage {stage_idx} rank {rank} invented a BC for node 4: "
            f"{lines}"
        )


@pytest.mark.parametrize("dofs", [(1, 1, 1, 1, 1, 1), (1, 0, 1, 0, 0, 0)])
def test_ghost_replays_the_owners_exact_fixity_vector(
    tmp_path, dofs,
) -> None:
    """The ghost replays the owner's flag vector verbatim, not a
    normalised or all-ones one — a partially-fixed node must stay
    partially fixed on the declaring rank."""
    ops = _staged_bc_ops(
        _frame_with_cross_rank_tie(),
        lambda s: s.fix(pg="Base", dofs=dofs),
        lambda s: s.equal_dof(name="x_tie"),
    )
    rank1 = _stage_rank_lines(_emit(ops, tmp_path))[(1, 1)]
    expected = "fix 1 " + " ".join(str(d) for d in dofs)
    assert expected in rank1, f"expected {expected!r} in {rank1}"
