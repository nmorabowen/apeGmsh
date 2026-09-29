"""ADR 0051 §4 — broker supports / masses the deck never restated.

``g.constraints.bc(...)`` resolves into homogeneous SP records on
``fem.nodes.sp`` and ``g.masses.*`` into ``fem.nodes.masses``, but the
bridge emits neither on its own: a deck carries them only through
``ops.fix`` / ``s.fix`` / ``s.support`` / ``ops.fix_from_model()`` and
``ops.mass`` / ``s.mass`` / ``ops.mass_from_model()``.  A deck that
restated nothing used to run unsupported and massless with no word
(an audit at 07f757e0: 174 SP + 339 mass records -> 0 ``fix`` / 0
``mass`` lines).  Emit now warns with
:class:`UnconsumedModelDefinitionWarning` and stays silent once every
record is restated.

Also covers ``ops.fix_from_model()``, the fix twin of
``ops.mass_from_model()``.

The records are injected straight onto a hand-rolled one-brick stub so
each test controls exactly which (node, DOF) pairs the broker holds.
"""
from __future__ import annotations

import warnings

import pytest

from apeGmsh._kernel.record_sets import MassSet, SPSet
from apeGmsh._kernel.records._loads import SPRecord
from apeGmsh._kernel.records._masses import MassRecord
from apeGmsh.opensees import UnconsumedModelDefinitionWarning
from apeGmsh.opensees.apesees import BridgeError, apeSees
from apeGmsh.opensees.emitter.recording import RecordingEmitter

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


# One unit hex. Base = z=0 face (1-4); Side = y=0 face (1, 2, 5, 6).
def _brick_fem() -> FEMStub:
    return FEMStub(
        nodes=_NodesStub(
            ids=list(range(1, 9)),
            coords=[
                (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0), (1.0, 0.0, 1.0),
                (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
            ],
            node_pgs={"Base": [1, 2, 3, 4], "Side": [1, 2, 5, 6]},
        ),
        elements=_ElementsStub(
            elem_pgs={
                "Block": _ElementGroupView(
                    ids=(1,), connectivity=((1, 2, 3, 4, 5, 6, 7, 8),),
                ),
            },
        ),
    )


def _bc(pg_nodes: "list[int]", mask: "tuple[int, ...]") -> list:
    """The records ``g.constraints.bc`` resolves to: one per flagged DOF."""
    return [
        SPRecord(node_id=n, dof=d, value=0.0, is_homogeneous=True)
        for n in pg_nodes
        for d, flag in enumerate(mask, start=1)
        if flag
    ]


def _model(*, sp: "list | None" = None, masses: bool = True) -> FEMStub:
    """Broker: bc(Base, 111) + bc(Side, 010) -> 12 + 4 = 16 SP records on
    6 nodes; a lumped mass on all 8 nodes."""
    fem = _brick_fem()
    fem.nodes.sp = SPSet(  # type: ignore[attr-defined]
        sp if sp is not None
        else _bc([1, 2, 3, 4], (1, 1, 1)) + _bc([1, 2, 5, 6], (0, 1, 0))
    )
    if masses:
        fem.nodes.masses = MassSet([  # type: ignore[attr-defined]
            MassRecord(node_id=n, mass=(2.0, 2.0, 2.0, 0.0, 0.0, 0.0))
            for n in range(1, 9)
        ])
    return fem


def _ops(fem: FEMStub) -> apeSees:
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.stdBrick(pg="Block", material=mat)
    return ops


def _chain(ops: apeSees) -> dict[str, object]:
    return {
        "test":        ops.test.NormDispIncr(tol=1e-6, max_iter=10),
        "algorithm":   ops.algorithm.Newton(),
        "integrator":  ops.integrator.LoadControl(dlam=1.0),
        "constraints": ops.constraints.Plain(),
        "numberer":    ops.numberer.RCM(),
        "system":      ops.system.UmfPack(),
        "analysis":    ops.analysis.Static(),
    }


def _emit_silent(ops: apeSees) -> RecordingEmitter:
    """Emit with this category escalated, so the warning fails the test."""
    rec = RecordingEmitter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnconsumedModelDefinitionWarning)
        ops.build().emit(rec)
    return rec


def _emit_warns(ops: apeSees) -> str:
    rec = RecordingEmitter()
    with pytest.warns(UnconsumedModelDefinitionWarning) as caught:
        ops.build().emit(rec)
    hits = [w for w in caught
            if issubclass(w.category, UnconsumedModelDefinitionWarning)]
    assert len(hits) == 1, "one aggregated warning per emit"
    return str(hits[0].message)


def _calls(rec: RecordingEmitter, name: str) -> list[tuple]:
    return [c[1] for c in rec.calls if c[0] == name]


# ---------------------------------------------------------------------------
# The warning
# ---------------------------------------------------------------------------


def test_category_is_a_public_userwarning() -> None:
    assert issubclass(UnconsumedModelDefinitionWarning, UserWarning)


def test_nothing_restated_warns_with_counts_and_the_fix() -> None:
    msg = _emit_warns(_ops(_model()))
    assert "16 homogeneous SP record(s) on 6 node(s)" in msg
    assert "8 nodal mass record(s)" in msg
    assert "ops.fix(" in msg and "ops.fix_from_model()" in msg
    assert "ops.mass_from_model()" in msg


def test_fully_restated_deck_is_silent_and_unchanged() -> None:
    ops = _ops(_model())
    ops.fix(pg="Base", dofs=(1, 1, 1))
    ops.fix(nodes=[5, 6], dofs=(0, 1, 0))   # Side minus the Base overlap
    ops.mass_from_model()
    rec = _emit_silent(ops)
    assert _calls(rec, "fix") == [
        (1, 1, 1, 1), (2, 1, 1, 1), (3, 1, 1, 1), (4, 1, 1, 1),
        (5, 0, 1, 0), (6, 0, 1, 0),
    ]
    assert len(_calls(rec, "mass")) == 8


def test_only_the_missing_half_is_named() -> None:
    ops = _ops(_model())
    ops.mass_from_model()
    msg = _emit_warns(ops)
    assert "homogeneous SP" in msg and "nodal mass" not in msg

    ops = _ops(_model())
    ops.fix(pg="Base", dofs=(1, 1, 1))
    ops.fix(nodes=[5, 6], dofs=(0, 1, 0))
    msg = _emit_warns(ops)
    assert "8 nodal mass record(s)" in msg and "homogeneous SP" not in msg


def test_partial_dof_coverage_counts_the_gap() -> None:
    """Base fixed in x, y only: its 4 z-records + Side's 2 unshared nodes
    stay unconsumed."""
    ops = _ops(_model())
    ops.fix(pg="Base", dofs=(1, 1, 0))
    ops.mass_from_model()
    msg = _emit_warns(ops)
    assert "6 homogeneous SP record(s) on 6 node(s)" in msg


def test_a_wider_mask_than_the_geometry_is_silent() -> None:
    ops = _ops(_model())
    ops.fix(nodes=[1, 2, 3, 4, 5, 6], dofs=(1, 1, 1))
    ops.mass_from_model()
    _emit_silent(ops)


def test_explicit_mass_covers_node_by_node() -> None:
    ops = _ops(_model())
    ops.fix_from_model()
    ops.mass(nodes=[1, 2, 3], values=(1.0, 1.0, 1.0))
    msg = _emit_warns(ops)
    assert "5 nodal mass record(s)" in msg

    ops = _ops(_model())
    ops.fix_from_model()
    ops.mass(nodes=list(range(1, 9)), values=(1.0, 1.0, 1.0))
    _emit_silent(ops)


def test_node_aggregate_verbs_count() -> None:
    """``Node`` / ``NodeSet`` verbs (charter P1) delegate to ops.fix /
    ops.mass, so they consume records like the flat verbs."""
    ops = _ops(_model())
    ops.nodes.get(pg="Base").fix(dofs=(1, 1, 1))
    ops.nodes.get(tag=5).fix(dofs=(0, 1, 0))
    ops.nodes.get(tag=6).fix(dofs=(0, 1, 0))
    ops.nodes.get().mass(values=(1.0, 1.0, 1.0))
    _emit_silent(ops)


def test_stage_fix_support_and_mass_count() -> None:
    ops = _ops(_model())
    with ops.stage(name="build") as s:
        s.fix(pg="Base", dofs=(1, 1, 1))
        s.mass(nodes=list(range(1, 9)), values=(2.0, 2.0, 2.0))
        s.analysis(**_chain(ops))
        s.run(n_increments=1, dt=1.0)
    with ops.stage(name="hold") as s:
        s.support(nodes=[5, 6], dofs=(0, 1, 0))
        s.analysis(**_chain(ops))
        s.run(n_increments=1, dt=1.0)
    _emit_silent(ops)


def test_dofs_beyond_the_node_ndf_are_not_counted() -> None:
    """bc(dofs=[1]*6) on 3-DOF solid nodes: the rotation records address
    DOFs no deck can carry, so they are not 'dropped'."""
    fem = _model(sp=_bc([1, 2, 3, 4], (1, 1, 1, 1, 1, 1)))
    ops = _ops(fem)
    ops.fix(pg="Base", dofs=(1, 1, 1))
    ops.mass_from_model()
    _emit_silent(ops)


def test_prescribed_sp_is_not_a_model_definition_record() -> None:
    """A non-zero SP belongs to a load case (p.from_model imports it);
    only homogeneous records are model-level."""
    fem = _model(sp=[
        SPRecord(node_id=7, dof=3, value=0.01, is_homogeneous=False,
                 pattern="push"),
    ])
    ops = _ops(fem)
    ops.mass_from_model()
    _emit_silent(ops)


def test_empty_broker_is_silent() -> None:
    ops = _ops(_brick_fem())   # the stub carries no sp / masses at all
    _emit_silent(ops)


# ---------------------------------------------------------------------------
# 2-D: bc masks are SPATIAL (ux uy uz rx ry rz), deck masks positional
# ---------------------------------------------------------------------------


def _frame2d(sp: list) -> apeSees:
    """ndm=2, ndf=3 frame (DOFs ux, uy, rz): one planar column."""
    fem = FEMStub(
        nodes=_NodesStub(
            ids=[1, 2], coords=[(0.0, 0.0, 0.0), (0.0, 3.0, 0.0)],
            node_pgs={"Base": [1]},
        ),
        elements=_ElementsStub(elem_pgs={
            "Col": _ElementGroupView(ids=(1,), connectivity=((1, 2),)),
        }),
    )
    fem.nodes.sp = SPSet(sp)  # type: ignore[attr-defined]
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=3)
    t = ops.geomTransf.Linear()
    ops.element.elasticBeamColumn(pg="Col", transf=t, A=0.01, E=2e11, Iz=1e-5)
    return ops


def _solid2d(sp: list) -> apeSees:
    """ndm=2, ndf=2 plane quad (DOFs ux, uy)."""
    fem = FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4],
            coords=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                    (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
            node_pgs={"Base": [1, 2]},
        ),
        elements=_ElementsStub(elem_pgs={
            "Plate": _ElementGroupView(ids=(1,), connectivity=((1, 2, 3, 4),)),
        }),
    )
    fem.nodes.sp = SPSet(sp)  # type: ignore[attr-defined]
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Plate", thickness=1.0, material=mat)
    return ops


def test_2d_frame_pinned_base_is_consumed_by_a_pin() -> None:
    """bc default [1,1,1] = ux, uy, uz. On a 2-D frame DOF 3 is rz, so a
    pin ``fix 1 1 1 0`` restates it fully (uz has no DOF); it must not
    read the uz record as an unfixed rz."""
    ops = _frame2d(_bc([1], (1, 1, 1)))
    ops.fix(nodes=[1], dofs=(1, 1, 0))
    rec = _emit_silent(ops)
    assert _calls(rec, "fix") == [(1, 1, 1, 0)]


def test_2d_frame_fix_from_model_pins_and_leaves_rz_free() -> None:
    ops = _frame2d(_bc([1], (1, 1, 1)))
    ops.fix_from_model()
    rec = _emit_silent(ops)
    assert _calls(rec, "fix") == [(1, 1, 1, 0)]


def test_2d_frame_rz_restraint_lands_on_dof_3() -> None:
    """bc(dofs=[1,1,0,0,0,1]) (spatial rz) -> fix 1 1 1 1; a pin alone
    leaves the rz record unconsumed."""
    ops = _frame2d(_bc([1], (1, 1, 0, 0, 0, 1)))
    ops.fix_from_model()
    rec = _emit_silent(ops)
    assert _calls(rec, "fix") == [(1, 1, 1, 1)]

    ops = _frame2d(_bc([1], (1, 1, 0, 0, 0, 1)))
    ops.fix(nodes=[1], dofs=(1, 1, 0))
    assert "1 homogeneous SP record(s) on 1 node(s)" in _emit_warns(ops)


def test_2d_solid_default_bc_maps_to_x_and_y() -> None:
    ops = _solid2d(_bc([1, 2], (1, 1, 1)))
    ops.fix_from_model()
    rec = _emit_silent(ops)
    assert sorted(_calls(rec, "fix")) == [(1, 1, 1), (2, 1, 1)]

    ops = _solid2d(_bc([1, 2], (1, 1, 1)))
    ops.fix(pg="Base", dofs=(1, 1))
    _emit_silent(ops)


def test_message_names_displacement_hold_cases() -> None:
    ops = _ops(_model(sp=[
        SPRecord(node_id=7, dof=2, value=0.0, is_homogeneous=True,
                 pattern="push"),
    ]))
    ops.mass_from_model()
    msg = _emit_warns(ops)
    assert "zero-valued g.displacements holds in case(s) 'push'" in msg
    assert "g.constraints.bc" not in msg.split("have no fix")[0]


def test_warning_points_at_the_callers_line() -> None:
    """stacklevel walks out of apeGmsh: the reported file is this test."""
    rec = RecordingEmitter()
    with pytest.warns(UnconsumedModelDefinitionWarning) as caught:
        _ops(_model()).build().emit(rec)
    assert caught[0].filename == __file__


# ---------------------------------------------------------------------------
# ops.fix_from_model()
# ---------------------------------------------------------------------------


def test_fix_from_model_one_fix_per_node_with_the_union_mask() -> None:
    """Base (111) and Side (010) overlap on nodes 1, 2: one fix each,
    never a second SP on an already-constrained DOF."""
    ops = _ops(_model())
    ops.fix_from_model()
    ops.mass_from_model()
    rec = _emit_silent(ops)
    assert sorted(_calls(rec, "fix")) == [
        (1, 1, 1, 1), (2, 1, 1, 1), (3, 1, 1, 1), (4, 1, 1, 1),
        (5, 0, 1, 0), (6, 0, 1, 0),
    ]


def test_fix_from_model_pads_a_short_mask_to_the_node_ndf() -> None:
    """A z-only restraint resolves to one dof=3 record -> fix n 0 0 1."""
    fem = _model(sp=_bc([1, 2], (0, 0, 1)))
    ops = _ops(fem)
    ops.fix_from_model()
    ops.mass_from_model()
    rec = _emit_silent(ops)
    assert sorted(_calls(rec, "fix")) == [(1, 0, 0, 1), (2, 0, 0, 1)]


def test_fix_from_model_leaves_out_dofs_the_node_lacks() -> None:
    """bc(dofs=[1]*6) on 3-DOF solid nodes fixes the three DOFs that
    exist (no G3 error on a 6-long mask); a node whose only restrained
    DOF is missing gets no fix line at all."""
    fem = _model(
        sp=_bc([1, 2, 3, 4], (1, 1, 1, 1, 1, 1)) + _bc([7], (0, 0, 0, 1)),
    )
    ops = _ops(fem)
    ops.fix_from_model()
    ops.mass_from_model()
    rec = _emit_silent(ops)
    assert sorted(_calls(rec, "fix")) == [(n, 1, 1, 1) for n in (1, 2, 3, 4)]


def test_following_the_warnings_advice_silences_it() -> None:
    fem = _model(sp=_bc([1, 2, 3, 4], (1, 1, 1, 1, 1, 1)))
    msg = _emit_warns(_ops(fem))
    assert "ops.fix_from_model()" in msg and "ops.mass_from_model()" in msg
    ops = _ops(fem)
    ops.fix_from_model()
    ops.mass_from_model()
    _emit_silent(ops)


def test_fix_from_model_skips_prescribed_records() -> None:
    fem = _model(sp=_bc([1], (1, 1, 1)) + [
        SPRecord(node_id=7, dof=3, value=0.01, is_homogeneous=False,
                 pattern="push"),
    ])
    ops = _ops(fem)
    ops.fix_from_model()
    ops.mass_from_model()
    rec = _emit_silent(ops)
    assert _calls(rec, "fix") == [(1, 1, 1, 1)]


def test_fix_from_model_with_no_records_is_a_noop() -> None:
    ops = _ops(_brick_fem())
    ops.fix_from_model()
    rec = _emit_silent(ops)
    assert _calls(rec, "fix") == []


def test_fix_from_model_combines_with_disjoint_explicit_fix() -> None:
    fem = _model(sp=_bc([1, 2, 3, 4], (1, 1, 1)))
    ops = _ops(fem)
    ops.fix_from_model()
    ops.fix(nodes=[7], dofs=(1, 0, 0))
    ops.mass_from_model()
    rec = _emit_silent(ops)
    assert sorted(_calls(rec, "fix")) == [
        (1, 1, 1, 1), (2, 1, 1, 1), (3, 1, 1, 1), (4, 1, 1, 1),
        (7, 1, 0, 0),
    ]


def test_fix_from_model_overlapping_explicit_fix_raises() -> None:
    ops = _ops(_model())
    ops.fix_from_model()
    ops.fix(nodes=[3], dofs=(0, 0, 1))
    with pytest.raises(BridgeError, match="fix_from_model"):
        ops.build()


def test_fix_from_model_overlapping_stage_fix_raises() -> None:
    ops = _ops(_model())
    ops.fix_from_model()
    with ops.stage(name="late") as s:
        s.support(nodes=[5], dofs=(0, 1, 0))
        s.analysis(**_chain(ops))
        s.run(n_increments=1, dt=1.0)
    with pytest.raises(BridgeError, match="fix_from_model"):
        ops.build()
