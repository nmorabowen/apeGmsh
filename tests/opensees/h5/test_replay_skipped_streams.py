"""D8 (#1412): deck replay warns when it skips a stream.

``OpenSeesModel.from_h5(path).build("tcl"|"py"|"live")`` re-emits the
``/opensees`` deck records through ``_replay_into``.  The neutral-zone
streams that the forward bridge fans out at emit time (equalDOF /
rigidLink / rigidDiaphragm / node-to-surface couplings, penalty and
equation ties, ``g.embed`` ties, contacts, phantom-bridged interfaces) have
no deck record, so the replayed deck silently left them out.  Replay now
raises :class:`ReplaySkippedStreamWarning`, naming each stream and its
count, iff the model carries one; a model without any stays silent.

The oracle is the forward deck: a stream is "skipped" when the forward
``apeSees`` deck carries its lines and the replayed deck does not.  The
end-to-end tests check the warning against exactly that difference.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh._kernel.records._constraints import (
    InterpolationRecord,
    NodeGroupRecord,
    NodePairRecord,
)
from apeGmsh.mesh.FEMData import ElementComposite, FEMData, NodeComposite
from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees._internal.compose import (
    ReplaySkippedStreamWarning,
    _replay_staged_into,
    _skipped_replay_streams,
)

from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_fem


def _frame_fem(*, node_constraints=(), **element_streams) -> FEMData:
    """The one-column frame, with extra constraint / side-list streams."""
    base = build_simple_frame_fem()
    nodes = NodeComposite(
        node_ids=np.array([1, 2], dtype=np.int64),
        node_coords=np.asarray(base.nodes.coords),
        physical=base.nodes.physical,
        labels=base.nodes.labels,
        constraints=list(node_constraints),
    )
    elements = ElementComposite(
        groups=dict(base.elements._groups),
        physical=base.elements.physical,
        labels=base.elements.labels,
        **element_streams,
    )
    return FEMData(nodes=nodes, elements=elements, info=base.info)


def _archive(fem: FEMData, tmp_path: Path) -> "tuple[str, str]":
    """Write ``fem`` + an elastic column through apeSees; return the
    forward Tcl deck text and the ``model.h5`` path."""
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    sec = ops.section.Elastic(
        E=2e11, A=0.01, Iz=1e-4, Iy=1e-4, G=8e10, J=1e-4,
    )
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=3)
    ops.element.forceBeamColumn(pg="Cols", transf=transf, integration=integ)
    fwd = tmp_path / "forward.tcl"
    ops.tcl(str(fwd))
    path = tmp_path / "model.h5"
    ops.h5(str(path))
    return fwd.read_text(encoding="utf-8"), str(path)


def _lines(deck: str, token: str) -> "list[str]":
    return [ln for ln in deck.splitlines() if ln.split()[:1] == [token]]


def _element_lines(deck: str, ele_type: str) -> "list[str]":
    return [
        ln for ln in deck.splitlines() if ln.split()[:2] == ["element", ele_type]
    ]


def _replay_silently(om: OpenSeesModel, target: str) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ReplaySkippedStreamWarning)
        return om.build(target)


_EQUAL_DOF = NodePairRecord(
    kind="equal_dof", master_node=1, slave_node=2, dofs=[1, 2, 3],
)


# ---------------------------------------------------------------------------
# End to end: forward deck vs replayed deck
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["tcl", "py"])
def test_clean_model_replays_silently(tmp_path, target):
    _fwd, path = _archive(_frame_fem(), tmp_path)
    deck = _replay_silently(OpenSeesModel.from_h5(path), target)
    assert "forceBeamColumn" in deck


def test_equal_dof_replay_warns_and_names_the_stream(tmp_path):
    fwd, path = _archive(_frame_fem(node_constraints=[_EQUAL_DOF]), tmp_path)
    # The oracle: the forward deck ties the nodes ...
    assert _lines(fwd, "equalDOF") == ["equalDOF 1 2 1 2 3"]
    om = OpenSeesModel.from_h5(path)
    with pytest.warns(ReplaySkippedStreamWarning) as caught:
        deck = om.build("tcl")
    # ... and the replayed deck does not, which the warning says.
    assert _lines(deck, "equalDOF") == []
    msgs = [str(w.message) for w in caught
            if w.category is ReplaySkippedStreamWarning]
    assert len(msgs) == 1
    assert "fem.nodes.constraints (equal_dof: 1)" in msgs[0]
    assert "1 stream(s)" in msgs[0]


def test_h5_target_stays_silent_with_constraints(tmp_path):
    # build("h5") passes no fem: the rewritten archive keeps the
    # constraint in its neutral zone, so nothing is skipped.
    _fwd, path = _archive(_frame_fem(node_constraints=[_EQUAL_DOF]), tmp_path)
    om = OpenSeesModel.from_h5(path)
    out = tmp_path / "rewrite.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("error", ReplaySkippedStreamWarning)
        om.build("h5", out=str(out))
    assert len(OpenSeesModel.from_h5(str(out)).fem.nodes.constraints) == 1


def test_element_form_coupling_replays_and_stays_silent(tmp_path):
    # A kinematic coupling goes out as an ``element`` line, which the deck
    # zone archives and replay re-emits: not a skipped stream.
    rbe2 = NodeGroupRecord(
        kind="kinematic_coupling", master_node=1, slave_nodes=[2],
        dofs=[1, 2, 3],
    )
    fwd, path = _archive(_frame_fem(node_constraints=[rbe2]), tmp_path)
    fwd_lines = _element_lines(fwd, "LadrunoKinematicCoupling")
    assert len(fwd_lines) == 1
    deck = _replay_silently(OpenSeesModel.from_h5(path), "tcl")
    assert _element_lines(deck, "LadrunoKinematicCoupling") == fwd_lines


# ---------------------------------------------------------------------------
# Staged replay: stage-claimed constraints are re-emitted, not skipped
# ---------------------------------------------------------------------------


def _staged_replay(fem: FEMData, claimed_name: str, **buckets) -> None:
    """Replay one stage whose MP buckets are ``buckets`` (default: the
    equalDOF 1 -> 2 on dofs 1-3, named ``claimed_name``)."""
    from apeGmsh.opensees._internal.typed_records import (
        EqualDOFRecord,
        StageRecordRO,
    )
    from apeGmsh.opensees.emitter.recording import RecordingEmitter

    if not buckets:
        buckets = {
            "equal_dofs": (EqualDOFRecord(
                master=1, slave=2, dofs=(1, 2, 3), name=claimed_name,
            ),),
            "equal_dof_seq": (1,),
        }
    st = StageRecordRO(name="s1", analyze_steps=1, **buckets)
    _replay_staged_into(
        RecordingEmitter(), stages=(st,), ndm=3, ndf=6, fem=fem,
    )


def test_staged_replay_silent_when_every_constraint_is_stage_claimed():
    named = NodePairRecord(
        kind="equal_dof", name="tie1", master_node=1, slave_node=2,
        dofs=[1, 2, 3],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ReplaySkippedStreamWarning)
        _staged_replay(_frame_fem(node_constraints=[named]), "tie1")


def test_staged_replay_warns_for_an_unclaimed_constraint():
    named = NodePairRecord(
        kind="equal_dof", name="tie1", master_node=1, slave_node=2,
        dofs=[1, 2, 3],
    )
    unclaimed = NodePairRecord(
        kind="equal_dof", master_node=1, slave_node=2, dofs=[4, 5, 6],
    )
    fem = _frame_fem(node_constraints=[named, unclaimed])
    with pytest.warns(
        ReplaySkippedStreamWarning,
        match=r"fem\.nodes\.constraints \(equal_dof: 1\)",
    ):
        _staged_replay(fem, "tie1")


def test_staged_replay_silent_for_a_stage_claimed_tied_contact():
    # Review finding 1 on b54b4665: a tied_contact slave row carries
    # name=None (only the parent is named), so a name match missed it and
    # warned although the stage block replays the slave as embeddedNode.
    from apeGmsh._kernel.records._constraints import SurfaceCouplingRecord
    from apeGmsh.opensees._internal.typed_records import EmbeddedNodeRecord

    tc = SurfaceCouplingRecord(
        kind="tied_contact", name="tc",
        slave_records=[_interp("tie")],   # slave 2 tied to master 1
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ReplaySkippedStreamWarning)
        _staged_replay(
            _frame_fem(constraints=[tc]), "tc",
            embedded_nodes=(EmbeddedNodeRecord(
                ele_tag=7, cnode=2, args=(1,),
            ),),
            embedded_node_seq=(1,),
        )


def test_staged_claim_by_name_does_not_excuse_another_kind():
    # Review finding 2 on b54b4665: stage claims filter by kind, so an
    # equalDOF "x" claimed by the stage must not hide a rigidDiaphragm
    # "x" that stayed global and is lost on replay.
    eq = NodePairRecord(
        kind="equal_dof", name="x", master_node=1, slave_node=2,
        dofs=[1, 2, 3],
    )
    rd = NodeGroupRecord(
        kind="rigid_diaphragm", name="x", master_node=1, slave_nodes=[2],
    )
    with pytest.warns(
        ReplaySkippedStreamWarning,
        match=r"fem\.nodes\.constraints \(rigid_diaphragm: 1\)",
    ):
        _staged_replay(_frame_fem(node_constraints=[eq, rd]), "x")


# ---------------------------------------------------------------------------
# The stream table, one row at a time
# ---------------------------------------------------------------------------


def _interp(kind: str, enforce: str = "penalty") -> InterpolationRecord:
    return InterpolationRecord(
        kind=kind, slave_node=2, master_nodes=[1], dofs=[1, 2, 3],
        enforce=enforce,
    )


@pytest.mark.parametrize(
    ("record", "skipped"),
    [
        (_interp("tie"), True),            # embeddedNode: no deck record
        (_interp("embedded"), True),
        (_interp("tie", "equation"), True),  # equationConstraint: ledger
        (_interp("tie", "penalty_al"), False),  # LadrunoEmbeddedNode element
        (_interp("distributing"), False),  # LadrunoDistributingCoupling
        (_interp("distributing", "equation"), True),
    ],
)
def test_interpolation_rows(record, skipped):
    out = _skipped_replay_streams(
        _frame_fem(constraints=[record]), stage_mp_keys=frozenset(),
    )
    if skipped:
        assert out == {"fem.elements.constraints": {record.kind: 1}}
    else:
        assert out == {}


@pytest.mark.parametrize(
    ("record", "skipped"),
    [
        (NodePairRecord(kind="rigid_beam", master_node=1, slave_node=2), True),
        (NodeGroupRecord(kind="rigid_diaphragm", master_node=1,
                         slave_nodes=[2]), True),
        (NodeGroupRecord(kind="rigid_body", master_node=1,
                         slave_nodes=[2]), True),
        (NodeGroupRecord(kind="rigid_body", master_node=1,
                         slave_nodes=[2], as_element=True), False),
        (NodeGroupRecord(kind="kinematic_coupling", master_node=1,
                         slave_nodes=[2]), False),
    ],
)
def test_node_constraint_rows(record, skipped):
    out = _skipped_replay_streams(
        _frame_fem(node_constraints=[record]), stage_mp_keys=frozenset(),
    )
    if skipped:
        assert out == {"fem.nodes.constraints": {record.kind: 1}}
    else:
        assert out == {}


def test_side_list_streams_are_named_with_counts():
    # Only the count is read here, so plain stand-ins carry the rows.
    fem = _frame_fem(
        embed_ties=[object(), object()],
        contacts=[object()],
        contact_planes=[object()],
        interfaces=[
            SimpleNamespace(kind="interface", name=None, phantom_node=None),
            SimpleNamespace(kind="interface", name=None, phantom_node=99),
        ],
        reinforce_ties=[object()],   # replayed by step 8b
        rebar_elements=[object()],   # element lines: replayed
    )
    out = _skipped_replay_streams(fem, stage_mp_keys=frozenset())
    assert out == {
        "fem.elements.interfaces": {"interface": 1},
        "fem.elements.embed_ties": {"record": 2},
        "fem.elements.contacts": {"record": 1},
        "fem.elements.contact_planes": {"record": 1},
    }


def test_a_fem_without_a_stream_fails_loud():
    # No None-means-skip: a broker missing a stream is an error, not empty.
    real = _frame_fem()
    fem = SimpleNamespace(
        nodes=real.nodes,
        elements=SimpleNamespace(
            constraints=real.elements.constraints,
            interfaces=[], embed_ties=[], contact_planes=[],
        ),
    )
    with pytest.raises(AttributeError, match="contacts"):
        _skipped_replay_streams(fem, stage_mp_keys=frozenset())
