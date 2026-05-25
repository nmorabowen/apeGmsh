"""S5 — E2E partitioned mixed-ndf shell-on-solid emit (ADR 0033).

Closes the shell-to-solid coupling stream (S1 + S2 + S5).  S1 + S2
(PRs #325 / #328) wired per-node ``ndf`` into every emit path; the
flat case is locked by ``test_emit_mixed_ndf_shell_on_solid_flat`` in
``tests/test_node_ndf.py``.  This file pins the **partitioned** case:
the per-foreign-node lookup at
``apeGmsh.opensees._internal.build.emit_mp_constraints_partitioned``
must fire per-tag (not as a scalar envelope fallback) when a cross-
partition MP constraint references a node owned by another rank.

Fixture
=======

A minimal 4-node, 2-partition stub modelling a "shell tile on a solid
block" interface — node 2 sits on rank 0 carrying ``ndf=6`` (shell-
owned), node 4 sits on rank 1 carrying ``ndf=3`` (solid-owned).  A
single cross-partition ``equalDOF(2, 4, 1, 2, 3)`` straddles the
interface, so:

* rank 0 owns nodes 1, 2 (both shell, ``ndf=6``); declares node 4 as
  foreign — the foreign-node decl MUST source ``ndf=3`` from the
  broker, not fall back to the bridge envelope ``ndf=6``.
* rank 1 owns nodes 3, 4 (both solid, ``ndf=3``); declares node 2 as
  foreign — the foreign-node decl MUST source ``ndf=6`` from the
  broker.

The envelope is ``ndf=6`` (the max — validated by
``validate_envelope_covers_broker_ndf`` at ``apeSees.model``).  Solid
nodes inside rank 1's owned-node pass MUST emit with ``-ndf 3`` (the
broker override against the 6-envelope); shell nodes on rank 0 MUST
emit with ``-ndf 6`` (still broker-sourced — the broker has an
explicit declaration there).
"""
from __future__ import annotations

from typing import cast

import pytest

from apeGmsh._kernel.records._constraints import NodePairRecord
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.tag_resolution import (
    ATTR_PHANTOM_NODE_TAGS,
)
from apeGmsh.opensees.emitter.recording import RecordingEmitter

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


# Silence the bridge's documented MP-auto-emit warnings (constraint
# handler / numberer / system) so the file is runnable under
# ``pytest -W error::UserWarning``.  These warnings are expected and
# fire because the test does not pre-declare an MP-friendly chain
# (the test's surface is per-rank ndf emit, not chain declaration).
# Matched by substring against the warning messages emitted by
# :mod:`apeGmsh.opensees.apesees._maybe_auto_emit_*` per ADR 0027 INV-5.
_MP_AUTO_EMIT_FILTERS = (
    "ignore:MP constraints are present in the model:UserWarning",
    "ignore:len.fem.partitions. > 1 with no user-declared numberer:UserWarning",
    "ignore:len.fem.partitions. > 1 with no user-declared system:UserWarning",
)
pytestmark = [pytest.mark.filterwarnings(f) for f in _MP_AUTO_EMIT_FILTERS]


# ---------------------------------------------------------------------------
# Per-rank call bucketing helper (mirrors the
# test_emit_partitioned_mp_constraint_replication.py shape).
# ---------------------------------------------------------------------------


def _per_rank_calls(rec: RecordingEmitter) -> dict[int, list[tuple]]:
    """Bucket emit calls by the enclosing ``partition_open(K)`` rank.

    Calls outside any ``partition_open`` / ``partition_close`` bracket
    (the global analysis-chain pass) are excluded.
    """
    out: dict[int, list[tuple]] = {}
    cur: int | None = None
    for name, args, kwargs in rec.calls:
        if name == "partition_open":
            cur = int(args[0])
            out.setdefault(cur, [])
        elif name == "partition_close":
            cur = None
        elif cur is not None:
            out[cur].append((name, args, kwargs))
    return out


def _node_emits_by_tag(
    calls: list[tuple],
) -> dict[int, dict]:
    """Return ``{tag: kwargs}`` for every ``node(tag, *coords, ...)``
    call in ``calls`` (one rank's bucket).  Asserts each tag appears
    at most once per rank — duplicate emits within one rank's block
    would be a bridge bug, not a test condition we want to mask.
    """
    out: dict[int, dict] = {}
    for name, args, kwargs in calls:
        if name != "node":
            continue
        tag = int(args[0])
        assert tag not in out, (
            f"node {tag} emitted twice within one rank's block: "
            f"first={out[tag]!r}, second={kwargs!r}"
        )
        out[tag] = dict(kwargs)
    return out


# ---------------------------------------------------------------------------
# Fixture — 4-node shell-on-solid interface with one cross-partition equalDOF.
# ---------------------------------------------------------------------------


def _make_shell_on_solid_partitioned_fem(
    *, per_node_ndf: dict[int, int],
) -> FEMStub:
    """Build the canonical S5 partitioned shell-on-solid fixture.

    Geometry: 4 nodes (a tile) split into two columns by rank.

      * rank 0 (shell-owned): nodes 1, 2 — declared ``ndf=6``
      * rank 1 (solid-owned): nodes 3, 4 — declared ``ndf=3``

    Each rank carries one degenerate "element" so the bridge has
    something to iterate over.  A single cross-partition
    ``equalDOF(master=2, slave=4, dofs=[1, 2, 3])`` ties the shell
    interface to the solid interface — making node 2 foreign on rank
    1 and node 4 foreign on rank 0, which is exactly the path the
    per-foreign-node ``_emit_node_with_broker_ndf`` call at
    ``build.py::emit_mp_constraints_partitioned`` covers.

    Parameters
    ----------
    per_node_ndf
        ``{node_id: ndf}`` mapping installed on the stub via
        :meth:`_NodesStub.set_per_node_ndf`.  Pass ``{}`` to leave the
        broker in the "no declarations" state (used by the no-overrides
        regression test).
    """
    nodes = _NodesStub(
        ids=[1, 2, 3, 4],
        coords=[
            (0.0, 0.0, 1.0),  # 1 — shell, rank 0
            (1.0, 0.0, 1.0),  # 2 — shell interface, rank 0
            (1.0, 0.0, 0.0),  # 3 — solid interface, rank 1
            (0.0, 0.0, 0.0),  # 4 — solid, rank 1
        ],
        node_pgs={
            "shell_pg": [1, 2],
            "solid_pg": [3, 4],
        },
    )
    elements = _ElementsStub(
        elem_pgs={
            "shell_pg": _ElementGroupView(
                ids=(1,), connectivity=((1, 2),),
            ),
            "solid_pg": _ElementGroupView(
                ids=(2,), connectivity=((3, 4),),
            ),
        },
    )
    stub = FEMStub(nodes=nodes, elements=elements)
    stub.set_partitions([
        (0, [1, 2], [1]),
        (1, [3, 4], [2]),
    ])
    if per_node_ndf:
        stub.nodes.set_per_node_ndf(per_node_ndf)
    return stub


# ---------------------------------------------------------------------------
# 1. Headline test — per-rank ndf emit semantics with per-foreign-node lookup.
# ---------------------------------------------------------------------------


def test_emit_partitioned_mixed_ndf_shell_on_solid() -> None:
    """Each rank's deck carries broker-sourced ``ndf`` for both its
    owned nodes AND its foreign-node decls (S5 — closes the shell-to-
    solid stream).

    Locked invariants:

    1. Owned nodes use broker ndf (``-ndf 6`` for shell on rank 0;
       ``-ndf 3`` for solid on rank 1).
    2. Foreign nodes use broker ndf via the per-tag lookup at
       ``build.py::emit_mp_constraints_partitioned`` (NOT the scalar
       envelope fallback) — so the shell-foreign on rank 1 emits
       ``ndf=6`` and the solid-foreign on rank 0 emits ``ndf=3``.
    """
    fem = _make_shell_on_solid_partitioned_fem(
        per_node_ndf={1: 6, 2: 6, 3: 3, 4: 3},
    )
    # Cross-partition equalDOF between shell interface (rank 0) and
    # solid interface (rank 1).  Drives the foreign-node decl path on
    # both ranks per ADR 0027 INV-2.
    fem.add_node_constraints([
        NodePairRecord(
            kind=ConstraintKind.EQUAL_DOF,
            master_node=2, slave_node=4,
            dofs=[1, 2, 3],
            name="interface_tie",
        ),
    ])

    ops = apeSees(cast("object", fem))
    # Envelope must cover the broker max (6); validator enforces this.
    ops.model(ndm=3, ndf=6)

    bm = ops.build()
    rec = RecordingEmitter()
    bm.emit(rec)
    per_rank = _per_rank_calls(rec)
    assert set(per_rank.keys()) == {0, 1}, (
        f"expected per-rank buckets for ranks 0 and 1; got "
        f"{sorted(per_rank.keys())}"
    )

    rank0_nodes = _node_emits_by_tag(per_rank[0])
    rank1_nodes = _node_emits_by_tag(per_rank[1])

    # --- Rank 0 (shell): owned 1, 2 with ndf=6; foreign 4 with ndf=3 ----
    assert set(rank0_nodes.keys()) == {1, 2, 4}, (
        f"rank 0 must emit nodes {{1, 2}} owned + {{4}} foreign; "
        f"got {sorted(rank0_nodes.keys())}"
    )
    assert rank0_nodes[1] == {"ndf": 6}, (
        f"rank 0 owned shell node 1: expected ndf=6, got "
        f"{rank0_nodes[1]!r}"
    )
    assert rank0_nodes[2] == {"ndf": 6}, (
        f"rank 0 owned shell node 2: expected ndf=6, got "
        f"{rank0_nodes[2]!r}"
    )
    assert rank0_nodes[4] == {"ndf": 3}, (
        f"rank 0 foreign solid node 4: expected ndf=3 from the per-"
        f"foreign-node broker lookup at "
        f"build.py::emit_mp_constraints_partitioned, got "
        f"{rank0_nodes[4]!r}.  A value of 6 would mean the lookup "
        f"silently fell back to the bridge envelope — masking the "
        f"per-tag wiring."
    )

    # --- Rank 1 (solid): owned 3, 4 with ndf=3; foreign 2 with ndf=6 ----
    assert set(rank1_nodes.keys()) == {2, 3, 4}, (
        f"rank 1 must emit nodes {{3, 4}} owned + {{2}} foreign; "
        f"got {sorted(rank1_nodes.keys())}"
    )
    assert rank1_nodes[3] == {"ndf": 3}, (
        f"rank 1 owned solid node 3: expected ndf=3 (broker override "
        f"against the ndf=6 envelope), got {rank1_nodes[3]!r}"
    )
    assert rank1_nodes[4] == {"ndf": 3}, (
        f"rank 1 owned solid node 4: expected ndf=3 (broker override), "
        f"got {rank1_nodes[4]!r}"
    )
    assert rank1_nodes[2] == {"ndf": 6}, (
        f"rank 1 foreign shell node 2: expected ndf=6 from the per-"
        f"foreign-node broker lookup, got {rank1_nodes[2]!r}"
    )


# ---------------------------------------------------------------------------
# 2. Foreign-node decl precedes the constraint line on each rank (INV-2).
# ---------------------------------------------------------------------------


def test_partitioned_foreign_node_decl_precedes_constraint() -> None:
    """The per-foreign-node broker lookup (S5) fires BEFORE the
    ``equalDOF`` line on every rank that emits the constraint
    (preserves ADR 0027 INV-2: foreign node decl must precede the
    constraint that references it).
    """
    fem = _make_shell_on_solid_partitioned_fem(
        per_node_ndf={1: 6, 2: 6, 3: 3, 4: 3},
    )
    fem.add_node_constraints([
        NodePairRecord(
            kind=ConstraintKind.EQUAL_DOF,
            master_node=2, slave_node=4,
            dofs=[1, 2, 3],
            name="interface_tie",
        ),
    ])

    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    bm = ops.build()
    rec = RecordingEmitter()
    bm.emit(rec)
    per_rank = _per_rank_calls(rec)

    def _first_index(calls: list[tuple], pred) -> int:
        for i, (name, args, kwargs) in enumerate(calls):
            if pred(name, args, kwargs):
                return i
        return -1

    for rank, foreign_tag in ((0, 4), (1, 2)):
        calls = per_rank[rank]
        decl_idx = _first_index(
            calls,
            lambda n, a, _k, t=foreign_tag: (
                n == "node" and len(a) >= 1 and int(a[0]) == t
            ),
        )
        ed_idx = _first_index(
            calls,
            lambda n, _a, _k: n == "equalDOF",
        )
        assert decl_idx != -1, (
            f"rank {rank}: foreign node {foreign_tag} was not declared"
        )
        assert ed_idx != -1, (
            f"rank {rank}: equalDOF line was not emitted"
        )
        assert decl_idx < ed_idx, (
            f"INV-2 regressed on rank {rank}: foreign node "
            f"{foreign_tag} decl at index {decl_idx} must precede "
            f"equalDOF at index {ed_idx}"
        )


# ---------------------------------------------------------------------------
# 3. Phantom-tag predicate set is populated correctly per rank.
# ---------------------------------------------------------------------------


def test_partitioned_phantom_predicate_set_empty_when_no_n2s() -> None:
    """The phantom-tag predicate set
    (``set_phantom_node_tags`` / ``ATTR_PHANTOM_NODE_TAGS``) installed
    by ``emit_mp_constraints_partitioned`` contains ONLY broker-
    synthetic phantom tags, never real broker node tags.

    This fixture has no :class:`NodeToSurfaceRecord` so the predicate
    must be the empty frozenset on every rank after the constraint
    pass — guards against the regression where a real broker tag is
    accidentally classified as a phantom and emitted twice with
    ``ndf=6``.
    """
    fem = _make_shell_on_solid_partitioned_fem(
        per_node_ndf={1: 6, 2: 6, 3: 3, 4: 3},
    )
    fem.add_node_constraints([
        NodePairRecord(
            kind=ConstraintKind.EQUAL_DOF,
            master_node=2, slave_node=4,
            dofs=[1, 2, 3],
            name="interface_tie",
        ),
    ])
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    bm = ops.build()
    rec = RecordingEmitter()
    bm.emit(rec)

    # The MP-constraint pass set the predicate on the emitter (and
    # never reset it across ranks because it is stateless ADR 0033 —
    # per-rank brokers carry identical phantom-tag sets per ADR 0027
    # INV-3).  No NodeToSurfaceRecord exists, so the set must be
    # empty.
    predicate = getattr(rec, ATTR_PHANTOM_NODE_TAGS, None)
    assert predicate is not None, (
        "emit_mp_constraints_partitioned must install the phantom-tag "
        "predicate on the emitter (ADR 0033 / set_phantom_node_tags)"
    )
    assert predicate == frozenset(), (
        f"no NodeToSurfaceRecord exists; the phantom-tag predicate "
        f"must be the empty frozenset.  Got {predicate!r}.  A non-"
        f"empty set means a real broker tag was classified as a "
        f"phantom — bug in plan_rank_constraints or "
        f"_gather_phantom_nodes."
    )


# ---------------------------------------------------------------------------
# 4. Cross-rank broker consistency — fem._ndf is shared across ranks.
# ---------------------------------------------------------------------------


def test_partitioned_per_node_ndf_consistent_across_ranks() -> None:
    """Per ADR 0033 §"OpenSeesMP consistency is hash-guaranteed":
    every rank's per-rank broker deserialises the same ``_ndf`` array,
    so the per-foreign-node ndf for any shared node is the same on
    every emitting rank.

    Under the apeGmsh stub the broker is a single object (not
    re-deserialised per rank) so the hash-fold path is not exercised
    here — but the *consumer* invariant (rank 0's foreign-node ndf
    for tag T equals rank 1's owned-node ndf for tag T) IS the
    observable surface of that contract, and IS testable on the
    recorded emit calls.  This pins it.
    """
    fem = _make_shell_on_solid_partitioned_fem(
        per_node_ndf={1: 6, 2: 6, 3: 3, 4: 3},
    )
    fem.add_node_constraints([
        NodePairRecord(
            kind=ConstraintKind.EQUAL_DOF,
            master_node=2, slave_node=4,
            dofs=[1, 2, 3],
            name="interface_tie",
        ),
    ])
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    bm = ops.build()
    rec = RecordingEmitter()
    bm.emit(rec)
    per_rank = _per_rank_calls(rec)

    rank0_nodes = _node_emits_by_tag(per_rank[0])
    rank1_nodes = _node_emits_by_tag(per_rank[1])

    # Tag 4 — owned on rank 1, foreign on rank 0.
    assert rank0_nodes[4].get("ndf") == rank1_nodes[4].get("ndf") == 3, (
        f"tag 4 must emit identical ndf on every rank that declares "
        f"it (rank 0 foreign={rank0_nodes[4]!r}, rank 1 owned="
        f"{rank1_nodes[4]!r}); broker per-node consistency regressed"
    )
    # Tag 2 — owned on rank 0, foreign on rank 1.
    assert rank0_nodes[2].get("ndf") == rank1_nodes[2].get("ndf") == 6, (
        f"tag 2 must emit identical ndf on every rank that declares "
        f"it (rank 0 owned={rank0_nodes[2]!r}, rank 1 foreign="
        f"{rank1_nodes[2]!r}); broker per-node consistency regressed"
    )


# ---------------------------------------------------------------------------
# 5. Backcompat — uniform-ndf partitioned emit produces NO per-node tokens.
# ---------------------------------------------------------------------------


def test_partitioned_byte_identical_when_no_overrides() -> None:
    """A partitioned model with NO ``g.node_ndf`` declarations emits
    byte-identical decks pre-S2 and post-S2.

    Per ADR 0033 §"Zero user-facing migration cost":

    * **Owned nodes** — NO per-node ``-ndf`` token (the model
      directive's envelope supplies it).  Pre-S2 owned-node decls
      had no ``-ndf`` either, so byte-identicality holds.
    * **Foreign nodes** — emit ``-ndf <envelope>`` as the documented
      fallback (see :func:`_emit_node_with_broker_ndf` docstring +
      ``emit_mp_constraints_partitioned`` step 3).  Pre-S2 emitted
      ``node(tag, *xyz, ndf=foreign_node_ndf)`` the SAME way, so
      byte-identicality holds on foreign-decl sites too.

    Locks S2 backcompat on the partitioned path so existing decks
    (and the ~285 existing ``apeSees(fem)`` test sites that touch
    partitioned models) keep producing identical Tcl / Py output.
    """
    fem = _make_shell_on_solid_partitioned_fem(per_node_ndf={})
    fem.add_node_constraints([
        NodePairRecord(
            kind=ConstraintKind.EQUAL_DOF,
            master_node=2, slave_node=4,
            dofs=[1, 2, 3],
            name="interface_tie",
        ),
    ])
    ops = apeSees(cast("object", fem))
    envelope = 3
    ops.model(ndm=3, ndf=envelope)
    bm = ops.build()
    rec = RecordingEmitter()
    bm.emit(rec)
    per_rank = _per_rank_calls(rec)

    # Per-rank owned vs foreign sets — derived from the fixture.
    rank_owned = {0: {1, 2}, 1: {3, 4}}
    for rank, calls in per_rank.items():
        nodes = _node_emits_by_tag(calls)
        owned = rank_owned[rank]
        for tag, kwargs in nodes.items():
            if tag in owned:
                # Owned-node decls MUST NOT carry per-node ndf —
                # the envelope wins via the model directive.
                assert "ndf" not in kwargs, (
                    f"rank {rank} OWNED node {tag}: emitted "
                    f"ndf={kwargs.get('ndf')!r} despite no g.node_ndf "
                    f"declarations.  Envelope (apeSees.model(ndf="
                    f"{envelope})) must win for undeclared owned nodes."
                )
            else:
                # Foreign-node decls MUST carry the envelope ndf as
                # the documented fallback (matches pre-S2 behaviour).
                assert kwargs.get("ndf") == envelope, (
                    f"rank {rank} FOREIGN node {tag}: expected "
                    f"ndf={envelope} (envelope fallback per ADR 0033 / "
                    f"_emit_node_with_broker_ndf docstring), got "
                    f"{kwargs!r}.  Pre-S2 emitted the envelope as the "
                    f"foreign-decl ndf; backcompat regressed if this "
                    f"is missing."
                )


# ---------------------------------------------------------------------------
# 6. Validator fires at apeSees.model() under partitioned construction path.
# ---------------------------------------------------------------------------


def test_partitioned_validator_fires_at_apesees_model() -> None:
    """The envelope-vs-broker validator at ``apeSees.model`` fires
    BEFORE the partitioned emit path splits the broker — so a too-
    small envelope is caught at the user's ``ops.model(...)`` call,
    not later in the per-rank emit fan-out.

    Pins ADR 0033 §"Validator at three sites" for the partitioned
    construction path: even though the broker eventually splits into
    per-rank views (each with its own max ndf), the validator sees
    the full broker's max and the user gets a clear ``BridgeError``
    naming the offending node.
    """
    import pytest

    from apeGmsh.opensees._internal.build import BridgeError

    fem = _make_shell_on_solid_partitioned_fem(
        per_node_ndf={1: 6, 2: 6, 3: 3, 4: 3},
    )
    ops = apeSees(cast("object", fem))
    # ndf=3 envelope cannot host the shell nodes (broker max=6).
    with pytest.raises(BridgeError) as exc_info:
        ops.model(ndm=3, ndf=3)
    msg = str(exc_info.value)
    assert "ndf=6" in msg, (
        f"BridgeError must name the declared per-node ndf=6: {msg!r}"
    )
    assert "raise" in msg or "model" in msg, (
        f"BridgeError must name the fix (raise envelope ndf via "
        f"apeSees.model): {msg!r}"
    )
