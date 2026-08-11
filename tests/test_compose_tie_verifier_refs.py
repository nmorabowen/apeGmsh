"""The compose verifier's cover-set check (check 3) sees tie records.

``_bundle_constraint_refs`` yields a :class:`ConstraintReference` per
tag-bearing field on every bundle constraint record; check 3 then
confirms each lands inside the module's reservation window (cover-set
drift detection, ADR 0038).  Reinforce ties (ADR 0067 P5.1) and embed
ties (ADR 0073 g.embed) were carried through compose but absent from
the ref stream — an out-of-window tie tag (a rewrite that missed a
field) passed the verifier silently.  These tests pin the tie streams
into the verifier's view: a hand-built bundle with an out-of-window
tie tag must trip ``tag_collision_verify``; in-window ties must pass.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh._kernel.records._constraints import (
    EmbedTieRecord,
    ReinforceTieRecord,
)
from apeGmsh.core._compose_errors import ComposeInvariantError
from apeGmsh.core._tag_collision_verifier import (
    ImportedRecords,
    ReservationRecord,
    tag_collision_verify,
)
from apeGmsh.mesh._compose import _bundle_constraint_refs, _RewrittenBundle


def _min_bundle(**over) -> _RewrittenBundle:
    """A minimal hand-built bundle — every stream empty unless overridden."""
    base = dict(
        label="A",
        source_path="mem://none",
        source_fem_hash="0",
        source_neutral_schema_version="0.0.0",
        translate=(0.0, 0.0, 0.0),
        rotate=None,
        partition_rank=None,
        properties={},
        composed_at="1970-01-01T00:00:00+00:00",
        base=1_000,
        size=1_000,
        source_span=10,
        source_min_tag=1,
        node_ids=np.array([], dtype=np.int64),
        node_coords=np.zeros((0, 3)),
        node_ndf=None,
        element_groups={},
        node_physical={},
        elem_physical={},
        node_labels={},
        elem_labels={},
        mesh_selection=None,
        part_node_map={},
        part_elem_map={},
        node_constraints=(),
        elem_constraints=(),
        nodal_loads=(),
        element_loads=(),
        sp_records=(),
    )
    base.update(over)
    return _RewrittenBundle(**base)


def _rtie(rebar_node, host_nodes):
    return ReinforceTieRecord(
        kind="reinforce", rebar_node=rebar_node, host_nodes=list(host_nodes),
        weights=np.full(len(host_nodes), 1.0 / len(host_nodes)),
        direction=np.array([0.0, 0.0, 1.0]), perfect=1.0e12)


def _etie(node, host_nodes):
    return EmbedTieRecord(
        kind="embed", node=node, host_nodes=list(host_nodes),
        weights=np.full(len(host_nodes), 1.0 / len(host_nodes)),
        k=1.0e12)


def _verify(bundle):
    """Feed the bundle's refs into check 3 the way _run_compose_verifier
    does: one reservation, the bundle's own window."""
    res = ReservationRecord(
        label=bundle.label, base=bundle.base, size=bundle.size)
    tag_collision_verify(
        reservations=(res,),
        host_pg_names=(),
        module_imports={
            bundle.label: ImportedRecords(
                tags=(),
                constraint_refs=tuple(_bundle_constraint_refs(bundle)),
                source_span=bundle.source_span,
            ),
        },
    )


def test_refs_include_reinforce_and_embed_tie_tags():
    bundle = _min_bundle(
        reinforce_ties=(_rtie(1_100, [1_101, 1_102, 1_103, 1_104]),),
        embed_ties=(_etie(1_200, [1_201, 1_202]),),
    )
    refs = list(_bundle_constraint_refs(bundle))
    tags = {(r.kind, r.tag) for r in refs}
    assert ("ReinforceTieRecord", 1_100) in tags        # rebar_node scalar
    assert ("ReinforceTieRecord", 1_103) in tags        # host_nodes array
    assert ("EmbedTieRecord", 1_200) in tags            # node scalar
    assert ("EmbedTieRecord", 1_202) in tags            # host_nodes array
    fields = {r.field_name for r in refs}
    assert {"rebar_node", "node", "host_nodes[0]"} <= fields


def test_out_of_window_reinforce_tie_trips_verifier():
    # host_nodes[1] = 5 is below the [1000, 2000) window — a cover-set
    # miss that previously sailed through unverified.
    bundle = _min_bundle(
        reinforce_ties=(_rtie(1_100, [1_101, 5, 1_103, 1_104]),),
    )
    with pytest.raises(ComposeInvariantError, match=r"host_nodes\[1\]"):
        _verify(bundle)


def test_out_of_window_embed_tie_trips_verifier():
    # constrained node 999_999 escapes the window entirely.
    bundle = _min_bundle(
        embed_ties=(_etie(999_999, [1_201, 1_202]),),
    )
    with pytest.raises(ComposeInvariantError, match=r"\b999999\b"):
        _verify(bundle)


def test_in_window_ties_pass_verifier():
    bundle = _min_bundle(
        reinforce_ties=(_rtie(1_100, [1_101, 1_102, 1_103, 1_104]),),
        embed_ties=(_etie(1_200, [1_201, 1_202]),),
    )
    _verify(bundle)                                     # no raise
