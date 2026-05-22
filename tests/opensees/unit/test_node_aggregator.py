"""Unit tests for the Node aggregator (Phase 5A).

Covers :class:`Node`, :class:`NodeSet`, and :class:`_NodeAccessor`
(exposed as ``ops.nodes``). The accessor is tested against the
hand-rolled FEM stub from :mod:`tests.opensees.fixtures.fem_stub`.
"""
from __future__ import annotations

from typing import cast

import pytest

from apeGmsh.opensees import Node, NodeSet, apeSees

from tests.opensees.fixtures.fem_stub import (
    make_two_column_frame,
    make_two_node_beam,
)


# ===========================================================================
# Node — single-node aggregate
# ===========================================================================

class TestNodeConstruction:
    def test_tag_and_coords_exposed(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        n = ops.nodes.get(tag=2)
        assert isinstance(n, Node)
        assert n.tag == 2
        assert n.coords == (0.0, 0.0, 1.0)

    def test_repr_includes_tag_and_coords(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        n = ops.nodes.get(tag=1)
        r = repr(n)
        assert "Node" in r
        assert "tag=1" in r
        assert "0.0" in r

    def test_int_coercion_returns_tag(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        n = ops.nodes.get(tag=2)
        assert int(n) == 2

    def test_equality_on_tag_and_bridge(self) -> None:
        fem = make_two_node_beam()
        ops_a = apeSees(cast("object", fem))  # type: ignore[arg-type]
        ops_b = apeSees(cast("object", fem))  # type: ignore[arg-type]
        n_a1 = ops_a.nodes.get(tag=2)
        n_a2 = ops_a.nodes.get(tag=2)
        n_b = ops_b.nodes.get(tag=2)
        assert n_a1 == n_a2          # same bridge, same tag
        assert n_a1 != n_b           # different bridge, same tag
        # ... and hashable so they can live in sets / dicts:
        assert len({n_a1, n_a2}) == 1
        assert len({n_a1, n_b}) == 2

    def test_get_with_missing_tag_raises(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        with pytest.raises(KeyError, match="no such node"):
            ops.nodes.get(tag=999)


# ===========================================================================
# Node verbs — fix / mass route through the bridge
# ===========================================================================

class TestNodeVerbs:
    def test_node_fix_records_via_bridge(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        ops.model(ndm=3, ndf=6)
        n = ops.nodes.get(tag=1)
        n.fix(dofs=(1, 1, 1, 1, 1, 1))

        # The fix landed on the bridge's record list as nodes=(1,).
        assert len(ops._fix_records) == 1
        rec = ops._fix_records[0]
        assert rec.pg is None
        assert rec.nodes == (1,)
        assert rec.dofs == (1, 1, 1, 1, 1, 1)

    def test_node_mass_records_via_bridge(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        ops.model(ndm=3, ndf=6)
        n = ops.nodes.get(tag=2)
        n.mass(values=(50.0, 50.0, 50.0, 0.0, 0.0, 0.0))

        assert len(ops._mass_records) == 1
        rec = ops._mass_records[0]
        assert rec.pg is None
        assert rec.nodes == (2,)
        assert rec.values == (50.0, 50.0, 50.0, 0.0, 0.0, 0.0)


# ===========================================================================
# NodeSet — multi-node aggregate
# ===========================================================================

class TestNodeSetFromPG:
    def test_get_by_pg_returns_nodeset(self) -> None:
        fem = make_two_column_frame()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        base = ops.nodes.get(pg="Base")
        assert isinstance(base, NodeSet)
        assert len(base) == 2
        assert base.tags == (1, 3)

    def test_nodeset_iterates_as_nodes(self) -> None:
        fem = make_two_column_frame()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        base = ops.nodes.get(pg="Base")
        nodes = list(base)
        assert len(nodes) == 2
        assert all(isinstance(n, Node) for n in nodes)
        assert {n.tag for n in nodes} == {1, 3}

    def test_nodeset_fix_applies_to_all_members(self) -> None:
        fem = make_two_column_frame()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        ops.model(ndm=3, ndf=6)
        ops.nodes.get(pg="Base").fix(dofs=(1, 1, 1, 1, 1, 1))

        # Single record with both node tags — no PG fan-out needed at
        # emit time because the resolution happened in NodeSet.fix.
        assert len(ops._fix_records) == 1
        rec = ops._fix_records[0]
        assert rec.pg is None
        assert rec.nodes == (1, 3)

    def test_nodeset_mass_applies_to_all_members(self) -> None:
        fem = make_two_column_frame()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        ops.model(ndm=3, ndf=6)
        ops.nodes.get(pg="Top").mass(values=(10.0, 10.0, 10.0, 0.0, 0.0, 0.0))

        assert len(ops._mass_records) == 1
        rec = ops._mass_records[0]
        assert rec.pg is None
        assert rec.nodes == (2, 4)
        assert rec.values == (10.0, 10.0, 10.0, 0.0, 0.0, 0.0)

    def test_empty_pg_nodeset_is_inert(self) -> None:
        """An empty NodeSet's verbs produce no records (the operation
        is well-defined as a no-op)."""

        # An empty stub PG — build one manually.
        from tests.opensees.fixtures.fem_stub import _NodesStub, FEMStub, _ElementsStub
        nodes = _NodesStub(
            ids=[1], coords=[(0.0, 0.0, 0.0)], node_pgs={"Empty": []},
        )
        elements = _ElementsStub(elem_pgs={})
        fem = FEMStub(nodes=nodes, elements=elements)

        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        ops.model(ndm=3, ndf=6)
        ops.nodes.get(pg="Empty").fix(dofs=(1, 0, 0, 0, 0, 0))
        ops.nodes.get(pg="Empty").mass(values=(1.0, 0, 0, 0, 0, 0))
        assert ops._fix_records == []
        assert ops._mass_records == []

    def test_nodeset_indexing_and_contains(self) -> None:
        fem = make_two_column_frame()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        base = ops.nodes.get(pg="Base")
        assert base[0].tag == 1
        assert base[1].tag == 3
        # Membership by tag or by Node.
        assert 1 in base
        assert 3 in base
        assert 2 not in base
        n1 = ops.nodes.get(tag=1)
        assert n1 in base

    def test_nodeset_summary_dataframe(self) -> None:
        fem = make_two_column_frame()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        df = ops.nodes.get(pg="Base").summary()
        assert list(df.columns) == ["tag", "x", "y", "z"]
        assert len(df) == 2
        assert set(df["tag"]) == {1, 3}


# ===========================================================================
# _NodeAccessor (ops.nodes) — container surface
# ===========================================================================

class TestNodeAccessor:
    def test_len_matches_fem(self) -> None:
        fem = make_two_column_frame()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        assert len(ops.nodes) == 4

    def test_iter_yields_every_node(self) -> None:
        fem = make_two_column_frame()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        tags = sorted(n.tag for n in ops.nodes)
        assert tags == [1, 2, 3, 4]

    def test_getitem_by_tag(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        n = ops.nodes[2]
        assert isinstance(n, Node)
        assert n.tag == 2

    def test_contains_by_tag_and_by_node(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        assert 1 in ops.nodes
        assert 999 not in ops.nodes
        n = ops.nodes.get(tag=2)
        assert n in ops.nodes

    def test_get_no_args_returns_all_nodes(self) -> None:
        fem = make_two_column_frame()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        all_nodes = ops.nodes.get()
        assert isinstance(all_nodes, NodeSet)
        assert len(all_nodes) == 4

    def test_get_rejects_both_tag_and_pg(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="supply at most one"):
            ops.nodes.get(tag=1, pg="Base")

    def test_get_with_unknown_pg_raises(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        with pytest.raises(KeyError, match="no such node PG"):
            ops.nodes.get(pg="DoesNotExist")

    def test_accessor_summary_matches_fem_count(self) -> None:
        fem = make_two_column_frame()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        df = ops.nodes.summary()
        assert len(df) == 4
        assert list(df.columns) == ["tag", "x", "y", "z"]


# ===========================================================================
# Plain.load(node=Node) — pattern accepts Node instances
# ===========================================================================

class TestPatternLoadAcceptsNode:
    def test_load_with_node_instance(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        ops.model(ndm=3, ndf=6)

        ts = ops.timeSeries.Linear()
        tip = ops.nodes.get(tag=2)
        with ops.pattern.Plain(series=ts) as p:
            p.load(node=tip, forces=(100.0, 0.0, 0.0, 0.0, 0.0, 0.0))

        # The pattern recorded a node= load with the right tag.
        loads = ops._primitives[-1].loads  # type: ignore[attr-defined]
        assert len(loads) == 1
        assert loads[0].target_kind == "node"
        assert loads[0].target == "2"
        assert loads[0].forces == (100.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def test_load_with_plain_int_still_works(self) -> None:
        """Backward compat — passing an int tag continues to work."""
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        ops.model(ndm=3, ndf=6)

        ts = ops.timeSeries.Linear()
        with ops.pattern.Plain(series=ts) as p:
            p.load(node=2, forces=(100.0, 0.0, 0.0))

        loads = ops._primitives[-1].loads  # type: ignore[attr-defined]
        assert loads[0].target == "2"

    def test_sp_with_node_instance(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        ops.model(ndm=3, ndf=6)

        ts = ops.timeSeries.Linear()
        tip = ops.nodes.get(tag=2)
        with ops.pattern.Plain(series=ts) as p:
            p.sp(node=tip, dof=1, value=0.005)

        sps = ops._primitives[-1].sps  # type: ignore[attr-defined]
        assert len(sps) == 1
        assert sps[0].target_kind == "node"
        assert sps[0].target == "2"
        assert sps[0].value == 0.005


# ===========================================================================
# apeSees.fix / mass with Node + mixed iterables
# ===========================================================================

class TestBridgeFixMassWithNodes:
    def test_fix_with_node_iterable(self) -> None:
        fem = make_two_column_frame()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        ops.model(ndm=3, ndf=6)

        nodes_to_fix = (ops.nodes.get(tag=1), ops.nodes.get(tag=3))
        ops.fix(nodes=nodes_to_fix, dofs=(1, 1, 1, 1, 1, 1))

        rec = ops._fix_records[0]
        assert rec.nodes == (1, 3)

    def test_fix_mixed_node_and_int(self) -> None:
        fem = make_two_column_frame()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        ops.model(ndm=3, ndf=6)

        ops.fix(nodes=(ops.nodes.get(tag=1), 3), dofs=(1, 1, 1, 1, 1, 1))

        rec = ops._fix_records[0]
        assert rec.nodes == (1, 3)

    def test_mass_with_node_iterable(self) -> None:
        fem = make_two_column_frame()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        ops.model(ndm=3, ndf=6)

        ops.mass(
            nodes=(ops.nodes.get(tag=2), ops.nodes.get(tag=4)),
            values=(10.0, 10.0, 10.0, 0.0, 0.0, 0.0),
        )

        rec = ops._mass_records[0]
        assert rec.nodes == (2, 4)


# ===========================================================================
# Region — Node.region / NodeSet.region / apeSees.region
# ===========================================================================

class TestNodeRegion:
    """Node.region(name) routes through the bridge into a
    :class:`RegionAssignmentRecord`.  At emit time the bridge merges
    members by name into one ``region $tag -node ...`` line per name.
    """

    def test_node_region_records_via_bridge(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        ops.model(ndm=3, ndf=6)
        ops.nodes.get(tag=1).region("damping")

        assert len(ops._region_records) == 1
        rec = ops._region_records[0]
        assert rec.name == "damping"
        assert rec.pg is None
        assert rec.nodes == (1,)

    def test_nodeset_region_records_via_bridge(self) -> None:
        fem = make_two_column_frame()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        ops.model(ndm=3, ndf=6)
        ops.nodes.get(pg="Base").region("supports")

        assert len(ops._region_records) == 1
        rec = ops._region_records[0]
        assert rec.name == "supports"
        assert rec.pg is None
        assert rec.nodes == (1, 3)

    def test_empty_nodeset_region_is_inert(self) -> None:
        from tests.opensees.fixtures.fem_stub import (
            FEMStub, _ElementsStub, _NodesStub,
        )
        nodes = _NodesStub(
            ids=[1], coords=[(0.0, 0.0, 0.0)], node_pgs={"Empty": []},
        )
        elements = _ElementsStub(elem_pgs={})
        fem = FEMStub(nodes=nodes, elements=elements)
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        ops.model(ndm=3, ndf=6)
        ops.nodes.get(pg="Empty").region("empty")
        assert ops._region_records == []

    def test_apesees_region_with_pg(self) -> None:
        fem = make_two_column_frame()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        ops.model(ndm=3, ndf=6)
        ops.region(name="top", pg="Top")

        assert len(ops._region_records) == 1
        rec = ops._region_records[0]
        assert rec.name == "top"
        assert rec.pg == "Top"
        assert rec.nodes is None

    def test_apesees_region_accepts_node_instances(self) -> None:
        fem = make_two_column_frame()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        ops.model(ndm=3, ndf=6)
        ops.region(
            name="mixed",
            nodes=(ops.nodes.get(tag=2), 4),
        )

        rec = ops._region_records[0]
        assert rec.nodes == (2, 4)

    def test_apesees_region_requires_pg_xor_nodes(self) -> None:
        fem = make_two_column_frame()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="exactly one of"):
            ops.region(name="x")  # neither
        with pytest.raises(ValueError, match="exactly one of"):
            ops.region(name="x", pg="Top", nodes=(2,))  # both

    def test_apesees_region_requires_nonempty_name(self) -> None:
        fem = make_two_column_frame()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="name= must be non-empty"):
            ops.region(name="", nodes=(2,))


class TestRegionEmitFanOut:
    """The build pipeline merges region records by name into one
    ``emitter.region`` call per name (one freshly allocated tag,
    de-duped node list in first-seen order).
    """

    def test_two_nodes_same_name_emit_one_region(self) -> None:
        from apeGmsh.opensees.emitter.recording import RecordingEmitter

        fem = make_two_column_frame()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        ops.model(ndm=3, ndf=6)
        ops.nodes.get(tag=1).region("damping")
        ops.nodes.get(tag=2).region("damping")

        rec = RecordingEmitter()
        ops.build().emit(rec)
        region_calls = [c for c in rec.calls if c[0] == "region"]
        assert len(region_calls) == 1
        _, args, _ = region_calls[0]
        # args = (tag, '-node', 1, 2)
        tag, flag, *node_tags = args
        assert isinstance(tag, int) and tag > 0
        assert flag == "-node"
        assert tuple(node_tags) == (1, 2)

    def test_two_names_emit_two_regions_with_distinct_tags(self) -> None:
        from apeGmsh.opensees.emitter.recording import RecordingEmitter

        fem = make_two_column_frame()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        ops.model(ndm=3, ndf=6)
        ops.nodes.get(pg="Base").region("supports")
        ops.nodes.get(pg="Top").region("loaded")

        rec = RecordingEmitter()
        ops.build().emit(rec)
        region_calls = [c for c in rec.calls if c[0] == "region"]
        assert len(region_calls) == 2
        tags = [c[1][0] for c in region_calls]
        assert len(set(tags)) == 2  # distinct
        # Insertion order: "supports" first, then "loaded".
        first_nodes = tuple(region_calls[0][1][2:])
        second_nodes = tuple(region_calls[1][1][2:])
        assert first_nodes == (1, 3)  # Base
        assert second_nodes == (2, 4)  # Top

    def test_duplicate_nodes_dedupe_within_name(self) -> None:
        from apeGmsh.opensees.emitter.recording import RecordingEmitter

        fem = make_two_column_frame()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        ops.model(ndm=3, ndf=6)
        # Node 1 is in Base; mention it twice in two ways.
        ops.nodes.get(tag=1).region("damping")
        ops.nodes.get(pg="Base").region("damping")  # includes node 1 again

        rec = RecordingEmitter()
        ops.build().emit(rec)
        region_calls = [c for c in rec.calls if c[0] == "region"]
        assert len(region_calls) == 1
        node_tags = tuple(region_calls[0][1][2:])
        # First-seen order: 1 (from .region(tag=1)), then the new
        # member from Base (3); node 1 from Base is skipped (dedup).
        assert node_tags == (1, 3)

    def test_no_region_records_emits_no_region(self) -> None:
        from apeGmsh.opensees.emitter.recording import RecordingEmitter

        fem = make_two_column_frame()
        ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
        ops.model(ndm=3, ndf=6)
        rec = RecordingEmitter()
        ops.build().emit(rec)
        assert [c for c in rec.calls if c[0] == "region"] == []
