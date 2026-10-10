"""ADR 0120 D1 — node-pair springs and element-less nodes under partitioned emit.

A shell plate on a spring bed, Gmsh-partitioned into 2 and 4 ranks. Each
grid node carries one spring (``zeroLength`` from a fixed, decoupled ground
node), in one of the two forms a ground-spring bed takes:

* **side form** — the spring reaches a decoupled 3-dof *side* node, tied to
  the 6-dof plate node by ``equalDOF 1 2 3`` (a broker record);
* **direct form** — the spring reaches the plate node itself.

Oracles, each independent of the code under test:

* a decoupled node is in no Gmsh partition, so the rank of a spring is the
  lowest partition that holds its plate node (read off ``fem.partitions``);
* every spring, its ground and side nodes, the ground ``fix`` and the
  side ``equalDOF`` are written exactly once, all in that rank's block — a
  spring line whose nodes are declared on another rank cannot run;
* a ground node's ``mass`` is written once (it is additive under MP);
* a node-pair between two plate nodes held by no common rank raises
  instead of being written on a rank that lacks an endpoint;
* routing does not touch the snapshot the bridge was built on.

The runtime oracle (serial vs 2 / 4 ranks, eigen and transient) is
``tests/opensees/subprocess/test_partitioned_spring_bed_compose_twin.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

import gmsh
from apeGmsh import apeGmsh
from apeGmsh._kernel.records._constraints import NodePairRecord
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build import (
    BridgeError,
    RoutedPartitionSet,
    route_element_less_nodes,
)
from apeGmsh.opensees.element.zero_length import ZeroLengthMatDir
from apeGmsh.opensees.emitter.recording import RecordingEmitter

LX, LY, NX, NY = 6.0, 4.0, 6, 4
GRID = [(i * LX / NX, j * LY / NY) for i in range(NX + 1) for j in range(NY + 1)]


def plate_on_springs(n_parts: int):
    """The plate FEM, its decoupled ``{k: handle}`` side / ground nodes."""
    with apeGmsh(model_name=f"bed_p{n_parts}", verbose=False) as g:
        g.model.geometry.add_rectangle(0, 0, 0, LX, LY, label="plate")
        g.physical.add_surface("plate", name="Plate")
        sides, grounds = {}, {}
        for k, (x, y) in enumerate(GRID):
            if k % 2 == 0:
                sides[k] = g.decouple_node(coords=(x, y, 0.0), label=f"side_{k}")
            grounds[k] = g.decouple_node(coords=(x, y, -1.0), label=f"gnd_{k}")
        surf = gmsh.model.getEntities(2)[0][1]
        for _d, c in gmsh.model.getBoundary([(2, surf)], oriented=False):
            b = gmsh.model.getBoundingBox(1, abs(c))
            gmsh.model.mesh.setTransfiniteCurve(
                abs(c), NX + 1 if abs(b[3] - b[0]) > 1e-3 else NY + 1)
        gmsh.model.mesh.setTransfiniteSurface(surf)
        gmsh.model.mesh.setRecombine(2, surf)
        g.mesh.generation.generate(dim=2)
        if n_parts > 1:
            g.mesh.partitioning.partition(n_parts)
        fem = g.mesh.queries.get_fem_data(dim=2)
    return fem, sides, grounds


def mesh_node_at(fem, x: float, y: float) -> int:
    dec = {int(n) for n in fem.nodes.decoupled_ids}
    for n, c in zip(fem.nodes.ids, np.asarray(fem.nodes.coords)):
        if int(n) not in dec and np.allclose(c, (x, y, 0.0), atol=1e-9):
            return int(n)
    raise AssertionError(f"no plate node at {(x, y)}")


def declare(fem, sides, grounds, *, ground_mass: bool = False):
    """The bed on ``fem`` (side ties added as broker records) -> ``(ops, fem)``."""
    for k, side in sides.items():
        fem = fem.with_constraint(NodePairRecord(
            kind="equal_dof", master_node=mesh_node_at(fem, *GRID[k]),
            slave_node=int(side.tag), dofs=[1, 2, 3]))
    ops = apeSees(fem, _artifacts=False)
    ops.model(ndm=3, ndf=6)
    sec = ops.section.ElasticMembranePlateSection(E=30e9, nu=0.2, h=0.4, rho=2400.0)
    ops.element.ASDShellQ4(pg="Plate", section=sec)
    spring = ops.uniaxialMaterial.ElasticMaterial(E=2.0e8, eta=4.0e6)
    dirs = tuple(ZeroLengthMatDir(material=spring, dof=d) for d in (1, 2, 3))
    for k, (x, y) in enumerate(GRID):
        j = sides[k] if k in sides else mesh_node_at(fem, x, y)
        ops.element.ZeroLength(nodes=(grounds[k], j), mat_dirs=dirs)
        ops.fix(nodes=[int(grounds[k].tag)], dofs=(1, 1, 1))
        ops.ndf(grounds[k], ndf=3)
        if k in sides:
            ops.ndf(sides[k], ndf=3)
        if ground_mass:
            ops.mass(nodes=[int(grounds[k].tag)], values=(1.0, 1.0, 1.0))
    return ops, fem


def per_rank(ops) -> dict[int, list[tuple]]:
    rec = RecordingEmitter()
    ops.build().emit(rec)
    out: dict[int, list[tuple]] = {}
    cur = None
    for name, args, _kw in rec.calls:
        if name == "partition_open":
            cur = int(args[0])
            out.setdefault(cur, [])
        elif name == "partition_close":
            cur = None
        elif cur is not None:
            out[cur].append((name, args))
    return out


def lowest_rank(fem, node: int) -> int:
    ranks = [idx for idx, p in enumerate(fem.partitions)
             if node in set(int(n) for n in p.node_ids)]
    assert ranks, f"node {node} is in no partition"
    return min(ranks)


@pytest.mark.parametrize("n_parts", [2, 4])
def test_each_spring_lands_whole_on_the_lowest_rank_of_its_plate_node(n_parts):
    fem, sides, grounds = plate_on_springs(n_parts)
    assert len(fem.partitions) == n_parts
    ops, fem = declare(fem, sides, grounds, ground_mass=True)
    blocks = per_rank(ops)

    springs = {r: [a for n, a in calls if n == "element" and a[0] == "zeroLength"]
               for r, calls in blocks.items()}
    declared = {r: {int(a[0]) for n, a in calls if n == "node"}
                for r, calls in blocks.items()}
    fixes = {r: [int(a[0]) for n, a in calls if n == "fix"] for r, calls in blocks.items()}
    masses = {r: [int(a[0]) for n, a in calls if n == "mass"] for r, calls in blocks.items()}
    ties = {r: [(int(a[0]), int(a[1])) for n, a in calls if n == "equalDOF"]
            for r, calls in blocks.items()}
    assert sum(len(v) for v in springs.values()) == len(GRID)

    for k, (x, y) in enumerate(GRID):
        plate = mesh_node_at(fem, x, y)
        g_tag = int(grounds[k].tag)
        j_tag = int(sides[k].tag) if k in sides else plate
        want = lowest_rank(fem, plate)
        where = [r for r, lines in springs.items()
                 if any((int(a[2]), int(a[3])) == (g_tag, j_tag) for a in lines)]
        assert where == [want], (k, where, want)
        # the spring's nodes exist on its rank, the decoupled ones nowhere else
        assert {g_tag, j_tag} <= declared[want]
        for r in blocks:
            if r != want:
                assert g_tag not in declared[r]
                if k in sides:
                    assert j_tag not in declared[r]
        assert [r for r in blocks if g_tag in fixes[r]] == [want]
        assert [r for r in blocks if g_tag in masses[r]] == [want]
        if k in sides:
            # the side tie is written once, beside its spring: no
            # equalDOF crosses ranks
            assert [r for r in blocks if (plate, j_tag) in ties[r]] == [want]


def test_routing_leaves_the_snapshot_alone_and_pins_every_decoupled_node():
    fem, sides, grounds = plate_on_springs(2)
    ops, fem = declare(fem, sides, grounds)
    before = [np.asarray(p.node_ids).copy() for p in fem.partitions]
    bm = ops.build()
    assert isinstance(bm.fem.partitions, RoutedPartitionSet)
    assert bm.fem is not fem
    assert [np.asarray(p.node_ids).tolist() for p in fem.partitions] == [
        b.tolist() for b in before]
    pinned = bm.fem.partitions.pinned
    assert set(pinned) == {int(n) for n in fem.nodes.decoupled_ids}
    # the archive's partition back-store is the mesh partition, not the routing
    assert bm.fem.nodes._partitions is fem.nodes._partitions


def test_nothing_to_route_returns_the_snapshot_itself():
    # every node held by a partition: nothing to route
    with apeGmsh(model_name="bare_p2", verbose=False) as g:
        g.model.geometry.add_rectangle(0, 0, 0, LX, LY, label="plate")
        g.physical.add_surface("plate", name="Plate")
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(dim=2)
        g.mesh.partitioning.partition(2)
        bare = g.mesh.queries.get_fem_data(dim=2)
    assert route_element_less_nodes(bare, []) is bare
    # an unpartitioned snapshot is never routed
    one, _s, gr = plate_on_springs(1)
    assert route_element_less_nodes(one, [(int(gr[0].tag), int(gr[1].tag))]) is one
    # an unlinked decoupled node still gets the first rank
    fem, _sides, _grounds = plate_on_springs(2)
    routed = route_element_less_nodes(fem, [])
    assert set(routed.partitions.pinned.values()) == {0}


def test_a_node_pair_between_two_ranks_raises():
    fem, sides, grounds = plate_on_springs(2)
    parts = [set(int(n) for n in p.node_ids) for p in fem.partitions]
    only0 = sorted(parts[0] - parts[1])
    only1 = sorted(parts[1] - parts[0])
    ops = apeSees(fem, _artifacts=False)
    ops.model(ndm=3, ndf=6)
    sec = ops.section.ElasticMembranePlateSection(E=30e9, nu=0.2, h=0.4, rho=2400.0)
    ops.element.ASDShellQ4(pg="Plate", section=sec)
    spring = ops.uniaxialMaterial.ElasticMaterial(E=1.0e6)
    ops.element.ZeroLength(nodes=(only0[0], only1[0]),
                           mat_dirs=(ZeroLengthMatDir(material=spring, dof=1),))
    with pytest.raises(BridgeError, match="no rank that holds both endpoints"):
        per_rank(ops)
