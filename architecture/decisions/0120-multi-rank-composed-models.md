# ADR 0120 — Multi-rank composed models: routed element-less nodes, one-graph repartition, a hosted assembly

**Status:** Proposed (2026-10-10).

**Owner:** nmora

**Builds on** [ADR 0027](0027-cross-partition-mp-constraints.md) (per-rank
emit and cross-partition MP constraints), [ADR 0049](0049-decoupled-nodes.md)
(decoupled nodes, and the node-pair refusal it left deferred),
[ADR 0117](0117-assembly-compose-v2.md) (Assembly, the private merge engine
and its one-rank-per-instance layout) and, when it lands,
ADR 0119 (the ground-spring bed, PR #1614).

## Context

Two San Ramón cells cannot run under OpenSeesMP.

1. **A spring bed is refused under partitioning.** A Tier-2 foundation is
   2 487 `zeroLength` springs, each from a fixed decoupled *ground* node to
   a decoupled 3-dof *side* node tied by `equalDOF` to a mat or wall node
   (ADR 0119). A decoupled node belongs to no element, so it is in no Gmsh
   partition and no rank declares it; a node-pair element has no FEM id, so
   `build_element_partition_owner` cannot place it. `apeSees` therefore
   refuses any node-pair element on a partitioned emit (the ADR 0049
   deferral), and Tier 2 runs serial.
2. **A composed model gets one rank per module.** Tier 4 is a structure FEM
   with a soil box grafted onto it and the building's below-grade nodes
   embedded in the pit faces. The merge engine ranks the host 0 and each
   module next (ADR 0038 rank model, ADR 0117 D6): two ranks, one of them
   holding a 100 k-hex soil box. Partitioning needs a Gmsh session, and the
   composed snapshot has none. The port also reached the merge engine
   through privates (`_compose_module`, `route_def_to_fem`), because
   `g.compose` was removed (ADR 0117 D7) and `Assembly` namespaces every
   instance, which the Tier-1 declarations (bare group names) cannot use.

## Decision

### D1 — Element-less nodes take the rank of what they are tied to

On a partitioned FEM, `apeSees.build()` gives every element-less node one
rank before the emit (`route_element_less_nodes`):

- a node is routed when it is in no partition and is decoupled, an endpoint
  of a node-pair element, or a node of an MP constraint record;
- node-pair elements and MP records join the routed nodes into components;
  a component takes the **lowest rank among the partitioned nodes it
  touches** (the primary-owner rule of ADR 0027's additive quantities), and
  the first rank when it touches none;
- each routed node is added to that one rank's partition in a copy of the
  snapshot the `BuiltModel` holds (`RoutedPartitionSet`, with `pinned =
  {node: rank}`). The snapshot passed to `apeSees` is not modified, and the
  copy shares `nodes._partitions`, so `model.h5` stores the mesh partition,
  not the routing.

Everything keyed by partition membership then follows with no new pass: the
routed node's `node` line, `ndf`, `fix`, `mass` (primary owner), named
regions and ghost SP replay. Two rules are added:

- a **node-pair element** is written on the lowest rank that holds both
  endpoints. A pair whose endpoints share no rank (two mesh nodes of
  different partitions) raises `BridgeError`: it would need a ghost
  endpoint, which is not designed here. A stage-claimed node-pair element on
  a partitioned emit raises too;
- a **node-pair MP record whose slave is pinned** (a spring's side-node
  `equalDOF`) is written on the slave's rank only, not replicated on every
  rank that holds its master (ADR 0027's rule stays for every other record).
  No `equalDOF` of a spring crosses ranks.

For a spring bed this is the routing the port specified (`tier2.MPI_ROUTING`):
spring `i` belongs to the rank owning its structural node; its side and
ground nodes, `zeroLength`, ground `fix`, `ndf` and `equalDOF` are written
there; decoupled-node tags are rank-invariant (ADR 0049); materials are
global, as every material of a partitioned deck already is.

### D2 — `FEMData.repartition(n_parts, *, weights=None)`

A snapshot method that returns a copy split into `n_parts` ranks as **one
graph**: every element, whatever module or instance it came from, by
recursive coordinate bisection of the element centroids, balanced on the
element weights (node count by default; a weight per element type name, or a
callable `f(type_name, ids, centroids)`; zero is allowed). A rank holds the
nodes of its elements, so cut nodes are shared as OpenSeesMP expects; a node
no element references is in no partition and D1 routes it. Partition ids run
`1 .. n` like the mesh partitioner's; `n_parts=1` is the unpartitioned copy.
It replaces whatever partition the snapshot carried, including the merge
engine's one-rank-per-module layout. The input is unchanged and the result
round-trips through `model.h5`.

RCB needs no graph library (pymetis has no Windows wheel) and keeps what is
close together on one rank: an embedded structure node is usually on the
rank of the soil element hosting it. Couplings that do cross a cut are the
cases ADR 0027 already writes (the `ASDEmbeddedNodeElement` on its host's
rank, the foreign node ghosted with its SP replay).

### D3 — A hosted assembly is the public graft

`Assembly(name, host=fem_or_model_h5)` grafts the instances onto an existing
snapshot instead of an empty broker:

- the host keeps its node and element ids and its bare group and label
  names; instances are namespaced and relocated exactly as in ADR 0117 D2;
- a bare port may name a host group or label (`asm.embedded("soil.pit_bottom",
  "Slabs_Foundation_Set")`); a dotted host name is taken whole when its
  prefix is no instance label;
- `Assembly.fem()` (new, on every assembly) returns the merged snapshot
  `bridge()` builds on, without rehydration. A hosted `fem()` is
  unpartitioned (the merge engine's host / module ranks are dropped);
  `asm.fem().repartition(n)` cuts it for MPI;
- a hosted assembly refuses reference nodes (their ids would collide with
  the host's), `partition_rank`, `bridge()` (the host carries no `/opensees`
  content to rehydrate) and `h5()` (there is no `/assembly` row for a host).
  The caller builds `apeSees(asm.fem())` and writes it with `ops.h5()`.

This amends ADR 0117 D1 ("there is no host") for the FEM side only: the
archive zone, rehydration and the namespacing of instances do not change,
and `g.compose` stays removed.

## Invariants

1. **INV-1.** On a partitioned emit, every node-pair element, its two
   endpoints (when element-less), the ground `fix` and a pinned slave's
   `equalDOF` are written in exactly one rank block, the same one.
2. **INV-2.** The rank of a routed node is the lowest rank holding a
   partitioned node of its component, else the first rank; the same
   snapshot gives the same routing.
3. **INV-3.** `apeSees.build()` and `FEMData.repartition` never modify the
   snapshot they are given.
4. **INV-4.** `repartition` assigns every element to exactly one rank, and a
   rank holds exactly the nodes of its elements.
5. **INV-5.** A hosted `fem()` holds the host's node ids, coordinates,
   element ids and names unchanged.
6. **INV-6.** A model emitted serial and on 1, 2 and 4 ranks reproduces the
   serial eigenvalues to 1e-9 and a short transient to round-off.

## Gates (measured 2026-10-10, local OpenSeesMP + Intel MPI)

`tests/opensees/subprocess/test_partitioned_spring_bed_compose_twin.py`
(skips without `APEGMSH_OPENSEES_BIN` and Intel MPI):

| model | ranks | max eigen rel. error | max disp error / peak |
|---|---|---|---|
| plate on 35 springs (Gmsh partition) | 1 | 6.1e-15 | 1.3e-15 |
| | 2 | 6.4e-15 | 1.9e-15 |
| | 4 | 8.2e-15 | 1.1e-15 |
| hosted shell box + soil box, embedded, H5DRM (N, mm) | 1 | 4.1e-12 | 3.6e-12 |
| | 2 | 1.3e-11 | 2.4e-12 |
| | 4 | 7.2e-12 | 1.2e-11 |

Rank 1 is the serial deck under OpenSeesMP (ParallelPlain + Mumps): the
hosted model's 1e-11 level is the Mumps-vs-UmfPack round-off of a stiff
embedded model in N/mm units, not the partition (it is already there on one
rank). The hosted model is in mm because the fork's H5DRM matches a node
without a station to the nearest one within 10 model units (its ADR-88
fallback), which in metres pulls the whole structure into the DRM set.

With ADR 0119's `g.decouple_node_set` + `ops.spring_bed` merged locally (not
part of this change), a 83-spring bed matched serial to 3.7e-14 (eigen) and
1.1e-14 (displacement) on 2 and 4 ranks. The port's 4D (86 182 nodes,
131 571 FEM elements) composed with D3 and cut with D2 emits on 16 ranks in
6 s, every rank building its domain under `mpiexec -n 16`.

## Consequences

- A Tier-2 deck runs under MPI with no bridge call beyond ADR 0119's; the
  ADR 0119 consequence "partitioned emit is refused" no longer holds once
  both land, and `ops.spring_bed`'s docstring sentence saying so goes with
  it.
- A partitioned deck now declares a decoupled node with no element on its
  routed rank instead of only as a ghost of an MP record; the physics is
  unchanged, the deck bytes of such models change.
- The routing reads `node_pairs` and MP records once per `build()`; its cost
  is linear in the element-less nodes and the records.
- RCB balances the declared weights, not the solve. Shells with layered
  sections cost far more than elastic bricks; the caller sets `weights`.

## Deferred

- A node-pair element between mesh nodes on two ranks (a ghost endpoint).
- Stage-claimed node-pair elements on a partitioned emit.
- A graph partitioner (METIS) for `repartition`, behind an optional extra.
- `-nodeOnly` regions: the bridge writes named regions as `-node`, per rank;
  a deck that needs `-nodeOnly` (EnergyBalance regions) still writes them.
- A hosted assembly archive (`/assembly` with a host row) and rehydration of
  a host's `/opensees` content.
