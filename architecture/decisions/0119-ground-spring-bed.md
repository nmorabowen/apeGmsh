# ADR 0119 — Ground-spring bed: decoupled node sets on the session, a spring bed on the bridge

**Status:** Proposed (2026-10-10).

**Owner:** nmora

**Builds on** [ADR 0049](0049-decoupled-nodes.md) (a decoupled node is a
broker node: identity on the session, `ndf` on the bridge),
[ADR 0048](0048-infer-per-node-ndf-from-elements.md) (adaptive zeroLength
endpoints), [ADR 0022](0022-mp-constraint-emission-fanout.md) (MP records
emit themselves) and the `uniaxialMaterial Parallel -factors` primitive.

## Context

A distributed soil-spring foundation (NIST GCR 12-917-21, Chapter 2) puts
one spring and one dashpot on every node of a mat and of the basement walls:
2 487 springs in the San Ramón Tier-2 decks (1 551 under the mat, 936 on the
walls). The decks wire each spring as

```
uniaxialMaterial Elastic 9001 1.0
uniaxialMaterial Viscous 9002 1.0 1.0
uniaxialMaterial Parallel <m> 9001 9002 -factors <k> <c>        (one per direction)
element zeroLength <e> <ground> <side> -mat <mx> <my> <mz> -dir 1 2 3 -doRayleigh 0 -orient ...
equalDOF <structural> <side> 1 2 3
fix <ground> 1 1 1
```

with a ground node offset from the structure and fixed, and a 3-dof *side*
node at the structural node, tied to it by `equalDOF` because the structural
node carries 6 dofs and a zeroLength needs equal `ndf` at both ends. `k` and
`c` differ per node: tributary area times an intensity, with end-zone
factors on the mat and per-face intensities in the face's local axes on the
walls.

Today apeGmsh needs one `g.decouple_node(coords=...)` per ground and per side
node, declared before meshing (so the user must already know every
structural node's coordinates), one `ops.element.ZeroLength(nodes=...)` and
three materials per spring, one `ops.fix` and one `ops.ndf` per node. The
side-node `equalDOF` cannot be declared at all: `g.constraints.equal_dof`
refuses a decoupled role.

## Decision

Two halves, split along ADR 0049's line: where a node comes from is the
session's question; what is attached to it is the bridge's.

### D1 — `g.decouple_node_set(source, *, offset=(0, 0, 0), label=None, tie_dofs=None)`

A *set* of decoupled nodes, one per node of a mesh node set, declared by
name and resolved at extraction:

- `source` is a label or physical-group name (label first, then physical
  group, the session's resolution order). At `get_fem_data` it resolves to
  the mesh nodes of that group, in ascending tag order; an empty or unknown
  source raises.
- `offset` is a 3-tuple, or a callable `f(xyz) -> offsets` taking the
  `(n, 3)` source coordinates and returning `(n, 3)` offsets (outward from
  each wall face, for example). Each new node sits at `xyz + offset`.
- The nodes are appended to `fem.nodes` with `provenance == "decoupled"`,
  tags continuing after the single decoupled nodes (`getMaxNodeTag() + i`,
  rank-invariant), exactly like `g.decouple_node`. They fold into the
  snapshot hash and the `model.h5` round trip with no schema change.
- `tie_dofs`, when given, adds one `equal_dof` record per pair to
  `fem.nodes.constraints` (retained = the source node, constrained = the
  new node, on those dofs). This is the side node: it is a broker MP record,
  so the handler auto-selection, staged and partitioned constraint emit and
  the assess checks see it like any other `equalDOF`.
- The handle (`DecoupledNodeSetDef`) carries, after extraction,
  `source_ids` and `tags`, paired by index. The pairing lives on the handle
  only, as a single decoupled node's tag does; a `from_h5` session has the
  nodes and the `equalDOF` records but no handle.

### D2 — `ops.spring_bed(ground, *, at=None, k, c=None, orient=None, tributary=None, dirs=(1, 2, 3), do_rayleigh=False, fix=True, ndf=3, name=None)`

A builder on the bridge. It reads the frozen snapshot once, at the call,
and registers ordinary primitives, so the emit path, tag law, `ndf` gates,
recorders and the h5 opensees zone see nothing new:

- one spring per node of `ground`: `zeroLength` from the ground node to
  `at`'s node with the same source (the side node) or, with `at=None`, to the
  source node itself;
- one `uniaxialMaterial Elastic 1.0` and one `Viscous 1.0 1.0` shared by
  every bed of the bridge, and per spring and direction one
  `Parallel(units, factors=(k, c))` (`factors=(k,)` without dashpots), so
  every per-spring number is a factor and the deck carries no per-spring
  base material;
- `k` and `c` are arrays of shape `(n, len(dirs))`, a `len(dirs)` sequence
  broadcast to every spring, or a callable `f(xyz, area)` returning the
  array, where `xyz` are the source coordinates and `area` the tributary
  areas (`None` without `tributary=`). End zones, per-face local intensities
  and corner sums are the callable's arithmetic, not bed options;
- `tributary` names a 2-D label or physical group; a node's area is the sum
  over its tri3 / quad4 (corner nodes of higher-order) elements of the
  element area over its corner count;
- `orient` is a 6-tuple, an `(n, 6)` array or a callable `f(xyz)`;
  `None` leaves OpenSees' global axes;
- the ground nodes are fixed in all `ndf` dofs (`ops.fix`), and `ndf` is
  stated for the ground and `at` nodes (`ops.ndf`, ADR 0049); `ndf` must
  equal the `ndf` of the node the spring reaches;
- `do_rayleigh=False` leaves the springs out of Rayleigh damping (the
  zeroLength default, `-doRayleigh` not written): the dashpots carry the
  foundation damping and the global `rayleigh` pair does not reach the
  springs;
- negative or non-finite `k` / `c` raise; the returned `SpringBed` holds the
  per-spring node tags, `k`, `c`, areas and orientations for checks.

## Rationale

- **Node existence stays on the broker.** ADR 0049 rejected bridge-born
  nodes because the lineage hash and the viewer would not see them; D1
  creates the nodes where `g.decouple_node` does, only by name instead of by
  coordinates, because a mat's nodes are known only after meshing.
- **The side `equalDOF` is a broker record.** Emitting it from the bridge
  would bypass the constraint-handler auto-selection and the staged and
  partitioned constraint paths; a record on `fem.nodes.constraints` takes
  all of them.
- **Factors over per-spring materials.** One `Elastic 1.0` and one
  `Viscous 1.0 1.0` with `Parallel -factors k c` is the deck's own form,
  byte for byte, and keeps the per-spring numbers in one place.
- **A builder, not a new emit pass.** Registering existing primitives costs
  some Python per spring but adds no tag-plan family, no h5 zone and no emit
  pass to keep in step with the partitioned and staged paths.

## Consequences

- A Tier-2 foundation is two session calls and one bridge call per spring
  group; the `ζ = c / (2 sqrt(k m))` single-node check of the T2S patch is a
  test.
- **Partitioned emit is refused**, as for every node-pair zeroLength
  (`apeSees` raises before the rank fan-out, ADR 0049): per-rank routing of
  node-pair elements is still deferred. A partitioned Tier-2 deck needs that
  routing first.
- The emitted `zeroLength` line writes `-orient` and omits `-doRayleigh`
  when it is off (OpenSees default 0, `ZeroLength.cpp` `OPS_ZeroLength`); a
  checker that parses the as-run deck form must accept the absent flag.
- `DecoupledNodeSetDef` labels are informational: constraint verbs do not
  resolve them as roles (single decoupled nodes keep that path).
- Staged models declare the bed before any stage; claiming it in a stage is
  not designed here.

## Open questions

1. A `fem.nodes` named set for the decoupled set (so a `from_h5` session can
   address it by name) needs a neutral-schema field; deferred until a reader
   needs it.
2. Per-rank routing of node-pair elements (ADR 0049) would let the bed emit
   partitioned.
