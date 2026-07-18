# Broker internals

This page explains how a `FEMData` snapshot is actually built and stored —
the resolution pipeline that turns declarations into records, the design of
the record sets those records live in, the zoned HDF5 layout, and the
versioning policy that keeps old files readable.
[The concepts page](../concepts/fem-broker.md) covers what the broker *is*;
this one is for contributors who need to know how it works inside.

## Two layers, one direction

The broker code is split across two packages with a strict import law
between them. `src/apeGmsh/mesh/` holds the orchestration: the `FEMData`
composites, the factory that populates them, the content hash, and the HDF5
readers and writers. `src/apeGmsh/_kernel/` holds the pure layer underneath:
the pre-mesh definition dataclasses (`defs/`), the resolved record
dataclasses (`records/`), the resolvers that turn one into the other
(`resolvers/`), the record-set sub-composites (`record_sets.py`), and the
`SelectionChain` mixin that gives geometry, mesh, broker, and results
selection their shared verb surface (`chain.py`).

The law is that `core`, `mesh`, `viz`, and `results` import strictly
downward into `_kernel`, and `_kernel` imports nothing back. That is what
makes the resolved layer testable without a Gmsh session and reusable by
every phase of the session lifecycle — the same `ConstraintResolver` runs
during the mesh-time freeze and again, gmsh-free, when a constraint is added
to an already-frozen model.

## The resolution pipeline

`FEMData.from_gmsh()` delegates to a factory in `mesh/_fem_factory.py` that
runs one fixed sequence. Understanding it explains most of the broker's
invariants.

**Extract.** One walk over the live mesh pulls raw node and element arrays,
builds one `ElementGroup` per Gmsh element type, and snapshots every
physical group, label, and partition into name-first sets. Nothing after
this step reads mesh topology from Gmsh again.

**Inject synthetic nodes.** Decoupled nodes declared via
`g.decouple_node(...)` are not Gmsh vertices; they are appended to the node
pool *before* constraint resolution, so the phantom-node tag base
(`max(node_ids) + 1`) lands strictly above them and the two synthetic
ranges are disjoint by construction (ADR 0049).

**Resolve, fail-loud.** Each declaration family on the session resolves in
turn: MP-style constraints, embedded reinforcement, node embedment, contact
and contact-plane interactions, auto-emitted rebar elements, loads,
boundary conditions, prescribed displacements, masses. Every resolver
receives the same kwargs — the node pool, coordinates, element tags, and
the part node/face maps — and none of the calls is wrapped in a broad
`except`. A reference that binds zero nodes, the wrong dimension, or an
ambiguous target raises with an actionable message, because a structural
model that silently drops a tie is worse than one that errors.

**Split.** Resolution produces one flat list of records; the factory then
sorts them into node-side sets (pair and group constraints, nodal loads,
SP records, masses) and element-side sets (interpolations, surface
couplings, element loads, plus the additive tie and contact record lists).
The node/element split — not record kind — is the broker's organizing axis.

**Filter and finish.** Optional orphan removal drops unused nodes, but
first collects every node referenced by a resolved record into a protected
set so a constraint can never lose an endpoint. The factory then computes
the semi-bandwidth, snapshots the part-label maps so `target="col_01"`
works with the session closed, populates the per-node `ndf` vector from
explicit declarations (apeGmsh refuses to infer it from element class), and
assembles the composites.

The one sanctioned exception to "never call Gmsh again" is the raw
`(dim, tag)` target: a name-free tuple has no snapshot representation, so
it queries the live session — and raises a `RuntimeError` pointing at the
`label=`/`pg=` alternative if the session is gone. Adding a second gmsh
call site to the broker requires a design-review note.

`FEMData.from_msh()` runs the same extract step against an imported file
and skips resolution entirely, which is why imported snapshots carry
groups but no records.

## Records and record sets

Every object in the resolved layer is one of three flavours, and each new
class must declare its own: a **record** is a slotted, immutable dataclass
(`NodePairRecord`, `SPRecord`, `MassRecord`, ...); a **record set** is the
iterable container that owns one kind of record and its query methods; a
**composite** is a mutable orchestrator like `NodeComposite`. `__slots__`
on records is not cosmetic — at mesh scale a million elements times a
handful of records each is millions of Python objects, and slots halve
their memory while preventing attribute drift on something that must stay
immutable.

All record sets derive from `_RecordSetBase`, which supplies iteration,
`len()`, truthiness, and `by_kind()`. Constraint kinds are constants
(`fem.nodes.constraints.Kind.EQUAL_DOF`), so a typo is a linter error, not
a silently empty branch.

The load-bearing convention is the **atomic versus compound split**. Some
records are compound — a rigid diaphragm is one record carrying many
slaves; a node-to-surface constraint carries phantom-node coordinates; a
mortar coupling carries its sparse operator. Any set holding compounds
must therefore offer two iterator tiers: atomic iterators (`pairs()`,
`equal_dofs()`, `interpolations()`) that expand compounds into flat
solver-ready commands, and compound iterators (`rigid_diaphragms()`,
`couplings()`, `phantom_nodes()`) that preserve the side-band fields. The
two are never mixed in one iterator, because a caller cannot tell compound
from atomic at the call site — and the resulting bug is silently lost
phantom nodes.

One more cross-cutting rule: every ID array is `dtype=object`, coerced once
at construction, so iteration yields plain Python `int`. C-extension
solvers that accept `int` but not `numpy.int64` — OpenSees is the
motivating case — consume broker output without a single cast.

## Mutation after the freeze

`FEMData` has no `refresh()` and never will: an in-place update would leak
live-session semantics back into the solver contract. But immutability
does not mean the model is finished. Once a session is in chain phase (its
broker snapshot exists), new declarations route through a chain-phase
router that resolves the definition *against the snapshot* — using the
same `_kernel` resolvers, with a `FEMDataSource` adapter standing in for
the live mesh — and applies the result through `with_constraint()` /
`with_load()` / `with_mass()` transforms. Each transform returns a **new**
`FEMData` with the record appended; the session swaps its reference. The
snapshot you were handed never changes; the session's current snapshot
advances. This is also how composition works on gmsh-free reopened models.

## What the snapshot hash covers

`snapshot_id` is a 128-bit blake2b digest over a canonical, sort-stable
walk of: node IDs and coordinates (plus the per-node `ndf` and provenance
channels when non-empty), per-type element connectivity, physical-group
membership on both the node and element side, label membership on both
sides, and the composition provenance records.

Deliberately *not* hashed: constraints, loads, masses, SP records, and
every tie and contact record list. The hash identifies the mesh and its
naming — the thing results bind to — not the physics riding on it, so
adding a load case does not orphan every results file recorded against the
same mesh. Empty channels are skipped entirely, keeping the digest of a
plain model identical across broker generations that added optional
features. The flip side of that stability is that the fold order is
frozen: reordering the walk would rotate every existing `snapshot_id`.

## The zoned HDF5 layout

A `model.h5` is partitioned into zones, each owned by exactly one writer.
The broker writes the **neutral zone** (`mesh/_femdata_h5_io.py`) — the
solver-agnostic groups at the file root:

```
/meta                     schema stamps, snapshot_id, lineage
/nodes                    ids, coords, optional ndf / provenance
/elements/{type}          per-type ids + connectivity
/physical_groups  /labels /mesh_selections
/constraints/{kind}       MP-style records
/loads/{kind}/{pattern}   per-pattern load records
/masses  /partitions  /parts
/reinforce_ties  /embed_ties  /contacts  /contact_planes  /rebar_elements
/composed_from            composition provenance
```

The OpenSees bridge writes its zone under `/opensees/`, and the results
runtime writes under `/stages/`; the broker never reads or writes either.
The same neutral layout also embeds as a sub-group (`/model/` inside a
composed `results.h5`) — there is one neutral-zone writer, and the only
difference between a root file and an embed is the parent group passed in.

Records are stored as structured NumPy arrays whose dtypes live in
`mesh/_record_h5.py`, one payload dtype per record family. Optional
per-record groups (ties, contacts) are omitted entirely when empty, so a
model without the feature stays byte-identical to one written before the
feature existed. On read, the neutral-zone reader rebuilds the composites
and recomputes `snapshot_id`, refusing to return a snapshot whose hash
does not match the one stored in `/meta` — corruption fails loud, never
silently.

## Versioning policy

Each zone carries its own semver stamp in `/meta`
(`neutral_schema_version`, `opensees_schema_version`,
`results_schema_version`), and the zones version independently — the
policy is ADR 0023. The bump cadence is strict: patch means fix-only,
minor means additive (new group, new column; old required fields remain),
major means breaking. Nearly every change lands as an additive minor, and
the writer module keeps a ledger docstring entry for each one.

Readers enforce a **two-version window**: code at X.Y accepts files at X.Y
and X.(Y−1), refuses anything older, and — deliberately — refuses anything
*newer*, because silently tolerating a file the reader does not fully
understand is worse than an explicit "upgrade apeGmsh" error. Within the
window, additive columns are presence-probed (`"omega" in p.dtype.names`)
and decode to their dataclass defaults when absent, which is what lets a
one-minor-old file round-trip without a migration step. Legacy files that
predate per-zone stamps fall back to the single envelope
`schema_version` key; new code never branches on the envelope.

The discipline this buys is concrete: a feature can add a column or a
group without touching any existing bytes, an old file loads with the
feature simply off, and the reader/writer constants live in one module per
zone so they cannot drift apart.

---

*Next: [Parts & assembly internals](parts-assembly.md).*
