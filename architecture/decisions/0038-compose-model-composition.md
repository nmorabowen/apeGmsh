# ADR 0038 — `g.compose()` model composition: flat-after-merge with namespaced tag-offset import

**Status:** Accepted (2026-05-26, Phase 3 of the model-composition
work stream). Builds on the three-broker refactor
([ADR 0019](0019-opensees-model-read-side-broker.md) through
[ADR 0023](0023-per-zone-schema-versioning.md)), references
[ADR 0026](0026-h5modelreader-protocol-contract.md) (H5ModelReader
Protocol), [ADR 0027](0027-cross-partition-mp-constraints.md)
(cross-partition MP-constraint emission), and
[ADR 0036](0036-embedded-host-decomposition.md) (embedded-host
decomposition). The tag-collision verifier discipline is folded in
as §"Tag-collision verifier". Deferred follow-ons: ADR 0039
(`interface.export(...)` encapsulation primitive), ADR 0040
(`g.loads.imposed_strain(...)` eigenstrain primitive for hybrid
composite strain transfer), and a future ADR for the two-level
partitioner (intra-module METIS while honoring module boundaries
as hard cuts).

## Context

Today's `Part` abstraction is geometry-only
([core/Part.py:130-137](../../../core/Part.py)) — the composites
exposed on a `Part` are `model`, `labels`, `physical`, `inspect`,
`plot`, and `edit`. There is no `mesh`, no `constraints`, no
`materials`, no `loads`. Composing complex apeGmsh models —
high-fidelity connection details bolted into a frame, hybrid
soil/structure SSI assemblies — is therefore forced through a single
gmsh session at one element order globally, with all OCC kernel
state in one place and all PG / label / material namespaces flat.
The result is that two engineers cannot author parts of a model in
isolation and combine them later without merging gmsh scripts by
hand; iterating on a connection detail forces a re-mesh of the
hosting frame; and a reusable component (say, a standard base-plate
or a brace-gusset) must be re-authored into every assembly that
needs it.

This ADR introduces `g.compose(source_h5, label, ...)` — merge a
previously-saved apeGmsh model (the **module**) into the current
session (the **host**). The merged content — mesh, physical groups,
labels, constraints, loads, masses, materials, sections, integration
rules, parts sub-records — is namespaced under the compose `label`
and tag-offset into a reserved range. Interfaces between host and
composed module are necessarily non-conformal (the module was meshed
independently of the host) and are bridged by the existing
constraint primitives — `g.constraints.embedded(...)` per ADR 0036,
`g.constraints.tied_contact(...)`, `g.constraints.equalDOF(...)`,
`g.constraints.rigid_link(...)`. The compose step does not introduce
new constraint primitives; it ensures that the imported module's
nodes / elements / PGs are referenceable by host-side constraint
declarations under the namespaced labels.

After `g.compose(...)` returns, the host `FEMData` is **flat**:
downstream subsystems (constraint resolver, OpenSees emitter,
ADR 0027 cross-partition fan-out, MPCO recorders, viewers, results
parsers) see one unified canonical broker — no compose-awareness
required downstream. The flat-after-merge decision is the load-
bearing one; the rest of the ADR is the contract that makes the
flattening safe.

The namespacing convention is not new. The instance-of-a-Part
mechanism at
[core/\_parts\_registry.py:705-736](../../../core/_parts_registry.py)
already prefixes the instance's PG names with
`{instance_label}.{pg_name}` so two copies of the same Part don't
collide. Compose extends the same `{label}.{original_name}`
convention to a full saved model rather than introducing a new
pattern.

## Decision

### `g.compose()` signature

```python
def g.compose(
    self,
    source: str | Path,
    *,
    label: str,
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotate: tuple[float, float, float, float] | None = None,
    anchor: str | None = None,
    partition_rank: int | None = None,
    properties: dict[str, Any] | None = None,
) -> ComposedModule:
```

`source` is H5-path-only in v1 — the module is the `model.h5` file
produced by a previous `g.save()` call. STEP / Tcl / openseespy
sources are not in scope; the round-tripped H5 is the canonical
module format and gates determinism (the lineage chain is computed
against the merged neutral-zone bytes, see §"Lineage chain
extension" below).

`label` is required. Validation: non-empty string, no `.` (the
namespace separator), no whitespace, no collision with an already-
composed module's label, no collision with any top-level host label.
Invalid labels raise `ComposeLabelError` at call time.

`translate` and `rotate` are the rigid-body placement of the module
in the host's coordinate system. `rotate` is axis-angle
`(x, y, z, theta)` — same convention as the gmsh OCC `rotate(...)`
operations. `anchor` is sugar over `translate`: it resolves a host-
side physical-group name to its centroid and sets `translate` to
position the module's origin at that centroid. `anchor` and an
explicit non-zero `translate` are mutually exclusive
(`ComposeAnchorError` on conflict).

`partition_rank` is the 0-based MP rank hint — Layer 2 of the rank
model (see §"Rank model" below). `None` means "let the eager
populator assign a rank automatically".

`properties` is a free-form dict persisted to
`/composed_from/{label}/properties` on the host's next `g.save()`.
Informational only; not load-bearing.

The return value is a `ComposedModule` handle exposing
`label`, `source_path`, `translate`, `rotate`, `partition_rank`,
and methods to introspect the composed PGs / labels / record counts.
The handle is also retrievable via `g.compose_list()` (see below).

### Companion helpers (v1)

- **`g.compose_inspect(path: str | Path) -> dict`** — read the
  module's H5 header without composing it. Returns a summary with
  the module's `fem_hash`, declared neutral-zone schema version,
  PG / label inventory, record counts per kind, and any
  `properties` the author attached. Cheap H5 metadata read; does
  not parse the bulk record payload.

- **`g.compose_list() -> tuple[ComposedModule, ...]`** — the modules
  currently composed into this session. Returned in compose-call
  order. Empty tuple when nothing has been composed.

`g.uncompose(label)` (selective removal) and `g.recompose(label,
new_path)` (swap a module for a re-baked version) are deferred to a
future ADR — they would need to walk the broker and unwind only the
records that originated under the given `label`, which interacts with
the lineage chain in non-trivial ways. v1 ships compose-only;
"recompose" is `g = Model(...)` from scratch followed by fresh
`g.compose(...)` calls.

### Merge semantics — zone × record-kind verdicts

Each record kind in the module's neutral zone receives one of three
verdicts at compose time: **IMPORT** (with tag-offset and namespace
prefix as appropriate), **DISCARD** (re-derived at emit time by the
host pipeline), or **FILTER** (host owns this concern; the module's
version is dropped). The full table:

| Record kind | Verdict | Notes |
|---|---|---|
| nodes | IMPORT | tags rewritten through offset map |
| elements (all types) | IMPORT | tags + connectivity rewritten |
| physical groups | IMPORT | name prefixed `{label}.`; `node_ids` / `element_ids` rewritten |
| labels | IMPORT | name prefixed `{label}.`; targets rewritten |
| mesh selections | IMPORT | name prefixed; targets rewritten |
| parts sub-records | IMPORT | namespace prefixed; nested tag refs rewritten |
| constraints (all kinds) | IMPORT | every tag-ref field rewritten (`master_node`, `slave_node(s)`, `host_element_tag`, `cnode`, `rnodes`, `phantom_nodes`, `sr_*_nodes`); outer `target` re-stringified with the namespaced label name |
| loads (nodal + element + SP) | IMPORT | `node_id` / `element_id` rewritten; outer `target` re-stringified |
| masses | IMPORT | same as loads |
| materials | IMPORT | name prefixed; tags rewritten through offset map |
| sections | IMPORT | name prefixed; tags rewritten through offset map |
| integration rules | IMPORT | name prefixed; tags rewritten through offset map |
| per-element-type assignments | IMPORT | references the namespaced material / section / integration names |
| regions | DISCARD | re-derived at emit time; matches `MODEL_HASH_EXCLUDED_CHILDREN` |
| cuts | DISCARD | host re-derives |
| sweeps | DISCARD | host re-derives |
| module's own `PartitionSet` | DISCARD | host re-partitions per the 3-layer rank model |
| stages (PR #335) | FILTER + `UserWarning` | host owns analysis-time topology activation |
| time-series | FILTER + `UserWarning` | analysis-time, host owns |
| load patterns | FILTER + `UserWarning` | analysis-time, host owns |
| recorders (MPCO + native) | FILTER silently | host owns the output layer |
| analysis settings (numberer, system, algorithm, test, integrator, constraints handler) | FILTER silently | host owns the solve configuration |
| `/results/` zone | FILTER silently | results belong to a solve, not a module |

The DISCARD record kinds — `regions`, `cuts`, `sweeps` — match the
existing `MODEL_HASH_EXCLUDED_CHILDREN` frozenset at
[opensees/\_internal/lineage.py:85](../../_internal/lineage.py:85).
The module's own `PartitionSet` is also discarded but for a different
reason — the host re-partitions per the 3-layer rank model below —
and is not part of the `MODEL_HASH_EXCLUDED_CHILDREN` list.

The FILTER-with-warning kinds (`stages`, `time-series`,
`load patterns`) emit a one-line `UserWarning` per kind per compose
call:

```
ComposeFilterWarning: module 'connection_a' carries 3 stages; stages are
analysis-time and not inherited under compose. Re-declare on the host.
```

The FILTER-silently kinds (`recorders`, `analysis settings`, `/results/`)
do not warn — re-declaring a numberer or a system from a module
into the host is never what the caller wants, and warning every
compose call would be noise. `g.compose_inspect(path)` exposes what
was filtered for callers who want to audit.

### Namespace rule

All string-keyed records imported from the module are prefixed with
`{label}.`. A module called `g.compose(..., label="conn_a")` whose
PG inventory is `{"top_flange", "weld_zone"}` lands in the host
broker as `{"conn_a.top_flange", "conn_a.weld_zone"}`. Label
references in constraint / load / mass records that point at module-
local PGs are rewritten to the namespaced form
(`"conn_a.top_flange"`); references that point at host-side PGs are
forbidden (a module cannot reach into the host or into another
module — see INV-2 below).

**Nested composition (v1, depth-limited).** A source H5 whose
`/fem/composed_from/` group is non-empty is itself a composed
assembly. Compose handles this transparently via three rules:

1. **Depth limit.** Default `max_compose_depth=3` (configurable
   kwarg). `source_depth = 1 + max(child.depth)`; raise
   `ComposeDepthExceededError` if `source_depth >= max_compose_depth`.
   The default of 3 covers the canonical hierarchy (connection →
   frame → building) plus one level of headroom. The cap can be
   lifted in v1.1 once depth-2 cost is measured.
2. **Separator alternation.** When composing a nested source,
   inner-level names use `/` as the namespace separator at depth
   boundaries instead of `.`. A source whose own labels are
   `frame.beam_A.end` (composed from a sub-module) gets rewritten to
   `frame/beam_A.end` before the outer prefix is applied — yielding
   `bldg_1.frame/beam_A.end` after compose. The `.` ↔ `/` alternation
   at depth boundaries makes nesting depth structurally visible and
   unambiguous on parse.
3. **Provenance graft.** Source's `/fem/composed_from/` records are
   grafted into the host's `/fem/composed_from/` group with their
   labels joined per rule 2. See the **"Flat graft instead of tree
   graft" amendment (2026-05-27)** below for the as-shipped
   representation choice and the hash one-way door it sidesteps. The
   provenance group remains informational (per the lineage section);
   `host_fem_hash` is the canonical-bytes hash of the flat post-compose
   broker either way.

`ComposeNestedError` is removed. The new errors are
`ComposeDepthExceededError` (depth cap hit) and
`ComposeNamespaceCollisionError` (a literal label collision
post-rewrite, vanishingly rare with separator alternation but the
verifier still checks).

Notably: the existing `g.parts.add()` namespacing convention at
[\_parts\_registry.py:705-736](../../../core/_parts_registry.py)
already produces dotted names like `C1.face_top`. The v1 prohibition
was originally written to forbid such sources from being composed;
with separator alternation, **a model that used `g.parts.add()`
composes cleanly** — its dotted PG names are valid leaf-level labels,
not nested-compose markers. The detection now uses
`H5Lexists('/fem/composed_from')` on the source, NOT a name scan.

### Tag-offset scheme — per-module auto-sizing

Each compose call reserves a contiguous integer window in the host's
tag namespace for the module's records. The window size is computed
from the source's actual tag span:

```
source_span = source_max_tag - source_min_tag + 1
size = ceil(source_span / GRANULARITY) * GRANULARITY                  # default GRANULARITY = 1_000_000
base = ceil(host_max_tag / GRANULARITY) * GRANULARITY                  # first compose
     = previous_base + previous_size                                    # subsequent composes (cumulative)
```

`GRANULARITY` is a class attribute on the `Compose` facade (default
`1_000_000`) — a power-of-10 round-up unit for human-readable log
messages. The `compose_size_per_module=N` kwarg overrides the
computed size with an explicit floor (advisory headroom for users
who expect the source to grow):

```python
g.compose("building.h5", label="bldg_1", compose_size_per_module=50_000_000)  # reserve 50M
```

The size is read from the source at compose time via
`/fem/@tag_span_max` (a new attribute written at `g.save()` time per
the schema bump). H5s saved by pre-2.9.0 apegmsh lack this attribute;
compose falls back to a dataset scan with one-shot
`UserWarning("source written by pre-2.9.0 apeGmsh; tag span computed
by dataset scan — re-save under 2.9.0 to skip this on future composes")`.
The scan is O(N) over the node + element tag columns; ~50ms per 1M
tags on local disk — acceptable for a one-shot upgrade nudge that
doesn't repeat on subsequent composes of the same upgraded H5.

The rewrite rule for every tag in the source is:

```
new_tag = old_tag + (base - source_min_tag)
```

Source's internal sub-module offsets (at depth ≥ 2) are preserved as
relative spacings within the outer reservation. A nested source whose
internal tags span [1M, 11M) gets composed into the host at, say,
[25M, 35M) — the 11M internal span is preserved; the offsets just
shift uniformly.

The same window serves nodes AND elements — they live in disjoint
integer spaces in apeGmsh's broker (the `NodeRecord` table and the
per-type `ElementRecord` tables don't share tag-pool semantics with
each other), so a single offset suffices for both.

All tag-bearing records — nodes, elements, physical-group entity
tags, material tags, section tags, integration-rule tags,
`geomTransf` tags — get the uniform offset rewrite via
`new_tag = old_tag + (base - source_min_tag)`. The TagAllocator at
apeSees(fem) build time sees imported tags with their offset values
and does not re-allocate over them. Material and section *names*
remain the user-facing identity (namespaced as `{label}.{name}`);
tags are the bridge-internal handle.

### Tag-reference rewrite checklist

Every field that holds an integer tag must be rewritten through the
offset map before the record lands in the host broker. The complete
list (the implementation calls this the "tag-rewrite cover set"):

- `NodeRecord.tag`
- `ElementRecord.tag` and `ElementRecord.connectivity`
- `PhysicalGroupRecord.node_ids` and `.element_ids`
- `LabelRecord.target` (when target is a tag, not a string name)
- `MeshSelectionRecord.node_ids` and `.element_ids`
- parts sub-records: any nested `node_ids` / `element_ids` /
  `tag` / `connectivity` field
- constraint records: `master_node`, `slave_node`, `slave_nodes`,
  `host_element_tag`, `cnode`, `rnodes`, `phantom_nodes`,
  `sr_master_node`, `sr_slave_nodes` (the surface-coupling
  variants from ADR 0036), outer `target` field re-stringified to
  the namespaced label name
- load records: `node_id`, `element_id`, outer `target`
- mass records: `node_id`, outer `target`
- material / section / integration-rule records: `tag` field
- `geomTransf` records: `tag` field
- Element-to-material references (`mat_tag` on element records)
  are **tag-reference fields in `tag_rewrite_spec` and are
  rewritten via the offset map**, just like `connectivity` is.
  They are not a special-case "automatic" inheritance — they're
  an explicit rewrite target. The reference remains valid
  post-rewrite because the referenced material tag is also offset
  (per the uniform offset rule). The same applies to
  `section_tag`, `integration_tag`, and any other tag-reference
  field on element or constraint records.
- OpenSees `element_meta`: `fem_eids` (the bridge's record of
  which FEM element tags map to which OpenSees element tags)

A field omitted from the rewrite cover set is a fail-loud bug, not
a silent corruption — the tag-collision verifier (next section)
catches dangling references at compose time. Adding a new record
kind to the broker requires updating the cover set in lockstep; the
verifier protects against the forgetful path.

### Tag-collision verifier

The verifier runs at compose time, after the tag-rewrite pass and
before the merged records are committed to the host broker. It
asserts five properties. Checks 1-4 raise `PartTagCollisionError`
(the error names the offending module label, the kind of collision,
and the colliding tag or name); check 5 raises
`ComposeCapacityError`. Both fail-loud at compose time:

1. **No imported tag lands in the host's range.** Post-rewrite,
   every imported tag is `>= base[i]` for that module's reservation.
   The host's `TagAllocator` watermark is the lower bound; the
   verifier checks against it directly.
2. **No two modules' reservation ranges overlap.** Range-disjoint
   check over `fem.composed_from`. The auto-sizing formula
   guarantees this by construction, but the verifier double-checks
   the actual reservation extents in case the offset formula is
   ever changed.
3. **Every constraint reference resolves inside the owning module's
   reservation window.** "Module" here means the outer reservation
   window — the cumulative `[base, base+size)` range allocated to
   this compose call. A nested source's internal sub-module references
   are valid because they all sit inside the outer window after the
   uniform tag shift. Cross-module references between sibling composes
   in the host are still forbidden — a constraint imported under
   module `conn_a` referencing node tag `42_000_001` must have
   `42_000_001` fall inside `conn_a`'s `[base, base + size)`. A
   reference outside the reservation means either the rewrite missed
   a field (cover-set bug) or the module's source H5 carried a
   cross-module reference between sibling composes.
4. **No PG-name collision after namespacing.** The namespace prefix
   prevents this by construction in the typical case. **Separator
   alternation (the `.` ↔ `/` rule, INV-3) eliminates the
   cross-module-PG-collision class structurally** — a host PG
   `frame.beam_A` and a composed module's rewritten `frame/beam_A`
   are syntactically distinct. Check 4 catches only the rare case
   where a host author literally named a PG with a `{compose_label}.`
   prefix matching an actual compose label, which is a pre-existing
   host-side authoring choice the verifier surfaces with
   `ComposeNamespaceCollisionError`.
5. **Source tag span fits within the computed reservation size.**
   `source_max_tag - source_min_tag < size[i]`. **This check fires
   only when the caller passed an explicit
   `compose_size_per_module=N` smaller than the source's actual tag
   span — without an override the auto-sizing scheme computes `size`
   from the source's actual span and check 5 is correct by
   construction.** If the override is too small, raise
   `ComposeCapacityError` naming the source's span and the
   configured cap.

The verifier is a Phase 2 primitive that compose calls into; it is
also intended to be reused by future operations (e.g., merging two
sessions in v2, or cross-validating a module before re-baking).
This ADR pins its API contract — the five checks above are the
load-bearing surface.

### Rank model — 3 layers, eager populator

Compose feeds the partitioned-emit pipeline from ADR 0027. The rank
each composed module's elements land in is determined by a three-
layer model:

**Layer 1 (default).** When `len(fem.composed_from) > 0` and
`fem.partitions == ()`, the eager populator auto-assigns one rank
per module. The host's elements land on rank 0; modules are
counter-assigned starting at rank 1, skipping any pinned ranks
(see Layer 2). The default for a single-module compose with no
hints: host on rank 0, module on rank 1. Pinning a module to rank 0
(via `partition_rank=0`) means it shares with the host, which is
legal — multiple element groups can land in the same
`PartitionRecord`.

**Layer 2 (hint).** `partition_rank=K` validated at compose time:
`K >= 0` (integer), no other constraint. Multiple modules can share
a rank — their elements both land in `PartitionRecord(id=K)`.
`partition_rank=10` on a 3-module compose is legal: the populator
keeps empty `PartitionRecord`s between the populated ranks
(`max(highest_pinned + 1, n_modules + 1)` total ranks). Empty
partitions are valid records in `fem.partitions` and emit per
ADR 0027 as no-op `if {[getPID] == K} { }` blocks.

**Layer 3 (bypass).** An explicit
`g.mesh.partitioning.partition(N)` call overwrites `fem.partitions`
last-write-wins, with a `UserWarning` listing the overridden
Layer 2 hints:

```
ComposePartitionOverrideWarning: explicit partition(4) overrode
partition_rank= hints for modules: {conn_a: 1, conn_b: 2}
```

**Populator entry point — eager.** Every `Compose.add(...)` call
repopulates `fem.partitions` from the full current
`fem.composed_from` set, applying Layer 1 + Layer 2 rules.
Subsequent `g.mesh.partitioning.partition(N)` overwrites
`fem.partitions` unconditionally (Layer 3 bypass). The order rule
is **"last write to `fem.partitions` wins"** — but every overwrite
(in either direction: Layer 3 clobbering Layer 1's eager output, or
a subsequent `Compose.add()` re-running Layer 1 over an existing
Layer 3 METIS result) emits a `UserWarning` describing which layer
overwrote which. The user sees the warning stream and can audit
the interleaving sequence.

A worked example showing the state of `fem.partitions` at each step:

```python
g.compose("conn_a.h5", label="A")          # fem.partitions = (host, A)             — Layer 1 default
g.compose("conn_b.h5", label="B")          # fem.partitions = (host, A, B)          — Layer 1 default
g.mesh.partitioning.partition(4)           # fem.partitions = METIS(4) over all     — Layer 3 overwrite
                                           # UserWarning lists overridden hints (none here, but format is fixed)
g.compose("conn_c.h5", label="C")          # fem.partitions = (host, A, B, C)       — eager populator re-runs,
                                           # METIS partitioning lost; UserWarning emitted
```

Predictability matters here: `fem.partitions` is the single source of
truth for the bridge per ADR 0027 §"Tag determinism" and for the
cross-partition viewer's per-rank coloring per the cross-partition-
viewer work stream. A lazy populator that materialized
`fem.partitions` only at build time would make `len(fem.partitions)`
non-deterministic between `g.save()` and `apeSees(fem)` calls
(depends on whether build-time materialization had fired yet); the
eager rule avoids that ambiguity.

### Lineage chain extension

The compose operation extends the lineage chain from the bind-
contract ([opensees/\_internal/lineage.py](../../_internal/lineage.py))
via a single rule:

**Lineage hash composition.** The host's `fem_hash` after compose
is computed via a thin `compose_hash()` wrapper:

```python
def compose_hash(fem: FEMData) -> str:
    # Sort module records by module_label for compose-order independence
    by_module = group_by_module_label(fem.canonical_records)
    parts = [canonical_bytes(by_module[""])]  # host-native records first
    for label in sorted(by_module):
        if label == "":
            continue
        parts.append(b"||" + label.encode() + b"||")
        parts.append(canonical_bytes(by_module[label]))
    return blake2b(b"".join(parts), digest_size=32).hexdigest()
```

The wrapper sorts module records by `module_label` before hashing —
compose order (`A then B` vs `B then A`) produces the same hash
regardless of `canonical_bytes()`'s internal iteration order. This
structurally locks INV-1's compose-order invariance without
depending on future stability of `canonical_bytes()`'s sort
behavior. The function lives at
[opensees/_internal/lineage.py](../../_internal/lineage.py)
alongside the existing `canonical_bytes()` and `fem_hash` plumbing.

The `/composed_from/{label}/` provenance group is **informational,
not load-bearing**. It records `source_fem_hash`,
`source_neutral_schema_version`, `source_path` (string, not
re-fetched), `translate`, `rotate`, `partition_rank`,
`composed_at` (ISO timestamp), and the optional `properties`
dict. `FEMData.from_h5(...)` reads this group for inspection
purposes (`g.compose_list()` populates from it on round-trip) but
the flat broker is the authoritative source — if the recorded
`source_fem_hash` doesn't match the source file's current hash
(because the module was re-baked), the warn-not-raise path from the
bind contract handles it: the host's `fem_hash` already changed
when the module's bytes changed, which invalidates `model_hash` and
`results_hash` downstream, which surfaces as a `LineageStaleWarning`
at the appropriate broker boundary. No special-case logic in the
lineage code is required for compose.

## Invariants

- **INV-1.** After `g.compose()`, the host `FEMData` is a flat
  canonical broker. Downstream subsystems (constraint resolver,
  OpenSees emitter, ADR 0027 partition fan-out, MPCO recorders,
  viewers, results parsers) see exactly the same record shapes they
  see for a non-composed assembly. The contract test is
  **canonical-bytes equivalence**: a composed assembly and an
  equivalent assembled-directly assembly produce the same
  `compose_hash()` over the post-merge `/fem/` zone — see
  [opensees/_internal/lineage.py](../../_internal/lineage.py)
  `compose_hash()` (sorted-by-module-label wrapper over
  `canonical_bytes()`; ensures compose-order independence by
  construction). Line-order in the emitted Tcl/Py deck may differ
  (dict-iteration-dependent emission order over per-namespace
  material/section names) but the deck is semantically identical —
  same nodes, elements, constraints, materials by their canonical
  identifiers.

- **INV-2.** Tag uniqueness is preserved across compose: per the
  verifier, no two records share an integer tag after compose.
  **All tag-bearing record kinds** (nodes, elements, physical
  groups, materials, sections, integration rules, `geomTransf`)
  get the uniform offset rewrite via
  `new_tag = old_tag + (base - source_min_tag)`. Tag-reference
  fields inside imported records (`mat_tag`, `section_tag`,
  `integration_tag` on element records; `master_node` /
  `slave_node` etc. on constraint records) are **explicit entries
  in `tag_rewrite_spec` and are rewritten via the offset map**,
  just like `connectivity` is — not "automatic" inheritance. The
  references remain valid post-rewrite because every referenced
  tag is also offset (per the uniform offset rule). Cross-module
  references between sibling composes in the host are forbidden
  by verifier check 3.

- **INV-3.** Namespacing is total: every string-keyed record from a
  module is prefixed with `{label}.` at the outer boundary. At depth
  boundaries the inner separator is `/` (e.g.,
  `bldg_1.frame_A/conn_A.face_to_beam`). The `.` ↔ `/` alternation
  at depth boundaries makes depth structurally visible and
  unambiguous. Dotted leaf names from the `g.parts.add()` convention
  at [\_parts\_registry.py:705](../../../core/_parts_registry.py)
  are valid leaf-level labels, not nested-compose markers.

- **INV-4.** Rank assignment is **deterministic given the operation
  sequence** — replay of `compose(A) → compose(B) → compose(C)` with
  the same `partition_rank=` hints and the same composed sources
  produces the same `fem.partitions`. Explicit
  `g.mesh.partitioning.partition(N)` calls anywhere in the sequence
  overwrite `fem.partitions` last-write-wins with `UserWarning`
  listing affected hints. Interleaved operations are auditable via
  the warning stream.

- **INV-5.** Provenance in `/composed_from/` is informational —
  `FEMData.from_h5()` reads it for inspection but the flat assembly
  is authoritative. Source files are NOT re-fetched on load; a
  composed module persisted to the host H5 is fully self-contained
  there.

- **INV-6.** Analysis-time settings are never inherited from a
  composed module. Stages (PR #335), time-series, and load patterns
  emit a `UserWarning` per kind per compose (visible filter, so
  callers notice the drop). Recorders (MPCO + native) and analysis
  settings (numberer, system, algorithm, test, integrator,
  constraints handler) and `/results/` filter silently — re-
  declaring these from a module is never what the caller wants.
  All filtered kinds remain inspectable via `g.compose_inspect(path)`.
  Under the v1 scope-gate fallback (see §v1 scope gate), this
  invariant is conditional: constraints filter alongside analysis-time
  settings when the cost gate trips.

- **INV-7.** The H5 schema bump
  (`neutral_schema_version` 2.8.0 → 2.9.0) is additive-minor per
  [ADR 0023](0023-per-zone-schema-versioning.md). Schema 2.8.x
  readers continue to open 2.9.x files; they ignore the new
  `module_label` parallel datasets on `/fem/nodes/` and
  `/fem/elements/{type}/`, the `/fem/@tag_span_max` attribute, and
  the optional `/fem/composed_from/` group via `H5Lexists` probes.
  The two-version reader window from ADR 0023 keeps interoperability
  across the bump.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| **`Part.bake()` per the original proposal** | `Part` is geometry-only ([core/Part.py:130-137](../../../core/Part.py) — composites are `model`, `labels`, `physical`, `inspect`, `plot`, `edit`). Baking exposes only a subset of apeGmsh and forces callers into a constrained authoring model. The full apeGmsh session is the natural unit of saved-and-loaded; making the H5 the bake target lets the same `g.save()` machinery serve both straight-line persistence and module export. |
| **Save-side `g.save_as_module(...)` variant** | `g.save()` already persists the full broker per ADRs 0019-0023. No new save-side machinery is needed; "module-ness" is a load-time interpretation. Adding a save-side variant would duplicate logic and force authors to know at save time whether the H5 would later be composed (which they generally don't). |
| **Sub-FEMDatas stored separately in the host H5** | Would kink every downstream contract: the viewer would load N `FEMData`s and assemble at view time; results parsing would need module-aware tag lookup; MPCO stitching would need cross-FEMData logic; the lineage chain would need fan-out hashes. The flat-after-compose decision deliberately pays the cost once at compose time (namespace prefix + tag-offset rewrite + verifier) so every downstream consumer sees one canonical broker. |
| **Mandatory `interface_pgs=` declaration at save time** | Self-documenting but redundant — whatever PG is referenced by a `g.constraints.embedded(host="host_pg", embedded="module.pg")` call IS an interface by usage. Removing the declaration eliminates a foot-gun (forget to declare → interface mysteriously not exposed) and simplifies the API. The inspection helper `g.compose_inspect(path)` recovers self-documentation without locking save-time discipline on the module author. |
| **Lazy partition populator** | Would make `len(fem.partitions)` non-deterministic between `g.save()` and `apeSees(fem)` — the value depends on whether build-time materialization had fired. The eager populator at `Compose.add()` time means `fem.partitions` is the single source of truth throughout, and ADR 0027's tag-determinism rule (the canonical `TagAllocator` runs once per emit pass) extends cleanly. |
| **Explicit composition formula folding source `fem_hash`s into host hash** | Redundant: the canonical-bytes hash of the post-compose neutral zone already covers all merged content, and is order-canonical so compose-order doesn't affect determinism. Source `fem_hash` is preserved in `/composed_from/{label}/@source_fem_hash` as provenance for `g.compose_list()` introspection — there is no need to fold it into a separate composition hash. |
| **Lift v1 nesting prohibition + ship `interface.export(...)` encapsulation in v1** | Encapsulation is a sibling primitive with its own H5 group, viewer mode, and resolver behavior — deserving its own ADR (0039, deferred). Bundling it with the nesting flag flip hides a separate feature under one decision. Ship nesting in v1; defer encapsulation to ADR 0039 (deferred) with feedback from real depth-1 / depth-2 usage. |
| **Fixed 1M reservation per module (original ADR draft)** | Breaks at depth ≥ 2: a nested source whose internal span exceeds 1M cannot fit in a 1M outer reservation. Auto-sizing per source's actual span (per §"Tag-offset scheme — per-module auto-sizing") handles any depth without user intervention. The fixed-size scheme is preserved only as the fallback granularity. |
| **Re-issue material/section tags by host TagAllocator (original ADR position)** | Created a second rewrite pass (element-to-material references must update post-re-issue) and a silent-corruption risk if a record kind's material-tag-reference field was missed. Uniform offset eliminates the carve-out: one rule, one cover set, one verifier check. Material *names* remain the user-facing identity. |

## Consequences

- **Test surface (Phase 3+4 of the work stream).** Round-trip tests
  for compose-then-emit canonical-bytes-equivalence-to-direct-assembly
  (the INV-1 lock). **The test compares the hash of the post-merge
  `/fem/` zone, NOT byte-equality of the emitted Tcl/Py deck —
  dict-iteration-dependent emission order over per-namespace
  material/section names means line-order in the deck may differ
  between compose-then-emit and assemble-directly-then-emit.** Tag-offset verifier coverage at multiple
  module counts (1 module, 2 modules sharing a rank, 2 modules on
  different ranks, 3+ modules with mixed `partition_rank=` hints).
  Mixed-rank-hint partition tests asserting the three-layer
  precedence rule (Layer 1 default, Layer 2 hint, Layer 3 explicit
  METIS override) including the override `UserWarning`. Schema
  round-trip for `/composed_from/` and the `module_label`
  parallel datasets, including the 2.8.x reader compatibility
  check. Lineage chain warn-not-raise on stale source (re-bake the
  module, observe `LineageStaleWarning` at the appropriate
  broker boundary, no hard fail).

- **Runtime cost.** Compose-time cost: one H5 read of the module's
  neutral zone + a tag-rewrite pass over the imported records + a
  namespace-prefix sweep over string-keyed names + the four
  verifier checks. O(records in module). Bounded by the module's
  size, not by the host's size. Build-time cost on the bridge:
  zero — the bridge sees a flat `FEMData` per INV-1, so
  `apeSees(fem)` cannot tell whether `fem` was composed or
  authored directly. Cross-rank constraint cost at SSI scale is
  measured by the Phase 1 microbenchmark already in flight;
  thresholds gate whether v1 ships the full feature or falls back
  to mesh-cache-only on the constraint-pricing side.

- **Debugging.** Compose-aware error messages always name the
  offending module label (`ComposeLabelError`, `ComposeAnchorError`,
  `ComposeDepthExceededError`, `ComposeNamespaceCollisionError`,
  `ComposeCapacityError`, `PartTagCollisionError`,
  `ComposeFilterWarning`, `ComposePartitionOverrideWarning`). New
  inspection helpers (`g.compose_inspect(path)` for pre-compose
  audit, `g.compose_list()` for post-compose audit) surface what's
  in the host without forcing the caller to read the broker by
  hand. The cross-partition viewer's `ColorMode.MODULE` mode
  (planned alongside the existing rank / PG / label modes) gives
  visual diagnosis of which elements came from which module.

- **Schema.** `neutral_schema_version` bumps 2.8.0 → 2.9.0 — an
  additive minor per ADR 0023. The additions:
  - New `/fem/composed_from/{label}/` provenance sub-group, one
    per composed module, attributes `source_fem_hash` (string),
    `source_neutral_schema_version` (string), `source_path`
    (string), `translate` (float[3]), `rotate` (float[4] or
    missing), `partition_rank` (int or missing), `composed_at`
    (ISO timestamp string), and an optional `properties`
    sub-attribute group.
  - New `/fem/@tag_span_max` (int64) attribute on the `/fem/` group,
    recording the max tag span observed across nodes / elements in
    the file. Read by compose at load time to size the per-module
    reservation without a full dataset scan; pre-2.9.0 files lack
    this attribute and trigger a one-shot fallback scan with
    `UserWarning`.
  - Optional `module_label` (variable-length string) parallel
    dataset on `/fem/nodes/` and on each `/fem/elements/{type}/`
    group, recording the `{label}` that contributed each row.
    Empty string for host-owned rows. 2.8.x readers ignore both.
  The two-version reader window from ADR 0023 keeps schema 2.8.x
  decks readable; re-emit from a 2.8.x reader produces a 2.8.x
  deck without compose information (lossy round-trip in the
  downgrade direction, as designed).

- **H5ModelReader Protocol widening.** Three new methods are added
  to the `H5ModelReader` Protocol contract per
  [ADR 0026](0026-h5modelreader-protocol-contract.md):
  `iter_composed_from() → Iterable[ComposedRecord]`,
  `composed_for_node(node_id) → str | None`,
  `composed_for_element(elem_id) → str | None`. Per ADR 0026's
  structural-Protocol stance, **each adapter implements these
  methods explicitly**. The apegmsh-native `H5Model` implementation
  reads `/fem/composed_from/` and the parallel `module_label`
  datasets. Foreign-format adapters (LS-DYNA d3plot, Exodus, xDMF)
  that have no notion of composition implement the methods as
  one-line no-ops: `def iter_composed_from(self): return ()`. The
  Protocol does not provide default method bodies — `typing.Protocol`
  is purely structural; runtime conformance checks via
  `@runtime_checkable isinstance` require every method to be defined
  on the implementing class.

## Implementation pointer

Phase 3 (Compose facade + merge engine) and Phase 4 (schema bump +
H5 reader changes) touch:

- New: `src/apeGmsh/mesh/_compose.py` — `Compose` facade,
  `_rebuild_partitions_from_modules` helper for the eager
  populator, `_rewrite_tag_refs` helper for the tag-rewrite cover
  set, `RESERVATION_GRANULARITY = 1_000_000` class attribute on the
  `Compose` facade, `_compute_source_span(source_path)` helper that
  reads `/fem/@tag_span_max` with fallback dataset scan, and the new
  exception types `ComposeCapacityError`, `ComposeDepthExceededError`,
  `ComposeNamespaceCollisionError`.
- New: `src/apeGmsh/_kernel/records/_compose.py` — `ComposeRecord`
  and `ComposeSet`, mirroring the `PartitionRecord` / `PartitionSet`
  pattern at `src/apeGmsh/_kernel/records/_partitions.py`.
- New: `src/apeGmsh/_kernel/record_sets.py` — registration of the
  new `ComposeSet` so `FEMData` exposes `composed_from` as a
  first-class attribute.
- Modified: `src/apeGmsh/mesh/FEMData.py` — `composed_from:
  tuple[ComposeRecord, ...]` attribute, plumbed through `__init__`
  / `__eq__` / `_canonical_bytes` / repr.
- Modified: `src/apeGmsh/mesh/_femdata_h5_io.py` —
  `NEUTRAL_SCHEMA_VERSION = "2.9.0"`; write paths for the
  `module_label` parallel datasets on `/fem/nodes/` and
  `/fem/elements/{type}/`; write path for `/fem/composed_from/`;
  read paths with `H5Lexists` probes per the optional-child rule
  from the bind-contract memory (h5py optional-child `.get()`
  hazard).
- Modified: `src/apeGmsh/mesh/_record_h5.py` — `physical_group`
  dtype gets an optional `module_label` attr (empty string for
  host-owned, label name for module-owned).
- Modified: `src/apeGmsh/opensees/_internal/compose.py` — the
  bridge composer (separate from the `g.compose()` facade despite
  the name collision; this is the bridge-side merge that produces
  the OpenSees deck from the broker) passes `fem.composed_from`
  through unchanged. No behavioural change here; the bridge sees
  a flat `FEMData` per INV-1.
- Modified: `src/apeGmsh/opensees/_internal/lineage.py` — the
  lineage chain uses `compose_hash()` (see below bullet) for both
  composed and uncomposed assemblies. On uncomposed
  `fem.composed_from == ()` input, `compose_hash()` is
  byte-equivalent to today's `fem_hash` (the wrapper concatenates
  only the host-native `canonical_bytes()` output with no
  module-separator suffixes appended). The migration is in-place
  for existing models — a contract test asserts bit-equality of
  pre-2.9.0 `fem_hash` vs post-2.9.0 `compose_hash()` on
  uncomposed fixtures.
- New: `compose_hash(fem)` in
  [opensees/_internal/lineage.py](../../_internal/lineage.py) —
  sorts records by `module_label` before delegating to existing
  `canonical_bytes()` for per-module hashing; ensures INV-1's
  compose-order invariance is structural, not implementation-
  derived.
- Each record dataclass declares a
  `tag_rewrite_spec: tuple[TagField, ...]` class attribute naming
  its tag-bearing fields (e.g.,
  `NodeRecord.tag_rewrite_spec = (TagField("tag", int),)`;
  `ConstraintRecord.tag_rewrite_spec` includes `master_node`,
  `slave_nodes[]`, `cnode`, `rnodes[]`, etc.). The compose rewriter
  iterates `RecordSet.registered_kinds()` and applies each spec
  uniformly; new record kinds register a spec at class-definition
  time or fail at module import. Hardens R1 (tag-rewrite cover-set
  drift) by making the cover set programmatically iterable and
  self-documenting.
- Modified: `src/apeGmsh/opensees/emitter/h5_reader.py` — Protocol
  widening: `H5ModelReader` gains the three method signatures
  (`iter_composed_from`, `composed_for_node`, `composed_for_element`)
  per ADR 0026's structural-Protocol rule; the apegmsh-native
  `H5Model` adds concrete implementations reading from
  `/fem/composed_from/` and the `module_label` parallel datasets.
  Foreign-format adapters added in future ADRs implement the methods
  as one-line no-ops on their own.
- New tests: `tests/mesh/test_compose_round_trip.py` (compose
  round-trip lock), `tests/mesh/test_compose_verifier.py` (the
  five verifier checks), `tests/mesh/test_compose_partitions.py`
  (the three-layer rank model), `tests/mesh/test_compose_schema.py`
  (2.8.0 ↔ 2.9.0 reader-window cross-compatibility),
  `tests/mesh/test_compose_lineage.py` (warn-not-raise on stale
  source), `tests/mesh/test_compose_nesting.py` (depth-limited
  nesting + separator alternation).

Phase 3 lands the `Compose` facade + merge engine + verifier.
Phase 4 lands the schema bump + H5 reader changes + Protocol
widening on `H5ModelReader`. The two phases ship sequentially —
Phase 3 alone is testable (compose into an in-memory `FEMData`,
assert canonical-bytes equivalence of the post-merge `/fem/` zone)
without persisting through H5;
Phase 4 closes the loop on round-trip.

## v1 scope gate

Phase 1 of the compose work stream is a cross-rank-constraint-cost
microbenchmark whose results gate whether the full feature ships or
falls back to a lighter alternative. Thresholds at the realistic
SSI-class point (10k embedded-node interface, 4 ranks, tet+line
embedded-host):

| Metric | Threshold | Rationale |
|---|---|---|
| `deck_emit_sec` | < 5.0 | Model-build is rare; 5s is invisible. |
| `deck_parse_py_sec` | < 2.0 | Per-run; user-noticeable threshold. |
| `deck_lines` | < 500_000 | Tcl files >500k lines stress editors. |
| `peak_rss_mb` | < 1_500 | Single-rank during emit. |

**Branches:**

- **All thresholds pass at 10k × 4:** proceed to Phase 2 (full
  feature).
- **10k × 4 passes but 100k × 8 fails:** proceed with
  `WARN_INTERFACE_SIZE = 50_000` hard warning at compose time,
  advising users to refactor large interfaces.
- **Any threshold breaches at 10k × 4:** abandon the full feature in
  v1. Ship Phase 2 (the `gmsh.isInitialized()` guard + tag-collision
  verifier — both have independent value) plus a stripped Phase 3
  that supports `g.compose()` for **mesh-cache-only flows** —
  meaning the composed module's mesh, materials, sections, and loads
  import, but **any cross-module MP-constraint**
  (`g.constraints.embedded`, `tied_contact`, `equalDOF`, `rigid_link`,
  `rigid_diaphragm`) raises `ComposeUnsupportedError`. The cost
  problem is symmetric across ADR 0027's five replicated-on-both-ranks
  Protocol methods, so blocking only `embedded` would be incomplete.
  Intra-module constraints within a single composed module still
  work. The fallback ships until ADR 0027's replication path is
  hardened under a future OpenSeesMP build with measured cross-rank
  constraint cost.

The fallback contract is documented here, not at the implementation
site, so a future maintainer reading this ADR understands the kill
switch even years after Phase 1 runs.

## Cross-references

- [ADR 0019](0019-opensees-model-read-side-broker.md) through
  [ADR 0023](0023-per-zone-schema-versioning.md) — the three-broker
  refactor; the persistence substrate compose builds on. Compose
  does not change the broker shape; it merges into it.
- [ADR 0023](0023-per-zone-schema-versioning.md) — additive-minor
  rule and two-version reader window. The neutral 2.8.0 → 2.9.0
  bump follows this policy.
- [ADR 0026](0026-h5modelreader-protocol-contract.md) — H5ModelReader
  Protocol; the read-side abstraction that gets widened with three
  new methods (`iter_composed_from`, `composed_for_node`,
  `composed_for_element`). Per ADR 0026's structural-Protocol stance
  each adapter implements them explicitly — foreign-format adapters
  (LS-DYNA d3plot, Exodus, xDMF) provide one-line no-op
  implementations.
- [ADR 0027](0027-cross-partition-mp-constraints.md) — cross-
  partition MP-constraint emission; the rank fan-out that compose
  feeds via the three-layer rank model. The compose rank
  assignment writes `fem.partitions`; ADR 0027's emission contract
  consumes it.
- [ADR 0034](0034-stage-bound-bcs-and-recorders.md) — stage-bound
  BCs / recorders / constraints; explicitly NOT inherited from
  composed modules per INV-6. Stages are analysis-time and host-
  owned.
- [ADR 0036](0036-embedded-host-decomposition.md) — embedded-host
  decomposition; the canonical constraint primitive for compose
  interfaces (a module's mesh meets the host's mesh non-
  conformally, and `g.constraints.embedded(...)` bridges the
  interface using the namespaced PG names).
- ADR 0039 (deferred) — `interface.export(...)` encapsulation
  primitive for composed-module external interfaces. Sibling
  primitive to compose with its own H5 group, viewer mode, and
  resolver behavior; deferred to a separate ADR per the alternatives
  table.
- ADR 0040 (deferred) — `g.loads.imposed_strain(...)` eigenstrain
  primitive for hybrid composite strain transfer. Adjacent
  capability surfaced by SSI-class compose use; deferred for a
  standalone treatment.
- Future ADR (deferred) — two-level partitioner (intra-module METIS
  while honoring module boundaries as hard cuts in the host's
  partition graph). v2 work; this ADR ships the one-rank-per-module
  default that v1 needs.
- `src/apeGmsh/core/_parts_registry.py:705-736` — the
  `{instance_label}.{pg_name}` namespacing precedent that compose
  extends.
- `src/apeGmsh/opensees/_internal/lineage.py:85` —
  `MODEL_HASH_EXCLUDED_CHILDREN`, the existing derived-not-canonical
  list that compose's DISCARD verdicts match.

## Amendment 2026-05-27 — Flat graft instead of tree graft

**Context.** Rule 3 of the Nested composition section originally
specified a *tree graft*: source's `/fem/composed_from/` subtree
copied under the host's `/fem/composed_from/{label}/composed_from/`
recursively, preserving the compose tree shape on disk.

**As shipped (PR #369).** The nested-compose merge engine uses a
**flat graft** instead: source's compose records get their labels
joined per rule 2 (`.` ↔ `/` separator alternation) and surface as
top-level entries in the host's `composed_from` tuple. The H5 layout
is therefore a flat list of `/fem/composed_from/{joined_label}/`
groups regardless of depth — no recursive `/composed_from/` sub-groups.

**Decision process.** Synthesized via a multi-agent design review
(architecture / implementation / UX audits) during PR #369. The
audits reached consensus on three points:

1. **No observable v1 loss.** For today's actual consumers —
   `compose_hash`, `g.compose_list()`, `composed_for_node` /
   `composed_for_element`, H5 round-trip, the `ColorMode.MODULE`
   viewer (PRs #372 / #373 / #374), `iter_composed_from()` on
   `H5Model` (PR #368) — flat is observationally equivalent to tree.
   The phrase *"informational, not load-bearing"* in the lineage
   section qualifies the **hash** (`host_fem_hash` is independent of
   provenance shape), not the disk-representation choice — so either
   shape satisfies the load-bearing contract.

2. **Tree-shape recovery is a thin derived view.** Joined labels
   parse unambiguously back into a tree via the separator-alternation
   rule. `g.compose_tree() -> tuple[ComposeTreeNode, ...]` ships this
   as a session-level helper (canonical primitive on
   `FEMData.compose_tree()`, session shim on `Compose.compose_tree()`,
   passthrough on `apeGmsh.compose_tree()`). Implementation lives at
   [src/apeGmsh/mesh/_compose.py](../../../mesh/_compose.py) with
   parser `_split_joined_label` as the strict inverse of
   `_join_module_label`.

3. **The hash representation is a one-way door, the disk shape is
   not.** If a future PR moved `_hash_composed_from`
   ([src/apeGmsh/mesh/_femdata_hash.py:108-137](../../../mesh/_femdata_hash.py))
   from its current flat fold over sorted top-level labels to a
   recursive fold over a tree shape, the digest space would change
   for **every existing composed FEMData file** — breaking the bind
   contract for stored results. The disk shape can be upgraded
   compatibly (additive field on `ComposeRecord` per
   [ADR 0023](0023-h5-schema-evolution.md)'s additive-minor rule);
   the hash cannot.

**Implications.**

- **The hash is the load-bearing surface.** It MUST stay non-recursive
  (flat fold over the sorted top-level joined labels). A future
  tree-shape addition lands as an additive optional field on
  `ComposeRecord` (e.g.,
  `composed_from: tuple[ComposeRecord, ...] = ()`) per ADR 0023; the
  hash code stays unchanged.
- **The separator-alternation rule is now load-bearing for tree
  reconstruction.** `_split_joined_label` depends on the alternation
  being unambiguous; relaxing it (e.g., allowing `/` in user-supplied
  `label=`) would break the parser. The `ComposeLabelError` validator
  in `_compose.py` forbids both `.` and `/` in user labels, so the
  invariant holds by construction. Document this dependency if the
  validator is ever loosened.
- **The H5 group-name sanitization** (`safe = label.replace("/", "_")`
  in `_write_composed_from` at
  [src/apeGmsh/mesh/_femdata_h5_io.py](../../../mesh/_femdata_h5_io.py))
  can collide for distinct labels `a/b` and `a_b`. The writer dedups
  via an order-dependent `__N` suffix on the group name; the verbatim
  `label` attribute round-trips correctly regardless, so this is
  internal-layout-only and not a public-surface hazard. Worth a code
  comment but no behavioral fix needed.

**Cross-references.**
- PR #369 (nested compose merge engine, depth verifier, separator
  alternation, flat graft).
- PR #370 (`compose_tree()` derived view).
- PRs #372 / #373 / #374 (`ColorMode.MODULE` viewer — consumes flat
  joined labels directly via `view.elements.module_for(eid)`; tree
  shape not required for v1 viewer).
