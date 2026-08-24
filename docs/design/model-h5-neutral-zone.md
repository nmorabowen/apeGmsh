# The `model.h5` neutral zone

`model.h5` is apeGmsh's canonical model archive. Its **neutral zone** —
everything at the file root that is not under `/opensees/` — describes a
meshed model without reference to any solver: where the nodes are, which
elements connect them, and which named sets group them.

This page publishes that zone as a **contract**. It is written for a
consumer that cannot import apeGmsh and does not have h5py — a browser
reader over `h5wasm`, a Rust importer, another meshing tool. Everything
below is what the writer emits **today**, verified against
`mesh/_femdata_h5_io.py` and against the committed golden fixture.

If you are reading this to change apeGmsh rather than to consume its
output, the internal specification —
[`src/apeGmsh/opensees/architecture/h5-schema.md`](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/architecture/h5-schema.md)
— covers all three zones and every record group. This page covers the
consumer-facing subset, in more detail, and is machine-checked.

!!! info "The contract lives in one place"
    The authoritative, machine-readable inventory is
    [`tests/fixtures/neutral_zone/inventory.py`](https://github.com/nmorabowen/apeGmsh/blob/main/tests/fixtures/neutral_zone/inventory.py).
    This page is prose about that table. `tests/mesh/test_neutral_zone_schema_note.py`
    holds all three artifacts together: it fails when the emitter drifts
    from the inventory, and when this page drifts from the inventory.
    If they ever disagree, the inventory wins and this page is the thing
    to fix.

## Who writes it

Two calls produce the neutral zone, byte-identically:

* `fem.to_h5(path)` — broker-only. Neutral zone and nothing else. No
  `/opensees/` group, which is the correct "no solver was involved"
  signal.
* `apeSees(fem).h5(path)` — composed. The same neutral zone, with the
  OpenSees enrichment layered under `/opensees/`.

A consumer of this page never needs to tell the two apart. Read the
neutral zone; ignore what else is in the file.

## Version rule

**Read this before reading any data.** Getting it wrong is the one
mistake that produces plausible-looking garbage instead of an error.

Versions are **semver strings**, not integers — `"2.31.0"`, stored as
HDF5 variable-length UTF-8 string attributes. Each zone carries its own
independent version; they do not share a number.

| Attribute on `/meta` | Gates | Authoritative? |
|---|---|---|
| `neutral_schema_version` | everything on this page | **yes** |
| `opensees_schema_version` | the `/opensees/` zone | yes, for that zone |
| `schema_version` | — | **no** — legacy envelope |

The neutral zone is gated by **`neutral_schema_version` alone**. At the
time of writing the writer stamps `2.31.0`.

`schema_version` is a back-compatibility envelope that predates the
per-zone split. Its value is "whichever writer wrote last" — the
composer may overwrite it. **Do not branch on it.** Use it only as a
fallback when `neutral_schema_version` is absent entirely, which means
the file predates the split.

Note that a broker-only file still stamps `opensees_schema_version`,
even though it has no `/opensees/` group. Its presence is not evidence
that a solver zone exists; check for the group.

### The two-version window

A reader declares the version it was written against. Call that
`R.major.R.minor`. Given a file at `F`:

| Condition | Behaviour |
|---|---|
| `F.major != R.major` | **refuse** |
| `F.minor == R.minor` | accept |
| `F.minor == R.minor - 1` | accept |
| `F.minor < R.minor - 1` | **refuse** — too old |
| `F.minor > R.minor` | **refuse** — newer than the reader |

Patch differences are always ignored.

The window runs **backward only**. A current reader opens the previous
minor's files; a reader older than the file refuses it. There is no
forward tolerance, and that is deliberate: a newer minor may carry
meaning the reader cannot see, and silently rendering a model wrong is
worse than declining to render it.

So: **on an unknown version, refuse visibly.** Show the user the file's
version and the range you support. Never fall back to "parse what I
recognise and hope" — the whole point of the window is that this
failure mode is not available.

This is the rule apeGmsh's own reader enforces, in
`opensees/_internal/schema_version.py` (`validate_zone_version`). The
same rule, with the same window, governs the `/opensees/` bridge zone
via `opensees_schema_version` and the `/stages/` results zone via
`results_schema_version`; a consumer that later reads those zones
applies this section unchanged, once per zone, against that zone's own
key.

### What a minor bump means for you

A minor bump is **additive**: new datasets, new attributes, new
optional groups. Existing required fields keep their name, dtype and
meaning.

Which gives the compatibility promise: **ignore fields you do not
recognise.** A dataset or attribute appearing that this page does not
mention is a newer apeGmsh being additive, not a corrupt file. Skip it.

The exception, and the reason the window exists at all: a minor bump
that *restructures* required content walks the window forward and locks
the prior minor out. That has happened once in the neutral zone (the
`2.10.0` split of the named-index groups into two sides). Honour the
window and you are safe from it.

## Zone map

```
model.h5
├── /meta                    ← this page
├── /nodes                   ← this page
├── /elements                ← this page
├── /physical_groups         ← this page
├── /labels                  ← this page
├── /mesh_selections         ← this page
│
├── (record groups)          neutral, but not published here
│   /loads /masses /constraints /contacts /contact_planes
│   /interfaces /reinforce_ties /embed_ties /rebar_elements
│   /partitions /parts /composed_from
│
└── /opensees/               the solver bridge zone — separate version key
```

The record groups are part of the neutral zone and are written by the
same writer; they carry compound dtypes whose fields change more often
than the geometry does, so they are documented internally rather than
published. Every one of them is **omitted entirely when empty**, so a
model that declares no loads simply has no `/loads`.

## Conventions

Three rules hold everywhere below.

**Absence means "not declared".** Optional groups and datasets are
omitted, never written empty as a placeholder. A missing `/mesh_selections`
means the model had no selections — not that data was lost.

**Strings are HDF5 variable-length UTF-8.** Both string attributes and
the string datasets use `H5T_STRING`, `H5T_VARIABLE`, `cset =
H5T_CSET_UTF8`. There are no fixed-width string datasets in this zone.
(h5py hands these back as `bytes` and you must decode; h5wasm gives you
JavaScript strings.)

**Nothing is chunked or compressed.** Every dataset is contiguous with
no filter pipeline, so a reader can slice without a decompressor.

## `/meta`

Attributes only — no datasets, no children except the optional
`/meta/lineage`. Always present.

| Attribute | Type | Meaning |
|---|---|---|
| `neutral_schema_version` | string | **the gate for this zone.** See [Version rule](#version-rule). |
| `opensees_schema_version` | string | version of the `/opensees/` zone. Stamped even when that zone is absent. |
| `schema_version` | string | legacy envelope. Non-authoritative — do not branch on it. |
| `apeGmsh_version` | string | producing apeGmsh version. **May be the empty string** — `fem.to_h5()` does not fill it in. |
| `created_iso` | string | ISO 8601 UTC timestamp of the write. The only attribute that varies between two writes of the same model. |
| `ndm` | int64 | spatial dimension, derived as the maximum element dimension present (`3` when there are no elements to derive it from). Note this does **not** change the width of `/nodes/coords`. |
| `ndf` | int64 | DOFs per node. **`0` means "not declared"**, not "zero DOFs" — broker-only writes always pass 0. |
| `snapshot_id` | string | hash of the `FEMData` snapshot; 32 lowercase hex characters. |
| `model_name` | string | user-supplied name. May be the empty string. |
| `tag_span_max` | int64 | `max(max_node, max_elem) - min(min_node, min_elem) + 1` over nodes and elements together. Sizes tag reservations when composing. `0` for an empty mesh. |

Every string attribute here is a scalar variable-length UTF-8 string;
every integer attribute is a scalar `int64`.

### `/meta/lineage`

Optional sub-group, attributes only, no datasets. Carries the
content-hash chain.

| Attribute | Type | Present when |
|---|---|---|
| `fem_hash` | string | the neutral zone was written and the snapshot hash was computable |
| `model_hash` | string | an `/opensees/` zone was composed onto it |
| `results_hash` | string | this is a composed results archive |

On a broker-only file only `fem_hash` is written, and it equals
`/meta`'s `snapshot_id`. Treat a missing `/meta/lineage` as "older file,
no lineage recorded" — a warning at most, never a failure.

## `/nodes`

Always present.

| Dataset | dtype | Shape | Always? |
|---|---|---|---|
| `/nodes/ids` | int64 | `(N,)` | yes |
| `/nodes/coords` | float64 | `(N, 3)` | yes |
| `/nodes/module_label` | vlen UTF-8 | `(N,)` | yes |
| `/nodes/ndf` | int8 | `(N,)` | no |
| `/nodes/provenance` | int8 | `(N,)` | no |

All five are row-aligned: row `i` of every dataset describes the same
node.

`/nodes/coords` **always has three columns**, including in a 2D model
where `/meta`'s `ndm` is 2. The third column is present and zero. Do not
size your buffer from `ndm`.

`/nodes/module_label` records which composed source module each node came
from; it is the empty string for host-owned rows, which is every row in
an uncomposed model. It is always written, so its presence tells you
nothing about whether the model was composed.

`/nodes/ndf` is the per-node DOF count, `0` meaning undeclared. It is
absent unless the model actually declared per-node DOFs — the common
case has no such dataset, so treat absence as "no per-node DOF
information", not as an error.

`/nodes/provenance` marks how a node came to exist: `0` for a mesh node,
`1` for a decoupled node with no element attached. Absent when the model
has no decoupled nodes.

## `/elements`

Always present as a group, **even when the model has no elements**. One
sub-group per gmsh element-type alias — `hex8`, `tet4`, `tri3`,
`triangle3`, `line2`, and so on. A type with no elements gets no
sub-group.

These aliases are gmsh's taxonomy, deliberately not OpenSees type names.
`/opensees/element_meta/` keys the same elements by OpenSees type token;
the two namespaces are separate on purpose.

Each `/elements/{type}` group carries five attributes:

| Attribute | Type | Meaning |
|---|---|---|
| `code` | int64 | gmsh element-type code |
| `gmsh_name` | string | gmsh's human name, e.g. `"Tetrahedron 4"` |
| `npe` | int64 | nodes per element — the width of `connectivity` |
| `dim` | int64 | topological dimension (1 line, 2 surface, 3 volume) |
| `order` | int64 | interpolation order |

and three datasets:

| Dataset | dtype | Shape | Always? |
|---|---|---|---|
| `/elements/{type}/ids` | int64 | `(E,)` | yes |
| `/elements/{type}/connectivity` | int64 | `(E, npe)` | yes |
| `/elements/{type}/module_label` | vlen UTF-8 | `(E,)` | yes |

Row-aligned, as with nodes. Node ordering within a connectivity row is
gmsh's node ordering for that element type — take it from `gmsh_name`
and `code`, not from a guess.

## Index base — read this carefully

This is where a reader most easily goes wrong, because the failure is
usually silent and looks like a slightly scrambled mesh.

**Ids are identifiers, not indices.**

* `/nodes/ids` holds gmsh node tags. They are **1-based** and they are
  **not guaranteed contiguous or sorted**. A composed model reserves tag
  spans per module, so gaps are ordinary.
* `/elements/{type}/connectivity` holds **node ids** — values that
  appear in `/nodes/ids` — not row indices into `/nodes/coords`.

So the reader must build the map itself:

```
id_to_row = {}
for row, node_id in enumerate(nodes.ids):
    id_to_row[node_id] = row

# a triangle's three vertex positions
for element_row, conn_row in enumerate(connectivity):
    xyz = [coords[id_to_row[node_id]] for node_id in conn_row]
```

Subtracting 1 from every id happens to work on a simple single-part
model, where the ids come out as exactly `1..N`. It breaks the moment
anyone composes two modules or deletes an entity. Build the map.

**Element ids share one numbering space across all types**, and that
space also counts elements that were never exported. In the golden
fixture the eight hexahedra carry ids 5 through 12 — ids 1 through 4
belong to the boundary quads that a 3D extraction dropped. Element ids
are therefore neither 0-based, nor 1-based per type, nor dense.

Which has a direct consequence for named sets, below.

## `/physical_groups`

Optional — omitted entirely when the model declares no physical groups.

Physical groups are split into **two independent sides**:

```
/physical_groups/
├── node_side/{name}/
└── element_side/{name}/
```

A group declared on both composites is written into both sub-trees. A
group that exists only on one side appears only there. **Walk each side
independently**; do not infer one from the other, and do not merge them.
Either sub-group may be absent.

`{name}` is the group's name with `/` replaced by `_`. If two groups
sanitize to the same name, the second gets `__{dim}_{tag}` appended. The
**real** name is always the `name` attribute — use the sub-group key
only as a key.

Each entry carries three attributes and up to three datasets:

| Attribute | Type | Meaning |
|---|---|---|
| `dim` | int64 | topological dimension of the group |
| `tag` | int64 | gmsh physical tag |
| `name` | string | the authoritative name |

| Dataset | dtype | Shape | Always? |
|---|---|---|---|
| `/physical_groups/node_side/{name}/node_ids` | int64 | `(Np,)` | yes |
| `/physical_groups/node_side/{name}/node_coords` | float64 | `(Np, 3)` | yes |
| `/physical_groups/element_side/{name}/node_ids` | int64 | `(Np,)` | yes |
| `/physical_groups/element_side/{name}/node_coords` | float64 | `(Np, 3)` | yes |
| `/physical_groups/element_side/{name}/element_ids` | int64 | `(Ep,)` | no |

`element_ids` appears on the element side only, and only when the group
has `dim >= 1` and a non-empty element set.

!!! warning "`element_ids` may name elements that are not in the file"
    A physical group records the elements gmsh assigned to it. If the
    `FEMData` snapshot was extracted at a dimension that excludes them,
    those elements are **not** in any `/elements/{type}` group but their
    ids are still listed here.

    The golden fixture is built to exercise exactly this: its `Base`
    group is a face of the cube, and its `element_ids` are `[1, 2, 3, 4]`
    — quads that a `dim=3` extraction dropped. Nothing in the file
    resolves them.

    Resolve PG element ids **defensively**: skip ids you cannot find.
    The node-side membership (`node_ids` / `node_coords`) is always
    self-contained, so it is the safer thing to highlight from.

The `node_coords` datasets duplicate coordinates already in
`/nodes/coords`. They are a convenience for readers that want a group's
geometry without the id map; they are row-aligned with that entry's own
`node_ids`, not with `/nodes/ids`.

## `/labels`

apeGmsh's internal labels — the names given to geometry at construction
time, as distinct from gmsh physical groups. Optional, omitted when
there are none.

**Structurally identical to `/physical_groups`** in every respect: the
same two-sided split, the same `dim` / `tag` / `name` attributes, the
same datasets with the same dtypes and the same optionality.

| Dataset | dtype | Shape | Always? |
|---|---|---|---|
| `/labels/node_side/{name}/node_ids` | int64 | `(Np,)` | yes |
| `/labels/node_side/{name}/node_coords` | float64 | `(Np, 3)` | yes |
| `/labels/element_side/{name}/node_ids` | int64 | `(Np,)` | yes |
| `/labels/element_side/{name}/node_coords` | float64 | `(Np, 3)` | yes |
| `/labels/element_side/{name}/element_ids` | int64 | `(Ep,)` | no |

A single reader function should handle `/physical_groups/node_side`,
`/physical_groups/element_side`, `/labels/node_side` and
`/labels/element_side` — they differ only in what they mean, not in how
they are laid out.

## `/mesh_selections`

Post-mesh selection sets. Optional — omitted when the model has no
selection store or the store is empty.

**This group is flat.** It has no `node_side` / `element_side` split;
entries live directly under `/mesh_selections/{name}`. Selections are a
single store with no notion of sides, so there was nothing to split. Do
not reuse the named-index walker here without accounting for the missing
level.

Same three attributes as a named-index entry — `dim`, `tag`, `name` —
where `dim` is `0` for a node-level selection.

| Dataset | dtype | Shape | Always? |
|---|---|---|---|
| `/mesh_selections/{name}/node_ids` | int64 | `(Np,)` | yes |
| `/mesh_selections/{name}/node_coords` | float64 | `(Np, 3)` | yes |
| `/mesh_selections/{name}/element_ids` | int64 | `(Ep,)` | no |
| `/mesh_selections/{name}/connectivity` | int64 | `(Ep, npe)` | no |

`element_ids` and `connectivity` are written together, for `dim >= 1`
selections with a non-empty element set — either both are present or
neither is. `connectivity` rows align 1:1 with `element_ids`, and hold
node ids on the same terms as `/elements/{type}/connectivity`.

## What is always there

The minimum a valid neutral zone guarantees:

* `/meta`, with `neutral_schema_version`, `ndm`, `ndf`, `snapshot_id`,
  `tag_span_max`, and possibly-empty `apeGmsh_version` / `model_name`.
* `/nodes`, with `ids`, `coords` and `module_label`.
* `/elements` as a group — possibly with no children.

Everything else on this page is conditional. A reader that handles a
file containing only those three has handled the degenerate case
correctly.

## For consumers

**Ignore what you do not recognise.** Minor versions add datasets,
attributes and groups. Encountering one this page does not describe
means you are reading a newer file, not a broken one. Skip it and carry
on.

**Refuse what you cannot understand.** The corollary. If
`neutral_schema_version` falls outside your two-version window, stop and
say so, showing the file's version and your supported range. Do not
partially render.

**Check yourself against the golden.** The conformance artifact is
[`tests/fixtures/neutral_zone/box.h5`](https://github.com/nmorabowen/apeGmsh/blob/main/tests/fixtures/neutral_zone/box.h5)
— a 41 KB unit cube, 27 nodes and 8 hexahedra, deterministic, with two
physical groups on both sides, a label, and a flat mesh selection. It is
built to exercise the awkward parts: non-contiguous element ids, and a
physical group whose `element_ids` do not resolve. Vendor it into your
own test suite and read it there.

It is regenerated by
[`tests/fixtures/neutral_zone/_generate_fixtures.py`](https://github.com/nmorabowen/apeGmsh/blob/main/tests/fixtures/neutral_zone/_generate_fixtures.py),
so the bytes are reproducible from the repo's own API rather than
hand-rolled.

**A checklist for a new reader**, in the order the mistakes happen:

1. Read `/meta`'s `neutral_schema_version`; validate the window; refuse
   loudly if outside it.
2. Read `/nodes/ids` and `/nodes/coords`; build the id → row map.
3. Read each `/elements/{type}`; resolve connectivity **through the
   map**, not by subtracting 1.
4. Size vertex buffers at three components per node, whatever `ndm`
   says.
5. Walk the named-index sides independently; take names from the `name`
   attribute.
6. Resolve physical-group `element_ids` defensively; expect some to
   dangle.
7. Treat every optional dataset's absence as information, not failure.

---

*See also: [The broker](broker.md) — how a session's declarations become
the frozen `FEMData` snapshot this file serialises.*
