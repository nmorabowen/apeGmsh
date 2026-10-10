# `model.h5` — canonical model archive

`fem.to_h5(path)` and `apeSees(fem).h5(path)` both write an HDF5 file
that captures **the full model definition**:

* `fem.to_h5(path)` — broker-only.  Writes just the neutral zone:
  nodes, elements, physical groups, labels, broker-side constraints,
  loads, and masses.  Output is solver-agnostic and complete enough
  for a viewer to render the mesh without OpenSees loaded.
* `apeSees(fem).h5(path)` — composed.  Layers the OpenSees
  enrichment (materials, sections, transforms, beam_integration,
  time_series, patterns, bcs, recorders, analysis, and per-type
  element metadata) under `/opensees/` on top of a broker neutral
  zone.

The result is **the canonical model archive** for an apeGmsh
session.  It carries information that is in *neither* the FEMData
snapshot in memory (geometry only) nor the STKO/MPCO results
(response only).

## Design principles

1. **One file, navigable as a graph.** Cross-references are HDF5
   paths (`/opensees/sections/Fiber_1`), not numeric tags. Anyone
   with `h5py` or `h5dump` can walk the model.
2. **Structured groups, scalar attrs, array datasets.** No
   JSON-blob attributes. HDF5-native types throughout so introspection
   tools work.
3. **Schema-versioned per zone.** Readers MUST check the per-zone
   key for each zone they read (`neutral_schema_version` /
   `opensees_schema_version` / `results_schema_version` /
   `geometry_schema_version` / `provenance_schema_version`) and refuse
   incompatible files — not the legacy `/meta/schema_version`
   envelope. See [Schema versioning](#versioning).
4. **Lazy and partial.** `model.h5` may be written at any point in
   the bridge lifecycle; absent groups indicate "user did not declare
   this," not "data is missing." The viewer must tolerate any subset.
5. **HDF5 emit is decoupled from execution.** Writing the H5 does not
   imply analysis was run. The H5 is a definition snapshot, not a
   results file.

## Two zones

`model.h5` is partitioned into a **neutral zone** at the root and a
**solver-specific zone** under `/opensees/`.

* The **neutral zone** (broker-owned, Phase 8.5) holds geometry and
  pre-solver model declarations. Every top-level group/dataset is
  written by [`write_neutral_zone`](../src/apeGmsh/mesh/_femdata_h5_io.py) (the
  enumeration below tracks that function — the ratchet test
  `test_h5_schema_doc_names_every_neutral_zone_group` fails if a new
  one is added here without a matching doc mention): `/meta`, `/nodes`,
  `/elements/{type}`, `/physical_groups`, `/labels`,
  `/mesh_selections`, `/partitions`, `/parts`, `/constraints/{kind}`,
  `/reinforce_ties`, `/embed_ties`, `/rebar_elements`, `/contacts`,
  `/contact_planes`, `/interfaces`, `/loads/{kind}/{pattern}`,
  `/masses`, `/composed_from`. Most of these are omitted entirely when
  the model has nothing to report for them (e.g. no ties, no
  partitions) — absence is the "not declared" signal, not "data
  missing". These describe the model independent of which solver
  consumes it.
* The **OpenSees zone** (bridge-owned, Phase 8.4) holds anything the
  OpenSees adapter contributes: `/opensees/materials`,
  `/opensees/sections`, `/opensees/transforms`,
  `/opensees/beam_integration`, `/opensees/element_meta` (per-type
  OpenSees args + cross-refs), `/opensees/time_series`,
  `/opensees/patterns`, `/opensees/bcs`, `/opensees/recorders`,
  `/opensees/analysis`.

A second producer (Code_Aster, Abaqus, …) would plug in at
`/<solver>/` next to `/opensees/` without colliding.  The neutral
zone is always present in a full file; the OpenSees zone is present
only when `apeSees(fem).h5(...)` produced the file (or when the
user explicitly drove an `H5Emitter`).  `fem.to_h5(path)` writes
neutral-zone-only files.

**Schema bumps.**
* Phase 8.4 — bridge groups moved under `/opensees/`.  Breaking
  (`1.x.y → 2.0.0`); any external tool that read a pre-8.4 file by
  absolute path (`/materials/uniaxial/...`) sees a
  `SchemaVersionError` from the reference reader.
* Phase 8.5 — broker neutral zone added.  Additive (`2.0.0 →
  2.1.0`); old v2.0.0 readers tolerate the absence of the new
  groups and still parse pre-8.5 files unchanged.

## Top-level layout

```
model.h5
├── /meta                                  attrs only
│
├── ── neutral zone (broker-owned) ──
├── /nodes
│     ├── ids                              (N,) int64
│     └── coords                           (N, 3) float64
├── /elements
│     └── /{gmsh_alias}                    one group per element type (tet4, hex8, …)
├── /physical_groups                      schema 2.10: side-partitioned
│     ├── /node_side/{name}                node-side physical group entries
│     └── /element_side/{name}             element-side physical group entries
├── /labels                                schema 2.10: side-partitioned
│     ├── /node_side/{name}                node-side label entries
│     └── /element_side/{name}             element-side label entries
├── /mesh_selections                      (optional)
│     └── /{name}                          one group per post-mesh selection set
├── /partitions                           (optional, schema 2.5.0)
│     └── /{id}                            one group per Gmsh partition rank
├── /parts                                (optional, schema 2.5.0)
│     └── /{label}                         one group per Part label
├── /constraints
│     └── /{kind}                          one dataset per constraint kind
├── /reinforce_ties                       (optional, schema 2.15.0)
│     └── ties                             one symmetric-compound dataset
├── /embed_ties                           (optional, schema 2.22.0)
│     └── ties                             one symmetric-compound dataset
├── /rebar_elements                       (optional, schema 2.16.0)
│     └── elements                         one symmetric-compound dataset
├── /contacts                             (optional, schema 2.21.0)
│     └── contacts                         one symmetric-compound dataset
├── /contact_planes                       (optional, schema 2.24.0)
│     └── contact_planes                   one symmetric-compound dataset
├── /interfaces                           (optional, schema 2.29.0)
│     └── interfaces                       one symmetric-compound dataset
├── /loads
│     ├── /nodal/{pattern}                 one dataset per pattern
│     ├── /element/{pattern}               one dataset per pattern
│     └── /sp/{pattern}                    single-point constraints, one dataset
│                                           per case name (schema 2.26.1; earlier
│                                           files flatten every record into `default`)
├── /masses                                single dataset
├── /composed_from                        (optional, schema 2.9.0)
│     └── /{label}                         one group per composed source module
│
├── ── own zones (ADR 0112 D2/D3, own version keys) ──
├── /provenance                            (optional; see /provenance below)
│     ├── /files                           one row per source file
│     ├── /sites                           one row per (file, line, function)
│     └── /records                         one row per declaration path
├── /assembly                              (optional; Assembly.h5 only, see /assembly below)
│     ├── /instances                       one row per instance
│     └── /ties                            one row per tie, coupling or reference node
│
└── /opensees/                             ── OpenSees zone (bridge-owned) ──
      │   @will_solve, @solve_refusals,      the solve stamp (optional attrs,
      │   @requires                          opensees 2.24.0; see below)
      ├── /materials
      │     ├── /uniaxial/{name}           one group per material
      │     └── /nd/{name}
      ├── /sections
      │     └── /{name}                    one group per section
      ├── /transforms
      │     └── /{name}                    one group per geomTransf
      ├── /beam_integration
      │     └── /{name}                    one group per beamIntegration
      ├── /element_meta
      │     └── /{type_token}              one group per OpenSees element type
      ├── /time_series
      │     └── /{name}                    one group per series
      ├── /patterns
      │     └── /{name}                    one group per pattern
      ├── /bcs
      │     ├── /fix                       single dataset
      │     └── /mass                      single dataset
      ├── /recorders
      │     └── /{name}                    one group per recorder
      ├── /cuts                            (optional, v4)
      │     └── /cut_{i}                   one group per persisted SectionCutDef
      ├── /sweeps                          (optional, v4)
      │     └── /sweep_{i}                 one group per persisted SectionSweepDef
      ├── /analysis                        attrs + sub-attrs (optional)
      ├── /commands                        calls with no typed store (optional, opensees 2.23.0)
      └── /program                         run-length emit order (opensees 2.23.0)
```

The user's PG names, material names, etc. are HDF5 group names — they
must therefore avoid `/` characters. The producers enforce this at
declaration time.

`/opensees` is created lazily: a file produced by `fem.to_h5(path)`
contains no `/opensees` group at all (the broker doesn't know about
OpenSees).  Conversely, the neutral zone is only present when a real
:class:`FEMData` drove the writer; standalone `H5Emitter` test
output contains just `/meta` + `/opensees/...`.

`/elements/{gmsh_alias}` (broker) and `/opensees/element_meta/{type_token}`
(bridge) are two different element keyings of the same underlying
elements.  The broker key is a GMSH alias (`tet4`, `hex8`, `line2`);
the bridge key is the OpenSees type token (`FourNodeTetrahedron`,
`stdBrick`, `forceBeamColumn`).  Consumers cross-reference between
them via the element tag (the `ids` dataset is shared in shape and
content).

## `/meta`

Attributes only.

| Attribute | Type | Description |
|---|---|---|
| `schema_version` | string | **legacy envelope** — back-compat only; *not* authoritative (see [Schema versioning](#versioning)) |
| `neutral_schema_version` | string | per-zone version of the broker neutral zone (e.g. `"2.32.0"`) |
| `opensees_schema_version` | string | per-zone version of the `/opensees/` zone (e.g. `"2.20.0"`); forward-stamped even on broker-only files |
| `apeGmsh_version` | string | producing apeGmsh version |
| `created_iso` | string | ISO 8601 timestamp |
| `ndm` | int | the model's spatial dimension as declared by `ops.model(ndm=)`, never the mesh dimension: a line-only 2-D frame carries `2`. `0` on broker-only files (`fem.to_h5`), which declare none, and deck-building / capture readers refuse it (#1291, neutral 2.34.0; older composed files stamped the highest element dimension) |
| `ndf` | int | DOFs per node as declared by `ops.model(ndf=)`; `0` on broker-only files |
| `snapshot_id` | string | hash of FEMData snapshot the bridge was built from |
| `session_id` | string | the writing session's uuid4, canonical 36-character form; pairs the file with its geometry sibling and is **never hashed** (see [`/meta/session_id`](#metasession_id-and-the-geometry-sibling)). **Optional**: absent in files at neutral ≤ 2.34.0, and read **without a version gate**: an additive `/meta` attr bumps no zone, older readers ignore it (ADR 0023 INV-2), and a reader that finds none mints a fresh id |
| `geometry_schema_version` | string | per-zone version of `/geometry`; present only in a `<stem>.geometry.h5` sibling |
| `provenance_schema_version` | string | per-zone version of `/provenance`; present only when the file carries `/provenance` |
| `model_name` | string | user-provided model name |
| `tag_span_max` | int | `max(max_node,max_elem) - min(min_node,min_elem) + 1`; sizes compose tag-offset reservations (ADR 0038) |

> The `/meta/lineage/` sub-group (ADR 0021) is stamped whenever the
> neutral zone is written, not only by the composer: `write_fem_h5`
> (the broker-only `fem.to_h5(path)` path) stamps just `fem_hash`
> (there is no `/opensees/` zone to fold into a `model_hash` yet); the
> composer additionally computes and stamps `model_hash` once
> `/opensees/...` exists. The composed `results.h5` carries its
> version keys at the **file root** (not `/meta`) — see
> [Schema versioning](#versioning).

### `results.h5` root attrs: OpenSees build provenance

A `results.h5` written by the native writer (`NativeWriter.open`) may carry
two root attrs naming the OpenSees build that produced its data. They come
from the bridge's `BackendInfo` (`apeGmsh.opensees._target`), whose one fork
signal is `ladrunoBuild()` returning a git sha, and are written with the
other root attrs, before any stage data.

| Attribute | Type | Description |
|---|---|---|
| `opensees_backend` | string | `"fork"` (Ladruno) or `"stock"`. Absent when the writer's caller did not know which binary ran |
| `opensees_build` | string | the 40-character git sha the fork binary was compiled from (`ladrunoBuild()`). Absent on stock, on a fork build predating `ladrunoBuild` (which reads as stock), and when unknown; never an empty string |

Both are **optional** and read **without a version gate**: like
`/meta/session_id`, an additive provenance attr bumps no zone and older
readers ignore it (ADR 0023 INV-2). Neither is deck-affecting, so the
minor-bump rule for additive content does not apply.

Schema versioning is **per-zone**, **strict on major**, and a
**floor per zone** on minor
([ADR 0113](decisions/0113-compatibility-is-a-floor-per-zone.md), which
retired [ADR 0023](decisions/0023-per-zone-schema-versioning.md)'s
two-version reader window). A reader written for `X.Y.*` accepts every
same-major minor from the zone's floor up to `X.Y` and refuses lower
minors, newer minors, or any other major. See
[Schema versioning](#versioning) for the floors and the shim ledger.

## `/nodes`

Neutral-zone group at the root, broker-owned.

```
/nodes/
├── ids               (N,) int64
├── coords            (N, 3) float64
├── ndf               (N,) int8              optional, schema 2.7.0
├── provenance        (N,) int8              optional, schema 2.11.0 (0=mesh, 1=decoupled)
└── module_label      (N,) vlen-utf-8        always written, schema 2.9.0
```

`ndf` carries per-node DOF-count overrides declared via
`g.node_ndf.set(...)` / `g.node_ndf.set_default(...)` (shell-to-solid
coupling, S1b); omitted when the broker has no populated `_ndf` array.
`provenance` distinguishes ordinary mesh nodes from nodes synthesized
by `g.decouple_node` (ADR 0049); omitted when the broker has no
decoupled nodes. `module_label` is **always** written (unlike the two
above) so a compose-aware reader has a stable shape contract — it
carries the source module's label for compose-merged rows (ADR 0038)
and `""` for host-owned rows / the uncomposed case.

The viewer renders the mesh substrate from `/nodes/coords` keyed by
`/nodes/ids`.  Bridge-side data that refers to nodes (loads,
constraints, fix records) does so via the tag string in the
`target` field of the symmetric record compound — see
[Symmetric compound contract](#symmetric-compound-contract) below.

## `/elements`

Neutral-zone group at the root, broker-owned (Phase 8.5).  One
sub-group per **GMSH element type alias** (`tet4`, `hex8`,
`line2`, `triangle3`, …).  Each element of that type sits in the
matching group regardless of which PG it belongs to.

```
/elements/tet4/
├── attrs: code=4, gmsh_name="Tetrahedron 4", npe=4, dim=3, order=1
├── ids               (E_t,) int64                — element tags
├── connectivity      (E_t, 4) int64              — node tags per element
└── module_label      (E_t,) vlen-utf-8           — always written, schema 2.9.0

/elements/line2/
├── attrs: code=1, gmsh_name="Line 2", npe=2, dim=1, order=1
├── ids
├── connectivity      (E_l, 2) int64
└── module_label      (E_l,) vlen-utf-8
```

`module_label` is always written (same contract as `/nodes/module_label`
above) — the source module's label for compose-merged rows (ADR 0038),
`""` for host-owned rows / the uncomposed case.

OpenSees-specific element metadata (positional args, cross-references)
lives under `/opensees/element_meta/{type_token}` — a parallel index
keyed by OpenSees type name rather than GMSH alias.  See that section
below for the cross-reference contract.

## `/physical_groups`

Neutral-zone group at the root, broker-owned.  **Schema 2.10
side-partitioned**: node-side and element-side entries live under
separate `node_side/` and `element_side/` sub-trees rather than
flattening into a single namespace.

```
/physical_groups/
├── /node_side/Slab/
│     ├── attrs: dim=2, tag=100, name="Slab"
│     ├── node_ids          (Np,) int64
│     └── node_coords       (Np, 3) float64
└── /element_side/Slab/
      ├── attrs: dim=2, tag=100, name="Slab"
      ├── element_ids       (Ep,) int64
      ├── node_ids          (Np,) int64       — element-derived membership
      └── node_coords       (Np, 3) float64
```

A PG that exists on both composites (the gmsh-extracted convention,
where node membership is derived from element membership) gets one
entry per side. A PG that exists only on the element-side broker
composite (the hand-constructed case) writes only into
`element_side/`, with no phantom `node_side/` entry.

**The 2.10 split closes the snapshot_id drift bug**
([ADR 0021's 2026-05-28 amendment](decisions/0021-lineage-chain-replaces-snapshot-id.md#amendment--2026-05-28--inv-1-retired-with-schema-210-b2--pr-398)
and [project memory `project_h5_schema_2_10_b2_shipped`](../README.md)).
Prior to 2.10 the layout was flat (`/physical_groups/{name}/...`)
and the reader heuristically classified entries by field presence,
producing phantom node-side entries that flipped the hash on
hand-constructed FEMData round-trips.

**Reader semantics**: each sub-tree is walked independently. No
inference, no shared-dataset reuse across sides. A PG present on
both sides produces two restored composite entries that share the
`(dim, tag)` key — `nodes.physical[(dim, tag)]` and
`elements.physical[(dim, tag)]` — the broker treats them as
independent records (which they are, semantically).

## `/labels`

Same shape and side-partition as `/physical_groups`. Node-side
labels live under `labels/node_side/{name}`, element-side under
`labels/element_side/{name}`. Each entry carries the same fields
(`dim`, `tag`, `name`, `node_ids`, `node_coords`, optional
`element_ids`).

## `/mesh_selections`

Same per-entry payload as `/physical_groups` / `/labels` (`dim`,
`tag`, `name`, `node_ids`, `node_coords`, optional `element_ids`),
but **the layout remains flat** in schema 2.10 — entries live
directly under `/mesh_selections/{name}/`, not under `node_side/` /
`element_side/` sub-trees. Mesh selections are a single broker store
(not partitioned by side), so the flat write/read is unambiguous;
B2's side-split did not extend here. The H5Model reader auto-detects
layout by sub-tree presence and falls back to the flat walk for this
section.

Sourced from `fem.mesh_selection` (a
:class:`apeGmsh.mesh.MeshSelectionSet.MeshSelectionStore` captured
at ``get_fem_data()`` time when ``g.mesh_selection`` has entries).

```
/mesh_selections/base/
├── attrs: dim=0, tag=1, name="base"
├── node_ids          (Np,) int64
└── node_coords       (Np, 3) float64
```

Schema 2.4.0 addition (Phase 8.7 commit 2).  Omitted entirely when
the broker has no selection store or the store is empty.  Pre-2.4.0
readers ignore the group and lose only the `selection=` selector's
round-trip convenience — live mesh_viewer sessions still consult
the live ``fem.mesh_selection`` directly.

## `/constraints/{kind}`

One dataset per constraint kind (`equal_dof`, `rigid_beam`,
`rigid_diaphragm`, `tie`, `mortar`, `node_to_surface`, …).  Every
dataset uses the symmetric outer compound (see below); the inner
`payload` dtype is per-record-type:

| Kind family | Payload fields |
|---|---|
| NodePair (`equal_dof`, `rigid_beam`, `rigid_rod`, `penalty`) | `master_node`, `slave_node`, `dofs` (vlen-int), `offset` (3,)f64, `penalty_stiffness` |
| NodeGroup (`rigid_diaphragm`, `rigid_body`, `kinematic_coupling`) | `master_node`, `slave_nodes` (vlen-int), `dofs` (vlen-int), `offsets` (vlen-f64, flat `3*n_slaves`), `plane_normal` (3,)f64 |
| Interpolation (`tie`, `distributing`, `embedded`) | `slave_node`, `master_nodes` (vlen-int), `weights` (vlen-f64), `dofs` (vlen-int), `projected_point` (3,)f64, `parametric_coords` (2,)f64 |
| SurfaceCoupling (`tied_contact`, `mortar`) | `master_nodes`/`slave_nodes`/`dofs` (vlen-int), `mortar_operator_shape` (2,)i64, `mortar_operator` (vlen-f64, row-major) |
| NodeToSurface (`node_to_surface`, `node_to_surface_spring`) | `master_node`, `slave_nodes`/`phantom_nodes` (vlen-int), `phantom_coords` (vlen-f64, flat `3*n`), `dofs` (vlen-int) |

Per-record-type payload dtypes are defined in
[`mesh/_record_h5.py`](../src/apeGmsh/mesh/_record_h5.py); the writer in
[`mesh/_femdata_h5_io.py`](../src/apeGmsh/mesh/_femdata_h5_io.py) bins
records by `kind` and dispatches to the right dtype based on the
record class.

## `/loads/{kind}/{pattern}`

Per-pattern, per-kind datasets sharing the symmetric outer compound.

* `/loads/nodal/{pattern}` — `NodalLoadRecord` rows.  Payload:
  `node_id`, `force_xyz` (3,)f64, `moment_xyz` (3,)f64, `name`
  (utf-8, 2.5.0), `basis` (utf-8, 2.28.0: `lagrange` / `bernstein`,
  `""` = basis-insensitive), `source` (utf-8, 2.35.0: the definition
  kind the record was reduced from — `gravity`, `body`, `line`,
  `surface`, `point`, `point_closest`, `face_load`; `""` = unknown).
  Absent force / moment components NaN-filled; the string columns are
  presence-probed, so an older file decodes them as `None`.
* `/loads/element/{pattern}` — `ElementLoadRecord` rows.  Payload:
  `element_id`, `load_type` (utf-8), `params_json` (utf-8 JSON
  blob — element-load `*args` shape is too freeform for a fixed
  typed compound).
* `/loads/sp/{pattern}` — `SPRecord` rows (single-point constraints),
  one dataset per case name (schema 2.26.1, mirroring
  `/loads/nodal/{pattern}`). Payload: `node_id`, `dof`, `value`,
  `is_homogeneous` (int 0/1). Files written before 2.26.1 flattened
  every SP record into a single `/loads/sp/default` dataset regardless
  of `g.displacements.case(...)`; the reader has always iterated the
  group's keys as pattern names, so both layouts read back correctly.

`{pattern}` is the broker pattern name (e.g. `gravity`, `quake_x`)
or `default` for records that didn't carry one.

## `/masses`

Single symmetric-compound dataset (no per-pattern partitioning —
masses are model-time, not load-time).  Payload: `node_id`, `mass`
(6,)f64 = `(mx, my, mz, Ixx, Iyy, Izz)`.

## Symmetric compound contract

Every record-set dataset (`/constraints/{kind}`,
`/loads/{kind}/{pattern}`, `/masses`, `/opensees/bcs/fix`,
`/opensees/bcs/mass`, `/opensees/patterns/{name}/loads`) uses the
same outer 4-field compound so a viewer can dispatch with one
reader and per-kind decoders:

| Field | Type | Meaning |
|---|---|---|
| `target_kind` | vlen utf-8 | `"node"` / `"element"` / `"pg"` |
| `target` | vlen utf-8 | tag (str) or PG name |
| `payload_kind` | vlen utf-8 | record subtype (e.g. `"rigid_beam"`) |
| `payload` | compound | per-kind nested compound |

The `payload` dtype varies by record kind; the outer three fields
are uniform.  Readers dispatch on `payload_kind` and decode
`payload` with the matching per-kind dtype.

Helpers in [`mesh/_record_h5.py`](../src/apeGmsh/mesh/_record_h5.py):

* `make_record_dtype(payload_dtype)` returns the outer compound.
* Per-record-type factories (`node_pair_payload_dtype`,
  `nodal_load_payload_dtype`, `mass_payload_dtype`, …) return the
  inner payload dtypes.

## `/opensees/materials`

```
/opensees/materials/
├── /uniaxial/
│   ├── /Steel02_1/                   group, named {type}_{tag}
│   │   attrs: type="Steel02", tag=1,
│   │          params=[4.2e8, 2.0e11, 0.01, 20.0, 0.925, 0.15]
│   └── /Concrete02_2/
│       attrs: type="Concrete02", tag=2,
│              params=[-3e7, -2e-3, -2.5e7, -6e-3, 0.1, 2.5e6, 2e8]
└── /nd/
    └── /ElasticIsotropic_3/
        attrs: type="ElasticIsotropic", tag=3, params=[3e10, 0.2, 2400.0]
```

Each material is a **group with no datasets, only attributes**. The
constitutive parameters are stored **positionally**, in OpenSees
argument order after the tag, never by name:

* `params` — attribute, float64, shape `(n_args,)` (an empty float64
  array if the command has no arguments). A slot that holds a string
  token (a flag such as `-GJ`) is `NaN`.
* `params_str` — attribute, vlen UTF-8 string array, same shape, empty
  string where the slot is numeric. Written **only** when at least one
  slot is a string; pure-numeric parameter lists have no `params_str`.
* `type` (OpenSees token) and `tag` (int64) are the other attributes.

Slot `i` is read from whichever of `params[i]` / `params_str[i]` is not
the sentinel. Writer: `_write_param_array` in
`opensees/emitter/h5.py`. The meaning of each slot is the OpenSees
manual's argument order for `type`; the file carries no parameter
names. Storing parameters by name is a pending requirement of chain K
(K0-8, ratified on #1283); until it ships, a reader must carry its own
per-type name table.

Optional: a `/comments` attribute (string) for user-supplied notes.

## `/opensees/sections`

Sections that aggregate (Fiber, LayeredShell) carry compound datasets
for their components. Sections that don't (ElasticMembranePlateSection)
are attribute-only, like materials. Every section group is named
`{type}_{tag}` and carries `type`, `tag`, and the positional `params` /
`params_str` attribute pair described under `/opensees/materials`
(float64 `(n_args,)`, `NaN` + `params_str` for flag tokens).

### Fiber section

```
/opensees/sections/Cols/
├── attrs: type="Fiber", tag=1,
│         params=[nan, 1.0e9], params_str=["-GJ", ""]
│                                ← the `-GJ` flag is a positional slot, not a `GJ` attr
├── /patches             compound dataset, shape (n_patches,)
│     fields: kind (string), material_ref (string),
│             ny (int), nz (int),
│             coords (float[8])    ← (yI, zI, yJ, zJ) padded to 8
├── /fibers              compound dataset, shape (n_fibers,)
│     fields: y (float), z (float), area (float),
│             material_ref (string)
└── /layers              compound dataset, shape (n_layers,)
      fields: kind, material_ref, n_bars (int), area (float),
              line (float[6])      ← (y1, z1, y2, z2) for `straight`, padded with NaN to 6
```

`material_ref` is an HDF5 path string like
`"/opensees/materials/uniaxial/Steel_S420"`.  Readers resolve by
`f[material_ref]`.

### Plate / shell section

```
/opensees/sections/Slab/
├── attrs: type="ElasticMembranePlateSection", tag=2, E=30e9, nu=0.2,
│          h=0.20, rho=2400.0
└── (no sub-groups)
```

### Layered shell

```
/opensees/sections/Composite/
├── attrs: type="LayeredShellFiberSection", tag=3
└── /layers              compound dataset, shape (n_layers,)
      fields: material_ref (string), thickness (float), n_int_pts (int)
```

### Aggregator / Parallel

```
/opensees/sections/Combined/
├── attrs: type="Aggregator", tag=4
└── /components          compound dataset
      fields: section_ref (string), dof_ids (int[ndf])
```

## `/opensees/transforms`

```
/opensees/transforms/PDelta_5/   ← "{type}_{tag}", one group per emitted geomTransf call;
                                   no `params` array (the vecxz is the dataset below)
├── attrs: type="PDelta", tag=5,
│         __deviation__="per-emitted-call grouping"
├── per_element_vecxz       float64 (1, 3) in 3-D, (1, 0) in 2-D
│                            the vecxz of this one geomTransf call
└── per_element_emitted_tag int64 (1,)
                             the call's geomTransf tag (equals attr tag)
```

One group per emitted `geomTransf` call, not per element and not per
user-declared transform (`H5Emitter._write_transforms`). Both datasets
hold one row:

* **3-D.** `per_element_vecxz` is `(1, 3)`, the `vecxz` written on the
  `geomTransf` line.
* **2-D.** A 2-D `geomTransf` takes no `vecxz`, so the dataset is
  `(1, 0)`: one row with zero columns, not a missing value. The
  column count is the vector's length, and `OpenSeesModel.from_h5`
  infers `ndm=3` from a 3-column row when `/meta/ndm` is absent.
* **Orientation fan-out.** A transform declared with `orientation=`
  emits one `geomTransf` line per distinct per-element vecxz (ADR 0010).
  Each line is its own group. The first reuses the declared transform's
  tag and the rest take fresh tags. No orientation parameters are
  stored, only the resolved vectors.

`__deviation__` (string, always `"per-emitted-call grouping"`) marks
this departure from the original per-element design. The dataset names
keep their `per_element_` prefix from that design, and the attribute
tells a reader that each group is one call.

**Element ↔ transform join.** Row order does not map elements to
transforms. Each beam-column element's geomTransf tag is the
transf-tag slot of its row in `/opensees/element_meta/{type}/args`.
The element vocabulary gives that slot's position per element type and
`ndm`. `fem_eids` maps the row back to the broker element id.
`H5Model.element_local_axes_vecxz()` performs this join and returns
`{fem_element_id: vecxz}` (3-D only, because 2-D groups carry no vector).

**Authoring front doors.** Two surfaces produce this zone (one schema,
one writer in `H5Emitter`):

* `apeSees(fem).h5(path)` — typed-primitive `ops.geomTransf.<Type>(...)` →
  `BuiltModel.emit` → `H5Emitter.geomTransf(...)`.
* `apeGmsh.opensees.ModelData(fem).oriented_elements(pg=, ele_type=,
  vecxz=).write(path)` — declarative side-channel for users who write
  their model in vanilla openseespy without the bridge.  Sees ADR
  [0018](decisions/0018-modeldata-vanilla-opensees-enrichment.md) and
  [modeldata-enrichment-scope.md (deleted in N3, pinned at a9f8b670)](https://github.com/nmorabowen/apeGmsh/blob/a9f8b670df700dd5b1a9c40f5c7d1f25dd1d400b/src/apeGmsh/opensees/architecture/modeldata-enrichment-scope.md).
  Calls `H5Emitter.add_oriented_elements(...)` which appends one
  `_TransformRecord` + per-element `_ElementRecord`s; the on-disk
  layout below is identical (single source of truth, INV-1 / INV-3).

**Consumer-side pointer.** `Results.viewer(model_h5=)` is the unified
consumer for this zone (and for `/opensees/cuts`).  Explicit
`model_h5=` wins; otherwise the post-solve `ResultsViewer`
auto-resolves `results._path` itself when it was opened from disk via
`Results.from_native` AND the file probes positive for both
`/opensees/transforms` and `/opensees/element_meta`
(`viewers/data/_h5_probe.py::has_opensees_orientation`).
`ViewerData.from_h5` then carries per-element `vecxz` into the scene
via `view.elements.vecxz_for(eid)`.  Producer-agnostic — INV-16's
byte-equivalent output means the consumer needs zero `ModelData`
awareness.  Recorder / MPCO files lack `/opensees/`, so the probe
naturally excludes them and any non-default layout still needs an
explicit `model_h5=` (forwarded into the subprocess on
`blocking=False` via `--model-h5`).

## `/opensees/beam_integration`

One group per `beamIntegration` call.  Keyed by `{type}_{tag}`
(e.g. `/opensees/beam_integration/Lobatto_1`).

```
/opensees/beam_integration/Lobatto_1/
└── attrs: type="Lobatto", tag=1,
          params=[1.0, 5.0]            ← float64 (n_args,): sec_tag, n_ip
          (params_str only if a slot is a flag token)
```

Force / disp-based beam-column elements reference the integration
rule by tag through their positional args; the rule's section
reference is itself an OpenSees section tag inside `params`.

## `/opensees/element_meta`

OpenSees-specific element metadata, keyed by OpenSees type token
(`forceBeamColumn`, `FourNodeTetrahedron`, `Truss`, …) — a parallel
index to the broker's `/elements/{gmsh_alias}` keyed by GMSH alias.

```
/opensees/element_meta/forceBeamColumn/
├── attrs: type="forceBeamColumn"
├── ids               (N,) int64                  — OpenSees element tags
├── fem_eids          (N,) int64                  — FEM element ids (Phase 8.6;
│                                                    -1 sentinel for bridge-minted
│                                                    rows, see below)
├── args              (N, max_tail) float64       — parameter tail (NaN at string slots)
├── args_str          (N, max_tail) vlen-utf-8    — string tokens (present only
│                                                    when any slot is a string)
└── inline_connectivity (N,) vlen-int64           — node-pair endpoint tags
                                                    (schema 2.17.0; present only
                                                    when a row has fem_eid < 0)
```

`args` and `args_str` encode the element's positional `*args` list
*after dropping the connectivity prefix* (the drop is **per record** —
`args[len(connectivity):]` — so a node-pair row whose connectivity length
differs from its PG siblings still slices its own tail).  A
vocabulary-aware reader recovers cross-references (`transf_ref`,
`section_ref`, `integration_ref`, …) by indexing into the element type's
known signature.

`inline_connectivity` (schema 2.17.0, ADR 0049) carries the endpoint node
tags of every **bridge-minted** row whose args start with a node pair:
node-pair elements (`ops.element.ZeroLength(nodes=…)` and the rest of the
zeroLength family wired to a `g.decouple_node` ground), interface springs,
and auto-emitted rebar bars (see "Bridge-minted rows" below).  Such a row
has `fem_eid = -1` and **no row to join on** in the neutral `/elements`
zone: a node-pair element has no gmsh cell at all, and a rebar bar's line
cell is absent from `/elements` whenever the extraction dropped dim-1
cells.  Its connectivity therefore cannot be sourced from there on
re-emit, so it is stored inline here, one ragged row per element (empty
for ordinary PG-fanned rows whose connectivity lives in `/elements`).
The dataset is written **only** when a type group has at least one such
row, so PG-only models are byte-identical and their `model_hash` is
unperturbed; when present it folds into `model_hash` (connectivity is
model-defining).

Phase 8.5 split element storage across two zones (master plan §3):
broker owns geometry (`/elements/{gmsh_alias}` with ids +
connectivity); bridge owns OpenSees-specific args
(`/opensees/element_meta/{type_token}`).  The two are linked by
element tag — both groups' `ids` datasets contain the same tags.

Phase 8.6 added the `fem_eids` parallel array: the i-th entry is
the FEM element id (`fem.elements.ids[i_fem]`) that the bridge's
fan-out used to allocate the i-th OpenSees tag.  Together with the
broker's `/elements/{gmsh_alias}/ids` this gives consumers a
two-way mapping between FEM and OpenSees element identifiers — the
"tag_map" the master plan placed under `/opensees/tag_map/`,
embedded here next to the per-type metadata it concerns rather than
duplicating the type-keying.  Records emitted outside a bridge
fan-out (test scenarios that drive `.element(...)` directly) carry
the sentinel `-1`.

### Bridge-minted rows (`fem_eids = -1`)

The bridge also mints elements that no `ops.element.X(pg=...)` fan-out
produced.  Each such row carries `fem_eids = -1`
(`MISSING_FEM_ELEMENT_ID`), never the id of a neighbouring mesh row
([ADR 0049](decisions/0049-decoupled-nodes.md) convention,
applied to every mint by the ADR 0093 S10 fix, 75614e08):

| Minted row | Type group | `inline_connectivity` row |
|---|---|---|
| node-pair element (`ops.element.ZeroLength(nodes=...)`, …) | its own type | the endpoint pair |
| interface spring (`g.constraints.interface`, ADR 0093) | `zeroLength` | the endpoint pair |
| auto-emitted rebar bar (`g.rebar.place(emit_elements=True)`, ADR 0067 P5.2) | `CorotTruss` | the bar cell's `(i, j)` pair |
| coupling / rigid-body element | its own type | empty (code-derived: `H5Emitter.element` clears the node channel, and these sites do not set it; not yet observed in a written file) |

The first three rows are observed in written files
(`tests/opensees/unit/test_node_pair_zerolength.py`,
`tests/opensees/integration/test_interface_emit_e2e.py`,
`tests/rebar/test_rebar_h5_join.py`).  A
`LadrunoEmbeddedRebar` tie (`g.reinforce`, `coupling="embedded"`) and a
`LadrunoEmbeddedNode` tie (`g.embed`) are **not** `element_meta` rows:
the H5 emitter's `embedded_rebar` and `embedded_node` write nothing under
`/opensees`, and the ties live in the neutral `/reinforce_ties` and
`/embed_ties` groups.

The `-1` is load-bearing: readers restore a row's connectivity from
`inline_connectivity` only when `fem_eid < 0`, because a minted row's
nodes need not be a cell in `/elements`.  A dim-3 extraction
(`get_fem_data(dim=3)`) drops a rebar bar's line cells from `/elements`,
so a real gmsh id there would dangle.  `-1` therefore means "no
`/elements` row to join on", not "no geometry".

**Joining a rebar bar to its `CorotTruss` rows.**  Use the node pair, not
`fem_eids`:

1. The rebar rows are the `CorotTruss` rows with `fem_eids == -1`, and
   they form **one contiguous block**.  A user
   `ops.element.CorotTruss(pg=...)` writes into the same type group with
   real `fem_eids` and empty `inline_connectivity` rows (it is emitted in
   the element pass, before the rebar pass, so its rows precede the
   block).  `/rebar_elements/elements` holds one record per bar; the
   fields are under the symmetric compound's `payload` (table below).
   Inside the block, the rows follow the records in order, record by
   record and pair by pair, and each row's `inline_connectivity` equals
   the record's pair.
2. When the extraction kept the dim-1 cells, every cell of the bar's
   physical group (`/physical_groups/element_side/<pg>/element_ids`, cells
   in `/elements/<line alias>`) matches exactly one `CorotTruss` row by its
   first two (corner) nodes, in gmsh node order.
3. `args[:, 1]` is the uniaxial material tag.  `/opensees/names` maps
   `(kind="uniaxialMaterial", tag)` back to the record's `material` name.

`tests/rebar/test_rebar_h5_join.py` pins this join with raw h5py reads,
including a model that mixes a user `CorotTruss` PG with rebar bars.

**`/rebar_elements/elements`** (neutral schema 2.16.0, ADR 0067 P5.2 /
B1a.2) is one [symmetric compound](#symmetric-compound-contract) dataset,
written only when the model has auto-emitted bars.  The outer fields are
`target_kind = "pg"`, `target = <bar pg>` and
`payload_kind = "rebar_element"`; the `payload` fields
(`rebar_element_payload_dtype` in
[`mesh/_record_h5.py`](../src/apeGmsh/mesh/_record_h5.py)) are:

| `payload` field | Type | Meaning |
|---|---|---|
| `pg` | vlen utf-8 | the bar's physical-group label |
| `element` | vlen utf-8 | `"truss"` (emitted as `CorotTruss`) or `"beam"` |
| `material` | vlen utf-8 | uniaxial-material **name** |
| `area` | float64 | bar area `π·d_b²/4` |
| `role` | vlen utf-8 | bar role (`"longitudinal"`, `"tie"`, …), diagnostics only |
| `connectivity` | vlen int64 | the bar's line cells as flat `(i, j)` pairs, `2·n_cells` long |
| `n_cells` | int64 | `len(connectivity) // 2`, for validation |

## `/opensees/time_series`

```
/opensees/time_series/elcentro/
├── attrs: type="Path", factor=9.81, dt=0.01,
│         file_path="elcentro.txt"        ← if loaded from file
├── time              float dataset (n_steps,)
└── values            float dataset (n_steps,)
```

For algorithmic series (`Linear`, `Constant`, `Trig`, etc.), `time`
and `values` are sampled at a configurable resolution (default: 200
points across the natural domain) so the viewer can plot them
without re-implementing the algorithm.

For loading protocols (`ASCE41Protocol`, `FEMA461Protocol`,
`ATC24Protocol`), the time/values arrays are computed at construction
time and stored verbatim.

Compression: HDF5 gzip level 4 on `time` and `values`. Negligible cost,
significant savings for ground motions.

## `/opensees/patterns`

```
/opensees/patterns/Wind/
├── attrs: type="Plain", tag=1,
│         series_ref="/opensees/time_series/Linear_1"
├── /loads               compound dataset, shape (n_loads,)
│     fields: target_kind (string),    ← "node" | "pg"
│             target (string),         ← node tag (str) or PG name
│             forces (float[ndf])      ← padded to ndf length
├── /sps                 compound dataset
│     fields: target, dof (int), value (float)
└── /element_loads       compound dataset
      fields: target, kind (string),   ← "beamUniform" | "surfacePressure" | …
              params (float[6])         ← padded
```

`/opensees/patterns/Earthquake_X/` for `UniformExcitation`:

```
/opensees/patterns/Earthquake_X/
└── attrs: type="UniformExcitation", tag=2, direction=1,
          series_ref="/opensees/time_series/elcentro"
```

(no contained loads — uniform excitation IS the pattern's payload)

## `/opensees/bcs`

```
/opensees/bcs/fix         compound dataset, shape (n_fix_records,)
   fields: target_kind (string), target (string), dofs (int[ndf])

/opensees/bcs/mass        compound dataset, shape (n_mass_records,)
   fields: target_kind, target, values (float[ndf])

/opensees/bcs@mass_from_model   int8 attr, value 1 (optional, opensees 2.22.0)
```

`@mass_from_model` is written only when the bridge declared
`mass_from_model()` and the snapshot carries masses (ADR 0112 amendment
5, #1304). The archive then holds no `/opensees/bcs/mass` rows for those
masses, because they already persist in the neutral zone's `/masses`.
A reader that rebuilds the model (`OpenSeesModel.build`) must stream
`/masses` onto each node's effective ndf (`/opensees/nodes_ndf`, else
`/meta/ndf`), after the explicit `bcs/mass` rows; skipping the marker
drops every model mass. Any value other than 1 is malformed. The group
`bcs` exists whenever the marker does, even with no `fix` or `mass` rows.

## `/opensees/recorders`

Schema 2.3.0 (Phase 9 commit 6) unifies both recorder declaration
systems — typed primitives (`Node` / `Element` / `MPCO`) and
fan-out records produced by `ops.recorder.declare(...)` — under one
group shape. Every record carries a `kind` attr that distinguishes
the two; declared records additionally carry the original
declaration metadata as attrs.

### Typed primitives (`kind="typed"`)

1:1 with an OpenSees `recorder` command. Same shape as schema 2.2.0
with a new `kind` attr.

```
/opensees/recorders/Node_0/
├── attrs: kind="typed", type="Node", file="disp.out"
└── params           string dataset   ← raw OpenSees args

/opensees/recorders/Element_1/
├── attrs: kind="typed", type="Element", file="forces.out"
└── params           string dataset

/opensees/recorders/mpco_2/
├── attrs: kind="typed", type="mpco", file="model.mpco"
└── params           string dataset
```

### Declared records (`kind="declared"`)

Each fan-out call produced by `ops.recorder.declare(...)` lands as
its own record group, tagged with the original declaration's
metadata. One declaration may produce multiple groups when a
record's components map to multiple OpenSees tokens (e.g. mixing
`displacement` and `velocity` on one nodes record produces two
`recorder Node ...` commands, each as a separate group sharing the
same `declaration_name` / `record_name`).

```
/opensees/recorders/Node_3/
├── attrs:
│   kind="declared", type="Node", file="default__top__disp.out",
│   declaration_name="default", record_name="top",
│   category="nodes",
│   components=["displacement_x","displacement_y","displacement_z"],
│   raw=[],
│   pg=["Top"], label=[], selection=[],
│   ids absent ← name selectors used instead,
│   dt=0.01, n_steps=null,
│   file_root="results/"
└── params           string dataset
```

The declaration-metadata attrs are:

| Attr | Type | Notes |
|---|---|---|
| `declaration_name` | string | identifier of the `ops.recorder.declare(name=...)` call |
| `record_name` | string or null | user-supplied per-record name (auto-generated if absent) |
| `category` | string | `nodes` / `elements` / `line_stations` / `gauss` |
| `components` | string[] | canonical names, already shorthand-expanded against the bridge's `ndm` / `ndf` at declaration time |
| `raw` | string[] | raw OpenSees response tokens (`raw=` escape hatch) |
| `pg` / `label` / `selection` | string[] | named selectors; combined as a union at resolve time |
| `ids` | int[] | explicit IDs; present only when the user passed `ids=` |
| `dt` | float or null | recording cadence (wall-clock) |
| `n_steps` | int or null | recording cadence (step-count); at most one of dt / n_steps is set |
| `file_root` | string | directory prefix; the actual emitted file path is `<file_root>/<declaration_name>__<record_name>__<token>.out` |

### Legacy archives (schema 2.0.0 – 2.2.0)

Pre-2.3.0 archives wrote no `kind` attr. `H5Reader.recorders()`
synthesizes `kind="typed"` for those records so callers can branch
on `r["kind"]` uniformly without a version probe.

## `/opensees/cuts` (optional, v4)

Persisted `SectionCutDef` instances — post-process specs that travel
with the model definition. Present only when the producer was given a
non-empty `cuts=` kwarg (via `apeSees.h5(path, cuts=[...])`) or when
`apeGmsh.cuts.persist_to_h5(path, cuts=[...])` was called against an
existing file. Writer lives in
[`apeGmsh.cuts._h5_io.write_cuts_into`](../src/apeGmsh/cuts/_h5_io.py); reader
in `read_cuts_and_sweeps`. Full design rationale in
[`apeGmsh/cuts/ARCHITECTURE.md`](../src/apeGmsh/cuts/ARCHITECTURE.md) — "## v4
— Cuts persisted in `model.h5`".

One sub-group per cut, named positionally (`cut_0`, `cut_1`, …) in
writer order. Standalone cuts and sweep-member cuts use the same
group shape — sweep cuts live under `/opensees/sweeps/sweep_{i}/cuts/`,
described below.

```
/opensees/cuts/
├── attrs: count=N
└── /cut_0/, /cut_1/, ...
    ├── attrs:
    │   plane_point        (3,)  float64     — point on the cut plane
    │   plane_normal       (3,)  float64     — unit-normalized; reader does not re-normalize
    │   side               utf-8             — "positive" | "negative"
    │   label              utf-8             — display label; "" when has_label=0
    │   has_label          int8              — 0/1; distinguishes None from ""
    │   has_bounding       int8              — 0/1
    ├── element_ids        (Ne,) int64       — OpenSees element tags
    └── bounding_polygon   (Mb, 3) float64   — present iff has_bounding=1
```

`element_ids` carries OpenSees tags, not FEM eids. The kernel-side
consumer (`STKO_to_python`) ingests OpenSees tags directly; the
apeGmsh viewer routes them back through
`/opensees/element_meta/{type}/fem_eids` to reach the FEMData
connectivity. The two `has_*` flags are the workaround for HDF5's
lack of a native `None` — a missing label round-trips as `None`
(not `""` or zero-length array).

Standalone group iteration uses natural-integer sort on the `_N`
suffix (so `cut_10` follows `cut_2`, not `cut_1`); the reader does
not depend on alphabetic ordering.

## `/opensees/sweeps` (optional, v4)

Persisted `SectionSweepDef` instances — ordered sequences of cuts
sharing one element filter, typically used for story-shear-vs-
elevation profiles. Each sweep group owns its cuts: rather than
cross-referencing into `/opensees/cuts/`, the sweep's members live
under its own `cuts/` sub-group. This keeps each sweep
self-contained and avoids dedup logic between the two layouts.

```
/opensees/sweeps/
├── attrs: count=M
└── /sweep_0/, /sweep_1/, ...
    ├── attrs:
    │   count=K
    │   order               vlen utf-8       — ["cut_0", "cut_1", ...] in sweep order
    └── /cuts/
        └── /cut_0/, /cut_1/, ...             — same shape as /opensees/cuts/cut_N
```

The explicit `order` attribute drives reconstruction — HDF5's
alphabetic group iteration would scramble sweeps containing more
than 9 cuts (`cut_10` would land before `cut_2`). Writers
populate `order` in declaration order; readers walk it to rebuild
the `SectionSweepDef.cuts` tuple.

## `/opensees/analysis` (optional)

Present only if the user called the analysis primitives.

```
/opensees/analysis/
└── attrs: handler="Transformation",
          numberer="RCM",
          system="BandGeneral",
          test="NormDispIncr", test_tol=1e-6, test_max_iter=10,
          algorithm="Newton",
          integrator="LoadControl", integrator_increment=0.05,
          analysis="Static",
          analyze_steps=20,
          analyze_dt=null
```

Absent if `ops.h5(path)` was called before any analysis primitive.
The viewer must tolerate this group being missing.

## `/opensees/program` (opensees 2.23.0)

The order in which the bridge drove the H5 emitter (ADR 0114 R2): one
compound row per **run**, a maximal stretch of consecutive Protocol
calls with equal `(method, store, stage, decl)` and consecutive rows.
Every file the bridge writes from 2.23.0 on carries it; it is hashed.

```
/opensees/program          compound (N,)
    first   i4   1-based emit index of the run's first call
    count   i4   calls in the run (>= 1)
    method  i2   index into @methods (the Protocol method name)
    store   i2   index into @stores (the VERBS store template), -1 when
                 the call wrote nothing (a `ledger` verb)
    row     i4   the record's ordinal among the records `method` wrote
                 to `store` in this stage, in emit order; -1 with store -1
    stage   i4   -1 global, else the `stage_NNN` ordinal
    decl    i4   -1 (the `/opensees/decls` row, from K1-6)
  @methods     vlen str (M,)
  @stores      vlen str (S,)   VERBS templates, e.g. `{scope}/bcs/fix`;
                               `stage` resolves `{scope}`
  @emit_count  i4              the runs tile [1, emit_count] with no gap
```

A call a partition bracket replicates (a shared node's `fix` on each
owning rank) repeats the `row` of its first capture, so replication is
a run that points at a row already written. Side channels
(`set_stage_records`, `add_oriented_elements`, `mark_mass_from_model`)
are not Protocol calls and have no emit index. The reader exposes the
runs as `H5Model.program()` and a record's emit index as
`H5Model.emit_index(method, row, stage=-1)` (`OpenSeesModel` delegates);
there is no per-record `emit_index` column (ADR 0114 Q2). An H5 → H5
rewrite (`OpenSeesModel.to_h5`) echoes the table, because its
category-major replay cannot regenerate the order. A rewrite of a file
below 2.23.0 writes **no** program (it never invents an order), and
`emit_index` on such a file raises `ProgramAbsentError`. When the
rewrite does not carry a store an echoed run names (today
`/opensees/regions`), that run keeps its emit indices with `store` and
`row` `-1`, `@stores` drops the name, and the rewrite warns.

## `/opensees/commands` (optional, opensees 2.23.0)

The generic store for a call that has no typed one (ADR 0114 R3a): the
global `rayleigh`, `eigen` and `modal_damping` (ADR 0053 D1/D4) and a
stage's `s.profile` bracket (`profiler`). Written only when such a call
exists; hashed.

```
/opensees/commands/
    method       vlen str (N,)    the Protocol method
    token        vlen str (N,)    the OpenSees verb of a `command` row, else ""
    stage        i4 (N,)          -1 global, else the `stage_NNN` ordinal
    arg_offsets  i4 (N+1,)        row i owns args[arg_offsets[i]:arg_offsets[i+1]]
    args         f8 (A,)          numeric argument, NaN for a string
    args_str     vlen str (A,)    string argument, "" for a number
    arg_kinds    i1 (A,)          0 int, 1 float, 2 str
    arg_names    vlen str (A,)    "" positional, else the keyword
```

Replay calls `method(*positional, **keywords)` with each argument back
at its Python type, so the line is byte-identical. Global rows replay
at the bridge's slot (after the masses, before the patterns); a stage's
`profiler` rows bracket that stage's `analyze` on the side
`/opensees/program` records. A row whose method has no replay slot
raises.

## `/opensees` attributes: the solve stamp (optional, opensees 2.24.0)

What the archive says about the solve it was emitted for (ADR 0114 D6,
R4), as three attributes on the `/opensees` group itself:

```
/opensees
  @will_solve       i1              1 when the model carries a solve
                                    (`staged or any(Analysis)` at emit), else 0
  @solve_refusals   vlen str (R,)   ids of the solve-time gates that refused
                                    at emit, in order; empty when none did
  @requires         vlen str (Q,)   sorted union of the archived verbs'
                                    `VERBS.requires` tokens (`fork`: the
                                    Ladruno build); empty when none
```

`BuiltModel.emit` hands the writer `will_solve` and the refusal ids
through `H5Emitter.set_solve_stamp` on every `apeSees.h5` emit; the
writer derives `@requires` from the methods its `/opensees/program`
holds (ledger rows included: the verb was emitted even if it stored
nothing) and stamps the three together. The refusal ids are the
solve-time gates of `BuiltModel.emit` with the `validate_` prefix
dropped (`ladruno_up_solver`, `serial_mumps`, `up_pressure_datum`),
probed as a solve would run them, under the archive's own partition
facts. `H5Model.solve_stamp()` returns a `SolveStamp`
(`emitter/caps.py`), or `None` for a file that lacks `@will_solve`
(every file below 2.24.0). Absent is "unknown", never a default. A
`@will_solve` without its two companions, one that is not the integer 0
or 1, or a token array that is not sorted and unique raises
`MalformedH5Error`. `OpenSeesModel.build('tcl' | 'py' | 'live')` fails
closed on a non-empty `@solve_refusals`; `build('live')` refuses on a
backend that lacks a token in `@requires` (`fork` needs a Ladruno
build; an unknown token always refuses); `to_h5` echoes the stamp, and
`@requires` regenerates from the echoed program. A fork *element or
material type* rides the generic `element` / `nDMaterial` verb, whose
row requires nothing, so it does not reach `@requires` until K4 moves
typed fork verbs onto the command channel. Attributes of `/opensees`
fold into `model_hash`.

## `/meta/session_id` and the geometry sibling

ADR 0112 D1 makes geometry an artifact of its own, and the V0
ratification (#1283, Q1) puts the `/geometry` zone in a **sibling file
only**, never inside `model.h5`. For a model file `<stem>.h5` the
sibling is `<stem>.geometry.h5`, in the same directory.

`FEMData.session_id` is a uuid4. Every writer of the neutral zone
stamps it as `/meta/session_id`: `fem.to_h5`, `apeSees(fem).h5` (through
`_compose_model_h5`), the replay writers that rebuild a file from its own
FEMData, and the `/model/meta` of a composed `results.h5`.

**Who mints it.** The id belongs to the session, and derived snapshots
inherit it:

* **Inherited, never re-minted:** the record transforms (`with_constraint`,
  `with_load`, `with_mass`) and every copy or replace of a snapshot.
  `FEMData.from_h5` restores the stored id, so a replay writer
  (`OpenSeesModel.to_h5`, `ModelData.write`) re-stamps the source's id.
* **Freshly minted:** a newly constructed snapshot. That covers `compose()`,
  whose result merges several sources and so belongs to none of their
  sessions; `from_mpco` and `.ladruno` imports, which have no session; a
  pre-#1304 pickle; and `from_h5` of a file without the attr.
* **Owned by the session (V2b, #1305):** `apeGmsh.begin()` mints one
  uuid4 (`_SessionBase._session_id`), `FEMData.from_gmsh(session=...)`
  stamps it on every snapshot the session extracts, and the session
  stamps its geometry sibling with the id of the snapshot `model.h5`
  carries. Two `get_fem_data()` calls in one session share the id.

Pairing by equality therefore holds only for the artifacts of one
session: the snapshot that wrote `model.h5`, the snapshots derived from
it, and (from V2b) the geometry that session captured.

**Pairing rule.** A reader pairs `<stem>.h5` with `<stem>.geometry.h5`
only when both carry a `/meta/session_id` and the two strings are
equal. A sibling whose `session_id` differs, or is missing, is stale or
foreign: a reader must say so and must not draw it as this model's
geometry. A model file without `session_id` (written before #1304)
pairs with nothing.

**Hash exclusion.** `session_id` is identity metadata, not model
content. `snapshot_id`, `fem_hash` and `model_hash` are computed from
allowlists (`mesh/_femdata_hash.py::compute_snapshot_id`,
`opensees/_internal/lineage.py::compute_model_hash`) that never read
it, so two writes of one model under different sessions hash the same.
`tests/test_v2_zone_keys.py` holds this invariant, together with the
same invariant for adding or deleting `/geometry` and `/provenance`.

## Integer policy for the ADR 0112 zones

`/geometry` and `/provenance` store **no int64**, so a browser reader
(h5wasm) never receives BigInt for them:

* ids, tags, row indices, offsets and counts are `int32`;
* small enumerations and flags are `int8`;
* coordinates and lengths are `float64`;
* strings are variable-length UTF-8.

A writer that meets a value outside `int32` (a tag, an offset or a count
of 2³¹ or more) **refuses loudly**: it raises before writing anything
and never truncates or wraps (V0 amendment 2). 2³¹ is the documented
limit of these zones.

## `/geometry` (sibling `<stem>.geometry.h5`)

Version key: `/meta/geometry_schema_version` (current `1.0.0`). The
sibling file also carries `/meta/session_id` (see the pairing rule
above). Layout from V0 decision 5: concatenated arrays plus offsets per
dimension (CSR), never one group per entity, because a browser would
open about 10⁴ groups at STKO scale. There is no BRep, one level of
detail, and no normals or UVs.

```
/geometry   @source {mesh|temp_mesh}  @gmsh_version str  @curve_samples i4
            @lod_size f8  @bbox f8[6]  @status {ok|partial}
  /entities      dim i1 (K,) · tag i4 (K,) · bbox f8 (K,6) · ok i1 (K,)
  /points        entity i4 (P,) · xyz f8 (P,3)
  /curves        entity i4 (C,) · vertex_offsets i4 (C+1,) · vertices f8 (Vc,3)
  /surfaces      entity i4 (S,) · vertex_offsets i4 (S+1,) · vertices f8 (Vs,3)
                 triangle_offsets i4 (S+1,) · triangles i4 (T,3)
  /volumes       entity i4 (V,) · face_offsets i4 (V+1,) · faces i4 (F,)
  /memberships   dim i1 · tag i4 · kind {label|physical_group} · name str · pg i4
```

* `entities` lists every model entity once. `ok` is 0 for an entity
  whose tessellation failed; the file then carries `@status = partial`.
* In `points`, `curves`, `surfaces` and `volumes`, `entity` is a row of
  `entities`. Row `i`'s data is `[offsets[i], offsets[i+1])` of the
  concatenated array. `surfaces/triangles` index that surface's own
  slice of `vertices` (0-based, local to the surface).
* `volumes/faces` are rows of `surfaces/entity`, so a volume is drawn
  from its bounding surfaces.
* `memberships` is a flat table, one row per (entity, label or physical
  group). `pg` is the physical-group tag for `kind = physical_group`
  and -1 for a label.
* `@source` is `mesh` when surfaces come from the session's real mesh
  (written at the exit of `g.mesh.generation.generate()`), and
  `temp_mesh` when the session never meshed and a 2-D surface-only
  mesh at `@lod_size` was generated at `end()` and then cleared (V0
  decision 6 and amendment 3). Curves are sampled parametrically with
  `@curve_samples` points (32 by default) at `@lod_size = 0.03 · bbox
  diagonal` (Q9).
* Sessions with no kernel (`from_h5`, STKO import, a hand-built FEM)
  write no geometry file. A failed capture warns once and never raises.

## `/provenance`

Version key: `/meta/provenance_schema_version` (current `1.1.0`, floor
`1.0.0`; `1.1.0` added the `records/origin` column, #1378, and a
`1.0.0` file reads with every record as `origin = "user"`).
Layout from V0 decisions 11 to 14: the source location of every user
declaration, deduplicated into three tables. Each table is a group of
equal-length column datasets.

```
/provenance   @base_dir str
  /files      path str · sha256 str · kind str
  /sites      file i4 · line i4 · function str
  /records    path str (unique) · site i4 · script i4 · seq i4 · origin str
```

* **Key.** `records/path` is the **declaration path**
  `<zone>/<family>/<name|#k>`: the zone and family the declaration
  belongs to, then the user's name for it. An unnamed declaration gets
  `#k`, its 1-based order among the unnamed declarations of that family
  in this run, so `#k` is stable within one run only. The key is never
  an HDF5 group name or an OpenSees tag (Q3).
* **Families.** The session writes these paths (V2c,
  `src/apeGmsh/_internal/provenance.py`):
  * `neutral/labels/<name>` from `g.labels.add`, `g.labels.rename` (the
    new name) and `g.parts.add` (the instance label);
  * `neutral/physical_groups/<name|#k>` from `g.physical.add` and its
    shorthands, and from `g.labels.promote_to_physical`;
  * `geometry/<kind>/<label|#k>` from every geometry registration, where
    `<kind>` is the registering primitive (`box`, `line`, `polyline`, ...);
  * `neutral/<family>/<name|#k>` from the declaration verbs, where
    `<family>` is one of `constraints`, `bcs`, `contacts`,
    `contact_planes`, `interfaces`, `decoupled_nodes`, `displacements`,
    `embeds`, `loads`, `masses`, `rebar`, `rebar_members` or
    `reinforcements` (`core/_declarations.py::_PROVENANCE_FAMILY`).

  A named path keeps its first record. A later call that merges into a
  label or appends to a physical group adds no record.

  The bridge writes (V2d, `apeSees._register`):
  * `opensees/<kind>/<name|#k>` for every primitive the user declared,
    where `<kind>` is the OpenSees command (`element`,
    `uniaxialMaterial`, `nDMaterial`, `section`, `geomTransf`,
    `beamIntegration`, `timeSeries`, `pattern`, `damping`, `recorder`,
    `constraints`, `numberer`, `system`, `test`, `algorithm`,
    `integrator`, `analysis`) and `<name>` the `name=` alias;
  * `opensees/<kind>/<verb>:<owner>[/<role>]` for an object the bridge
    synthesises inside a verb the user called, with
    `origin = "synthesised"`: `opensees/timeSeries/support:<stage>/hold`
    and `opensees/pattern/support:<stage>` from `s.support`,
    `opensees/timeSeries/imposed_displacement:<name>` and
    `opensees/pattern/imposed_displacement:<name>` from
    `imposed_displacement` (`<name>` is its `name=`, else `#<k>`, the
    call's 1-based ordinal on the bridge; a `name=` starting with `#`
    is refused). The bridge allows a repeated stage name, so a stage
    whose pattern key `support:<stage>` is taken keys its support
    objects `<stage>@<n>` with the smallest `n >= 2` whose key is free
    (stages `s`, `s`, `s@2` give `support:s`, `support:s@2`,
    `support:s@2@2`). These keys never use the family's `#k` counter,
    so the user's first unnamed `Linear()` stays `#1`.

  User names and synthesised keys share one key space per family, and
  no name is restricted. A collision, whichever side comes second (a
  user `Linear(name="imposed_displacement:foo")` after
  `imposed_displacement(name="foo")`, a stage `x` whose HOLD key a user
  `Linear(name="support:x/hold")` took, a user name `#1` followed by an
  unnamed declaration), fails loud **before** the bridge allocates a
  tag: the refused call leaves no primitive, no tag and no record. A
  record is never overwritten and never dropped.

  In a file the bridge writes, the `opensees/` records are the bridge's
  own: a snapshot loaded from an earlier bridge-written file drops that
  file's `opensees/` records before the new ones are appended, so a
  repeated name does not collide and no stale record survives.
* **Files.** `path` is POSIX. It is relative to `@base_dir` when the
  file lies under it, and absolute otherwise (Q8). `sha256` is the hex
  digest of the file when it was captured, so go-to-source can tell
  that the file has been edited since. `sha256` is `""` for a pseudo-file
  (`<string>`, `<stdin>`, a notebook cell, which keep that name as
  `path`) and for a source that could not be read. `kind` is `script`
  for the run's `__main__` file and `module` for any other file.
* **Sites.** `file` is a row of `files`; `line` is 1-based.
* **Records.** `site` is a row of `sites`: the first frame outside
  apeGmsh and the standard library. `script` is a row of `sites`: the
  outermost `__main__` frame of the user code around `site`, which
  differs from `site` when the call came through a user helper. The
  walk for `script` starts at `site`, passes through standard-library
  frames (`contextlib`, `runpy`), and stops at the first apeGmsh or
  site-packages frame. A launcher that runs as `__main__` from
  site-packages (`pytest`, `ipykernel`) therefore never claims it, and
  a call with no user `__main__` frame around it (a test function) gets
  none. A `__main__` frame whose file lies under the apeGmsh tree is
  user code. A stdlib launcher's `__main__` frame (`python -m cProfile`,
  `pdb`, `trace`) is not, so the walk passes through it. A pseudo-file
  `__main__` frame (`<string>`) never overrides a real-file one the
  walk already found, so the `<string>` trampoline of `python -m pdb`
  does not claim it; with no real file around it (`-c`, `<stdin>`, a
  notebook cell) it is the script. Either is -1 when no such frame
  exists.
  `seq` is the declaration's 0-based capture order in the run.
  `origin` (1.1.0) is `user` for a declaration the user made and
  `synthesised` for an object apeGmsh created inside a verb the user
  called (the keys above); a viewer shows synthesised objects by
  default. Absent in a file below `1.1.0`, where every record reads as
  `user`; from `1.1.0` on the column is required and a file without it
  is malformed (`schema_version.py::PROVENANCE_ORIGIN_FROM`).
* There is one record per user call: none per emitted row, none per
  fanned-out element. A call that synthesises deck objects (series,
  patterns) gets one record per synthesised object, each pointing at
  the call's own site.
* Every artifact the session writes carries its own `/provenance`
  (decision 14). The replay writers copy it forward (Q7).
* No hash reads `/provenance` (the same allowlists as above).
* An assembly (ADR 0117 D5) writes `assembly/instances/<label>` from
  `Assembly.instance` and `assembly/ties/<name|#k>` from `Assembly.tie`,
  `Assembly.node` and every coupling verb (`equal_dof`, `rigid_link`,
  `rigid_diaphragm`, `embedded`, `couple`), all `origin = "user"`, at the
  declaring line. They ride the merged
  FEM of `Assembly.bridge`, so `ops.h5` writes them too.

## `/assembly`

Version key: `/meta/assembly_schema_version` (current `1.0.0`, floor
`1.0.0`; [ADR 0117](decisions/0117-assembly-compose-v2.md) D5, its own
key per ADR 0112 D2). Written by `Assembly.h5` only, after `apeSees.h5`
has written the whole `model.h5`
([`assembly/_h5.py`](../src/apeGmsh/assembly/_h5.py)). It re-lists what
was assembled; the flat zones stay authoritative, so no model reader
reads it. Each table is a group of equal-length column datasets.

```
/assembly     @name str
  /instances  label str · source_path str · source_fem_hash str ·
              source_opensees_hash str · translate (n, 3) f8 · rotate (n, 4) f8 ·
              fem_id_base i8 · fem_id_span i8 · partition_rank i8
  /ties       name str · kind str · master str · slave str · params str ·
              n_records i8
```

* **Instances.** `source_path` is the path as declared, in POSIX form;
  `source_fem_hash` and `source_opensees_hash` are the source's
  `fem_hash` and `model_hash` (ADR 0021) when it was bridged. `rotate`
  is `(ax, ay, az, theta)`, all zero for an unrotated instance (a
  declared axis is never zero). An instance declared with `anchor=`
  stores the translate its anchor resolved to; the anchor name is not
  stored (v1's `/composed_from` did not store it either). `fem_id_base` and `fem_id_span` are the
  relocated FEM-id window of the instance's nodes and elements: the
  source's smallest id maps to `fem_id_base`. `partition_rank` is `-1`
  without a rank hint.
* **Ties.** One row per assembly-level tie, coupling (ADR 0117 D3) and
  reference node. `name` is `""` for an unnamed tie or coupling.
  `master` and `slave` are the ports as declared: `{instance}.{pg|label}`
  or, where the kind accepts one, the name of a reference node. `params`
  is canonical JSON (sorted keys, no spaces) holding exactly the keys of
  its kind (`TIE_PARAMS` in `assembly/_h5.py`), plus any of the kind's
  knobs (`TIE_KNOBS`) that is set; a row with other keys, or with a knob
  stored at its default, is refused on write and on read, and a reader
  refuses an unknown kind.
  `n_records` is the number of constraint records the row resolved to,
  at least 1. The kinds of 1.0.0:

  | kind | master / slave | `params` keys | knobs (written only when set) |
  |---|---|---|---|
  | `tie` | port / port | `dofs`, `enforce`, `method`, `tolerance` | `stiffness` (`"auto"`), `stiffness_p`, `rotational` (`false`), `pressure` (`false`), `control`, `outward` |
  | `equal_dof` | port or node / port or node | `dofs`, `tolerance` | |
  | `equal_dof_mixed` | port or node / port or node | `dof_pairs`, `tolerance` | |
  | `rigid_link` | port or node / port or node | `link_type`, `master_point` | |
  | `rigid_diaphragm` | port or node / port or node | `constrained_dofs`, `master_point`, `plane_normal`, `plane_tolerance` | |
  | `rigid_body` | port or node / port or node | `as_element`, `master_point`, `mass`, `omega` | |
  | `embedded` | host port / embedded port | `stiffness`, `tolerance` | |
  | `kinematic_coupling` (RBE2) | reference node / port | `dofs` | `k`, `k_alpha`, `kr`, `enforce` (`"penalty"`), `al_update` |
  | `distributing_coupling` (RBE3) | reference node / port | `weighting` | `k`, `k_alpha`, `kr`, `enforce` (`"penalty"`) |
  | `node` | `""` / `""` | `coords` | |

  A knob's default is `null` unless the table names one. The knobs and
  the kinds `equal_dof_mixed` and `rigid_body` came with AS5-b (#1588);
  the version stays 1.0.0 because the zone is unreleased, and a row that
  sets no knob is the row written before them. `dof_pairs` is a list of
  `[retained, constrained]` DOF pairs. A tie's `control` is a
  `CouplingControl` as an object holding every one of its fields. RBE2 /
  RBE3 `k="auto"` and `k_alpha` need a host element, which an assembly
  coupling does not take, so the verbs and the reader refuse them.

* **Reference nodes.** A `node` row is an assembly-owned reference node
  (`Assembly.node`): `name` is the node's (non-empty, no `.`), both ports
  are `""`, `params` is `{"coords":[x,y,z]}` and `n_records` is 1. Node
  rows are written before every other row, so a reader knows each node
  before a coupling names it; in the flat zones the `k`-th node is the
  decoupled node with FEM id `k`, labelled with its name.
* **Empty and rewritten.** An assembly with no ties writes `/ties` with
  zero-length columns, never a missing group. Writing into a file that
  already has the zone replaces it. Rows are validated before the file
  is opened, and a failure part-way removes the partial group and key.
* `Assembly.from_h5` reads only this zone and never opens an instance
  file. A file without `/assembly` opens exactly as before in every
  reader; `Assembly.from_h5` refuses it.
* `/composed_from` is still written beside it. No hash reads
  `/assembly`: `fem_hash` reads the neutral zone and `model_hash`
  reads `/opensees`.

## Cross-references

Every reference uses an HDF5 path string. Examples:

| Reference attribute | Example value |
|---|---|
| `material_ref` | `/opensees/materials/uniaxial/Steel_S420` |
| `section_ref` | `/opensees/sections/Cols` |
| `transf_ref` | `/opensees/transforms/Cols` |
| `series_ref` | `/opensees/time_series/elcentro` |

Readers MUST resolve via `h5py.File["{ref}"]` and validate the
returned group's `type` attribute matches expectations.

## Compound dataset conventions

For variable-length string fields (`material_ref`, `target`, `kind`),
use HDF5 variable-length string type
(`h5py.string_dtype(encoding="utf-8")`).

For padded float arrays (`forces`, `params`), pad with `nan` to a
fixed length (e.g. `ndf` for forces, 6 for element-load params). Use
`np.dtype([...])` compound types.

## Versioning

Versioning is **per-zone**. Each producer owns an independent semver
axis stamped under its own `/meta` key (ADR 0023). They do **not**
share a number: a file can be `neutral=2.10.0` and `opensees=2.12.0`
at the same time. The legacy single `/meta/schema_version` envelope
predates the split and is **non-authoritative** (back-compat only —
see "Legacy envelope" below).

The central logic lives in
[`opensees/_internal/schema_version.py`](../src/apeGmsh/opensees/_internal/schema_version.py).
`reader_version(zone)` sources each zone's current value directly from
the writer's constant, so reader and writer **cannot drift**; readers
call `validate_zone_version(...)` for each zone before reading it.

### Zone registry

| Zone | `/meta` key | Root paths | Writer constant (source of truth) | Current | Floor |
|---|---|---|---|---|---|
| neutral (broker) | `neutral_schema_version` | `/nodes`, `/elements`, `/physical_groups`, `/labels`, `/mesh_selections`, `/partitions`, `/parts`, `/constraints`, `/reinforce_ties`, `/embed_ties`, `/rebar_elements`, `/contacts`, `/contact_planes`, `/interfaces`, `/loads`, `/masses`, `/composed_from` | [`mesh/_femdata_h5_io.py`](../src/apeGmsh/mesh/_femdata_h5_io.py) `NEUTRAL_SCHEMA_VERSION` | **2.35.0** | **2.10.0** |
| opensees (bridge) | `opensees_schema_version` | `/opensees/*` | [`opensees/emitter/h5.py`](../src/apeGmsh/opensees/emitter/h5.py) `SCHEMA_VERSION` | **2.24.0** | **2.12.0** |
| results | `results_schema_version` | `/stages/*` (composed `results.h5`, at file root) | [`results/schema/_versions.py`](../src/apeGmsh/results/schema/_versions.py) `RESULTS_SCHEMA_VERSION` | **1.1.0** | **1.0.0** |
| cuts (sub-zone of opensees) | — (no own key; rides the opensees zone) | `/opensees/cuts`, `/opensees/sweeps` | [`cuts/_h5_io.py`](../src/apeGmsh/cuts/_h5_io.py) `V4_SCHEMA_VERSION` | 2.5.0 | none of its own: it rides the opensees floor |
| geometry (ADR 0112 D2) | `geometry_schema_version` | `/geometry` (sibling `<stem>.geometry.h5` only) | [`opensees/_internal/schema_version.py`](../src/apeGmsh/opensees/_internal/schema_version.py) `GEOMETRY_SCHEMA_VERSION` | **1.0.0** | **1.0.0** |
| provenance (ADR 0112 D3) | `provenance_schema_version` | `/provenance` | [`opensees/_internal/schema_version.py`](../src/apeGmsh/opensees/_internal/schema_version.py) `PROVENANCE_SCHEMA_VERSION` | **1.1.0** | **1.0.0** |
| assembly (ADR 0117 D5) | `assembly_schema_version` | `/assembly` (`Assembly.h5` archives only) | [`opensees/_internal/schema_version.py`](../src/apeGmsh/opensees/_internal/schema_version.py) `ASSEMBLY_SCHEMA_VERSION` | **1.0.0** | **1.0.0** |

> The geometry, provenance and assembly keys never fall back to the legacy
> envelope: they postdate it, so an absent key means the zone was not
> written (`read_zone_version` returns `None`). Their writers import the
> version constants from `schema_version.py` until they have modules of
> their own.

> The registry's *current* values are a snapshot — the writer
> constants above are the authoritative source. The test
> [`tests/opensees/h5/test_h5_schema_compat.py`](../tests/opensees/h5/test_h5_schema_compat.py)
> (`test_reader_version_reflects_writer_constants`) pins reader↔writer
> agreement; [`tests/fixtures/schema.py`](../tests/fixtures/schema.py)
> centralizes the values fixtures stamp.

### The floor rule

Compatibility is a **floor per zone**, not a window
([ADR 0113](decisions/0113-compatibility-is-a-floor-per-zone.md) D1; it
retired ADR 0023's two-version window, which expired files the readers
could still parse). `validate_zone_version` accepts a file iff

```
file.major == reader.major  and  floor.minor <= file.minor <= reader.minor
```

The patch is ignored. A file below the floor refuses as "too old", a
newer minor refuses as "newer than this reader" (ADR 0023 INV-4: Python
readers have no forward tolerance), and another major refuses as a major
mismatch. Every refusal names both ends of the supported range, for
example `supports 2.10.x–2.34.x`.

- **Where the floor lives.** Each floor is a writer-owned constant beside
  the version it bounds (ADR 0113 D3): `NEUTRAL_SCHEMA_FLOOR`,
  `SCHEMA_FLOOR`, `RESULTS_SCHEMA_FLOOR`, `GEOMETRY_SCHEMA_FLOOR` and
  `PROVENANCE_SCHEMA_FLOOR`. `reader_floor(zone)` reads them, as
  `reader_version(zone)` reads the versions. The **Floor** column of the
  registry above is a snapshot of those constants.
- **Evidence.** A floor stands only where a committed corpus file from
  every minor, floor to current, opens through today's reader
  (`tests/fixtures/schema_corpus/`, built from git's frozen writers by
  `scripts/build_schema_corpus.py`; ADR 0113 D8). The opensees floor is
  2.12.0, not the 2.11.0 first ratified: every 2.11-era writer stamped
  a neutral zone below the neutral floor, so no 2.11 file could open.
- **A floor only rises**, and after the initial evidence gate only with a
  major bump, to `X.0.0`, by ADR. Raising it deletes the shims beneath it.
- **Files below a floor** are refused, except an embedded zone of a
  `results.h5`: a results file whose embedded `/model` or `/opensees` is
  below its floor still opens its `/stages`, read-only and flagged (D9).
  An embedded zone newer than the reader still refuses the whole file.
- **The app** carries the same floors in `apeGmshViewer/src/reader/read.ts`
  (`ZONE_FLOOR`); a Python test parses them as text. The app opens a
  same-major newer file with one banner instead of refusing (D7).

### Bump rules (per zone)

- **Major** bump → breaking change, and the only place a restructure
  (renamed group, changed dtype, layout split) may go. Readers refuse
  outright; the zone's floor is reset to `X.0.0` by ADR, and the
  migrator (ADR 0113 D6) ships with the first major bump of any zone.
- **Minor** bump → additive (new group, new attribute, new column), or a
  semantic change that ships a reader shim (see the ledger below). The
  stamp still moves for an additive change: an older reader would
  otherwise open the file and drop a deck-affecting column unseen (D5).
  Readers at or above the new minor open every minor from the floor,
  probing for what an older file lacks.
- **Patch** bump → internal/cosmetic. Readers must not depend.
- **Every minor bump adds one corpus file**: the outgoing minor's, built
  with `python scripts/build_schema_corpus.py --zone <Z> --minor <M>` and
  committed (see the bridge-feature guide's bump checklist).

### Shim ledger

A minor that changes the *meaning* of a field already in the file ships
a reader shim keyed on a named `*_FROM` constant at or above the zone's
floor, with a test on a corpus file below it (ADR 0113 D4, INV 8). This
table is the one list of them.

| Zone | Constant | Below it | At or above it |
|---|---|---|---|
| neutral | `META_NDM_IS_SPATIAL_FROM` = 2.34.0 (`opensees/emitter/h5_reader.py`) | `/meta/ndm` is the mesh dimension; `read_spatial_ndm(meta, f, *, coords)` salvages the spatial `ndm` from the `per_element_vecxz` widths: width 0 (a 2-D transform) resolves 2, width 3 lifts to 3, and conflicting evidence, an unknown width, or a result that would drop a non-zero coordinate column refuses (a pre-2.34 2-D truss with y≠0 and a 2-D surface at z≠0 refuse; #1358) | the attribute is trusted |
| neutral | SP loads before 2.26.1 (no named constant: the data carries it) | every SP record sits in one `default` case; the per-case split is not reconstructed | one group per case under `/loads/sp` |

A shim never forwards an old value under a newer stamp (ADR 0113 INV 9):
`NativeWriter` restamps the embedded `/model/meta` of a results twin,
so it forwards `ndm` through `read_spatial_ndm`, never the raw attribute.

### Legacy envelope (`/meta/schema_version`)

Pre-Phase-7a files carried a single `schema_version`. Today every
file apeGmsh writes also stamps the per-zone keys (broker-only files
forward-stamp `opensees_schema_version` too), so the envelope is read
**only as a fallback** for genuinely old single-stamp files
(`read_zone_version(..., envelope_fallback=True)`). Per ADR 0023
INV-2, **new code must not branch on the envelope** — it is "whichever
writer wrote last" (the composer and `cuts.persist_to_h5` may each
overwrite it independently), so its value is not a reliable zone
marker. The guarantee that the envelope never becomes load-bearing for
our own output is held by
`test_to_h5_stamps_both_per_zone_keys` (broker-only) and
`test_per_zone_keys_written_on_compose` / `_on_native_results`
(composed).

### Neutral-zone history

The list below is the **neutral-zone** lineage, condensed from the
canonical log — the `NEUTRAL_SCHEMA_VERSION` docstring in
[`mesh/_femdata_h5_io.py`](../src/apeGmsh/mesh/_femdata_h5_io.py), current
through **2.35.0**. The opensees zone's per-version history is
maintained inline in [`opensees/emitter/h5.py`](../src/apeGmsh/opensees/emitter/h5.py)
(`SCHEMA_VERSION` docstring), current through **2.20.0**; its post-2.10
additions are summarized after this list.

History (entries that mention ADR 0023's "two-version reader window" were
written under that rule and stay as the record of what each writer
promised then; ADR 0113 retired the window, and today every minor from the
zone's floor up opens):

- `1.0.0` — Phase 6 initial release.
- `1.1.0` — added `/beam_integration` group + widened fiber-layer
  `line` field from float[4] to float[6].
- `2.0.0` — Phase 8.4: bridge-written groups (materials, sections,
  transforms, beam_integration, time_series, patterns, bcs, recorders,
  analysis) moved under `/opensees/`.  `/meta` and `/elements` stay
  at root.  Breaking — any tool reading pre-8.4 files by absolute
  path needs to update.
- `2.1.0` — Phase 8.5: broker neutral zone added (`/nodes`,
  `/elements/{gmsh_alias}`, `/physical_groups`, `/labels`,
  `/constraints/{kind}`, `/loads/{kind}/{pattern}`, `/masses`).
  OpenSees-specific element metadata moved from the old
  `/elements/{type_token}` shape to `/opensees/element_meta/{type_token}`
  so the broker can own root `/elements`.  Additive — old v2.0.0
  readers tolerate the absence of the new groups.
- `2.2.0` — Phase 8.6: `fem_eids` int64 dataset added under each
  `/opensees/element_meta/{type_token}/` group, parallel to `ids`.
  Carries the FEM element id each OpenSees tag was fanned out from
  (master plan §3 "tag_map", embedded with the per-type metadata
  it concerns instead of duplicated under a standalone
  `/opensees/tag_map/` index).  Sentinel `-1` marks records
  emitted outside a bridge fan-out.  Additive — old v2.1.0 readers
  ignore the new dataset.
- `2.3.0` — Phase 9 commit 6: unified `/opensees/recorders/`
  archive.  Every record group gains a `kind` attr — `"typed"` for
  Node / Element / MPCO primitives (1:1 with an OpenSees `recorder`
  command), `"declared"` for fan-out calls produced by
  `ops.recorder.declare(...)`.  Declared records additionally carry
  the original declaration metadata as attrs: `declaration_name`,
  `record_name`, `category`, `components`, `raw`, `pg`, `label`,
  `selection`, `ids`, `dt`, `n_steps`, `file_root`.  Additive —
  old v2.2.0 readers see `kind="declared"` records as well-formed
  recorder groups (they just ignore the extra attrs).  See
  [phase-9-recorder-unification.md (deleted in N3, pinned at a9f8b670)](https://github.com/nmorabowen/apeGmsh/blob/a9f8b670df700dd5b1a9c40f5c7d1f25dd1d400b/src/apeGmsh/opensees/architecture/phase-9-recorder-unification.md)
  for the multi-commit phase that delivered this.
- `2.4.0` — Phase 8.7 commit 2: `/mesh_selections/` neutral-zone
  group added, mirroring `/physical_groups` / `/labels` shape.
  Carries post-mesh selection sets (``g.mesh_selection`` →
  ``fem.mesh_selection``) so the viewer's `selection=` selector
  round-trips through `model.h5`.  Additive — old v2.3.0 readers
  ignore the new group and lose only the `selection=` round-trip
  convenience (live mesh_viewer sessions still consult the live
  ``fem.mesh_selection`` directly).  See
  [phase-8.7-scope.md (deleted in N3, pinned at a9f8b670)](https://github.com/nmorabowen/apeGmsh/blob/a9f8b670df700dd5b1a9c40f5c7d1f25dd1d400b/src/apeGmsh/opensees/architecture/phase-8.7-scope.md) §1b for the rationale
  and [ADR 0014](decisions/0014-viewer-is-pure-h5-consumer.md) for
  the architectural decision.
- `2.5.0` — apeGmsh.cuts v4: `/opensees/cuts/` and
  `/opensees/sweeps/` groups added carrying `SectionCutDef` /
  `SectionSweepDef` persistence.  See
  `src/apeGmsh/cuts/ARCHITECTURE.md` "## v4 — Cuts persisted in
  `model.h5`" for the on-disk shape.  Writer lives in
  `apeGmsh.cuts._h5_io.write_cuts_into`; reader in
  `read_cuts_and_sweeps`.  Additive — pre-v4 readers (2.4.0 and
  earlier) ignore the new groups; missing groups return empty
  tuples.
- `2.6.0` through `2.9.0` — additive evolution: `/meta/lineage`
  (ADR 0021), `/meta/neutral_schema_version` (ADR 0023),
  `/nodes/ndf` (S1b shell-to-solid), interpolation-payload widening
  (ADR 0035), `/composed_from/` + `tag_span_max` + `module_label`
  parallel datasets (ADR 0038 / compose v1 Phase 3A.1).
- `2.10.0` — **B2 schema bump** (PR #398). `/physical_groups/` and
  `/labels/` split into `node_side/` and `element_side/` sub-trees,
  fixing a snapshot_id drift bug where the flat layout's
  side-classification heuristic produced phantom node-side entries
  on hand-constructed FEMData round-trips. Element-side groups now
  write their own `node_ids` / `node_coords` (the prior writer
  stripped them to `(0,)` on disk — adjacent latent bug). The
  `compute_snapshot_id` hash widened to fold element-side PGs and
  labels (previously element-side PGs were invisible to the hash,
  and labels weren't hashed at all). **Not additive** — 2.9
  files could not be read by the 2.10 reader (under the window of the
  day it slid forward; the floor rule makes 2.10.0 the neutral floor).
  ADR 0021 INV-1 retired; ADR 0023 window semantics reframed.
- `2.11.0` — ADR 0049 (decoupled-node provenance): additive — adds the
  optional `/nodes/provenance` int8 dataset (0=mesh, 1=decoupled),
  distinguishing ordinary mesh nodes from nodes synthesized by
  `g.decouple_node`. Omitted when the broker has no decoupled nodes.
  Per ADR 0023's two-version reader window, readers tolerate 2.10.x
  and 2.11.x.
- `2.12.0` — fork-coupling control knobs: additive — adds the
  `CouplingControl` columns (`cpl_has`, `cpl_k`, `cpl_kr`, `cpl_dtcr`,
  `cpl_enforce`, `cpl_absolute`) to `node_group_payload_dtype` /
  `interpolation_payload_dtype` (plus the `sr_cpl_*` per-slave mirror
  on `surface_coupling_payload_dtype`) so `g.constraints
  .kinematic_coupling` / `distributing_coupling`'s `k` / `kr` /
  `enforce` / `bipenalty_dtcr` / `absolute` knobs round-trip. Pre-2.12.0
  files lack the columns; the reader probes `p.dtype.names` and falls
  back to `control=None`. Per ADR 0023's two-version reader window,
  readers tolerate 2.11.x and 2.12.x.

**From 2.13.0 onward** the per-version rationale is maintained inline
as the canonical log in the `NEUTRAL_SCHEMA_VERSION` docstring in
[`mesh/_femdata_h5_io.py`](../src/apeGmsh/mesh/_femdata_h5_io.py) — this table
condenses it to one line per version; consult the docstring for the
full "why" and the exact affected dtype columns:

- `2.13.0` — fork-coupling host auto-scalers (`k="auto"`, `k_alpha`,
  `host` FEM element id, `bipenalty_wcap`) added to the 2.12.0 coupling
  lane.
- `2.14.0` — ADR 0068 equation-constraint tied interface: adds the
  `enforce` route (`"penalty"|"penalty_al"|"equation"`) to the
  interpolation / surface-coupling lanes.
- `2.15.0` — ADR 0067 P5.1: new `/reinforce_ties` group persists
  `g.reinforce`'s `LadrunoEmbeddedRebar` couplings (previously dropped
  with a deferral warning).
- `2.16.0` — ADR 0067 P5.2 / B1a.2: new `/rebar_elements` group
  persists a cage's auto-emitted (`emit_elements=True`) structural
  elements.
- `2.17.0` — ADR 0069 equalDOF_Mixed: adds the `master_dofs` column to
  the node-pair payload for the retained-node DOFs of an
  `equal_dof_mixed` record.
- `2.18.0` — ADR 0069 follow-up (EmbeddedNodeControl pressure tie):
  adds `cpl_pressure` / `cpl_kp` (+ `sr_cpl_*` mirrors) to the coupling
  lane.
- `2.19.0` — ADR 0071 LadrunoRigidBody: adds `rb_as_element` / `rb_mass`
  to `node_group_payload_dtype` for a rigid_body declared with
  `as_element=True`.
- `2.20.0` — ADR 0071 follow-up: adds the `omega` (3,)-float64 column
  (initial body-frame angular velocity) to `node_group_payload_dtype`.
- `2.21.0` — ADR 0073 follow-up: new `/contacts` group persists
  `g.constraints.contact` / `.mortar` NTS/mortar interactions
  (previously dropped with a deferral warning, no neutral persistence).
- `2.22.0` — ADR 0073 follow-up: new `/embed_ties` group persists
  `g.embed`'s `LadrunoEmbeddedNode` node-to-host couplings.
- `2.23.0` — ADR 0073 follow-up: adds the optional `cell` (broad-phase
  cell-size scale) column to `contact_payload_dtype`.
- `2.24.0` — ADR 0073 follow-up: new `/contact_planes` group persists
  `g.constraints.contact_plane` rigid-plane contacts.
- `2.25.0` — ADR 0073 follow-up: adds the edge-edge contact fallback
  columns (`edge_edge`, `edge_kn`, `edge_band`, `edge_mu`, …) to
  `contact_payload_dtype`.
- `2.26.0` — ADR 20 R3c: adds `corot` / `shape_b` / `has_shape_b`
  (co-rotated bar axis) to `reinforce_tie_payload_dtype`.
- `2.26.1` — patch: `_write_sp_loads` writes one `/loads/sp/{pattern}`
  dataset per case name instead of flattening every SP record into
  `/loads/sp/default` (see the `/loads` section above).
- `2.27.0` — tie `stiffness="auto"`: adds `stiffness_auto` (+
  `sr_stiffness_auto` mirror) to the interpolation / surface-coupling
  lanes.
- `2.28.0` — ADR 0091 load basis: adds the `basis` column
  (`"lagrange"`/`"bernstein"`) to `nodal_load_payload_dtype`.
- `2.29.0` — ADR 0093 S6: new `/interfaces` group persists
  `g.constraints.interface()` oriented coincident-pair zeroLength
  springs (previously refused outright with no persisted form).
- `2.30.0` — 2D contact (the wound chain): widens the value domain of
  `master_nps` / `slave_nps` in `contact_payload_dtype` from `{3, 4}`
  to `{2, 3, 4}` (fork 2D line-segment contact surfaces); also adds the
  additive `outward_mode` column.
- `2.31.0` — 2D mortar (`-slave-segments` lane): adds the `thickness`
  column (2D mortar plane-model out-of-plane thickness) to
  `contact_payload_dtype`.
- `2.32.0` — TIMs A10 S2 (`interface()` on a 3D surface master): adds
  the `orient_t2` / `has_orient_t2` columns to
  `interface_payload_dtype`, carrying the second in-plane tangent of a
  3D pair's `(n, t1, t2)` frame. The `orient` column is unchanged and
  still holds the zeroLength `-orient` argument at both master
  dimensions, so a 2D interface row is byte-for-byte what 2.29.0 wrote.
- `2.33.0` — fork PR #839 (`LadrunoKinematicCoupling -alUpdate`): adds
  the `cpl_al_update` column (uint8 `0`=unset / `1`=commit / `2`=iter)
  to the coupling-control lane on `node_group_payload_dtype` /
  `interpolation_payload_dtype`, plus its `sr_cpl_al_update` per-slave
  mirror on `surface_coupling_payload_dtype`, so
  `g.constraints.kinematic_coupling(..., al_update=...)` round-trips.
  `0` = the flag is omitted and the fork's own `commit` cadence
  applies, which is precisely what every pre-2.33.0 file meant.
- `2.34.0` — #1291: `/meta/ndm` is the `ops.model` spatial dimension.
  A minor bump: the attribute's meaning changes and a new value (`0`)
  appears, and readers branch on the version (ADR 0023 lets them
  branch on a minor, never on a patch). The writer used to stamp the
  highest element dimension of the mesh, so a line-only 2-D frame
  carried `ndm=1` and a shell-only 3-D model `ndm=2`. Composed files
  now stamp the bridge's declared `ndm`; broker-only files (`fem.to_h5`)
  stamp `0`, the undeclared sentinel `ndf` has always used, which
  `OpenSeesModel.build`/`to_h5` and `DomainCapture.from_h5` refuse.
  `h5_reader.read_spatial_ndm` trusts `/meta/ndm` from this minor on
  and salvages it for older files (which the floor admits) from the `per_element_vecxz` widths:
  `0` (a 2-D `geomTransf`, which has no vecxz) resolves 2, `3` lifts the
  stamp to 3, and conflicting evidence, an unknown width, or a result
  that would drop a non-zero coordinate column refuses (#1358);
  `NativeWriter` forwards the salvaged value (not the raw stamp) onto a
  composed file's `/model/meta`.
- `2.35.0` — #1338 nodal load `source`: adds the `source` column to
  `nodal_load_payload_dtype`, the `kind` of the definition a nodal load
  was reduced from (`NodalLoadSource`), so the bridge's
  `WarnBodyForceDoubleCount` guard can tell a reduced self-weight from
  a vertical footing line load after the definitions are gone. `""`
  decodes to `None` (unknown), and an older file reads the same way;
  the guard still warns on an unknown source. Additive, presence-probed
  on read; the minor bumps per ADR 0113 D5.

### OpenSees-zone history (post-2.10)

The window wording below is era-correct history (ADR 0113 retired the
window). The opensees floor is 2.12.0; the 2.11.0 entry records the
rank flip that made 2.10 unreadable, but no 2.11-era file can be opened
either, because its writer stamped a neutral zone below the neutral floor.

The opensees zone advanced past 2.10 independently of the neutral
zone (it shares the early lineage above through 2.10; the entries
below are opensees-only and have no neutral-zone counterpart). Full
detail lives in the `SCHEMA_VERSION` docstring in
[`opensees/emitter/h5.py`](../src/apeGmsh/opensees/emitter/h5.py):

- `2.11.0` — bug fix: the bridge emits **0-based runtime ranks**
  (matching `OpenSeesMP::getPID()`) instead of Gmsh's 1-based
  `PartitionRecord.id`. Per-partition `rank` attrs and the
  `partition_ids` column on `/opensees/element_meta/{type}` are now
  in `[0, N-1]`. Group naming (`partition_NN`) is unchanged. Breaking
  for any reader that mapped `rank`/`partition_ids` to `part.id`
  directly; under the window of the day 2.10 fell out of reach for the
  opensees zone.
- `2.12.0` — ADR 0035 (ASDEmbeddedNodeElement option exposure):
  `/opensees/constraints/embeddedNode` gains five typed columns —
  `stiffness` (`-K`), `stiffness_p` (`-KP`) + `has_stiffness_p`
  sentinel, `rotational` (`-rot`), `pressure` (`-p`). Defaults match
  the C++ parser, so legacy decks behave identically. Additive — old
  2.11.x readers ignore the new columns.
- `2.13.0`–`2.16.0` — see the `SCHEMA_VERSION` docstring in
  [`opensees/emitter/h5.py`](../src/apeGmsh/opensees/emitter/h5.py) (named primitives sidecar,
  `/opensees/nodes_ndf`, `/opensees/dampings`, `/opensees/initial_stress`).
  From 2.16.0 onward a 2.N.x reader REFUSES a 2.(N+1).x file (INV-4: a
  reader opens older files down to the floor, never newer ones).
- `2.17.0` — ADR 0049 (node-pair zeroLength): optional
  `inline_connectivity` vlen-int64 dataset under
  `/opensees/element_meta/{type}/` carrying the endpoint tags of node-pair
  spring elements (no neutral gmsh cell to source them from). Written only
  for type groups with a node-pair (`fem_eid < 0`) row, so PG-only models
  are byte-identical; folds into `model_hash`. Additive group; a 2.16.x
  reader refuses a 2.17.x file (INV-4).
- `2.18.0` — ADR 0055 Phase 2 (staged-model archival, non-partitioned):
  `/opensees/stages/stage_NNN` groups in registration order, each carrying
  the captured resolved per-stage emit stream — `owned_node_ids` /
  `owned_element_ids` (emit order == replay order), `bcs/{fix,mass}`,
  `regions/region_NNN` (with `kind` ∈ `rayleigh` | `damping_attach` |
  `node_or_filter` and an `emit_index` provenance stamp — four producers
  share the pool and the rayleigh-vs-region interleaving carries OpenSees
  overwrite semantics), `constraints/*`, `patterns/*` (stage load patterns
  plus the ADR 0052 HOLD pattern marked `role="hold"` with an `sp_holds`
  (n, 2) int64 dataset of `(node, dof)` pairs; every stage pattern group
  carries `emit_index`), `recorders/*`, `rayleigh` (n, 4) float64 +
  `rayleigh_emit_index`, `remove_sp` / `remove_element`, a per-stage
  `analysis` attr group, and the declarative complement —
  `activated_pgs`, `initial_stress/stress_NNN` (the 2.16.0 field set),
  `activate_absorbing/absorb_NNN` (`pg` attr XOR `elements` dataset).
  Tri-state mutators (`set_time`, `set_creep_on`, `pre_analyze_reset`,
  `analyze_dt`) are presence-encoded attrs — never-set means no attr.
  Staged files carry NO global `/opensees/analysis` group.  The
  `fem_eid → ops_tag` map is NOT duplicated — it is the existing
  `element_meta` `fem_eids`/`ids` columns.  Written only when ≥ 1 stage
  exists (vanilla byte-identical); folds into `model_hash`; a 2.17.x reader
  refuses it (INV-4).
- `2.19.0` — ADR 0055 Phase 5 (P5.1, partitioned staged archival): NO
  layout change — the bump marks that PARTITIONED staged archives now
  exist (the last `apeSees.h5` fail-loud guard is lifted). Per-rank
  replicated emission dedupes to one captured record; per-rank pattern
  and stage-region fragments merge by tag. Folds into `model_hash`;
  a 2.18.x reader refuses it (INV-4).
- `2.20.0` — ADR 0078 Amendment A1 (ComputedSection provenance):
  additive — new optional `/opensees/computed_sections` sidecar
  (`tag` / `analyzer_name` / JSON `payload`), written only when a
  `ComputedSection` emitted. Provenance metadata, not authored model
  state → excluded from `model_hash` (same carve-out as `names`).
  Additive minor (a 2.19.x reader refuses a 2.20.x file, INV-4).
- `2.21.0` — Phase SSI-2.E (`s.update_material_stage`): additive —
  new optional `update_material_stage` dataset (`(N, 2)` int64,
  `(mat_tag, stage)`) under `/opensees/stages/stage_NNN/`, written
  only when a stage actually flips a SANISAND material. Authored
  model state, not provenance → folds into `model_hash`. Additive
  minor (a 2.20.x reader refuses a 2.21.x file, INV-4).
- `2.22.0` — ADR 0112 amendment 5 (#1304): additive — new optional
  `/opensees/bcs@mass_from_model` marker (see [`/opensees/bcs`](#opensees-bcs)).
  Under `mass_from_model()` the archive skips the mass stream and
  replay streams the neutral zone's `/masses`. Authored model state →
  folds into `model_hash`. Additive minor (a 2.21.x
  reader refuses a 2.22.x file, INV-4).
- `2.23.0` — ADR 0114 R2/R3a (K1-4, #1461): additive — new
  [`/opensees/program`](#opensees-program-opensees-2230) (the emit
  order, every file) and optional
  [`/opensees/commands`](#opensees-commands-optional-opensees-2230)
  (global `rayleigh` / `eigen` / `modal_damping` and stage `profiler`,
  which earlier files dropped). Both fold into `model_hash`, so an
  identical model hashes differently once at this minor (ADR 0114 Q5).
  Additive minor (a 2.22.x reader refuses a 2.23.x file, INV-4).
- `2.24.0` — ADR 0114 D6/R4 (K1-5, #1462): additive — three optional
  attributes on `/opensees` itself, `@will_solve`, `@solve_refusals` and
  `@requires` (see [the solve stamp](#opensees-attributes-the-solve-stamp-optional-opensees-2240)),
  written together once the bridge hands the writer a `SolveStamp`; a
  file without them reads as "no stamp" (`None`). Attributes of
  `/opensees` fold into `model_hash`. Additive minor (a 2.23.x reader
  refuses a 2.24.x file, INV-4).

This is the **current** opensees-zone version (`SCHEMA_VERSION` in
[`opensees/emitter/h5.py`](../src/apeGmsh/opensees/emitter/h5.py)); check that constant
directly before trusting this list on a future read — it is a
condensed log, not the source of truth.

A reader skeleton:

```python
import h5py

def read_model_h5(path):
    with h5py.File(path, "r") as f:
        meta_attrs = f["/meta"].attrs
        # Read the per-zone key, not the legacy envelope. Both zones
        # share major 2 today; check the major of the zone you read.
        version = meta_attrs.get(
            "neutral_schema_version", meta_attrs.get("schema_version"),
        )
        major = int(str(version).split(".")[0])
        if major != 2:
            raise ValueError(
                f"Unsupported model.h5 neutral-zone major version "
                f"{major}; reader supports v2.x.y"
            )
        # Walk the file ...
```

## Worked example — minimal model

A single elastic column with one fiber section, one ground motion,
no analysis settings:

```
column.h5
├── /meta
│   schema_version="2.3.0", ndm=3, ndf=6, snapshot_id="abc123"
├── /nodes
│   ├── ids       [1, 2]
│   └── coords    [[0,0,0], [0,0,1]]
├── /elements/line2/                  ← broker keying (GMSH alias)
│   ├── attrs: code=1, gmsh_name="Line 2", npe=2, dim=1, order=1
│   ├── ids           [1]
│   └── connectivity  [[1, 2]]
└── /opensees/
    ├── /materials/uniaxial/Steel02_1/
    │   type="Steel02", tag=1,
    │   params=[4.2e8, 2.0e11, 0.01, 20.0, 0.925, 0.15]
    ├── /materials/uniaxial/Concrete02_2/
    │   type="Concrete02", tag=2,
    │   params=[-3e7, -2e-3, -2.5e7, -6e-3, 0.1, 2.5e6, 2e8]
    ├── /sections/Fiber_1/
    │   ├── attrs: type="Fiber", tag=1,
    │   │          params=[nan, 1.0e9], params_str=["-GJ", ""]
    │   ├── /patches  → 1 row: kind="rect",
    │   │              material_ref="/opensees/materials/uniaxial/Concrete02_2",
    │   │              ny=8, nz=8, coords=[-0.20,-0.20,0.20,0.20,nan,nan,nan,nan]
    │   └── /fibers   → 8 rows of (y, z, area,
    │                              material_ref="/opensees/materials/uniaxial/Steel02_1")
    ├── /transforms/PDelta_1/
    │   ├── attrs: type="PDelta", tag=1, __deviation__="per-emitted-call grouping"
    │   ├── per_element_vecxz       (1, 3) = [[1, 0, 0]]
    │   └── per_element_emitted_tag (1,)   = [1]
    ├── /element_meta/forceBeamColumn/        ← bridge keying (OpenSees type)
    │   ├── attrs: type="forceBeamColumn"
    │   ├── ids        [1]
    │   ├── fem_eids   [1]                      — Phase 8.6 mapping (broker's eid → ops_tag)
    │   └── args       [[1, 1]]                 — (transf_tag, integration_tag)
    ├── /time_series/elcentro/
    │   ├── attrs: type="Path", factor=9.81, dt=0.01, file_path="elcentro.txt"
    │   ├── time       (n_steps,)  = [0.00, 0.01, 0.02, ...]
    │   └── values     (n_steps,)  = [0.001, 0.005, ..., -0.012, ...]
    ├── /patterns/Quake/
    │   └── attrs: type="UniformExcitation", tag=1, direction=1,
    │              series_ref="/opensees/time_series/elcentro"
    └── /bcs/fix
        target_kind=["pg"], target=["Base"], dofs=[[1,1,1,1,1,1]]
```

This file is ~50 KB and tells the viewer everything it needs to
draw the column with its section, materials, orientation, and
ground motion — without reading a single OpenSees recorder output.
