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
      │   @target_mode                       pinned target mode (optional,
      │                                      opensees 2.36.0; not hashed)
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
      ├── /program                         run-length emit order (opensees 2.23.0)
      ├── /decls                           the bridge's declarations, hash-excluded
      │                                    (optional, opensees 2.25.0)
      └── /decl_params                     each declaration's parameters by field name,
                                           hash-excluded (optional, opensees 2.26.0)
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
| `units` | string | canonical consistent unit system ID (`"N_m_s"`, `"kN_m_s"`, `"tf_m_s"`, `"kgf_m_s"`, `"N_mm_s"`, `"kip_in_s"`, `"dyne_cm_s"`). **Optional**: written when the session or `FEMData` declares `units=`; absent when undeclared, and read without a version gate (ADR 0023 INV-2) |
| `tag_span_max` | int | `max(max_node,max_elem) - min(min_node,min_elem) + 1`; sizes compose tag-offset reservations (ADR 0038) |

> The `/meta/lineage/` sub-group (ADR 0021) is stamped whenever the
> neutral zone is written, not only by the composer: `write_fem_h5`
> (the broker-only `fem.to_h5(path)` path) stamps just `fem_hash`
> (there is no `/opensees/` zone to fold into a `model_hash` yet); the
> composer additionally computes and stamps `model_hash` once
> `/opensees/...` exists. The composed `results.h5` carries its
> version keys at the **file root** (not `/meta`) — see
> [Schema versioning](#versioning) and
> [`results.h5`: the native results file](#resultsh5-the-native-results-file).

### `results.h5` root attrs: OpenSees build provenance

A `results.h5` written by the native writer (`NativeWriter.open`) may carry
two root attrs naming the OpenSees build that produced its data. The whole
file layout, including the other root attrs, is in
[`results.h5`: the native results file](#resultsh5-the-native-results-file). They come
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
`opensees/emitter/_h5_write.py`. The meaning of each slot is the OpenSees
manual's argument order for `type`; the file carries no parameter
names. Each declaration's parameters by field name live in
[`/opensees/decl_params`](#opensees-decl_params-optional-opensees-2260)
(opensees 2.26.0; writer `_write_decl_params` in
`opensees/emitter/_h5_decls.py`); a reader that needs names reads them
there, not from `params`.

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

One group per emitted `timeSeries` call, named `{type}_{tag}`
(`time_series_name` in
[`opensees/emitter/_h5_write_model.py`](../src/apeGmsh/opensees/emitter/_h5_write_model.py);
a user-supplied `name=` is recorded in `/opensees/names` and
`/opensees/decls`, never as the HDF5 group name).

```
/opensees/time_series/Linear_1/
└── attrs: type="Linear", tag=1,
          params=[nan, 2.0], params_str=["-factor", ""]
          (params=[] and no params_str when factor == 1.0)

/opensees/time_series/Path_2/
└── attrs: type="Path", tag=2,
          params=[nan, 0.0, 0.5, 1.0, nan, 0.01, nan, 9.81],
          params_str=["-values", "", "", "", "-dt", "", "-factor", ""]
```

Each group (`H5Emitter._write_time_series`, `H5Model.time_series`)
stores only:

* `type` — vlen UTF-8 string attribute: the emitted OpenSees
  `timeSeries` command token (`"Linear"`, `"Constant"`, `"Path"`,
  `"Trig"`, `"Pulse"`).
* `tag` — int64 attribute: the emitted OpenSees time-series tag.
* `params` / `params_str` — the positional OpenSees arguments after
  `tag`, written by `_write_param_array` (`params` float64 `(n_args,)`
  with `NaN` at string slots; `params_str` vlen UTF-8 `(n_args,)` with
  `""` at numeric slots, omitted when all slots are numeric). Stored as
  attributes, or as uncompressed datasets of the same names when HDF5
  refuses either half for size (see below).

The emitted positional argument sequence after `tag` per `type`
([`opensees/time_series/time_series.py`](../src/apeGmsh/opensees/time_series/time_series.py)):

| `type` | Emitted positional arguments in `params` / `params_str` (optional flags emitted only when non-default) |
|---|---|
| `"Constant"`, `"Linear"` | `()` when `factor == 1.0`, else `("-factor", factor)` |
| `"Trig"` | `(t_start, t_end, period[, "-factor", factor][, "-shift", shift][, "-zeroShift", zero_shift])` |
| `"Pulse"` | `(t_start, t_end, period, width[, "-factor", factor][, "-shift", shift][, "-zeroShift", zero_shift])` |
| `"Path"` | `(("-filePath", file) \| ("-values", *values)[, ("-dt", dt) \| ("-time", *time)][, "-factor", factor][, "-startTime", start_time][, "-prependZero"][, "-useLast"])` |

**No pre-sampled `time` / `values` datasets on disk.** The writer does
not sample algorithmic series (`Linear`, `Constant`, `Trig`, `Pulse`)
and does not create standalone `time` or `values` datasets under
`/opensees/time_series/{name}`:

* **Closed-form series (`Constant`, `Linear`, `Trig`, `Pulse`)** store
  their command arguments in `params` / `params_str` and their
  dataclass fields (`factor`, `t_start`, `t_end`, `period`, `width`,
  `shift`, `zero_shift`) by field name in
  [`/opensees/decl_params`](#opensees-decl_params-optional-opensees-2260).
  Consumers (`apeGmshViewer/src/panels/windows/plot/series.ts`)
  evaluate the OpenSees closed forms directly from those parameters.
* **Inline `Path` series and generator primitives** (`Path(values=...)`,
  plus `Ricker`, `MomentStep`, `Yoffe`, `ASCE41Protocol`,
  `ModifiedATC24Protocol`, and `FEMA461Protocol`, which sample their
  waveform at emit time and delegate to `Path`) emit as `type="Path"`
  with `-values v0 v1 ...` and `-dt dt` (or `-time t0 t1 ...`) in the
  `params` / `params_str` pair. For a direct `Path(values=...)`
  declaration, `/opensees/decl_params` also carries `values` (and
  `time` or `dt`, `factor`, `start_time`, `prepend_zero`, `use_last`)
  as JSON fields; for a generator primitive, `/opensees/decl_params`
  carries the generator class and its own dataclass knobs while the
  tabulated samples sit in `/opensees/time_series/Path_{tag}`'s
  `params` / `params_str`.
* **File-backed `Path(file=...)` series** carry `-filePath <path>` in
  `params` / `params_str` (and `file` in `/opensees/decl_params`); the
  external file is not copied into the archive.

### Large arrays as datasets (opensees 2.34.0)

Every typed store above (materials, sections, beam integrations, time
series, patterns, dampings, regions, recorders, and their stage
copies) keeps a call's positional arguments as the `params` /
`params_str` pair: `params` float64 with `NaN` in string slots,
`params_str` UTF-8 with `""` in numeric slots, `params_str` absent when
no slot is a string. The writer tries the pair as **attributes**. When
HDF5 refuses either half for size (an attribute is one object-header
message of at most 65,535 bytes: about 8,183 float64 or 4,091 string
slots), it writes two **uncompressed datasets of the same names** in the
same group instead, as an element's `args` / `args_str` are. A long
`Path` series failed to archive before (#2404).

```
/opensees/time_series/Path_1/
├── attrs: type="Path", tag=1
├── params        float64 dataset (n,)   ← only when the attribute is refused
└── params_str    vlen str dataset (n,)  ← when any slot is a string
```

- The pair switches together: a half already written as an attribute is
  deleted, so a group holds both arrays in one form.
- A declared recorder's `ids` follows the same rule (an int64 attribute,
  or when HDF5 refuses that, an int64 dataset named `ids`).
- Any other sequence attribute HDF5 refuses for size raises at write,
  naming the key and the group.
- The reader reads either form and raises `MalformedH5Error` when a key
  is in both forms, when the pair is split across them, or when a file
  stamped below 2.34.0 holds the dataset form
  (`h5_reader.LARGE_ARRAY_DATASETS_FROM`).
- `model_hash` walks datasets as it walks attributes, so the values are
  hashed in either form. The fallback fires only where a 2.33 writer
  failed, so no archive a 2.33 writer could write changes bytes or hash.
- The app (`apeGmshViewer`) reads either form for every typed store and
  refuses the same malformed cases.

## `/opensees/patterns`

One group per emitted `pattern` block, named `{type}_{tag}`
(`pattern_name` in
[`opensees/emitter/_h5_write_model.py`](../src/apeGmsh/opensees/emitter/_h5_write_model.py)).
Global patterns live under `/opensees/patterns/{type}_{tag}`;
stage-scoped patterns live under
`/opensees/stages/stage_NNN/patterns/{type}_{tag}` with the identical
group layout plus an `emit_index` (int64) attribute and, for an ADR 0052
`s.support` HOLD pattern, `role="hold"` and an `sp_holds` `(n_holds, 2)`
int64 dataset of `(node_tag, dof)` pairs.

```
/opensees/patterns/Plain_1/
├── attrs: type="Plain", tag=1,
│         series_ref="/opensees/time_series/Linear_1",
│         params=[1.0]
├── /loads               compound dataset, shape (n_loads,)
│     fields: target_kind (vlen utf-8),  ← always "node" (PGs already fanned out)
│             target (vlen utf-8),       ← node tag as string
│             forces (float64[ndf])      ← padded with NaN to ndf
├── /sps                 compound dataset, shape (n_sps,)
│     fields: target_kind (vlen utf-8),  ← always "node"
│             target (vlen utf-8),       ← node tag as string
│             dof (int64), value (float64)
└── /element_loads       vlen utf-8 dataset, shape (n_calls, max_tokens)
      raw eleLoad command tokens, one row per emitted eleLoad call,
      right-padded with "" to max_tokens
```

Each child dataset (`loads`, `sps`, `element_loads`, `sp_holds`) is
created only when that pattern emitted at least one record of that kind.

### `element_loads`: 2-D token table

`/opensees/patterns/{name}/element_loads` (and
`/opensees/stages/stage_NNN/patterns/{name}/element_loads`) is a **2-D
variable-length UTF-8 string dataset** of shape `(n_calls, max_tokens)`
(`dtype=h5py.string_dtype(encoding="utf-8")`), written by
`H5Emitter._write_pattern_ele_loads`
([`opensees/emitter/_h5_write_model.py`](../src/apeGmsh/opensees/emitter/_h5_write_model.py))
and read by `H5Model._pattern_record`
([`opensees/emitter/h5_reader.py`](../src/apeGmsh/opensees/emitter/h5_reader.py))
and `apeGmshViewer` (`readTable` in `apeGmshViewer/src/reader/read.ts`):

* **Rows and padding.** Row `i` holds the positional arguments of the
  `i`-th `emitter.eleLoad(*args)` call in that pattern. String tokens
  are stored as `str(v)` and numeric arguments via `repr(v)`. When calls
  in the same pattern have different token counts, shorter rows are
  right-padded with `""` (empty string) to `max_tokens`. Readers drop
  `""` cells when reconstructing a call's token list.
* **Target tokens (before `-type`).** Every row starts with its target
  elements in **OpenSees element tag space** (the bridge's
  `resolve_ele_load_commands` has already fanned out any `pg=`,
  `label=`, `Element` handle, or `from_model(case)` declaration and
  translated FEM element IDs through the build's `fem_eid -> ops_tag`
  map; consumers map an OpenSees element tag back to its FEM element ID
  via `/opensees/element_meta/{type}/ids` and `fem_eids`):
  * `"-ele", "<tag_1>", ["<tag_2>", ...]` — one or more explicit
    OpenSees element tags (followed by `"-type"`).
  * `"-range", "<start_tag>", "<end_tag>"` or
    `"-eleRange", "<start_tag>", "<end_tag>"` — inclusive OpenSees
    element tag range `start_tag .. end_tag`.
* **Load-type tokens (`"-type", "<kind>", ...`).** Immediately after the
  target tokens sits `"-type"` followed by the OpenSees element-load
  kind token and its numeric arguments (`_beam_uniform_args` and
  `_self_weight_args` in
  [`opensees/pattern/pattern.py`](../src/apeGmsh/opensees/pattern/pattern.py),
  parsed by `parse_ele_load_beam_uniform` in
  [`results/_beam_diagrams.py`](../src/apeGmsh/results/_beam_diagrams.py)):

  | Kind token | Model `ndm` | Trailing float tokens (after `"-type", "<kind>"`) | Meaning (element-local axes `x, y, z`; fractional span `0 <= aL < bL <= 1`) |
  |---|---|---|---|
  | `"-beamUniform"` | `2` | 2 tokens: `wy, wx` | full-span uniform (`aL=0, bL=1`, `wya=wyb=wy`, `wxa=wxb=wx`) |
  | `"-beamUniform"` | `2` | 4 tokens: `wy, wx, aL, bL` | partial-span uniform on `[aL, bL]` (`wya=wyb=wy`, `wxa=wxb=wx`) |
  | `"-beamUniform"` | `2` | 6 tokens: `wya, wxa, aL, bL, wyb, wxb` | partial-span trapezoidal on `[aL, bL]` |
  | `"-beamUniform"` | `3` | 2 tokens: `wy, wz` | full-span uniform transverse (`aL=0, bL=1`, `wx=0`) |
  | `"-beamUniform"` | `3` | 3 tokens: `wy, wz, wx` | full-span uniform (`aL=0, bL=1`, `wya=wyb=wy`, `wza=wzb=wz`, `wxa=wxb=wx`) |
  | `"-beamUniform"` | `3` | 5 tokens: `wy, wz, wx, aL, bL` | partial-span uniform on `[aL, bL]` |
  | `"-beamUniform"` | `3` | 8 tokens: `wya, wza, wxa, aL, bL, wyb, wzb, wxb` | partial-span trapezoidal on `[aL, bL]` |
  | `"-selfWeight"` | `2` | 2 tokens: `wx, wy` | element self-weight factors `(wx, wy)` |
  | `"-selfWeight"` | `3` | 3 tokens: `wx, wy, wz` | element self-weight factors `(wx, wy, wz)` |
  | `"-surfaceLoad"` | `2` or `3` | 0 tokens | surface pressure follower marker |

`/opensees/patterns/UniformExcitation_2/` for `UniformExcitation`:

```
/opensees/patterns/UniformExcitation_2/
└── attrs: type="UniformExcitation", tag=2, direction=1,
          series_ref="/opensees/time_series/Path_1",
          params=[1.0, nan, 1.0], params_str=["", "-accel", ""]
```

(no contained load datasets — uniform excitation IS the pattern's payload)

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
                 the call wrote nothing (a `ledger` verb's dropped call)
    row     i4   the record's ordinal among the records `method` wrote
                 to `store` in this stage, in emit order; -1 with store -1
    stage   i4   -1 global, else the `stage_NNN` ordinal
    decl    i4   the `/opensees/decls` row of the declaration that made the
                 calls, when the emit proves it, else -1 (opensees 2.31.0,
                 A7; -1 throughout below 2.31.0)
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

### `decl`: each call's declaration (opensees 2.31.0)

From 2.31.0 (A7, #1839; the maintainer's ruling of 2026-10-11 on #1646
item 1) `decl` names the `/opensees/decls` row of the declaration that
made the call, so a stage's calls join to their declarations through the
program alone. The writer records it at emit time and never guesses:

| The call | Its `decl` |
|---|---|
| a tagged verb (`uniaxialMaterial`, `nDMaterial`, `section`, `section_open`, `geomTransf`, `beamIntegration`, `timeSeries`, `pattern_open`, `damping`, `region`, `element`) | the `/opensees/decls/tags` run that holds its `(kind, tag)`; a fan-out element its spec's |
| a tagless verb whose row has a `/opensees/decls/rows` entry filled at the call (`fix`, `mass`, `recorder`, `sp_hold`, `remove_sp`, `remove_element`, `update_material_stage`, a stage `rayleigh`, and the `/opensees/commands` rows: global `rayleigh`, `eigen`, `modal_damping`, `profiler`, `command`) | the declaration the bridge opened on it (`set_declaration`) |
| a side-channel store's call (`step_hook_ramp`, `addToParameter`, `flip_element_stage`, `update_parameter`, `set_node_vel`, `set_node_accel`), from 2.32.0 (#1853) | the declaration of the record that drove it (an initial stress, an absorbing flip, an `s.update_parameter`, an `s.zero_velocities`), which the bridge opens around that record's calls; `-1` below 2.32.0 |

Everything else is an explicit **`-1`**:

- **no declaration of its own**: `node` and the MP constraints
  `equalDOF`, `rigidLink`, `rigidDiaphragm`, `embeddedNode`,
  `embedded_rebar` (the FEM snapshot declares them, in the neutral zone);
  the parts of a tagged record (`section_close`, `patch`, `fiber`, `layer`,
  `pattern_close`, `load`, `eleLoad`, `sp`, the recorder declaration
  brackets, `mp_constraint_comment`); and the control calls (`model`, the
  analysis chain, `analyze`, the stage brackets and domain calls,
  partition brackets and runtime fallbacks), whose primitives are
  declarations in `/opensees/decls/tags` but whose calls carry no tag to
  join on;
- **not proved at emit time**: a tagged call whose `(kind, tag)` no run
  holds (an element a side pass mints with no declaration); and a tagless
  call with no declaration open, such as a ghost replay (ADR 0027 INV-2),
  whose row the `/opensees/decls/rows` column claims afterwards;
- the `ledger` verbs' dropped calls, which write no row.

A node that two `s.zero_velocities` records of one stage name is zeroed
once, so its `set_node_vel` / `set_node_accel` calls are the first
record's (the rule `BuiltModel._claim_ghost_fixes` uses for ghost fixes).

A run breaks where `decl` changes. The reader refuses a `decl` outside
`{-1} ∪ [0, n_decls)` (`n_decls = 0` when the file has no
`/opensees/decls`) with `MalformedH5Error`, and the writer refuses to
write one. A rewrite echoes the source's column, an older source's `-1`
values included. Below 2.31.0 the column is `-1` throughout and says
nothing about the call. The rewrite of such a file is stamped 2.31.0 but
keeps that column, since the rewrite cannot prove a call's declaration it
did not emit. The version therefore does not tell a reader whether the
joins are recorded; the completeness marker does:

```
/opensees/decls
  @program_decl   vlen str   "complete": /opensees/program's decl column
                             records every call's proved declaration (a
                             2.31+ apeSees emit); "absent": it was never
                             recorded (a to_h5 rewrite of a file below
                             2.31.0). A file below 2.31.0 has no marker
                             and reads as "absent".
```

The marker sits on `/opensees/decls`, not on `/opensees`. The
attributes of `/opensees` fold into `model_hash` (`canonical_bytes`
hashes the root group's attributes, all but those in
`MODEL_HASH_EXCLUDED_ATTRS`, which is `@target_mode` alone), so an
attribute there would move a
rewritten legacy file's hash. `decls` is hash-excluded, and it is the
table the joins point into: with no `decls` group there is nothing to
join to. A rewrite echoes the source's marker with its program, so a
legacy rewrite stays `"absent"` and keeps its hash, and a rewrite of a
2.31 emit stays `"complete"`. A marker outside those two values raises
`MalformedH5Error`.

Only decl-join queries fail closed. `H5Model.program_declarations()`
pairs each run with its `DeclarationRO` (or `None` for `-1`) and raises
`ProgramDeclAbsentError` ("joins not recorded") when
`records_program_decl(H5Model.declarations())` is false, that is when the
marker is `"absent"` or there is no `decls` group. It never presents an
unrecorded `-1` as "undeclared". `H5Model.program()`, `emit_index` and
every other accessor read the file as before.

`decl` is hashed, so every archive the bridge writes changes its
`model_hash` once at 2.31.0. Declaration ids are first-seen row indices,
so a name (a label) still leaves the hash alone; declaration *order*
now enters it. A file written before 2.31.0 keeps its stored hash, and a
recompute over it gives the same value (its column is still `-1`); only
a re-emit through `apeSees` writes the new column. A tool that compares a
model's `model_hash` with a results file's across that boundary reads the
`/opensees/decls@program_decl` marker on both sides, not
`/meta@opensees_schema_version`: a `to_h5` rewrite of an older file is
stamped current but keeps its old hash, so only the marker tells the era
(`records_program_decl`; `Results` names the cause this way, #1861).

At 2.32.0 (#1853) the side-channel calls gain their declarations, so an
archive that carries any of them (an initial stress, an absorbing flip,
`s.update_parameter`, `s.zero_velocities`) changes its `model_hash` once
more; any other archive is unchanged. The marker means the same thing at
both minors ("every provable join recorded"), so it does not tell a 2.31
emit from a 2.32 one: such a pair shows as a plain `model_hash` mismatch.

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
R4; verdicts per partition mode by the maintainer's ruling of
2026-10-10 on #1462), as five attributes on the `/opensees` group itself:

```
/opensees
  @will_solve            i1              1 when the model carries a solve
                                         (`staged or any(Analysis)` at emit), else 0
  @solve_mode            vlen str        the archive's own partition mode:
                                         `partitioned` (the FEM is partitioned and
                                         the target consumes the brackets) or `flat`
  @solve_refusals        vlen str (R,)   ids of the solve-time gates that refuse
                                         the solve under @solve_mode, in gate order;
                                         empty when none does
  @solve_refusals_flat   vlen str (F,)   the same gates' verdict under the flat
                                         (single-domain) mode; equals
                                         @solve_refusals when @solve_mode is `flat`
  @requires              vlen str (Q,)   sorted union of the emitted verbs'
                                         `VERBS.requires` tokens (`fork`: the
                                         Ladruno build) and, from 2.33.0, the
                                         value tokens `mpco` / `mpi`; empty
                                         when none
```

The `@requires` tokens (`REQUIRES_TOKENS` in `emitter/caps.py`):

| Token | From | Written when | Met by (`build('live')`, the live net) |
|---|---|---|---|
| `fork` | 2.24.0 | an emitted verb's `VERBS` row requires it | a Ladruno build (`ladrunoBuild()` answers a sha) |
| `mpco` | 2.33.0 | any `recorder mpco`, global or staged | a Ladruno build: pip openseespy ships no MPCO recorder |
| `mpi` | 2.33.0 | a Mumps-family or `Parallel*` `system` (global or staged) that has no runtime fallback, or `@solve_mode` `partitioned` | never in process: the deck needs a multi-rank MPI run |

`mpco` and `mpi` depend on argument values, so `VERBS` rows do not
carry them; the stamp writer derives them from its own stores
(`H5Emitter.solve_stamp`, `caps.value_requires`). The MPI systems are
`Mumps` and `ParallelProfileSPD` (`caps.is_mpi_system`; neither stock
openseespy nor the fork's serial build declares them). `MPIDiagonal` falls
back to the serial diagonal solver on both, so it needs nothing, and
neither does the ADR 0027 auto-emitted `Mumps`, which carries an `UmfPack`
runtime fallback (`caps.binding_systems`). `@requires` describes the
archive's own deck. A flat replay re-derives its need from the archived
chains (`OpenSeesModel.replay_requires`, `caps.flat_requires`): it keeps
`mpi` only when a chain declares an MPI system, so a partitioned mode
alone does not bind a flat replay. A file below 2.33.0
never names them, and its `@requires` keeps the meaning it was written
with (verb tokens only). A reader accepts any non-empty token, known or
not; a solving target refuses a token it does not know.

`BuiltModel.emit` hands the writer `will_solve`, the mode and the two
verdicts through `H5Emitter.set_solve_stamp` on every `apeSees.h5`
emit; the writer derives `@requires` from the methods its
`/opensees/program` holds (ledger rows included: the verb was emitted
even if it stored nothing) and stamps the five together. The refusal
ids are the solve-time gates of `BuiltModel.emit` with the `validate_`
prefix dropped (`ladruno_up_solver`, `serial_mumps`,
`up_pressure_datum`), each probed as a solve would run it, once under
the archive's own partition mode and once flat. The two verdicts
differ exactly where a gate reads the mode: a partitioned deck with no
`system` rides the ADR 0027 auto-emitted general solver (allowed) while
its flat replay would solve on ProfileSPD (refused), and an explicit
`Mumps` is legal partitioned and refused serial. `H5Model.solve_stamp()`
returns a `SolveStamp` (`emitter/caps.py`, with `refusals_for(mode)`),
or `None` for a file that lacks `@will_solve` (every file below
2.24.0). Absent is "unknown", never a default. A `@will_solve` without
its four companions, one that is not the integer 0 or 1, a mode outside
`SOLVE_MODES`, a `flat` stamp whose two verdicts differ, or a token
array that is not sorted and unique raises `MalformedH5Error`.
`OpenSeesModel.build('tcl' | 'py' | 'live')` emits flat (the read side
is partition-blind), so it fails closed on a non-empty
`@solve_refusals_flat`, naming the archive's own verdict beside it;
`build('live')` refuses on a backend that lacks a token in `@requires`
of its flat replay (`replay_requires`: `fork` and `mpco` need a
Ladruno build, `mpi` (an MPI system in the chain) always refuses in
process, and so does an unknown token);
`to_h5` echoes the stamp, and `@requires` regenerates from the echoed
program and the rewritten stores (a rewrite of a file below 2.33.0
gains the value tokens its model implies). A fork *element or
material type* rides the generic `element` / `nDMaterial` verb, whose
row requires nothing, so it does not reach `@requires` until K4 moves
typed fork verbs onto the command channel. Attributes of `/opensees`
fold into `model_hash`, except `@target_mode` (below).

## `/opensees@target_mode` (optional, opensees 2.36.0)

```
/opensees
  @target_mode           vlen str ()     "fork" or "stock": the bridge's
                                         pinned OpenSeesTarget.mode
```

The mode of the `OpenSeesTarget` the bridge was built with (#2278, ruling
A; #2375). It is written only for a pinned target. An unpinned model
(`mode="auto"`, or no target) writes nothing, and an absent attribute reads
as `"auto"`, so every file below 2.36.0 keeps its meaning. Only the mode is
kept: `binary` and `python` are machine paths and are never archived.

`H5Model.target_mode()` returns `"auto"`, `"fork"` or `"stock"`. A value
that is not one string of `TARGET_MODES` (`emitter/h5_reader.py`), the
empty string included, raises `MalformedH5Error` naming
`/opensees@target_mode`. `OpenSeesModel` uses it as the bridge uses the
target:

- `build('tcl' | 'py')` stamps the deck through `deck_backend`, so a pinned
  archive's replay writes the bridge's `# apeGmsh <v>; backend <mode>`
  line and an unpinned one writes none;
- a `"fork"` archive adds `"fork"` to `replay_requires()`, so
  `build('live')` refuses a stock openseespy, and the stage-close route
  checks on replay (#2465) treat the run as on the fork, which brings back
  the #2444 fork-reset refusal;
- `to_h5` echoes it.

The attribute is in `MODEL_HASH_EXCLUDED_ATTRS`
(`_internal/lineage.py`), so it never reaches `model_hash`: the same
model pinned or on auto has the same physics and the same hash, the
carve-out `regions` and `names` already have. Two archives with one hash
can therefore stamp different backends.

## `/opensees/decls` (optional, opensees 2.25.0)

The bridge's declarations (ADR 0114 R5, K1-6): which user declaration
each tag and each tagless row came from, under the declaration key of
its `/provenance` record (ADR 0112 D3). Written on every `apeSees.h5`
emit; a rewrite (`OpenSeesModel.to_h5`) echoes it. A label, not
structure: `decls` is in `MODEL_HASH_EXCLUDED_CHILDREN`, so `model_hash`
does not read it. From 2.31.0 the hashed `/opensees/program` `decl`
column names these rows (its section above).

```
/opensees/decls/
    key       vlen str (N,)   opensees/<family>/<name|#k>, or a synthesised
                              <verb>:<owner>[/<role>] key; one row per key
    family    vlen str (N,)   the allocator kind (uniaxialMaterial, element,
                              recorder, ...) or the tagless verb (fix, mass)
    name      vlen str (N,)   the user's name=, "" when unnamed
    synth     i1 (N,)         1 for an object apeGmsh synthesised in a user
                              verb (the HOLD series of s.support)
    tags/                     the tagged declarations, as runs
        kind  vlen str (T,)   allocator kind
        first i8 (T,)         first tag of the run
        count i8 (T,)         tags first .. first + count - 1
        decl  i8 (T,)         row of the declaration
    rows/<store>  i8 (R,)     one declaration row per record of a tagless
                              store, aligned to it: bcs/fix, bcs/mass,
                              recorders (one per group, in index order),
                              initial_stress (one per stress_NNN),
                              commands (one per /opensees/commands row),
                              and under stages/stage_NNN bcs/fix, bcs/mass,
                              recorders, initial_stress, remove_sp,
                              remove_element, update_material_stage,
                              activate_absorbing, rayleigh and
                              patterns/<hold>/sp_holds (s.support)
```

The tag join is the tag plan's `(kind, tag)` (#1445 open question 5):
every registered primitive gives its own tag, except an element spec,
whose registered tag is never written; its planned fan-out (a run of
element tags) inherits the spec's row, and an orientation fan-out's extra
`geomTransf` tags inherit their transform's. A `region` tag joins to its
site's owner: a named region to its name's first `ops.region` / `s.region`
call (one declaration per name), a damping attach's region to its
`Damping` object, a region-scoped Rayleigh's to its `rayleigh` record, a
filtered recorder's to the recorder. `rows/commands` gives every
`/opensees/commands` row its declaration (a global `rayleigh`, the `eigen`
+ `modal_damping` pair of `ops.damping.modal`, a stage's `s.profile`
bracket), and `stages/stage_NNN/rayleigh` each stage `rayleigh` row;
`ops.equation_constraint` is keyed but has no `rows` column (from opensees 2.35.0 its line is a row of [`/opensees/constraints/equationConstraint`](#opensees-constraints-equationconstraint-optional-opensees-2350), joined by its `/opensees/program` call). An unnamed object apeGmsh
registers inside a user call that already has its provenance record
(one record per user call) has no record of its own; it is keyed
`opensees/<family>/@k` with `synth` 1. The `rows` columns are filled at the head of each
tagless loop (`H5Emitter.set_declaration`); a partition replica whose
first capture had none open (a ghost replay before its owner's `fix`)
takes the owner's, and a row that no declaration claims refuses at
write; a ghost replay no replica claims (a node only ghosts carry) takes,
after the emit, the first fix or `remove_sp` record of its replay order
(global, then stages `0..i`). `fix`, `mass` (flat and staged) and the
recorders take `name=` for this table only: unique per family (both call
sites named on a repeat), never in `/opensees/names`;
`fix_from_model()` / `mass_from_model()` are one declaration each, and a
user name of the `@<k>` form is refused. `H5Model.declarations()` returns a
`DeclarationTable` (`for_tag(kind, tag)`, `for_row(store, row)`,
`by_key(key)`), or `None` for a file without the group;
`OpenSeesModel.declarations` delegates. A column whose length disagrees
with its store, or an index past the rows, raises `MalformedH5Error`.

## `/opensees/decl_params` (optional, opensees 2.26.0)

Every declaration's parameters **by field name** (ADR 0114 A6 and Q4,
K1-7): the K0-8 record a reader uses to show *what* a declaration was,
without an OpenSees syntax table on either side. At least one row (one
per owner) per `/opensees/decls` row: a registered primitive, a `fix` / `mass` /
`region` / `rayleigh` / `modal_damping` / `initial_stress` /
`equation_constraint` record, a stage record (`support`, `remove_sp`,
...), or the model-wide `fix_from_model()` / `mass_from_model()` object
(its one field is `verb`). **A key with several owners has one entry per
owner**, consecutive rows with the same `decl`, in call order: the one
such family is `region`, whose name is the declaration's identity, so
`ops.region(name="core", nodes=[1]); ops.region(name="core", nodes=[2])`
stores two `RegionAssignmentRecord` entries (the merged membership is
the `/opensees/regions` store's). `initial_stress` shares a key by name
too, but its own validation refuses a repeated name at build, so it never
reaches the archive with two owners. The per-node records
`fix_from_model()` expands to are that declaration's expansion
(`/opensees/bcs`), not owners; its one entry is the `verb`. Nothing is
dropped: the writer refuses a declaration with no owner, and the
reader returns every entry.
Written on every `apeSees.h5` emit; a rewrite (`OpenSeesModel.to_h5`)
echoes it verbatim. A derived view of the hashed stores, not structure:
`decl_params` is in `MODEL_HASH_EXCLUDED_CHILDREN`, so `model_hash` is
unchanged for every model.

```
/opensees/decl_params/
    decl             i8 (P,)         the /opensees/decls row of the declaration;
                                     repeated, consecutively, once per owner
    type             vlen str (P,)   the primitive class (Steel01, Fiber, ...)
    params           vlen str (P,)   one JSON object, field name -> value, in
                                     dataclasses.fields order (see below)
    params_names     vlen str (P,)   a JSON list naming the argv slots of the
                                     declaration's store row, when the argv
                                     equals the fields; "" otherwise
    transf_ref/      index run       every GeomTransf / BeamIntegration /
    integration_ref/                 Section declaration key the row's fields
    section_ref/                     reference, in dataclass field order and
        first  i8 (P,)               depth-first within a field: row i's keys
        count  i8 (P,)               are key[first[i] : first[i] + count[i]]
        key    vlen str (K,)         (a HingeRadau lists section_i, section_j,
                                     section_interior; count 0 = none)
```

The `params` encoding is generic, from `dataclasses.fields(prim)`, with
every field shape either stored or refused at write (never skipped):
`None`, `bool`, `int`, finite `float` and `str` as JSON values; a
referenced primitive as `{"$decl": <key>}`, its `/opensees/decls` key
(the one its own `(kind, tag)` joins to, so a reference is resolved with
`DeclarationTable.by_key`); a tuple or list as a list, nested lists
kept; a mapping with plain `str` keys as an object; a value dataclass a
field holds (a `Fiber` patch, a `ShellLayer`, a `ZeroLengthMatDir`) as
`{"$struct": <class>, "fields": {...}}`; and the listed opaque objects
(`Cartesian`, `Cylindrical`, `Spherical`, `AlongBeam`,
`SectionProperties`) as `{"$opaque": <class>}`, since their lowering is
in the hashed stores. An `ndarray`, a set, an `Enum`, a non-finite
float, a mapping keyed by anything else or an unlisted object raises
`H5DeclParamsError` from `H5Emitter.set_decl_params`, before the emit.

`params_names` (Q4) is derived at write from the stores whose rows carry
one flat argv per tag: `materials`, the one-line sections,
`beam_integration`, `time_series` and `dampings`. The fields are
flattened in order (a scalar one slot named after its field, a sequence
one slot per element as `points[0]`, nested as `m_ij[0][1]`, a
reference its tag, an omitted optional no slot); when the slot count and
every value equal the row's argv, the list names the slots, else `""`.
A `bool` (a flag), a mapping, a struct, an opaque or a reference without
a single tag has no slot form, so a flag without a field (`Parallel
-factors`), an element row (node tags), a complex section (`Fiber`) or a
chain component (no tag) stays unnamed. Every concrete primitive of the
registry that the archive does not name is a line of the ledger
`tests/opensees/contract/params_names_ledger.txt` (`unnamed`: argv is
not the fields; `uncheckable`: no roster sample to emit; `nostore`: a
family without a flat argv), checked by `test_verbs_lock.py` with the
same `decl_argv_names` the writer uses; the `unnamed` + `uncheckable`
count may only shrink, and a primitive neither named nor listed fails.

**`label`: provenance only (opensees 2.28.0; ADR 0115 amendment R2-f).**
The rows of `FixRecord`, `MassRecord` (flat and staged) and of the typed
`recorder.Node` / `recorder.Element` carry a `label` field. A declaration
made with `label=` stores the names it was given as a list (`"left"` is
stored `["left"]`, a list keeps its order) beside the ids the call
resolved them to (`nodes`; for `recorder.Element`, `elements`). A
`recorder.Element` row's `elements` are always **OpenSees element tags**,
whichever way it was declared: a `label=` declaration resolves to FEM ids,
and the archive stores them translated through the build's FEM id → tag
map, exactly as the deck's `-ele` list (identical under
`element_tags="fem"`). So one reader reads every row the same way. A declaration made with `pg=` or `nodes=` / `elements=`
stores **`label: null`** (the generic encoder writes every field, so the
key is present and empty rather than absent). The ids are authoritative:
no reader, replay (`OpenSeesModel.from_h5`, `derive_sequence`) or rewrite
re-resolves a label, so a label that no longer names those ids, or names
nothing, changes nothing that runs. An archive written before 2.28.0 has
no `label` key in these rows; it reads exactly as before.

**`precision` (opensees 2.30.0; R3-S5 #1789, maintainer ruling D7).**
The rows of the typed `recorder.Node` / `recorder.Element` carry a
`precision` field: the significant digits per written value that
`precision=N` asked for, which the deck carries as `-precision N` right
after `-file`. A recorder declared without it stores **`precision: null`**
and its deck line has no flag, byte-identical to an earlier writer's. An
archive written before 2.30.0 has no `precision` key in these rows; it
reads exactly as before.

`H5Model.declarations()` returns the rows as `DeclarationTable.params`
(`decls` row -> a tuple of `DeclParamsRO`, one per owner in call order:
`type`, `params` with `DeclRef` / `DeclStruct` / `DeclOpaque` values,
`params_names`, the three refs and the stored `params_json`);
`params_for(key)` looks a key's tuple up. A
column that differs in length, a `decl` out of range, a
`params` that is not a JSON object, a reference or `*_ref` to a key the
declarations lack, an object with an unknown `$` tag, or a
`params_names` that is not a list of str raises `MalformedH5Error`.

## Parameter tags (opensees 2.27.0)

The `parameter` tags of an initial stress, an absorbing flip and an
`s.update_parameter` are the build's (its tag plan mints them once, ADR
0114 D4); from 2.27.0 each record stores its own, so a deck replay writes
them and allocates none (`OpenSeesModel` INV-5).

```
/opensees/initial_stress/stress_NNN       @param_tags (3,) int64: XX, YY, ZZ ramp tags
/opensees/stages/stage_NNN/
├── initial_stress/stress_NNN             @param_tags (3,) int64, as above
├── activate_absorbing/absorb_NNN/
│     └── param_tags  (k, 2) int64        one (rank, pid) row per flip written
├── update_parameter/update_NNN/          (optional) one per s.update_parameter
│     @name, @value, @mat_tag (optional), @pg XOR elements (n,) int64
│     └── param_tags  (k, 2) int64        one (rank, pid) row per update written
└── zero_velocities/zero_NNN/             (optional) one per s.zero_velocities
      └── nodes  (n,) int64               absent: the whole domain
```

A flip or update record holds one `pid`, written once per rank that holds
its elements (`rank` is `-1` for a flat build), so its `(k, 2)` rows share
the `pid`. The writer attributes each call to its record by tag: the plan
mints the records' tags in record order, so the distinct tags, ascending,
are the records in order. A record whose target holds no element has no
tag and refuses the write. A replay resolves each record's elements on the
whole mesh (the deck replay is flat) and writes them under the stored
`pid`. `s.zero_velocities` replays over the archived node ids at their
`/opensees/nodes_ndf` width. A rewrite (`OpenSeesModel.to_h5`) of a pre-2.27
archive stores the tags that archive's replay mints.

## `/opensees/constraints/embeddedRebar` (optional, opensees 2.29.0)

One row per `g.reinforce` `LadrunoEmbeddedRebar` tie the build wrote
(K2b, #1795; formerly a `ledger` verb). The row holds the tie's planned
element tag and the positional arguments `embedded_rebar_args` built,
the bond material already resolved to its tag. Written only when a tie
exists; hashed.

```
/opensees/constraints/embeddedRebar/
    ele_tag      i8 (N,)          the tie's element tag (the tag plan's)
    name         vlen str (N,)    the mp_constraint_comment label, "" when none
    arg_offsets  i4 (N+1,)        row i owns args[arg_offsets[i]:arg_offsets[i+1]]
    args         f8 (A,)          numeric argument, NaN for a string
    args_str     vlen str (A,)    string argument, "" for a number
    arg_kinds    i1 (A,)          0 int, 1 float, 2 str
```

**Bond material tag (no column).** The group has no bond-material column.
`embedded_rebar_args` builds each row's arguments in this order:
`rebar_node`, the host (`-host ele` or `nHost h1..hN`), the weights
(`-xi ...` or `-shape ...`), `-dir dx dy [dz]`, then the axial law,
either `-perfect kAxial` or `-bond matTag [-bondScale bs]`, then the
optional `-kt`, `-ktAlpha`, `-corot`, `-enforce` and `-bipenalty` tokens.
The bond material tag therefore sits inside `args`, as the element
immediately after the literal token `-bond` (stored as an `arg_kinds` 0
int). A perfect-bond tie has no `-bond` token and so no bond tag. A
reader scans the row's `args_str` for `-bond` and takes exactly the one
argument after it, never a later `-bondScale` value. An explicit
`bond_tag` column is deferred to the next `/opensees` bump that is
already happening; under the V4t drift guard, that bump also moves the
viewer `OPENSEES_TARGET`. No schema bump is made for this note.

The argument encoding is `/opensees/commands`'s, so a replayed line is
byte-identical. A deck replay of a 2.29 archive writes the stored rows
after the fixes, masses, global commands and regions (the bridge's
slot) and mints no tag (`OpenSeesModel` INV-5). A replay of an older
archive re-derives the ties from the neutral `/reinforce_ties`, numbering
them past the highest replayed element tag (the
`pre-2-29-reinforce-tie-replay` tag-law waiver, behind
`EMBEDDED_REBAR_FROM` in `h5_reader.py`). A 2.29 archive whose neutral
zone carries ties that the store lacks, or a different count, refuses to
replay. A rewrite (`OpenSeesModel.to_h5`) of a pre-2.29 archive stores
the ties that archive's replay derives and links their `/opensees/program`
runs to the store. The writer refuses a tie inside a stage bracket and an
argument that is not an `int`, `float` or `str`, at emit.

## `/opensees/constraints/equationConstraint` (optional, opensees 2.35.0)

One row per `equationConstraint` line the bridge's global pass wrote, in
emit order (#2416; every such call was dropped before). Two kinds of line
reach it, and the store holds both: an `enforce="equation"` tie's
(`tie`, `distributing` or `embedded`, one line per tied DOF, ADR 0068)
and an `apeSees.equation_constraint` row. Written only when a line
exists; hashed.

```
/opensees/constraints/equationConstraint/
    cnode        i8 (N,)          the constrained node
    cdof         i4 (N,)          its 1-based DOF
    ccoef        f8 (N,)          its coefficient
    name         vlen str (N,)    the mp_constraint_comment label (a named tie's first line), "" when none
    ret_offsets  i4 (N+1,)        row i owns terms ret_offsets[i]:ret_offsets[i+1], at least one
    ret_node     i8 (M,)          retained node
    ret_dof      i4 (M,)          its 1-based DOF
    ret_coef     f8 (M,)          its coefficient
```

- **Join.** Each row is written by one `equationConstraint` call of
  `/opensees/program` (store `{scope}/constraints/equationConstraint`),
  so a replay writes it at the bridge's emit index, between the other
  MP rows (`compose.py` 9b′). A row no call wrote refuses.
- **Partitions.** A partitioned emit writes an equation tie on every rank
  that holds its slave or a master (ADR 0068 Open item 2); the store
  captures each line once (ADR 0027 INV-1), and its replicas' program
  runs repeat the first capture's row.
- **The user rows.** The bridge writes `apeSees.equation_constraint` rows
  last in its MP pass, so they are the last `K` rows, `K` the
  `equation_constraint` declarations of `/opensees/decls`. Their
  `/opensees/decl_params` records must equal those rows, or the reader
  raises `MalformedH5Error`. From this minor the rows are the replay's
  only source for every line.
- **Stage scope (opensees 2.37.0).** The lines of an equation tie a
  stage claims are rows of that stage block, in the same columns, beside
  a row-aligned stage MP sequence (#2448; a 2.35 or 2.36 writer dropped
  them):

  ```
  /opensees/stages/stage_NNN/constraints/
      equationConstraint/               the columns above, one row per line the stage wrote
      equationConstraint_emit_index     i8 (N,)   the stage's MP emit_index of each row
  ```

  The emit index comes from the stage's one sequence, which also stamps
  `equalDOF_emit_index`, `rigidLink_emit_index`,
  `rigidDiaphragm_emit_index` and `embeddedNode_emit_index`, so the
  replay merges the five kinds back into the bridge's order inside the
  stage. Their `/opensees/program` calls carry the stage's ordinal and the
  same store template. A partitioned stage captures each line once.
  Written only when a stage writes a line; hashed. `equationConstraint`
  is no `ledger` verb any more (ADR 0114, amendment of 2026-10-23).
- **Reader refusals** (`H5Model.equation_constraints` and
  `H5Model.stages`, all `MalformedH5Error`): the global group in a file
  below 2.35.0 (`h5_reader.EQUATION_CONSTRAINT_FROM`), a stage group in a
  file below 2.37.0 (`h5_reader.STAGE_EQUATION_CONSTRAINT_FROM`), a stage
  group without its `equationConstraint_emit_index` or the reverse, an
  emit-index length that differs from the rows, a missing column, columns
  that disagree in length, a row with no retained term, a DOF below 1,
  and a zero or non-finite coefficient.
- **Replay refusal.** An `enforce="equation"` record of the neutral zone
  whose lines are not all among the stored tie rows, global and stage
  (matched as a multiset), refuses with `BridgeError`: every archive below
  2.35.0 (#2416), a stage-claimed tie in an archive below 2.37.0 (#2448),
  and a file that lost rows. A pre-2.37 archive stores nothing a
  stage-claimed tie's lines could be rebuilt from, so it never replays;
  nor does a `to_h5` rewrite of one, although it is stamped 2.37.0. A
  partitioned archive whose stage holds equation rows and repeats a fix
  still in force (not fully released by a `remove_sp` since; the ghost
  replay of a rank that does not own the node, a zero-weight master's
  included) also refuses, since its flat replay would fix the node twice
  (#2530).
- **Rewrite.** `OpenSeesModel.to_h5` keeps the rows. A rewrite of a
  pre-2.35 archive stores its `apeSees.equation_constraint` rows and links
  the source's last `K` global `equationConstraint` calls to them; its
  equation ties' calls stay unlinked, since the source stores no line,
  and the rewritten file still refuses them.

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

## `/sequence`

Version key: `/meta/sequence_schema_version` (current `1.0.0`, floor
`1.0.0`; [ADR 0112](decisions/0112-files-are-the-model-and-a-read-only-app.md)
D2b, V0 decisions 8-10 on #1283). The readable view of the analysis
program: one row per stage, one row per change a stage makes. It is a
projection, not a second registry. `_compose_model_h5` writes it last,
after `/meta/lineage`, by reading the `/opensees` zone it has just
written back through `H5Model`
([`_internal/sequence_h5.py`](../src/apeGmsh/opensees/_internal/sequence_h5.py)).
The deriver imports no record or primitive module; an AST lock in
`tests/opensees/test_sequence_h5.py` holds that. Each table is a group
of equal-length column datasets.

```
/sequence   @derived_from str ("/opensees") · @model_hash str · @staged i1
  /stages   order i4 · name str · analysis · integrator · algorithm · test ·
            system · numberer · handler (str, "" when unset) ·
            integrator_args · algorithm_args · test_args · system_args ·
            handler_args (str, JSON list) · steps i4 · dt f8 · set_time f8 ·
            domain_change i1 · source str
  /changes  stage i4 · op str · what str · name str · count i4 · source str
```

* **Stages.** A staged archive gives one row per
  `/opensees/stages/stage_NNN`, in order, with `@staged` 1. A deck with
  no stage gives one implicit row from `/opensees/analysis`
  (`name` `""`, `@staged` 0). A model-only file (neither) gives an empty
  table. `steps` is `-1` when the archive holds no `analyze_steps` (a
  deck that loops by hand); `dt` is NaN for a static stage or when no
  `analyze_dt` is archived; `set_time` is NaN when the stage does not
  set it. The `*_args` columns are the chain's positional arguments as
  JSON (`[]` when there are none; a non-finite number becomes `null`).
  A partitioned deck's `numberer_runtime_fallback` and
  `system_runtime_fallback` have no column: the token column holds the
  primary, the canonical choice (ADR 0027 INV-5).
* **Changes.** `op` is one of `add`, `remove`, `activate`, `set`;
  `what` is one of `pattern`, `recorder`, `fix`, `mass`, `element`,
  `group`, `region`, `rayleigh`, `initial_stress`, `absorbing`,
  `material_stage`, `constraint`. The pairs are the `seq` column of
  `emitter/verbs.py::VERBS` plus `(activate, group)` (a stage's
  `activated_pgs`). Global declarations (`/opensees/patterns`,
  `recorders`, `regions`, `initial_stress`, `bcs/fix`, `bcs/mass`,
  `constraints/<kind>` and a global `rayleigh` command) are in force from
  the first analysis, so
  their rows carry stage `0`. `name` is the declaration's `name=` from
  `/opensees/decls`, `""` when unnamed or when the file has no `decls`
  (below opensees 2.25.0). A tagless store (`fix`, `mass`, `remove_sp`,
  ...) gives one row per declaration, with `count` its records; a
  pattern, recorder, region, initial stress or absorbing flip gives one
  row per record.
* **Source.** Every row names the archive path it was read from;
  `<path>[i]` is the `i`-th record of that container in reader order.
  Global `bcs/fix` and `bcs/mass` are read through the `decls` row join
  only, so a file below opensees 2.25.0 lists none. Global constraints are
  read through `/opensees/program` (one run per `add constraint` call):
  one row per declaration when `decls` covers `constraints/<kind>`, else
  one unnamed row (a constraint the mesh carried); a ledger constraint
  verb's dropped call, which the archive does not carry, names its
  `program[i]` run. A
  file below opensees 2.23.0 (no program) lists none.
* **Not projected.** The stage verbs whose `seq` is empty in `VERBS`
  (`update_parameter`, `zero_velocities`, `set_creep`, the pre-analyze
  `reset`) have no `what` in the vocabulary and give no row.
* **Integers** follow the integer policy above: a value outside `int32`
  raises `SequenceOverflowError` before anything is written.
* **Writes.** An existing `/sequence` and its key are dropped first, the
  tables are built in memory, written to a scratch group and moved into
  place, and the key is stamped last. A failure leaves neither a stale nor
  a half-written zone, and a rewrite (`OpenSeesModel.to_h5`) re-derives
  the zone. A chain attribute the zone has no column for refuses
  (`SequenceError`).
* No hash reads `/sequence`, and no model reader reads it.

## `results.h5`: the native results file

A native results file is written by `NativeWriter`
([`results/writers/_native.py`](../src/apeGmsh/results/writers/_native.py))
and read by `NativeReader`
([`results/readers/_native.py`](../src/apeGmsh/results/readers/_native.py)).
The path names and attribute keys are constants in
[`results/schema/_native.py`](../src/apeGmsh/results/schema/_native.py).
This section describes the layout those modules write at results zone
**1.2.0** (floor **1.0.0**). It is the contract apeGmshViewer reads
(ADR 0112 D4). It is not a `model.h5`: the version keys are root attrs,
`/meta` holds only the lineage, and the model is embedded under `/model`
rather than written at the root.

Six writers produce the file, and files from a seventh, deleted writer stay readable. `source_type` names the writer, but it is a stamp, not proof of origin:

| `source_type` | Writer | `/model` | `/opensees` | Stages |
|---|---|---|---|---|
| `eigen` | `EigenResult.to_native` (`ops.eigen(n)`, `opensees/analysis/eigen.py`) | yes | yes: the bridge's zone | one `mode` stage per mode |
| `eigen` | `capture_live_modes` (`results/live/_modes.py`), a plain openseespy deck | yes | no: the deck is not the bridge's | one `mode` stage per mode |
| `domain_capture` (legacy) | the in-process `DomainCapture`, deleted in #2191; its files stay readable until the native format retires | yes | yes when the bridge archived a `model.h5` sidecar, else absent | the user's `begin_stage` calls, then one `mode` stage per mode |
| `tcl_recorders` | the recorder transcoder (`Results.from_recorders`) | yes | when `model=` is given | one stage per transcode |
| `parallel_modal` | `ParallelModalResult` harvest (`opensees/analysis/modal.py`) | yes | no | one `mode` stage per mode |
| `footfall_walking` | `FootfallResult` (`opensees/analysis/footfall_result.py`) | no | no | one `static` stage named `footfall` |
| `domain_capture` (the `NativeWriter.open` default) | `make_demo_results` (`results/demo.py`): synthetic nodal displacements, no element groups | yes | yes | one `static` stage named `pushover` |

Because `domain_capture` is also `NativeWriter.open`'s default, that stamp
does not prove that `DomainCapture` wrote the file: the demo writer, or
any caller that keeps the default, stamps it too.

### Layout

```
results.h5
│   @schema_version, @results_schema_version,      root attrs (see below)
│   @neutral_schema_version, @opensees_schema_version,
│   @source_type, @source_path, @created_at, @apegmsh_version,
│   @analysis_label, @opensees_backend, @opensees_build,
│   @element_index_space                     (optional, results 1.2.0)
├── /meta                                  lineage only (always present from results 1.1.0)
│     └── /lineage                         @fem_hash, @model_hash, @results_hash
├── /model                                 (optional) the neutral zone, embedded
│     ├── /meta                            the model's /meta attrs, plus /meta/lineage
│     ├── /nodes, /elements, /physical_groups, …   same layout as a model.h5 root
├── /opensees                              (optional) the bridge zone, copied verbatim
└── /stages
      └── /{stage_id}                      @name, @kind (+ mode attrs)
            ├── time                       (T,) float64 (coerced by the writer)
            └── /partitions
                  └── /{partition_id}
                        ├── /nodes
                        │     ├── _ids                     (N,) int64
                        │     └── {component}              (T, N) float, see below
                        └── /elements
                              ├── _ids                     (reserved; see "What is absent")
                              ├── /gauss_points/{group_id}
                              ├── /line_stations/{group_id}
                              ├── /nodal_forces/{group_id}
                              ├── /fibers/{group_id}
                              └── /layers/{group_id}
```

Every component dataset has the step axis first, and its length equals
the stage's `time` length `T`; the writer refuses any other shape.
**Component dtypes are not guaranteed.** The writer stores a component
array as the caller passed it, with no dtype coercion. Every caller in
`src/` passes float64 today, so the tables below say float64, but that is
a convention: a float32 array round-trips as float32. A reader must accept
any floating dtype and cast on read. The `time` vector, the index
datasets (`_ids`, `_element_index`, …) and the stage attrs are coerced by
the writer to the dtypes stated.
Names that start with `_` are index and metadata datasets. A name with
no `_` prefix is a result component. A reader lists a group's
components by dropping the `_` names.

### Root attributes

All strings are variable-length UTF-8.

| Attribute | Type | Meaning |
|---|---|---|
| `results_schema_version` | string | the results-zone version (`"1.2.0"`). The reader validates it against the results floor before any read ([The floor rule](#the-floor-rule)). |
| `schema_version` | string | the envelope stamp (`SCHEMA_VERSION`, `"1.1.0"` today). The reader falls back to it only when `results_schema_version` is missing. A file with neither is refused as "not a native apeGmsh results.h5". |
| `neutral_schema_version` | string | forwarded from `/model/meta`. Present only when `/model` is embedded. |
| `opensees_schema_version` | string | forwarded from the source `model.h5`. Present only when `/opensees` is embedded. |
| `source_type` | string | the writer, from the table above |
| `source_path` | string | where the data came from: the recorder output directory, or a tag such as `<openseespy>`. Informational only. |
| `created_at` | string | ISO 8601 UTC timestamp |
| `apegmsh_version` | string | the producing apeGmsh version, or `""` if unknown |
| `analysis_label` | string | free text; `""` by default |
| `opensees_backend`, `opensees_build` | string | optional build provenance ([`results.h5` root attrs](#resultsh5-root-attrs-opensees-build-provenance)) |
| `element_index_space` | string | optional, results 1.2.0: the element row ids are provably FEM element ids ([`element_index_space`](#element_index_space)) |

### `element_index_space`

Results 1.2.0 (#1746) adds this optional root attribute. When present, it
says that every element row id in the file (each group's `_element_index`)
is provably a FEM element id, the id `/model/elements` carries. This is
the complete value set. A reader must treat any value not listed here as
unverified:

| Value | Meaning |
|---|---|
| `"fem"` | element row ids are provably FEM element ids. It is the only value any writer emits (`ELEMENT_INDEX_SPACES` in `results/schema/_native.py`; `NativeWriter.stamp_element_index_space` refuses any other) |

**Absent** means unverified: the file does not prove which ids its element
rows carry. Every 1.0.0 and 1.1.0 file lacks the attribute, and so does a
1.2.0 file whose writer could not prove it. Absence is not a refusal: the
file reads exactly as before. The version number alone never makes a file
verified, and neither does `source_type`: the rule is the same for every
writer, `domain_capture` included. A file is verified if and only if its
root attribute `element_index_space` is `"fem"`.

The on-disk contract is this one root attribute, `element_index_space`:
its value set is exactly `"fem"`, and absence means unverified. No other
attribute carries the verdict. apeGmsh's Python reader derives a
convenience property from it, `NativeReader.element_index_verified`
(true only when the attribute is `"fem"` in the form below), and exposes
the raw value as `NativeReader.element_index_space`. Both are Python
properties, not HDF5 attributes: no file holds an attribute named
`element_index_verified`.

**The rule (#1811): a file is verified if and only if `element_index_space`
is a variable-length UTF-8 string exactly equal to `"fem"`; a
variable-length bytes (ASCII) `"fem"`, a fixed-length string or bytes
`"fem"`, and any non-string value are not verified.**

**Its on-disk form.** The attribute is a **scalar,
variable-length, UTF-8 string**: an HDF5 string datatype of variable size
with character set `H5T_CSET_UTF8` (h5py `string_dtype("utf-8")`, what
h5py writes for a Python `str`), in a scalar dataspace, holding exactly
the three characters `fem`. Every writer emits that form. Any other form
reads as unverified, even when its text is `fem`:

- a fixed-length string, NUL- or space-padded or not. Some HDF5 readers
  strip the padding and would compare it equal to `"fem"`; that is not
  the stamp;
- an ASCII-charset string, fixed or variable length;
- an opaque or bytes value;
- an array of one or more strings;
- any non-string dtype.

A reader in another language must check the datatype (string,
variable-length, UTF-8, scalar) before it compares the text.

`Results.from_recorders` caches transcoded files under a key that includes
a digest of `model=` (its `lineage.fem_hash`, its `lineage.model_hash` and
its `(fem_eid, tag)` pairing). A cached file, its stamp and its embedded `/opensees` are
therefore never served to a call with another model.

Who writes it:

| Writer (`source_type`) | Writes `element_index_space`? |
|---|---|
| recorder transcoder (`tcl_recorders`) | `"fem"` when it can prove it, whatever apeGmsh version emitted the recorders; otherwise absent (below) |
| `DomainCapture` (`domain_capture`, legacy files; the writer was deleted in #2191) | `"fem"` on every capture from #1764 on. An element-scoped capture refused unless a `{fem_eid: ops_tag}` pairing covered every recorded id ([Domain capture eras](#domain-capture-eras)) |
| `EigenResult.to_native`, `capture_live_modes` (`eigen`) | never: they write nodal rows only |
| `make_demo_results` (`domain_capture`) | `"fem"`, from #1764 on. It writes nodal rows only, in every version |
| `ParallelModalResult` (`parallel_modal`) | never: absent |
| `FootfallResult` (`footfall_walking`) | never: absent |
| `NativeWriter` itself | only when a caller invokes `stamp_element_index_space`; the transcoder and `make_demo_results` are the callers in `src/` |
| MPCO (`Results.from_mpco`) | writes no `results.h5` at all: it reads the `.mpco` file directly |

The transcoder labels element rows with the recorder spec's FEM ids. Those
are the recorded elements only if the emit put each element's OpenSees tag
in the `-ele` list, and a `.out` file does not record which emit wrote it
(before #1740 the emits wrote the FEM ids unchanged). The transcoder therefore
stamps `"fem"` only when all of these hold:

1. `model=` was given, so the model's `/opensees` zone is embedded;
2. that model's `lineage.fem_hash` equals the spec's `fem_snapshot_id`;
3. either no record yields element rows (node ids are FEM ids, because the
   bridge emits every node under its FEM id), or every element id that a
   `gauss`, `line_stations` or `elements` record names is its own OpenSees
   tag in the model's `(fem_eid, tag)` pairing. Then an emit that
   translated the ids and one that did not wrote the same `-ele` list.

What the stamp does **not** certify: that the recorder files came from a
deck built from the embedded model, or that the deck recorded what the spec
names. It certifies the index space only, relative to the embedded
`/opensees`. Element records under `element_tags="sequential"` (the bridge
default), whose FEM ids differ from their tags, stay unverified even when a
translating emit wrote them, until an emit-side manifest can prove the
translation.

#### Domain capture eras

A domain capture asks OpenSees for each element by ops tag and labels the
row with the record's `fem_eid`. That row is the right element only if the
capture translated `fem_eid` to the ops tag first. Under the bridge's
default `element_tags="sequential"`, and on any mesh where gmsh numbers
lower-dimensional elements first, the two differ. **Unstamped
`domain_capture` files are not grandfathered: they read unverified**,
because the writer did not translate in every era:

| Era (commits) | Element-id translation | Element rows of an unstamped file |
|---|---|---|
| `483633df` → `fbf68c11` (2026-04-26 → 2026-05-29) | none: the capture queried OpenSees with the `fem_eid` | wrong element, or dropped, wherever `fem_eid` ≠ ops tag |
| `fbf68c11` → `2d159362` (→ 2026-08-04) | only for a composed model | wrong for an uncomposed model whose `fem_eid`s are offset from its ops tags |
| `2d159362` → #1764 | whenever a pairing was read, but silently none when it was not (no bridge or model, a sidecar failure, any read error), and none for a whole record the pairing only partly covered | wrong on those paths |
| #1764 on | refuses an element-scoped capture without a covering pairing; stamps `"fem"` | stamped, verified |

`2d159362`'s own note: the translator was gated on compose provenance,
"assuming an uncomposed model has ops tag == fem_eid by allocator
construction", which is false whenever gmsh numbers lower-dimensional
elements first. Every Gauss value was then attributed to a different
element, and elements whose `fem_eid` exceeded the largest ops tag were
dropped from filtered reads (it reproduced the fault on a 65,798-element
tet10 model offset by 683). On b23c182c a 2-D portal under sequential tags
(pairing `{5: 3, 4: 4, 6: 5}`) captured without a pairing wrote fem 5's row
as Fy = 6257.8, tag 5's response (column fem 6); fem 5's own value is
−6257.8.

A file whose model had `element_tags="fem"` and no offset mesh is correct
in every era, but nothing in an unstamped file proves that, so a reader
cannot tell it apart.

### The embedded model, and how to pair a file with its `model.h5`

`/model` is the neutral zone that `fem.to_native_h5` writes. It has the
same groups and the same `/meta` attrs as the root of a `model.h5`
(`/model/nodes/ids`, `/model/nodes/coords`, `/model/elements/{alias}`, and
so on), at the neutral version stamped in `/model/meta`. The writer
overwrites `/model/meta/ndm`, `ndf`, `model_name` and
`opensees_schema_version` with the source `model.h5`'s values, and copies
its `/meta/lineage` into `/model/meta/lineage`. `/opensees` is that
`model.h5`'s `/opensees` group, copied byte for byte. The copy keeps
`/opensees/cuts` and `/opensees/sweeps`.

`/meta/lineage` carries the ADR 0021 hash chain. Each value is a
32-character hex string:

| Attr | Present when | Value |
|---|---|---|
| `fem_hash` | `/model` is embedded | the neutral zone's content hash, equal to `/model/meta/snapshot_id` |
| `model_hash` | `/opensees` is embedded | `blake2b(fem_hash ‖ /opensees)`. The hash skips `cuts`, `sweeps`, `regions`, `names`, `computed_sections`, `decls` and `decl_params`, and the attribute `/opensees@target_mode`. |
| `results_hash` | always; stamped at close | `blake2b(model_hash ‖ /stages)`, chained on `""` when `model_hash` is absent |

**Pairing rule.** A results file belongs to a given `model.h5` when its
`/meta/lineage/model_hash` equals that `model.h5`'s
`/meta/lineage/model_hash`. When the results file has no `/opensees`, it
belongs when its `fem_hash` equals the `model.h5`'s `fem_hash`. That
matches the mesh only, not the OpenSees model. A file with no `fem_hash`
(no `/model`, or results 1.0.0, which predates `/meta/lineage`) cannot be
paired by content. Such a 1.1.0 file still has `/meta/lineage`, holding
`results_hash` alone.

`/model/meta/session_id`, when present, equals the session id of the
`model.h5` the run was built from. It is a weaker signal: it identifies
the session, not the content
([`/meta/session_id`](#metasession_id-and-the-geometry-sibling)).
Do not pair on `/model/meta/model_name`. A domain capture stamps the
sidecar's stem (`run.model` for `run.h5`), not the session's model name.

The stored hashes are the writer's claims. Recomputing `fem_hash` needs
apeGmsh, because it rebuilds the snapshot. `Results.lineage` recomputes
all three and warns on drift; it never raises (ADR 0021 INV-2).

An embedded zone below its floor does not refuse the file. `/stages`
still opens, read-only, and only the model zone is unavailable. An
embedded zone newer than the reader refuses the whole file
([The floor rule](#the-floor-rule), ADR 0113 D9).

### Stages and the time vector

`/stages/{stage_id}` holds one analysis stage. Stage ids are arbitrary
strings. The writers name them `stage_0`, `stage_1`, … in write order, and
readers order `stage_<int>` ids by their integer suffix (HDF5 lists
`stage_10` before `stage_2`), then any other ids in lexical order.

| Stage attr | Type | Meaning |
|---|---|---|
| `name` | string | the user's stage name (`begin_stage(name=)`), or `mode_<k>` for a mode stage |
| `kind` | string | `"static"`, `"transient"` or `"mode"`; the writer refuses anything else |
| `eigenvalue` | float64 | `mode` only: λ = ω² |
| `frequency_hz` | float64 | `mode` only: √λ / 2π; `0.0` when λ ≤ 0 |
| `period_s` | float64 | `mode` only: 2π / √λ; `0.0` when λ ≤ 0 |
| `mode_index` | int64 | `mode` only, optional: the 1-based mode number |

`time` is `(T,)` float64: the OpenSees domain time at each recorded step,
in the order recorded. For a static stage that is the load factor
(`getTime()` under `LoadControl`). A `mode` stage has `T = 1` and
`time = [0.0]`, and its nodal `displacement_*` / `rotation_*` are
eigenvector components, not displacements. Step `k` of every component in
a stage is `time[k]`.

### Partitions

`/stages/{sid}/partitions/{pid}` splits a stage's rows by partition. Every
writer in `src/` writes exactly one partition, `partition_0`, and native
results carry no `PARTITION_REDUCTION` rule (ADR 0123 makes one
partition per stage the contract). A file written with several partitions
through the public `NativeWriter` API still opens in the Python reader:

- a node id repeated within one partition's `nodes/_ids` refuses at open
  (`NativeRepeatedNodeId`);
- a node several partitions hold reads once when its copies are bit-for-bit
  equal over the stage; if any node's copies of a component differ, every
  read of that component raises `NativePartitionCopiesDisagree` (#2360),
  because a per-partition partial value cannot be told from a replicated
  one;
- element rows are concatenated in partition-name order.

The viewer refuses a stage that does not hold exactly one partition
(`unsupported`). Partitioned runs reach a reader as `.ladruno` or MPCO part
sets, which carry `PARTITION_REDUCTION`.

### Nodal results: `nodes/`

| Dataset | Shape, dtype | Meaning |
|---|---|---|
| `_ids` | `(N,)` int64 | node ids. They are the ids of `/model/nodes/ids` (the same ids `model.h5` carries). They are ascending when a stage merges several nodal records; otherwise they keep the record's order, so do not assume they are sorted |
| `{component}` | `(T, N)` float64 | column `j` is node `_ids[j]` |

**Example.** To read the nodal displacement of node `n` at step `k` of
stage `s`: open `/stages/s/partitions/partition_0/nodes`, find `j` with
`_ids[j] == n`, and read `displacement_x[k, j]`, `displacement_y[k, j]`
and, in 3-D, `displacement_z[k, j]`.

A component is present only if it was recorded. A shorthand request
such as `"displacement"` is clipped to the model's `ndm` and `ndf`, so a
2-D file normally has no `displacement_z`, and a file has `rotation_*`
only when the node DOFs include rotations. When a stage merges several nodal records, the
id list is their union, and a node that a record did not cover is `NaN`
in that record's components.

### Element results: the five categories

Element results live in **groups**. Each group holds one homogeneous
bucket: one element class and one integration layout. `group_id` is an
arbitrary string; the writers use `<record name>_<class name>_<i>`, for
example `wall_stress_FourNodeQuad_0`. A component can appear in several
groups. The reader concatenates its rows across groups and partitions.

**Element ids.** `_element_index` holds **element ids**, not row
positions. Every writer means them as the ids of `/model/elements`
(`fem_eid`), but only the root attribute `element_index_space = "fem"`
proves it.

> **Verified only by the stamp.** A file's element mapping is verified if
> and only if its root attribute `element_index_space` is `"fem"`
> ([`element_index_space`](#element_index_space)). The rule is the same for
> every `source_type`; no `source_type` is an exception, `domain_capture`
> included ([Domain capture eras](#domain-capture-eras)). For any other
> file, a reader must show an "unverified" state for its element results,
> and it must never guess a join between `_element_index` and
> `/model/elements`.

| Category | Group attrs | Index datasets | Component shape |
|---|---|---|---|
| `gauss_points` | `class_tag` int64 (OpenSees classTag), `int_rule` int64, `custom_rule_idx` int64 | `_element_index` `(E,)` int64; `_natural_coords` `(G, dim)` float64; `_local_axes_quaternion` `(E, 4)` float64, shells only, optional | `(T, E, G)` float64: `[k, e, g]` is element `_element_index[e]` at Gauss point `g` |
| `line_stations` | `class_tag`, `int_rule` int64 | `_element_index` `(E,)` int64; `_station_natural_coord` `(S,)` float64, ξ ∈ [-1, +1] | `(T, E, S)` float64: `[k, e, s]` is element `e` at station `s` |
| `nodal_forces` | `class_tag` int64, `frame` string (`"global"` or `"local"`) | `_element_index` `(E,)` int64 | `(T, E, n)` float64: `[k, e, a]` is element `e` at its `a`-th node, in connectivity order |
| `fibers` | `section_tag` int64, `section_class` string (`-1` and `"(domain_capture)"` from capture) | one row per fiber, each `(F,)`: `_element_index` int64, `_gp_index` int64 (0-based section), `_y`, `_z`, `_area` float64, `_material_tag` int64, `_station_natural_coord` float64 (optional; `NaN` where unknown) | `(T, F)` float64: column `f` is the fiber described by row `f` of the index datasets |
| `layers` | none | one row per (element, surface Gauss point, layer, sub-point), each `(L,)`: `_element_index`, `_gp_index`, `_layer_index`, `_sub_gp_index` int64 (all 0-based), `_thickness` float64; `_local_axes_quaternion` `(L, 4)` float64 | `(T, L)` float64 |

* `_natural_coords` row `g` is Gauss point `g` in the parent element's
  natural coordinates (`dim` is the parent dimension; a truss has
  `(1, 1)`). The rows are in the element's own Gauss-point order.
* Quaternions are scalar-first `(w, x, y, z)` and rotate global to local.
* `nodal_forces` components are named `nodal_resisting_force_{x,y,z}` and
  `nodal_resisting_moment_{x,y,z}`, with `_local_` before the axis when
  `frame` is `"local"`.

**Example.** To read element `e`'s stress components at step `k` of stage
`s`: for each group under
`/stages/s/partitions/partition_0/elements/gauss_points/`, find `i` with
`_element_index[i] == e`. `stress_xx[k, i, :]` then holds one value per
Gauss point, at the points `_natural_coords[g]`. The other `stress_*`
components follow the same pattern.

### Component names and units

Component names are the canonical vocabulary in
[`src/apeGmsh/_vocabulary.py`](../src/apeGmsh/_vocabulary.py), for
example `displacement_x`, `rotation_z`, `reaction_force_y`,
`stress_xx`, `stress_xy`, `axial_force`, `bending_moment_z`,
`membrane_force_xx` and `fiber_stress`. Tensors use Voigt suffixes
`xx yy zz xy yz xz`, and shear strains are engineering strains. A
plane-stress element stores only `stress_xx`, `stress_yy` and
`stress_xy`. A writer outside the recording path may use names that are
not canonical: footfall writes `footfall_ap`, `footfall_ratio` and
`footfall_fdom`. A reader must take component names from the file, not
from a fixed list. Derived scalars such as `von_mises_stress` and the
principal stresses are computed on read and are never stored.

The file stores **no units**. Values are in the model's own consistent
unit system, the one its coordinates and material constants were given
in. Nothing in `results.h5` or `/model` records that system.

### What is absent

* **No springs.** The native writer has no spring path. A reader asked
  for springs returns an empty result. Spring data exists only in MPCO
  and `.ladruno` files.
* **No model change across steps or stages.** `/model` and `/opensees`
  are written once, at open. A stage does not record which elements were
  active. The stage `name` is the only link to a staged build's program
  stages, and the viewer pairs the two by name
  (`viewers/data/_stage_activation.py::pair_capture_to_program`). That
  pairing is a viewer convention, not a stored reference.
* **No element-level `_ids`.** `NativeWriter.write_element_ids` can write
  `elements/_ids`, but no writer in `src/` calls it and no reader reads
  it. Take element ids from each group's `_element_index`.
* **No `/model`** in a footfall file, so its `/meta/lineage` holds only
  `results_hash` (chained on `""`). There is no `model_hash` when
  `/opensees` is absent.
* **Results 1.0.0 files** (the floor) have no `/meta/lineage`: 1.1.0
  added only that sub-group (ADR 0021). The `results_schema_version` key
  itself arrived with the embedded `/opensees` zone (ADRs 0020, 0023), so a
  1.0.0 file may carry `/opensees`. A file that predates the key carries
  only the envelope `schema_version` and no `/opensees`. Otherwise all of
  them have the same `/stages` layout.
* **No `element_index_space`** in 1.0.0 and 1.1.0 files, nor in a 1.2.0
  file whose writer could not prove its element ids
  ([`element_index_space`](#element_index_space)). Such a file reads as
  unverified.

## Cross-references

Every reference uses an HDF5 path string. Examples:

| Reference attribute | Example value |
|---|---|
| `material_ref` | `/opensees/materials/uniaxial/Steel_S420` |
| `section_ref` | `/opensees/sections/Cols` |
| `transf_ref` | `/opensees/transforms/Cols` |
| `series_ref` | `/opensees/time_series/Path_1` |

Readers MUST resolve via `h5py.File["{ref}"]` and validate the
returned group's `type` attribute matches expectations.

## Compound dataset conventions

For variable-length string fields (`material_ref`, `target`, `kind`),
use HDF5 variable-length string type
(`h5py.string_dtype(encoding="utf-8")`).

For padded float arrays (`forces` in `/opensees/patterns/{name}/loads`,
`coords` / `line` in `/opensees/sections/{name}`), pad with `nan` to a
fixed length (`ndf` for forces, 8 for patch `coords`, 6 for layer
`line`). Use `np.dtype([...])` compound types.

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
| opensees (bridge) | `opensees_schema_version` | `/opensees/*` | [`opensees/emitter/h5.py`](../src/apeGmsh/opensees/emitter/h5.py) `SCHEMA_VERSION` | **2.37.0** | **2.12.0** |
| results | `results_schema_version` | `/stages/*` (composed `results.h5`, at file root; layout in [`results.h5`](#resultsh5-the-native-results-file)) | [`results/schema/_versions.py`](../src/apeGmsh/results/schema/_versions.py) `RESULTS_SCHEMA_VERSION` | **1.2.0** | **1.0.0** |
| cuts (sub-zone of opensees) | — (no own key; rides the opensees zone) | `/opensees/cuts`, `/opensees/sweeps` | [`cuts/_h5_io.py`](../src/apeGmsh/cuts/_h5_io.py) `V4_SCHEMA_VERSION` | 2.5.0 | none of its own: it rides the opensees floor |
| geometry (ADR 0112 D2) | `geometry_schema_version` | `/geometry` (sibling `<stem>.geometry.h5` only) | [`opensees/_internal/schema_version.py`](../src/apeGmsh/opensees/_internal/schema_version.py) `GEOMETRY_SCHEMA_VERSION` | **1.0.0** | **1.0.0** |
| provenance (ADR 0112 D3) | `provenance_schema_version` | `/provenance` | [`opensees/_internal/schema_version.py`](../src/apeGmsh/opensees/_internal/schema_version.py) `PROVENANCE_SCHEMA_VERSION` | **1.1.0** | **1.0.0** |
| assembly (ADR 0117 D5) | `assembly_schema_version` | `/assembly` (`Assembly.h5` archives only) | [`opensees/_internal/schema_version.py`](../src/apeGmsh/opensees/_internal/schema_version.py) `ASSEMBLY_SCHEMA_VERSION` | **1.0.0** | **1.0.0** |
| sequence (ADR 0112 D2b) | `sequence_schema_version` | `/sequence` | [`opensees/_internal/schema_version.py`](../src/apeGmsh/opensees/_internal/schema_version.py) `SEQUENCE_SCHEMA_VERSION` | **1.0.0** | **1.0.0** |

> The geometry, provenance, assembly and sequence keys never fall back to the legacy
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
- `2.24.0` — ADR 0114 D6/R4 (K1-5, #1462): additive — five optional
  attributes on `/opensees` itself, `@will_solve`, `@solve_mode`,
  `@solve_refusals`, `@solve_refusals_flat` and `@requires` (see
  [the solve stamp](#opensees-attributes-the-solve-stamp-optional-opensees-2240)),
  written together on every `apeSees.h5` emit; a file without them
  reads as "no stamp" (`None`). Attributes of `/opensees` fold into
  `model_hash`. Additive minor (a 2.23.x reader refuses a 2.24.x file,
  INV-4).
- `2.25.0` — ADR 0114 R5 (K1-6, #1463): additive — new optional
  [`/opensees/decls`](#opensees-decls-optional-opensees-2250), the
  bridge's declarations joined on the tag plan's `(kind, tag)` and to
  the rows of the tagless stores, carrying the new `name=` of `fix` /
  `mass` and the recorders' forwarded one. Hash-excluded
  (`MODEL_HASH_EXCLUDED_CHILDREN`), so `model_hash` is unchanged for
  every model. Additive minor (a 2.24.x reader refuses a 2.25.x file,
  INV-4).
- `2.26.0` — ADR 0114 A6/Q4 (K1-7, #1464): additive — new optional
  [`/opensees/decl_params`](#opensees-decl_params-optional-opensees-2260),
  every registered primitive's parameters by field name (references as
  declaration keys), with `params_names` where the argv equals the
  fields and the `transf_ref` / `integration_ref` / `section_ref`
  columns. Hash-excluded (`MODEL_HASH_EXCLUDED_CHILDREN`), so
  `model_hash` is unchanged for every model. Additive minor (a 2.25.x
  reader refuses a 2.26.x file, INV-4).
- `2.27.0` — K1-8 (#1465; ADR 0114 D4 amended, the T ruling on #1465
  option A): additive — the archive stores every
  [parameter tag](#parameter-tags-opensees-2270) the build planned, per
  record, and the two stage verbs the writer refused before archive
  (`s.update_parameter`, `s.zero_velocities`). A replay of a 2.27 archive
  reads the tags and mints none; a replay of an older one keeps re-minting
  them (the pre-2.27 tag-law waivers, behind `PARAM_TAGS_FROM` in
  `h5_reader.py`). The tags fold into `model_hash`, so an initial-stress or
  absorbing archive hashes differently once at this minor; every other
  file hashes as before. Additive minor (a 2.26.x reader refuses a 2.27.x
  file, INV-4).
- `2.28.0` — ADR 0115 amendment R2-f (#1735; the maintainer's option (a)
  on #1661): additive — the [`decl_params`](#opensees-decl_params-optional-opensees-2260)
  rows of fix, mass and typed Node / Element recorder declarations gain a
  `label` field, the names a `label=` declaration was given (`null` for
  `pg=` or explicit ids). Provenance only: the ids stay authoritative and
  replay never re-resolves a label. No path or dtype changes, and
  `decl_params` is hash-excluded. Additive minor (a 2.27.x reader refuses
  a 2.28.x file, INV-4).
- `2.29.0` — K2b (#1795; ADR 0114 Q3, the K1-8 option-A precedent): additive
  — new optional
  [`/opensees/constraints/embeddedRebar`](#opensees-constraints-embeddedrebar-optional-opensees-2290),
  every `g.reinforce` tie the build wrote (its tag and arguments); the
  `embedded_rebar` verb leaves the ledger. A replay of a 2.29 archive
  writes the stored ties and mints none; a replay of an older one keeps
  re-deriving them from the neutral zone (the pre-2.29 tag-law waiver,
  behind `EMBEDDED_REBAR_FROM` in `h5_reader.py`). The store folds into
  `model_hash`, so a reinforced archive hashes differently once at this
  minor; every other file hashes as before. Additive minor (a 2.28.x
  reader refuses a 2.29.x file, INV-4).
- `2.30.0` — R3-S5 (#1789; maintainer ruling D7 on #1372, the R2-f
  precedent): additive — the
  [`decl_params`](#opensees-decl_params-optional-opensees-2260) rows of
  typed Node / Element recorder declarations gain a `precision` field
  (`null` when unset; the deck then carries no `-precision`). No path or
  dtype changes, and `decl_params` is hash-excluded, so every file hashes
  as before. Additive minor (a 2.29.x reader refuses a 2.30.x file,
  INV-4).
- `2.31.0` — A7 (#1839; the maintainer's ruling of 2026-10-11 on #1646
  item 1): the [`/opensees/program`](#opensees-program-opensees-2230)
  `decl` column, `-1` on every run until now, holds the `/opensees/decls`
  row of the declaration that made each call when the emit proves it,
  and an explicit, documented `-1` otherwise; runs also break where it
  changes. No path or dtype changes. The column is hashed, so every
  archive the bridge writes hashes differently once at this minor; an
  untouched older file keeps (and recomputes to) its stored hash, and a
  rewrite echoes its source's column. A hash-excluded
  `/opensees/decls@program_decl` marker (`"complete"` | `"absent"`) says
  whether the joins are recorded, so a legacy rewrite, stamped 2.31, still
  reads as "not recorded". The reader refuses a `decl` outside
  `{-1} ∪ [0, n_decls)`. Additive minor (a 2.30.x reader refuses a 2.31.x
  file, INV-4).
- `2.32.0` — #1853 (chain K; the maintainer's ruling of 2026-10-13): the
  six side-channel verbs (`step_hook_ramp`, `addToParameter`,
  `flip_element_stage`, `update_parameter`, `set_node_vel`,
  `set_node_accel`) carry the `/opensees/program` `decl` of the record
  that drove them, opened around each record's calls at emit; `-1` below
  this minor. No path or dtype changes, and the `program_decl` marker
  keeps its meaning. The column is hashed, so an archive with any of
  these calls hashes differently once at this minor; any other archive is
  unchanged. Additive minor (a 2.31.x reader refuses a 2.32.x file,
  INV-4).
- `2.33.0` — #2372 (chain K; the maintainer's ruling of 2026-10-21 on
  #2220): `/opensees@requires` gains two value tokens the stamp writer
  derives from argument values: `mpco` (any `recorder mpco`) and `mpi`
  (a Mumps-family or `Parallel*` `system` without a runtime fallback, or
  `@solve_mode` `partitioned`; a flat replay keeps `mpi` only for the
  system); see
  [the solve stamp](#opensees-attributes-the-solve-stamp-optional-opensees-2240).
  No path or dtype changes; below this minor `@requires` names verb
  tokens only and keeps that meaning on read. `@requires` is hashed, so
  an archive with an MPCO recorder, a Mumps system or partitions hashes
  differently once at this minor; any other archive is unchanged.
  Additive minor (a 2.32.x reader refuses a 2.33.x file, INV-4).
- `2.34.0` — #2404 (chain K, slice #2418; the maintainer's ruling (i) of
  2026-10-21): a `params` / `params_str` pair, or a declared recorder's
  `ids`, that HDF5 refuses as an attribute for size is stored as
  uncompressed datasets of the same names in its group; see
  [large arrays as datasets](#large-arrays-as-datasets-opensees-2340).
  The writer failed on such an archive before. Any other oversized
  sequence attribute refuses by name. Every array that fitted keeps the
  attribute form, so every archive a 2.33 writer could write keeps its
  bytes and hash. A 2.33 reader would read a dataset-form pair as no
  arguments, hence the minor. Additive minor (a 2.33.x reader refuses a
  2.34.x file, INV-4).
- `2.35.0` — #2416 (chain K, slice #2447; the maintainer's ruling of
  2026-10-22): a new optional
  [`/opensees/constraints/equationConstraint`](#opensees-constraints-equationconstraint-optional-opensees-2350)
  group holds every global `equationConstraint` line, an
  `enforce="equation"` tie's and an `apeSees.equation_constraint` row's,
  so a replay writes an equation-tied model tied. Below this minor the
  replay refuses such a tie by name. A stage-claimed tie's lines stay
  unstored in this minor (#2448). The group is written only when a line
  exists, so every other archive keeps its bytes and hash; an archive
  with one hashes differently once. A 2.34 reader would replay the model
  untied, hence the minor. Additive minor (a 2.34.x reader refuses a
  2.35.x file, INV-4).
- `2.36.0` — #2278 (chain K, slice #2375; the maintainer's ruling A of
  2026-10-21): the optional attribute
  [`/opensees@target_mode`](#openseestarget_mode-optional-opensees-2360)
  keeps a pinned `OpenSeesTarget(mode="fork" | "stock")`, so a replay
  deck stamps its backend as the bridge's did and a fork pin survives the
  round trip. Written only for a pinned target, so every unpinned archive
  keeps its bytes; absent reads as `"auto"`. It is not hashed: the same
  model pinned or on auto has one `model_hash`. A 2.35 reader would replay
  a fork pin as auto, hence the minor. Additive minor (a 2.35.x reader
  refuses a 2.36.x file, INV-4).
- `2.37.0` — #2448 (chain K; the maintainer's ruling (a) on #2447, the
  ADR 0114 amendment of 2026-10-23 that reverses #1341 Q3 for
  `equationConstraint`): a stage block's `constraints` group gains
  [`equationConstraint`](#opensees-constraints-equationconstraint-optional-opensees-2350)
  and `equationConstraint_emit_index`, the lines of every
  `enforce="equation"` tie the stage claims, so a replay writes them in
  the stage's own position. `equationConstraint` leaves the `VERBS`
  ledger. Below this minor the replay refuses a stage-claimed tie by
  name. The group is written only when a stage writes a line, so every
  other archive keeps its bytes and hash. A 2.36 reader refuses the stage
  group by name, hence the minor. Additive minor (a 2.36.x reader refuses
  a 2.37.x file, INV-4).

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
    ├── /time_series/Path_1/
    │   └── attrs: type="Path", tag=1,
    │              params=[nan, nan, nan, 0.01, nan, 9.81],
    │              params_str=["-filePath", "elcentro.txt", "-dt", "", "-factor", ""]
    ├── /patterns/UniformExcitation_1/
    │   └── attrs: type="UniformExcitation", tag=1, direction=1,
    │              series_ref="/opensees/time_series/Path_1",
    │              params=[1.0, nan, 1.0], params_str=["", "-accel", ""]
    └── /bcs/fix
        target_kind=["node"], target=["1"], dofs=[[1,1,1,1,1,1]]
```

This file is ~50 KB and tells the viewer everything it needs to
draw the column with its section, materials, orientation, and
ground motion — without reading a single OpenSees recorder output.
