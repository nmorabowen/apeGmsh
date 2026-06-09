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
   `opensees_schema_version` / `results_schema_version`) and refuse
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
  pre-solver model declarations: `/meta`, `/nodes`,
  `/elements/{type}`, `/physical_groups`, `/labels`,
  `/constraints/{kind}`, `/loads/{kind}/{pattern}`, `/masses`.
  These describe the model independent of which solver consumes it.
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
├── /mesh_selections
│     └── /{name}                          one group per post-mesh selection set
├── /constraints
│     └── /{kind}                          one dataset per constraint kind
├── /loads
│     ├── /nodal/{pattern}                 one dataset per pattern
│     ├── /element/{pattern}               one dataset per pattern
│     └── /sp/default                      single-point constraints
├── /masses                                single dataset
│
└── /opensees/                             ── OpenSees zone (bridge-owned) ──
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
      └── /analysis                        attrs + sub-attrs (optional)
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
| `neutral_schema_version` | string | per-zone version of the broker neutral zone (e.g. `"2.10.0"`) |
| `opensees_schema_version` | string | per-zone version of the `/opensees/` zone (e.g. `"2.12.0"`); forward-stamped even on broker-only files |
| `apeGmsh_version` | string | producing apeGmsh version |
| `created_iso` | string | ISO 8601 timestamp |
| `ndm` | int | spatial dimension |
| `ndf` | int | DOFs per node |
| `snapshot_id` | string | hash of FEMData snapshot the bridge was built from |
| `model_name` | string | user-provided model name |
| `tag_span_max` | int | `max(max_node,max_elem) - min(min_node,min_elem) + 1`; sizes compose tag-offset reservations (ADR 0038) |

> The `/meta/lineage/` sub-group (ADR 0021) is additionally stamped by
> the composer; the composed `results.h5` carries its version keys at
> the **file root** (not `/meta`) — see [Schema versioning](#versioning).

Schema versioning is **per-zone**, **strict on major**, **bounded on
minor** via the two-version reader window
([ADR 0023](decisions/0023-per-zone-schema-versioning.md)). A reader
written for `X.Y.*` accepts `X.Y.*` and `X.(Y-1).*` and refuses
older minors, newer minors, or any other major. The window applies
to **additive-only** changes; layout-perturbing minor bumps (such
as the B2 2.10 split of `/physical_groups/` and `/labels/`) walk
the window forward and the prior minor falls out of reach — see
[ADR 0023's 2026-05-28 amendment](decisions/0023-per-zone-schema-versioning.md#amendment--2026-05-28--window-applies-to-additive-only-changes-b2--pr-398).

## `/nodes`

Neutral-zone group at the root, broker-owned.

```
/nodes/
├── ids               (N,) int64
└── coords            (N, 3) float64
```

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
└── connectivity      (E_t, 4) int64              — node tags per element

/elements/line2/
├── attrs: code=1, gmsh_name="Line 2", npe=2, dim=1, order=1
├── ids
└── connectivity      (E_l, 2) int64
```

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
and [project memory `project_h5_schema_2_10_b2_shipped`](../../../README.md)).
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
[`mesh/_record_h5.py`](../../mesh/_record_h5.py); the writer in
[`mesh/_femdata_h5_io.py`](../../mesh/_femdata_h5_io.py) bins
records by `kind` and dispatches to the right dtype based on the
record class.

## `/loads/{kind}/{pattern}`

Per-pattern, per-kind datasets sharing the symmetric outer compound.

* `/loads/nodal/{pattern}` — `NodalLoadRecord` rows.  Payload:
  `node_id`, `force_xyz` (3,)f64, `moment_xyz` (3,)f64.  Absent
  force / moment components NaN-filled.
* `/loads/element/{pattern}` — `ElementLoadRecord` rows.  Payload:
  `element_id`, `load_type` (utf-8), `params_json` (utf-8 JSON
  blob — element-load `*args` shape is too freeform for a fixed
  typed compound).
* `/loads/sp/default` — `SPRecord` rows (single-point constraints).
  Payload: `node_id`, `dof`, `value`, `is_homogeneous` (int 0/1).

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

Helpers in [`mesh/_record_h5.py`](../../mesh/_record_h5.py):

* `make_record_dtype(payload_dtype)` returns the outer compound.
* Per-record-type factories (`node_pair_payload_dtype`,
  `nodal_load_payload_dtype`, `mass_payload_dtype`, …) return the
  inner payload dtypes.

## `/opensees/materials`

```
/opensees/materials/
├── /uniaxial/
│   ├── /Steel_S420/                  group
│   │   attrs: type="Steel02", tag=3, fy=420e6, E=200e9, b=0.01,
│   │          R0=20.0, cR1=0.925, cR2=0.15
│   └── /Concrete_C30/
│       attrs: type="Concrete02", tag=4, fpc=-30e6, epsc0=-0.002, ...
└── /nd/
    └── /Concrete_3D/
        attrs: type="ElasticIsotropic", tag=1, E=30e9, nu=0.2, rho=2400.0
```

Each material is a **group with no datasets, only attributes**. The
attributes are the constitutive parameters, named exactly as in the
typed dataclass (`fy`, `E`, `b`, …). The OpenSees type token lives in
the `type` attribute.

Optional: a `/comments` attribute (string) for user-supplied notes.

## `/opensees/sections`

Sections that aggregate (Fiber, LayeredShell) carry compound datasets
for their components. Sections that don't (ElasticMembranePlateSection)
are attribute-only, like materials.

### Fiber section

```
/opensees/sections/Cols/
├── attrs: type="Fiber", tag=1, GJ=1.0e9
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
/opensees/transforms/Cols/
├── attrs: type="PDelta", tag=5,
│         orientation_kind="Cylindrical",   ← optional, present if orientation was used
│         orientation_origin=[0.0, 0.0, 0.0],
│         orientation_axis=[0.0, 0.0, 1.0],
│         roll_deg=0.0
├── per_element_vecxz       float dataset (n_elements, 3)
│                            row i corresponds to /elements/Cols/ids[i]
└── per_element_emitted_tag int dataset (n_elements,)
                             which OpenSees geomTransf tag was assigned
                             (multiple if orientation fan-out)
```

When the user supplied an explicit `vecxz=` (no orientation), `per_element_vecxz`
is still present — every row holds the same vector — so the viewer
can read uniformly.

**Authoring front doors.** Two surfaces produce this zone (one schema,
one writer in `H5Emitter`):

* `apeSees(fem).h5(path)` — typed-primitive `ops.geomTransf.<Type>(...)` →
  `BuiltModel.emit` → `H5Emitter.geomTransf(...)`.
* `apeGmsh.opensees.ModelData(fem).oriented_elements(pg=, ele_type=,
  vecxz=).write(path)` — declarative side-channel for users who write
  their model in vanilla openseespy without the bridge.  Sees ADR
  [0018](decisions/0018-modeldata-vanilla-opensees-enrichment.md) and
  [modeldata-enrichment-scope.md](modeldata-enrichment-scope.md).
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
          params=[sec_tag, n_ip, ...]
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
│                                                    -1 sentinel for records
│                                                    emitted outside a bridge fan-out)
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
tags of **node-pair** elements (`ops.element.ZeroLength(nodes=…)` and the
rest of the zeroLength family wired to a `g.decouple_node` ground).  Such
an element has `fem_eid = -1` and **no gmsh cell** in the neutral
`/elements` zone, so its connectivity cannot be sourced from there on
re-emit — it is stored inline here instead, one ragged row per element
(empty for ordinary PG-fanned rows whose connectivity lives in
`/elements`).  The dataset is written **only** when a type group has at
least one node-pair row, so PG-only models are byte-identical and their
`model_hash` is unperturbed; when present it folds into `model_hash`
(connectivity is model-defining).

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
```

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
[`apeGmsh.cuts._h5_io.write_cuts_into`](../../cuts/_h5_io.py); reader
in `read_cuts_and_sweeps`. Full design rationale in
[`apeGmsh/cuts/ARCHITECTURE.md`](../../cuts/ARCHITECTURE.md) — "## v4
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
[`opensees/_internal/schema_version.py`](../_internal/schema_version.py).
`reader_version(zone)` sources each zone's current value directly from
the writer's constant, so reader and writer **cannot drift**; readers
call `validate_zone_version(...)` for each zone before reading it.

### Zone registry

| Zone | `/meta` key | Root paths | Writer constant (source of truth) | Current |
|---|---|---|---|---|
| neutral (broker) | `neutral_schema_version` | `/nodes`, `/elements`, `/physical_groups`, `/labels`, `/mesh_selections`, `/constraints`, `/loads`, `/masses`, `/partitions`, `/parts`, `/composed_from` | [`mesh/_femdata_h5_io.py`](../../mesh/_femdata_h5_io.py) `NEUTRAL_SCHEMA_VERSION` | **2.10.0** |
| opensees (bridge) | `opensees_schema_version` | `/opensees/*` | [`opensees/emitter/h5.py`](../emitter/h5.py) `SCHEMA_VERSION` | **2.12.0** |
| results | `results_schema_version` | `/stages/*` (composed `results.h5`, at file root) | [`results/schema/_versions.py`](../../results/schema/_versions.py) `RESULTS_SCHEMA_VERSION` | **1.1.0** |
| cuts (sub-zone of opensees) | — (no own key; rides the opensees zone) | `/opensees/cuts`, `/opensees/sweeps` | [`cuts/_h5_io.py`](../../cuts/_h5_io.py) `V4_SCHEMA_VERSION` | 2.5.0 |

> The registry's *current* values are a snapshot — the writer
> constants above are the authoritative source. The test
> [`tests/opensees/h5/test_h5_schema_compat.py`](../../../../tests/opensees/h5/test_h5_schema_compat.py)
> (`test_reader_version_reflects_writer_constants`) pins reader↔writer
> agreement; [`tests/fixtures/schema.py`](../../../../tests/fixtures/schema.py)
> centralizes the values fixtures stamp.

### Bump rules (per zone)

- **Major** bump → breaking change. Readers refuse outright.
- **Minor** bump → additive (new group, new attribute). Readers
  one minor behind continue to parse via the two-version window,
  ignoring unknown groups.
- **Patch** bump → internal/cosmetic. Readers must not depend.

**Window caveat.** The two-version window applies to **additive-only**
minor bumps. A minor bump that perturbs the canonical bytes of a
required structure (the B2 layout split at neutral 2.10, and the
opensees 2.11 0-based-rank flip, are the canonical examples) walks the
window forward and the prior minor falls out of reach. See
[ADR 0023's 2026-05-28 amendment](decisions/0023-per-zone-schema-versioning.md#amendment--2026-05-28--window-applies-to-additive-only-changes-b2--pr-398).

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

The list below is the **neutral-zone** lineage. The opensees zone's
per-version history is maintained inline in
[`opensees/emitter/h5.py`](../emitter/h5.py) (`SCHEMA_VERSION`
docstring), current through **2.12.0**; its post-2.10 additions are
summarized after this list.

History:

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
  [phase-9-recorder-unification.md](phase-9-recorder-unification.md)
  for the multi-commit phase that delivered this.
- `2.4.0` — Phase 8.7 commit 2: `/mesh_selections/` neutral-zone
  group added, mirroring `/physical_groups` / `/labels` shape.
  Carries post-mesh selection sets (``g.mesh_selection`` →
  ``fem.mesh_selection``) so the viewer's `selection=` selector
  round-trips through `model.h5`.  Additive — old v2.3.0 readers
  ignore the new group and lose only the `selection=` round-trip
  convenience (live mesh_viewer sessions still consult the live
  ``fem.mesh_selection`` directly).  See
  [phase-8.7-scope.md](phase-8.7-scope.md) §1b for the rationale
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
  files cannot be read by the 2.10 reader, the window slid forward.
  ADR 0021 INV-1 retired; ADR 0023 window semantics reframed.

### OpenSees-zone history (post-2.10)

The opensees zone advanced past 2.10 independently of the neutral
zone (it shares the early lineage above through 2.10; the entries
below are opensees-only and have no neutral-zone counterpart). Full
detail lives in the `SCHEMA_VERSION` docstring in
[`opensees/emitter/h5.py`](../emitter/h5.py):

- `2.11.0` — bug fix: the bridge emits **0-based runtime ranks**
  (matching `OpenSeesMP::getPID()`) instead of Gmsh's 1-based
  `PartitionRecord.id`. Per-partition `rank` attrs and the
  `partition_ids` column on `/opensees/element_meta/{type}` are now
  in `[0, N-1]`. Group naming (`partition_NN`) is unchanged. Breaking
  for any reader that mapped `rank`/`partition_ids` to `part.id`
  directly; the two-version window slid forward (2.10 falls out of
  reach for the opensees zone).
- `2.12.0` — ADR 0035 (ASDEmbeddedNodeElement option exposure):
  `/opensees/constraints/embeddedNode` gains five typed columns —
  `stiffness` (`-K`), `stiffness_p` (`-KP`) + `has_stiffness_p`
  sentinel, `rotational` (`-rot`), `pressure` (`-p`). Defaults match
  the C++ parser, so legacy decks behave identically. Additive — old
  2.11.x readers ignore the new columns.
- `2.13.0`–`2.16.0` — see the `SCHEMA_VERSION` docstring in
  [`opensees/emitter/h5.py`](../emitter/h5.py) (named primitives sidecar,
  `/opensees/nodes_ndf`, `/opensees/dampings`, `/opensees/initial_stress`).
  From 2.16.0 onward a minor bump is a producer **hard floor** — a 2.N.x
  reader REFUSES a 2.(N+1).x file (the window only lets a newer reader open
  an older file, never the reverse).
- `2.17.0` — ADR 0049 (node-pair zeroLength): optional
  `inline_connectivity` vlen-int64 dataset under
  `/opensees/element_meta/{type}/` carrying the endpoint tags of node-pair
  spring elements (no neutral gmsh cell to source them from). Written only
  for type groups with a node-pair (`fem_eid < 0`) row, so PG-only models
  are byte-identical; folds into `model_hash`. Additive group; a 2.16.x
  reader refuses a 2.17.x file (hard floor), a 2.17 reader opens 2.16 + 2.17.
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
  exists (vanilla byte-identical); folds into `model_hash`; hard-floor
  window semantics as above.

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
    ├── /materials/uniaxial/Steel/
    │   type="Steel02", tag=1, fy=420e6, E=200e9, b=0.01, R0=20.0,
    │   cR1=0.925, cR2=0.15
    ├── /materials/uniaxial/Concrete/
    │   type="Concrete02", tag=2, fpc=-30e6, epsc0=-0.002,
    │   fpcu=-25e6, epsu=-0.006, lambda_val=0.1, ft=2.5e6, Ets=200e6
    ├── /sections/Col/
    │   ├── attrs: type="Fiber", tag=1, GJ=1.0e9
    │   ├── /patches  → 1 row: kind="rect",
    │   │              material_ref="/opensees/materials/uniaxial/Concrete",
    │   │              ny=8, nz=8, coords=[-0.20,-0.20,0.20,0.20,nan,nan,nan,nan]
    │   └── /fibers   → 8 rows of (y, z, area,
    │                              material_ref="/opensees/materials/uniaxial/Steel")
    ├── /transforms/Col/
    │   ├── attrs: type="PDelta", tag=1, orientation_kind="Cartesian",
    │   │          orientation_origin=[0,0,0], orientation_axis=[0,0,1], roll_deg=0.0
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
