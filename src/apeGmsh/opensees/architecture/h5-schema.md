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
3. **Schema-versioned at the root.** Readers MUST check
   `/meta/schema_version` and refuse incompatible files.
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
├── /physical_groups
│     └── /{name}                          one group per Gmsh physical group
├── /labels
│     └── /{name}                          one group per apeGmsh label
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
| `schema_version` | string | semver, e.g. `"2.2.0"` |
| `apeGmsh_version` | string | producing apeGmsh version |
| `created_iso` | string | ISO 8601 timestamp |
| `ndm` | int | spatial dimension |
| `ndf` | int | DOFs per node |
| `snapshot_id` | string | hash of FEMData snapshot the bridge was built from |
| `model_name` | string | user-provided model name |

Schema versioning is **strict on major**, **lax on minor/patch**. A
reader written for `2.x.y` MUST refuse to read `1.x.y` or `3.x.y`
files. Within `2.x.y`, additions are allowed without breaking
readers.

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

Neutral-zone group at the root, broker-owned.  One sub-group per
Gmsh physical group; combines node-side and element-side membership.

```
/physical_groups/Slab/
├── attrs: dim=2, tag=100, name="Slab"
├── node_ids          (Np,) int64
├── node_coords       (Np, 3) float64
└── element_ids       (Ep,) int64        — present for dim>=1
```

The combined-side shape matches the master plan's "top-level index for
viewer discovery": one `physical_groups[name]` walk gives the viewer
everything it needs to colour a PG.

## `/labels`

Same shape as `/physical_groups`, with apeGmsh-internal labels in
place of Gmsh PG taxonomy.  Each label entry carries the same
fields (`dim`, `tag`, `name`, `node_ids`, `node_coords`, optional
`element_ids`).

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
│         csys_kind="Cylindrical",        ← optional, present if csys was used
│         csys_origin=[0.0, 0.0, 0.0],
│         csys_axis=[0.0, 0.0, 1.0],
│         roll_deg=0.0
├── per_element_vecxz       float dataset (n_elements, 3)
│                            row i corresponds to /elements/Cols/ids[i]
└── per_element_emitted_tag int dataset (n_elements,)
                             which OpenSees geomTransf tag was assigned
                             (multiple if csys fan-out)
```

When the user supplied an explicit `vecxz=` (no csys), `per_element_vecxz`
is still present — every row holds the same vector — so the viewer
can read uniformly.

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
└── args_str          (N, max_tail) vlen-utf-8    — string tokens (present only
                                                    when any slot is a string)
```

`args` and `args_str` encode the element's positional `*args` list
*after dropping the connectivity prefix*.  A vocabulary-aware reader
recovers cross-references (`transf_ref`, `section_ref`,
`integration_ref`, …) by indexing into the element type's known
signature.

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

```
/opensees/recorders/disp/
├── attrs: type="Node", file="disp.out", response="disp",
│         dT=0.0
├── target_nodes      int dataset
├── target_dofs       int dataset
└── time_format       string attr     ← "step" | "dt"

/opensees/recorders/forces/
├── attrs: type="Element", file="forces.out", response="globalForce"
└── target_elements   int dataset

/opensees/recorders/main_mpco/
├── attrs: type="MPCO", file="model.mpco"
├── nodal_responses   string dataset   ← ["displacement", "reactionForce"]
└── elem_responses    string dataset   ← ["stresses", "section.force"]
```

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

`/meta/schema_version` follows semver:

- **Major** bump → breaking change. Readers refuse.
- **Minor** bump → additive (new group, new attribute). Readers
  ignore unknown groups.
- **Patch** bump → internal/cosmetic. Readers must not depend.

The current schema version is **`2.2.0`**.

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

A reader skeleton:

```python
import h5py

def read_model_h5(path):
    with h5py.File(path, "r") as f:
        meta = f["/meta"]
        major = int(meta.attrs["schema_version"].split(".")[0])
        if major != 2:
            raise ValueError(
                f"Unsupported model.h5 schema major version {major}; "
                f"reader supports v2.x.y"
            )
        # Walk the file ...
```

## Worked example — minimal model

A single elastic column with one fiber section, one ground motion,
no analysis settings:

```
column.h5
├── /meta
│   schema_version="2.2.0", ndm=3, ndf=6, snapshot_id="abc123"
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
    │   ├── attrs: type="PDelta", tag=1, csys_kind="Cartesian",
    │   │          csys_origin=[0,0,0], csys_axis=[0,0,1], roll_deg=0.0
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
