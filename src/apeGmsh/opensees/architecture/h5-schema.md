# `model.h5` — schema for the bridge enrichment file

`apeSees.h5(path)` writes an HDF5 file that captures **everything the
bridge knows about the model definition**: materials, sections,
transforms with per-element vecxz, elements grouped by PG, time
series with their values, patterns with their loads, BCs, recorders,
and analysis settings.

This file is **the canonical model archive**. It carries information
that is in *neither* the FEMData snapshot (geometry only) nor the
STKO/MPCO results (response only).

## Design principles

1. **One file, navigable as a graph.** Cross-references are HDF5
   paths (`/sections/Cols`), not numeric tags. Anyone with `h5py` or
   `h5dump` can walk the model.
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

## Top-level layout

```
model.h5
├── /meta                                  attrs only
├── /materials
│     ├── /uniaxial/{name}                 one group per material
│     └── /nd/{name}
├── /sections
│     └── /{name}                          one group per section
├── /transforms
│     └── /{name}                          one group per geomTransf
├── /elements
│     └── /{pg_name}                       one group per assigned PG
├── /time_series
│     └── /{name}                          one group per series
├── /patterns
│     └── /{name}                          one group per pattern
├── /bcs
│     ├── /fix                             single dataset
│     └── /mass                            single dataset
├── /recorders
│     └── /{name}                          one group per recorder
└── /analysis                              attrs + sub-attrs (optional)
```

The user's PG names, material names, etc. are HDF5 group names — they
must therefore avoid `/` characters. The bridge enforces this at
declaration time.

## `/meta`

Attributes only.

| Attribute | Type | Description |
|---|---|---|
| `schema_version` | string | semver, e.g. `"1.0.0"` |
| `apeGmsh_version` | string | producing apeGmsh version |
| `created_iso` | string | ISO 8601 timestamp |
| `ndm` | int | spatial dimension |
| `ndf` | int | DOFs per node |
| `snapshot_id` | string | hash of FEMData snapshot the bridge was built from |
| `model_name` | string | user-provided model name |

Schema versioning is **strict on major**, **lax on minor/patch**. A
reader written for `1.x.y` MUST refuse to read `2.x.y` files. Within
`1.x.y`, additions are allowed without breaking readers.

## `/materials`

```
/materials/
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

## `/sections`

Sections that aggregate (Fiber, LayeredShell) carry compound datasets
for their components. Sections that don't (ElasticMembranePlateSection)
are attribute-only, like materials.

### Fiber section

```
/sections/Cols/
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
              line (float[4])      ← (y1, z1, y2, z2)
```

`material_ref` is an HDF5 path string like `"/materials/uniaxial/Steel_S420"`.
Readers resolve by `f[material_ref]`.

### Plate / shell section

```
/sections/Slab/
├── attrs: type="ElasticMembranePlateSection", tag=2, E=30e9, nu=0.2,
│          h=0.20, rho=2400.0
└── (no sub-groups)
```

### Layered shell

```
/sections/Composite/
├── attrs: type="LayeredShellFiberSection", tag=3
└── /layers              compound dataset, shape (n_layers,)
      fields: material_ref (string), thickness (float), n_int_pts (int)
```

### Aggregator / Parallel

```
/sections/Combined/
├── attrs: type="Aggregator", tag=4
└── /components          compound dataset
      fields: section_ref (string), dof_ids (int[ndf])
```

## `/transforms`

```
/transforms/Cols/
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

## `/elements`

One group per PG that received an `assign(...)` call.

```
/elements/Cols/
├── attrs: type="forceBeamColumn", n_ip=5,
│         section_ref="/sections/Cols",
│         transf_ref="/transforms/Cols"
├── ids               int dataset (n_elements,)
│                      OpenSees element tags
└── connectivity      int dataset (n_elements, n_corners)
                       node tags per element

/elements/Body/
├── attrs: type="FourNodeTetrahedron",
│         material_ref="/materials/nd/Concrete_3D"
├── ids
└── connectivity      shape (n_elements, 4)
```

Element types that take additional scalar params (`A`, `E`, etc.
on `elasticBeamColumn`) carry them as attributes alongside the refs.

## `/time_series`

```
/time_series/elcentro/
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

## `/patterns`

```
/patterns/Wind/
├── attrs: type="Plain", tag=1, series_ref="/time_series/Linear_1"
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

`/patterns/Earthquake_X/` for `UniformExcitation`:

```
/patterns/Earthquake_X/
└── attrs: type="UniformExcitation", tag=2, direction=1,
          series_ref="/time_series/elcentro"
```

(no contained loads — uniform excitation IS the pattern's payload)

## `/bcs`

```
/bcs/fix                  compound dataset, shape (n_fix_records,)
   fields: target_kind (string), target (string), dofs (int[ndf])

/bcs/mass                 compound dataset, shape (n_mass_records,)
   fields: target_kind, target, values (float[ndf])
```

## `/recorders`

```
/recorders/disp/
├── attrs: type="Node", file="disp.out", response="disp",
│         dT=0.0
├── target_nodes      int dataset
├── target_dofs       int dataset
└── time_format       string attr     ← "step" | "dt"

/recorders/forces/
├── attrs: type="Element", file="forces.out", response="globalForce"
└── target_elements   int dataset

/recorders/main_mpco/
├── attrs: type="MPCO", file="model.mpco"
├── nodal_responses   string dataset   ← ["displacement", "reactionForce"]
└── elem_responses    string dataset   ← ["stresses", "section.force"]
```

## `/analysis` (optional)

Present only if the user called the analysis primitives.

```
/analysis/
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
| `material_ref` | `/materials/uniaxial/Steel_S420` |
| `section_ref` | `/sections/Cols` |
| `transf_ref` | `/transforms/Cols` |
| `series_ref` | `/time_series/elcentro` |

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

The current schema version is **`1.0.0`**.

A reader skeleton:

```python
import h5py

def read_model_h5(path):
    with h5py.File(path, "r") as f:
        meta = f["/meta"]
        major = int(meta.attrs["schema_version"].split(".")[0])
        if major != 1:
            raise ValueError(
                f"Unsupported model.h5 schema major version {major}; "
                f"reader supports v1.x.y"
            )
        # Walk the file ...
```

## Worked example — minimal model

A single elastic column with one fiber section, one ground motion,
no analysis settings:

```
column.h5
├── /meta
│   schema_version="1.0.0", ndm=3, ndf=6, snapshot_id="abc123"
├── /materials/uniaxial/Steel/
│   type="Steel02", tag=1, fy=420e6, E=200e9, b=0.01, R0=20.0,
│   cR1=0.925, cR2=0.15
├── /materials/uniaxial/Concrete/
│   type="Concrete02", tag=2, fpc=-30e6, epsc0=-0.002,
│   fpcu=-25e6, epsu=-0.006, lambda_val=0.1, ft=2.5e6, Ets=200e6
├── /sections/Col/
│   ├── attrs: type="Fiber", tag=1, GJ=1.0e9
│   ├── /patches  → 1 row: kind="rect", material_ref="/materials/uniaxial/Concrete",
│   │              ny=8, nz=8, coords=[-0.20,-0.20,0.20,0.20,nan,nan,nan,nan]
│   └── /fibers   → 8 rows of (y, z, area, material_ref="/materials/uniaxial/Steel")
├── /transforms/Col/
│   ├── attrs: type="PDelta", tag=1, csys_kind="Cartesian",
│   │          csys_origin=[0,0,0], csys_axis=[0,0,1], roll_deg=0.0
│   ├── per_element_vecxz       (1, 3) = [[1, 0, 0]]
│   └── per_element_emitted_tag (1,)   = [1]
├── /elements/Col/
│   ├── attrs: type="forceBeamColumn", n_ip=5,
│   │          section_ref="/sections/Col",
│   │          transf_ref="/transforms/Col"
│   ├── ids           [1]
│   └── connectivity  [[1, 2]]
├── /time_series/elcentro/
│   ├── attrs: type="Path", factor=9.81, dt=0.01, file_path="elcentro.txt"
│   ├── time       (n_steps,)  = [0.00, 0.01, 0.02, ...]
│   └── values     (n_steps,)  = [0.001, 0.005, ..., -0.012, ...]
├── /patterns/Quake/
│   └── attrs: type="UniformExcitation", tag=1, direction=1,
│              series_ref="/time_series/elcentro"
└── /bcs/fix
    target_kind=["pg"], target=["Base"], dofs=[[1,1,1,1,1,1]]
```

This file is ~50 KB and tells the viewer everything it needs to
draw the column with its section, materials, orientation, and
ground motion — without reading a single OpenSees recorder output.
