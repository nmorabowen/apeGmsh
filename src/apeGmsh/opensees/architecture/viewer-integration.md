# Viewer integration — what the viewer team needs to implement

This document is the contract the viewer team works against to
consume the bridge enrichment HDF5 file (see [h5-schema.md](h5-schema.md)).
It lists the specific UI features the H5 enables, the discovery and
loading conventions, the required vs optional reads, and the test
fixtures that will be provided.

## TL;DR for the viewer team

You will be given **one new optional input** alongside the existing
FEM scene: `model.h5`. When present, it tells you about materials,
sections, transforms with per-element vecxz, time series, patterns,
recorders, and analysis settings — none of which exist in the FEM or
in MPCO results.

When absent, viewer behavior is unchanged. **Every feature below is
opt-in**: the viewer continues to work without `model.h5`, and
gracefully degrades when groups are missing within the file.

## File discovery

The viewer accepts the H5 path through one of three channels, in
priority order:

1. **Explicit argument** — the application code passes
   `model_h5=Path("...")` when constructing the viewer.
2. **Sidecar to a Tcl/py deck** — if the viewer was opened on a
   model deck `frame.tcl` or `frame.py`, look for `frame.h5` in the
   same directory.
3. **Sidecar to a results MPCO** — if the viewer was opened on
   `frame.mpco`, look for `frame.h5` in the same directory.

If no H5 is found, log at INFO level and continue. Do not raise.

## Required reads

When `model.h5` is loaded, the viewer MUST validate and read these
groups before attempting any enrichment:

| Group | Required | Behavior if missing |
|---|---|---|
| `/meta` | yes | Refuse to load; surface schema-version mismatch clearly. |
| `/nodes` | yes | Refuse to load (mesh substrate is mandatory). |
| `/elements/{gmsh_alias}` | yes (per element type rendered) | Skip enrichment for that element type. |

Phase 8.5 made the broker the source-of-truth for `/nodes` and
`/elements/{gmsh_alias}` (keys are GMSH aliases: `tet4`, `hex8`,
`line2`, …).  OpenSees-specific per-type metadata lives in a
parallel `/opensees/element_meta/{type_token}/` zone keyed by
OpenSees type name (`forceBeamColumn`, `FourNodeTetrahedron`); the
viewer joins the two via shared element tags when the user clicks
into an element-detail panel.

Schema version check (mandatory):

```python
major = int(f["/meta"].attrs["schema_version"].split(".")[0])
if major != EXPECTED_MAJOR:
    raise SchemaVersionError(
        f"model.h5 schema v{major}.x.y is not supported by this "
        f"viewer (expected v{EXPECTED_MAJOR}.x.y)"
    )
```

## Optional reads → UI features

For each group below, the viewer team implements the listed UI
feature when the group is present. When the group is absent, the
feature is hidden — no error, no warning.

### `/opensees/materials/*` — Material inspector panel

**Trigger:** user clicks an element OR the "Materials" tab.
**UI:**

- A side panel listing all materials, grouped by family (uniaxial,
  nd).
- Each entry shows: name, OpenSees type, parameters as a table.
- Optional: a small backbone plot inset for uniaxial materials, if
  the viewer team wants to add a constitutive plotter (the bridge's
  typed Material classes can produce this independently;
  out-of-scope here).

**Read pattern:** lazy. Don't load all params at file-open time;
read group attributes when the panel is opened.

### `/opensees/sections/*` — Section panel

**Trigger:** user clicks a beam-column or shell element.
**UI:**

- For Fiber sections: render the patches and fibers as a 2-D
  canvas in a side panel. Patches as filled rectangles colored by
  material (look up `material_ref` to get the material type, then
  hash to a palette); fibers as small circles.
- For ElasticMembranePlateSection: show E, ν, h as a parameter
  table.
- For LayeredShell: show layer stack as a 1-D bar chart with layer
  thicknesses and material colors.

**Read pattern:** when section panel is opened, read
`/opensees/sections/{section_ref}/patches` and `/fibers`. Cache by
section name; sections are reused across many elements.

### `/opensees/transforms/*/per_element_vecxz` — Local-axis glyph overlay

**Trigger:** user enables "Show local axes" overlay on beam-column
elements.
**UI:**

- For each beam element, draw three glyphs at the centroid:
  - **local-x** along the gmsh tangent (already known to the viewer)
  - **local-y** computed from `vecxz × tangent` (the H5 gives
    `per_element_vecxz`; viewer computes `local_y = unit(vecxz × tangent)`)
  - **local-z** = `vecxz` itself (or the unit-normalized projection
    perpendicular to tangent)
- Color glyphs by axis (x=red, y=green, z=blue per the existing
  apeGmsh convention).
- Glyph length proportional to local element length.

**Read pattern:** at file-open time, read all
`/opensees/transforms/*/per_element_vecxz` and the matching
`/elements/{gmsh_alias}/ids`. Build an `element_id → vecxz` lookup
table.

### `/physical_groups/*` / `/labels/*` — group inspector

**Trigger:** "Groups" tab or click a group in the tree.
**UI:** highlight member nodes / elements, list counts, allow
toggling visibility.  Source data lives at the root level since
Phase 8.5 (broker-owned); read attrs (`dim`, `tag`, `name`) and
datasets (`node_ids`, `node_coords`, optional `element_ids`) per
sub-group.

### `/constraints/{kind}/*` and `/loads/{kind}/{pattern}/*` — record explorer

**Trigger:** "Constraints" / "Loads" tab.
**UI:** list per-kind rows; on click, walk the symmetric outer
compound (`target_kind`, `target`, `payload_kind`, `payload`) and
decode the payload using the per-kind dtype.  Helpers live in
[`mesh/_record_h5.py`](../../mesh/_record_h5.py); the contract is
documented in [`h5-schema.md`](h5-schema.md) under "Symmetric
compound contract".

### `/opensees/element_meta/{type_token}/*` — element-detail panel

**Trigger:** user clicks an element in the mesh.
**UI:** join the broker's `/elements/{gmsh_alias}/ids` index against
the bridge's `/opensees/element_meta/{type_token}/ids` to find the
matching row, then expose the parameter tail (`args` /
`args_str`) plus cross-refs (`transf_ref`, `section_ref`,
`integration_ref`) for the clicked element.

Phase 8.6 added a `fem_eids` parallel array alongside `ids`: row
`i`'s `fem_eids[i]` is the FEM element id (broker's `ids[i_fem]`)
that the bridge fanned the OpenSees tag `ids[i]` out from.  This
gives the viewer a round-trip: from an OpenSees tag in MPCO recorder
output, look up the matching `fem_eid` and highlight the broker-side
geometry without an external mapping table.  Sentinel value `-1`
marks records emitted outside a bridge fan-out (rare; only happens
in standalone H5Emitter test cases).

**Note:** today the viewer already has CSys overlays through the
diagram path (`viewers/diagrams/_beam_geometry.py`). This H5
read replaces the live-bridge dependency that overlay had — the
viewer can show vecxz from a frozen H5 even after the bridge is
gone.

### `/opensees/time_series/*` — Time-series plot panel

**Trigger:** user clicks a pattern OR opens the "Time series" tab.
**UI:**

- Plot `time` vs `values`.
- Show the type label (`Path`, `ASCE41Protocol`, etc.) and
  parameters (`factor`, `dt`).
- For ground-motion records: a 3-pane plot (acc / vel / disp via
  trapezoidal integration on the fly).
- A legend showing which patterns reference this series
  (cross-referenced from `/opensees/patterns/*/series_ref`).

**Read pattern:** lazy. Load `time` and `values` when the panel is
opened. Compression in the H5 is gzip-4; expect ~30% file size for
typical ground motions.

### `/opensees/patterns/*` — Pattern explorer

**Trigger:** "Patterns" tab in the tree view.
**UI:**

- Hierarchical view: pattern → loads → targets.
- For each load: target (PG name or node tag), force vector, the
  series it follows.
- Visualize loads on the 3-D mesh: arrows at target nodes scaled
  by force magnitude, colored by pattern.
- Toggle per-pattern visibility.

**Read pattern:** read at file-open time (small data — even a 100-load
pattern is < 10 KB).

### `/opensees/recorders/*` — Recorder coverage map

**Trigger:** "Recorders" tab.
**UI:**

- For each Node recorder: highlight target nodes, list responses.
- For each Element recorder: highlight target elements.
- For MPCO: list the requested response tokens.
- An "uncovered" highlight: nodes/elements that don't appear in
  any recorder. Useful for catching incomplete output requests
  before running an analysis.

**Read pattern:** read at file-open time.

### `/opensees/analysis` — Analysis summary panel

**Trigger:** "Analysis" tab.
**UI:**

- A simple table: handler, numberer, system, test (+tol+max_iter),
  algorithm, integrator (+params), analysis, analyze_steps,
  analyze_dt.
- Read-only.

**Read pattern:** attribute reads only. Cheap.

## Performance budget

For a model with 50,000 nodes / 100,000 elements:

| Operation | Target time |
|---|---|
| Open H5 + validate version | < 50 ms |
| Build `element_id → vecxz` table for all transforms | < 100 ms |
| Load all patterns + recorders into memory | < 50 ms |
| Lazy-read one section panel | < 20 ms |
| Lazy-read one time series (200 KB compressed) | < 100 ms |

These are advisory. Bigger models scale linearly with their data;
the H5 itself is small (KB to low MB) for typical models.

## Error handling

**Schema version mismatch:** raise `SchemaVersionError`, surface in
UI as an error dialog.

**Missing required group (`/meta`):** raise `MalformedH5Error`.

**Missing optional group:** silently skip the corresponding feature.
Log at DEBUG level: "no /opensees/sections group; section panel disabled."

**Type mismatch on cross-reference** (e.g. `material_ref` points at
something that isn't a material group): log WARNING, skip the
specific record. Do not crash.

**Truncated compound dataset:** log WARNING, skip incomplete rows.

## File-level invariants the viewer can rely on

- All cross-references are valid HDF5 paths within the same file.
  No external references.
- Numeric attributes are stored as native HDF5 numeric types
  (float64, int64) — no JSON strings.
- String attributes are UTF-8 encoded.
- Variable-length string fields in compound datasets use
  `h5py.string_dtype(encoding="utf-8")`.
- Compound dataset row ordering is meaningful where the matching
  array (e.g. `/elements/{type}/ids`) shares the same index.
  Specifically: row `i` of
  `/opensees/transforms/{name}/per_element_vecxz` corresponds to
  row `i` of `/elements/{type}/ids` for the element type that
  references this transform.
- All datasets that hold per-element data are 1-D or 2-D dense
  arrays. No ragged data.

## Test fixtures

The bridge team will ship the following fixture files under
`tests/fixtures/h5/`:

| Fixture | Contents |
|---|---|
| `minimal.h5` | One column, one fiber section, one ground motion, no analysis |
| `frame_3d.h5` | The moment-frame example from charter — multiple PGs, mixed sections, multi-pattern |
| `arch_csys.h5` | Shoebuckle arch with cylindrical CS — many distinct vecxz |
| `dome_spherical.h5` | Spherical CS on dome ribs |
| `tank_cylindrical.h5` | Tank with ring beams + vertical stiffeners |
| `incomplete.h5` | `/meta` + `/elements` only — viewer must show mesh, hide all enrichment panels |
| `wrong_major.h5` | Schema major v3 — viewer must refuse |

Each fixture has a sibling `.json` describing its expected viewer
output (which panels populated, what counts) so the viewer team can
write integration tests.

## Integration points in the existing viewer

These are the files in `apeGmsh/viewers/` the viewer team will most
likely touch. The bridge team flags them as references; final design
is the viewer team's call.

| Concern | Existing module | Likely change |
|---|---|---|
| Overlay system | `viewers/overlays/` | Add `model_h5_overlay.py` driving glyph overlays |
| Side panels | `viewers/panels/` (or wherever the dock lives) | Add Material, Section, TimeSeries, Pattern, Recorder, Analysis panels |
| Diagrams | `viewers/diagrams/_beam_geometry.py` | Replace live-bridge vecxz lookup with H5 read |
| Scene | `viewers/scene/` | New "Compositions" entry: "Model definition (H5)" |
| Discovery | viewer entry point | Add H5 sidecar discovery (see "File discovery" above) |

## Versioning policy

The bridge / broker teams own the schema jointly.  Schema bumps
follow semver:

- **Major** bump (2.x → 3.x): breaking. Coordinate ahead of time.
  Both teams ship paired releases.  (Phase 8.4 was the most recent
  major: `1.x.y → 2.0.0`, introducing the `/opensees/` namespace.)
- **Minor** bump (2.0 → 2.1): additive. Viewer team gets a heads-up
  but isn't required to update — old viewer reads new file with
  reduced functionality (the new groups go unused).  Recent minors:
  - `2.0.0 → 2.1.0` (Phase 8.5) — broker neutral zone (`/nodes`,
    `/elements/{gmsh_alias}`, `/physical_groups`, `/labels`,
    `/constraints/{kind}`, `/loads/{kind}/{pattern}`, `/masses`).
  - `2.1.0 → 2.2.0` (Phase 8.6) — `fem_eids` int64 dataset under
    each `/opensees/element_meta/{type_token}/`, mapping each
    OpenSees element tag to its source FEM element id.  Pre-8.6
    viewers ignore the new dataset and lose only the round-trip
    convenience.
- **Patch** bump (2.0.0 → 2.0.1): clarifications, doc-only. No
  reader changes.

Schema additions over the v2.x lifetime that the viewer team should
expect (each will land via minor bump, additive only):

- `/opensees/recipes` — when the bridge supports recipes that produce
  more than one primitive, recording the recipe used (so the viewer
  can show "this section was generated by
  RectangularConfinedColumn(width=...)").
- `/opensees/regions` — OpenSees regions (damping, recorders).
- `/constraints/multi_point` — equal_dof, rigid_link records when the
  bridge promotes them from FEM-side records to first-class
  primitives.  Lives in the neutral zone (Phase 8.5).

## What the bridge team commits to

1. Schema is authoritative — `h5-schema.md` is the spec; H5 files
   conform.
2. Backward compatibility within v2.x.y.
3. Test fixtures will land before the viewer team starts work.
4. Schema changes are announced via PRs that update both
   `h5-schema.md` and `viewer-integration.md` together.
5. Reference reader code in the bridge package — viewer team can
   borrow validation logic.

## What the viewer team commits to

1. Graceful degradation for any optional group missing.
2. Schema version check on file load.
3. Performance budget targets met for the test fixtures.
4. UI features wired through the existing dock/panel system, not
   bypassed.
5. Integration tests against the provided fixtures.

## Out of scope for v1

- Editing the H5 from the viewer (write-back). Read-only.
- Streaming updates (live link to a running bridge). The H5 is a
  snapshot; the bridge writes once.
- Result enrichment (overlaying analysis results on model
  definition). That's STKO/MPCO's job; this file is model only.
