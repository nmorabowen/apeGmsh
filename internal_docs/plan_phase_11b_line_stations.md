# Phase 11b — Line stations + nodal forces topology levels

> [!note] Status
> Scoped April 2026 after Phase 11a Round A (trusses) landed. Phase 11a
> v1 covered the Gauss-level topology (`elements/gauss_points/`) for
> continuum solids, shells, and 1-GP trusses. Phase 11b extends the
> Results module to the two remaining line-element topology levels:
>
> - **`elements/line_stations/`** — distributed-plasticity beam-columns
>   that integrate section forces along the element length.
> - **`elements/nodal_forces/`** — closed-form elastic beams that
>   expose only per-element-node force vectors (no integration points).
>
> The catalog framework needs a new model: **per-instance integration
> rules**. Force-/disp-based beam-columns let the user choose any of
> ~18 beam-integration schemes (Lobatto, Legendre, Radau,
> NewtonCotes, Simpson, HingeMidpoint, HingeRadau, FixedLocation,
> UserDefined, …). Each element instance's `n_IP` and IP locations
> depend on the assigned `beamIntegration`, not on the element class.
> MPCO classifies all of these as `CustomIntegrationRule = 1000` and
> stores per-element `GP_X` natural coordinates as an attribute on
> the element-connectivity dataset.
>
> The Phase 11a catalog model — `(class_name, int_rule, token) →
> ResponseLayout(n_GP, natural_coords, ...)` — was designed for
> fixed standard rules. Phase 11b introduces a parallel **custom-rule
> catalog** keyed only on `(class_name, token)`, plus a runtime
> resolver that builds a concrete `ResponseLayout` from the
> per-element `GP_X` array.

## Scope

Three interlocking pieces, each shippable independently once the
infrastructure lands:

### Piece 1 — Custom-rule catalog model

A new dataclass alongside `ResponseLayout` in
`src/apeGmsh/solvers/_element_response.py`:

```python
@dataclass(frozen=True)
class CustomRuleLayout:
    """Layout for elements whose IP geometry is per-instance.

    Used by force-/disp-based beam columns: the beam integration
    scheme (Lobatto, UserDefined, HingeRadau, …) is element-level
    metadata that varies independently of the C++ class. ``n_IP``
    and ``natural_coords`` come from the per-element ``GP_X`` array
    in MPCO, or from ``ops.eleResponse(eid, "integrationPoints")``
    at runtime.
    """
    n_components_per_location: int
    component_layout: tuple[str, ...]
    class_tag: int
    coord_system: str               # "isoparametric_1d"
    topology_level: str             # "line_stations" | "nodal_forces"


CUSTOM_RULE_CATALOG: dict[tuple[str, str], CustomRuleLayout] = {
    # (class_name, token) — int_rule is always Custom (1000).
    ("ForceBeamColumn3d", "section_force"): CustomRuleLayout(
        n_components_per_location=6,
        component_layout=(
            "axial_force", "shear_y", "shear_z",
            "torsion", "bending_moment_y", "bending_moment_z",
        ),
        class_tag=ELE_TAG_ForceBeamColumn3d,
        coord_system="isoparametric_1d",
        topology_level="line_stations",
    ),
    # ... ~12 entries for all beam-column families ...
}


def resolve_layout_from_gp_x(
    custom: CustomRuleLayout, gp_x: np.ndarray,
) -> ResponseLayout:
    """Build a concrete ResponseLayout from the catalog entry + per-element
    IP coords."""
    return ResponseLayout(
        n_gauss_points=gp_x.size,
        natural_coords=gp_x.reshape(-1, 1),
        n_components_per_gp=custom.n_components_per_location,
        component_layout=custom.component_layout,
        class_tag=custom.class_tag,
        coord_system=custom.coord_system,
    )
```

The existing `RESPONSE_CATALOG` (Phase 11a) stays for fixed-rule
elements. The two catalogs are non-overlapping (a class is in one or
the other; force-beam columns are *only* in `CUSTOM_RULE_CATALOG`).

### Piece 2 — `elements/line_stations/` topology

**MPCO read path** — extend `MPCOReader.read_line_stations` (currently
returns empty slabs). The challenge: each bucket's element layout
is per-element. Algorithm:

1. Walk `RESULTS/ON_ELEMENTS/<token>/<bracket_key>` for each token
   that maps to a line-stations component (e.g. `"section.force"`).
2. For each bucket, parse the bracket → `(class_tag, class_name,
   int_rule, custom_rule_idx, header_idx)`. For line-stations
   buckets, `int_rule == 1000`.
3. Look up `CUSTOM_RULE_CATALOG[(class_name, catalog_token)]`.
4. Read per-element `GP_X` from
   `MODEL/ELEMENTS/<class_tag>-<class_name>[1000:custom_rule_idx]`'s
   attributes. **Note**: `GP_X` is *per element*, but elements
   sharing the same `(rule_type, x_vector)` get the same
   `customRuleIdx` and share one `GP_X` array. So per-bucket the
   `GP_X` is uniform.
5. Call `resolve_layout_from_gp_x(...)` to build a concrete
   `ResponseLayout` for this bucket.
6. Read `META`, validate as in Phase 11a.
7. Read `DATA/STEP_<k>`, slice rows for filtered `element_ids`,
   `unflatten` via the resolved layout.
8. Return a `LineStationSlab` (already exists in `_slabs.py`).

**DomainCapture path** — new `_LineStationCapturer` class in
`_domain.py`. Per element:
1. On first `step()`, call `ops.eleResponse(eid,
   "integrationPoints")` to get the IP coords (returns Vector(n_IP)
   in `[-1, +1]` for Tier-1 elements).
2. Build the `ResponseLayout` via `resolve_layout_from_gp_x`.
3. Each step: for each IP `i = 1..n_IP`, call
   `ops.eleResponse(eid, "section", str(i), "force")` and stack
   into a `(T, E_g, n_IP × 6)` flat array.
4. At `end_stage()`, `unflatten` and write via
   `NativeWriter.write_line_stations_group(...)` (already exists).

**TXT transcoder path** — extend `RecorderTranscoder` to handle
line-stations records. The `.out` file's column layout depends on
the `recorder Element ... section force` form; needs the
custom-rule resolver to know how many columns per element and how
to demux into stations × components.

**Coverage target** for v1:
- `ForceBeamColumn2d` / `3d`
- `DispBeamColumn2d` / `3d`
- `ElasticForceBeamColumn2d` / `3d`
- `ForceBeamColumnCBDI2d` (CBDI variant)
- `ForceBeamColumnWarping2d`

That's ~9 classes × 2 tokens (section force + section deformation) =
~18 entries.

### Piece 3 — `elements/nodal_forces/` topology

Simpler than line_stations — no per-station coordinate, just a
per-element-node force vector.

**Catalog entries** — fixed shape per class (no custom rule):

```python
("ElasticBeam3d", "global_force"): NodalForceLayout(
    n_nodes_per_element=2,
    n_components_per_node=6,
    component_layout=(
        "force_x", "force_y", "force_z",
        "moment_x", "moment_y", "moment_z",
    ),
    class_tag=ELE_TAG_ElasticBeam3d,
    frame="global",
    topology_level="nodal_forces",
)
```

Or — alternative design — the elastic beams could go in
`RESPONSE_CATALOG` with a fixed (synthetic) "rule" like
`NodalForce_2node_3D = -1`, mirroring the Phase 11a pattern.
Decision deferred until Round B starts.

**MPCO read path** — extend `MPCOReader.read_elements`. MPCO writes
elastic-beam global forces under `ON_ELEMENTS/globalForce/<bucket>/`
with a column layout of `n_nodes × 6` (or `× 3` in 2-D). Reader
unpacks per-node.

**DomainCapture path** — new `_NodalForcesCapturer` calling
`ops.eleResponse(eid, "globalForce")` (returns Vector(12) for a 3-D
2-node beam) per step, reshaping to `(T, E_g, n_nodes, 6)`.

**Coverage target**:
- `ElasticBeam2d` / `3d`
- `ElasticTimoshenkoBeam2d` / `3d`
- `ModElasticBeam2d`
- `elasticBeamColumnIO` (if present)

5 classes × 2 tokens (global / local force) = ~10 entries.

## Vocabulary additions

Section-force / line-diagram components — already partially in
`_vocabulary.LINE_DIAGRAMS`:

```python
LINE_DIAGRAMS: tuple[str, ...] = (
    "axial_force",
    "shear_y", "shear_z",
    "torsion",
    "bending_moment_y", "bending_moment_z",
)
```

These are scalar canonicals (no axis suffix). The Phase 11a
`split_canonical_component` + full-name fallback in
`gauss_keyword_for_canonical` already handles this pattern (proven
in Round A for `axial_force`). Phase 11b adds the keyword routing:

```python
_GAUSS_PREFIX_TO_KEYWORD.update({
    "axial_force":      "section.force",  # scalar — full name
    "shear_y":          "section.force",
    "shear_z":          "section.force",
    "torsion":          "section.force",
    "bending_moment_y": "section.force",
    "bending_moment_z": "section.force",
})
```

For nodal forces — use existing `force_*` / `moment_*` from
`NODAL_FORCES`. The token mapping:

```python
"force":  "globalForce",   # or "localForce" — frame-dependent
"moment": "globalForce",
```

Note the `force` / `moment` prefixes already exist in
`_GAUSS_PREFIX_TO_KEYWORD` (Phase 11a Round A doesn't claim them).
Wiring needs to disambiguate `force_x` (nodal) from `axial_force`
(line-station scalar) — different topology levels routed to
different keywords. Doable via a per-topology routing table
instead of one global table.

## Custom-rule complexity (the hardest part)

The element-compatibility skill's three-tier discovery for
`getCustomGaussPointLocations` (`MPCORecorder.cpp:4089–4265`)
already encodes the OpenSees-side complexity:

- **Tier 1**: `ops.eleResponse(eid, "integrationPoints")` — works
  for all `ForceBeamColumn*`. Returns the `nIP`-vector directly.
- **Tier 2/3**: probing via `setResponse(["section", "dummy"], ...)`
  — needed for `DispBeamColumn*` (no `integrationPoints` response).
- **Special case**: `MVLEM_3D` family swaps the keyword to
  `"material"` (1-D MVLEM rules in the cohesive-band sense).

For DomainCapture, `ops.eleResponse(eid, "integrationPoints")` works
for all force-based beams. Disp-based beams need the Tier 2 fallback,
which is non-trivial to invoke from Python — may need a special
path or accept that DomainCapture for disp-based beams requires
the user to query MPCO instead.

For MPCO read, every beam-column's `GP_X` is already on disk —
trivial to read. The per-element customization is fully handled by
MPCO's `customRuleIdx` indexing, so different IP layouts naturally
land in different buckets.

## Naming question — section_force vs separate components

The MPCO recorder writes line-station data under `ON_ELEMENTS/
section.force/<bucket>/`. Internally the META describes:

```
META/COMPONENTS = "1.section.0.axial,shear_y,shear_z,torsion,My,Mz;
                   1.section.1.axial,...;
                   ...;
                   1.section.<n_IP-1>.axial,..."
```

(Pseudo — exact form per `mpco-recorder` skill.) Each section's
6 components per-IP, GP-slowest, exactly matching the Phase 11a
unflatten pattern.

For the apeGmsh canonical components, the existing `LINE_DIAGRAMS`
names map naturally:
- MPCO `axial` → apeGmsh `axial_force`
- MPCO `shear_y` → apeGmsh `shear_y`
- MPCO `Mz` → apeGmsh `bending_moment_z`

Component-name translation: a small dict in `_mpco_translation.py`
parallel to the existing nodal one.

## Estimated effort

| Piece | LoC (src) | LoC (tests) | New files |
|---|---:|---:|---:|
| 1. Custom-rule catalog model | ~250 | ~150 | 0 |
| 2a. line_stations MPCO read | ~300 | ~200 | 1 (`_mpco_line_io.py`) |
| 2b. line_stations DomainCapture | ~250 | ~200 | 0 (extend `_domain.py`) |
| 2c. line_stations TXT transcoder | ~200 | ~150 | 0 |
| 3a. nodal_forces MPCO read | ~150 | ~100 | 0 |
| 3b. nodal_forces DomainCapture | ~150 | ~100 | 0 |
| Catalog entries + vocab | ~100 | ~150 | 0 |
| Real-openseespy fixtures | — | ~300 | 1 (`test_results_catalog_lines_real.py`) |
| **Total** | **~1400** | **~1350** | **2** |

Roughly 1.5× the size of all three Phase 11a element-transcoding
sites together, mostly because the custom-rule resolver touches
both the read path and the capture path with very different
upstream APIs.

## Real upstream issues to expect

Based on Phase 11a Rounds A–E:

- `DispBeamColumn*` element-level "stresses" probe likely returns
  zeros (same pattern as BbarBrick / older shells); MPCO works around
  via material-level probing. Catalog correct, DomainCapture
  may fail for disp-based beams in some OpenSees builds.
- `ElasticBeam*` is closed-form, no per-IP state — `globalForce`
  and `localForce` work; `"stresses"` likely raises (no IPs to
  iterate). Catalog should not declare a `gauss` entry for them.
- `Hinge*` integration schemes have known edge cases at the
  hinge → elastic-segment interface; the IP-coord array can have
  zero-weight markers that need filtering before catalog use.

## Build sequence

The dependencies suggest this order:

1. **Custom-rule catalog model** (Piece 1) — purely additive, can
   be merged independently. Adds `CustomRuleLayout` and
   `CUSTOM_RULE_CATALOG` with ~12 entries; adds
   `resolve_layout_from_gp_x`. No reader/capture changes yet.
2. **MPCO `read_line_stations`** (Piece 2a) — opens the value
   stream first; lots of users have `.mpco` files from STKO
   workflows and want to read them. Requires fixture: a real
   `.mpco` from a `forceBeamColumn` model.
3. **DomainCapture `_LineStationCapturer`** (Piece 2b) — extends
   the in-process capture. Real-openseespy validation against the
   fixture from step 2.
4. **TXT transcoder for line-stations** (Piece 2c) — emit `.out` →
   custom-rule resolver → write native HDF5. Validates against
   real Tcl-subprocess output.
5. **`nodal_forces` topology** (Piece 3) — simpler than 2; lands
   after the line_stations pattern is proven.

## Decisions to nail down before starting

1. **CustomRuleLayout vs extending ResponseLayout**? Two dataclasses
   keeps the type system honest (a fixed-rule layout always has
   coords; a custom-rule layout doesn't). Lean: separate dataclass.
2. **Per-topology keyword routing**? Phase 11a has a single global
   `_GAUSS_PREFIX_TO_KEYWORD` → adequate for stress/strain/shells.
   Adding line-station scalars (`axial_force` for line stations)
   conflicts with Round A's `axial_force` for trusses. Need
   topology-aware routing. Lean: pass a topology hint to
   `gauss_keyword_for_canonical(name, topology="line_stations")`,
   default `"gauss"`.
3. **Element-class hint plumbing on capture / transcoder**? Already
   plumbed for the .out transcoder via
   `ResolvedRecorderRecord.element_class_name` (Phase 11a). Should
   capture also use it (vs `ops.eleType`)? Lean: capture continues
   to use `ops.eleType` (faster and authoritative); the spec hint
   is just for the .out transcoder.
4. **Disp-based beam-column DomainCapture**? Tier 2/3 fallback is
   complex. v1 may declare DomainCapture only works for force-based
   beams; disp-based beams via MPCO only. Document and move on.

## Out of scope for Phase 11b

Pushed to later phases:

- `MVLEM_3D` / `SFI_MVLEM_3D` family (cohesive-band 2-D rules,
  `material X`-keyword swap)
- ZeroLength element family (per-DOF spring forces — different
  topology again, simpler than line_stations but with its own
  component naming)
- Bearings (`ElastomericBearing*`, `LeadRubberX`, `HDR`,
  `ElastomericX`) — these are 1-GP line elements but their response
  is per-DOF (similar to ZeroLength). Phase 11d.
- Fiber-section through-thickness probing for layered shells —
  Phase 11c (`elements/fibers/` and `elements/layers/` topologies)

## See also

- [[plan_element_transcoding]] — Phase 11a parent plan (already shipped)
- [[Results_architecture]] — schema and reader-protocol spec
- `references/integration-rules-and-gauss.md` (mpco-recorder skill) —
  authoritative reference for the three-tier discovery + GP_X format
- `references/element-compatibility.md` (mpco-recorder skill) — full
  catalog of element-class → (geometry, rule) mappings
- `src/apeGmsh/solvers/_element_response.py` — Phase 11a catalog
  (extends in this plan)
