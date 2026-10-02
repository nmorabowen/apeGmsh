# ADR 0111 — STKO translator: a mesh-faithful `.scd` → session + apeSees deck, driven by an `XOBJ_META` registry

**Status:** Proposed (2026-10-02). Interop-side feature on top of the `.scd`
reader (`apeGmsh.interop.stko`, PR #1267) plus one opt-in bridge option
(§D2, element tags = FEM element ids). No Emitter Protocol change, no schema
change. Sibling of [ADR 0072](0072-etabs-bridge-decompose-build.md) (the ETABS
interop, whose `translate → get_fem_data → build_opensees` shape this mirrors).

**Implementation contract:** `internal_docs/stko_translator_rules.md` holds the
exact module interfaces, the shared dataclasses (verbatim), every export rule
with its numeric evidence, the type registry table and the deck-parity
contract. This ADR records the decisions and why; the rules file is what the
implementers build against.

**Sources read for this ADR:** STKO's own exporter, installed with STKO at
`C:\Program Files\STKO\external_solvers\opensees\` (`conditions/`,
`element_properties/`, `physical_properties/`, `analysis_steps/`,
`utils/write_node.py`, `utils/time_increment_utils.py`); the San Ramon Tier-1
documents and STKO Tcl exports (research repo
`models/stko-rev0-tcl/Tier_1/{1A,1B,1C,1D}/`: 1A's `input files/`, and the
campaign decks `campaign_sta2_rup1/` copied read-only from esmeralda); the
reader (`src/apeGmsh/interop/stko/`); the ETABS interop
(`src/apeGmsh/interop/etabs_import.py`); the bridge's element-tag allocator
(`opensees/_internal/build.py::allocate_element_tags`, `ElementPlanRows`).

## Context

The San Ramon port moves 16 STKO models (Tiers 1–4 × cases A–D) to apeGmsh. The
building is rebuilt parametrically by hand (`sanramon/` in the research repo),
and each case needs two things the hand-built recipe cannot give itself: an
**automatic reference** (the same model STKO ran, to compare against) and a
**definitions source** (section, material, load and mass values keyed to the
model's groups). Reading values off the Tcl export does not scale: the Tcl has
no geometry, no group names, no load-case names, is split over MPI partitions,
and 1A's `analysis_steps.tcl` alone is 7 MB of per-element-corner `load` lines.

The `.scd` (HDF5) has everything: the mesh with STKO's node and element IDs,
selection sets, element/physical-property assignments per sub-shape, conditions,
interactions, definitions and analysis steps, each object typed by its
`XOBJ_META`. The reader exposes it as plain data. What is missing is the
translation into an apeGmsh session and an apeSees deck — and the translation
must be **mesh-faithful**: same nodes, same elements, same IDs as STKO, so a
deck diff against STKO's own export is a real proof and results can be compared
node by node.

Facts that shaped the decisions (all from the sources above):

- `MESH/ELEMENTS` holds every meshed entity; OpenSees receives only
  `ScdModel.analysis_elements()` (sub-shapes with an element property, plus
  interaction "link" elements).
- STKO's exporter derives everything per element from the mesh: `-local` is
  column 0 and `vecxz` column 2 of the element's orientation quaternion
  (`MESH/ELEMENT_ORIENTATION_QUATERNIONS`); shell connectivity is rotated by one
  position when the local x is closer to the element's η direction than to ξ.
- Distributed conditions are lumped per element with the consistent shape-function
  integral (∫Nᵢ q dA), not tributary area; masses come from **every** `Mass.*`
  condition in the document, loads only from conditions a load pattern references.
- Documents carry unused template properties; STKO exports them anyway. Only what
  is reachable from assigned elements matters.
- apeSees allocated element tags sequentially per spec; before this ADR it had
  no way to emit a FEM element id as the OpenSees tag.

## Decision

### D1 — Mesh route: gmsh discrete entities carrying STKO's own mesh

The STKO mesh enters the session directly through gmsh's discrete-entity API
(`addDiscreteEntity`, `mesh.addNodes`, `mesh.addElementsByType`) with **STKO's node
tags and element tags**. One discrete entity per *referenced* STKO sub-shape
`(geometry, kind, index)` — referenced by an element property, a condition, a
selection set or an interaction — split further only if one sub-shape's analysis
elements carry more than one local axis (never the case in Tier 1). Physical
groups are unions of these entities:

- one **element group** per `(element property, physical property, local axis)` —
  the key every element declaration uses (`ops.element.X(pg=...)`);
- one PG per selection set (per kind when a set mixes kinds);
- one PG per condition (per kind);
- per rigid-diaphragm master: a master PG and a slave PG on point "carrier"
  entities that **own** those nodes (§D4).

Non-analysis mesh elements on referenced sub-shapes (shell edge meshes, vertex
point elements) stay in the session as *carriers*: they make PG node sets equal
STKO's sub-shape node sets and are never declared, so never emitted. Vertex
point elements and diaphragm carriers get synthetic ids above `max(STKO id)`.
Each node is classified (owned) on exactly one entity, in the priority
diaphragm carriers → vertices → edges → faces → solids; `g.constraints`
resolves nodes by ownership (`getNodes`), while `FEMData` PG node sets come from
element connectivity, so ownership matters only for diaphragms. The session must
**never** call `g.mesh.generation.generate()` or `g.mesh.partitioning.renumber()`
(either destroys the STKO mesh or its IDs); `get_fem_data(dim=None)` is called
directly.

### D2 — STKO element IDs are the OpenSees element tags (opt-in bridge option)

`apeSees(fem, element_tags="fem")` makes every PG-fanned element's OpenSees tag
its FEM element id, and advances the element counter past `max(eid)` so any
bridge-synthesised element (node-pair specs, springs) is allocated above it. The
default stays `"sequential"` (byte-identical to today). The change is local to
`allocate_element_tags` (`ElementPlanRows` already carries explicit per-row
`tags`, and `FemToOpsTagMap.from_plan` already honours them); the implementer
audits every other `allocate("element")` call site so none can land below the
FEM range. `build_opensees` requests `"fem"` (its default) and raises
`TypeError` on a bridge without the option rather than renumbering silently;
`element_tags="sequential"` keeps the bridge's numbering. The parity check
(§D8) requires our element tags to equal STKO's ids and compares elements by
tag, with the connectivity bijection kept as a separate row. Section,
material, transform and pattern tags are **not** preserved: they are compared
structurally.

*Implemented (2026-10-02):* `apeSees(fem, element_tags="fem")` reserves the FEM
element-id range right after seeding the tag allocator
(`build.reserve_fem_element_tags`: refuses an id ≤ 0 or an element fanned out
twice, and moves the counter past the snapshot's largest element id, so a
synthesised tag never reuses a FEM id even of a non-emitted point element);
`allocate_element_tags(..., element_tags=)` gives each PG row `tags=eids`, on
the flat, staged, split, partitioned and partitioned-staged paths. A source
test pins every element-tag allocation site. `MeshSelection.ids`,
`.connectivity` and its `(eid, conn)` iteration now share one order (the
connectivity used to follow storage order while `.ids` did not).

### D3 — A registry keyed by `XOBJ_META`; anything else fails loudly, listed in full

Every STKO type the translator can meet is a registry entry with a status:
`supported`, `tier_only` (known, deliberately not in this version: `stdBrick`,
`ElasticIsotropic`, `ASDAbsorbingBoundary3DAuto` and its material, `zeroLength` /
`zeroLengthMaterial` / `uniaxial.Elastic`, `ASDEmbeddedNodeElement`, `H5DRM` load
and pattern, `ASDAbsorbingBoundaryActivate`) or `ignored` (explicitly not part of
the model: recorders, regions, monitors, raw `customCommand` Tcl, and
`ImplexAutoErrorControlActivate` *only when no analysis follows it* — before an
analysis it changes how that analysis runs and is refused, the same position rule
as a pattern after the last analysis). Before touching the session,
`translate_scd` scans what STKO would export — assigned element properties, the
physical-property closure reachable from them (including fiber-group materials
held inside a fiber section's custom object, which are not `INDEX` attributes),
every `Mass.*` condition, every condition a pattern references, every referenced
definition, every analysis step, every interaction and the element / physical
properties it carries (2A's `zeroLength` links are named, not just "interaction"),
every mesh element type, and the node set (the session must hold exactly the
analysis elements' nodes plus the referenced diaphragms' nodes, and those must be
every mesh node, because STKO writes them all) — and
also checks per-type options it does not translate (`Mode = function`,
non-standard beam integration, section offsets, releases, `implexAlpha ≠ 1`,
non-empty `eleLoad`/`sp`/`genericLoad` in a pattern, a non-square
`sections.Elastic` whose `PROPS[1]`/`PROPS[2]` → `Iyy`/`Izz` order is unverified, a
static stage after the transient stage, …). Anything not
`supported` and not `ignored` goes into one `UnsupportedSTKOTypes` error that
lists **every** offender (category, `XOBJ_META`, ids, names, reason) — never the
first one, never a warning, never a silent skip. `ignored` items are returned in
`TranslateResult.ignored` so the omission is visible. Adding a type later is one
registry entry plus its rule.

### D4 — Conditions are reproduced by STKO's own rules, declared at node level

The translator computes STKO's nodal values itself (pure numpy over the
`ScdModel`) and declares them at the bridge per node: `ops.mass(nodes=[n], ...)`,
`p.load(node=n, ...)` inside each pattern, `ops.fix(nodes=..., dofs=mask)`. It does
**not** route FaceMass/FaceForce through `g.masses.surface` / `g.loads.surface`:
those apply apeGmsh's reductions, which are not STKO's rules (consistent lumping,
AutoEdgeMass's averaged section area, local-axis rotation), and exact parity is the
point. Each condition still gets its PGs and a `ConditionSummary` (type, per-unit
value, PGs, patterns using it), which is the *definitions source* the parametric
recipes read. Rigid diaphragms are the exception that goes through the session:
`g.constraints.rigid_diaphragm(master_pg, slave_pg, ...)` per STKO master, on
carriers that own exactly STKO's link nodes, so the public constraint path
resolves them (the ETABS route injects private `NodeGroupRecord`s; not repeated).
apeGmsh has no node-level master/slave diaphragm API, and the resolver is
geometric (it keeps nodes within `plane_tolerance` of the master's plane and picks
the node nearest `master_point`), so the translator makes the geometry reproduce
STKO's links verbatim: the carriers hold only `{master, *slaves}`, `master_point`
is the master's own coordinates (a slave at those coordinates is refused), and
`plane_tolerance` exceeds the farthest slave's offset (OpenSees keeps an
off-plane slave, so STKO's pair must not be dropped). The carriers are built from
the same groups as the plan's pairs (`diaphragm_groups`: referenced conditions
only), and after `get_fem_data` the resolved records are asserted equal to
`plan.diaphragm_pairs` (`verify_diaphragms`, run by `build_conditions`).

### D5 — Properties map value-for-value; the only computation is STKO's own

Sections and materials are built from the `.scd` attribute values with STKO's
exporter formulas — not recomputed from physics. Two places need STKO's
computation ported verbatim: the `ASDConcrete3D/1D` "Concrete (9P)" preset
(backbone tables and `-autoRegularization` length) and fiber extraction (the fiber
section's stored fibers, offset by its stored centroid). Raw primitives are
registered with `ops.register(...)` where the namespace only offers a
physics constructor (`ASDConcrete3D.from_fc` is not STKO's law).

### D6 — Analysis steps: model-defining steps are translated, drivers are data

Load patterns (Plain with conditions + `massToLoad`, UniformExcitation), the
constraint pattern, Rayleigh damping and the static stage sequence
(`AnalysesCommand` with `loadConst`) are translated; static stages are emitted
through `ops.stage(...)`, each starting with STKO's IMPL-EX reset of its targets
(`dTimeCommit`, `dTimeInitial`, `dTime` = `duration/numIncr`, through
`s.update_parameter`). The transient stage, STKO's adaptive time-step driver
(the test written with `2 × iter` when adaptive, the desired count kept apart),
and the per-step `dTime` update are returned as data (`StageSpec`,
`ConditionsPlan.implex_dt_targets`); the San Ramon case scripts own the
time-history driver, and once the static reset ran the driver *must* set `dTime`
every transient step, as STKO's transient template does. Time-series values come from the `.scd`
(a one-value placeholder in every San Ramon document) unless the caller passes the
real record (`records=`).

### D7 — Four modules, one public entry, the ETABS shape

`translate_mesh.py` (mesh + groups), `translate_props.py` (element/section/material
registry), `translate_conditions.py` (masses, loads, fixities, diaphragms, time
series, patterns, damping, stages), and `translate.py` (registry scan, public
entry, bridge wiring), exchanging frozen dataclasses from `translate_types.py`
(fixed verbatim in the rules file so three implementers work in parallel):

```python
scd = read_scd("1C_TH_000.scd")
with apeGmsh(model_name="1C") as g:
    result = translate_scd(g, scd)                 # raises UnsupportedSTKOTypes
    fem = g.mesh.queries.get_fem_data(dim=None)    # no generate(), no renumber()
ops = build_opensees(fem, result)                  # element_tags="fem", stages=True
ops.tcl("1C.tcl")
```

As in ADR 0072, the pieces stay callable separately (`build_props`,
`build_conditions`), so a recipe can take STKO's elements and replace its loads, or
the reverse.

### D8 — Deck parity against STKO's own export is the acceptance gate

The integrator ships a canonicaliser for STKO Tcl exports and apeSees decks
(partition-merged: union of repeated nodes/elements/fixes, the mass from the one
copy that carries it, loads summed per pattern and node, diaphragm pairs
deduplicated; tags resolved to structural signatures) and a per-category
comparison: IDs (element tags included) and connectivity exact, element
options exact, local axes to STKO's print precision, section/material values
relative 1e-12 over the reachable closure, nodal masses and loads relative 1e-9
per node, fixity and diaphragm sets exact (the plan's pairs too), stage
structure exact, the IMPL-EX reset exact, and the transient stage's solver
chain (test `max_iter` included) exact against the plan. The full table is in
the rules file. A case is translated when every gated category passes.

## Evidence (2026-10-02)

- **Rules.** A scratch prototype of every rule (`stko_rules.py` + a canonicaliser,
  `verify.py`) checked against the four Tier-1 STKO exports: **132 checks, 132
  pass** (1A 25, 1B 33, 1C 37, 1D 37). Nodal mass max |Δm| 5.4e-10 t (STKO prints 10
  significant digits) with totals equal to the last digit (1A/1C/1D 11720.837250 t,
  1B 11736.737000 t); per-pattern nodal loads max |ΔF| ≤ 9e-11 N (1A pattern 9:
  14003 nodes from 53344 load lines); fixity sets exact (1551 / 1563 / 212 / 1551
  nodes); diaphragm pairs exact (1B 360, 1C 840); ASDConcrete backbones relative
  error 0 and `lch_ref` exact (6.377551020408164, 4.081632653061225,
  3.188775510204082); fibers exact (1B 273, 1C/1D 276); shell `-local` and beam
  `vecxz` exact on every element (12968 / 736 in 1A); shell node rotation applies to
  11134 of 12968 1A shells.
- **Mesh route.** Spike on 1A (one shell group + one beam group + fix carriers):
  2287 emitted node IDs equal STKO's, coordinates within STKO's print precision
  (4.3e-6 mm on 4.1e4 mm), 2208 elements with identical connectivity. Full scale
  (4381 discrete entities, 68 PGs, 9175 carrier elements, entities 2.7 s +
  `get_fem_data` 1.2 s): nodes, elements (13704, connectivity and node order),
  fixity (1551) and nodal mass identical to STKO's export for 1A and 1D; 1B and 1C
  additionally reproduce every rigid-diaphragm pair through
  `g.constraints.rigid_diaphragm`. With the §D2 option prototyped as a scratch
  monkeypatch of `allocate_element_tags` (≈15 lines), every element tag equals the
  STKO id (13704/13704 in 1A; 0 without it). With the option implemented, the
  translated decks carry STKO's element ids in all four cases (1A 13704, 1B 2456,
  1C 3704, 1D 13160; every gated parity row passes), and the 1A and 1B gravity +
  eigen smoke against STKO's deck run serially agree to 5.0e-11 and 3.4e-13 in
  T1–T6.
- **Solve.** The full-scale 1A spike deck run on the Ladruno build
  (`piles-bin/fd87e396d`, eigen 6) gives T1–T6 = 1.66477, 1.65103, 1.09871, 0.33839,
  0.33243, 0.21159 s; STKO's own run of 1A gives 1.6648, 1.6510, 1.0987, 0.3384,
  0.3324, 0.2116 s (research repo `validation/parity_1A.md`).

## Alternatives considered

- **Re-mesh the BREP (`write_brep` → `load_brep` → `generate`).** Rejected: not
  mesh-faithful; IDs and the shell node order are lost, so neither the deck diff nor
  node-by-node results comparison is possible. It stays the route of the
  hand-built parametric model.
- **Write a `.msh` and `g.loader.from_msh` / `g.model.io.load_msh` it.** Equivalent
  in result (gmsh keeps tags from `.msh`), but adds a file format round trip and a
  second writer to keep in step; the discrete-entity API is the same data without
  the detour.
- **One entity per element group only.** Rejected: selection sets and conditions
  address sub-shapes that cut across element groups, so they would need per-node
  PGs; per-sub-shape entities make every STKO grouping a plain union.
- **Declare FaceMass / FaceForce through `g.masses` / `g.loads` reductions.**
  Rejected for the parity path (§D4): different lumping rules and no AutoEdgeMass
  area averaging. The condition PGs keep that route open for the recipes.
- **Inject diaphragm `NodeGroupRecord`s into the snapshot (the ETABS route).**
  Rejected: private API; the public `rigid_diaphragm` reproduces STKO's pairs exactly
  once the carriers own the nodes.
- **Renumber the emitted deck's element tags afterwards.** Rejected: recorders,
  regions and the model.h5 tag maps would disagree with the deck.
- **Warn and skip unknown types.** Rejected by requirement: a silently missing
  property, load or mass is the failure this translator exists to prevent (§D3).

## Consequences

- One small bridge change (§D2), opt-in, default byte-identical.
- FEMData of a translated session carries carrier elements (line2 edge meshes,
  point elements) that are never declared; 9175 of them in 1A. Anything that
  iterates *all* FEM elements must expect them.
- STKO quirks are reproduced on purpose and named in the rules file: AutoEdgeMass
  uses the mean area over the edge's section and the sections it references, and a
  fiber section's area is its surface fibers only; `lch_ref` is a hundredth of the
  tension/compression minimum; STKO lists non-analysis edge elements among its
  IMPL-EX `dTime` targets (the translator lists analysis elements only).
- Allowed differences, never gated: section/material/transform/pattern tags; one
  `geomTransf` per element (STKO) vs one per local axis (apeSees); inline
  `forceBeamColumn ... Lobatto sec np` vs a `beamIntegration` object; STKO's
  export of unreachable template properties; defaults written explicitly on one side
  only (`-implexAlpha 1.0`, `-eta 0.0`, Hysteretic `beta 0`).
- Not decided here, not verified: `g.save()` / `model.h5` round trip of a
  translated session; Tier 2–4 types (they fail loudly until added); the transient
  driver and its per-step `dTime` update; which of `PROPS[1]` / `PROPS[2]` is
  `Iyy` (non-square elastic sections are refused until a case decides it).
