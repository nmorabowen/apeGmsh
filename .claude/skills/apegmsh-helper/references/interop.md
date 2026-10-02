# `apeGmsh.interop` — import analytical structural models (ADR 0009)

<!-- skill-freshness: verified against apeGmsh main@8eeda7a3 (2026-07-06) · signatures: python -m apeGmsh.studio.lookup SYMBOL (ADR 0096); src/ is not the authoring lookup -->

`apeGmsh.interop` turns an **analytical** structural model (joints, frames,
areas, restraints, loads — exported from apeETABS as a neutral `*.sm.json`)
into a **conformal beam + shell FE mesh**, then optionally into an `apeSees`
OpenSees deck. It is a **standalone module** (separate import, NOT a session
composite).

```python
from apeGmsh.interop import (
    StructuralModel, import_structural_model, apply_subgrade_springs,
    build_opensees, solve_and_extract, SolveResult, ImportResult,
    FrameGroup, AreaGroup, RestraintGroup, DiaphragmSpec, SpringGround,
)
```
`# src/apeGmsh/interop/__init__.py` (exports verified verbatim)

## The pipeline

```python
model = StructuralModel.from_json("wall_slab_frame.sm.json")   # interop/model.py

with apeGmsh(model_name="bldg") as g:
    result = import_structural_model(g, model)     # geometry + PGs + loads + masses
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=2)              # dim=1 for frames-only models
    g.mesh.partitioning.renumber(base=1)
    apply_subgrade_springs(g, model, result)       # optional — after mesh, before fem
    fem = g.mesh.queries.get_fem_data(dim=None)    # the usual FEMData snapshot

ops = build_opensees(fem, model, result)           # → an apeSees bridge
ops.tcl("bldg.tcl")                                # or .py(...) / native apeSees API
```

The output of the import is an ordinary `FEMData` snapshot fed to the
ordinary `apeSees` bridge — interop is a *front door*, not a parallel
solver path. Everything downstream (`results.md`, `opensees-bridge.md`) is
unchanged.

## Entry points

```python
import_structural_model(g, model: StructuralModel, *, self_mass=True) -> ImportResult
#   src/apeGmsh/interop/etabs_import.py:134
#   Builds on the live session g: nodes→points, frames→shared lines grouped by
#   (section, orientation) into PGs, areas→conformal shell surfaces into PGs,
#   restraints→point PGs per DOF mask. Declares loads on g.loads (nodal, frame
#   tributary, area traction) grouped by pattern, and (self_mass=True) lumped
#   self-mass from material density. Returns ImportResult metadata.

apply_subgrade_springs(g, model, result) -> int                # etabs_import.py:297
#   Run AFTER meshing + renumber(), BEFORE get_fem_data(). No-op without springs.
#   Point springs: per-node 6-diagonal k in the global frame.
#   Area (subgrade/Winkler) springs: per-unit-area k in the area LOCAL axes
#   (U3 = surface normal), distributed by tributary area as oriented zeroLengths —
#   correct for inclined/vertical areas. Returns the count of grounded springs.

build_opensees(fem, model, result, *, ndm=3, ndf=6,
               shell_element="ASDShellT3") -> apeSees           # etabs_import.py:508
#   Wires the elastic deck: elastic beam-columns (one geomTransf per orientation),
#   ASDShellT3 + ElasticMembranePlateSection, fixities, oriented grounded springs,
#   self-mass (ops.mass_from_model), one Linear timeSeries + Plain pattern per load
#   pattern (p.from_model(name)), and injects non-shell-backed rigid diaphragms as
#   RIGID_DIAPHRAGM constraints. Returns the apeSees bridge (you choose .tcl()/.py()).
#   Areas carrying ETABS property modifiers get a LadrunoShellModifier wrapper
#   around their plate section -- such a deck needs a LADRUNO build, not stock.

solve_and_extract(model, *, case=None, global_size=1.0, ndm=3, ndf=6,
                  tol=1e-6, max_iter=50) -> SolveResult         # interop/solve.py:46
#   All-in-one static solve (needs openseespy). model = StructuralModel or *.sm.json
#   path. Meshes, applies springs, runs a linear static analysis, returns joint-keyed
#   results. case= picks a load pattern (default: first; solved in isolation, no
#   superposition). Finer global_size → closer to the analytical field.
```

## Area property modifiers (cracked sections)

An `Area` may carry `modifiers` -- the ten-entry ETABS OAPI array
`[f11, f22, f12, m11, m22, m12, v13, v23, mass, weight]`, all defaulting to 1.0.
The producer writes the *effective* modifiers already in force on that object
(ETABS lets an area object override its section property; resolving that
override is ETABS-side knowledge).

```json
{"id": "W1", "nodes": ["1","2","6","5"], "section": "WALL",
 "modifiers": {"f11": 0.35, "f22": 0.35, "f12": 0.35}}
```

Three things this changes, all of which were silent bugs before:

- **Areas bucket by `(section, modifiers)`, not by section name.** Two walls on
  one section can legitimately be cracked differently. A section with more than
  one distinct modifier set gets `__m1`/`__m2`/... PG suffixes, ordered by the
  modifier values so they are stable across runs; `AreaGroup.modifiers` records
  which is which. A section with a single variant keeps its bare name.
- **The nine stiffness modifiers ride a `LadrunoShellModifier`** wrapper (fork
  ADR 91) around the plate section. An all-1.0 set is treated as gross and emits
  no wrapper.
- **`weight != 1.0` is REFUSED** at import with an actionable message. OpenSees
  ties shell self-weight to the same density as the mass matrix, so it could
  only alias `mass`. Scale self-weight at the load level instead.

The `mass` modifier is applied to the *lumped* areal density as well as the
section, because this path lumps shell mass on the apeGmsh side
(`g.masses.surface`) and would otherwise ignore it.

Two things NOT to expect from this path. Walls cracked with `m11`/`m22`/`m12`
come through unchanged in plane — in-plane bending is membrane action, so only
`f11`/`f22`/`f12` soften a wall (fork gate G10). And results are not expected to
match ETABS bit-for-bit: the congruence used for the Poisson coupling term is
not known to match CSI's for strongly unequal `f11`/`f22`, and that parity is
deliberately not validated. The accepted validation is the fork's frame-vs-shell
equivalence cross-check.

`SolveResult` (`solve.py:30`): `case`, `displacements: dict[str, Vec6]`
(etabs joint id → Ux,Uy,Uz,Rx,Ry,Rz), `reactions: dict[str, Vec6]`,
`converged: bool`, `n_mesh_nodes: int`. Only mesh nodes that coincide with
input joints are keyed back (interior mesh nodes have no ETABS id);
`reactions` covers supported joints (restraints + point springs + area-spring
boundary nodes). It is the apeGmsh half of the ADR 0009 cross-check —
output aligns 1:1 by joint id with apeETABS.

## ADR 0072 — `apeSees` is THE modeling surface (the durable principle)

The interop's job ends at handing off **labelled geometry**: ETABS section
and material **names** become PGs (`"COL400"`, `"SLAB200"`, with a
`__v`/`__h` orientation suffix when a section spans both), restraint / load
/ diaphragm PGs are named, and joint ids are preserved on `FEMData`. ETABS
is a regenerable **scaffold** (geometry, loads, supports, masses), **not**
the research model — that's the scaffold *plus your own modeling decisions*
(fiber columns, plastic-damage concrete, force-based elements, refined
mesh). You make those decisions in the **native `apeSees` API** with `pg=`
selectors keyed on the stable section names — there is deliberately **no
overlay DSL** (a `ModelProfile`-style grammar was explicitly rejected).

```python
# Geometry / mesh / extra constraints: the native g API, between
# import_structural_model and get_fem_data (§5 of the ADR — already available).
g.mesh.sizing.set_size_by_physical("COL400", 0.1)
g.constraints.equal_dof(...)
```

> ⚠️ **Status: ADR 0072 is *Proposed* (2026-06-25), not yet implemented.**
> The decomposition into public, composable emit steps
> (`emit_elements(skip={"COL400"})` to emit the elastic default for every
> section *except* the ones you model yourself, plus `inject_diaphragms` /
> `emit_supports` / `emit_loads` / `emit_mass`) is the **planned** seam — it
> does **not** exist on `main`. Today `build_opensees` is still
> **all-or-nothing**: you can only **add** native `apeSees` declarations
> *after* it (which would double up on a group it already emitted), not
> *replace* its elastic default for a subset. Verify against
> `src/apeGmsh/interop/etabs_import.py` (the decomposition functions are
> still the private `_inject_diaphragms` / `_emit_springs`) before relying on
> the `skip=`/`only=` API.

## STKO `.scd` documents — `apeGmsh.interop.stko`

STKO (ASDEA's OpenSees pre/post-processor) saves its CAE document as HDF5.
`read_scd` reads it into plain frozen dataclasses (no gmsh, no OCC, no
OpenSees): geometries with per-sub-shape property assignments, the mesh,
selection sets, physical / element properties, conditions, interactions,
local axes, definitions and analysis steps. `write_brep` extracts the OCC
geometry for `g.model.io.load_brep` (needs `cadquery-ocp`, the
`apeGmsh[stko]` extra). Use it to harvest an STKO model's values and to
check an apeGmsh rebuild against it.

```python
from apeGmsh.interop.stko import read_scd, write_brep

scd = read_scd("1A_TH_000.scd")
slab = scd.physical_property("Slab_Elastic")      # XObject
slab.type, slab["E"], slab["h"]                   # "sections.ElasticMembranePlateSection", 25000.0, 200.0
scd.set_elements("Columns_Set")                   # element IDs (as STKO's .mpco.cdata)
scd.set_nodes("Columns_Set")                      # + nodes on the set's vertices
scd.analysis_elements()                           # what OpenSees receives: {eid: AnalysisElement}
scd.condition("mass_Slabs").geometry              # {geom_id: SubShapes(faces=(...))}
write_brep(scd, "1a.brep", geometry=353)          # one geometry, or the whole compound
```
`# src/apeGmsh/interop/stko/model.py, read_scd.py, geometry.py`
`# verified: tests/interop/test_stko_reader.py::test_analysis_elements_skip_the_edge_meshes`

- **Addressing is STKO's:** a geometry is one child of the document's OCC
  compound; sub-shapes are 0-based per kind (`vertices`, `edges`, `faces`,
  `solids`). After `write_brep` → `load_brep`, STKO face *i* is gmsh face
  tag *i + 1* and edge *i* is edge tag *i + 1*; **vertices are reordered** —
  match them by coordinate through `scd.mesh.vertex_nodes`.
- **`scd.mesh.elements` is every meshed entity**, not the analysis mesh:
  edge meshes of shells and face meshes of solids are in it. Use
  `analysis_elements()` (sub-shapes with an element property, plus
  interaction-generated elements).
- **Parameters** keep STKO's names (`"E"`, `"Drilling DOF Type"`); `INDEX` /
  `INDEX_VEC` values are IDs of other objects (listed in
  `XObject.references`). Nested custom objects (section outlines, fiber
  sections) come back as dicts of arrays.
- **Documents carry unused template properties** — map what is assigned
  (`Geometry.element_property` / `physical_property`), not every definition.

### Translating an STKO document — `translate_scd` + `build_opensees` (ADR 0111)

The translator is **mesh-faithful**. STKO's own nodes and elements go into
the session with STKO's ids, as gmsh discrete entities. The session is never
meshed or renumbered. The deck then reproduces STKO's export: sections,
materials, elements, masses, loads, fixities, rigid diaphragms, time series,
Rayleigh damping and the static stages.

```python
from apeGmsh import apeGmsh
from apeGmsh.interop.stko import read_scd, translate_scd, build_opensees, collect_unsupported

scd = read_scd("1C_TH_000.scd")
collect_unsupported(scd)                        # () when everything is translatable
with apeGmsh(model_name="1C") as g:             # an EMPTY session
    result = translate_scd(g, scd, records={2: accel_values})   # Path series values
    fem = g.mesh.queries.get_fem_data(dim=None) # NEVER generate() / renumber() here
ops = build_opensees(fem, result)               # apeSees: props, elements, conditions, static stages
ops.tcl("1C.tcl")
result.plan.patterns, result.plan.stages        # UniformExcitation + transient stage: data only
result.mesh.element_groups                      # PG per (element prop, physical prop, local axis)
```
`# src/apeGmsh/interop/stko/translate.py (+ translate_mesh/_props/_conditions/_types.py)`
`# verified: tests/interop/test_stko_translate.py::test_build_opensees_emits_the_whole_document`

- **Fails loud, all at once.** An unknown or tier-only STKO type (stdBrick,
  zeroLength, absorbing boundaries, ASDEmbeddedNodeElement, H5DRM, hex8
  meshes, …), or an option that is not translated, raises one
  `UnsupportedSTKOTypes`. It lists every offender (`.items`) before the
  session is touched. Unassigned template properties are not scanned.
- **Element tags are STKO's element ids.** `build_opensees` builds the bridge
  with `apeSees(fem, element_tags="fem")`; `element_tags="sequential"` keeps
  the bridge's own numbering. Tags the bridge makes itself (springs,
  couplings) land above the largest FEM element id. Node tags are STKO's
  either way.
- **Serial by default.** `chain="serial"` maps STKO's `Mumps` to `Pardiso`
  and `ParallelRCM` to `RCM`. `chain="stko"` keeps STKO's choices, for a
  partitioned deck.
- **Time-series values.** The `.scd` holds one-value placeholders
  (`TimeSeriesSpec.placeholder`). Pass the real record with `records=`.
- **IMPL-EX `dTime`: your time-history driver must set it every step.** Each
  static stage starts with STKO's reset of `dTimeCommit` / `dTimeInitial` /
  `dTime` on `result.plan.implex_dt_targets` (`duration/numIncr`). After it
  the ASDConcrete materials stop reading OpenSees' own increment, so before
  each transient step set `dTime` on those elements (and `dTimeCommit` /
  `dTimeInitial` at the first step), as STKO's deck does.
  `build_conditions(..., implex_dtime=False)` writes no reset.
- **Rigid diaphragms are STKO's links, checked.** One
  `g.constraints.rigid_diaphragm` per master, on carriers that hold only that
  master and its slaves, with a tolerance that keeps an off-plane slave.
  `build_conditions` checks the FEM's resolved records against
  `result.plan.diaphragm_pairs` (`translate_conditions.verify_diaphragms`).
- **Recipes:** `declare_translation(ops, result)` returns `(props, series)`
  for a bridge you made yourself. `translate_props.asdconcrete_9p(attrs)` and
  `translate_props.fiber_rows(section)` are STKO's computations, usable as a
  source of definitions. `result.plan.summaries` gives each condition as an
  intent (value, per area/length/node, its PGs, the patterns that use it).
