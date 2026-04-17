# apeGmsh Learning Curriculum

**Status: DRAFT — awaiting user review before execution.**

This document is the planning spec for a structured learning path
through apeGmsh, OpenSees-first. Every example ends with
``ops.analyze()`` and at least one extracted result, verified against
either an analytical solution or a published benchmark.

The curriculum preserves every existing notebook. Nothing is deleted
or renamed. Slots that already have an existing notebook either
**ADAPT** it (copy → verify against current library → fix whatever
has drifted → save as the curriculum version) or mark it **NEW** when
no existing notebook is close enough.

> ### Health warning about existing notebooks
>
> Several notebooks under `examples/` were written against earlier
> versions of the composite APIs (`NodeResult` before pair-iter,
> `MeshViewerV2`, constraint records before the `_kinds` split, …).
> **They are not assumed to run as-is.** Every curriculum notebook
> — even those derived from an existing file — is executed end-to-end
> against `HEAD` before being accepted into the curriculum. Where an
> existing notebook covers the right topic but uses stale APIs, it is
> still valuable as a starting point: the engineering setup (geometry,
> section properties, loads) rarely needs to change, only the
> apeGmsh calls that consume/produce composite records.

---

## How to read this document

Each row in a tier table has:

| Field | Meaning |
|---|---|
| **#** | Curriculum ordinal (01..19). |
| **Title** | Short descriptive name (what the learner sees in the README). |
| **File** | Notebook path. Prefix: `NEW` = to be written from scratch. `ADAPT` = existing notebook is a starting point; verify + fix stale APIs + save under `examples/curriculum/`. `REVIEW` = existing file likely fits but I need to read it before confirming the classification. No "reuse as-is" column — every existing notebook is treated as a draft. |
| **Learns** | Single-sentence learning objective. |
| **Features** | apeGmsh features introduced (comma-separated). |
| **Analysis** | OpenSees analysis type (static / eigen / static_nonlinear / transient). |
| **Verify** | How correctness is checked (analytical, benchmark, cross-code). |
| **Prereq** | Earliest preceding example the learner must have done. |

See the _Summary of authoring work_ section at the bottom for the
counts of NEW vs ADAPT slots and the total effort estimate.

---

## Tier 1 — Fundamentals

Goal: a new user can build, mesh, solve, and extract results from a
linear-elastic model in under 100 lines. Every example here has a
closed-form answer.

| # | Title | File | Learns | Features | Analysis | Verify | Prereq |
|---|---|---|---|---|---|---|---|
| 01 | Hello Plate | ADAPT `examples/EOS Examples/example_plate_basic.ipynb` | The full apeGmsh pipeline: `g.model` → `g.mesh` → `FEMData` → OpenSees → displacement. | geometry primitives, physical groups, `fix`, nodal load, `g.opensees.ingest`, `ops.analyze("Static")` | static linear | Max displacement within 1% of Timoshenko plate theory or analytical beam strip. | — |
| 02 | 2D Cantilever Beam | **NEW** `examples/curriculum/02_cantilever_beam_2D.ipynb` | Simplest 1D model with analytical check. | beam geometry, `elasticBeamColumn`, point load at free end, tip displacement | static linear | `δ = PL³/(3EI)` within 0.1%. | 01 |
| 03 | Simply-Supported Beam | **NEW** `examples/curriculum/03_simply_supported_beam.ipynb` | Distributed load + two supports; midspan displacement + moment. | line load, two pin supports, reaction extraction | static linear | `δ_mid = 5wL⁴/(384EI)` and `M_mid = wL²/8`. | 02 |
| 04 | 2D Portal Frame | **NEW** `examples/curriculum/04_portal_frame_2D.ipynb` | First multi-element model; joint moment distribution. | multiple beams, rigid connections, combined loads, reaction + drift | static linear | Hand-calculation via stiffness method; drift within 1%. | 03 |

**Tier 1 gaps to write:** 3 notebooks (02, 03, 04).

---

## Tier 2 — Model building blocks

Goal: the learner sees each major composite (`labels`, `physical`,
`sections`, `loads`, `masses`, `mesh.sizing`) in isolation so they
know where each concept lives.

| # | Title | File | Learns | Features | Analysis | Verify | Prereq |
|---|---|---|---|---|---|---|---|
| 05 | Labels and Physical Groups | **NEW** `examples/curriculum/05_labels_and_pgs.ipynb` | When to use labels (Tier 1) vs PGs (Tier 2); how both feed load/BC targeting. | `g.model.labels.*`, `g.physical.*`, auto-resolution precedence in `g.loads.point(target=)` | static linear | Identical result whether the target is given as a label, a PG, or a raw DimTag list. | 02 |
| 06 | Section Catalog | ADAPT `examples/moment_curvature_fiber_section.ipynb` + short section-catalog prelude | Fiber sections (concrete + rebar), elastic sections, shell sections; moment-curvature response. | `g.opensees.sections.fiber`, `g.opensees.sections.elastic`, moment-curvature driver | static nonlinear (section only) | Cracked/uncracked stiffness ratio matches hand calc; yield moment within 2%. | 02 |
| 07 | Load Patterns and Combinations | **NEW** `examples/curriculum/07_load_patterns.ipynb` | Multiple named patterns (`dead`, `live`, `seismic`), `g.loads.pattern()` context manager, combination at analysis time. | `g.loads.pattern()`, multiple `timeSeries`, `LoadPattern` separation in OpenSees | static linear (superposed) | Pattern-wise displacements sum to the combined result within 1e-10. | 01 |
| 08 | Boundary Conditions Walkthrough | **NEW** `examples/curriculum/08_boundary_conditions.ipynb` | `fix`, `face_sp`, prescribed displacement; homogeneous vs non-homogeneous SPs; when each is correct. | `g.loads.face_sp`, `g.loads.face_load`, `SPRecord`, `fem.nodes.sp` | static linear with prescribed disp | Reaction equals applied prescribed force via FEM work balance. | 03 |
| 09 | Mesh Sizing and Refinement | ADAPT `examples/example_mesh_selection.ipynb` + **NEW** prepend mesh-refinement section | Sizing fields, boundary layers, transfinite, recombine; convergence study. | `g.mesh.sizing.*`, `g.mesh.field.*`, `g.mesh.structured.*` | static linear | Displacement converges under mesh refinement toward a known reference. | 01 |

**Tier 2 gaps to write:** 3 fully new (05, 07, 08) + 2 partially new (06 prelude, 09 prepend).

---

## Tier 3 — Assemblies

Goal: the learner can compose multi-instance models from parts, do
boolean operations safely, and tie conformal interfaces.

| # | Title | File | Learns | Features | Analysis | Verify | Prereq |
|---|---|---|---|---|---|---|---|
| 10 | Parts Basics | **NEW** `examples/curriculum/10_parts_basics.ipynb` | Register parts with `g.parts.register()`, place multiple instances, identify entities by `(part, dim, tag)`. | `g.parts.register`, instances, `g.parts.from_model`, part labels as load/BC targets | static linear | Reactions equal to sum over instance contributions. | 04 |
| 11 | Boolean Operations in Assemblies | REVIEW `examples/example_gusset.ipynb` — does it use `g.model.boolean` with parts? If not, **NEW**. | `fuse`, `cut`, `fragment` applied to imported CAD, staying part-aware. | `g.model.io.load_step`, `g.model.boolean.*`, `g.parts.fragment_all` | static linear | Assembled geometry passes `make_conformal` without orphans. | 10 |
| 12 | Interfaces via Tie | **NEW** `examples/curriculum/12_interfaces_via_tie.ipynb` | Two parts meshed separately, tied at a shared surface; the `tie` constraint resolver in action. | `g.constraints.tie`, `InterpolationRecord`, shape-function weights | static linear | Displacement continuity across the interface within mesh-size tolerance. | 10 |

**Tier 3 gaps to write:** 2 fully new (10, 12) + 1 review (11).

---

## Tier 4 — Mixed-dimension and contact

Goal: the learner can couple beam-to-solid, add contact springs,
embed rebar, and tie non-matching meshes — the four hardest
real-world interface problems.

| # | Title | File | Learns | Features | Analysis | Verify | Prereq |
|---|---|---|---|---|---|---|---|
| 13 | Beam-to-Solid Coupling | ADAPT `examples/EOS Examples/example_column_nodeToSurface_v6.ipynb` | 6-DOF master node coupled to 3-DOF surface via phantom nodes + rigid beams. | `g.constraints.node_to_surface`, `NodeToSurfaceRecord`, phantom node tag space | static linear | Reduces to an equivalent beam-theory cantilever stiffness. | 04, 12 |
| 14 | Contact Springs Under a Footing | ADAPT `examples/EOS Examples/example_footing_contact_springs_v2_flexure.ipynb` | Soil-structure contact via springs; separation allowed under tension. | `g.constraints.node_to_surface_spring`, `RIGID_BEAM_STIFF` routing, `stiff_beam_groups()` | static nonlinear | Contact pressure redistributes under eccentric load; matches rigid-footing limit analysis. | 13 |
| 15 | Embedded Rebar in Concrete | ADAPT `examples/01_embedded_rebars.ipynb` | 1D reinforcement embedded in a 3D concrete host via `ASDEmbeddedNodeElement`. | embedded constraint, host vs embedded entities, kinematic coupling | static nonlinear | Cracked flexural capacity matches hand-calculated Mn. | 06, 13 |
| 16 | Tied Contact with Non-Matching Meshes | **NEW** `examples/curriculum/16_tied_contact_nonmatching.ipynb` | Two solids meshed with different element sizes, tied via `tied_contact`. | `g.constraints.tied_contact`, bidirectional projection, `SurfaceCouplingRecord` | static linear | Homogenous stress through the interface within ε · ℎ_max. | 12 |

**Tier 4 gaps to write:** 1 fully new (16).

---

## Tier 5 — Analysis types

Goal: the learner can run every analysis type OpenSees exposes:
modal, buckling, pushover, and time-history.

| # | Title | File | Learns | Features | Analysis | Verify | Prereq |
|---|---|---|---|---|---|---|---|
| 17 | Modal Analysis | ADAPT `examples/example_ibeam_modal.ipynb` | Eigenvalue analysis, mass assembly, mode shape extraction and animation. | `g.masses.lumped`, `ops.analysis("Eigen")`, results viewer mode playback | eigen | First 3 natural frequencies within 2% of closed-form for a cantilever. | 06 |
| 18 | Lateral-Torsional Buckling | ADAPT `examples/EOS Examples/example_LTB_shell.ipynb` | Linearized buckling (`EigenAnalysis` with load stiffness), critical-load extraction, mode visualization. | shell elements, geometric stiffness, `ASDShellQ4` corotational | eigen (buckling) | Critical moment within 5% of the standard LTB formula. | 17 |
| 19 | Nonlinear Static Pushover | ADAPT `examples/frame_pushover_from_cad.ipynb` | Displacement-controlled pushover of a moment frame; plastic hinge formation. | displacement control, fiber sections, capacity-curve extraction | static nonlinear | Matches published pushover for the same benchmark frame. | 06, 04 |
| 20 | Time-History Dynamic Analysis | **NEW** `examples/curriculum/20_time_history_beam.ipynb` | Transient analysis with ground acceleration; Newmark integration; damping. | `MultiSupport` ground motion pattern, Rayleigh damping, `transient` analyze, response history extraction | transient | Peak displacement within 3% of Newmark-integrated closed-form for a SDOF with equivalent mass/stiffness/damping. | 17 |

**Tier 5 gaps to write:** 1 fully new (20).

---

## Summary of authoring work

| Category | Count | Which |
|---|---|---|
| Full new notebooks (write from scratch) | 10 | 02, 03, 04, 05, 07, 08, 10, 12, 16, 20 |
| Adapt an existing notebook (copy → verify against HEAD → fix stale API → save under `curriculum/`) | 8 | 01, 06, 09, 13, 14, 15, 17, 18, 19 |
| Needs review before deciding NEW vs ADAPT | 1 | 11 |
| **Total curriculum slots** | **19** | |

Adapting is **not** free work — early-composite-API drift is real and
every existing notebook will need its constraint/load/viewer calls
updated. Budget ~1-2 hours for a mature ADAPT slot (e.g. 13, 17) and
2-3 hours for one where the engineering side also needs a refresh
(e.g. 09, 19). Budget 2-4 hours per NEW slot.

Rough total effort: **~50-65 hours** of authoring. At one session per
week, that's ~10-12 weeks.

---

## Executing the curriculum

Proposed ordering for writing sessions:

1. **Session 1 (Tier 1):** Write 02, 03, 04. Three small analytical examples; establishes the template every later notebook follows.
2. **Session 2 (Tier 2, part 1):** Write 05, 07, 08. The pure-apeGmsh composite tutorials.
3. **Session 3 (Tier 2, part 2):** Polish 06 and 09. Minimal touches on existing notebooks.
4. **Session 4 (Tier 3):** Write 10, 12. Review 11.
5. **Session 5 (Tier 4 closure):** Write 16.
6. **Session 6 (Tier 5 closure):** Write 20.
7. **Session 7 (integration):** Write `examples/README.md` that ties the whole curriculum together with links and a prerequisite diagram.

Each session keeps CLAUDE.md's surgical discipline: one notebook per commit, follow the template established in 02, verify every analytical claim before shipping.

---

## Template for a curriculum notebook

Every NEW notebook follows this 8-section structure so learners know
what to expect:

1. **Problem statement** — a paragraph and a sketch. State the
   analytical answer up front.
2. **Geometry** — minimal Gmsh geometry using `g.model.geometry.*`.
3. **Physical groups / labels** — tag what loads and BCs will
   attach to.
4. **Mesh** — one or two parameter sweeps, not a convergence study
   (that lives in 09).
5. **FEM build** — sections, loads, constraints, masses.
6. **OpenSees ingest + analysis** — `g.opensees.ingest.*` calls,
   explicit `ops.analyze(...)`.
7. **Result extraction** — pull the specific number the problem
   statement named, compute error vs the analytical/benchmark
   reference.
8. **Viewer check (optional)** — `g.mesh.results_viewer(...)` with a
   one-line "here's what you should see".
