# apeGmsh workflows — end-to-end patterns
<!-- skill-freshness: verified against apeGmsh main@20f5f091 (2026-07-18) · if weeks old, re-verify signatures in src/apeGmsh/ before trusting exact tags/signatures -->

Concrete recipes for the workflows that come up most often. Each one is a
working skeleton — fill in the geometry and it runs against the **v2.0.0**
API. The `examples/` folder in the project is the authoritative gallery;
these are the patterns worth memorizing.

Deep references (read on demand): the broker/selection chain is in
`fem-broker.md`; the typed OpenSees bridge (`apeSees`) is in
`opensees-bridge.md`; one-line API signatures are in `api-cheatsheet.md`.

The session shape is always the same spine:

```
build geometry → PGs/labels → declare loads/masses/constraints (pre-mesh)
  → mesh → fem = g.mesh.queries.get_fem_data(dim=...)   # resolution happens here
  → apeSees(fem) bridge → emit deck / .h5  → (later) Results post-processing
```

Two top-level facts that the older skill got wrong and you must not repeat:

- `g.masses` (not `g.mass`). `g.opensees` was **removed** — the OpenSees
  entry point is the post-session bridge `apeSees(fem)` from
  `apeGmsh.opensees`.
- "The session **is** the assembly" at the *top level* — `apeGmsh.Assembly`
  does not exist. The declarative `Assembly`+`couple` builder **shipped** as
  a **sub-path** import (`from apeGmsh.assembly import Assembly`, PR #433) —
  see `compose.md`.

---

## 1. Single-session solid — copy-paste quickstart

**The minimal end-to-end example. Runs as-is** (it builds a box, so you can
paste and execute it with zero edits) — start here, then swap in your own
geometry. One session, one kernel, one mesh → OpenSees, all in one `with`.

```python
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

with apeGmsh(model_name="block") as g:
    # Geometry (add_box auto-synchronizes; sync=True is default)
    box = g.model.geometry.add_box(0, 0, 0, 10, 5, 2, label="body")

    # Physical groups — the solver contract (idiomatic: by label / the
    # .select() chain, never raw entity tags). The base face is a thin bbox
    # slab at z = 0; .to_physical() names it as a PG in one call.
    g.physical.add_volume("body", name="Body")
    (g.model.select(None, dim=2)
        .in_box((-1e-6, -1e-6, -1e-6), (10 + 1e-6, 5 + 1e-6, 1e-6))
        .to_physical("Base"))

    # Loads / masses declared pre-mesh; resolved by get_fem_data
    with g.loads.case("dead"):
        g.loads.gravity("Body", g=(0, 0, -9.81), density=2400)
    g.masses.volume("Body", density=2400)

    # Mesh
    g.mesh.sizing.set_global_size(0.4)
    g.mesh.generation.generate(dim=3)
    g.mesh.partitioning.renumber(dim=3, method="rcm", base=1)

    # Snapshot — this is the resolution point
    fem = g.mesh.queries.get_fem_data(dim=3)
    print(fem.info.summary())

# OpenSees — post-session bridge, typed primitives. Loads are opt-in
# (ADR 0051): a g.loads.case reaches the deck only via p.from_model(case)
# inside a pattern (here gravity is an element body_force instead). Masses
# + fixities are re-declared on the bridge; MP constraints auto-emit.
ops = apeSees(fem)
ops.model(ndm=3, ndf=3)
conc = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
ops.element.FourNodeTetrahedron(
    pg="Body", material=conc, body_force=(0.0, 0.0, -9.81 * 2400),
)
ops.fix(pg="Base", dofs=(1, 1, 1))
ops.py("block.py")     # writes the openseespy deck to the cwd — no out/ dir needed
# verified: tests/opensees/integration/test_runnable_deck.py::test_tcl_deck_contains_constraint_lines
```

---

## 2. Multi-part assembly via `Part`

Parts from separate CAD files (or instanced) build in their own sessions,
then import into an assembly session via `g.parts`.

```python
from apeGmsh import apeGmsh, Part
from apeGmsh.opensees import apeSees

with Part("girder") as girder:
    girder.model.geometry.add_box(0, 0, 0, 20, 0.6, 1.5, label="girder")
    # no save() → auto-persists to a tempfile on __exit__

with Part("deck") as deck:
    deck.model.geometry.add_box(-0.5, -2, 1.5, 21, 4, 0.25, label="deck")

with apeGmsh(model_name="bridge") as g:
    g.parts.add(girder, label="girder")
    g.parts.add(deck,   label="deck")

    # Fragment so shared interfaces become conformal. fragment_all
    # syncs internally. Do it BEFORE creating PGs at the fragmented dim.
    g.parts.fragment_all(dim=3)

    for label in g.parts.labels():
        inst = g.parts.get(label)
        for tag in inst.entities.get(3, []):
            g.physical.add(3, [tag], name=label.capitalize())

    with g.loads.case("dead"):
        g.loads.gravity("girder", g=(0, 0, -9.81), density=7850)
        g.loads.gravity("deck",   g=(0, 0, -9.81), density=2400)
    g.masses.volume("girder", density=7850)
    g.masses.volume("deck",   density=2400)

    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
    g.mesh.partitioning.renumber(dim=3, method="rcm", base=1)
    fem = g.mesh.queries.get_fem_data(dim=3)

ops = apeSees(fem)
ops.model(ndm=3, ndf=3)
steel = ops.nDMaterial.ElasticIsotropic(E=200e9, nu=0.3, rho=7850)
conc  = ops.nDMaterial.ElasticIsotropic(E=30e9,  nu=0.2, rho=2400)
ops.element.FourNodeTetrahedron(pg="Girder", material=steel,
                                body_force=(0.0, 0.0, -9.81 * 7850))
ops.element.FourNodeTetrahedron(pg="Deck", material=conc,
                                body_force=(0.0, 0.0, -9.81 * 2400))
ops.py("out/bridge.py")
```

Key points:

- Each `Part` owns its own Gmsh session; `with Part(...)` auto-persists the
  geometry to a tempfile on exit so `g.parts.add(part)` finds it on disk.
- `g.parts.fragment_all(dim=3)` makes shared faces/edges conformal *before*
  you create PGs at that dimension.
- String selectors in `g.loads.*`, `g.masses.*`, `g.constraints.*`, and
  `fem.nodes.select(target=...)` accept part labels directly — no need to
  promote them into PGs first.
- This is the *imperative* multi-part path (one live gmsh session). The
  *declarative* cross-session path (build modules separately, save each to
  `.h5`, then graft) is `g.compose` — see `compose.md`.

---

## 3. Constraints — two-stage pipeline (emission now automatic)

Coupling is declared pre-mesh and *resolved* at `get_fem_data`. As of
**ADR 0022** the resolved MP constraints **auto-emit** into the runnable
deck — you no longer hand-write `ops.equalDOF`/`ops.rigidLink` loops.

### 3.1  Stage 1 — declare before meshing

```python
# Node-to-node: a colocated DOF tie, resolved by coincidence within tolerance
g.constraints.equal_dof("col", "beam", dofs=[1, 2, 3], tolerance=1e-3)

# Surface tie: master surface drives slave entity (interpolation record)
g.constraints.tie("shell", "beam",
                  master_entities=[(2, face_tag)],
                  slave_entities=[(1, edge_tag)],
                  dofs=[1, 2, 3, 4, 5, 6], tolerance=5.0)
```

Declaration verbs on `g.constraints` and where their records land:

| Level | Method | Lives on |
|---|---|---|
| Node-to-node | `equal_dof`, `rigid_link`, `penalty` | `fem.nodes.constraints` |
| Node-to-group | `rigid_diaphragm`, `rigid_body`, `kinematic_coupling` | `fem.nodes.constraints` |
| Mixed-DOF | `node_to_surface`, `node_to_surface_spring` | `fem.nodes.constraints` |
| Surface | `tie`, `distributing_coupling`, `embedded` | `fem.elements.constraints` |
| Surface-to-surface | `tied_contact` | `fem.elements.constraints` |
| Fork contact | `contact`, `contact_plane` (+ deprecated `mortar` alias) | `fem.elements.contacts` (ADR 0073) |

**`tie` vs `equal_dof`**: `equal_dof` ties *colocated* nodes DOF-for-DOF
(a node-pair record, resolved by coincidence within `tolerance`). `tie` is
an *interpolation* between non-matching meshes — a master surface drives a
slave entity even when their nodes don't line up (a surface/interpolation
record on `fem.elements.constraints`). Reach for `tie` across non-conformal
interfaces; `equal_dof` only when nodes coincide.

### 3.2  Stage 2 — resolution at `get_fem_data`

`get_fem_data` resolves every declared constraint into records on two
composites:

- `fem.nodes.constraints` (`NodeConstraintSet`) — node-pair, node-group,
  node_to_surface
- `fem.elements.constraints` (`SurfaceConstraintSet`) — surface ties,
  distributing/embedded interpolations, tied_contact couplings (fork
  `contact(...)` records land on `fem.elements.contacts` instead)

```python
fem = g.mesh.queries.get_fem_data(dim=None)   # dim=None = all dims; needed
# verified: tests/test_femdata_from_h5.py::test_round_trip_node_to_surface_record
# when shells or tied interfaces (dim<3 entities) must reach the bridge.
```

### 3.3  Stage 3 — emission is AUTOMATIC in the bridge (ADR 0022)

`apeSees(fem)` now **fans the resolved MP constraints out into the
runnable Tcl/Py deck** — `equalDOF`, `rigidLink`, `rigidDiaphragm`, and the
`ASDEmbeddedNodeElement` embedded/tied_contact path all emit. When MP
constraints are present the bridge also auto-selects the `Transformation`
constraint handler. (The old skill's "constraint emission is DEFERRED /
the Emitter protocol has no MPC verb" claim is **false** — do not repeat it.)

```python
ops = apeSees(fem)
ops.model(ndm=3, ndf=3)
# ... materials, elements, fixities ...
ops.tcl("out/coupled.tcl")   # the deck contains equalDOF/rigidLink/... lines
# verified: tests/opensees/integration/test_runnable_deck.py::test_tcl_deck_contains_constraint_lines
# verified: tests/opensees/unit/test_emitter_protocol.py::test_equalDOF_records_master_slave_dofs
```

You no longer iterate `fem.nodes.constraints` / `fem.elements.constraints`
by hand to drive a solver. Those accessors remain for **inspection** and
for the viewer/`Results`, but the deck is complete on its own. The emitted
deck `analyze`s end-to-end (`test_equalDOF_deck_analyzes`,
`test_rigid_link_deck_analyzes`, `test_tied_contact_deck_analyzes`).

### Worked example: solid soil ↔ frame column

```python
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

with apeGmsh(model_name="hybrid") as g:
    g.model.geometry.add_box(-10, -10, -20, 20, 20, 20, label="soil")
    p_base = g.model.geometry.add_point(0, 0, 0, label="col_base")
    p_top  = g.model.geometry.add_point(0, 0, 6, label="col_top")
    g.model.geometry.add_line(p_base, p_top, label="col")

    g.physical.add_volume("soil", name="Soil")
    g.physical.add_curve("col",  name="Column")
    g.physical.add_point("col_base", name="ColBase")
    g.physical.add_point("col_top",  name="ColTop")
    # Soil top face via a thin bbox slab at z = 0 → named PG in one call.
    (g.model.select(None, dim=2)
        .in_box((-10 - 1e-6, -10 - 1e-6, -1e-6), (10 + 1e-6, 10 + 1e-6, 1e-6))
        .to_physical("SoilTop"))

    # Couple the column base into the soil top surface. This resolves into
    # fem.elements.constraints AND auto-emits into the deck (§3.3).
    g.constraints.tie("SoilTop", "ColBase")

    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(dim=3)
    g.mesh.partitioning.renumber(dim=3, method="rcm", base=1)
    fem = g.mesh.queries.get_fem_data(dim=None)   # all dims (frame is 1-D)

ops = apeSees(fem)
ops.model(ndm=3, ndf=6)                          # ndf=6: column needs rotations
soil = ops.nDMaterial.ElasticIsotropic(E=50e6, nu=0.3, rho=2000)
ct   = ops.geomTransf.Linear(vecxz=(1, 0, 0))
ops.element.stdBrick(pg="Soil", material=soil)
ops.element.elasticBeamColumn(pg="Column", transf=ct,
    A=0.09, E=30e9, G=12.5e9, J=1e-3, Iy=6.75e-4, Iz=6.75e-4)
ops.fix(pg="SoilTop", dofs=(0, 0, 0))            # example fixity
ops.tcl("out/hybrid.tcl")
```

---

## 4. Pushover-style second pattern

Patterns are **explicit** in the bridge — each is its own
`with ops.pattern.Plain(series=...) as p:` block opened on `ops`. Gravity
is best expressed as an element `body_force=`; the lateral pushover is a
nodal `p.load`.

```python
from apeGmsh.opensees import apeSees
# ... (g built + meshed; fem = g.mesh.queries.get_fem_data(dim=3)) ...

ops = apeSees(fem)
ops.model(ndm=3, ndf=3)
conc = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)

# Gravity — body force on the elements (pattern 1 equivalent)
ops.element.FourNodeTetrahedron(
    pg="Body", material=conc, body_force=(0.0, 0.0, -9.81 * 2400))
ops.fix(pg="Base", dofs=(1, 1, 1))

# Pushover — explicit pattern: unit lateral load at the control PG
with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
    p.load(pg="ControlNode", forces=(1.0, 0.0, 0.0))

ops.py("out/pushover.py")
```

Patterns appear in the deck in the order you open them (`pattern Plain <i>
Linear { ... }`, indexed from 1). On the openseespy side, drive gravity
with `LoadControl` and the lateral step with `DisplacementControl`.

For staged construction (excavation, lift, soil-structure interaction) use
the stage builder `with ops.stage(name=...) as s:` (`s.fix`/`s.mass`/
`s.region`/`s.recorder`/`s.embedded`/`s.initial_stress`/`s.analysis`/
`s.run`) — see `opensees-bridge.md`.

### The `name=` constraint handshake (staged SSI)

A constraint that must **activate in a later stage** is declared once at
apeGmsh time **with `name=`**, resolved into the snapshot, then **claimed
by that name** in the stage where it turns on. This is the one contract
that spans the session → bridge boundary; the stage mechanics
(PUSH/PULL/CLAIM, the required `s.analysis(...)` + `s.run(...)`) live in
`opensees-bridge.md`.

```python
# ── pre-mesh (session): DECLARE with name= ──────────────────────────────
with apeGmsh(model_name="ssi") as g:
    # ... soil volume + lining shell geometry, PGs "Soil" / "Lining" / "Iface" ...
    g.constraints.embedded("Soil", "Lining", name="lining_embed")   # named, not yet active
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)        # resolves the named constraint into the snapshot

# ── bridge: CLAIM the name in the install stage ─────────────────────────
ops = apeSees(fem)
ops.model(ndm=3, ndf=3)
ops.element.stdBrick(pg="Soil", material=soil)
ops.fix(pg="Base", dofs=(1, 1, 1))

with ops.stage(name="insitu") as s:                 # 1) gravity on soil only
    s.initial_stress(name="k0", pg="Soil", sigma_xx=-100, sigma_yy=-100, sigma_zz=-200, ramp_steps=10)
    s.analysis(...); s.run(n_increments=10, dt=0.1)

with ops.stage(name="install_lining") as s:         # 2) lining online + embed activates
    s.activate(pgs=["Lining"])
    s.embedded(name="lining_embed")                 # CLAIM by name → domainChange brings it live
    s.analysis(...); s.run(n_increments=20, dt=0.05)

ops.tcl("out/ssi.tcl", run=True)                    # staged decks emit via tcl/py text only
```

A typo or missing `name=` on either side raises `ValueError`; a name claimed
in two stages raises. Other claimable constraints: `s.tie` / `s.distributing`
/ `s.equal_dof` / `s.rigid_link` / `s.rigid_diaphragm` / `s.kinematic_coupling`
/ `s.node_to_surface` / `s.node_to_surface_spring` / `s.tied_contact`
(all `(name=)`).

---

## 5. Persistence round-trip — build → save → reload

Build once, autosave the **neutral-zone** `model.h5` on context exit, then
reload it in a later script without touching gmsh. Two distinct reload
entry points (don't confuse them):

- `FEMData.from_h5(path)` → rebuilds a **FEMData** (nodes/elements/PGs/
  labels/constraints/loads/masses). Integrity-checked: `/meta/snapshot_id`
  is re-verified against the recomputed hash (`MalformedH5Error` on
  mismatch; wrong schema major → `SchemaVersionError`).
- `apeGmsh.from_h5(path)` → rebuilds a **chain-phase session** (no gmsh
  state). Only `compose()`/`compose_inspect()`/`compose_list()`/`save()`
  work; `g.model.*` / `g.mesh.*` will fail.

```python
from apeGmsh import apeGmsh, FEMData

# Build + autosave the neutral zone on g.end()/context-exit.
with apeGmsh(model_name="plate", save_to="plate.h5", overwrite=True) as g:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 0.1, label="body")
    g.physical.add(3, ["body"], name="Body")
    g.mesh.generation.generate(dim=3)
    # g.save("ckpt.h5") also works for an explicit checkpoint.

# Resume later — pure broker reload, no gmsh:
fem = FEMData.from_h5("plate.h5")
print(fem.info.summary())
# verified: tests/test_femdata_from_h5.py::test_session_save_then_from_h5

# Or rehydrate a chain-phase session (compose/save still work):
g2 = apeGmsh.from_h5("plate.h5")
```

What writes what: `g.save()` / `FEMData.to_h5(path)` write **only the
neutral zone**; `apeSees(fem).h5(path)` writes **both** zones (neutral +
`/opensees/`) — the canonical two-zone file the viewer and `Results`
consume. Write traps (`save_to=` autosaves on `end()` not eagerly;
`overwrite=` / `RuntimeError` cases) and the per-zone schema constants +
reader window are all in **`fem-broker.md` Part B** — not repeated here.

---

## 6. Post-processing — run analysis → Results → query → plot/web

`Results` constructors **require a model** (the three-broker chain, ADR
0020): `Results.model` is always non-None, FEMData is reached via
`results.model.fem`, and omitting the model kwarg raises `TypeError`. The
full constructor table (`from_native` / `from_mpco` / `from_ladruno` /
`from_recorders`, each kwarg) is in **`results.md` §1** — the end-to-end
spine here uses the two common ones:

```python
from apeGmsh import Results
from apeGmsh.opensees import OpenSeesModel

# Native apeGmsh two-zone HDF5 (model+results often in one file)
model = OpenSeesModel.from_h5("run.h5")          # model= is REQUIRED
results = Results.from_native("run.h5", model=model)
# verified: tests/test_results_bind.py::test_from_native_without_model_raises_typeerror

# STKO .mpco: model_h5= (sibling model archive) is REQUIRED
results = Results.from_mpco("run.mpco", model_h5="model.h5")
# verified: tests/test_results_bind.py::test_from_mpco_without_model_h5_raises_typeerror

# Three-broker chain forward + lineage
osm = results.model              # OpenSeesModel broker (never None)
fem = results.model.fem          # neutral FEMData zone
lin = results.lineage            # Lineage(...) — NEVER raises
# verified: tests/test_results_bind.py::test_lineage_propagates_from_model
# Mismatches surface as strings in lin.warnings; escalate with:
# lin.assert_clean()  -> LineageError.  (BindError was deleted; pairing the
# right fem with a run is the USER's responsibility.)

disp = results.nodes.get(component="displacement_z", pg="Top")

# Static matplotlib (headless; needs the [plot] extra)
results.plot.contour("displacement_z", step=-1)
```

**Viewers — kernel safety.** `results.viewer(blocking=True)` is the
DEFAULT and **crashes the Jupyter kernel** (blocking VTK+Qt). In notebooks
use the web viewer (kernel-safe trame) or the subprocess viewer:

```python
results.show_web()                  # inline trame/pyvista; step slider + layer toggles
results.viewer(blocking=False)      # subprocess; kernel keeps running

# Standalone web app (outside a notebook; blocks until Ctrl-C):
# results.serve_web(render_mode="client", port=8080)

# Zero-setup sample — no .mpco/model.h5 needed:
Results.demo().show_web()
# verified: tests/test_results_demo.py::test_results_demo_classmethod
```

`show_web`/`serve_web` take `render_mode` in `{"client"` (default, browser
WebGL, fast)`, "server"` (kernel-side, image-streamed)`, "hybrid"}`, and
need the `[viewer]` extra (trame; ipywidgets for the inline controls). Full
surface (`results.stages`, `eigen_modes`, slabs, controls) in `results.md`.

---

## 7. Compose — build modules → graft → inspect/save

Build reusable modules in isolation, save each to `.h5`, then graft them
into a host via `g.compose` (tag-offset namespacing, no gmsh re-run). The
cross-session path runs in a chain-phase session (`apeGmsh.from_h5`).

```python
from apeGmsh import apeGmsh

# 1. Build + save a reusable module
with apeGmsh(model_name="bolt", save_to="bolt.h5", overwrite=True) as g:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 5, label="shaft")
    g.physical.add(3, ["shaft"], name="shaft")
    g.mesh.generation.generate(dim=3)

# 2. Cross-session composition — chain phase, no gmsh build
g = apeGmsh.from_h5("host.h5")
g.compose("bolt.h5", label="bolt", translate=(10.0, 0.0, 0.0))
g.compose("bolt.h5", label="bolt2", anchor="mount_pad")  # anchor XOR translate
print(g.compose_list())          # (ComposedModule(label='bolt'), ...)
g.compose_inspect("bolt.h5")     # metadata-only dict, no merge
g.save("assembly.h5")
# verified: tests/test_compose_end_to_end.py::test_from_h5_session_compose_workflow
# verified: tests/test_compose_end_to_end.py::test_cross_session_compose_via_from_h5
```

Composed-module PGs are namespaced `{label}.{pg}` (the host stays bare).
Interface-bridging constraints (`tie`/`equal_dof`/`tied_contact`/...) DO
work in chain phase and route onto the FEMData. `label=` is fail-loud (no
`.`/`/`/whitespace; can't start/end with `_`); `anchor=` and a non-zero
`translate=` are mutually exclusive. Full rules, nested compose, depth
limits, and the viewer `'Module'` color mode in `compose.md`.

---

## Patterns worth knowing (not full workflows)

### Label → physical group promotion

Labels (Tier 1) don't commit to a dimension. To make a label visible to a
consumer that reads the raw `.msh`, promote it:

```python
g.labels.promote_to_physical("col.web")
```

The bridge accepts label names directly (via the `_label:` prefix), so
promotion is only needed for external `.msh` consumers.

### Selection sets for post-mesh queries

Build a named set **after meshing** with the fluent
`g.mesh_selection.select(...).save_as(name)` chain (point family —
`on_plane` takes `(point, normal, *, tol)`; spatial verbs test
coordinates). It snapshots into `FEMData` and round-trips via HDF5:

```python
g.mesh.generation.generate(dim=3)
(g.mesh_selection.select(level="node", dim=2)
    .on_plane((0, 0, 10), (0, 0, 1), tol=1e-6)
    .save_as("top_nodes"))

fem = g.mesh.queries.get_fem_data(dim=3)
ids = fem.mesh_selection.node_ids("top_nodes")          # ndarray of node ids
```

Alternative (PG → set bridge): name a PG with the geometry selector, then
import it — `g.model.select(None, dim=2).on_plane((0,0,10),(0,0,1),tol=1e-6).to_physical("top")`
followed by `g.mesh_selection.from_physical(dim=2, "top", ms_name="top_nodes")`.

### Diagnosing disjoint topology (arc-line wires, IGES imports)

A wire built from a partial arc (`add_ellipse(angle1, angle2)`) plus lines,
or an un-welded IGES import, often leaves OCC unable to weld the arc/line
endpoints — the mesh carries two distinct nodes at every corner with no
continuity, and corner moments read wrong. Fix at the geometry layer:

```python
import math
g.model.geometry.add_ellipse(0, 0, 0, 2.55, 2.75,
                             angle1=0, angle2=math.pi, label="arch")
g.model.geometry.add_line("p_left",   "arch_start")
g.model.geometry.add_line("arch_end", "p_right")

g.model.queries.make_conformal(dims=[1])   # weld BEFORE Parts / meshing

g.mesh.generation.generate(1)
fem = g.mesh.queries.get_fem_data(dim=1)
pairs = fem.inspect.find_coincident_node_pairs(pg="cimbra", tol=1e-6)
unbridged = {k: v for k, v in pairs.items() if not v}
assert not unbridged, f"unbridged corners: {sorted(unbridged)}"
```

Reading the diagnostic:

* `pairs == {}` — no coincident pairs; topology clean.
* `pairs[(a, b)] == []` — **bug**: two nodes share XYZ, nothing ties them.
  Re-fragment (`make_conformal`) or add `equal_dof` / `rigid_link`.
* `pairs[(a, b)] == ["element zeroLength#7"]` — legitimate.
* `pairs[(a, b)] == ["constraint equal_dof"]` — legitimate, explicitly tied.

---

## Workflow-level pitfalls

- **Calling synchronize by hand.** `g.model.geometry.add_*` and
  `g.model.boolean.*` sync internally (`sync=True` default). You almost
  never need `gmsh.model.occ.synchronize()`.
- **Labels on the assembly session expected solver-visible without
  `promote_to_physical`.** Part sessions auto-promote labels; assembly
  sessions do not.
- **Fragmenting after PGs are attached by tag.** `fragment_all` rewrites
  entity tags; PGs added by tag (not by label) go stale. Create PGs after
  fragmentation, or use labels (they survive it).
- **`make_conformal()` after building `Part` instances.** A `Part` built
  before fragmenting holds stale tag dicts and misresolves silently. Weld
  *before* constructing Parts, or rebuild Parts after.
- **Asking for `dim=3` FEMData on a tet+shell model.** `get_fem_data(dim=3)`
  drops the 2-D mesh — use `dim=None` when shells or tied interfaces must
  reach the bridge.
- **`ops.model(ndm=3, ndf=3)` then declaring a beam.** Beams need
  rotational DOFs — use `ndf=6` for 3-D frame/shell models.
- **Omitting `model=`/`model_h5=` on a `Results` constructor.** Raises
  `TypeError` since ADR 0020 — every constructor needs a model.
- **`results.viewer()` in a notebook.** Default `blocking=True` crashes the
  kernel — use `results.show_web()` or `results.viewer(blocking=False)`.
