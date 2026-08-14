# Section properties — `SectionProperties` analyzer + bridge handoff (ADR 0078)
<!-- skill-freshness: verified against apeGmsh main@ad5ea52c (2026-07-18) · signatures: python -m apeGmsh.studio.lookup SYMBOL (ADR 0096); src/ is not the authoring lookup -->

In-process cross-section property analyzer (the full PyPI
`sectionproperties` capability set, natively): geometric, Saint-Venant
warping (`J` / shear centre / `Γ` / shear areas), plastic, and stress
recovery on **any meshed 2-D face** — plus a declarative OpenSees
handoff (`ComputedSection`) and a Qt inspector. Shipped complete
(ADR 0078 Accepted, slices S1–S6 = PRs #802–#810 + #811); gates G-A/G-B
adversarially verified the warping math and the OpenSees axis mapping
against upstream C++ source and closed-form cantilevers.

Two different things share the "sections" name — don't confuse them:

- **`g.sections`** (session composite) = the *geometry builder*
  (solid/shell/flat-face recipes).
- **`SectionProperties`** (standalone broker, sibling of `Results` /
  `apeSees`, NOT a composite) = the *analyzer*. Import:
  `from apeGmsh import SectionProperties`;
  `from apeGmsh.sections import SectionMaterial`.

## 1. Workflow

```python
# author a face (any route works: *_face builders, raw OCC, load_dxf, STEP)
g.sections.W_face(bf=400.0, tf=25.0, h=1200.0, tw=12.0, label="girder")
g.mesh.sizing.set_global_size(15.0)
g.mesh.generation.generate(dim=2)
g.mesh.generation.set_order(2)            # tri6 — warping-grade (see §6)
fem = g.mesh.queries.get_fem_data(dim=2)

sec = SectionProperties(
    fem,
    materials={"girder": SectionMaterial(E=200e3, nu=0.3, fy=345.0,
                                         density=7.85e-9)},  # PG name -> material
    name="PG1200x400",                    # handle used in fail-loud messages
    disconnected="raise",                 # or "sum" — see §5
)
geo  = sec.geometric()    # pure quadrature — A, centroid, EI*, phi, Z moduli
warp = sec.warping()      # FE solve (memoized) — GJ, shear centre, EGamma, GAs_*
plas = sec.plastic()      # needs fy on EVERY material
st   = sec.stress(N=-800e3, Vy=350e3, Mxx=1.9e9); st.plot("von_mises")
sec.summary(); sec.plot_section(); sec._repr_html_  # notebook-first
```

The analyzer **snapshots the fem at construction** (session-independent
after; ADR 0001 doctrine) and **memoizes every analysis** — call
`geometric()`/`warping()` freely, they solve once. Constructor gates
(fail-loud `SectionMeshError`): 2-D elements only
(tri3/tri6/quad4/quad8/quad9), face authored in the **global XY plane**,
`materials=` PGs must exactly cover the elements (uncovered AND
doubly-covered both raise).

## 2. The naming law (read this before touching any property)

Rigidity-form fields carry the modulus prefix and are **always valid**:
`EA`, `EQx/EQy`, `EIxx_c/EIyy_c/EIxy_c`, `EI11_c/EI22_c`, `EZ*_plus/minus`,
`GJ`, `EGamma`, `GA`, `GAs_x/GAs_y/GAs_xy`, `Mp_xx/Mp_yy/Mp_11/Mp_22`.

Unprefixed accessors (`Ixx_c`, `Zxx_plus`, `J`, `Gamma`, `As_y`, `Sxx`,
…) divide by the section's **single** modulus:
- geometric-only mode (no `materials=`): placeholder is unit-`E`
  isotropic (`E=1, ν=0` → `G=1/2`) so they are the classic numbers;
- homogeneous: divide by the one material's `E`/`G`;
- **composite: they raise `CompositeSectionError`** — use
  `transformed(e_ref=...)` (geometric) / `transformed(e_ref=, g_ref=)`
  (warping); never a silently-chosen reference.

**Exemption — reference-free ratios**: `rx/ry/r11/r22` (`√(EI/EA)`) and
`alpha_x/alpha_y` (`GAs/GA`) are valid in every mode (modulus cancels;
on composites they are the transformed-section values).

`SectionMaterial(E, nu, G=, fy=, density=, name=)` — `G=` is an
independent **override** (default isotropic `E/2(1+ν)`); the solver
assembles the E-field (geometric) and G-field (warping) separately, so
the override is exact, not a fudge — this is the mechanism for
*authored* partial shear transfer (§5).

## 3. Composite authoring law — PGs must PARTITION the face

Material regions must be **disjoint**: an element claimed by two
`materials=` PGs raises at construction. The trap: authoring an inner
shape **overlapping** an outer one and then fragmenting —
`fragment_pair` maps the overlap piece to BOTH parents' PGs
(double-cover → raise). Carve first, then fragment for conformity:

```python
conc  = g.sections.rect_face(b=600.0, h=600.0, label="concrete")
steel = g.sections.W_face(bf=250.0, tf=17.0, h=250.0, tw=10.0, label="steel")
g.model.boolean.cut(conc.entities[2], steel.entities[2],
                    dim=2, remove_tool=False)      # W-shaped hole in the concrete
g.parts.fragment_pair("concrete", "steel", dim=2)  # conformal shared boundary
```

Hand-authored multi-region faces must share **lines, not just points**
— two rectangles drawn with their own coincident edges mesh as
*disconnected* parts (duplicate coincident lines). Build the shared
edge once and use it in both curve loops (see
`tests/sections/test_warping_s2.py::_three_strip_section` for the
canonical pattern).

## 4. Analyses — what to know beyond the signatures

- **geometric()** — quadrature-**exact** for straight-sided meshes
  (assert `rel=1e-9` against hand integrals in tests, any mesh size).
  Connectivity-blind (valid for disconnected sections: common-centroid
  Steiner terms, no flag needed).
- **warping()** — per-material **G-weighted** Laplacian for torsion
  (exact for heterogeneous G — what makes `G=` overrides physical);
  shear functions use the package convention (E-weighted with a single
  `ν_eff = EA/(2·GA) − 1` — identical to exact when ν is uniform).
  Pure-Neumann singularity regularized by a Lagrange row (∫ω dA = 0),
  never node pinning. **tri3/quad4 warp poorly** →
  `SectionAccuracyWarning`; the fix is `g.mesh.generation.set_order(2)`.
- **`GAs_xy` diverges on (near-)symmetric sections** — that IS the
  answer ("no coupling", `Δ²/κ_xy` convention, same as the PyPI
  package). Compare couplings via `1/GAs_xy`, never the raw value.
- **plastic()** — needs `fy` on every material (a `fy`-less region
  raises naming the PG). Mixed-`fy` composite: `Sxx` raises; the
  `Mp_*` fields ARE the capacities. Shape factors are first-yield
  based (`sf = Mp/My`), reducing to the classic `Z/S` for homogeneous.
- **stress(...)** — a **linear blend of precomputed unit-load nodal
  fields** (never re-solves; the inspector's live inputs ride this).
  Sign conventions (equilibrium-tested): `N` tension+, `Mxx` tension
  at `+y`, `Myy` tension at `+x`, `Mzz` CCW. Recovery is exact nodal
  evaluation averaged **within material regions only** — flat arrays
  take max-|value| at interface nodes; `st.get(component, pg=...)` is
  the exact per-region field (NaN outside). Per-action components kept
  (`sigma_zz_mxx`, `tau_zy_vy`, …) + `tau`, `von_mises`.

## 5. Disconnected sections (`disconnected="raise" | "sum"`)

Default `"raise"` makes `warping()` fail loud on a disconnected mesh —
usually the forgot-to-fragment / duplicate-edge authoring bug (§3), and
a silently-summed `J` would be garbage. Explicit `"sum"` = the
classical multi-girder **lower bound**: per-component Saint-Venant
solves, `GJ = ΣGJᵢ`, `GAs = ΣGAsᵢ`, GJ-weighted shear centre, per-part
results on `warp.parts` (equal twist rate, no inter-part shear
transfer). Effective deck width is **authored, never inferred**.
Partial shear transfer (battens/lacing) is **authored, never a knob**:
draw a thin connecting strip with near-zero `E` + calibrated `G=`.
`stress()` on a `"sum"` section distributes the actions per the ADR
policy: `N`/`Mxx`/`Myy` use the **global** plane-sections composite
state (common centroid, Steiner terms); `Mzz` goes to parts
∝ `GJᵢ/ΣGJ`; `Vx`/`Vy` ∝ the part flexural-rigidity shares
(`EIyyᵢ` / `EIxxᵢ`, equal curvature; scalar per axis — exact for
axis-aligned parts, approximate for in-plane-rotated ones) — each
part recovers τ from its own ω/Ψ/Φ solves. Consistent lower bound; per-part fields equal a
standalone analysis of each part under its distributed share
(exactness-tested), *except* `Myy`/`Mxx` σ when part centroids are
offset — there the global Steiner state governs, deliberately.

## 6. OpenSees handoff — the axis contract

Analyzer results live in gmsh **authoring (x, y)** axes (`Ixx = ∫y²dA`,
PyPI-package convention). One shared lowering
(`sections/_lowering.py`) owns the OpenSees mapping — **authoring x ≡
local z, authoring y ≡ local y**:

| analyzer | OpenSees `ElasticSection` |
|---|---|
| `Ixx_c` (strong axis of an upright I) | `Iz` |
| `Iyy_c` | `Iy` |
| `J` | `J` |
| `As_y / A` | `alphaY` |
| `As_x / A` | `alphaZ` |

(G-B verified against upstream `ElasticShearSection3d.cpp` — `Iz↔MZ`,
`alphaY↔VY` families — and machine-precision Timoshenko cantilevers in
both axes.) **"Local y up ⇒ Iz strong" is the `geomTransf` author's
responsibility** — the lowering is orientation-agnostic; pick `vecxz`
so local y lands on the section's authoring-y axis (for a column along
global Z, `vecxz=(1,0,0)` puts local y on global −Y and local z on
global +X).

Two routes, one lowering:

```python
girder = p.section.ComputedSection(analysis=sec)   # LAZY — resolves at emit
integ  = p.beamIntegration.Lobatto(section=girder, n_ip=5)
p.element.forceBeamColumn(pg="girders", transf=transf, integration=integ)

es = sec.to_elastic_section(E=..., G=..., ndm=3)   # EAGER ElasticSection
```

- `ComputedSection` subclasses the `Section` base → slots into
  `Lobatto` / `Aggregator.base_section` / element `section=` fields
  with zero consumer changes. N references to one analyzer = **one**
  memoized solve; the deck line is **byte-identical** to a hand-typed
  `ElasticSection`.
- Reference-moduli rules (fail-loud at emit naming the handle):
  homogeneous → `E`/`G` default from the single material;
  **composite → explicit `E=`/`G=` REQUIRED** (transformed-section
  `EA/E`, `EI/E`, `GJ/G` — the deck reproduces the analyzer's
  rigidities exactly, whatever reference you pick); geometric-only →
  both required (classic geometry + your deck moduli).
- **`ndm=` selects the `ElasticSection` form** (`3` default →
  `E A Iz Iy G J alphaY alphaZ`; `2` → `E A Iz G alphaY`). It lives on
  the primitive because sections emit before the bridge's
  `ops.model(ndm=)` is visible to any `_emit` — match it to the model
  envelope yourself.
- **`kind="fiber"` lowering (Amendment A2)**: auto-generated `section
  Fiber` — one fiber per Gauss point of the analyzer mesh (3/tri,
  9/quad; `area = w·|J|`, exact area partition), coordinates about the
  **elastic centroid**, same axis identification as elastic
  (authoring x ≡ local z, y ≡ local y — gate G-D verified the signed
  mapping):

  ```python
  conc  = p.uniaxialMaterial.Concrete01(...)   # construct via the
  steel = p.uniaxialMaterial.Steel02(...)      #   bridge (P11!)
  col = p.section.ComputedSection(
      analysis=sec, kind="fiber",
      fibers={"concrete": conc, "steel": steel},  # pg -> UniaxialMaterial,
      GJ=None,                                    #   EXACT cover, never inferred
  )                                               # GJ=None -> warp.GJ; -GJ always emitted
  ```

  `kind="fiber"` forbids `E=`/`G=`/`ndm=` (raise at construction);
  geometric-only analyzers rejected. Fiber count knob = the authored
  mesh size (author a coarser face for nonlinear runs). Caveat: if the
  uniaxial initial moduli differ from `SectionMaterial.E`, the fiber
  section's effective centroid shifts off the element axis
  (documented, no knob).
- **H5 provenance persists (Amendment A1, schema 2.20.0)**: the
  resolved numbers already ride the ordinary section capture
  (`/opensees/sections/*`); every emitted `ComputedSection` also gets
  a row in the `/opensees/computed_sections` sidecar — `(tag,
  analyzer_name, JSON payload)` with kind, materials, disconnected
  policy, reference moduli (elastic) or `GJ`+`fiber_pgs` (fiber).
  Read back via `OpenSeesModel.from_h5(...).computed_sections()`.
  Hash-excluded (a `ComputedSection` deck has the same `model_hash`
  as the hand-typed equivalent); the analyzer mesh is NOT persisted;
  `g.compose` still drops the whole `/opensees/` zone (ADR 0055
  FILTER) — re-declare `ComputedSection`s in the composing script.

## 7. Flat-face builders (`g.sections.*_face`)

Solid recipes minus the extrude — no `length`/`anchor`/`align`;
in-plane `translate=(dx, dy)` + scalar `rotate` (degrees, about the
world origin, applied before translate); **auto-PG named after
`label`** (exactly what `materials=` consumes); return `Instance`:

```
W_face(bf, tf, h, tw)   rect_face(b, h)          rect_hollow_face(b, h, t)
pipe_face(r)            pipe_hollow_face(r, t)   angle_face(b, h, t)
channel_face(bf, tf, h, tw)                      tee_face(bf, tf, h, tw)
```

Accuracy expectation vs AISC catalog (fillet-less plate assembly):
`A`/`Ix`/`Iy` land within ~1–2 % (fillets add area); **`J` lands
5–15 % below catalog** (J is fillet-sensitive) — don't chase that gap
with mesh refinement, it's a modeling difference, not an error.

## 7b. Plotting surface (all matplotlib, all headless-safe)

| call | what you get |
|---|---|
| `sec.plot()` | one-call overview **Figure**: glyphed section view + the `summary()` report panel |
| `sec.plot_mesh(ax=)` | PG-colored wireframe |
| `sec.plot_section(centroid=, shear_centre=, principal_axes=, ax=)` | mesh + glyph overlay (`shear_centre=False` for disconnected-`raise` sections) |
| `sec.plot_warping(shear_flow=, ax=)` | Saint-Venant ω contour (per part under `"sum"`); `shear_flow=True` overlays the unit-torsion τ quiver (under `"sum"` each part shows its `GJᵢ/ΣGJ` share) |
| `sec.stress(...).plot(component, ax=)` | filled tricontour of any component (incl. per-action `sigma_zz_mxx`, `tau_zy_vy`, …) |
| `st.plot_vector(action=None, ax=)` | (τ_zx, τ_zy) quiver; `action ∈ "mzz"/"vx"/"vy"` for one term |
| `st.plot_mohrs_circle(at=(x, y), pg=)` | Mohr's circle of the beam state (σ_zz, τ) at the node nearest `at`; `pg=` picks the exact per-region value at interfaces |
| `g.sections.plot_faces(ax=)` | **pre-mesh** geometry preview — face outlines + auto-PG name annotations (sanity-check placements before meshing) |

Sanity oracle worth remembering: a **circular** section's ω is ~0
everywhere (circles don't warp) — a visibly non-zero ω contour on a
disk means the mesh/solve is wrong.

## 8. Inspector (`sec.viewer()`)

Standalone Qt + matplotlib panel — deliberately NOT the
0014/0042/0056 viewer family. Left: mesh + glyphs (centroid, shear
centre, principal axes, PG colors) switching to stress contours;
right: Geometric/Warping/Plastic tables (composite → `e_ref` input
drives a transformed column) + six load spinboxes re-blending the unit
fields live (no solve ever on the UI thread). `sec.viewer` defaults
to `blocking=True` — **notebooks must pass `blocking=False`** (`%gui qt`).
(`results.viewer` is different: it auto-detects the kernel; see results.md.)
Qt absent → `ImportError` with guidance, `QT_QPA_PLATFORM=offscreen`
on Windows → `RuntimeError`. Everything is equally reachable headless:
`summary()`, `plot_mesh()`, `plot_section()`, `stress(...).plot()`.

## 9. Testing / validation lessons

- Analytic oracles are the primary gate: rectangle/circle exact
  integrals, `J = πr⁴/2`, rect-`J` series, thin-channel shear centre,
  `Z = bh²/4`. Cheap closed forms catch what plausible-looking numbers
  hide (a transposed Jacobian produced all-wrong-but-plausible
  gradients once — an oracle caught it, 28 review agents didn't).
- The PyPI `sectionproperties` package is a **dev-only oracle**
  (`pip install -e ".[section-oracle]"`; tests `importorskip` it; only
  the CI suite lane installs it). Never a runtime dep.
- Deck-level assertions: compare `ComputedSection` emission
  byte-for-byte against a hand-typed `ElasticSection` via
  `TclEmitter().lines()`; count solves by monkeypatching
  `apeGmsh.sections._analysis.compute_warping`.

## 10. Section documents + the builder GUI (ADR 0080)

`SectionDocument` is versioned JSON (`SECTION_DOC_VERSION`, ADR 0023
additive-minor window) describing ONE section. It is the source of
truth; `launch_builder()` is an editor for it. **Parity law** — every
GUI action is a document mutation, so anything the window does a
script can do. One document, one lane.

```python
from apeGmsh.sections import (
    SectionDocument, launch_builder, moment_curvature, handoff_snippet,
)
doc = SectionDocument.new(name="col500", kind="fiber", units="N, mm")   # or "continuum"
doc = SectionDocument.open("col500.section.json"); doc.save(path)
```

**Continuum lane** — `add_shape(<one of the eight *_face>, id=, material=,
translate=, rotate=, **params)`, `add_polygon(points, id=)`,
`set_mesh(lc=, order=2)`, `set_disconnected("raise"|"sum")`;
`build()` → `SectionProperties`. Booleans: **`add_embed(outer, inner)`
is THE composite primitive** (cut with `remove_tool=False` then
`fragment_pair`, one step) — the double-cover trap of §3 is not
expressible through it. Raw `add_cut(target, tool, remove_tool=)` /
`add_fragment_pair(a, b)` remain for non-overlapping work and
sacrificial hole tools.

**Fiber lane** — `add_patch_rect` / `add_patch_circ` /
`add_layer_straight` / `add_point` / `set_GJ`, plus RC templates via
`add_template(name, materials={role: mat_name}, **params)` with
`rc_rect_column` / `rc_circ_column` / `rc_beam`. Templates are stored
AS PARAMETERS and re-expanded on every `build()` (→ `FiberRecipe`), so
editing `cover` moves every bar. `core_split=True` splits the concrete
into confined-core + cover patches — roles become `("core", "cover",
"bars")` instead of `("concrete", "bars")`, and `materials=` must cover
the roles **exactly**.

Template traps: `cover` is to the **bar centre**; `bars_x`/`bars_y`
count bars per face **including corners** (4/4 → 12 bars, not 16: two
straight layers of 4 plus `bars_y-2` interior points per side).

**Materials** are dual-role: `set_material(name, E=, nu=, G=, fy=,
density=, uniaxial=("<Type>", {...}))`. The continuum build needs the
first role on used materials; the fiber handoff needs the second.
`<Type>` is the token `ops.uniaxialMaterial.<Type>` takes (so
`"ElasticMaterial"`, not `"Elastic"`).

**Bars overlay** (continuum only, ADR 0080 B3): `add_bar(material=, x=,
y=, area=)` / `add_bar_line(material=, n=, area=, start=, end=)` in
AUTHORING coords. Resolved at handoff by appending bar `FiberPoint`s to
the Gauss fibers. Concrete area is NOT deducted at bar locations.

**Handoff** — fiber: `doc.to_section(ops)` → registered
`ops.section.Fiber`. Continuum: `ops.section.ComputedSection(
analysis=doc.build())` for the elastic lowering, `doc.to_section(ops)`
when a bars overlay exists (the elastic lowering would silently drop
it). `handoff_snippet(doc, path=...)` renders those lines paste-ready
(the GUI's "Copy handoff"); `doc.export_script(path)` writes the
equivalent plain apeGmsh script — one-way, round-trip editing is the
JSON's job.

**Moment–curvature** (fiber lane only):

```python
mc = moment_curvature(doc, axis="z", kappa_max=8e-5, n_steps=60, axial=-1500e3)
mc.EI0        # first-step secant = the EXACT fiber sum ΣE·A·r²
mc.M_max      # largest |M|, signed;  mc.curvature / mc.moment are tuples
mc.complete   # False -> a step failed before κ max; the partial curve is valid
```

`axis="z"` is DOF 6 (elastic slope = `EIxx_c`), `"y"` is DOF 5
(`EIyy_c`). `kappa_max` is SIGNED. `axial` is OpenSees-signed
(compression negative), applied first then held with `loadConst`.
**Two hard contracts**: it `wipe()`s the process-global OpenSees domain
(never call it with a live analysis open), and it runs synchronously on
the calling thread — openseespy is a non-reentrant C++ runtime, so it
must NOT be moved onto a worker. (The B6 properties worker is threaded
only because it drives Gmsh.) Documents with no `GJ` get an inert
placeholder — the harness fixes the torsional DOF at both nodes.

**GUI** — `launch_builder(path_or_doc=None, blocking=True)`; notebooks
pass `blocking=False` (`%gui qt`). Same S6 contract as the inspector
(Qt absent → `ImportError`; `QT_QPA_PLATFORM=offscreen` on Windows →
`RuntimeError` from the launcher, window class stays
offscreen-constructible). Drafting aids on the polygon tool: **F7**
grid, **F9** object snap, **F8** ortho, typed `length<angle` / `dx,dy`
/ `x,y`; the resolvers live Qt-free in `_drafting.py`. The live
properties panel solves on a worker thread and **meshes in a child
process** — Gmsh is one process-global, non-reentrant runtime, so
driving it from a background thread of a process that also drives it
from the main thread aborts the interpreter (no traceback). The
process-wide runtime lock in `_session.py` serializes every *other*
threaded Gmsh use; the worker sidesteps it entirely, because a session
left open would hold that lock for its lifetime. Two optional deps,
both fail-soft and
never required headless: **apeSteel** (catalog picker prefills the
`W_face` form in MILLIMETRES — and its `h` is the CLEAR web height, not
catalog `d`; absent → picker hidden) and **openseespy / the Ladruno
fork** (M–κ button; absent → greyed with guidance).

## Source map

`src/apeGmsh/sections/`: `_analysis.py` (broker) · `_snapshot.py`
(gates) · `_geometric.py` / `_warping.py` / `_plastic.py` /
`_stress.py` (analyses) · `_lowering.py` (THE axis mapping — nothing
else maps axes) · `_inspector.py` (Qt panel) · `_builder.py`
(`g.sections` incl. `*_face`) · `_materials.py` / `_errors.py`.
ADR 0080: `_document.py` (`SectionDocument`, and
`typed_fiber_items` — THE recipe→primitives construction, shared by
the handoff and the M–κ harness) · `_rc_templates.py` ·
`_script_export.py` (export + the shared fiber renderers) ·
`_handoff.py` · `_mc.py` · `_catalog.py` · `_builder_gui.py` ·
`_drafting.py` · `_properties.py`.
Bridge: `src/apeGmsh/opensees/section/computed.py`. Authoritative
contracts: ADR 0078 (analyzer) + ADR 0080 (document/builder); user
guides: <https://nmorabowen.github.io/apeGmsh/concepts/sections/> and
<https://nmorabowen.github.io/apeGmsh/how-to/author-a-section-document/>.
