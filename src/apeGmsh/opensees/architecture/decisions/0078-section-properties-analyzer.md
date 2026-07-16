# ADR 0078 — In-process cross-section property analyzer (`SectionProperties`)

**Status:** Proposed (2026-07-16)

## Context

Every beam-line primitive in `opensees/section/` *demands* section
properties from the user and computes none of them. `ElasticSection`
takes `E, A, Iz, Iy, G, J, alphaY, alphaZ` and only validates `> 0`
(`section/beam.py`); `Fiber` takes a hand-laid patch/layer geometry;
`Aggregator` composes user-supplied uniaxial laws. For catalog shapes
the numbers come from AISC/EN tables (apeSteel owns that path), but for
anything non-catalog — built-up plates, haunched girders, CFTs, stiffened
cores, retrofit jackets, arbitrary DXF imports — the user today leaves
apeGmsh, runs the PyPI `sectionproperties` package (or hand integrals),
and types the numbers back in. That round-trip abandons exactly the
things apeGmsh is good at: OCC geometry with holes and booleans,
labeled multi-material regions, and quality meshing.

The reference capability is the `sectionproperties` package (Pilkey,
*Analysis and Design of Elastic Beams*): on a mesh of the cross-section
it computes

- **geometric** — area, perimeter, first moments, centroid,
  `Ixx/Iyy/Ixy` (global + centroidal), principal axes and angle φ,
  section moduli, radii of gyration; modulus-weighted for composites;
- **warping** — the Saint-Venant torsion constant `J`, warping constant
  `Γ`, shear centre, shear areas `As_x/As_y` (a scalar FEM solve on the
  section mesh);
- **plastic** — plastic centroid, plastic moduli `S/Z`, shape factors;
- **stress** — the linear-elastic σ_zz/τ fields from
  `N, Vx, Vy, Mxx, Myy, Mzz`.

Three structural facts shape the design:

1. **apeGmsh already owns the hard input.** `add_plane_surface`
   supports holes natively (first loop = boundary, rest = holes),
   boolean `fragment` + physical groups give multi-material regions,
   and 2-D meshing produces `tri3/tri6/quad4/quad8/quad9` — a strictly
   better front end than the `shapely`+`triangle` (Tri6-only) mesher
   the PyPI package carries. `FEMData` exposes node coords, per-type
   connectivity, and PG membership — the complete solver input.

2. **The element kernel already exists and is shared.** `apeGmsh.fem`
   was created (three-broker refactor) as the neutral home for shape
   functions and quadrature precisely so multiple consumers could
   integrate over elements: `_shape_functions.get_shape_functions()`
   covers every 2-D type above with `N`/`dN`/Jacobian helpers, and
   `_quadrature` has matching Gauss rules (`gauss_tri`,
   `gauss_quad_2d`). Today its consumers are HRZ mass lumping and
   Gauss-point extrapolation; a section assembly is a third consumer,
   not new machinery.

3. **The solve is too small to outsource.** The warping problem is one
   symmetric-positive-semidefinite scalar Laplacian (plus two RHS
   re-solves for shear), thousands of DOFs at most. Round-tripping
   through an OpenSees deck (our only solver today) would be absurd
   overhead and would put a pre-processing-time computation behind the
   bridge. `scipy.sparse` — already a dependency — solves it in
   milliseconds. This is deliberately apeGmsh's **first in-process
   solve**; ADR 0009's "no back-compat with `apeGmsh.solvers`" bans a
   *structural* solver competing with OpenSees, not element-level
   integrals on the pre-processing side (HRZ lumping already crossed
   that line in spirit).

Naming hazard: `g.sections` is **taken** — it is the parametric *solid*
builder (`W_solid`, `rect_hollow`, …, all extruding a `length` into
3-D). A cross-section analyzer consumes a *flat* 2-D mesh and is not
session-bound, so it must not be forced into that composite.

## Decision

Add a **standalone analyzer class `SectionProperties`** — a
FEMData-consuming broker in the mold of `apeSees(fem)` / `Results` —
that computes geometric, warping, plastic, and stress results for an
arbitrary meshed 2-D cross-section, **natively in-process** over
`apeGmsh.fem` shape functions + quadrature + `scipy.sparse`. No new
runtime dependency; the PyPI `sectionproperties` package is used only
as a **dev-only test oracle**.

```python
from apeGmsh import apeGmsh, SectionProperties
from apeGmsh.sections import SectionMaterial

g = apeGmsh("box_girder")
# author the flat face in the gmsh (x, y) plane: OCC loops, holes,
# fragment for multi-material regions, PGs per material
...
g.mesh.generation.generate(dim=2)
g.mesh.generation.set_order(2)                    # tri6 / quad9
fem = g.mesh.queries.get_fem_data(dim=2)

sec = SectionProperties(
    fem,
    materials={                                   # pg name -> material
        "deck":  SectionMaterial(E=30e9,  nu=0.2,  fy=30e6),
        "steel": SectionMaterial(E=200e9, nu=0.3,  fy=350e6),
    },
)

geo  = sec.geometric()      # -> GeometricProperties (frozen)
warp = sec.warping()        # -> WarpingProperties   (frozen)
plas = sec.plastic()        # -> PlasticProperties   (frozen)

warp.J, (warp.x_sc, warp.y_sc), warp.As_y
geo.Ixx_c, geo.phi, geo.I11_c

st = sec.stress(N=1e3, Vy=5e2, Mxx=2e4, Mzz=1e3)  # -> SectionStress
st.sigma_zz, st.tau_zx, st.tau_zy, st.von_mises   # per-node arrays
st.plot("von_mises")                               # matplotlib contour

# declarative binding: the frame model consumes the *declaration*;
# properties are resolved at emit, like every other apeSees resolution
p = apeSees(frame_fem)
girder = p.section.ComputedSection(analysis=sec)   # deferred lowering
integ = p.beamIntegration.Lobatto(section=girder, n_ip=5)

elastic = sec.to_elastic_section()                 # eager escape hatch
```

### Placement and lifecycle

- Module: the analyzer lives in the existing `src/apeGmsh/sections/`
  package as siblings of `_builder.py` — `_analysis.py` (class +
  orchestration), `_geometric.py`, `_warping.py`, `_plastic.py`,
  `_stress.py`, `_materials.py`. One "sections" package, two entries:
  the session-bound solid builder (`g.sections`) and the
  FEMData-consuming analyzer (`SectionProperties`).
- Export: `from apeGmsh import SectionProperties` top-level, mirroring
  `Results` / `FEMData` / `Part`.
- The constructor snapshots what it needs from `FEMData` (coords,
  per-type connectivity, PG membership) — after construction it is
  independent of the gmsh session, per the ADR 0001 doctrine.
- Results are cached per analysis kind; each `*Properties` object is a
  frozen dataclass. `warping()` and `plastic()` internally require
  `geometric()` (auto-run, cached). No mutable "run flags" surface.

### Input contract (fail-loud, per the resolution-contract law)

- **Any meshed face qualifies** — the three authoring routes are
  equals: raw OCC geometry (`add_point`/`add_spline`/`add_curve_loop`/
  `add_plane_surface` with holes — arbitrary irregular outlines),
  imported CAD (`g.model.io.load_dxf` / `load_step`), or the
  parametric builders (§ flat-face builders — convenience only, never
  required).
- The mesh must be **2-D elements only** (`tri3/tri6/quad4/quad8/quad9`),
  all in one plane. 3-D or mixed-dim `FEMData` → `SectionMeshError`
  naming the offending types. Straight-sided elements are assumed
  (affine gmsh output); curved boundary faces are not detected.
- `materials=` maps **physical-group names → `SectionMaterial(E, nu,
  G=None, fy=None, density=None)`** (`G` defaults isotropic; see the
  API contract for the override's purpose). Every 2-D element must be
  covered by exactly one named PG; uncovered or doubly-covered elements →
  `SectionMeshError` listing them by PG. Omitting `materials=` entirely
  runs the **geometric-only mode**: unit `E`, results are pure
  geometric properties (the classic non-composite numbers).
- **Disconnected sections** (twin girders, spaced boxes, a girder plus
  an equivalent deck strip) are governed by a constructor-level policy
  `disconnected="raise" | "sum"`:
  - `geometric()` and `plastic()` are **connectivity-blind** in either
    mode — their integrals sum over parts, which is exactly the
    plane-sections / equal-curvature assumption an equivalent spine
    beam already makes.
  - `warping()` under the default `"raise"` requires a **single
    connected domain** (node-adjacency graph; disconnected → fail-loud
    with the component count). The default is deliberate: a
    disconnected mesh is *usually* the authoring bug of touching faces
    never fragmented (duplicated interface nodes), which a permissive
    solver would convert into a silently garbage `J`.
  - `warping()` under explicit `"sum"` solves Saint-Venant **per
    connected component** and combines: `GJ = ΣGJᵢ`, `GAs = ΣGAsᵢ`,
    `EΓ = ΣEΓᵢ`, shear centre = the `GJᵢ`-weighted mean of part shear
    centres. Mechanical meaning, stated in the docstring: equal twist
    rate, each part twisting about its own shear centre, **no
    inter-part shear transfer** — the classical multi-girder lower
    bound. Cross-frame / diaphragm warping interaction is *out of
    plane* and out of scope; the real coupled system is stiffer.
    Per-part results are kept on `WarpingProperties.parts`.
  - `stress()` follows the policy: `N/Mxx/Myy` recovery uses the
    global plane-section assumption unchanged; under `"sum"`, `Mzz`
    is distributed to parts ∝ `GJᵢ/ΣGJ` and `Vx/Vy` ∝ the part
    flexural-rigidity shares (consistent with equal curvature).
  - Shear lag / effective width is the **user's modeling decision**,
    here as everywhere: the analyzer computes what is drawn — author
    the effective deck width into the face, don't expect the solver
    to infer it.
  - **Partial shear transfer is authored, never a knob.** Built-up
    members whose parts *are* coupled — battened or laced columns, two
    chords plus discrete connectors — sit between the two bounds, and
    the solver deliberately has no coupling factor. The bounds come
    free: `geometric()` on the disconnected mesh already integrates
    about the common centroid, so the Steiner `ΣAᵢdᵢ²` terms give the
    fully-composite flexural rigidity (the code-level shear-flexibility
    slenderness knockdown, AISC E6, is a *member*-design concern —
    apeSteel's layer, not the section's); `disconnected="sum"` is the
    zero-transfer torsion floor. For the in-between: **draw the
    connection** — a thin strip joining the parts (fragmented, so the
    mesh is connected), carrying a `SectionMaterial` with near-zero `E`
    and an explicit `G` calibrated so the strip's shear stiffness per
    unit length matches the connector system's (Timoshenko built-up
    member theory / code `Sv` formulas — the calibration is the user's,
    the mechanism is the `G` override). Rigid `G` recovers the
    closed-section upper bound. Stresses reported inside a fictitious
    strip are meaningless by construction — read real parts via
    `stress().get(component, pg=...)`.
- `plastic()` requires `fy` on every material.
- `tri3`/`quad4` meshes are accepted for `geometric()`/`plastic()` but
  **warn** (`SectionAccuracyWarning`) on `warping()`/`stress()`:
  constant-strain elements converge poorly on `J` and shear areas —
  the guidance is `set_order(2)`. (This is why the PyPI package is
  Tri6-only; we keep the flexibility but say so out loud.)
- Units are the user's, consistently (N-mm, N-m, …) — same
  unit-agnostic stance as the rest of apeGmsh.

### Axis and load conventions

The section is authored in the **gmsh (x, y) plane**, and all analyzer
results are reported in those authoring axes — `Ixx = ∫y²dA`,
`Iyy = ∫x²dA`, identical to the `sectionproperties` package so oracle
comparison and user intuition are 1:1. Loads on `stress()` use the
same frame: `N` (axial, out-of-plane), `Vx, Vy` (shear), `Mxx, Myy`
(bending about x/y), `Mzz` (torsion).

The **bridge handoff owns the OpenSees mapping**: OpenSees frame
sections live in local (y, z) with local-y conventionally "up". The
identification is *authoring x ≡ local z, authoring y ≡ local y*, so:

| analyzer (authoring axes) | OpenSees `ElasticSection` |
|---|---|
| `Ixx_c` (about horizontal axis) | `Iz` (strong axis of an upright I) |
| `Iyy_c` | `Iy` |
| `J` | `J` |
| `As_y / A` | `alphaY` |
| `As_x / A` | `alphaZ` |

Both bridge paths (below) perform exactly this mapping in one shared
lowering function. For a homogeneous section with materials supplied,
`E`/`G` default to the (single) material's values; in geometric-only
mode they are required. For a **composite** section the properties are
modulus-weighted (see below) and the lowering requires an explicit
reference `E` — it produces the transformed-section properties
`EA/E, EI/E, GJ/G` and says so in its docstring.

### Declarative binding to the bridge (`ComputedSection`)

The analyzer is a **declaration, not just a calculator**. It is
session-decoupled at construction, its inputs are frozen, and its
analyses are cached frozen values — so the bridge can hold a reference
to it and resolve *at emit time*, exactly like every other apeSees
resolution (material tags, PG membership, per-node ndf, damping
scopes). Concretely:

- New primitive `opensees/section/computed.py` —
  `ComputedSection(analysis=<SectionProperties>, E=None, G=None)`, a
  frozen `kw_only`/`slots` dataclass subclassing the same `Section`
  base as `ElasticSection`. Because every downstream consumer
  (`Lobatto`/`beamIntegration`, `Aggregator.base_section`,
  `zeroLengthSection`, element `section=` fields) types against the
  `Section` base with emit-time `resolve_tag`, **`ComputedSection`
  slots into all of them with zero consumer changes**.
- At emit, its `_emit` calls the shared lowering (running
  `geometric()` + `warping()` lazily if not yet cached — results are
  memoized on the analyzer, so N references = one solve) and emits a
  plain `section Elastic $tag $E $A $Iz $Iy $G $J <alphaY alphaZ>`
  line. **No `Emitter` Protocol widening, no H5 schema change** — the
  deck is indistinguishable from a hand-typed `ElasticSection`.
- Analysis failure at emit (disconnected domain, missing reference `E`
  on a composite) fails loud with the section's name/handle — never a
  silent fallback, per the resolution-contract law.
- Two escape hatches remain eager: `sec.to_elastic_section()` returns
  a plain populated `ElasticSection` (inspectable numbers, decoupled
  from the analyzer), and the properties objects are plain data for
  hand-use.
- Reserved, not implemented: a `kind=` axis on `ComputedSection` for
  future lowerings (e.g. `"fiber"` — auto-generated `FiberPoint`s from
  the mesh with per-region materials). Elastic is the only lowering
  this ADR ships.
- Persisting the declaration into `model.h5` (so a composed model
  carries its computed sections) is **deferred** — bridge-side section
  primitives are not H5-persisted today, and this ADR does not change
  that.

### Analyses

**Geometric** — Gauss-loop over elements accumulating
`∫E dA, ∫E x dA, ∫E y dA, ∫E x² dA, …` (modulus-weighted throughout;
unit `E` reduces to the classic integrals). Derived: centroid,
centroidal + principal second moments, φ, elastic section moduli,
radii of gyration, perimeter (boundary-edge walk), mass (when
`density` given). This is pure quadrature — no solve.

**Warping (Saint-Venant)** — assemble the scalar Laplacian
`K = ∫ ∇Nᵀ G ∇N dA` (shear-modulus field; per material `G` defaults to
`E/(2(1+ν))` unless overridden) over the section mesh and solve three
RHS's:

1. warping function ω: `K ω = f_ω` — a **pure-Neumann (singular)**
   system, regularized by a Lagrange-multiplier row enforcing
   `∫ω dA = 0` (not node-pinning, which distorts the shear-centre
   integrals on coarse meshes);
2. shear functions Ψ, Φ (Pilkey): two more solves against the same
   factorization.

Derived: `J` (torsion constant), shear centre (elasticity + Trefftz),
warping constant `Γ`, shear areas `As_x, As_y` and the cross term,
monosymmetry constants. One `scipy.sparse.linalg` factorization
(`splu`), three back-substitutions. Under `disconnected="sum"` the
same machinery runs once per connected component (one factorization
each) and the results combine per the input-contract policy.

**Plastic** — bisection on the neutral-axis position along the
centroidal and principal axes, each trial evaluating a signed
`∫fy dA` imbalance by Gauss quadrature with per-element side
classification (the standard `sectionproperties` algorithm). Derived:
plastic centroids, `S_xx/S_yy` (+ principal), shape factors. Documented
as **invalid for strain-softening/nonlinear materials** (e.g. plain
concrete) — same caveat as the reference package.

**Stress** — closed-form linear-elastic recovery on the already-solved
fields: `σ_zz` from `N/Mxx/Myy` (transformed-section), `τ` from `Mzz`
(∇ω) and `Vx/Vy` (Ψ, Φ). Evaluated at Gauss points, extrapolated to
nodes with the existing `results/_gauss_extrapolation` machinery, and
averaged per material region (no averaging across material
interfaces). `SectionStress` exposes per-node arrays plus a small
matplotlib `plot()` (tricontour over the mesh).

Because the solves are load-independent, stress recovery is a **linear
blend of six precomputed unit-load fields** (per unit `N, Vx, Vy, Mxx,
Myy, Mzz`) — evaluating a new load vector is a weighted sum, not a
re-solve. This is what makes the inspector's live load inputs (below)
cheap.

### Section inspector (`sec.viewer()`) — own panel, not the viewer family

The **ADR 0014/0042/0056 viewer family is explicitly not joined**: those
contracts (out-of-process H5 consumption, SceneLayer IR, VTK render
backends, owner-fired event dispatch) exist for 3-D scenes with time
axes and picking. A cross-section is a small, static, 2-D domain — the
right tool is a **standalone lightweight Qt panel with an embedded
matplotlib canvas**, in `sections/_inspector.py`, no `model.h5`, no
render seam, no session sidecar:

- **Left — canvas**: the meshed section with glyph overlays (elastic
  centroid, plastic centroid, shear centre, principal axes at φ, PG
  coloring/legend); switches to stress contours when a component is
  selected.
- **Right — properties**: tabbed read-only tables (Geometric / Warping
  / Plastic — tabs appear as each analysis is available or on demand;
  composite sections show the rigidity-form values plus an `e_ref`
  input driving a transformed column).
- **Right — loads**: six spinboxes (`N, Vx, Vy, Mxx, Myy, Mzz`) + a
  stress-component picker; edits re-blend the unit fields and redraw
  live (no solve in the UI thread, ever — solves happen once, before
  or at panel open).
- API: `sec.viewer(*, blocking: bool = True)` — same contract as
  `results.viewer`: **notebooks must pass `blocking=False`** (a
  blocking Qt loop kills the kernel), and the offscreen-platform guard
  (`QT_QPA_PLATFORM=offscreen` raises) is reused from `ViewerWindow`.
  Qt absent → `ImportError` with guidance; every capability is equally
  reachable headless (`summary()`, `plot_section()`, `stress().plot()`).

Notebook-first affordances ship with S1, not S6: `summary()` returns a
plain-text properties report, and the analyzer + every `*Properties`
dataclass implement **`_repr_html_`** so evaluating `sec` or
`sec.geometric()` in a Jupyter cell renders the properties table with
no window at all.

### API contract

The full landing surface. All result objects are frozen `slots`
dataclasses; all arrays are `float64` numpy.

**One naming law for composite-weighted values.** Rigidity-form fields
carry the modulus prefix (`EA`, `EIxx_c`, `GJ`, `GAs_y`) and are
*always* valid. The familiar unprefixed names (`Ixx_c`, `J`, `As_y`,
`Zxx_plus`, …) are **properties that divide by
the section's single modulus**: in geometric-only mode `E = G = 1` so
they are the classic numbers; for a homogeneous section they divide by
its one material's `E`/`G`; for a **composite** they raise
`CompositeSectionError` naming `transformed(e_ref=...)` — never a
silently-chosen reference modulus. `transformed(e_ref)` returns the
same dataclass shape with every rigidity divided by `e_ref`.

```python
# apeGmsh/sections/_materials.py
@dataclass(frozen=True, kw_only=True, slots=True)
class SectionMaterial:
    E: float                      # Young's modulus (> 0)
    nu: float                     # Poisson's ratio
    G: float | None = None        # shear-modulus OVERRIDE; default isotropic
                                  #   E/(2(1+nu)). Independent G exists for
                                  #   equivalent shear media (smeared battens/
                                  #   lacing, corrugated webs): tiny E + tuned G
                                  #   transfers shear without parasitic
                                  #   flexural area. The solver assembles the
                                  #   E-field (geometric) and G-field (warping)
                                  #   separately, so this is free.
    fy: float | None = None       # required by plastic()
    density: float | None = None  # enables mass / per-length weight
    name: str | None = None       # display only

# apeGmsh/sections/_analysis.py
class SectionProperties:
    def __init__(
        self,
        fem: FEMData,
        *,
        materials: Mapping[str, SectionMaterial] | None = None,  # PG name -> material
        name: str | None = None,   # handle used in fail-loud messages
        disconnected: Literal["raise", "sum"] = "raise",  # multi-part policy
    ) -> None: ...
        # snapshots coords/connectivity/PG membership; all input gates
        # (2-D-only, planarity, PG coverage) raise SectionMeshError here

    # -- analyses: memoized, frozen returns, auto-run prerequisites --
    def geometric(self) -> GeometricProperties: ...
    def warping(self) -> WarpingProperties: ...      # needs geometric()
    def plastic(self) -> PlasticProperties: ...      # needs geometric(); fy on every material
    def stress(
        self, *,
        N: float = 0.0, Vx: float = 0.0, Vy: float = 0.0,
        Mxx: float = 0.0, Myy: float = 0.0,
        M11: float = 0.0, M22: float = 0.0,          # principal-axis moments
        Mzz: float = 0.0,
    ) -> SectionStress: ...                          # needs geometric()+warping()
    def analyze(self) -> "SectionProperties": ...    # geometric+warping+plastic; returns self

    # -- handoff --
    def to_elastic_section(
        self, *, E: float | None = None, G: float | None = None,
    ) -> ElasticSection: ...                         # eager; shared lowering

    # -- introspection / display --
    name: str | None
    materials: Mapping[str, SectionMaterial]         # read-only view
    def plot_mesh(self, *, ax=None) -> "Axes": ...   # matplotlib, PG-colored
    def plot_section(self, *, centroid=True, shear_centre=True,
                     principal_axes=True, ax=None) -> "Axes": ...  # glyph overlay
    def summary(self) -> str: ...                    # plain-text report
    def _repr_html_(self) -> str: ...                # Jupyter table (also on
                                                     #   every *Properties class)
    def viewer(self, *, blocking: bool = True) -> None: ...  # Qt inspector
                                                     #   (notebooks: blocking=False)
```

```python
@dataclass(frozen=True, slots=True)
class GeometricProperties:
    # pure geometry — valid in every mode
    area: float                   # sum of element areas
    perimeter: float              # exterior boundary walk (holes excluded)
    mass: float | None            # per unit length; None unless density given
    cx: float; cy: float          # elastic (modulus-weighted) centroid
    phi: float                    # principal-axis angle, degrees
    # rigidity form — always valid
    EA: float
    EQx: float; EQy: float                     # first moments, global axes
    EIxx_g: float; EIyy_g: float; EIxy_g: float
    EIxx_c: float; EIyy_c: float; EIxy_c: float
    EI11_c: float; EI22_c: float
    # unprefixed properties (homogeneous / geometric-only; else raise):
    #   Qx Qy Ixx_g Iyy_g Ixy_g Ixx_c Iyy_c Ixy_c I11_c I22_c
    #   Zxx_plus Zxx_minus Zyy_plus Zyy_minus     (elastic section moduli)
    #   Z11_plus Z11_minus Z22_plus Z22_minus
    #   rx ry r11 r22                             (radii of gyration)
    def transformed(self, e_ref: float) -> "GeometricProperties": ...

@dataclass(frozen=True, slots=True)
class WarpingProperties:
    x_sc: float; y_sc: float          # shear centre (elasticity), authoring axes
    x_sc_t: float; y_sc_t: float      # shear centre (Trefftz), package parity
    # rigidity form
    GJ: float
    EGamma: float                     # warping rigidity
    GAs_x: float; GAs_y: float        # shear rigidities
    GAs_xy: float                     # cross term
    beta_x_plus: float; beta_x_minus: float   # monosymmetry constants
    beta_11_plus: float; beta_11_minus: float
    beta_22_plus: float; beta_22_minus: float
    parts: tuple["WarpingProperties", ...]    # per-component results under
                                              #   disconnected="sum"; () when
                                              #   the domain is connected
    # unprefixed properties (same law): J Gamma As_x As_y As_xy
    #   alpha_x alpha_y                (= As/area shear-area factors)
    def transformed(self, *, e_ref: float, g_ref: float) -> "WarpingProperties": ...

@dataclass(frozen=True, slots=True)
class PlasticProperties:
    x_pc: float; y_pc: float          # plastic centroid, authoring axes
    x11_pc: float; y22_pc: float      # plastic centroid, principal axes
    # fy-weighted form — always valid (composite: these ARE Mp values)
    Mp_xx: float; Mp_yy: float
    Mp_11: float; Mp_22: float
    sf_xx_plus: float; sf_xx_minus: float     # shape factors (± fibre)
    sf_yy_plus: float; sf_yy_minus: float
    sf_11_plus: float; sf_11_minus: float
    sf_22_plus: float; sf_22_minus: float
    # unprefixed properties (homogeneous: divide by the one fy):
    #   Sxx Syy S11 S22                (plastic section moduli)

class SectionStress:
    loads: Mapping[str, float]        # echo of the applied actions
    # per-node arrays (n_nodes,); material-interface nodes averaged
    # within each region only — see get(pg=) for the exact per-region view
    sigma_zz: np.ndarray              # combined axial + bending
    tau_zx: np.ndarray; tau_zy: np.ndarray   # combined torsion + shear
    tau: np.ndarray                   # magnitude
    von_mises: np.ndarray
    def get(self, component: str, *, pg: str | None = None) -> np.ndarray: ...
        # component ∈ {"sigma_zz", "sigma_zz_n", "sigma_zz_mxx", ...,
        #   "tau_zx_mzz", "tau_zy_vy", ...} — per-action terms kept
    def plot(self, component: str = "von_mises", *,
             ax=None, cmap="coolwarm", levels: int = 15) -> "Axes": ...
```

```python
# apeGmsh/opensees/section/computed.py
@dataclass(frozen=True, kw_only=True, slots=True)
class ComputedSection(Section):
    analysis: SectionProperties       # the declaration (identity-hashed)
    E: float | None = None            # reference moduli; required iff
    G: float | None = None            #   composite (else default from
                                      #   the single material / must be
                                      #   given in geometric-only mode)
    # _emit(): shared lowering -> emitter line identical to
    #   ElasticSection(E, A, Iz, Iy, G, J, alphaY, alphaZ)
    # dependencies(): ()  — no upstream primitives
```

Errors and warnings (in `apeGmsh/sections/_errors.py`, re-exported
from `apeGmsh.sections`): `SectionMeshError` (input gates),
`CompositeSectionError` (unprefixed accessor on a composite),
`SectionAnalysisError` (solve failures — disconnected domain at
`warping()` under the default `disconnected="raise"`, non-bracketing
bisection), `SectionAccuracyWarning` (linear elements in
`warping()`/`stress()`).

### Flat-face parametric builders

`g.sections` gains flat-face siblings of the existing solid recipes,
same shape parameters minus the extrusion arguments
(`length`/`anchor`/`align`), returning the same `Instance` handle with
an auto-PG on the surface:

```python
g.sections.W_face(bf, tf, h, tw, *, label="W_face", lc=1e22,
                  translate=(0.0, 0.0), rotate=None)      # -> Instance
g.sections.rect_face(b, h, *, ...)
g.sections.rect_hollow_face(b, h, t, *, ...)
g.sections.pipe_face(r, *, ...)
g.sections.pipe_hollow_face(r, t, *, ...)
g.sections.angle_face(b, h, t, *, ...)
g.sections.channel_face(bf, tf, h, tw, *, ...)
g.sections.tee_face(bf, tf, h, tw, *, ...)
```

`translate` is in-plane `(dx, dy)` and `rotate` an in-plane angle in
degrees — enough to compose multi-region sections (steel `W_face`
inside a concrete `rect_face`, then `fragment`). These close the loop
for the common case (catalog-ish shape, non-catalog dimensions)
without leaving the session. The analyzer itself never requires them —
any meshed face works.

### Worked examples

**Homogeneous — welded plate girder** (N-mm; unprefixed accessors
valid; bridge reference moduli defaulted from the single material):

```python
gs = apeGmsh("girder_section")
gs.sections.W_face(bf=400.0, tf=25.0, h=1200.0, tw=12.0, label="girder")
gs.mesh.sizing.set_global_size(15.0)
gs.mesh.generation.generate(dim=2)
gs.mesh.generation.set_order(2)              # tri6 — warping-grade
sec_fem = gs.mesh.queries.get_fem_data(dim=2)

steel = SectionMaterial(E=200e3, nu=0.3, fy=345.0, density=7.85e-9)
sec = SectionProperties(sec_fem, materials={"girder": steel}, name="PG1200x400")

geo, warp, plas = sec.geometric(), sec.warping(), sec.plastic()
geo.area, geo.Ixx_c, geo.Zxx_plus            # homogeneous → unprefixed OK
warp.J, (warp.x_sc, warp.y_sc), warp.alpha_y
plas.Sxx, plas.sf_xx_plus
sec.stress(N=-800e3, Vy=350e3, Mxx=1.9e9).plot("von_mises")

p = apeSees(frame_fem)                       # separate frame model
transf = p.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
girder = p.section.ComputedSection(analysis=sec)   # E/G default from steel
integ  = p.beamIntegration.Lobatto(section=girder, n_ip=5)
p.element.forceBeamColumn(pg="girders", transf=transf, integration=integ)
p.tcl("frame.tcl")
# -> section Elastic $tag 200000.0 $A $Iz $Iy 76923.1 $J $alphaY $alphaZ
```

**Composite — steel W encased in concrete (SRC column)** (rigidity
form or explicit `transformed(e_ref)`; explicit reference moduli
required at the bridge; `plastic()` fails loud — concrete has no `fy`):

```python
gc = apeGmsh("src_column_section")
gc.sections.rect_face(b=600.0, h=600.0, label="concrete")
gc.sections.W_face(bf=250.0, tf=17.0, h=250.0, tw=10.0, label="steel")
gc.parts.fragment_pair("concrete", "steel", dim=2)  # conformal; PGs follow labels
gc.mesh.sizing.set_global_size(20.0)
gc.mesh.generation.generate(dim=2)
gc.mesh.generation.set_order(2)
sec_fem = gc.mesh.queries.get_fem_data(dim=2)

sec = SectionProperties(
    sec_fem,
    materials={
        "concrete": SectionMaterial(E=25e3, nu=0.2),
        "steel":    SectionMaterial(E=200e3, nu=0.3, fy=345.0),
    },
    name="SRC600",
)
geo, warp = sec.geometric(), sec.warping()
geo.EA, geo.EIxx_c, warp.GJ          # rigidity form — always valid
geo.Ixx_c                            # CompositeSectionError -> transformed(e_ref=...)
geo.transformed(e_ref=200e3).Ixx_c   # steel-transformed section

p = apeSees(frame_fem)
col   = p.section.ComputedSection(analysis=sec, E=200e3, G=76.9e3)  # explicit, required
integ = p.beamIntegration.Lobatto(section=col, n_ip=5)
p.element.forceBeamColumn(pg="columns", transf=transf_col, integration=integ)
# omitting E= raises at emit naming "SRC600" — never a silent reference pick
```

**Disconnected — twin girders + deck strip (equivalent spine beam)**
(parts intentionally not touching; policy declared at construction):

```python
gt = apeGmsh("twin_girder_section")
gt.sections.W_face(bf=400., tf=25., h=1400., tw=14., label="girder_L",
                   translate=(-1750.0, 0.0))
gt.sections.W_face(bf=400., tf=25., h=1400., tw=14., label="girder_R",
                   translate=(+1750.0, 0.0))
gt.sections.rect_face(b=6000., h=250., label="deck",       # EFFECTIVE width —
                      translate=(0.0, 1525.0))             # authored, not inferred
# no fragment: the three faces never touch
...mesh, get_fem_data...

sec = SectionProperties(sec_fem, materials={...}, name="spine",
                        disconnected="sum")     # explicit intent, or warping() raises

warp = sec.warping()      # 3 per-part Saint-Venant solves
warp.GJ                   # ΣGJᵢ — equal twist rate, no inter-part shear transfer
warp.parts                # the per-component WarpingProperties
sec.geometric().EIxx_c    # connectivity-blind either way (plane sections)
```

### Validation contract

- **Analytic oracles**: rectangle (exact A/I/J-series/S), circle
  (exact everything incl. `J = πr⁴/2`, `As = 0.9·A`), thin-walled
  channel shear centre, circular tube.
- **Package oracle**: the PyPI `sectionproperties` package as a
  **dev-only** dependency; CI compares an I-section, a composite
  concrete-steel section, and a holed box on matched meshes to
  tight tolerances (`J`, shear centre, `As`, plastic moduli within
  mesh-convergence distance). Skip-if-not-installed, same pattern as
  the openseespy live oracles.
- Mesh-convergence test on `J` for tri3 → tri6 documents the
  accuracy-warning rationale.

## Alternatives considered

1. **Wrap the PyPI `sectionproperties` package.** Least work,
   battle-tested numbers. Rejected: it re-meshes internally via
   `shapely`+`triangle`, so gmsh geometry/mesh/PG machinery would be
   serialized out and thrown away; Tri6-only; adds a runtime dependency
   stack (shapely) we don't otherwise carry; and its API/mesh churn
   would sit permanently under ours. Kept as the *test oracle*, where
   its maturity is pure upside.
2. **Hybrid — native geometric, wrapped warping.** Rejected: splits
   one result object across two engines with two meshes; the warping
   solve is the part where owning the mesh matters most (quads,
   structured meshes, refinement studies).
3. **Solve via an OpenSees round-trip.** Rejected: a scalar Poisson
   problem does not warrant deck emission, process spawn, and recorder
   parsing; it would also couple pre-processing to a solver install.
4. **Extend `g.sections` with a `.properties()` method.** Rejected:
   muddles a session-bound geometry builder with a mesh-consuming
   analyzer; the analyzer's lifecycle (post-mesh, session-independent)
   matches the `Results`/`apeSees` broker pattern, not a composite verb.
5. **Hang it off the bridge (`ops.section.from_mesh`).** Rejected:
   pure-geometry computation behind the OpenSees bridge is the wrong
   layer; the bridge is a *consumer* (`to_elastic_section`).

## Consequences

**Positive:**

- Arbitrary cross-sections (holes, composites, imported DXF/STEP faces)
  get `A/I/J/As/S/Z` without leaving apeGmsh; the numbers flow straight
  into `ElasticSection` via one documented axis mapping.
- The declaration/resolution split means a frame model references the
  *section*, not hand-copied numbers: edit the face geometry and re-run
  the script, and the emitted deck follows — no transcription step to
  get stale. The `kind=` axis leaves a landing for fiber lowering
  without a new surface.
- `apeGmsh.fem` gains its third consumer, validating the shared-kernel
  bet; quad and structured meshes work where the reference package is
  Tri6-only.
- The oracle strategy gives us the reference package's maturity in CI
  without carrying it at runtime.
- Natural future consumers: the planned `recipes/section_recipes.py`
  (ADR 0004), apeSteel interop (plastic moduli), `Fiber` layout
  validation (compare fiber-sum A/I against the meshed truth).

**Negative / risks:**

- First in-process solver — a new competence the team must maintain
  (singular-system regularization, shear-integral subtleties). Bounded
  by the small, classical problem and the dual oracle net.
- Two "sections" concepts in one package (builder vs analyzer) —
  mitigated by co-location and naming (`g.sections` builds, 
  `SectionProperties` analyzes).
- A new Qt surface (the inspector) to maintain — bounded by staying
  outside the viewer-family contracts (no SceneLayer/H5/dispatcher
  obligations), by matplotlib doing all drawing, and by every
  capability having a headless equivalent so the panel is never the
  only path.
- Plastic analysis is meaningless for softening materials; documented,
  not guarded — same stance as the reference package.
- Modulus-weighted composite results require the user to understand
  the transformed-section convention on `to_elastic_section`; the
  explicit required `E` makes the choice loud.

## Slices

| # | Deliverable | Verify |
|---|---|---|
| S1 | `SectionMaterial` + `SectionProperties` skeleton + geometric analysis (incl. composite weighting, geometric-only mode, input gates) + `summary()`/`_repr_html_` | analytic oracles; PG-coverage fail-loud tests |
| S2 | Warping solve (`J`, shear centre, `Γ`, `As`) + `disconnected="raise"/"sum"` policy + accuracy warning | analytic + package oracle; tri3→tri6 convergence; two-rectangle sum test (`GJ = ΣGJᵢ` exactly, per-part shear centres); unfragmented-touching-faces fail-loud test; `G`-override bound test (two chords + connecting strip: `G→0` recovers the sum, rigid `G` recovers the connected solve) |
| S3 | Plastic analysis | analytic (rect/circle Z) + package oracle |
| S4 | Stress recovery + `SectionStress.plot()` | closed-form σ checks (pure N, pure M); package oracle fields |
| S5 | Bridge binding: `ComputedSection` (emit-time lowering) + eager `to_elastic_section()` + flat-face builders in `g.sections` | round-trip test: `W_face` → analyzer → `ComputedSection` → emitted deck line vs AISC W-shape table values; deck byte-equality vs hand-typed `ElasticSection`; fail-loud test on composite without reference `E` |
| S6 | Section inspector: `sec.viewer()` Qt panel (glyph canvas, tabbed property tables, live load inputs blending unit stress fields) + `plot_section()` glyph overlay | offscreen-guard + import-guard tests; unit-field blend == direct `stress()` arrays; QTimer-driven screenshot smoke test (same pattern as the viewer gallery) |

## Reference

- ADR 0004 (sections outside `material/`), ADR 0009 (no back-compat
  solvers), ADR 0014 (viewer is an H5 consumer), ADR 0001 (decouple
  from the gmsh session).
- Pilkey, *Analysis and Design of Elastic Beams* (2002) — the warping /
  shear-area formulation, same basis as the reference package.
- `sectionproperties` (PyPI) — https://sectionproperties.readthedocs.io
  — reference capability + CI oracle.
- Kernel: `src/apeGmsh/fem/_shape_functions.py`,
  `src/apeGmsh/fem/_quadrature.py`; input broker:
  `src/apeGmsh/mesh/FEMData.py`; consumer:
  `src/apeGmsh/opensees/section/beam.py`.
