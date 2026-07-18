# Compute section properties for a custom section

Get `A / I / J / shear centre / As / Z / S` — plus stress fields — for a
cross-section that isn't in a catalog: built-up plates, an SRC column, a
holed box, an imported DXF outline. Author the flat face, mesh it at
order 2, hand the `FEMData` to `SectionProperties`, and either read the
numbers directly or bind the analyzer to the OpenSees bridge with
`ComputedSection` so the emitted deck always follows the drawn geometry.

## Recipe — homogeneous section

```python
from apeGmsh import apeGmsh, SectionProperties
from apeGmsh.sections import SectionMaterial

g = apeGmsh(model_name="girder_section")
g.begin()

# 1. Author the flat face in the gmsh (x, y) plane. Any route works:
#    *_face builders, raw OCC loops (holes supported), load_dxf/load_step.
g.sections.W_face(bf=400.0, tf=25.0, h=1200.0, tw=12.0, label="girder")
g.sections.plot_faces()                    # optional pre-mesh sanity check

# 2. Mesh — 2-D, then SECOND ORDER (tri6/quad9). Linear elements are
#    accepted but warn on warping()/stress(): tri3/quad4 converge poorly
#    on J and shear areas.
g.mesh.sizing.set_global_size(15.0)
g.mesh.generation.generate(dim=2)
g.mesh.generation.set_order(2)
fem = g.mesh.queries.get_fem_data(dim=2)

# 3. Analyze. materials= maps PG names -> SectionMaterial; the builder's
#    label IS the auto-PG name.
steel = SectionMaterial(E=200e3, nu=0.3, fy=345.0, density=7.85e-9)  # N-mm
sec = SectionProperties(fem, materials={"girder": steel}, name="PG1200x400")

geo  = sec.geometric()     # quadrature: A, centroid, I*, phi, Z moduli
warp = sec.warping()       # FE solve: J, shear centre, Gamma, As_x/As_y
plas = sec.plastic()       # needs fy on every material: S, Mp, shape factors

geo.area, geo.Ixx_c, geo.Zxx_plus          # homogeneous -> unprefixed OK
warp.J, (warp.x_sc, warp.y_sc), warp.alpha_y
plas.Sxx, plas.sf_xx_plus

# 4. Stress recovery + plots (all matplotlib, all headless-safe).
st = sec.stress(N=-800e3, Vy=350e3, Mxx=1.9e9)
st.plot("von_mises")                       # filled tricontour
st.plot_vector(action="vy")                # (tau_zx, tau_zy) quiver
st.plot_mohrs_circle(at=(0.0, 600.0))      # beam state at the top flange
sec.plot_warping()                         # Saint-Venant omega contour
sec.plot()                                 # one-call overview figure
print(sec.summary())                       # plain-text report
# sec.viewer()                             # Qt inspector (notebooks:
                                           #   sec.viewer(blocking=False))

# 5. OpenSees handoff — the frame model references the DECLARATION;
#    properties resolve at emit time into a plain `section Elastic` line.
from apeGmsh.opensees import apeSees

p = apeSees(frame_fem)                     # your separate frame model
transf = p.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
girder = p.section.ComputedSection(analysis=sec)   # E/G default from steel
integ  = p.beamIntegration.Lobatto(section=girder, n_ip=5)
p.element.forceBeamColumn(pg="girders", transf=transf, integration=integ)
p.tcl("frame.tcl")
# -> section Elastic $tag 200000.0 $A $Iz $Iy 76923.1 $J $alphaY $alphaZ

es = sec.to_elastic_section()              # eager escape hatch: a plain
                                           #   populated ElasticSection
```

## Recipe — composite (SRC column: steel W inside concrete)

Material PGs must **partition** the face — every 2-D element covered by
exactly one `materials=` PG. The trap: authoring the inner shape
*overlapping* the outer one and fragmenting maps the overlap piece to
**both** parents' PGs (double-cover → `SectionMeshError`). Carve first
with `cut(remove_tool=False)`, then fragment for a conformal mesh:

```python
g = apeGmsh(model_name="src_column_section")
g.begin()

conc  = g.sections.rect_face(b=600.0, h=600.0, label="concrete")
steel = g.sections.W_face(bf=250.0, tf=17.0, h=250.0, tw=10.0, label="steel")
g.model.boolean.cut(                       # W-shaped hole in the concrete;
    conc.entities[2], steel.entities[2],   # keep the tool so the W face
    dim=2, remove_tool=False,              # survives as its own region
)
g.parts.fragment_pair("concrete", "steel", dim=2)   # conformal shared boundary

g.mesh.sizing.set_global_size(20.0)
g.mesh.generation.generate(dim=2)
g.mesh.generation.set_order(2)
fem = g.mesh.queries.get_fem_data(dim=2)

sec = SectionProperties(
    fem,
    materials={
        "concrete": SectionMaterial(E=25e3, nu=0.2),
        "steel":    SectionMaterial(E=200e3, nu=0.3, fy=345.0),
    },
    name="SRC600",
)

geo, warp = sec.geometric(), sec.warping()
geo.EA, geo.EIxx_c, warp.GJ                # rigidity form — always valid
geo.transformed(e_ref=200e3).Ixx_c         # steel-transformed section
# geo.Ixx_c on a composite raises CompositeSectionError — there is no
#   silently-chosen reference modulus.

col = p.section.ComputedSection(analysis=sec, E=200e3, G=76.9e3)
# composite -> explicit reference E/G REQUIRED; omitting them fails loud
#   at emit naming "SRC600". The deck reproduces the analyzer's
#   rigidities exactly (EA/E, EI/E, GJ/G), whatever reference you pick.
```

## Notes / gotchas

- **Order 2 or warnings.** `warping()`/`stress()` on a tri3/quad4 mesh
  emit `SectionAccuracyWarning` — constant-strain elements converge
  poorly on `J` and shear areas. `g.mesh.generation.set_order(2)` is the
  fix, not finer linear meshing.
- **The naming law.** Rigidity-form fields (`EA`, `EIxx_c`, `GJ`,
  `GAs_y`, `Mp_xx`) are always valid. Unprefixed accessors (`Ixx_c`,
  `J`, `As_y`, `Sxx`) divide by the section's *single* modulus and raise
  `CompositeSectionError` on composites — use `transformed(e_ref=...)`.
  Reference-free ratios (`rx/ry/r11/r22`, `alpha_x/alpha_y`) are valid
  in every mode.
- **Composite authoring = cut, then fragment.** Overlap + fragment
  double-covers the inner region in both PGs and the exact-cover gate
  fires. Hand-authored multi-region faces must share **lines**, not just
  points — two rectangles drawn with their own coincident edges mesh as
  *disconnected* parts.
- **Disconnected parts are a policy, not an accident.** `warping()` on a
  disconnected mesh raises by default (usually the forgot-to-fragment
  bug). For intentionally separate parts (twin girders + deck strip)
  pass `disconnected="sum"` — the classical equal-twist-rate lower
  bound, per-part results on `warp.parts`. Effective deck width is
  authored, never inferred.
- **Axis contract.** Analyzer results live in gmsh authoring `(x, y)`
  axes (`Ixx = ∫y²dA`). The lowering maps *authoring x ≡ local z,
  authoring y ≡ local y*: `Ixx_c → Iz`, `Iyy_c → Iy`, `As_y/A → alphaY`,
  `As_x/A → alphaZ`. Making local y land on the section's authoring-y is
  the `geomTransf` author's job — pick `vecxz` accordingly.
- **`ndm=` on `ComputedSection` selects the deck form** (`3` →
  `E A Iz Iy G J alphaY alphaZ`; `2` → `E A Iz G alphaY`). Match it to
  your `ops.model(ndm=)` envelope yourself — sections emit before the
  bridge ndm is visible.
- **Catalog comparisons: expect `J` low.** The fillet-less plate
  assembly lands `A`/`Ix`/`Iy` within ~1–2 % of AISC tables, but `J`
  5–15 % below catalog (J is fillet-sensitive). That's a modeling
  difference — don't chase it with mesh refinement.
- **One analyzer, one solve.** Analyses are memoized; N
  `ComputedSection` references to the same analyzer trigger a single
  solve at emit, and the deck line is byte-identical to a hand-typed
  `ElasticSection`.

## See also

- **Concept:** [Sections guide](../internal_docs/guide_sections.md) —
  the solid/shell builders, and the analyzer section with the full
  property tables.
- **Contract:** ADR 0078 (`src/apeGmsh/opensees/architecture/decisions/`)
  — the authoritative API contract, axis conventions, and
  disconnected-section semantics.
- **Bridge:** [OpenSees bridge guide](../internal_docs/guide_opensees.md)
  — where `section` / `beamIntegration` / `element` primitives fit.
- **Tutorial:** [A simply-supported beam](../tutorials/beam-and-composites.md)
  — frame modeling that consumes section properties.

---

*Next: [Export to a Tcl or openseespy script](export-script.md).*
