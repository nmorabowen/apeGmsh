# Sections

This page explains what a *cross-section* is in apeGmsh — one word the
library uses in three related places — so you always know which layer you're
working in: sections you **mesh** (`g.sections` builds the shape as labeled
geometry), sections you **declare** to the solver (`ops.section` primitives
on the bridge, elastic or fiber), and sections you **compute** (the
`SectionProperties` analyzer, which turns a drawn face into the numbers a
frame model needs).

The layer is decided by one question: *does the mesh carry the cross-section,
or does an element property?* In a continuum model — a wide-flange column
meshed with bricks — the section is literally the geometry, and the mesh
resolves stress across the flanges and web. In a frame model the member is a
line, the mesh knows nothing about its shape, and the cross-section is a
constitutive statement you hand to the element. And when the constants for
that statement should come from drawn geometry rather than a handbook table,
the third layer computes them. The three meet at the bridge, but they enter
the workflow at different moments — geometry before meshing, primitives after
the snapshot — which is why they live in different namespaces.

## Sections you mesh: `g.sections`

`g.sections` is a catalog of parametric structural shapes built as real
geometry, directly in the session — no Part, no file, no import step. Ask for
a wide-flange column and you get a 3D solid, extruded along Z:

```python
col = g.sections.W_solid(
    bf=150, tf=20, h=300, tw=10, length=2000,
    label="col",
)
```

The shape itself is nothing you couldn't build from boxes and boolean cuts.
What the builder adds is *structural naming*: the solid arrives pre-sliced at
the flange–web boundaries, and every sub-region carries a label derived from
yours — `col.top_flange`, `col.bottom_flange`, `col.web`, plus `col.start_face`
and `col.end_face` for the two ends. (The returned handle spells them for you:
`col.labels.web` is the string `"col.web"`.) This is the naming doctrine from
[the mental model](mental-model.md) applied to structural shapes — the
sub-regions an engineer actually targets exist as names from the moment the
geometry does:

```python
g.physical.add_surface("col.start_face", name="Base")   # fix the bottom
with g.loads.case("dead"):
    g.loads.gravity("col.web", density=7850)            # target sub-regions
```

The catalog covers the common prismatic shapes — `W_solid`, `rect_solid`,
`rect_hollow` (HSS), `pipe_solid`, `pipe_hollow`, `angle_solid`,
`channel_solid`, `tee_solid` — each with the label set its shape calls for
(an angle names its `horizontal_leg` and `vertical_leg`; simple shapes get a
single `body`). `W_shell` is the mid-surface sibling: same parameters, but
flanges and web arrive as surfaces at their mid-planes, ready for shell
meshing at `dim=2` instead of solids at `dim=3`.

Two knobs matter in practice. `translate` and `rotate` place the section
(built at the origin, using the same axis–angle convention as
`g.model.transforms.rotate`), and `lc` sets a local target element size on
the section's points — the default imposes no constraint, so the global size
governs unless you say otherwise. After placing several members you
`g.parts.fragment_all()` for a conformal mesh at the joints, exactly as with
any other geometry.

If that sounds close to a Part, it is — both produce labeled geometry — and
the boundary is worth stating. A [Part](parts-and-assembly.md) is a reusable
*template* with its own session, persisted to a STEP file you can hand
around: reach for it when one column design repeats twenty times. A section
is a *one-off parametric member* that exists only in the assembly session:
reach for it when this girder, with these plate dimensions, appears once.
Parts survive session restarts; sections are rebuilt by the script that
declared them — which, for parametric work, is precisely what you want.

## Sections you declare: `ops.section`

In a frame model the cross-section crosses to the other side of the snapshot.
The mesh is a line of beam elements; the section is no longer geometry but a
typed primitive you build on the [OpenSees bridge](opensees-bridge.md), under
`ops.section.<Type>(...)`. Like every bridge primitive, it registers itself
when constructed — tags resolve automatically, and you pass the returned
handle wherever a section is consumed (a `beamIntegration`, a
`zeroLengthSection`, an element's `section=` field).

The fundamental choice is **elastic versus fiber**, and it is a modeling
decision, not an API detail. An elastic section is the cross-section reduced
to a handful of constants:

```python
sec = ops.section.Elastic(E=200e3, A=8.5e3, Iz=2.0e8)   # 2-D form
```

Give it `A`, `E`, `Iz` (plus `Iy`, `G`, `J` for the 3-D form) and it bends
forever along a straight line in the moment–curvature plane — the right
answer whenever the member is meant to stay linear. A fiber section makes the
opposite bet: instead of handing the element an `Iz`, you tile the
cross-section with small material cells, each carrying its own uniaxial
stress–strain law, and the section *derives* its response — elastic at first,
then yielding from the extreme fiber inward, then a plastic plateau:

```python
steel = ops.uniaxialMaterial.Steel02(fy=Fy, E=E, b=0.005)

section = ops.section.Fiber(patches=(
    RectPatch(material=steel, ny=4,  nz=1,                       # top flange
              yI=d/2 - tf, zI=-bf/2, yJ=d/2,        zJ=bf/2),
    RectPatch(material=steel, ny=24, nz=1,                       # web
              yI=-(d/2 - tf), zI=-tw/2, yJ=d/2 - tf, zJ=tw/2),
    RectPatch(material=steel, ny=4,  nz=1,                       # bottom flange
              yI=-d/2, zI=-bf/2, yJ=-(d/2 - tf), zJ=bf/2),
))
```

The building blocks are `RectPatch` (fill a rectangle with `ny × nz` fibers),
`StraightLayer` (a line of fibers — rebar), and `FiberPoint` (one fiber),
laid out in the section's local (y, z) frame; `ops.section.W_fiber(...)` is a
parametric shortcut that assembles the three-patch wide-flange for you. One
rule to internalize: build the materials and the section *through the bridge*
(`ops.uniaxialMaterial...`, `ops.section.Fiber(...)`), not by constructing
dataclasses standalone, so the bridge can register them and resolve every
fiber's material tag at build time.

The catalog rounds out with the shell-side sections —
`ElasticMembranePlateSection` for a single-layer plate, `LayeredShell` and
`LayeredShellFiberSection` for stacked layers (an RC wall with cover,
core, and smeared-rebar layers) — and `Aggregator`, which composes uniaxial
laws DOF-by-DOF on top of an optional base section (a shear spring bolted
onto a flexural section). To watch a fiber section reproduce first-yield and
plastic moments against hand calculations, work through
[Fiber sections & moment–curvature](../examples/fiber-moment-curvature.md).

## Sections you compute: the analyzer

Between the drawn shape and the declared constants sits an obvious gap: where
do `A`, `Iz`, `J`, and the shear areas *come from* when the section isn't in
a catalog — built-up plates, an SRC column, an imported DXF outline? The
`SectionProperties` analyzer closes that gap in-process: mesh the
cross-section as a flat 2-D face, and it computes the full property set from
that mesh — no external tool, no re-entered dimensions.

The workflow reuses everything you already know. Every solid recipe has a
flat-face sibling (`W_face`, `rect_face`, `rect_hollow_face`, `pipe_face`,
`pipe_hollow_face`, `angle_face`, `channel_face`, `tee_face`) — though any
meshed face works; raw OCC loops and DXF or STEP imports are equal routes,
the builders are convenience, not a requirement. You mesh it like any other
model, at second order — warping and shear-area solves are FE problems in
their own right, and constant-strain triangles converge poorly on them:

```python
g.sections.W_face(bf=400.0, tf=25.0, h=1200.0, tw=12.0, label="girder")
g.mesh.sizing.set_global_size(15.0)
g.mesh.generation.generate(dim=2)
g.mesh.generation.set_order(2)          # tri6 — warping-grade
fem = g.mesh.queries.get_fem_data(dim=2)

from apeGmsh import SectionProperties
from apeGmsh.sections import SectionMaterial

sec = SectionProperties(
    fem, materials={"girder": SectionMaterial(E=200e3, nu=0.3, fy=345.0)},
)
geo, warp, plas = sec.geometric(), sec.warping(), sec.plastic()
```

The analyzer is a *declaration* — frozen inputs, memoized results — and its
three analyses split by cost: `geometric()` is pure quadrature (area,
centroid, inertias, elastic moduli), `warping()` is the FE solve (torsion
constant, shear center, shear areas), `plastic()` locates the plastic
neutral axes (plastic moduli, `Mp`, shape factors). Stress recovery and a
family of matplotlib plots — von Mises contours, shear-flow quivers, a Qt
inspector via `sec.viewer()` — hang off the same object.

Materials make it composite-aware: `materials=` maps physical-group names to
`SectionMaterial`s, so an SRC column is just a face partitioned into a
`"concrete"` PG and a `"steel"` PG. Composites impose one naming law worth
knowing at concept level — rigidity-form fields (`EA`, `EIxx_c`, `GJ`) are
always valid, while unprefixed accessors (`Ixx_c`, `J`) divide by *the*
modulus and therefore raise `CompositeSectionError` on a composite, where no
single modulus exists; you pick a reference explicitly with
`transformed(e_ref=...)`. Omit `materials=` entirely for geometric-only mode,
the classic unit-modulus numbers.

The last step is the reason the analyzer lives in apeGmsh at all. Instead of
copying numbers into an `ops.section.Elastic(...)` call, you bind the
*declaration* to the bridge:

```python
girder = ops.section.ComputedSection(analysis=sec)   # lazy; resolves at emit
integ  = ops.beamIntegration.Lobatto(section=girder, n_ip=5)
ops.element.forceBeamColumn(pg="girders", transf=transf, integration=integ)
```

`ComputedSection` holds a reference to the analyzer and resolves it at emit
time into a plain `section Elastic` line — byte-identical to one you'd type
by hand, so it slots into every consumer unchanged. The emitted deck
therefore always follows the drawn geometry: edit the face, re-run, and the
frame model updates with it. The lowering owns the axis mapping (authoring
x becomes the element's local z, authoring y its local y — so `Ixx_c → Iz`,
`Iyy_c → Iy`, and the shear-area ratios become `alphaY`/`alphaZ`), analyses
are memoized so many references to one analyzer cost one solve, and a
composite without explicit reference `E=`/`G=` fails loud at emit rather than
guessing a modulus. The full recipes — the SRC composite, the disconnected
multi-part policy, the gotchas — are in
[Compute section properties](../how-to/section-properties.md).

---

*Next: [Constraints](constraints.md).*
