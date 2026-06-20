# apeGmsh reinforcement cages (`g.rebar`)

A guide to authoring reinforcing-bar cages for RC solid models — standard
column/beam/circular members with code-aware ACI detailing, plus
hand-authored bars and stirrups. This document covers the `g.rebar`
abstraction (ADR 0067); see `guide_constraints.md` for the embedded /
conformal coupling it delegates to, and `guide_fem_broker.md` for how the
emitted curves land in the broker.

All snippets assume an open session with a host solid:

```python
from apeGmsh import apeGmsh
from apeGmsh._kernel.defs.rebar import BarLayout, TieLayout, Hook
from apeGmsh.rebar.detailing import ACI318, ACI318_seismic, BarCatalog

g = apeGmsh(model_name="demo")
g.begin()
vol = g.model.geometry.add_box(0, 0, 0, 0.6, 0.6, 3.0, label="Col")  # the concrete
```


## Tasks on this page

- [Pick a detailing standard](#1-detailing-standards) · [Build a rectangular column](#2-the-column-generator) · [Beams](#3-the-beam-generator) · [Circular columns](#4-circular-columns) · [Walls](#4a-walls) · [Place + couple the cage](#5-placing-the-cage) · [Author bars/stirrups by hand](#6-hand-authoring) · [Bundled bars](#6a-bundled-bars-aci-318-19-256) · [What is and isn't generated](#7-limits)


## 1. Detailing standards

A `DetailingStandard` resolves the *code-aware* numbers a cage needs —
bar diameter/area from a designation (`"#8"`, `"20mm"`), minimum bend
diameters, hook-tail extensions, and the ACI confinement rules — and
resolves `"<k>db"` length tokens. It is bound to the session (or a cage)
and applied at `place` time, so the bar specs stay unitless data.

```python
g.rebar.use_standard(ACI318())            # ACI 318-19 development hooks
g.rebar.use_standard(ACI318_seismic())    # + seismic 135° hooks + §18.6/18.7 confinement
```

**Units.** apeGmsh is unit-agnostic; the single unit knob is the
`BarCatalog`. For a **metres** model, scale the catalogue so the absolute
ACI floors (18 in `l_o`, 3 in seismic hook tail, the `s_o` equation) land
correctly:

```python
g.rebar.use_standard(ACI318_seismic(BarCatalog(unit_length=0.0254)))  # 1 in = 0.0254 m
```

`Raw()` is the escape hatch — it resolves catalogue diameters/areas but
raises on every code-derived bend/hook, so you must give every length
yourself.


## 2. The column generator

`g.rebar.column(...)` builds a rectangular column cage: perimeter
longitudinal bars + tie rings + lateral support for the intermediate bars.

```python
cage = g.rebar.column(
    section=("rect", 0.6, 0.6), height=3.0, cover=0.05,
    longitudinal=BarLayout(n_x=3, n_y=3, db=0.025),   # 3×3 perimeter bars
    ties=TieLayout(db=0.01, spacing=0.3))             # hoop bar + spacing
```

- **`longitudinal=BarLayout(n_x, n_y, db, material)`** — bars per face
  (corners shared); `n_x, n_y ≥ 2`.
- **`ties=TieLayout(db, spacing, material, hinge_spacing=, hinge_length=)`**
  — the hoop. `spacing` is the regular tie pitch; the optional `hinge_*`
  fields densify the ends.
- Bars/ties are inset interior (`cover + tie + db/2`), so the cage meshes
  under conformal coupling without a boundary-facet error.

**Cross-ties (ACI 318 §25.7.2.3).** When a face has interior bars
(`n>2`), every intermediate bar gets a cross-tie by default
(`crossties=True`): a transverse leg at every tie level with a 135°
seismic hook + a 90° hook, alternated end-for-end (§18.7.5.2). For wide
sections use `confinement_style="overlapping_hoops"` instead — the core
is tiled with closed overlapping cell-hoops (every bar at a hoop corner).

```python
cage = g.rebar.column(
    section=("rect", 0.8, 0.8), height=3.0, cover=0.05,
    longitudinal=BarLayout(n_x=4, n_y=4, db=0.025),
    ties=TieLayout(db=0.012, spacing=0.3),
    confinement_style="overlapping_hoops")   # vs the default "crossties"
```

**Seismic confinement zone (ACI 318 §18.7.5).** With an `ACI318_seismic`
standard and no explicit `hinge_*`, the confined-end length
`l_o = max(depth, ln/6, 18 in)` and dense spacing
`s_o = min(b/4, 6·d_b, 4+(14−h_x)/3 in ∈ [4,6] in)` are auto-derived from
the geometry (a warning reports the values). Pass `TieLayout(hinge_length=,
hinge_spacing=)` to override, or use a non-seismic standard for uniform
spacing.


## 3. The beam generator

`g.rebar.beam(...)` builds a rectangular beam cage: top + bottom
longitudinal bars + vertical stirrups.

```python
cage = g.rebar.beam(
    section=("rect", 0.4, 0.6), length=5.0, cover=0.04,
    top=BarLayout(n_x=4, db=0.02), bottom=BarLayout(n_x=4, db=0.02),
    stirrups=TieLayout(db=0.01, spacing=0.2))
```

- `top`/`bottom` bar counts use `BarLayout.n_x` (`n_y` is ignored).
- Intermediate bars (`n>2`) get supplementary legs: each interior bar is
  tied to the nearest bar on the opposite face (vertical when the counts
  align, slightly inclined when they differ — every interior bar is
  supported either way; a count mismatch is warned).
- For wide beams, `confinement_style="overlapping_hoops"` tiles the
  cross-section with closed overlapping cell-stirrups (every bar at a hoop
  corner) instead of straight legs — needs equal top/bottom counts.
- **Seismic hoop zone (ACI 318 §18.6.4).** With `ACI318_seismic` and no
  explicit `hinge_*`, the hoop zone length `2h` and spacing
  `min(d/4, 6·d_b, 6 in)` are auto-derived.


## 4. Circular columns

`g.rebar.circular_column(...)` builds a round column: `n_bars` evenly
spaced on a circle + circular confinement (discrete hoops or a spiral).

```python
# circular hoops (one closed ring per tie level)
cage = g.rebar.circular_column(
    diameter=0.6, height=3.0, cover=0.05, n_bars=8, bar_db=0.025,
    ties=TieLayout(db=0.01, spacing=0.3))

# continuous spiral at pitch = ties.spacing
cage = g.rebar.circular_column(
    diameter=0.6, height=3.0, cover=0.05, n_bars=8, bar_db=0.025,
    ties=TieLayout(db=0.01, spacing=0.075), spiral=True)
```

- Bars sit on radius `D/2 − cover − tie − db/2`; the hoop centerline on
  `D/2 − cover − tie/2`.
- `spiral=True` emits a single `role="spiral"` truss helix; `spiral=False`
  emits one closed ring per (hinge-densified) tie level.
- Rings/helix are polygon-approximated with `n_segments` sides per turn
  (default 24). Circular confinement supports every bar — no cross-ties.
- The §18.7.5 seismic confinement auto-derives (`h_x` = the bar chord
  spacing on the circle).


## 4a. Walls

`g.rebar.wall(...)` builds a wall panel (plane x-z, thickness along y):
vertical bars (along the height, spaced along the length) + horizontal bars
(along the length, spaced up the height), in one or two curtains.

```python
cage = g.rebar.wall(
    length=4.0, thickness=0.25, height=3.0, cover=0.04,
    vertical_db=0.016, vertical_spacing=0.30,       # spacing, not count
    horizontal_db=0.012, horizontal_spacing=0.30,
    curtains=2, crosstie_spacing=0.60)              # ties the two curtains
```

- **Spaced, not counted** — `vertical_spacing` / `horizontal_spacing` are
  max centre-to-centre pitches, rounded to an even division between the
  `end_cover` insets.
- `curtains=2` (default) places a layer `cover + db/2` in from each face;
  `curtains=1` a single layer at mid-thickness. Vertical and horizontal bars
  of a curtain are idealised co-planar (a truss model).
- For a double curtain, `crossties=True` (default) ties the curtains through
  the thickness on a grid (`crosstie_spacing`, default twice the coarser bar
  spacing) with a 135°/90° hooked leg (ACI 318 §11.7.4 / §18.10.2.7). A
  single-curtain wall warns and emits none.
- Bars carry `role="vertical"` / `"horizontal"` / `"crosstie"`.
- **Boundary elements** are out of scope — model a confined wall end with
  `column()` over the boundary zone and place both cages.


## 5. Placing the cage

`g.rebar.place(cage, into, ...)` emits the cage geometry and couples each
member to the host solid `into`. Call it **before** meshing.

```python
g.physical.add_volume([vol], name="Col")        # embedded needs a PG host
g.rebar.place(cage, into="Col", coupling="embedded", perfect=1.0e8)
g.mesh.generation.generate(dim=3)
```

- **`coupling="conformal"`** (default) embeds the bar curves into the host
  so the mesh conforms (shared nodes → perfect bond). Needs a single host
  volume, run before meshing. Cross-ties / overlapping hoops form bar/tie
  T-junctions that need `g.mesh.editing.make_conformal`.
- **`coupling="embedded"`** meshes the bars independently and forwards to
  `g.reinforce` (→ `LadrunoEmbeddedRebar`); pass `bond=<LadrunoBondSlip
  name>` *or* `perfect=<axial penalty>`. Single-process only. This is the
  robust path for full cages.
- **`per_member_coupling={role: coupling}`** mixes per role (e.g.
  longitudinal conformal + ties embedded). Roles: `"longitudinal"`,
  `"tie"`, `"crosstie"`, `"top"`, `"bottom"`, `"spiral"`, `"vertical"`,
  `"horizontal"`.
- **`twin_tail=True`** (default) emits the real two-tail hoop seam (both
  ends hooked, overlapping at the seam corner); `twin_tail=False` for the
  simplified single hook.


## 6. Hand authoring

For non-standard layouts, author bars and stirrups directly and assemble a
`Cage`. The fluent `BarBuilder` chains into a bar; `stirrup_rect` builds a
rectangular tie.

```python
from apeGmsh._kernel.defs.rebar import Cage

bar = (g.rebar.bar(db=0.025, material="rebar")
       .through([(0.1, 0.1, 0.0), (0.1, 0.1, 3.0)])
       .hook_end(Hook.standard_90())
       .as_("corner_NE"))
tie = g.rebar.stirrup_rect(0.5, 0.5, 0.04, db=0.01, material="rebar", z=1.0)

g.rebar.place(Cage(bars=(bar,), stirrups=(tie,)), into="Col",
              coupling="embedded", perfect=1.0e8)
```

Hooks default their tail/bend radius to the bound standard
(`Hook.standard_90()`, `Hook.seismic_135()`, …); pass explicit numbers or
`"<k>db"` tokens to override. A hook with no standard and no numeric tail
is dropped with a warning (longitudinal development hooks raise instead).


## 6a. Bundled bars (ACI 318-19 §25.6)

Pack 2–4 bars in contact at each position. On a generator, add `bundle=` to
the `BarLayout` (or to `circular_column` directly):

```python
cage = g.rebar.column(
    section=("rect", 0.6, 0.6), height=3.0, cover=0.05,
    longitudinal=BarLayout(n_x=2, n_y=2, db="#11", bundle=3),  # 3-bar corner bundles
    ties=TieLayout(db="#4", spacing=0.30))
```

- `bundle_pattern="auto"` (default) → `line` (2, side-by-side), `triangle`
  (3), `square` (4, 2×2); an explicit pattern must match the count.
- The outer bars sit on the nominal cover line and the cluster **stacks
  inward**; cross-ties and hoops still engage the outer bar at the nominal
  position. A bundle is realised as that many individual bar members, so
  coupling and detailing are unchanged.
- Limits: 1–4 bars (`#14`/`#18` capped at 2 per ACI §25.6.1.1); the generator
  fails loud if the inward stack would cross the section centre. At a true
  **corner** a tangentially-spread pair leans toward a face by ≤ √2⁄2·d_b —
  for strict corner cover, inset the layout for the equivalent diameter
  `√n·d_b`.

For free-form bundles, `g.rebar.bundle(...)` returns a tuple of `Bar`:

```python
bars = g.rebar.bundle(
    [(0.1, 0.1, 0.0), (0.1, 0.1, 3.0)], n=2, db="#10", material="rebar",
    toward=(0.3, 0.3, 0.0))          # the cluster leans toward this interior point
g.rebar.place(Cage(bars=bars), into="Col", coupling="embedded", perfect=1e8)
```

`spacing=` overrides the centre-to-centre offset (default = the bar diameter,
a contact bundle); a curved polyline is offset rigidly by its chord frame
(exact for straight bars).


## 7. Limits

- A beam with mismatched top/bottom counts ties each interior bar to its
  nearest opposite-face bar (legs may be inclined; warned).
- Circular hoops/spirals are polygon-approximated (not true NURBS circles).
- Composed-`Part` rebar libraries are not yet persisted through `model.h5`
  (the embedded-tie record is H5-dropped today); author cages in the same
  session as the host. `element="beam"` (dowel-action) rebar on a
  curved/hooked bar is gated on the ADR-0010 orientation fan-out.
