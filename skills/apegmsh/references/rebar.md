# `g.rebar` — reinforcement-cage authoring (ADR 0066/0067)
<!-- skill-freshness: verified against apeGmsh main@8eeda7a3 (2026-07-06) · if weeks old, re-verify signatures in src/apeGmsh/ before trusting exact tags/signatures -->

`g.rebar` is a session composite (`RebarComposite`, registered at
`src/apeGmsh/_core.py:53`) that authors **RC reinforcement cages** —
longitudinal bars + ties/hoops + ACI-318 detailing — as solver-agnostic
geometry, then couples them to a host continuum. It sits **above** the
lower-level `g.reinforce` embedded-rebar generator: `g.rebar` is the
"draw a code-compliant cage" layer; it forwards to `g.reinforce` /
`LadrunoEmbeddedRebar` when you ask for embedded coupling.

```python
from apeGmsh.rebar import (              # L1 specs + detailing standards
    Bar, BarBuilder, BarLayout, Cage, Hook, Path, Stirrup, TieLayout, Vec3, METADATA,
    ACI318, ACI318_seismic, BarCatalog, Raw, DetailingStandard, DetailingError,
)
```

The realised FE elements are always **straight 2-node chords** — OpenSees
has no curved line element, so "true-arc" geometry only seeds the *mesh
nodes* on the true curve.

## Mental model — three layers

1. **L1 specs** (frozen, serialisable): `Bar` / `Stirrup` / `Cage` /
   `Hook` / `Path` / `BarLayout` / `TieLayout`. A `Cage` is just a tuple
   of `Bar`s + a tuple of `Stirrup`s (+ an optional detailing standard,
   not serialised).
2. **L2 composite** (`g.rebar`): generators that *produce* cages
   (`column` / `beam` / `circular_column` / `wall`) and hand-authoring
   primitives (`bar` / `stirrup` / `bundle`), plus `place(...)` which
   emits geometry and wires coupling.
3. **L3 fluent builder** (`BarBuilder`): `g.rebar.bar()` with no points
   returns a chainable builder (`.through(...).hook_end(...).as_(name)`).

## Detailing standard — set it first

```python
g.rebar.use_standard(standard) -> None       # src/apeGmsh/core/RebarComposite.py:99
```

Binds the default `DetailingStandard` for the session; resolves `"<k>db"`
length tokens (`"12db"`, `"6db"`) and hook factories at bind time.

- **`ACI318(catalog)`** — ACI 318-19 Table 25.3 min bend diameters +
  standard-hook tails.
- **`ACI318_seismic(catalog)`** — adds the §18.7.5 column-confinement and
  §18.6.4 beam-hoop zone auto-derivation, 3-in 135° hook-tail floor.
- **`Raw(catalog)`** — explicit-only; **raises** `DetailingError` on every
  code-derived number ("I'll give every dimension myself").
- **`BarCatalog(unit_length=1.0, base="imperial"|"metric")`** — resolves
  bar designations (`"#3"`–`"#18"` imperial ASTM A615, `"20mm"` metric, or
  a raw float in model units) to diameter / area. `unit_length` = model
  units per canonical unit (e.g. `0.0254` if you model in metres with #-bars).

Without a standard set, resolving a *designation* (e.g. `"#8"`) raises —
pass a numeric `db` in model units or call `use_standard(...)` first.

## Standardised member generators → a `Cage`

All keyword-only; all return a `Cage`. (`src/apeGmsh/core/RebarComposite.py`
:147 `column`, :269 `beam`, :386 `circular_column`, :503 `wall`.)

```python
g.rebar.column(*, section, height, cover, longitudinal: BarLayout, ties: TieLayout,
    base_z=0.0, origin=(0.0, 0.0), standard=None, top_hook=None, bottom_hook=None,
    end_cover=None, crossties=True, confinement_style="crossties") -> Cage
#   section=("rect", bx, by); perimeter bars + tie rings + §25.7.2.3 cross-ties
#   + §18.7.5 seismic confinement zone (with ACI318_seismic). confinement_style:
#   "crossties" (straight legs) | "overlapping_hoops" (cell-hoops tiling the core).

g.rebar.beam(*, section, length, cover, top: BarLayout, bottom: BarLayout,
    stirrups: TieLayout, base_x=0.0, origin=(0.0, 0.0), standard=None, end_cover=None,
    crossties=True, confinement_style="crossties") -> Cage
#   section=("rect", w, h); top/bottom use BarLayout.n_x only; stirrups along the
#   span; §18.6.4 seismic hoop zone auto-derived.

g.rebar.circular_column(*, diameter, height, cover, n_bars, bar_db, bar_material="rebar",
    ties: TieLayout, base_z=0.0, origin=(0.0, 0.0), standard=None, top_hook=None,
    bottom_hook=None, end_cover=None, spiral=False, n_segments=24, bundle=1,
    bundle_pattern="auto", true_arc=False) -> Cage
#   n_bars >= 3 on a circle. spiral=True → one continuous helix at pitch ties.spacing;
#   spiral=False → discrete circular hoops. n_segments = polygon sides/turn (polyline
#   only). true_arc=True → mesh-native arcs/spline (nodes land on the true curve).

g.rebar.wall(*, length, thickness, height, cover, vertical_db, vertical_spacing,
    horizontal_db, horizontal_spacing, curtains=2, material="rebar",
    vertical_material=None, horizontal_material=None, crossties=True, crosstie_db=None,
    crosstie_spacing=None, base_z=0.0, origin=(0.0, 0.0), standard=None,
    end_cover=None) -> Cage
#   vertical panel (plane x-z, thickness along y); 1 or 2 curtains + through-thickness
#   cross-ties (double-curtain only). Bars evenly spaced ≤ the given pitch.
```

`BarLayout(n_x, n_y, db, material, bundle=1, bundle_pattern="auto")` and
`TieLayout(db, spacing, material, hinge_spacing=None, hinge_length=None,
db_value=None, hook=None)` are the layout records. `hinge_spacing` +
`hinge_length` (both or neither) override the auto-derived seismic dense zone.

## Hand-authoring primitives

```python
g.rebar.bar(points=None, *, db, material, role="longitudinal", element="truss",
    start_hook=None, end_hook=None, corner_radius=METADATA, curve="polyline",
    arc_center=None, name=None) -> Bar | BarBuilder            # :105
#   points given → Bar; points None → fluent BarBuilder.
#   element: "truss" (CorotTruss) | "beam" (dowel; gated, curved bars NotImplemented).
#   curve: "polyline" (default) | "arc" (true arc about arc_center) | "spline" (C2).

g.rebar.stirrup(points, *, db, material, closure_hook=None, role="tie",
    corner_radius=METADATA, curve="polyline", arc_center=None, name=None) -> Stirrup   # :131
#   open polyline returning near origin; closure_hook defaults to Hook.seismic_135().
g.rebar.stirrup_rect(bx, by, cover, *, db, material, **kw) -> Stirrup                   # :142

g.rebar.bundle(points, *, n, db, material, toward, pattern="auto", spacing=None,
    role="longitudinal", element="truss", start_hook=None, end_hook=None,
    name=None) -> tuple[Bar, ...]                              # :841 (ACI 318-19 §25.6)
#   n in 1..4 parallel contact bars; pattern "line"/"triangle"/"square"/"auto";
#   stacks toward an interior point. #14/#18 capped at 2 per §25.6.1.1.
```

`Hook.standard_90/135/180(...)` and `Hook.seismic_135(...)` are the hook
factories; `Stirrup.rect(bx, by, cover, *, db, material, z=, plane=, ...)`
builds a rectangular tie directly.

## `place(...)` — emit geometry + couple to host

```python
g.rebar.place(cage: Cage, into: str, *, coupling="conformal",
    per_member_coupling=None, bond=None, perfect=None, kt=None, kt_alpha=None,
    enforce="penalty", bipenalty=False, dtcr=None, tolerance=1.0e-6, snap=False,
    host_dim=None, true_arc=False, on_conformal_infeasible="fail", twin_tail=True,
    emit_elements=False, name=None) -> RebarPlacement           # :1040
```

**Call before `g.mesh.generation.generate()`** for conformal coupling
(it uses gmsh `embed` to weld bar nodes into the host before meshing).

- **`coupling="conformal"`** (default) — gmsh embed → shared nodes /
  perfect bond. **MPI-safe.** Needs a single un-meshed host volume.
- **`coupling="embedded"`** — forwards to `g.reinforce` →
  `LadrunoEmbeddedRebar`. Needs a **physical-group** host and **exactly
  one** of `bond=<LadrunoBondSlip name>` or `perfect=<axial penalty>`.
  **Single-process only** (warns under partitioned emit).
- **`per_member_coupling={role: coupling}`** — mix per role, e.g.
  `{"longitudinal": "conformal", "tie": "embedded"}`.
- **`on_conformal_infeasible`** — `"fail"` (raise on embed error) or
  `"embedded"` (fall back to embedded coupling).
- **`emit_elements=True`** — auto-emit the bar's structural element at
  `get_fem_data()` time: `element="truss"` → `CorotTruss` (one per line
  cell). Coupling is emitted regardless of this flag; `emit_elements`
  only controls whether the *bar itself* becomes FE elements.
- **`twin_tail=True`** (default) — real hoop seam (both ends hooked) vs a
  single closure hook.

`kt` / `kt_alpha` / `enforce` / `bipenalty` / `dtcr` / `tolerance` / `snap`
forward to `g.reinforce` for the embedded path (see
`project_reinforce_embedded_rebar` lineage). `place()` returns a
`RebarPlacement` record.

The lower-level `g.reinforce(host, bars, *, ...)` additionally takes
`corot=False` (#769, ADR 20 §10.5): `corot=True` co-rotates the bar axis
each step from the current host geometry (`-corot`; the generator computes
the `-shapeB` point-B weights automatically) — keeps the axial/transverse
split frame-objective under **large host rotation**. Default `False` = the
frozen reference `-dir`. `place()` does **not** forward `corot`; call
`g.reinforce` directly when you need it.

## Persistence & compose

- Auto-emitted rebar structural elements round-trip through the **neutral**
  `model.h5` (`/rebar_elements`, neutral schema ≥ 2.16.0).
- Embedded-reinforcement ties (the `g.reinforce` coupling metadata)
  persist under `/reinforce_ties` (the `corot` flag rides along, neutral
  schema ≥ 2.26.0) and survive `g.compose(...)` (node-tag offset + name/bond
  prefix); a tie that would cross a Part boundary raises
  `ComposeReinforceCrossPartError`. H5 **deck-replay** re-emits them too:
  `OpenSeesModel.build()` from a `model.h5` regenerates the
  `LadrunoEmbeddedRebar` tie lines (ADR 0067 P5.1, #771).
- `Cage.to_dict()` / `from_dict()` serialise geometry + intent only
  (schema `"apeGmsh.rebar.cage"`) — **no** detailing standard, **no**
  OpenSees handles baked in.

## Fail-loud guards worth knowing

- Conformal embedding must run **before** meshing and needs **one** host
  volume; embedded coupling needs a **PG** host and is **single-process**.
- Embedded coupling: pass **exactly one** of `bond=` / `perfect=`.
- `bundle ∈ [1, 4]`; the inward stack must not cross the section centre.
- `circular_column` needs `n_bars ≥ 3`; a `Cage` must hold ≥ 1 member.
- `element="beam"` on curved/hooked bars raises `NotImplementedError`
  (orientation fan-out deferred, ADR-0010 Phase-4).
- `curve="arc"` requires `arc_center`.

## Minimal example

```python
from apeGmsh import apeGmsh
from apeGmsh.rebar import ACI318_seismic, BarCatalog, BarLayout, TieLayout

with apeGmsh(model_name="col") as g:
    g.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 3.0, label="col")
    g.physical.add_volume("col", name="Col")
    g.rebar.use_standard(ACI318_seismic(BarCatalog(unit_length=0.0254)))  # metres
    cage = g.rebar.column(
        section=("rect", 0.5, 0.5), height=3.0, cover=0.04,
        longitudinal=BarLayout(n_x=3, n_y=3, db="#8", material="rebar"),
        ties=TieLayout(db="#3", spacing=0.15, material="rebar"),
    )
    g.rebar.place(cage, into="col", coupling="conformal", emit_elements=True)
    g.mesh.recipe.unstructured(max_size=0.1)      # bars already embedded
    fem = g.mesh.queries.get_fem_data(dim=None)   # carries host + CorotTruss bars
```
