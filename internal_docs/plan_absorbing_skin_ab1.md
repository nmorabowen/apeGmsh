# Plan — AB-1: absorbing-boundary skin generator

Implements slice **AB-1** of [ADR 0054](../src/apeGmsh/opensees/architecture/decisions/0054-asd-absorbing-boundary.md):
generate the one-element-thick `ASDAbsorbingBoundary3D` skin around an
axis-aligned structured box, with per-cell `btype`, as physical groups the bridge
fans out over. AB-2 (the bridge element) and AB-3 (the staged flip) ride on top of
AB-1's output and are comparatively trivial.

## The crux: the skin is an offset SHELL, not per-quad extrusion

The obvious algorithm — "find each boundary quad, extrude it outward by
`thickness`, btype = that face" — **is wrong**, and the STKO reference proves it.
That deck contains `LF`/`LK`/`RF`/`RK` (vertical-edge) and `BLF`/`BLK`/`BRF`/`BRK`
(bottom-corner) elements (ADR 0054 validation section). Per-quad extrusion would
produce only single-face panels (`L`, `R`, `F`, `K`, `B`) and leave a
one-element-wide **gap along every box edge** where the `−X`-extruded L-panel and
the `−Y`-extruded F-panel fail to meet. Waves leak through those slots and the
panels don't share outer nodes at the edge.

The correct object is a **morphological one-element dilation (offset shell)** of
the box on its 5 truncation faces (top excluded), decomposed into:

- **face panels** — `L R F K B` (cell sits outside exactly one face)
- **edge prisms** — `LF LK RF RK` (vertical edges) + `BL BR BF BK` (bottom edges)
- **corner cubes** — `BLF BLK BRF BRK` (bottom corners)

`btype` of a shell cell = **the set of truncation faces it lies outside of**, OR-
combined. This is exactly the 17-case set the element handles internally
(`ASDAbsorbingBoundary3D.cpp:1524-1619`) and exactly the distribution observed in
STKO.

## Build approach — geometry-time offset shell (shared core)

DRMBox is **not** the home for this — it serves the Domain Reduction Method
(injecting the incident field), a separate concern. The absorbing skin gets its
own facilities, and there are **two entry points over one shared core**.

> **Adversarial panel (2026-06-07) — verdict: geometry + btype VALIDATED;
> two framing fixes.** A 4-agent panel cross-checked this plan against the
> element source and the real `waveletExample` STKO deck. Confirmed: the offset
> shell IS a plain rectangular block with **uniform** one-element thickness
> (measured 5575 on all 5 faces — the earlier per-face-anisotropy worry was a
> misread), a complete product grid, and a btype tally matching the closed-form
> counts to the unit. The btype→axis mapping is proven twice (source + data):
> **L=min-X, R=max-X, F=min-Y, K=max-Y, B=min-Z, top(+Z)=free surface**; letter
> order is irrelevant (`strstr`) but opposite-face combos (`LR`/`FK`) are illegal
> and must never be emitted. Two adjustments are folded in below: **(F5)** build
> on **session geometry, not the Part/STEP vehicle**; **(sig)** `skin_thickness`
> optional w/ a match-adjacent-soil default, and `z`/`material` accept the layered
> type but reject-for-now. The waveletExample is a **pure soil column (no
> structure)** — this builder produces soil only.

**Shared core — a geometry-time axis-aligned offset-shell generator (session
geometry, NOT a Part).** Build the soil box **and** its absorbing shell as one
structured grid: add one extra 1-element segment at `X−`, `X+`, `Y−`, `Y+`, `Z−`
(never `Z+`). The full grid is `(nx+2)·(ny+2)·(nz+1)` cells; the **inner**
`nx·ny·nz` block is the intact soil box, the wrapping outermost segments are the
shell. Gmsh meshes it conformally (shared faces → shared nodes), so:

- **Correct by construction.** Edge prisms and corner cubes are just the grid
  cells at the corners of the outer layer — Gmsh produces them, shares their
  nodes, and the soil box stays fully intact (the shell is genuinely *extra*
  material outside it, satisfying the physics that killed reinterpret).
- **`btype` falls out of grid position** — no fragile post-hoc classifier, no
  L-vs-R ambiguity, no floating-point face tests.
- **Rides the whole pipeline** (partition / H5 / viewer / renumber) like any other
  structured region. Reuses the layered-axis + transfinite machinery (the same
  `Axis1D` / transfinite cascade DRMBox uses, without depending on DRMBox).

The post-mesh element-surgery alternative (walk the hex mesh, `addNodes`/
`addElements` into a discrete entity, hand-build the offset-shell topology and
node dedup) is **deferred** — it is only needed to wrap an *arbitrary already-
meshed / foreign* mesh, but ASD is axis-aligned-box-only, so the geometry-time
core covers every real case. Revisit only if a "wrap an existing unstructured
mesh" requirement appears.

### Entry point 1 — `PlainWaveBox` (turnkey)

`g.parts.add_plane_wave_box(...)` — builds the structured soil box + the 1-element
absorbing shell on the 5 truncation faces + optional base plane-wave input wiring
(a `timeSeries` on every `B`-containing skin PG via the AB-2 `-fx/-fy/-fz` path).
This is the common SSI case — the `waveletExample` *is* a plane-wave-from-base
model. Params mirror DRMBox's `(size, n_elements)` layered axes plus
`skin_thickness`, `center`, `rotation_z_deg` (axis-aligned + Z only, fail-loud),
and the base series + active directions. Returns `AbsorbingSkinResult`.

### Entry point 2 — user-defined model

The user builds their own axis-aligned structured soil box (their geometry/mesh,
their PGs) and applies the shell:
`g.model.geometry.add_absorbing_shell(box=<vol/PG>, thickness=…, faces=…)` — runs
the **same geometry-time core** before meshing, adding the offset-shell volumes
around the user's box (conformal once meshed) and tagging the btype PGs. The op
identifies the 5 truncation faces from the box's axis-aligned extent (or from
user-named face PGs); `faces=` lets the user restrict which faces get a skin
(e.g. omit a symmetry plane). This keeps the user in control of the model while
still using the correct-by-construction shell.

Both entry points return the same `AbsorbingSkinResult` and feed AB-2/AB-3
identically.

### btype classification (grid-position, local frame)

In the structured grid, label each axis segment index. A shell cell at
`(ix, iy, iz)`:

```
faces = set()
if ix == 0:            faces.add("L")     # local −X outermost segment
if ix == NX-1:         faces.add("R")     # local +X
if iy == 0:            faces.add("F")     # local −Y   (map F/K to your ± convention)
if iy == NY-1:         faces.add("K")     # local +Y
if iz == 0:            faces.add("B")     # local −Z (bottom); +Z (top) never shelled
btype = "".join(sorted(faces, key="BLRFK".index))   # OR-combined, canonical order
```

A cell with `faces == {}` is interior soil (not shell). Classification is on the
**local** frame (reuse the DRMBox `to_local` rotation, `parts/drm_box.py:230-235`)
so `rotation_z` and a non-zero `center` are handled; refuse any non-Z rotation.

The btype→axis mapping is **PROVEN** (panel cross-checked the element source's
FF-node-row/sorter geometry against STKO centroids — they agree exactly):

| btype | face | letter order | illegal combos |
|---|---|---|---|
| `L` | minimum-X | order irrelevant (`strstr`); | `LR` and `FK` (opposite |
| `R` | maximum-X | emit canonical `BLRFK` order | faces) are **illegal** — |
| `F` | minimum-Y | (`"BLRFK".index`) to match | the element has no branch |
| `K` | maximum-Y | STKO and lock the golden test | for them and silently |
| `B` | minimum-Z (bottom) | | mis-sizes. The generator |
| top(+Z) | free surface — **no skin, no letter** | | must never produce them. |

Because the grid never puts a cell in both `L` and `R` (or `F`/`K`) segments,
opposite-face combos cannot arise by construction — but assert it anyway.

### Physical groups emitted

One PG per distinct btype present, named e.g. `absorbing_L`, `absorbing_LF`,
`absorbing_BLF`, … plus a roll-up `absorbing_all` (for the AB-3 stage-flip set and
for region/Rayleigh assignment). The base-input PGs are every btype containing
`B`. PG creation via `PhysicalGroups.add(dim=3, tags, name=…)`
(`mesh/PhysicalGroups.py:66-162`); the bridge fans out through
`expand_pg_to_elements(fem, pg)` (`opensees/_internal/build.py:1002-1035`) which
reads `fem.elements.select(pg=pg)`.

### Result dataclass

```
@dataclass(frozen=True)
class AbsorbingSkinResult:
    soil_pg: str
    skin_pgs: dict[str, str]      # btype -> PG name   (e.g. "LF" -> "absorbing_LF")
    skin_all_pg: str              # roll-up over every skin cell
    bottom_pgs: tuple[str, ...]   # the B-containing PGs (base-input targets)
    center: tuple[float, float, float]
    rotation_z: float
```

`skin_all_pg` is what AB-2 emits over (one `ASDAbsorbingBoundary3D` decl per
`skin_pgs` entry, fixed btype) and what AB-3 flips / Rayleigh-regions.

### Guards (fail-loud, ADR 0054)

- Non-axis-aligned or non-Z rotation → raise.
- A skin cell whose source soil face is non-planar / skewed beyond tol → raise
  (the element silently idealizes skew in 3D, has no guard in 2D).
- Degenerate/zero-thickness shell cell → raise (3D element `exit(-1)`s on singular
  Jacobian).
- `skin_thickness <= 0` → raise.
- Warn on high skin-cell aspect ratio (grading) — degrades absorption.

## Hooks into existing code

- **shared core — a SESSION-GEOMETRY helper, NOT a Part (F5).** Build directly in
  the live session (`add_box` + `slice` on `g.model.geometry`, then classify),
  avoiding the DRMBox Part/STEP round-trip and its `setCurrent` footgun
  (`_parts_registry.py:607-649`). Lift the reusable **classify + transfinite-
  cascade block** out of `add_DRM_box` (`_parts_registry.py:651-736` — the
  COM-in-local-frame `region_of` lookup + per-volume `set_transfinite`) into the
  helper. Reuse `Axis1D` (`parts/_axis1d.py`) and `to_local`
  (`parts/drm_box.py:230-235`), but **build NEW per-segment-named axes**
  (`("L",…)("soil",…)("R",…)`) — **not** `symmetric_layered`, which labels both
  ±X `"outer"` and would collapse L+R into one PG (F1).
- **Volume PGs only (F3).** Do **not** reuse the line-PG subsystem
  (`classify_drm_box_lines`/`rebuild_drm_box_line_pgs`) — it is the fragile,
  drift-prone part (5° edge-cone tests). The bridge fans out over volume
  elements; one in-memory `PhysicalGroups.add(3, …)` per btype is all that's
  needed (PGs are assembly-side, never persisted through STEP — F4).
- **`g.model.geometry`** — `add_plane_wave_box` (entry 1) is a thin facade over
  the session-geometry helper; `add_absorbing_shell` (entry 2, AB-1b) feeds the
  **same classify/transfinite helper** but via a distinct geometry path
  (boolean-`fragment`-weld of new outer segments onto the user's existing box —
  NOT the monolithic slice, so "one shared core" applies to classify/PG/
  transfinite only, F2).
- **`mesh/PhysicalGroups.py`** — `add()` for the btype PGs (existing API).
- **No DRMBox changes** — DRM and absorbing pair *in a model* (DRM injects,
  the skin absorbs) but are independent facilities.
- **Soil only.** The builder produces soil + skin; embedding a foundation/
  structure is the user's job against `result.free_surface_pg` via the existing
  embed/tie APIs (the waveletExample has no structure — pure soil column).

## Slice breakdown

- **AB-1a** — ✅ **DONE.** The shared **session-geometry** offset-shell helper
  (`src/apeGmsh/parts/plane_wave_box.py`: `build_plane_wave_box` +
  `AbsorbingSkinResult`) + `g.parts.add_plane_wave_box` (entry 1), axis-aligned,
  no rotation: grid construction, btype classification, btype PGs + roll-up,
  `skin_thickness` default = match adjacent soil element per face, fail-loud guards
  (reject layered-`z`, `rotation_z_deg != 0`, non-positive sizes/thickness). Built
  local then translated to `center` (slice plane is origin-sized). 11 tests in
  `tests/parts/test_plane_wave_box.py` (btype distribution, no-illegal-combo,
  scalar thickness, name/center, guards) — all green; full `tests/parts` 59/59.
  `material`/`base_series` deliberately kept off the builder (consumed at the
  bridge in AB-2).
- **AB-1b** — `add_absorbing_shell` user-defined entry (entry 2): boolean-weld of
  outer segments onto the user's box; truncation-face detection from box extent /
  named face PGs, `faces=` restriction.
- **AB-1c** — `center` + `rotation_z` via local-frame classification (refuse other
  rotations); **layered-Z stratigraphy** (`z: list[...]` + per-layer `material`);
  grading + aspect-ratio warning (generous threshold — STKO ships ~2:1 bottom);
  per-axis `skin_thickness`.
- (deferred) — post-mesh surgery path for wrapping an arbitrary/foreign mesh;
  3-component/oblique base input (`base_series: dict[dir, TimeSeries]`).

## AB-1a — concrete API

### The construction is a plain rectangular block (low-risk insight)

The soil + shell together form **one axis-aligned rectangular structured block**
spanning `[−Lx/2−t, Lx/2+t] × [−Ly/2−t, Ly/2+t] × [−Lz−t, 0]` — no L-shaped or
non-convex geometry, no fragmenting. It is structurally identical to what
`add_DRM_box` already builds; only the region scheme differs:

```
axis_x = Axis1D("x", (("L",  -Lx/2 - t, -Lx/2, 1),
                      ("soil", -Lx/2,    Lx/2, nx),
                      ("R",     Lx/2,    Lx/2 + t, 1)))
axis_y = Axis1D("y", (("F",  -Ly/2 - t, -Ly/2, 1),
                      ("soil", -Ly/2,    Ly/2, ny),
                      ("K",     Ly/2,    Ly/2 + t, 1)))
axis_z = Axis1D("z", (("B",  -Lz - t,  -Lz, 1),
                      ("soil", -Lz,      0.0, nz)))   # no skin above z=0 (free surface)
```

Each sub-volume cell classifies by its `(x_region, y_region, z_region)` exactly as
DRMBox does (`region_of` per axis, in the local frame):

```
faces = set()
if x_region in ("L", "R"): faces.add(x_region)
if y_region in ("F", "K"): faces.add(y_region)
if z_region == "B":        faces.add("B")
# faces == {} -> soil cell -> soil_pg
# else -> btype = "".join(sorted(faces, key="BLRFK".index)) -> skin PG
```

So AB-1a is mostly DRMBox's slice + transfinite-cascade + classify with this
3/3/2-region scheme and a btype join — reusing `Axis1D`, `to_local`, and the
transfinite machinery, without depending on DRMBox.

### `add_plane_wave_box` (entry point 1)

```python
def add_plane_wave_box(
    self,
    *,
    # lateral soil extent (symmetric, centred): (full_size, n_elements)
    x: tuple[float, int],
    y: tuple[float, int],
    # vertical soil: single (depth, n) for AB-1a; list[(depth, n)] = layered Z
    # (stratigraphy) is accepted in the TYPE but REJECTED with a clear error in
    # AB-1a, so AB-1c can add it without an API break.
    z: tuple[float, int] | list[tuple[float, int]],
    # absorbing skin (one element thick). None => match the adjacent soil element
    # size per face (STKO-faithful default). Scalar or per-axis (tx,ty,tz) to override.
    skin_thickness: float | tuple[float, float, float] | None = None,
    # placement — AB-1a requires rotation_z_deg == 0.0 (fail-loud otherwise); AB-1c adds Z-rotation
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation_z_deg: float = 0.0,
    # PG naming (mirrors add_DRM_box)
    name: str | None = None,            # prefix; default "pwbox"
    names: dict[str, str] | None = None,
    apply_transfinite: bool = True,
) -> "AbsorbingSkinResult":
```

`center=(0,0,0)` puts the soil top-face centre at the origin, free surface at
`z=0` (DRMBox convention). AB-1a delivers a **pure-geometry** result — soil +
skin PGs only. **`material` / `base_series` are deliberately NOT on the geometry
builder** (F8 abstraction-leak): the bridge consumes `result.skin_pgs` /
`result.bottom_pgs` at the `apeSees(fem)` call site in AB-2/AB-4, where the
material (skin derives `G=E/(2(1+ν))`, ν, ρ per face) and the base series already
belong. A later turnkey one-liner can wrap geometry-result + material + series,
but the geometry builder stays free of bridge types. The per-axis `skin_thickness`
tuple is retained for flexibility but is over-engineering for the golden case — a
single scalar (or the match-adjacent default) reproduces `waveletExample`.

### `add_absorbing_shell` (entry point 2 — AB-1b)

```python
def add_absorbing_shell(
    self,
    *,
    box,                                # the user's soil volume / its PG name
    skin_thickness: float | tuple[float, float, float],
    faces: tuple[str, ...] | None = None,   # restrict skin to these faces; default all 5
    name: str | None = None,
    names: dict[str, str] | None = None,
) -> "AbsorbingSkinResult":
```

Runs the same shell core at geometry time, deriving the box's axis-aligned extent
(and the 5 truncation faces) from `box`; `faces=` drops faces (e.g. a symmetry
plane). Truncation-face detection rule is the AB-1b open sub-decision.

### `AbsorbingSkinResult`

```python
@dataclass(frozen=True)
class AbsorbingSkinResult:
    soil_pg: str                       # the intact interior soil
    skin_pgs: dict[str, str]           # btype -> PG name, e.g. {"L": "...", "LF": "...", "BLF": "..."}
    skin_all_pg: str                   # roll-up over every skin cell (AB-3 flip set + Rayleigh region)
    bottom_pgs: tuple[str, ...]        # the B-containing PG names (base-input targets)
    free_surface_pg: str               # top soil face (z=0) — handy for output / checks
    axes: dict[str, "Axis1D"]          # x/y/z descriptors for downstream sizing
    center: tuple[float, float, float]
    rotation_z: float                  # radians
```

`skin_all_pg` is what AB-2 iterates (one `ASDAbsorbingBoundary3D` declaration per
`skin_pgs` entry, fixed btype) and what AB-3 flips / Rayleigh-regions.

## Verification

- **btype distribution test** — build a small box, assert the PG counts match the
  closed-form face/edge/corner counts (mirrors the STKO tally: faces +
  4·vertical-edges + 4·bottom-edges + 4·bottom-corners, no top).
- **conformality test** — every skin cell's 4 inner nodes are existing soil
  boundary nodes; outer nodes are new and shared between adjacent shell cells (no
  duplicate-coincident outer nodes).
- **soil-intact test** — the inner `nx·ny·nz` soil block is unchanged vs the same
  box built without a skin (node/element parity on the interior).
- **no-illegal-combo test** — assert no emitted btype contains both `L`&`R` or
  both `F`&`K`, and none carries a top letter.
- **golden cross-check (`waveletExample`)** — build a box with `nx=22, ny=20,
  nz=16` and a **single scalar** `skin_thickness=5575`; assert the exact btype
  tally `B=440, L=R=320, F=K=352, LF=LK=RF=RK=16, BL=BR=20, BF=BK=22,
  BLF=BLK=BRF=BRK=1` (total 1936), uniform 5575 thickness on all 5 faces, no top
  skin, and the 4-soil + 4-new node pattern per element. Confirms the scalar path
  reproduces STKO.
- **round-trip** — skin elements + PGs survive `to_h5`/`from_h5` and the viewer.

## Resolved direction

- DRMBox is left alone (it serves the DRM). The skin gets its own facilities.
- **`PlainWaveBox`** (`add_plane_wave_box`) is the turnkey entry; a user-defined
  `add_absorbing_shell` is the bring-your-own-box entry. Both ride the shared
  geometry-time offset-shell core (correct-by-construction).
- Post-mesh surgery is deferred (not needed for axis-aligned ASD).

## Open sub-decision (resolve at AB-1b)

How the user-defined entry identifies the 5 truncation faces of a BYO box:
auto-detect from the box's axis-aligned bounding extent (default), vs require
user-named face PGs. Lean to auto-detect with a `faces=`/named-PG override.
