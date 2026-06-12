# apeGmsh absorbing boundaries

A guide to truncating a soil domain with the `ASDAbsorbingBoundary3D` / `2D`
skin (ADR 0054): building the plane-wave box, declaring the skin on the
bridge, the staged hold→absorbing lifecycle, and the run-verified modeling
recipe — mass, damping, base input, and how to *validate* that the boundary
is actually working. This document covers the apeGmsh abstraction; see
`guide_opensees.md` for the bridge in general and `guide_masses.md` for the
tributary-mass channel the recipe leans on.

All snippets assume the usual pipeline:

```python
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.material.nd import ElasticIsotropic
from apeGmsh.opensees.time_series.time_series import Ricker
```

## Tasks on this page

- [The 30-second picture](#1-the-30-second-picture)
- [Build the box](#2-build-the-box-gpartsadd_plane_wave_box) · [What comes back](#the-absorbingskinresult)
- [Declare the skin](#3-declare-the-skin-opselementabsorbing_boundary) · [Why the impedance material is never emitted](#the-impedance-material-is-never-emitted)
- [Inject the input motion](#4-inject-the-input-motion)
- [The staged lifecycle](#5-the-staged-lifecycle-hold--absorbing)
- [The modeling recipe](#6-the-modeling-recipe-run-verified)
- [Validate the boundary](#7-validating-the-boundary)
- [Pitfalls](#8-pitfalls)


## 1. The 30-second picture

Three calls and one stage verb, each on the layer where it belongs:

```python
# 1 ── SESSION: soil box + absorbing skin in one call
with apeGmsh(model_name="site") as g:
    skin = g.parts.add_plane_wave_box(
        x=(200.0, 4), y=(200.0, 4),
        z=[(45.0, 8), (75.0, 4), (80.0, 4)],   # layered, TOP → BOTTOM
    )
    for pg, layer in zip(skin.soil_pgs, LAYERS):
        g.masses.volume(pg, density=layer.gamma)   # soil only — never the skin
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data()

# 2 ── BRIDGE: fan the whole skin from one declaration
ops = apeSees(fem)
soil_mats = [ops.register(ElasticIsotropic(E=L.E, nu=L.nu, rho=0.0))
             for L in LAYERS]                                   # rho=0: mass is nodal
impedance = [ElasticIsotropic(E=L.E, nu=L.nu, rho=L.gamma)
             for L in LAYERS]                                   # never emitted (§3)
ricker = ops.register(Ricker(f_n=5.0, t_total=2.5, dt=0.005,
                             t_center=0.3, kind="velocity", factor=0.01))
for mat, pg in zip(soil_mats, skin.soil_pgs):
    ops.element.stdBrick(pg=pg, material=mat)
ops.element.absorbing_boundary(skin=skin, materials=impedance,
                               base_series=ricker, base_dirs=("x",))

# 3 ── STAGE: flip the skin from HOLD to ABSORBING, then run
with ops.stage(name="dynamic") as st:
    st.activate_absorbing(pg=skin.skin_all_pg)
    st.analysis(...)
    st.run(n_increments=500, dt=0.005)
```

The skin replaces *distance*: instead of pushing fixed boundaries kilometres
away, the lateral cells enforce the free-field motion at the truncation and
the base is a compliant (dashpot) boundary that injects the incident wave and
absorbs the outgoing one through the same elements.


## 2. Build the box: `g.parts.add_plane_wave_box`

```python
skin = g.parts.add_plane_wave_box(
    x=(200.0, 4),                # (extent, n_elements) along X
    y=(200.0, 4),
    z=(200.0, 10),               # homogeneous: one (thickness, n) pair …
    # z=[(45, 8), (75, 4), (80, 4)],  # … or layered, TOP → BOTTOM
    # skin_thickness=…,          # default: adjacent soil element size; scalar or (tx, ty, tz)
    # center=(x0, y0, z0), names={…},
)
```

The builder constructs, **in the live session** (no Part/STEP round-trip), an
axis-aligned structured (transfinite hex) soil box plus a one-element-thick
absorbing **offset shell** on its five truncation faces — the local `+Z` top
is the free surface and is never shelled. Soil + shell are sliced only at
region breakpoints, giving 18 sub-volumes: one soil region and up to 17 skin
regions (5 face panels, 4 vertical edges, 4 bottom edges, 4 bottom corners).
Each skin region is tagged with its `btype` — the set of truncation faces it
lies outside of, OR-combined (`L`=min-X, `R`=max-X, `F`=min-Y, `K`=max-Y,
`B`=min-Z; e.g. a bottom corner is `"BLF"`). A layered `z=` stratifies the
soil *and* the lateral skin, so each layer's boundary can carry its own
impedance.

If the skin comes out much thicker than the adjacent soil element (above
~4:1; STKO ships ~2:1) the builder warns with `WarnAbsorbingSkinAspect` —
elongated boundary hexes absorb poorly. Fail-soft: a thick skin is legal,
just flagged.

Siblings, same contract:

- `g.parts.add_absorbing_shell(...)` — bring-your-own-box: wraps an existing
  soil volume you already built.
- `g.parts.add_plane_wave_box_2d(...)` / `add_absorbing_shell_2d(...)` —
  plane-strain rectangle in the global X–Y plane (X lateral, Y vertical,
  free surface on top); skin regions `L R B BL BR` fan out to
  `ASDAbsorbingBoundary2D` quads.

### The `AbsorbingSkinResult`

The return value is a bag of physical-group names, so nothing downstream
touches a raw tag:

| field | what it names |
|---|---|
| `soil_pg` / `soil_pgs` | the intact interior soil volume (per layer when stratified) |
| `skin_pgs` | `btype → PG` for every skin region (`{"L": "absorbing_L", "BLF": …}`) |
| `skin_pgs_by_layer` | `layer → (btype → PG)` for stratified skins |
| `skin_all_pg` | roll-up over every skin region — the stage-flip target |
| `bottom_pgs` | the `B`-containing PGs — where the base input attaches |
| `free_surface_pg` | dim-2 PG of the soil top face (tie structures here) |
| `axes`, `n_layers`, `ndm` | sizing descriptors and the 2D/3D dispatch flag |


## 3. Declare the skin: `ops.element.absorbing_boundary`

One facade call registers one element declaration per `btype` PG; the
bridge's fan-out then writes one `element ASDAbsorbingBoundary3D` line per
hex cell with the shared properties:

```python
# homogeneous skin — a single material (or raw G=, v=, rho=)
ops.element.absorbing_boundary(skin=skin, material=imped)

# stratified skin — one material per layer, TOP → BOTTOM
ops.element.absorbing_boundary(skin=skin, materials=[m_top, m_mid, m_bot])

# 2D skin — thickness is REQUIRED (out-of-plane plane-strain slab; match your quads')
ops.element.absorbing_boundary(skin=skin2d, material=imped, thickness=1.0)
```

With `materials=` the length must equal `skin.n_layers`; each layer's lateral
cells get that layer's properties and the base skin takes the **bottom**
layer's. The raw per-PG primitive
(`ops.element.ASDAbsorbingBoundary3D(pg=…, G=…, v=…, rho=…, btype=…)`) is the
escape hatch when you built the skin PGs yourself.

### The impedance material is never emitted

The upstream element command has **no material-tag slot** — it takes raw
`G v rho` doubles:

```tcl
nDMaterial ElasticIsotropic 1 2567314.3 0.3 0.0                    # soil: registered, has a tag
element stdBrick 219  40 217 400 204 …  1                          # brick references the tag
element ASDAbsorbingBoundary3D 19  265 88 …  987428.6 0.3 2.4 F    # skin: raw G v rho inline
```

So the facade reads `G = E/(2(1+ν))`, `ν`, `ρ` off the `ElasticIsotropic`
you pass **at construction time** and stores three floats; the material is
not a dependency and no `nDMaterial` line is ever written for it. Accepting
a material object at all is ergonomics — you already have per-layer
materials around. The practical consequence: your *soil* materials carry
`rho=0` (mass is nodal, §6) while the *impedance* objects carry the real
`ρ` the dashpots need — same `E`/`ν`, different `ρ`, no collision.


## 4. Inject the input motion

```python
ricker = ops.register(Ricker(f_n=5.0, t_total=2.5, dt=0.005,
                             t_center=0.3, kind="velocity", factor=0.01))
ops.element.absorbing_boundary(skin=skin, materials=impedance,
                               base_series=ricker,
                               base_dirs=("x",))    # -fx → vertically propagating S-wave
```

`base_series` is the incident motion as a **velocity history**; it attaches
as `-fx` / `-fy` / `-fz` (per `base_dirs`) to the `B`-containing cells
*only* — the upstream element consumes base input exclusively on bottom
boundaries, and the typed primitive rejects a series anywhere else rather
than let it silently no-op. `("x",)` or `("y",)` injects a shear wave;
`("z",)` a P-wave (`"z"` is invalid on a 2D skin).

This is a *compliant base*: the same dashpots that inject the wave absorb
whatever comes back down. The injection rides the element internal-force
channel (`addBaseActions`), which is why, in the energy balance, the input
shows up as **negative internal work** (see §7).


## 5. The staged lifecycle: hold → absorbing

The element is staged by design:

- **Stage 0 — HOLD** (initial): the skin is penalty-frozen to the soil box.
  A gravity / initial-state stage equilibrates against rigid lateral
  support, exactly like the classic fixed-boundary gravity step.
- **Stage 1 — ABSORBING**: dashpots + free-field columns active.

```python
with ops.stage(name="dynamic") as st:
    st.activate_absorbing(pg=skin.skin_all_pg)   # the flip
    ...
```

`s.activate_absorbing` resolves the PG to element tags and emits a
self-contained one-shot block —
`parameter` / `addToParameter … stage` / `updateParameter 1` /
`remove parameter` — at the top of that stage. Partitioned decks get the
right semantics automatically: each rank flips only the elements it owns
(an id owned elsewhere is silently skipped), while in a sequential deck an
unresolvable id is a hard `BridgeError`. Staged decks emit and run as text
(`ops.tcl` / `ops.py`), never live in-process.

The canonical two-stage pattern is gravity-then-dynamic; for a pure
plane-wave study a single dynamic stage that flips immediately (as in §1)
is enough.


## 6. The modeling recipe (run-verified)

Every item below was established by running, not by reading manuals.

**Mass: nodal, soil-only.** Declare tributary mass on the session
(`g.masses.volume(pg, density=γ)` per layer) and re-declare it per node on
the bridge (`ops.mass`); emit the soil bricks with `rho=0` materials so
element mass never double-counts it. **Never put nodal mass on the skin** —
its lateral cells carry their own free-field mass internally, and extra
mass detunes the absorption (measured: late-time spurious motion grows from
~20 % to ~80 % of peak).

```python
for mrec in fem.nodes.masses:
    ops.mass(nodes=[int(mrec.node_id)],
             values=tuple(float(v) for v in mrec.mass[:3]))
```

**Fix the bottom outer node plane when running explicit.** In absorbing
mode the element penalty-fixes its below-soil outer nodes anyway
(`addKPenaltyStage1`); declaring the same constraint as a real `ops.fix`
removes those massless, springs-only DOFs from the system, which an
explicit mass-only-LHS solver cannot tolerate (`Diagonal aii = 0`).
Implicit results are unchanged.

**Damping: global mass-proportional Rayleigh.** For explicit studies use
`ops.damping.rayleigh(alpha_m=4πζf_anchor)` — βK in any flavour collapses
the explicit stable step and drags a non-diagonal C onto stock
central-difference LHSs. Two source-verified facts make the **global** form
the right one here:

1. Region-scoped Rayleigh (`on=…`) attaches to *elements* only — and the
   soil elements have `rho=0`, so element-side αM·M is zero. Only the
   global `rayleigh` command also reaches the **nodes**, where the mass
   actually lives.
2. The skin is *supposed* to feel the same α: `ASDAbsorbingBoundary3D`
   deliberately applies the domain's Rayleigh α to its lateral free-field
   columns (`addCff`; the bottom face is skipped). The free field must
   match a damped interior, or the truncation mismatch radiates spurious
   waves.

**Lateral extent can be small.** The free-field columns enforce the 1-D
site response at the truncation, so a few wavelengths of plan extent
replace kilometres of domain. Widen it when a structure sits inside.


## 7. Validating the boundary

Quietness is **not** a valid check for stratified profiles — a soft layer
over a stiff one is a physical resonator that keeps ringing long after the
pulse (verified: removing the layers drops the late-time surface envelope
by an order of magnitude, ~16 % → ~1 % of peak). Use these instead:

- **Plane-wave coherence** — every free-surface node moves identically
  (cross-node scatter at machine zero); the free-field boundary is doing
  its 1-D job.
- **Windowed decay** — the surface envelope falls window-over-window after
  the pulse (and its reflections) have passed.
- **Energy balance** — record with `ops.recorder.Ladruno(..., energy=True)`
  and read `results.energy()` (`KE/IE/DW/ULW/RES/ERR`, fork builds). With
  base injection through the elements, `ULW = 0` and the closure reads
  `−IE = KE + DW + RES`: the boundary is working when `DW` climbs onto
  `−IE` (≈ 98 % absorbed on the reference runs) and `ERR` stays well under
  1 %. Note `DW` also contains material (Rayleigh) damping work — run a
  ζ = 0 twin to isolate the dashpots.
- **Sequential ↔ partitioned identity** — same physics across 1 vs N ranks
  to ~1e-15 of peak; partitioning may change wall time, never the answer.
- **Know the residual.** A Lysmer-type base is exact only in the continuum
  limit: expect a small re-reflection of the returning free-surface echo,
  arriving one extra round trip later (4H/Vs) at a few % of peak amplitude
  (~5 % measured on the wavelength-rule mesh — consistent with ≈ 98 %
  energy absorption).


## 8. Pitfalls

- **Mass on the skin.** The single most damaging mistake — see §6.
- **`btype` is validated, not trusted.** Letters outside `BLRFK` (or
  outside `BLR` in 2D), repeated letters, and opposite-face pairs
  (`LR`/`FK`) are construction-time errors — a real box cell can never
  carry them, and the element would silently mis-size if it could.
- **Base series off the bottom.** `-fx/-fy/-fz` on a non-`B` cell is
  rejected at construction (upstream gates them on `BND_BOTTOM`; at best a
  silent no-op).
- **`thickness=` is 2D-only.** Required for a 2D skin (out-of-plane slab
  thickness — match your soil quads'), rejected for a 3D one.
- **Skewed 2D skins.** `ASDAbsorbingBoundary2D` self-sorts its nodes and
  has no distortion handling; the builders keep the skin axis-aligned —
  don't hand-build skewed quads.
- **Region-scoped Rayleigh on `rho=0` soil damps nothing** — see §6.
- **Stock `CentralDifference` + skin dashpots.** That integrator assembles
  C into its LHS; the skin's dashpots make C non-diagonal, so the
  `Diagonal` solver is invalid for it (the fork's `ExplicitBathe` /
  `CentralDifferenceLadruno` and stock `ExplicitDifference` are
  mass-only-LHS and unaffected).
- **Thick-skin warning.** Heed `WarnAbsorbingSkinAspect`; keep the skin
  near the adjacent soil element size.

Worked end-to-end examples: the staged-gravity SSI example shipped with
ADR 0054 (AB-4), and the plane-wave study notebooks in the
`Plain Wave JAA` project folder (stratified physics reference, faithful-
geometry scaling, local integrator/damping rig, single-layer control).
