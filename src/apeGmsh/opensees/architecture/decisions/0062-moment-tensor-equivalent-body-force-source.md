# ADR 0062 — Moment-tensor equivalent body-force source (embedded seismic source)

**Status:** Proposed (2026-06-17; design draft). **MT-1 → MT-3 + MT-4a SHIPPED
(2026-06-17)** — the MT math core, the point→host→nodal-force build with the
`p.moment_tensor` authoring surface, the `S(t)` moment-function helpers, and
the bridge-level `ops.fault.from_shakermaker` finite-fault adapter. **MT-4b
(`from_ffsp`) is deferred to MT-5**: an adversarial review source-verified that
FFSP `get_subfaults()` units differ from the original premise (coords are
**metres** not km — `ffsp_wrapper.f90` ×1000; `peak_time` is the dimensionless
ratio `pktm/(rstm+pktm)` not seconds — `spfield_n.f90:589,598`; `slip` is
rescaled by `area_sub/μ` when `is_moment>1`), so a correct `M0` needs
validation against a real FFSP run. MT-5 (FK-vs-FEM validation example +
absorbing-skin guard + the corrected `from_ffsp`) remains. Pairs with [0054](0054-asd-absorbing-boundary.md) (the
absorbing skin the source radiates into), rides the load/pattern machinery of
[0005](0005-patterns-explicit.md) /
[0007](0007-time-series-separated-from-pattern.md) /
[0051](0051-bridge-load-consumption.md), and reuses the point-in-host
shape-function machinery of [0036](0036-embedded-host-decomposition.md). The
ShakerMaker adapter consumes `FaultSource` / `FFSPSource.get_subfaults()` and
mirrors the Aki & Richards convention already in ShakerMaker's
`core/radiats.c` — **verified term-for-term against `nmtensor` during MT-1.**

## Context

We can drive a solid box three ways today: a prescribed base plane-wave
(`add_plane_wave_box` + a `Ricker` on the bottom absorbing cells, ADR 0054),
DRM boundary forces (the `DRMBox` facility + ShakerMaker FK), or hand-authored
`p.load` patterns. **None of them let you put an earthquake *source* inside the
mesh and let the solver radiate it.** That is the missing capability: model the
solid **and** the fault, embed a seismic moment tensor at depth, and propagate
the wavefield with the explicit (or implicit) integrator. It is exactly what
SW4 / SPECFEM / EQdyna do internally, and it is the prerequisite for using a
ShakerMaker rupture (a single double-couple now; an FFSP finite fault later) as
the excitation of a Ladruno-fork explicit run.

A seismic point source is a **moment tensor** $M_{ij}(t)$. By the representation
theorem (Aki & Richards 2002, Ch. 3) its radiated field equals that of an
equivalent **body force** $f_i = -\partial_j M_{ij}$; for a point source at
$\xi$, $f_i(x,t) = -M_{ij}(t)\,\partial_j\delta(x-\xi)$. Projected onto the FE
basis, the **consistent nodal force** on node $a$ of the host element is

$$
F^a_i(t) = M_{ij}(t)\,\frac{\partial N_a}{\partial x_j}\bigg|_{\xi},
\qquad M_{ij}(t) = M_0\,m_{ij}\,S(t-t_0),
$$

with $m_{ij}$ the **unit** moment tensor (double-couple from strike/dip/rake),
$M_0=\mu A\bar D$ the scalar moment, $S(\cdot)$ the **normalized moment
function** (the *time-integral of the slip-rate*, rising $0\to1$), and $t_0$ the
rupture onset. This is a pure **right-hand-side** contribution — no stiffness,
no constraint, integrator-agnostic — which is why it suits the explicit fork
(its LHS is the mass matrix alone) with zero solver change.

Unit moment tensor (Aki & Richards Eq. 4.83–4.88; geographic $x$=N, $y$=E,
$z$=**Down**), the same convention as ShakerMaker `core/radiats.c`
("double-couple specified by az-strike, dip, rake"):

$$
\begin{aligned}
m_{xx} &= -(\sin\delta\cos\lambda\sin2\phi + \sin2\delta\sin\lambda\sin^2\phi)\\
m_{yy} &= \ \ \sin\delta\cos\lambda\sin2\phi - \sin2\delta\sin\lambda\cos^2\phi\\
m_{zz} &= \ \ \sin2\delta\sin\lambda = -(m_{xx}+m_{yy})\\
m_{xy} &= \ \ \sin\delta\cos\lambda\cos2\phi + \tfrac12\sin2\delta\sin\lambda\sin2\phi\\
m_{xz} &= -(\cos\delta\cos\lambda\cos\phi + \cos2\delta\sin\lambda\sin\phi)\\
m_{yz} &= -(\cos\delta\cos\lambda\sin\phi - \cos2\delta\sin\lambda\cos\phi)
\end{aligned}
$$

## Decision

Add a **moment-tensor source** as a first-class, bridge-authored **load** that
resolves a point (or a fault of points) into nodal `load` records modulated by a
`timeSeries` moment function. It is a load, **not** a new element: OpenSees has
no double-couple source element, and forces-on-the-RHS is the clean,
integrator-agnostic, fork-change-free path.

1. **Authoring surface — a source on the bridge pattern.** A moment-tensor
   source is born on the bridge (it needs `fem` coords/connectivity to locate
   the host and evaluate $\partial N/\partial x$), so it lives next to `p.load`,
   not as a `g.loads` geometry case:

   ```python
   ts = ops.timeSeries.Path(dt=dt, values=S)        # normalized moment function S(t)
   with ops.pattern.Plain(series=ts) as p:
       p.moment_tensor(                              # single double-couple
           position=(x, y, z),                       # mesh coords; host found automatically
           M0=6.3e15,                                # scalar moment (deck units, e.g. kN·m)
           mech=dict(strike=350, dip=40, rake=113),  # OR m_ij=<3x3>
           t0=0.0,                                    # rupture onset (s)
           method="consistent",                      # "consistent" (∂N/∂x) | "dipole"
       )
   ```

   Stage-scoped via `s.pattern(series=...)` under the no-mixing rule (ADR 0051
   §5). The per-source forces are constant vectors $M_0 m_{ij}\,\partial N_a/\partial x_j$;
   `S(t)` is the shared `Path` series. **Apply $S(t)$ (the moment function), not
   $\dot S$ (the slip-rate).**

2. **Two force-build methods, both mesh-objective.**
   - **`"consistent"`** — locate the host element, evaluate the trilinear/tet
     shape-function gradients at $\xi$, emit $F^a = M\!\cdot\!\nabla N_a$ on the
     host's corner nodes. Reuses ADR 0036's **point-in-host finder**
     (`_inverse_map.locate_point`) but adds its own $\partial N/\partial x$
     projection (`shape_gradient_phys`; the finder exposes only $\xi$ / weights
     — open-Q #3, resolved).
   - **`"dipole"`** — place the source at the nearest mesh node and apply
     force-dipoles on its $\pm$ axis neighbours ($F^a_i \mathrel{+}= \pm
     M_{ij}/(2h_j)$). Trivial on a structured `add_plane_wave_box` grid; the
     validation fallback. Net force and net torque are zero for symmetric $M$
     (a physical seismic source) — asserted at build.

3. **Moment tensor from `(strike,dip,rake)` with an explicit frame contract.**
   `mech=` builds $m_{ij}$ via the formulas above (or pass `m_ij=` directly).
   A required `frame=` flag maps **Aki & Richards $z$-down** to the **mesh
   $z$-up** convention (flip $m_{xz}, m_{yz}$; the source depth is a positive
   number below the free surface). This is the #1 footgun — it silently mirrors
   the radiation pattern — so it is explicit and the FK cross-check (below) is
   the catch-all.

4. **Normalized moment function helpers → `Path`.** Ship `S(t)` builders: a
   smoothed step (error-function ramp, $\dot S$ Gaussian — clean, one-sided,
   integral 1) and the **modified-Yoffe** used by FFSP (shaped by
   `rise_time`/`peak_time`). All band-limited to a caller `f_max`; a build-time
   warn fires if $\dot S$ carries energy above the mesh's resolvable frequency
   (the P=3 dispersion lesson, ADR 0054 AB-4).

5. **ShakerMaker adapter — a finite fault is a loop of sources.** *(Shipped as
   bridge-level `ops.fault.*` returning one pattern per source, not `p.from_*` —
   see open-Q #2 / the slice plan; the prose below predates that resolution.)*
   `p.from_shakermaker(fault)` ingests a ShakerMaker `FaultSource` (a list of
   `PointSource`, each with `x`, `angles=[φ,δ,λ]`, `tt`, `stf`); each becomes one
   `moment_tensor(...)`. `p.from_ffsp(subfaults, crust)` ingests
   `FFSPSource.get_subfaults()` (aligned arrays `x,y,z,slip,strike,dip,rake,
   rupture_time,rise_time,peak_time`) — per subfault $M_0^{(k)}=\mu_k A_k D_k$
   with $\mu_k=\rho V_s^2$ read from the medium at the subfault depth (the same
   quantity FFSP computes internally as `amu(k)`), $t_0^{(k)}=$ `rupture_time`,
   and a per-subfault Yoffe `S` from `rise_time`/`peak_time`. The FK Green's
   functions are shared across an FFSP ensemble, but that amortization is
   ShakerMaker's; here every realization is just a different fault of point
   sources. **The dependency direction is fixed by decision 8 — apeGmsh imports
   nothing from ShakerMaker.**

6. **Units contract (fail-loud).** ShakerMaker/FFSP author in **km** and
   **N·m**; the deck is **kN·m·s** (`baseUnits`). The adapter converts coords
   ×1000 (km→m), $M_0$ ÷1000 (N·m→kN·m), and derives $\mu$ in deck-consistent
   units. Frame and unit flags are required, not defaulted.

7. **No schema bump for v1** — the source emits ordinary `load` lines + a `Path`
   `timeSeries`, which already round-trip. A `/opensees/sources` provenance
   group (strike/dip/rake, $M_0$, position, $t_0$) is a **deferred follow-up**
   for `model.h5` so the viewer can draw beachballs and `Results` can label the
   source.

8. **Dependency contract — ShakerMaker is NOT an apeGmsh dependency; the
   interchange is plain data.** A general meshing/FEM library must not couple to
   a niche FK-seismology package (most apeGmsh users never touch it), consistent
   with the decoupling doctrine of [0001](0001-decouple-from-gmsh-session.md) /
   [0009](0009-no-backwards-compat-with-solvers.md). **The dependency arrow never
   points apeGmsh → ShakerMaker.** Concretely:
   - **Neutral core.** `p.moment_tensor(...)` and the force / $S(t)$ builders
     take only scalars and numpy arrays (`position`, `M0` or $\mu A\bar D$,
     `strike/dip/rake` or `m_ij`, `rise/peak` times). They import nothing
     seismological.
   - **Primary path is the plain dict.** `p.from_ffsp(subfaults, crust)` consumes
     the `get_subfaults()` **dict of aligned numpy arrays** + a plain crust
     description, and imports **no** ShakerMaker symbol — the interchange is data,
     not objects.
   - **Live-object path is duck-typed.** `p.from_shakermaker(fault)` iterates the
     object and reads each sub-source's documented public attributes
     (`.x`, `.angles`, `.tt`) — no `import shakermaker`. A typed,
     `isinstance`-validating variant, *if* ever wanted, hides behind an **optional
     extra** `apeGmsh[shakermaker]` with a **lazy import inside the function**,
     never at module top.
   - **apeGmsh owns the STF.** `S(t)` is built by the decision-4 helpers from
     `rise/peak` times; ShakerMaker's STF classes are not imported. A caller may
     instead pass a pre-sampled `(t, S)` array, used verbatim.
   - Net: the glue, if any, is a few lines the user owns (e.g. the Ladruno
     project) or the optional lazy adapter — `core/test` apeGmsh has zero
     seismology imports.

## Why not the alternatives

- **A "source element" (new C++ class).** Would need a fork element and break
  the "no solver change" property. The RHS-force formulation is exact (it *is*
  the discrete representation theorem) and integrator-agnostic — rejected.
- **Split-node / Day kinematic fault.** Prescribing slip across a *meshed* fault
  surface (duplicated nodes + relative-displacement constraint) is the right
  tool for **on-fault near-field / spontaneous dynamic rupture**, but it
  requires the fault to be a discontinuity surface in the mesh and is far
  heavier. For radiating a **known kinematic** source into the medium, point-MT
  equivalent forces are the standard, mesh-light choice. Split-node is a
  separate future facility, not this ADR.
- **Hand `p.load` only.** That stays the escape hatch. The ADR's value is
  centralizing the MT→force math, the frame/units contract, the host projection,
  and the ShakerMaker adapter — precisely the footguns that must not be
  re-derived per study.
- **A `g.loads` geometry case.** Rejected: a point source needs host-element
  location and $\partial N/\partial x$ (a bridge/FEM concern), and its time
  series is born on the bridge. It is opt-in pattern content (ADR 0051), not
  tributary geometry load.

## Gotchas the implementation must honor

- **Frame (A&R $z$-down vs mesh $z$-up).** Silent radiation-pattern mirror if
  unhandled. Required `frame=`; the FK cross-check is the guard.
- **$S(t)$ vs $\dot S(t)$.** The displacement-based nodal force takes the
  **moment function** (integral, $0\to1$), not the slip-rate. Wrong choice =
  one time-derivative off in every trace.
- **Band-limit to the mesh.** Inject no energy above $V_s^{\min}/(P\,h)$ — warn
  at build (mirrors the ADR 0054 P=3 dispersion finding).
- **Interior-source requirement.** The source must sit inside the intact
  continuum, not in the absorbing skin and not on the free surface (the
  `"dipole"` fallback needs all six neighbours; `"consistent"` needs a
  non-degenerate host). A point *outside* every host fails loud (MT-2).
  **Skin-cell detection is deferred to MT-5** (it needs the absorbing-PG /
  element set threaded into the emit-time host walk; the skin is real hex/quad
  geometry, so it is a geometrically valid host the current guard cannot
  distinguish). Until MT-5, a skin-placed source emits silently — call it out
  in the worked example.
- **Net force/torque zero.** Assert $\sum_a F^a = 0$ and, for a symmetric $M$,
  zero net torque — a non-zero residual means a frame/winding bug.
- **$\mu$ from the medium, per subfault.** $M_0=\mu A\bar D$ uses the *local*
  shear modulus; a single $\mu$ over a layered crust mis-scales slip↔moment
  (FFSP does this depth lookup deliberately).

## Validation (acceptance test — mirror the ADR 0054 rigor)

ShakerMaker FK is the **exact 1-D oracle** for a point double-couple. Ship a
run-verified example:

1. Single double-couple at depth in a layered box (`add_plane_wave_box` with the
   crust as layers, absorbing skin on the 5 truncation faces, free surface on
   top), explicit, modest `f_max`.
2. The **same** `PointSource` through ShakerMaker FK at matching surface
   receivers.
3. Gates: (a) the four-lobe **double-couple radiation pattern** in first-motion
   amplitude vs azimuth; (b) FEM-vs-FK surface velocity within a few % of peak
   at $P\ge6$; (c) absorbing late-time energy small (the ADR 0054 quiet-base
   check). Register under `docs/examples/` + mkdocs, like `plane-wave-ssi.md`.

## Slice plan

- **MT-1 — MT math + frame contract (pure). ✅ SHIPPED.**
  `_kernel/geometry/_moment_tensor.py`: $m_{ij}$ from $(\phi,\delta,\lambda)$,
  the A&R-down↔mesh-up flip (required `frame`), $M_0$ from $M_w$ (SI Hanks &
  Kanamori) or $\mu A\bar D$. Unit-tested against the hand-computed value
  ($\phi{=}350,\delta{=}40,\lambda{=}113$ →
  $m=[-0.113,-0.793,0.906;-0.391,0.323,0.105]$, trace 0) **and** the
  ShakerMaker `nmtensor` oracle (the ⚠owed convention check — done).
- **MT-2 — point→host→nodal force. ✅ SHIPPED.** `"consistent"` (host finder
  `_inverse_map.locate_point` + the new `shape_gradient_phys` Jacobian-inverse
  projection — see open-Q #3) and `"dipole"` fallback (compensating-arm couple
  — net-force-zero on *graded* grids too, not just uniform);
  `Plain.moment_tensor(position, frame, M0, mech|m_ij, method)` authoring
  surface; bridge emit, **flat *and* partitioned (OpenSeesMP)** — the
  per-rank load lines reproduce the flat deck (the SSI/wave-prop workload runs
  partitioned). Out-of-continuum points fail loud (MT-flavoured `BridgeError`),
  and the **exact** net-force-zero invariant is asserted. Decisive test:
  emitted forces recover the moment tensor as their first moment
  ($\sum_a x_a\otimes F_a = M$), unit + e2e on a real meshed hex box.
  **`t0 \neq 0` fails loud — per-source onset delay is MT-4.** ⚠ The
  *interior-source* guard rejects points outside the continuum but does **not**
  yet detect a point that lands *inside the absorbing skin* (a valid host
  geometrically) — that needs the absorbing-PG plumbing wired in **MT-5** when
  the plane-wave-box integration lands; until then a skin-placed source emits
  silently. Documented on `p.moment_tensor` ("must lie inside the intact
  continuum").
- **MT-3 — moment-function helpers → `Path`. ✅ SHIPPED.**
  `ops.timeSeries.MomentStep` (erf ramp) + `Yoffe` (regularized modified-Yoffe,
  Tinti 2005) façades; shared spectral band-limit warn
  (`WarnMomentFunctionBandwidth`).
- **MT-4a — `ops.fault.from_shakermaker`. ✅ SHIPPED.** **Bridge-level**
  (open-Q #2 resolved: **one `Plain` pattern per source**, since OpenSees binds
  one `timeSeries` per pattern). Each source's rupture onset rides its own
  `Yoffe(t0=tt)`; the `moment_tensor` record's own `t0` stays 0 (no clash with
  the MT-2 guard). Duck-types the `FaultSource` (reads `.x`/`.angles`/`.tt`, no
  import — **`.angles` is radians**, converted to degrees) and takes `M0`
  +rise/peak directly (a `PointSource` carries none). Required `length_scale`
  (km→deck). Guards: `WarnFaultSubfaultSkipped` (zero moment/rise/peak),
  `WarnFaultPeakClamped` (peak_time clamped below rise/2 for the Yoffe bound),
  `WarnFaultSubfaultTruncated` (onset/rise outside `t_total`). Per-subfault
  patterns emit flat and partitioned.
- **MT-4b — `ops.fault.from_ffsp`. ⏭ DEFERRED to MT-5.** The FFSP
  `get_subfaults()` unit contract is source-verified to differ from the design
  premise on all three axes (coords metres; `peak_time` ratio; `slip`
  `is_moment`-rescaled), and a correct `M0=\mu A\bar D` needs end-to-end
  validation against a real FFSP run. The corrected adapter (metres→deck
  coords, `peak_seconds = ratio·rise_time`, `is_moment==1` slip guard,
  `area_m2 = source.area·1e6`) lands with the MT-5 validation example.
- **MT-5 — validation example.** FK-vs-FEM overlay + radiation lobes, run-verified, mkdocs.
- **Deferred:** `/opensees/sources` provenance + viewer beachball; split-node
  kinematic fault (separate facility).

## Open questions

1. `p.moment_tensor` (pattern-scoped) vs a dedicated `ops.source.*` namespace —
   does a source ever need to span patterns/series? **RESOLVED (MT-2):
   pattern-scoped** — `p.moment_tensor` lives on `Plain` next to `p.load`
   (it needs the pattern's `series` to carry $S(t)$); the `S(t)` helpers live
   under `ops.timeSeries`. A finite fault with per-subfault $t_0$ (MT-4) will
   share one $S$ shifted per-load, or one series per unique rise time.
2. Per-subfault $t_0$ delay — a single `Path` with per-`load` time shift, or
   N series? **RESOLVED (MT-4): one pattern + one `Yoffe` series per subfault**,
   the series carrying the onset (`t0=rupture_time`). Cleanest mapping onto
   OpenSees' one-series-per-pattern binding; deck size scales with subfault
   count (inherent to a propagating finite-fault rupture). MT-2's
   `moment_tensor` `t0` stays 0 — the delay lives in the series, not the load.
3. Does ADR 0036's host decomposition expose $\partial N/\partial x$ at an
   arbitrary interior point, or only the corner-coupling weights?
   **RESOLVED (MT-2): only the latter.** `_inverse_map` returns $\xi$, the
   shape weights $N(\xi)$, and (internally) $\partial N/\partial\xi$
   (`_hex8_dN`/`_quad4_dN`), but **not** $\partial N/\partial x$. MT-2 adds
   `shape_gradient_phys` = $\partial N/\partial\xi\cdot J^{-1}$ (with
   $J = X^\top\partial N/\partial\xi$, guarded on $\det J$). So MT-2
   "consistent" reuses the host *finder* but the spatial-gradient projection
   is new.
4. Live (`ops.run`) support, or Tcl/py emit first? (Esmeralda path is Tcl;
   lean emit-first, live as follow-up. MT-2 is emit-side — works in any
   emit target.)
