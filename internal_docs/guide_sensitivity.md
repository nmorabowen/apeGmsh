# Guide — black-box sensitivity (damping & beyond) in apeGmsh

> **Status:** guide + forward-looking API proposal. The mechanics work **today**
> with the shipped `apeSees` bridge + `Results`; the first-class `apeGmsh`
> sensitivity driver in §6 is proposed, not yet implemented.
> See also `plan_damping_integration.md`, `guide_opensees.md` (the `ops.damping.*`
> surface), `guide_obtaining_results.md`.

## 1. What this is, in one breath

**Sensitivity = how a response changes when a parameter changes:**
`∂R/∂p`. apeGmsh can compute this for *any* parameter the model exposes —
**every damping channel included** — by **finite differences**: perturb the
parameter, re-run the analysis, difference the response. No solver edits, no
analytic derivative, no C++.

```
        p ── apeSees(fem) ──► OpenSees ──► Results ──► R(p)     (one forward solve)
   p + h ── apeSees(fem) ──► OpenSees ──► Results ──► R(p+h)
                                            ∂R/∂p ≈ ( R(p+h) − R(p−h) ) / 2h
```

## 2. Why this belongs in the apeGmsh layer

Two ways to get a sensitivity out of OpenSees:

| | Where it lives | Cost to build | Coverage |
|---|---|---|---|
| **DDM / analytic** (`getDampSensitivity`, …) | **C++ inside OpenSees** | per-element/material C++ work; several damping channels have **no** derivative at all | partial |
| **Finite differences** | **Python, on top of the solver** | a `forward(params)` wrapper | **every** channel — it never inspects the damping |

Finite differences is a **black-box driver on top of the forward solve** — which
is exactly what apeGmsh already *is* (build → `apeSees` → run → `Results`). So FD
sensitivity is apeGmsh's natural domain; DDM would be an OpenSees-core change.

The FD approach is proven and validated upstream in the Ladruno OpenSees fork
(bundle `Ladruno_implementation/damping_sensitivity/`, PR #241): all six damping
channels demonstrated, SDOF + viscous cases matching closed forms to **0.00%**,
and a 14-agent adversarial review confirmed the match is **non-circular**. This
guide ports that capability into the apeGmsh workflow.

## 3. The `forward(params)` contract

Everything hinges on one function you write:

```python
def forward(params) -> float:
    """Build + solve the model for these parameter values; return ONE scalar."""
```

The golden rule: **mesh once, re-run the analysis many times.** Geometry and
meshing (the expensive apeGmsh half) are done *outside* `forward`; inside, you
only rebuild the `apeSees` deck with the new parameter and re-run.

```python
from apeGmsh import apeGmsh, Results
from apeGmsh.opensees import apeSees

# ---- build the mesh ONCE (outside the FD loop) -------------------------
with apeGmsh(model_name="frame", save_to="frame.h5") as g:
    g.model.geometry.add_box(0, 0, 0, 6, 6, 12, label="col")
    g.physical.add_volume("col", name="Col")
    g.physical.add_surface([1], name="Base")
    g.physical.add_surface([2], name="Roof")
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)          # the solver contract


def forward(params):
    (xi,) = params                                    # damping ratio here
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    conc = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
    ops.element.FourNodeTetrahedron(pg="Col", material=conc)
    ops.fix(pg="Base", dofs=(1, 1, 1))
    ops.mass(pg="Roof", values=(m, m, m))

    # ---- the parameter under study: a damping knob -------------------
    ops.damping.rayleigh(ratio=xi, f_i=2.0, f_j=12.0)     # <-- perturbed each call

    # ---- excitation + a recorder for the response --------------------
    ts = ops.timeSeries.Trig(t_start=0.0, t_end=T, period=1.0 / f_drive)
    with ops.pattern.UniformExcitation(series=ts, dof=1) as p:
        pass
    ops.recorder.Node(pg="Roof", component="displacement_x", path="roof.out")  # see guide_recorders_reference.md

    ops.run()                                          # in-process openseespy
    ops.analyze(steps=NSTEPS, dt=DT)

    # ---- read ONE scalar back via Results ----------------------------
    results = Results.from_recorders(spec, "out/", fem=fem, model=ops.build())
    u = results.nodes.get(pg="Roof", component="displacement_x").values
    return float(abs(u).max())                         # peak roof drift
```

> Response read: `Results` is the apeGmsh-idiomatic path (label/PG selectors, not
> raw tags). For a tight FD loop an in-process `ops.domain_capture(spec, path=)`
> avoids file round-trips — see `guide_obtaining_results.md` /
> `guide_recorders_reference.md` for the exact recorder/capture signatures.

## 4. Driving it — the FD helper

The proven helper is `FDSensitivity` (from the fork bundle; a ~150-line,
OpenSees-agnostic class). It holds `forward` + step config, **memoizes** solves
(each is a full transient — the expensive thing), and **counts** them:

```python
from fd_sensitivity import FDSensitivity     # bundle: Ladruno_implementation/damping_sensitivity/

fd   = FDSensitivity(forward, rel_step=1e-2, scheme="central")
grad = fd.gradient([0.05])         # d(peak drift)/d(xi)
print(fd.step_study([0.05]))       # step-size plateau — ALWAYS check before trusting
print(fd.n_solves)                 # honest cost
```

**Always confirm the step-size plateau.** Too large = truncation error, too small
= round-off; the flat middle is the trustworthy gradient.

## 5. The damping-parameter menu (every channel is a knob)

FD is **source-agnostic**: point `forward`'s parameter at any `ops.damping.*` /
material / integrator knob. All six OpenSees damping channels are reachable:

| Channel | apeGmsh knob (set inside `forward`) | Notes |
|---|---|---|
| Rayleigh (raw) | `ops.damping.rayleigh(alpha_m=…, beta_k=…)` | global or `on="PG"` |
| Rayleigh (ratio fit) | `ops.damping.rayleigh(ratio=ξ, f_i=…, f_j=…)` | β lands as βK0 by default |
| Modal | `ops.damping.modal(ξ, modes=N)` | bundles its own eigen |
| Uniform / SecStif / URD objects | `ops.damping.uniform(ratio=ξ, freq_lower=…, freq_upper=…, on="PG")` | **input ζ ≠ realised ξ** (band-fit operator) |
| Viscous / dashpot material | `ops.uniaxialMaterial.Viscous(C=…, alpha=…)` | no DDM derivative exists — FD is the only no-touch route |
| Numerical (integrator) | `ops.integrator.HHT(alpha=…)` etc. | algorithmic dissipation, not a physical source — reachable, rarely wanted |

Two honest facts the upstream validation surfaced, worth carrying into apeGmsh:

- **`ops.damping.uniform` input `ratio` is not the realised SDOF ξ.** It fits a
  damping operator over `[freq_lower, freq_upper]`; at a single frequency the
  realised ξ_eff differs (e.g. input 0.05 → ξ_eff ≈ 0.029). FD differentiates the
  *input* knob correctly regardless — but interpret the result as ∂R/∂(input ζ),
  not ∂R/∂(realised ξ).
- **Validate with the parameter-free oracle `dU/dp = −U/p`** for resonant
  amplitude checks — it is exact whenever `U ∝ 1/p`, and (unlike per-channel
  closed forms) is immune both to algebra typos and to the input≠realised-ξ issue.

## 6. Multi-variable and cross-channel — one call, full gradient

`forward` can take a **vector** of parameters; `gradient` returns the whole vector
(cost `2N` solves central, `N+1` forward). The parameters can be **heterogeneous
and cross-channel** — per-component relative steps keep a damping ratio (~0.05)
and a stiffness (~3e10) on the same footing:

```python
def forward(params):
    aM, bK, E = params
    ops = apeSees(fem)
    ...
    ops.damping.rayleigh(alpha_m=aM, beta_k=bK)
    conc = ops.nDMaterial.ElasticIsotropic(E=E, nu=0.2, rho=2400)
    ...

fd.gradient([0.10, 0.001, 30e9])   # -> [∂R/∂αM, ∂R/∂βK, ∂R/∂E]
```

So: **scope one or many variables, get all their sensitivities at once.** What
this does *not* yet do is *solve for* parameter values (calibration/inversion) —
that is the gradient feeding an optimiser (`scipy.optimize`), a thin wrapper on
top (see §7). Nor does it return a full multi-response Jacobian — `forward` is
scalar-valued; many responses = loop the response (a clean extension).

## 7. Proposed first-class API (`apeGmsh.sensitivity`)

The `forward` boilerplate above is repetitive — building the deck, setting the
knob, recording, running, reducing. A first-class driver would hide it, in the
apeGmsh declarative style:

```python
# PROPOSED — not yet implemented
from apeGmsh.sensitivity import Sensitivity, Param, Response

sens = Sensitivity(
    fem,
    build   = build_deck,                      # callable(ops, params) -> None : declares the deck
    params  = [Param("damping.rayleigh.ratio", value=0.05, f_i=2.0, f_j=12.0)],
    response= Response(pg="Roof", component="displacement_x", reduce="peak"),
)
grad = sens.gradient()                         # ∂(peak roof drift)/∂(ξ)
sens.step_study()                              # plateau check
cal  = sens.solve(target=0.012)                # OPTIONAL: invert for ξ matching a target drift
```

Design notes:
- `Param` keys address `ops.damping.*` / material / integrator knobs by path, so
  the driver sets them generically — no per-channel code.
- `Response` reuses the `Results` selector grammar (`pg=`, `component=`) + a
  reducer (`peak` / `rms` / `at_time` / `energy`).
- `Sensitivity` wraps the proven `FDSensitivity` core; `solve()` hands its
  gradient to an optimiser for calibration/inversion.
- Mesh-once is enforced by construction (the `fem` snapshot is bound once).

## 8. Limitations (carried from the adversarial review)

FD assumes the response is a **smooth** function of the parameter:

- **Non-smooth functionals.** A `max(|u|)` peak-pick is only piecewise-smooth:
  under hysteresis / contact / yielding, or if the peak timestep switches between
  the `+h` and `−h` runs, the difference quotient is noisy. Linear/at-resonance
  cases (the validated ones) are smooth; nonlinear models want a smoother reducer
  (energy, RMS, response-at-fixed-time) or a wider step.
- **Steady-state proxy needs the transient dead.** "Peak over the last K cycles ≈
  steady amplitude" needs `ξ·2π·(n_cycles − K) ≫ 1`. Light damping (ξ ≲ 0.005)
  leaves residual transient that biases the amplitude high — re-check the window.
- **Sampling bias is benign.** Discrete `max` under-reads the crest by
  `≈cos(π·dt/period)`, a near-constant factor that **cancels in the FD ratio**
  (why the *gradient* error beats the *amplitude* error).
- **Cost is `N+1` solves per gradient.** Linear in the number of parameters —
  fine for a handful of damping knobs. For *many* parameters in a tight
  optimisation/reliability loop, that wall is where analytic DDM (for the channels
  that have it) or an adjoint earns its keep.

## 9. Provenance

The mechanics, validation, and adversarial review live in the Ladruno OpenSees
fork: `Ladruno_implementation/damping_sensitivity/` (helper `fd_sensitivity.py`,
four demos, Zone-A test `tests/test_fd_damping_sensitivity.py` — 11 passed), PR
#241. This guide is the apeGmsh-side port and API proposal.
