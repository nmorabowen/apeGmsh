# Staged SSI — gravity first, then the absorbing flip

The [plane-wave SSI example](plane-wave-ssi.md) was a *pure-wave* skeleton:
no gravity, just a pulse from the base. A real soil model runs **gravity
first** — the soil settles under its own weight while the boundary holds
it, the in-situ state is frozen, and only *then* does the dynamic shaking
begin on top of that equilibrated state. This example adds that gravity
stage, which is the canonical soil–structure-interaction sequence:

1. **Gravity stage.** The absorbing skin is still in its *hold* state
   (a penalty support), so it pins the box while self-weight settles the
   column. `loadConst` freezes the result at `t = 0`.
2. **Flip + transient.** `s.activate_absorbing` switches the skin to
   *absorbing*, baking in the gravity displacement as its reference, and
   the base wavelet shakes the equilibrated column.

The interesting design question this example answers is **how to apply
self-weight to a continuum**, because OpenSees gives you two routes that
behave very differently.

## Two ways to load gravity — and why this example picks one

A continuum element (`stdBrick`, `FourNodeQuad`, …) takes an optional
`body_force` triple. It is tempting to use it for gravity:

```python
ops.element.stdBrick(pg="soil", material=soil,
                     body_force=(0.0, 0.0, -rho * 9.81))   # ← always on
```

But `body_force` is **applied every step, unconditionally** — it is *not*
a load pattern. Verified against the OpenSees source
(`Brick.cpp:1268-1274`, `FourNodeQuad.cpp:890-906`) and a live single-
element probe: the element integrates the constructor `b` into its
resisting force whether or not any `eleLoad` exists. That means
`body_force` gravity:

- cannot be **ramped** in over a few steps,
- cannot be **scaled** by a time series,
- is **not frozen** by `loadConst` — it is on from the first step of the
  first stage, through the transient, forever.

For a *staged* model that is the wrong tool: you want gravity to ramp in
during the gravity stage, hold, and then sit quietly (frozen) under the
dynamics. So this example uses the **pattern-controlled nodal route**
instead — author the weight at the geometry and import it into the
gravity stage's pattern:

```python
with g.loads.case("dead"):
    g.loads.gravity(res.soil_pg, density=rho, g=(0.0, 0.0, -9.81))
# ... later, in the bridge:
with ops.stage(name="gravity") as s:
    with s.pattern(series=grav_ts) as p:
        p.from_model("dead")        # ← pattern-controlled, loadConst-freezable
```

`g.loads.gravity` lumps each element's weight `ρ·V·g` onto its nodes.
For the structured hex boxes used here that tributary split is *exactly*
the consistent (Gauss-integrated) body-force vector — so it is not an
approximation, it is the same load `body_force` would assemble, just
under your control.

!!! warning "Don't do both"
    If you put `body_force=` on the brick **and** import a gravity case
    onto the same nodes, the region carries its self-weight **twice**.
    The bridge catches the collision and raises
    `WarnBodyForceDoubleCount` at build time. Pick one route.

## The whole model

```python
import os, sys
import numpy as np
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.material.nd import ElasticIsotropic
from apeGmsh.opensees.time_series.time_series import Ricker

# --- Problem data (consistent SI: m, N, Pa, kg, s) ---
rho, nu, Vs = 2000.0, 0.3, 200.0     # density, Poisson, target shear-wave speed
G = rho * Vs**2                      # 8.0e7 Pa
E = 2.0 * G * (1.0 + nu)             # 2.08e8 Pa
H, g_acc = 40.0, 9.81                # column depth [m], gravity [m/s^2]
traveltime = H / Vs                  # 0.20 s

f_n, t_center, dt, t_total = 4.0, 0.15, 0.002, 1.0
n_steps = round(t_total / dt)

surf, base_disp = "surf_vel.out", "base_disp.out"

# --- 1. Geometry: soil box + absorbing skin; author the gravity case ---
with apeGmsh(model_name="grav_ssi") as g:
    res = g.parts.add_plane_wave_box(x=(20.0, 2), y=(20.0, 2), z=(H, 16))
    with g.loads.case("dead"):
        g.loads.gravity(res.soil_pg, density=rho, g=(0.0, 0.0, -g_acc))
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data()

# --- 2. Bridge: soil brick (NO body_force) + absorbing skin ---
ops = apeSees(fem)
ops.model(ndm=3, ndf=3)

soil = ops.register(ElasticIsotropic(E=E, nu=nu, rho=rho))
base = ops.register(Ricker(f_n=f_n, t_total=t_total, dt=dt,
                           t_center=t_center, kind="velocity", factor=0.01))
grav_ts = ops.register(ops.timeSeries.Linear())

ops.element.stdBrick(pg=res.soil_pg, material=soil)
ops.element.absorbing_boundary(skin=res, material=soil,
                               base_series=base, base_dirs=("x",))

# --- 3a. Gravity stage: the skin HOLDS while self-weight settles ---
with ops.stage(name="gravity") as s:
    with s.pattern(series=grav_ts) as p:
        p.from_model("dead")                      # ramp gravity in
    s.recorder(ops.recorder.Node(
        file=base_disp, response="disp",
        pg=res.free_surface_pg, dofs=(3,), time_format="dt",
    ))
    s.analysis(
        test=ops.test.NormDispIncr(tol=1e-8, max_iter=30),
        algorithm=ops.algorithm.Newton(),
        integrator=ops.integrator.LoadControl(dlam=0.1),
        constraints=ops.constraints.Transformation(),
        numberer=ops.numberer.RCM(),
        system=ops.system.UmfPack(),
        analysis=ops.analysis.Static(),
    )
    s.run(n_increments=10)            # gravity -> full; loadConst freezes it

# --- 3b. Dynamic stage: FLIP the skin, then shake from the base ---
with ops.stage(name="dynamic") as s:
    s.activate_absorbing(pg=res.skin_all_pg)      # hold -> absorbing
    s.recorder(ops.recorder.Node(
        file=surf, response="vel",
        pg=res.free_surface_pg, dofs=(1,), time_format="dt",
    ))
    s.analysis(
        test=ops.test.NormDispIncr(tol=1e-8, max_iter=20),
        algorithm=ops.algorithm.Newton(),
        integrator=ops.integrator.Newmark(gamma=0.5, beta=0.25),
        constraints=ops.constraints.Transformation(),
        numberer=ops.numberer.RCM(),
        system=ops.system.UmfPack(),
        analysis=ops.analysis.Transient(),
    )
    s.run(n_increments=n_steps, dt=dt)

ops.py("grav_ssi.py", run=True)        # emit the openseespy deck and run it

# --- 4. Read both outputs, run the checks ---
settle = np.loadtxt(base_disp)[-1, 1:].mean()          # vertical, after gravity

data = np.loadtxt(surf)
t = data[:, 0]
v = data[:, 1:].mean(axis=1)
absv = np.abs(v)
peak = absv.max()
t_arr = t[int(np.argmax(absv > 0.05 * peak))]
late = float(absv[t >= 0.7].max())

print(f"surface settlement (gravity) = {settle:.3e} m")
print(f"H/Vs            = {traveltime:.3f} s")
print(f"first arrival   = {t_arr:.3f} s")
print(f"late max (>0.7) = {late:.3e} m/s   ({late/peak:.2%} of peak)")
```

Run it. You should see:

```
surface settlement (gravity) = -5.606e-02 m
H/Vs            = 0.200 s
first arrival   = 0.198 s
late max (>0.7) = 6.488e-06 m/s   (0.93% of peak)
```

Three things land:

1. **Gravity settled the column.** The free surface drops 5.6 cm under
   self-weight, with the absorbing skin holding the boundary — the
   in-situ state a real analysis needs.
2. **The wave still arrives at $H/V_s$.** First motion shows up at
   0.198 s after the flip, matching the shear traveltime — the gravity
   stage did not disturb the wave physics.
3. **It still radiates.** Late-window surface motion holds at 0.93 % of
   peak, identical to the pure-wave run: `loadConst` froze gravity as a
   static offset and the linear dynamic response rides on top, undisturbed.

## Why the skin has to hold during gravity

![The settled column at the end of the gravity stage — 5.6 cm of vertical displacement, skin holding](../assets/tut/staged-ssi-settlement.png)


This is the whole reason the absorbing layer is *extra* material outside
an intact soil box (ADR 0054). In its **hold** state the
absorbing element behaves as a fixed support with **zero** gravity mass —
so it pins the truncation boundary while the interior settles, without
itself contributing spurious weight. At the flip, `s.activate_absorbing`
bakes the current displacement and the frozen stage-0 reactions into the
element's reference state, then switches on the dashpots. Settle, freeze,
flip, shake — in that order.

## Going further — a true geostatic stress

This example reaches its in-situ state by *settling* an elastic column.
For a nonlinear soil you usually want to **install** the geostatic stress
directly rather than compress your way into it, which is what
`s.initial_stress(...)` does (parameter-ramped $\sigma$, ADR 0028) — add
it to the gravity stage alongside the `from_model` weight. And a structure
on the surface ties or embeds against `res.free_surface_pg`, driven by the
radiated motion. The skeleton here is the foundation those build on.

---

*Next: [Moment-tensor seismic source](moment-tensor-source.md).*
