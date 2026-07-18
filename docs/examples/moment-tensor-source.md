# Moment-tensor seismic source (ADR 0062)

Embed an earthquake **source** inside a solid mesh and let the solver radiate
it. A seismic point source is a moment tensor $M_{ij}$; by the representation
theorem its radiated field equals that of an equivalent body force, which on
the FE basis is the **consistent nodal force**

$$
F^a_i(t) = M_{ij}\,\frac{\partial N_a}{\partial x_j}\bigg|_{\xi}\,S(t),
\qquad M_{ij} = M_0\,m_{ij},
$$

with $m_{ij}$ the unit double-couple from strike/dip/rake, $M_0=\mu A\bar D$
the scalar moment, and $S(t)$ the **normalized moment function** (rising
$0\to1$). This is a pure right-hand-side load — no new element, no constraint,
integrator-agnostic — so it drops straight into an explicit or implicit run.

This page validates it: a double-couple in a box, run as a transient, must
radiate the textbook **four-lobe double-couple pattern**.

## A single double-couple

```python
import numpy as np
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.material.nd import ElasticIsotropic

L = 40.0
ctr = (L / 2, L / 2, L / 2)

g = apeGmsh(model_name="mt_source", verbose=False)
g.begin()
g.model.geometry.add_box(0, 0, 0, L, L, L, label="soil")
g.physical.add(3, "soil", name="soil")
g.mesh.structured.set_transfinite("soil", n=17)
g.mesh.generation.generate(dim=3)
fem = g.mesh.queries.get_fem_data()
g.end()

ops = apeSees(fem)
ops.model(ndm=3, ndf=3)
soil = ops.register(ElasticIsotropic(E=2.0e8, nu=0.25, rho=2000.0))
ops.element.stdBrick(pg="soil", material=soil)

# The pattern's time series IS the moment function S(t): apply S(t),
# never the slip-rate. MomentStep is a smooth 0->1 ramp (Gaussian rate).
dt, t_total = 5e-4, 0.12
S = ops.timeSeries.MomentStep(half_duration=0.008, t_total=t_total, dt=dt, t0=0.02)
with ops.pattern.Plain(series=S) as p:
    p.moment_tensor(
        position=ctr,
        frame="z-down",                 # REQUIRED: mesh vertical convention
        M0=1.0e8,                        # scalar moment in deck units
        mech=dict(strike=0, dip=90, rake=0),   # vertical strike-slip
        method="consistent",            # host ∂N/∂x (any solid mesh)
        region="soil",                   # restrict the host to the soil PG
    )
```

`frame` is mandatory — `"z-up"` (typical apeGmsh: free surface on top, depth
positive downward) flips the Aki & Richards z-down tensor; getting it wrong
silently mirrors the radiation pattern. `region="soil"` confines the source to
the named continuum so a point in an absorbing skin (or outside the region)
**fails loud** instead of sourcing into boundary cells. Pass `m_ij=<3×3>`
instead of `mech=` to supply an explicit unit tensor.

## Running it and checking the radiation pattern

Drive a transient and read the P-wave **first-motion radial velocity** on a
ring of receivers around the source. For a vertical strike-slip,
$m_{ij}=\begin{psmallmatrix}0&1&0\\1&0&0\\0&0&0\end{psmallmatrix}$ and the
horizontal P radiation is $A_P(\hat\gamma)=\hat\gamma\cdot M\cdot\hat\gamma
\propto \sin2\theta$ — four lobes with nodal planes along $x$ and $y$.

```python
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh._kernel.geometry._moment_tensor import unit_moment_tensor

ops.constraints.Plain(); ops.numberer.RCM(); ops.system.UmfPack()
ops.test.NormDispIncr(tol=1e-8, max_iter=10); ops.algorithm.Linear()
ops.integrator.Newmark(gamma=0.5, beta=0.25); ops.analysis.Transient()

emitter = LiveOpsEmitter(wipe=True)
ops.build().emit(emitter)
o = emitter.ops

# receivers on a horizontal ring at the source depth
ids = np.asarray(fem.nodes.ids); coords = np.asarray(fem.nodes.coords, float)
ring, dirs = [], []
for a in np.linspace(0, 2*np.pi, 24, endpoint=False):
    tgt = np.array(ctr) + np.array([10*np.cos(a), 10*np.sin(a), 0.0])
    j = int(np.argmin(np.linalg.norm(coords - tgt, axis=1)))
    ring.append(int(ids[j])); dirs.append((coords[j]-ctr)/np.linalg.norm(coords[j]-ctr))

vr = {n: [] for n in ring}
for _ in range(int(t_total/dt)):
    o_rc = emitter.analyze(steps=1, dt=dt)
    for n, d in zip(ring, dirs):
        vr[n].append(np.array([o.nodeVel(n, k) for k in (1, 2, 3)]) @ d)

fem_amp = np.array([max(vr[n], key=abs) for n in ring])
M = unit_moment_tensor(strike=0, dip=90, rake=0)
analytic = np.array([d @ (M @ d) for d in dirs])     # γ·M·γ
corr = np.corrcoef(fem_amp/np.abs(fem_amp).max(),
                   analytic/np.abs(analytic).max())[0, 1]
print(f"radiation-pattern correlation = {corr:.3f}")   # ~0.90 (coarse) .. 0.97
```

The FEM first-motion amplitude tracks $\hat\gamma\cdot M\cdot\hat\gamma$:
positive in two opposing quadrants, negative in the other two, with minima on
the nodal planes — the double-couple signature. The run-verified regression
lives in `tests/opensees/live/test_moment_tensor_radiation_live.py`.

!!! note "Resolution"
    Sharper lobes need the receiver ring in the far field ($r\gg\lambda$) and
    $\ge 6$ elements per wavelength; near-field terms keep the nodal planes
    from going fully to zero at modest resolution (the correlation still
    exceeds 0.9). Band-limit `S(t)` to the mesh with `MomentStep(..., f_max=)`.

## Finite faults

A finite fault is a loop of point sources, each with its own location, moment,
mechanism, and rupture onset. `ops.fault.from_shakermaker` ingests a
ShakerMaker `FaultSource` and emits **one pattern per source** — each a
`Yoffe` moment function shifted to that source's rupture time:

```python
patterns = ops.fault.from_shakermaker(
    fault,                              # a ShakerMaker FaultSource (duck-typed)
    frame="z-up", region="soil",
    M0=6.3e15, rise_time=1.0, peak_time=0.2,
    dt=dt, t_total=t_total, length_scale=1000.0,   # km -> m
)
```

It imports nothing from ShakerMaker (the interchange is plain data) and reads
each `PointSource`'s `.x` (km), `.angles` (radians, converted to degrees), and
`.tt` (rupture onset). Subfaults whose onset falls outside `t_total`, or whose
`peak_time` must be clamped for the Yoffe bound, warn rather than silently
distort the source.

## Validating against ShakerMaker FK (procedure)

ShakerMaker's frequency–wavenumber (FK) solution is the exact 1-D oracle for a
point double-couple in a layered crust. Because the FK stack and OpenSees live
in separate environments, the overlay is run cross-venv and compared offline:

1. **FK reference** (in `shakermaker_venv`): build the same crust as a
   `CrustModel`, the same `PointSource(x, [strike, dip, rake], tt, stf)`, and
   `Station`s at the surface receiver locations; run `ShakerMaker(...).run()`
   and save the surface velocity traces.
2. **FEM** (in `opensees_venv`): build the crust as soil layers in an
   `add_plane_wave_box` (absorbing skin on the five truncation faces, free
   surface on top), embed the same source with `p.moment_tensor`, run the
   transient, and record the same receivers.
3. **Compare**: FEM-vs-FK surface velocity within a few percent of peak at
   $P\ge6$, the four-lobe radiation pattern in first-motion amplitude vs
   azimuth, and a quiet late-time signal at the absorbing base.

The radiation-pattern check above is the self-contained half of gate (b)/(a);
the full layered-crust FK waveform overlay is the acceptance test to run where
ShakerMaker is installed.

---

*Next: the ladder ends here; the [how-to recipes](../how-to/index.md) cover the rest.*
