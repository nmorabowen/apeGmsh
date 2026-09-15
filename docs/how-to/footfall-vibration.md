# Evaluate footfall vibration

Check a floor or footbridge against walking-induced vibration — AISC Design
Guide 11, 2nd edition, Chapter 7. Reach for this once you have a modal
model (see [Run a modal analysis](../examples/modal-analysis.md)) and want
to know whether people walking on it will feel it.

## The bay

One steel floor beam, simply supported, with the tributary floor mass
spread along the span as **distributed element mass** — not lumped at one
point — small enough to run in seconds, large enough to show a real
pass/fail with a modal basis that actually covers the Guide's 20 Hz band.

```
  Left (pin)                Quarter          Midspan           Right (roller)
     o───────────────────────o───────────────────o───────────────────o
     |<---------- 6 m span (one bay) ------------------------------->|

  Section: composite steel beam, Iz = 3.5e-4 m^4 (bare steel ~1.2e-4,
           boosted for composite action with the slab — see below)
  Mass:    500 kg/m distributed (the bay's tributary floor mass per unit
           length: slab + ceiling + partitions + a share of live load) —
           puts the first mode at ~16 Hz, a sensible occupied-floor
           frequency, and the second at ~65 Hz, comfortably clear of the
           20 Hz band the Guide's Eq 7-5 sum runs over.
```

```python
from apeGmsh import apeGmsh, Results
from apeGmsh.opensees import apeSees

L, E, Iz, A = 6.0, 200e9, 3.5e-4, 0.0083   # SI: m, Pa, m^4, m^2
mass_per_len = 500.0                        # distributed floor mass, kg/m

with apeGmsh(model_name="footfall-bay") as g:
    p0 = g.model.geometry.add_point(0.0, 0.0, 0.0)
    pq = g.model.geometry.add_point(L / 4, 0.0, 0.0)
    pm = g.model.geometry.add_point(L / 2, 0.0, 0.0)
    p1 = g.model.geometry.add_point(L, 0.0, 0.0)
    l0 = g.model.geometry.add_line(p0, pq)
    l1 = g.model.geometry.add_line(pq, pm)
    l2 = g.model.geometry.add_line(pm, p1)
    g.model.sync()

    g.physical.add(1, [l0, l1, l2], name="Beam")
    g.physical.add(0, [p0], name="Left")
    g.physical.add(0, [p1], name="Right")
    g.physical.add(0, [pq], name="Quarter")
    g.physical.add(0, [pm], name="Midspan")

    g.mesh.sizing.set_global_size(L / 12.0)   # L/8 or finer — 12 here
    g.mesh.generation.generate(1)
    fem = g.mesh.queries.get_fem_data(dim=1)

ops = apeSees(fem)
ops.model(ndm=2, ndf=3)
transf = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
ops.element.elasticBeamColumn(
    pg="Beam", transf=transf, A=A, E=E, Iz=Iz, mass=mass_per_len,
)   # lumped element mass by default — do not pass c_mass=True (see below)
ops.fix(pg="Left", dofs=(1, 1, 0))
ops.fix(pg="Right", dofs=(0, 1, 0))

midspan = ops.nodes.get(pg="Midspan")
quarter = ops.nodes.get(pg="Quarter")

result = ops.footfall_walking(
    num_modes=2,         # highest extracted mode (~65 Hz) clears f_max=20 Hz
    body_weight=747.0,   # Q = 168 lb (the Guide's recommended walker weight)
    g=9.81,
    response_nodes=midspan,
    dof=2,               # vertical translation in this 2-D frame
    occupancy="office",
    damp=0.03,           # must be > 0 — an undamped FRF has no finite
                         # resonant peak, and the driver refuses it
)

print(result.f_dom, result.a_p, result.regime, result.limit, result.ratio)
# -> [16.326] [0.012007] ['high'] [0.010204] [1.177]

result.to_results(fem, "footfall_bay.h5")
Results.from_fem(fem, "footfall_bay.h5", kind="native").viewer()
```

That prints a dominant frequency of **16.33 Hz**, a governing acceleration
of **1.20 %g**, and a **ratio of 1.177** against the office limit — this
bay fails, and by enough that stiffening the section or adding damping is
worth considering. Here is what each part is doing.

## Modelling the bay (§7.2)

The Guide asks for the floor's **actual in-place mass** (not a code live
load) and a **composite-boosted stiffness** — cracked-section bending
underestimates how stiff a slab-on-beam floor really is, so §7.2 has you
transform the slab into the beam at `1.35 Ec` and use the resulting
composite `I`. Here that shows up as a single number: `Iz = 3.5e-4 m^4`
against a bare-steel `1.2e-4 m^4` — do the transformed-section calc for
your own beam and section.

## `num_modes` and the 20 Hz warning

`footfall_walking` builds its own live domain and issues one `eigen` +
`modalProperties` pair for `num_modes` modes — it does not need a prior
`ops.analyze()` call. The high-frequency branch (Eq 7-4…7-6) sums over
**every** mode up to `f_max` (20 Hz by default), so a basis that stops
short warns. Drop the worked example above to `num_modes=1` and the
highest (and only) extracted mode is 16.33 Hz, still under `f_max=20.0`:

```
UserWarning: apeSees.footfall_walking: the highest extracted mode is
16.326 Hz, below f_max=20.0 Hz — the Eq 7-5 impulse sum is missing modes
it should carry. Raise num_modes, or on the fork use eigen_feast(...) to
get the band exactly.
```

That basis is genuinely short: the Eq 7-5 sum needs every mode up to
20 Hz, and this beam's second mode (~65 Hz) sits well outside it — so the
sum evaluated with `num_modes=1` is missing a real contribution, not a
spurious one. `num_modes=2` (used above) clears the warning because its
highest extracted mode is already past `f_max`. On a real floor, raise
`num_modes` until the basis covers 20 Hz, or use the fork's band solve
(`eigen_feast`) to get it exactly.

`footfall_walking` defaults to `solver="-genBandArpack"`; leave it at the
default here. On this build `-genBandArpack` returns unit-modal-mass
eigenvectors under both lumped and consistent element mass, while
`-fullGenLapack` does not under consistent mass and is correctly refused
by the ADR 0109 D2 eigenvector-scale check.

## `body_weight` and `g`

Both are **required**, in the model's own units — apeGmsh has no unit
system, and a silently wrong `g` is a factor of 386 in the answer. The
Guide's recommended walker weight is **Q = 168 lb ≈ 747 N**; this model is
SI, so `body_weight=747.0, g=9.81`. In kip-inch units you'd pass
`body_weight=0.168, g=386.1` instead — same walker, different numbers.

## `excitation="self"` vs `"full"`

`"self"` (the default) evaluates the diagonal: the walker stands exactly
where the occupant sits — the Guide's conservative default, both at
mid-bay. `"full"` checks every excitation/response pair and reports which
walker governs:

```python
full = ops.footfall_walking(
    num_modes=2, body_weight=747.0, g=9.81,
    response_nodes=midspan, excitation="full",
    excitation_nodes=(*midspan, *quarter),
    dof=2, occupancy="office", damp=0.03,
)
full.exc_node   # -> [3] — Midspan itself, not Quarter
```

Here the occupant's own position still governs even under `"full"`,
because this bay's dominant mode shape peaks at midspan — the walker
standing anywhere else excites it less. On a floor where a different mode
dominates at a different response node, a different excitation node can
win, and `exc_node` tells you which.

## Reading the result

Per response node, `result.a_p` is the governing acceleration (the larger
of `a_p_lf`, Eq 7-1, and `a_espa_hf`, Eq 7-4…7-6) as a **fraction of g**,
`result.regime` says which branch produced it (`"low"` / `"high"` /
`"both"`), and `result.ratio = a_p / limit` is the demand/capacity check —
below 1 is acceptable. `result.limit` is read at `f_dom` on the Fig 2-1
curve by default (`limit="curve"`), which is why it comes out at 1.02 %g
here rather than the flat 0.5 %g office value: the curve rises above 8 Hz,
and this bay's dominant frequency (16.33 Hz) sits well into that rise.

## Rendering the ratio map

`to_results` writes one native results file — one frame, three nodal
components (`footfall_ap`, `footfall_ratio`, `footfall_fdom`), `NaN`
outside the response nodes — that `Results.from_fem` binds like any other
native run:

```python
result.to_results(fem, "footfall_bay.h5")
r = Results.from_fem(fem, "footfall_bay.h5", kind="native")
r.nodes.get(component="footfall_ratio").values   # (1, 13), 1.177 at Midspan, NaN elsewhere
r.viewer()                                        # ordinary nodal-scalar map — no recorder involved
```

On a real floor with many response nodes this is what turns a table of
per-node ratios into a picture: red where the floor fails, green where it
doesn't. Checked here headless with `r.render("ratio.png", view="contour",
component="footfall_ratio")`: `from_fem` binds a real FEM broker off the
same `fem` object with no extra wiring, and the nodal-scalar map draws
correctly with the NaN nodes off the response set — VTK's default
autoscaling ignores them rather than blowing up the color range.

## Why the 2nd edition, not Robot

Robot's own *Footfall* case is built on the Design Guide's **1st**
edition — resonant-only under its AISC option, with no equivalent to the
high-frequency effective-impulse branch this driver evaluates as
`a_espa_hf`; ADR 0109 has the full comparison.

## See also

- Example: [Modal analysis of a cantilever](../examples/modal-analysis.md) — mass, `eigen`, and reading modes back, the ingredients this recipe builds on.
- ADR: `architecture/decisions/0109-footfall-vibration-frf-method.md` — the FRF method, the eigenvector-scale assertion, and why Robot isn't the oracle.
- API: `apeGmsh.opensees.apeSees.footfall_walking`, `apeGmsh.opensees.analysis.footfall_result.FootfallResult`.

---

*Next: [Read a node's displacement and reactions](read-results.md).*
