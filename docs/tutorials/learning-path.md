# The learning path

This page is the whole staircase on one screen: every tutorial and
worked example in the order they are meant to be read, what each one
teaches, and the known answer it is checked against. If you ever wonder
"where am I, and what should I read next?" — the answer is here.

The rules of the ladder are simple. The four **tutorials** are
hand-held: one path, no forks, and each ends in a number you can verify
by hand. The **examples** are recognizable structural problems that
each introduce one new capability on top of the rungs before them. You
don't need all of them — climb until you reach your own problem, then
switch to the [how-to recipes](../how-to/index.md) for the tasks that
remain.

## Tutorials — the guaranteed path

| | Tutorial | You learn | Checked against |
|---|---|---|---|
| 1 | [Your first model in 10 minutes](first-model.md) | the whole spine in under 40 lines: geometry → mesh → typed `apeSees` bridge → solve → results in the browser | tip deflection $\delta = PL^3/3EI$ |
| 2 | [A plate in tension](plate-in-tension.md) | the same bridge on a 2D solid: `nDMaterial`, elements by physical group, reading a field, a contour | $u_x = \sigma L / E$ |
| 3 | [A simply-supported beam, the apeGmsh way](beam-and-composites.md) | the composites — `g.loads`, `g.masses`, `ops.section` — declare on the geometry, resolve at the bridge | $\delta = 5wL^4/384EI$, $M = wL^2/8$ |
| 4 | [Save, reload, view](save-reload-view.md) | persistence (`save_to` / `from_h5`) and the notebook-safe results loop | the reloaded model reproduces tutorial 3's deflection |

## Examples — the ladder

Each example names the rung it builds on. Same bridge, same read-side
API throughout — every rung adds exactly one idea.

| | Example | Builds on | The new idea | Checked against |
|---|---|---|---|---|
| 1 | [Portal frame (2D)](../examples/portal-frame.md) | tutorial 4 | several element groups, gravity + lateral patterns, drift | base shear = applied load, exactly |
| 2 | [Modal analysis (cantilever)](../examples/modal-analysis.md) | portal frame | `g.masses` + `ops.eigen`, modes through `Results` | Euler–Bernoulli frequencies |
| 3 | [Fiber sections & moment–curvature](../examples/fiber-moment-curvature.md) | modal analysis | `uniaxialMaterial` + fiber sections (via apeSteel) | $M_y = F_y S_x$, $M_p = F_y Z_x$, shape factor 1.10 |
| 4 | [Multi-part assembly](../examples/multipart-assembly.md) | portal frame | one `Part`, stamped three times, addressed by label | each copy exactly $PL^3/3EI$ |
| 5 | [Tie non-matching meshes](../examples/tie-non-matching-meshes.md) | multi-part assembly | `g.constraints.tie` auto-emitted by the bridge | matches the monolithic bar to ~3% |
| 6 | [STEP import: plate with a hole](../examples/step-plate-with-hole.md) | plate in tension | CAD import, healing, naming edges by query | stress concentration $K_t \approx 3$ |
| 7 | [Choosing a results strategy](../examples/results-strategies.md) | tutorial 4 | the same model read via `from_native` and `from_mpco` | both agree on the drift to zero |
| 8 | [Compose modules](../examples/compose-modules.md) | tie + assembly | `g.compose` — build once, save, import twice | each bay drifts the portal frame's exact answer |
| 9 | [Pushover of a steel moment frame](../examples/pushover-steel-frame.md) | fiber sections | DisplacementControl to a sway mechanism | $K = 2\cdot12EI_c/H^3$ (2.7%), $V_p = 4M_p/H$ (1.9%) |
| 10 | [Plane-wave SSI: absorbing soil column](../examples/plane-wave-ssi.md) | modal analysis | the absorbing-boundary skin and a propagating wave | arrival at $H/V_s$; late motion < 1% of peak |
| 11 | [Staged SSI: gravity then the absorbing flip](../examples/staged-gravity-ssi.md) | plane-wave SSI | `ops.stage` — settle under gravity, freeze, then shake | 5.6 cm settlement, then clean radiation |
| 12 | [Moment-tensor seismic source](../examples/moment-tensor-source.md) | staged SSI | a double-couple source embedded in the solid mesh | the analytic double-couple radiation pattern |

More rungs (shell-on-solid, transient response-history) are landing;
each will slot into this table when it does.

---

*Next: start the climb with
[your first model in 10 minutes](first-model.md).*
