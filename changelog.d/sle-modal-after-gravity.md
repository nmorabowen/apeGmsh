### ADDED — `modal_after_gravity`: modes on the gravity-preloaded tangent, on one live domain (ADR 0124)

`apeSees.modal_after_gravity(num_modes, *, p_delta=True, solver="-genBandArpack")` runs the declared
`Static` + `LoadControl(dlam=1/N)` chain to `λ = 1` (every registered pattern is the gravity preload),
`loadConst -time 0`, then `eigen` and `modalProperties -return` on the same domain, so the modes see
`K - Kg` (P-Delta the ETABS "iterative based on loads" way). The other modal drivers build a fresh,
unloaded domain and see `K` only. `p_delta=False` solves the modes before the preload, on `K`, and runs
the same gravity analysis afterwards. The `GravityModalResult` carries the `ModalPropertiesResult` and a
`GravityState` snapshot (every node's displacement and reaction after `loadConst`); the driver raises if
the eigen solve moved it, and `gravity_drift()` re-checks it. An adaptive or non-`LoadControl`
integrator, a transient analysis and a staged model are refused. Checked against the exact discrete
reference of a PDelta cantilever (`T/T0` to `1e-9` at `P/Pcr` = 0.25, 0.5, 0.75) and a `ShellNLDKGQ`
wall strip at `0.5 Pcr` (`T/T0` = 1.4087, continuum 1.4094, `sqrt(2)` 1.4142), on stock and fork.
`ops.element.ShellNLDKGQ` joins the typed shells: its tangent carries the initial-stress stiffness.
