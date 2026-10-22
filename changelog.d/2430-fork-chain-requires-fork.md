### FIXED — every fork-only analysis-chain command stamps `requires=("fork",)`; on stock, a `Ladruno*` integrator no longer solves silently with the default integrator (#2430)

Stock openseespy 3.7.1.2 does not know the fork's `Ladruno*`, SMS and
Bathe integrators. It prints `WARNING unknown integrator type`, then
`analysis` solves with its default integrator, and the deck exits 0 with
the wrong answer: on a 1-DOF truss, `LadrunoLoadControl 0.1` gives
u = 0.01 on stock and 0.001 on the fork. Stock rejects outright
`numberer LadrunoParallelRCM`, `test LadrunoStabilizedUnbalance` and
`constraints LadrunoProjection` / `LadrunoContact`. Their archives all
stamped `/opensees@requires = ()`, which told a stock backend it could
solve the model.

- **One map.** `caps.FORK_CHAIN` (`{family: casefolded types}` over
  `system`, `integrator`, `numberer`, `test` and `constraints`) replaces
  `caps.FORK_SYSTEMS` and `caps.is_fork_system`. `caps.CHAIN_ATTRS` names
  the chain attribute each family is stored under. `caps.is_fork_chain`
  raises on a family it does not hold.
- **Emit.** `H5Emitter.solve_stamp` hands every analysis chain, global and
  staged, to `caps.value_requires(chains=...)`, which adds `"fork"` for a
  fork-only type of any family. `"fork"` was already a stamp token, so
  the schema does not change.
- **Read.** `caps.flat_requires(chains=...)` adds `"fork"` the same way.
  `build('live')` on stock refuses before it touches the domain, also for
  an archive written before this fix (`requires=()`) and for one with no
  solve stamp at all (below 2.24.0), which used to need nothing.
- **Pinned.** `_CHAIN_TOKENS` pins every member of the four families, and
  `FORK_CHAIN` must equal the types the bridge's fork-only members emit.
  The live net runs one deck per member on the bound build and checks
  that `build('live')` of each fork-only archive refuses on stock.
- **`LadrunoContact` is archived.** The H5 emitter used to drop that
  handler, so its archive stamped nothing. It is recorded now, and the
  stamp carries `"fork"`.
- **In-process gates.** `LiveOpsEmitter` refuses every fork-only chain
  type on stock: integrator, numberer, test, constraints and system, read
  from `FORK_CHAIN`. Before, it did not know the SMS integrators'
  consistent tokens (`CentralDifferenceSMSConsistent`,
  `ExplicitBatheSMSConsistent`, `ExplicitBatheLNVDSMSConsistent`).
  `apeSees.analyze` on stock then returned 0 with the default
  integrator's answer.
- **Live-net guard.** A deck whose stderr says `WARNING unknown <family>
  type` does not count as having run, even when it exits 0. A command the
  deck itself tries in `try:` (the ADR 0027 `Mumps` fallback) is exempt.
