### FIXED — a `system Pardiso` archive stamps `requires=("fork",)`, so `build('live')` on stock refuses it instead of aborting (#2393)

`system Pardiso` exists only on the Ladruno fork. A one-process solve on
stock openseespy 3.7.1.2 rejects the command, and the fork 79e06 solves it.
Its archive still stamped `/opensees@requires = ()`, which claimed that a
stock backend could solve it, the same false claim as #2220.

- **Emit.** `caps.value_requires` adds `"fork"` when the analysis chain,
  global or staged, declares a fork-only system (`caps.is_fork_system`;
  `caps.FORK_SYSTEMS` is `{"pardiso"}`). `"fork"` was already a stamp
  token, so the schema does not change.
- **Read.** `caps.flat_requires` adds `"fork"` for a stored fork-only
  system. An archive written before this fix, whose stamp says
  `requires=()`, still fails closed: `build('live')` on stock raises a
  `BridgeError` naming `'fork'` before it touches the domain.
- **Pinned.** The audit was repeated on both builds. Every LinearSystem
  member's tokens are pinned in `_SYSTEM_TOKENS`. The live net now runs
  each member's deck on the bound build and checks it against those
  tokens.

Not fixed here: several fork-only analysis commands still stamp no
`"fork"`. They are the `Ladruno*` integrators, the SMS and Bathe
integrators, `numberer LadrunoParallelRCM`,
`test LadrunoStabilizedUnbalance`, and `constraints
LadrunoProjection` / `LadrunoContact`. Stock openseespy treats them in
one of two ways. It rejects the numberer, the test and both handlers.
It accepts an unknown integrator with a warning and solves with its
default instead. Deriving their token needs the chain attributes at the
stamp writer, in `emitter/h5.py`, so it is a follow-up (#2430).
