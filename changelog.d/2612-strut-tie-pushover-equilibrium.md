### FIXED — `strut_tie_pushover` checks equilibrium on every step and refuses a run it cannot vouch for, instead of returning a diverged capacity (#2612, #2602)

`strut_tie_pushover` used to return `capacity = max(P)` over every recorded
point, whatever `stopped` said. Its convergence test, `NormDispIncr`, bounds
the last correction and not the residual, and the fallback algorithms ran at
100 times that tolerance. A step could therefore be accepted out of
equilibrium, and a run that stopped converging still returned its peak with
`stopped="divergence"`. The #2602 diagnosis found the Cook and Mitchell
corbel peak moving between 287 and 385 kN per side with the MKL code path,
and between 318 and 414 kN with a 1e-12 change in `ft`.

Every accepted step is now checked for equilibrium. The 2-norm of the
unbalanced force over the free DOFs, read from `reactions()`, must not
exceed `equilibrium_tolerance × |λ|·|P|`, the load the curve reports at that
step. The new keyword `equilibrium_tolerance` defaults to `5e-3`
(`EQUILIBRIUM_TOLERANCE_0_005`), the 0.5 % force tolerance of Abaqus and
ANSYS. It must be positive.

A step that fails the check, or that does not converge after
`max_halvings` halvings, raises `PushoverRefusedError`, and no result
object is returned. The error names the step, the unbalanced force and the
tolerance, and carries `.points`, the curve up to the last step in
equilibrium. A returned curve therefore stops at `"target"` or
`"post_peak"`; `"divergence"` is no longer returned. `fe_curve` gains an
additive `equilibrium` block, `{"tolerance", "max_relative_unbalance"}`.
A converged run keeps every existing key. The CLI
(`python -m apeGmsh.interop.strut_tie --pushover`) now exits with the error
instead of writing a curve.

At the pinned fork, `LadrunoConcrete3D`'s return map fails silently: it
returns 0 with the trial stress. A `LadrunoConcrete3D` pushover past
cracking is therefore refused. The Cook and Mitchell corbel test now expects
that refusal. Its bracket, 0.7 × 471 kN < per side < 1.1 × 502 kN with
`stopped == "post_peak"`, is unchanged, and returns when the fork's return
map fails loudly (#2602 route 2). The `ASDConcrete3D` corbel still reaches
its plateau of about 600 kN per side.
