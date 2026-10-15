### FIXED — upstream shells under `algorithm Linear`: the emit warns, DomainCapture keeps Newton data (program slice #2001)

After a step solved with `algorithm Linear`, ShellMITC4, ShellDKGQ, ShellDKGT, ShellNLDKGQ and ShellNLDKGT
answer every section probe from state that never saw the converged displacement: zero at step 1, then the
previous step's state (or zero again for the NL shells). The `.out`, MPCO and `.ladruno` recorders wrote
those values with no error. A bridge emit that solves with Linear now raises a `StaleShellProbeWarning`,
once per recorder record and class, when a recorder reads such a shell's stresses, strains or section
response (and, for the NL shells, their nodal forces, which are also stale). The warning names the fix:
solve with Newton, which converges in one iteration on a linear problem. DomainCapture now skips these
classes only under Linear or an unknown algorithm, and `ops.domain_capture` passes the declared one, so
under Newton it records them with the closed-form values instead of discarding them. The skip warning no
longer says that MPCO is correct.
