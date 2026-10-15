### FIXED — the upstream-shell "zero probe" note names its real cause, algorithm Linear (#1992)

`ZERO_GAUSS_PROBE_CLASSES` said ShellMITC4, ShellDKGQ, ShellNLDKGQ, ShellDKGT
and ShellNLDKGT return zeros through `eleResponse`. Measured on stock
openseespy 3.7.1 and the Ladruno fork, that holds only after a step solved
with `algorithm Linear`, which never re-forms the residual, so the sections
keep stale state. Step 1 then reads exactly zero; from step 2 on the
linear-kinematics shells read the previous step's state. `eleResponse`, a
`.out` Element recorder and (on the fork) MPCO all agree, so a Linear run
records these values silently on every route. Under Newton every route is correct.

The note and the `has_dead_gauss_probe` docstring now say this. Behaviour is
unchanged: DomainCapture still skips these classes with a warning, which keeps
the Linear case loud but also drops valid data under Newton. A strict-`xfail`
`live` test holds the Newton answer until the capture call site can pass the
algorithm. New `live` tests in `tests/test_results_shell_resultant_vocabulary.py`
pin each route under both algorithms against the closed form N11 = 2·P·λ.
