### FIXED — Assembly v2 instances a source with `/interfaces` or a point group (#1586, #1587)

`Assembly.bridge()` refused a source that carries `g.constraints.interface`
records, because the rehydrator met the build-synthesised `ENT` uniaxial.
Each record's `zeroLength` row and its tributary uniaxials are now left to
the build, which synthesises them again from the carried `/interfaces`
stream; a unit that does not match its record exactly still raises
`AssemblyError`. A source with a dim-0 (point) physical group no longer
crashes `bridge()` with a raw `ValueError`: point groups carry no element
spec and travel as `{instance}.{pg}`.
