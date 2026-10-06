### ADDED — `FORK_PIN` and a nightly `live-fork` lane (program slice F2-a, #1497)

`FORK_PIN` names the Ladruno fork build apeGmsh is tested against. The
non-required `live-fork` workflow downloads that release asset nightly and
runs the `ladruno_fork` tests, writing pass/fail/skip counts (and the
`ladruno_mkl` skip count) to the job summary. Under
`APEGMSH_FORK_PIN_ENFORCE=1` a build that differs from the pin skips every
`ladruno_fork` test with the pinned and imported shas; local runs are
unaffected. New `ladruno_mkl` marker for tests that need Pardiso.
