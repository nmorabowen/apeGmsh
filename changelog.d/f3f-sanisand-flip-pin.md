### FIXED — SANISAND bit-identity test pins `-flipAlphaIn vanilla`; the `init` default gets an effect test (#1523, #1680)

The `LadrunoSANISAND` against `ManzariDafalias` bit-identity test (I1 in
`tests/opensees/integration_ladruno/test_ladruno_sanisand_live.py`) failed on
every fork build because the fork's default `-flipAlphaIn init` (WP-112,
fork `25136af9c`) sets `alpha_in := alpha` at the stage flip, which
`ManzariDafalias` does not do. That default is intended (#1523), so I1 now
pins `flip_alpha_in="vanilla"`, keeps its exact assertion, and loses its
Linux `xfail`. A new effect test, I5, flips at a sheared state (norm of
`alpha` 0.347) and asserts what the two rules predict: under `init`,
`alpha_in` equals the flip's `alpha` bit for bit, under `vanilla` it stays
below 1e-12, and `init` settles about half as much. Roster row D9 now records
the default as a policy row with a contract oracle.
