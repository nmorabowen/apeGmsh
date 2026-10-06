### FIXED — mark the Ladruno SaniSand bit-identity test strict-xfail on Linux (program link F2, #1523)

`test_i1_pinned_ladruno_sanisand_reproduces_manzari_bit_identically` is now
`xfail(strict=True)` on Linux, where the fork build differs from ManzariDafalias by about 4e-10.
Other platforms are unchanged. The F3 roster (#1523) tracks the underlying difference.
