### CI — F821 hard gate on the hub packages and a fork-token ratchet (C3.4)

`static-gates` now runs `ruff check --select F821` on `_kernel`, `core`,
`mesh`, `results`, `viewers` and `opensees`. `lock-tests` runs the new
`tests/test_fork_token_ratchet.py`: the count of `Ladruno` identifiers and
non-docstring strings per package glob (`_kernel`, `core`, `mesh`, `results`,
exempting `results/readers/_ladruno*.py`) may not exceed its pinned baseline.
No runtime change.
