# MPM contract fixtures (vendored from the fork)

Byte-identical copies of the Ladruno fork's MPM contract fixtures and executable
validators (ADR 0134 D3, "Before M2"). **Never edit them here**: a change belongs in the
fork, then a refresh. The validators patch `sys.path` from their own location, so tests load
them by putting this directory on `sys.path` (see `tests/results/test_mpm_fixtures_baseline.py`).

Source: `nmorabowen/OpenSees`, branch `ladruno`, commit
`55f1831f2bdcf7ef7d51acd8344017b9c2274a84`.

| File here | Fork path | git blob sha |
|---|---|---|
| `particle_set_v1.h5` | `tests/data/mpm/particle_set_v1.h5` | `458821a33debe76c83568145f59a43d196047c2d` |
| `particles_v1m2.ladruno` | `tests/data/mpm/particles_v1m2.ladruno` | `c29ea1f52f60c02cc2545c0f40ca76c03fffe115` |
| `particles_mixed_v1m2.ladruno` | `tests/data/mpm/particles_mixed_v1m2.ladruno` | `0ffe3d5a556faa4435934d69eaa711dbc4aa5a70` |
| `mpm_contracts.py` | `tests/mpm_contracts.py` | `431dd5b7a5a55ab35e998de3fc2a9229c028315a` |
| `ladruno_format.py` | `Ladruno_scripts/ladruno_recorder_tests/ladruno_format.py` | `b1cc954d7be20827c06eefa1da430a3f3636e6a5` |
| `ladruno_basis.py` | `Ladruno_scripts/ladruno_recorder_tests/ladruno_basis.py` | `2c1f178877797e7977f66b3bf826fa8f9b1292a7` |

Check a copy: `git hash-object tests/fixtures/mpm/<file>` must print its sha above.

Refresh one file (set `S` to the new fork commit, then update the table):

```bash
S=55f1831f2bdcf7ef7d51acd8344017b9c2274a84
gh api "repos/nmorabowen/OpenSees/contents/<fork path>?ref=$S" \
  -H "Accept: application/vnd.github.raw" > tests/fixtures/mpm/<file>
```
