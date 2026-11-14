# MPM contract fixtures (vendored from the fork)

Byte-identical copies of the Ladruno fork's MPM contract fixtures and executable
validators (ADR 0134 D3, "Before M2"). **Never edit them here**: a change belongs in the
fork, then a refresh. The validators patch `sys.path` from their own location, so tests load
them by putting this directory on `sys.path` (see `tests/results/test_mpm_fixtures_baseline.py`).

Source: `nmorabowen/OpenSees`, branch `ladruno`, commit
`e30872e34d060322f3622ba262bcf6cf18cad7fa` (the merge of OpenSees#1080, the §2.5 crash-rule
follow-up; M2-S6 #2990). It added `particles_torn_v1m2.ladruno` and `committed_outputs` in
`mpm_contracts.py`. The other files were first vendored at `55f1831f2` (M2-S0 #2979) and
their blobs are unchanged at `e30872e34`.

| File here | Fork path | git blob sha |
|---|---|---|
| `particle_set_v1.h5` | `tests/data/mpm/particle_set_v1.h5` | `458821a33debe76c83568145f59a43d196047c2d` |
| `particles_v1m2.ladruno` | `tests/data/mpm/particles_v1m2.ladruno` | `c29ea1f52f60c02cc2545c0f40ca76c03fffe115` |
| `particles_mixed_v1m2.ladruno` | `tests/data/mpm/particles_mixed_v1m2.ladruno` | `0ffe3d5a556faa4435934d69eaa711dbc4aa5a70` |
| `particles_torn_v1m2.ladruno` | `tests/data/mpm/particles_torn_v1m2.ladruno` | `8fdac14ae0c7876ae18bd4f17d1a9ac9d00b2675` |
| `mpm_contracts.py` | `tests/mpm_contracts.py` | `aa3a2117b8d53966fb298ad43ffc1d56aa4c35cc` |
| `ladruno_format.py` | `Ladruno_scripts/ladruno_recorder_tests/ladruno_format.py` | `b1cc954d7be20827c06eefa1da430a3f3636e6a5` |
| `ladruno_basis.py` | `Ladruno_scripts/ladruno_recorder_tests/ladruno_basis.py` | `2c1f178877797e7977f66b3bf826fa8f9b1292a7` |

Check a copy: `git hash-object tests/fixtures/mpm/<file>` must print its sha above.

Refresh one file (set `S` to the new fork commit, then update the table):

```bash
S=e30872e34d060322f3622ba262bcf6cf18cad7fa
gh api "repos/nmorabowen/OpenSees/contents/<fork path>?ref=$S" \
  -H "Accept: application/vnd.github.raw" > tests/fixtures/mpm/<file>
```
