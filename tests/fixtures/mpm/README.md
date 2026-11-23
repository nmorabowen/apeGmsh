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

## A killed run, and the stage rule's references (M2-S7 #2991)

Build and fixtures stay at the pin, `f0b9a1df699f9634fd89c5942d69d90fe5ad27c4` (`FORK_PIN`,
release `ladruno-v20261114`). The fork commits no killed-run file, and no Ladruno build at the
pin is on the machine that made this slice (the installed one predates the MPM writer), so the
killed run was **produced with the fork's own tool** at the pin, not vendored and not a real
process kill.

| File here | How it was made | git blob sha |
|---|---|---|
| `particles_killed_swmr_replay.h5` | `make_killed_swmr_replay.py <fork files at f0b9a1df6>`: the fork's `Ladruno_scripts/mpm_crash_tests/hdf5_crash_replay.py` (blob `99772f89ceda9596f63027800300c4eb5d886cd9`), unmodified: `write_run(fobj, 6, False, 8 << 20, 0, True)` (SWMR, the current writer's HDF5 call pattern), the image after write op 237, then the tool's own `check()` (flags 0x05 with a new lookup3: sec2's SWMR flags, which the fileobj driver does not write). h5py 3.15.1 / HDF5 1.14.6. | `145223563319571019c32d2f144ff8665b040080` |
| `particles_torn_last_stage_v1m2.ladruno` | `make_torn_stage_fixtures.py`, from `particles_v1m2.ladruno` | `732ad5cd110307e0d38d3f8b33ac0d91ec4a3938` |
| `particles_torn_early_stage_v1m2.ladruno` | `make_torn_stage_fixtures.py`, from `particles_v1m2.ladruno` | `f1e21eaea23c3d28f5e8889706c4eaa3359cae96` |
| `particles_two_stages_v1m2.ladruno` | `make_torn_stage_fixtures.py`, from `particles_v1m2.ladruno` (both stages whole) | `a0dd6da74661634f36fd6710bb3f5945f92f2510` |

- `particles_killed_swmr_replay.h5` is an HDF5 image of the particle group's layout as the
  replay models it, **not** a contract-conformant `.ladruno` (no `INFO`, and its field
  attributes are placeholders), so it is named `.h5`, and only the open rule and the crash rule
  are read from it: plain open refused ("already open for write"), SWMR-read open (step 2), 4
  committed outputs by the fork's `committed_outputs`, a partial fifth output on disk, and the
  stored end-of-allocation (38115) behind the file size (39580). Its rows are the replay's closed
  form (`rows(k, nc) = k*1000 + i*10 + c + 0.5`, ids `IDS`). A rebuild is not byte-identical (the
  tool's layout tracks object times) but reads the same.
- The two torn-stage files model a stage torn while it was laid out, at the layout level:
  two particle stages in a version-3 superblock file, closed (flags 0x00). The tests copy one and
  set its flags to 0x05 or 0x01 (with the fork's `set_superblock`, or the viewer's `lookup3`
  checked against HDF5's own checksums) to reach steps 2 and 3. See the script's docstring. The
  third case, a last stage with no `RESULTS/ON_PARTICLES` yet, is built per test and not committed
  (`build(p, "half")` in Python, `rewritten(..., { noLastOnParticles: true })` in the viewer):
  `LadrunoReader` refuses it at open (C2-PAIR), and every committed `.ladruno` must open
  (`tests/results/test_ladruno_duplicate_node_ids.py`).
- `particles_two_stages_v1m2.ladruno` is the base of the HDF5-level damage cases: both test
  suites flip the first byte of the same B-tree node signatures (`TREE`/`SNOD`, at byte offsets
  the tests list and check) in a copy, and expect the same stage outcome from both readers. The
  offsets hold as long as the file's blob sha above holds; a rebuild that changes the bytes
  needs the offsets found again (the tests assert the signature at each one).
- `damage_sweep.py` builds 966 damaged copies of it, in a temp folder, never committed: every
  object signature flipped (146 sites), 256 zero bytes at every 4 KiB boundary (31), 1 KiB of
  zeros off that grid (42), the file cut every 2 KiB through the last stage's layout (31), and
  a byte flipped in attribute messages (72), each at flags 0x00/0x05/0x01. `damage_sweep_expected.json` is the Python reader's outcome per
  copy, recorded by running the script (each case in a worker with a timeout, so an HDF5 hang is
  recorded as `hang` and never opened by a test). Both readers' tests check every copy against
  it, so they agree file by file, but for two named, counted differences on attribute damage
  (`KNOWN_FIELD_ONLY`, `KNOWN_KIND_REFUSAL` in the viewer's test); the Python test also checks that Python never reads a stage
  the fork's reference `particle_stage_report` skips or refuses. Re-record it only for a
  deliberate change to the stage rule.

## B-spline grids (#3013)

Fork release `ladruno-v20261118` (`3e738308c`, WP-232i, OpenSees#1099) amends 232a §2.4:
a Contract 2 `GRID/BASIS` is one of `gimp | linear | bspline2 | bspline3`, with no minor
bump; Contract 1 keeps `gimp | linear` (the fork checker's `C1_BASES`, `C2_BASES` in
`ladruno_format.py`). The fork commits no `.ladruno` with a B-spline grid, so these are
**made** from `particles_v1m2.ladruno` by `make_basis_fixtures.py`, which rewrites only
`BASIS`, in the fork's encoding (a 1-element, fixed-length, null-terminated ASCII string).
The outputs are still the GIMP run's: the files pin what a reader accepts, not what a
B-spline run computes. The fork's checker at the tag accepts both (`validate` returns
`[]`) and refuses the same edit with `bspline4` (C2-GRID). The live test
`tests/opensees/integration_ladruno/test_mpm_bspline_readers_live.py` runs the fork and
checks that its BASIS has this encoding.

| File here | How it was made | git blob sha |
|---|---|---|
| `particles_bspline2_v1m2.ladruno` | `make_basis_fixtures.py`, from `particles_v1m2.ladruno` | `b8df927bd194a013f6b1d1a2516eec6e716c247e` |
| `particles_bspline3_v1m2.ladruno` | `make_basis_fixtures.py`, from `particles_v1m2.ladruno` | `9a4e989a38256ed10db5004546b80721c85f2fe2` |

The vendored `ladruno_format.py` and `mpm_contracts.py` above predate the amendment and
still refuse a B-spline grid; they are not refreshed here.

`fork_a1d4c7260/` holds the stage rule's **reference code** at
`a1d4c72601fe394a58a33e1596bb21a2efbb3d4c` (OpenSees#1088: 232a §2.5's stage rule), read as an
oracle only: `particle_stage_report`, `recover_particle_file`, `open_particle_file` and
`committed_outputs` are not in the pinned copies above. The owner's ruling of 2026-11-15 on #2991
builds the readers against 232a §2.5 at that commit; the fixtures and the build stay at the pin.

| File here | Fork path at `a1d4c7260` | git blob sha |
|---|---|---|
| `fork_a1d4c7260/mpm_contracts.py` | `tests/mpm_contracts.py` | `230bef441403dd09a15466ad5e1b91ce2698528d` |
| `fork_a1d4c7260/ladruno_format.py` | `Ladruno_scripts/ladruno_recorder_tests/ladruno_format.py` | `786af6ef9ca7617e254877c3aea8470f216d65a2` |
| `fork_a1d4c7260/ladruno_basis.py` | `Ladruno_scripts/ladruno_recorder_tests/ladruno_basis.py` | `2c1f178877797e7977f66b3bf826fa8f9b1292a7` |

## PORE_PRESSURE and the field catalogue (#3121)

Fork release `ladruno-v20261123b` (`3eb195bbccb8b78a8982267179a8a0b2b7e03977`, the `FORK_PIN`
when this was made; WP-232l, OpenSees#1116, P12) adds one reserved particle field to 232a
§2.6: `PORE_PRESSURE`, `COMPONENTS` `p` in 2-D and 3-D, `DIMENSION` `F/L^2`, `DATA_TYPE` 0,
the excess pore pressure of a two-phase particle, compression positive (0 on a single-phase
particle), with no minor bump. A two-phase particle's `STRESS` is σ′. Both readers carry the
checker's `PARTICLE_FIELDS` as their catalogue and refuse a reserved field with other
`COMPONENTS` (C2-CAT); an unreserved name reads.

`fork_3eb195bbc/` holds the checker at the pin, read as an oracle (its `PARTICLE_FIELDS`, and
`validate()` on the fixture below and on the live test's file):

| File here | Fork path at `3eb195bbc` | git blob sha |
|---|---|---|
| `fork_3eb195bbc/ladruno_format.py` | `Ladruno_scripts/ladruno_recorder_tests/ladruno_format.py` | `3413e97d0287fb336398bc1b34f4cc291eef1f49` |
| `fork_3eb195bbc/ladruno_basis.py` | `Ladruno_scripts/ladruno_recorder_tests/ladruno_basis.py` | `2c1f178877797e7977f66b3bf826fa8f9b1292a7` |

The fork commits no `.ladruno` with a `PORE_PRESSURE` field, so this one is **made** from
`particles_v1m2.ladruno` by `make_pore_pressure_fixture.py`, which adds the group in the
writer's encoding (`LadrunoMPMIO.cpp`'s field table at the pin). Its values are
`9.81 * (2 - y)` of each row's position, not a two-phase run's: the file pins what a reader
accepts. The live test `tests/opensees/integration_ladruno/test_mpm_pore_pressure_readers_live.py`
runs a two-phase column at the pin and checks the fork writes this encoding. A rebuild is
byte-identical.

| File here | How it was made | git blob sha |
|---|---|---|
| `particles_pore_pressure_v1m2.ladruno` | `make_pore_pressure_fixture.py`, from `particles_v1m2.ladruno` | `822901c999e5f93cb40b93c856c7c83ae04be1e9` |

Refresh one file (set `S` to the new fork commit, then update the table):

```bash
S=e30872e34d060322f3622ba262bcf6cf18cad7fa
gh api "repos/nmorabowen/OpenSees/contents/<fork path>?ref=$S" \
  -H "Accept: application/vnd.github.raw" > tests/fixtures/mpm/<file>
```
