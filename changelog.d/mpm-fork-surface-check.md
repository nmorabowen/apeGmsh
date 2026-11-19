### TESTS — the `mpm` parity check reads the fork's subcommands at `FORK_PIN` (#3063)

`tests/opensees/parity/test_mpm_command_parity.py` compared each `mpm`
member with a fork command line typed in at fork `f0b9a1df6`, and never read
the fork. A pin bump that added a subcommand passed: #3055 moved to
`ladruno-v20261118b`, which added `mpm stage` and `mpm particleResponse`, and
nothing failed.

`scripts/refresh_mpm_fork_surface.py` now reads the parser's dispatch
(`sub == "<name>"` in `OPS_LadrunoMPM`, cross-checked with the parser's own
usage line) at `FORK_PIN`'s sha and writes the names to
`tests/fixtures/mpm_fork_surface/surface.json`. The suite lane reads that
snapshot offline and fails:

- when its sha or tag is not `FORK_PIN`'s, saying to run the script;
- by name, for each fork subcommand that is neither a member's nor a row of
  `NOT_OFFERED` (`tests/opensees/parity/mpm_fork_surface.py`), which gives the
  reason apeGmsh has no member;
- for each row the fork no longer has, and each member the fork does not
  accept.

On the `live-fork` lane, `tests/opensees/integration_ladruno/
test_mpm_fork_surface_live.py` reads the parser at `FORK_PIN` from the
public fork and runs the same extraction. It fails when the snapshot differs,
so a snapshot whose sha was typed in by hand does not pass. A failed read
fails the test.

At `ladruno-v20261118b` the fork has 36 subcommands: 13 are members, and 23
are rows. Among the rows, `pforce` and `condForce` are mode A only and
refused in mode F, and `particles` is written by the deck emitters rather
than declared. Run the script at every pin bump. The check works at the
subcommand level: a new form of an existing subcommand, such as `gridvel …
affine`, is not seen.
