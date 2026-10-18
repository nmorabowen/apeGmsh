### ADDED - `check_quirks.py` flags a fork token in an added string under `results/` (#2146)

The quirk lint has a diff rule, `fork-token-string`. It flags a line a branch adds
under `src/apeGmsh/results/` whose non-docstring string literal names the Ladruno fork.
Such a string counts against the `results/**` fork-token ratchet, which never rises, and
twice (#1999, #2023) review caught it instead of the author. Keep user-facing text
fork-neutral, put a fork name behind `results/readers/_ladruno*.py`, or waive one site
with `# apegmsh-lint: fork-token-string-ok <reason>`. The ratchet test now reads its
token definition from `scripts/check_quirks.py`, so the two cannot drift.
