### FIXED — `scripts/land_pr.py` check 7 refuses an amendment note added or flipped to Accepted

Check 7 (ADR 0121 D4, invariant 8) read Status lines only in the first 30
lines of an ADR, the header `adr_index.parse_adr` reads. An amendment note
lower down carries its own `**Status:**` line, so a PR that added one reading
Accepted passed the check: #3078 added ADR 0138's D5 note, Accepted, at line
160. The maintainer did accept that one in chat, but accepting an amendment
is the maintainer's act, as #3031 ("ADR 0114: mark the 2026-11-13 amendment
note Accepted") was treated.

`_sets_accepted` now also refuses a patch whose body, below the header, gains
more `**Status:** Accepted` lines than it loses. The header rule is unchanged.
Counting keeps a re-wrapped or moved Accepted amendment passing, and an
amendment added as Proposed passes. Replayed over the last 600 commits that
touched `architecture/decisions/`, every new refusal is a real amendment
acceptance, several of them landed by agent PRs. The replay of #3078 and the
pass cases are in `tests/test_land_pr.py`.
