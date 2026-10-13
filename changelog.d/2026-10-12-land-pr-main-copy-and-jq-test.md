### CHANGED — landing runs main's copy of `land_pr.py`; a real-jq test pins `FILES_JQ` (program slice #1715 part b)

AGENTS.md "How work lands" now says to land with `origin/main`'s copy of `scripts/land_pr.py`
(`git show` into a temp file, then run it), because a PR that deletes the script's self-hop would
otherwise run its own copy. `tests/test_land_pr.py` gains a test that runs the real `FILES_JQ`
expression (the `jq` binary, else the `jq` module, else a skip) over a recorded `pulls/N/files` page,
so a mutation such as `.previous_filename` to `null` now fails it.
