### FIXED — AGENTS.md no longer counts the `lock-tests` files

AGENTS.md said the `lock-tests` job runs "five lock files". It runs seven
since K1-1 (#1370) added `tests/opensees/contract/test_verbs_lock.py`. The
row now says "the lock files the job names", so the job's own list in
`.github/workflows/tests.yml` stays the only count.
