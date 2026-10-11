### CHANGED — orphan detector: acknowledged-orphans list (#1237)

`scripts/find_orphans.py` gains `ACKNOWLEDGED`, a `(PR, kind)` list of nightly-report rows verified
redone on `main`: #858 by #1169, #1057 by #1058, #1097 (post-merge tip a7237433) by #1099. Those
rows no longer open the `[Orphans]` issue; any other orphan still reports. Replay tests in
`tests/test_find_orphans.py`.
