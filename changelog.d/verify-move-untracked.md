### verify_move flags ignored and untracked files under `--head WORKTREE` (#1705)

`scripts/verify_move.py --head WORKTREE` now warns, on stderr and per file, about every checked `.py` file that git ignores or does not track, since a green WORKTREE verify proves nothing about the commit for those files (the S2-a `build/` ignore rule hid 23 files). `--strict` turns the warning into exit 1. SHA mode is unchanged.
