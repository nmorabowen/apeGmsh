### land_pr BASELINE covers the LOC-ratchet escape guards

`tests/test_loc_ratchet_escapes.py` joins `scripts/land_pr.py::BASELINE`, so a PR that edits its pins needs the `human-gate` label (check 8), like the ratchet's own baselines (#1886).
