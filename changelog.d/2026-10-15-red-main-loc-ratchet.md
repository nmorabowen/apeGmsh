### FIXED — `main` red on the LOC ratchet: dead imports removed from the recorder transcoder (#2040)

`tests/test_loc_ratchet.py` failed with `from_recorders: 1238 LOC > baseline 1237` after #2019 and #2017 landed back to back. `results/transcoders/_recorder.py` loses two unused imports (`CustomRuleLayout`, `line_station_gpx_path`); no behaviour changes and the baseline is untouched.
