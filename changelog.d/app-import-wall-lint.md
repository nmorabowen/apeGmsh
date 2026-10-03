### ADDED — D4 import-wall lint for `apeGmshViewer/` (ADR 0112, program slice V1b, #1286)

`scripts/check_app_wall.py` fails if the app in `apeGmshViewer/` imports `apeGmsh` (W1 Python
imports, W2 module specifiers that are bare `apeGmsh` or relative paths leaving the app) or spawns
a subprocess into it (W3). It scans only `git ls-files apeGmshViewer`, skips comment lines, has no
waivers, and passes with a "nothing to check" note while the directory does not exist. It runs as a
step of `static-gates` after the quirk lint; `tests/test_check_app_wall.py` is the self-test.
