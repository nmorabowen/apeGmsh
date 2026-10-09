### CHANGED — CI installs against a constraints file; nightly latest-deps canary (#1576)

Every `pip install` in `tests.yml` now resolves inside `.github/ci-constraints.txt`
(the exact versions of the last green environment, applied through `PIP_CONSTRAINT`), so an
upstream release can no longer turn `main` red (#1571). The new `deps-canary.yml` workflow runs the
`suite` selection nightly against the latest releases and opens one `[Deps canary]` issue naming the
drifted packages; `scripts/update_constraints.py` regenerates the file from a green canary run.
