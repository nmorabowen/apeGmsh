### CHANGED — `land_pr.py` refuses a PR that changes `apeGmshViewer/` unless `app-tests` is green on its head (#2864, VH0)

`app-tests`, the apeGmshViewer app's own lane, is path-filtered: its first
step skips the app checks unless the PR changes `apeGmshViewer/`. So it was
neither a required check nor in the ruleset, and a viewer PR could land with
its own lane red or absent. Check 6 of `scripts/land_pr.py` now adds
`app-tests` to the five required checks when a changed file is under
`apeGmshViewer/`, counting both names of a rename and a deleted file. Every
run of it must be green on the head SHA, as for the other required checks.
A PR that touches no viewer path is unaffected, and the GitHub ruleset is
unchanged.
