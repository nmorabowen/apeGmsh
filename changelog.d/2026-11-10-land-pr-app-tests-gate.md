### CHANGED — `land_pr.py` refuses a PR that changes what `app-tests` checks unless that lane is green on its head (#2864, VH0)

`app-tests`, the apeGmshViewer app's own lane, is path-filtered: its first
step skips the checks unless the PR changes a path in its pathspec list.
That list is `apeGmshViewer/`, the workflow, and the `.ladruno` pairing
oracle's test, fixture and `stamp_provenance`. So the lane was neither a
required check nor in the ruleset, and a viewer PR, or one that broke the
pairing oracle, could land with that lane red or absent. Check 6 of
`scripts/land_pr.py` now adds `app-tests` to the five required checks when
a changed file matches `APP_PATHS`, counting both names of a rename and a
deleted file. A test holds `APP_PATHS` equal to the filter's list in
`tests.yml`. Every run of `app-tests` must be green on the head SHA, as for
the other required checks. Any other PR is unaffected, and the GitHub
ruleset is unchanged.
