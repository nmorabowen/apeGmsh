### FIXED — schema-corpus ancestry test now runs in CI; neutral 2.35 re-anchored on main (program slice B2-5, #1416, closes #1407)

The `suite` lane checks out with `fetch-depth: 0`, so the MANIFEST "every non-current era SHA is an ancestor of origin/main" test runs instead of skipping. The neutral 2.35 corpus entry and the MANIFEST `base` are regenerated on main's commits (they were PR-branch SHAs), and the stale "Caveat (#1365)" paragraph is gone from the bridge-feature guide. No source change.
