### FIXED — the app's results and geometry doors read only the open set's files (#2753)

apeGmshViewer's `results:open`, `results:readStep`, `results:scan` and
`geometry:open` used to read any path the renderer sent. They now read only
the window's set: in view mode the set on screen, which a double-click or a
drop opened; in render mode the job's set, from the spec; in describe,
capture and measure the set of `--file`. Each door reads the set's own path,
never the request's. Any other path is refused with the reason: a sibling
results or geometry file, a `..` path, an absolute path elsewhere, the old
set's results after a drop moved the set, and every path when no set is open.
Their error line no longer uses `JSON.stringify`, so a request holding a BigInt
or a cycle resolves with the reason instead of rejecting.
