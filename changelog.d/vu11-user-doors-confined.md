### FIXED — the app's `.user.h5` doors read only the open set's files; a 0-step stage term refuses (#2750)

apeGmshViewer's `user:read` and `user:readCase` used to read any path the
renderer sent. They now read only the window's set: the set's user file (its
user slot, else `<stem>.user.h5`), judged against the set's own model and
results. The open set in view mode, or the job's set in render mode. Any other
path, including a sibling `.user.h5`, a `..` path or an absolute path elsewhere,
is refused with the reason, and so is every path when no set is open. Their
error line no longer uses `JSON.stringify`, so a request holding a BigInt or a
cycle resolves with the reason instead of rejecting. In Python,
`Results.combinations().evaluate` over a `stage` term on a stage with no steps
(San Ramón's static stage, whose modes are mode stages) now refuses as
`unresolved`. Before, it raised `IndexError`.
