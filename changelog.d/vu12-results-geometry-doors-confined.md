### FIXED — the app's page no longer names the files main opens; the results and geometry doors read only the open set (#2753)

apeGmshViewer used to trust its page to name the files to open and read.
Following the owner's ruling, an open the page starts now needs a real user
gesture. A dropped file reaches main as the path Electron gives the File:
the preload's `requestOpen` takes the File, not a path, and reports the drop
to main before the open. `viewer:requestOpen` and `model:open` then open only
a file the user dropped, or a path main itself sent the page: the argv /
`--file` set, the set `onOpen` names, `config().file` and the render job.
`viewer:offerOpen` opens only the files of a set main offered, or the set an
offer left (its Back). It refuses an empty list of needed files. Any other path
is refused and the open set stays as it was. Opening a set from the page by
script, without a gesture, is no longer possible. The agent and render
command lines are unaffected.

`results:open`, `results:readStep`, `results:scan` and `geometry:open` read
only the window's set: in view mode the set on screen, in render mode the
job's set, and in describe, capture and measure the set of `--file`. Each
reads the set's own path, never the request's. They refuse a sibling, a `..`
path, an absolute path elsewhere, the old set's results after a drop, and
every path when no set is open. The error lines of these doors and of the
other open doors no longer use `JSON.stringify`, so a request holding a
BigInt or a cycle resolves with the reason instead of rejecting.
