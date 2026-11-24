### CHANGED — apeGmshViewer's page and main share one typed channel map (review finding 5)

Every message between the viewer's page and its main process is now listed once, in
`apeGmshViewer/src/main/ipc.ts`, with its request and its answer. Main answers through it, the
preload is checked against it, and the page's doors take what the preload exposes without a cast, so
a renamed field or a changed answer fails the typecheck instead of the running app. Writing the map
found three places where the page's types had drifted from main's (the run's mode, the model's open
answer, the particle run's job) and three where the preload's requests were looser than the page's.
The loader process also answers a request it does not serve with why, instead of never answering.
No behaviour changes for the user.
