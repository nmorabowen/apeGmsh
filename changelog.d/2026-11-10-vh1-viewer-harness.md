### CHANGED — apeGmshViewer: the measure and capture harness moves out of `renderer/app.ts` (VH1, #2865)

`npm run measure` and `npm run capture` now run from `apeGmshViewer/src/renderer/harness/`, which takes one context
(`store`, `blobs`, `effects`, `viewport`, `bridge`) from the renderer's composition root. `renderer/app.ts` keeps the
wiring, the view mode and the mode dispatch (1,744 lines down to 397). A pure move: the same stills under the same
names, the same measure JSON keys, the same order of captures.
