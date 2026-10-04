### CHANGED — apeGmshViewer: the D6 state store, BlobStore and effects; unmistakable selection; distinct group hues (ADR 0112, V2e)

The app's state is now one immutable record keyed by declaration path
(`<zone>/<family>/<name|#k>`, V0 decision 11) with a pure `reduce(state, event)`
over the twenty events of decision 17 plus `fileClosed`, a name index from
`/opensees/names`, `/labels` and `/physical_groups` (null-prototype records, so
a group named `constructor` is a name), and a reserved, empty `overrides`. The
file's arrays live in a `BlobStore` outside the snapshot; the state holds
`BlobRef`s, an element holds a row range into its block's connectivity blob
rather than a node copy, and the viewport's GPU cache is keyed by the refs.
Every side effect (open with latest-wins sequencing and a re-read after a
mid-read change, drop, V2f's `onOpen` / `onFileChanged` / `requestOpen` /
`goToSource`, each feature-detected) is in `src/effects.ts`; the DOM panels
under `src/panels/` dispatch events and read selectors only, and may import
nothing but `state/store`, `state/selectors`, `state/types` and `ui/`. The
chain the inspector shows is a selector over the state and is held equal to
the V1 resolver on the fixture and on the synthetic fail-closed models.
`scripts/check_app_wall.py` gains the `panel-import` allow-list rule (W4),
`npm run lint` runs dependency-cruiser over the same graph, a reducer-purity
test freezes the state and forbids TypedArray, ArrayBuffer, Map and Set in
it, and CI's new `app-tests` job runs typecheck, test and lint when
`apeGmshViewer/` changes. A zone a reader refuses is shown as "written by an
older (or newer) apeGmsh (zone, version, accepted range)", never a blank view.

From the V1 gate: a selected element gets a thick outline in the selection
colour over everything and the rest of the model dims (beams and shells, at
full-model zoom); group colours are generated in OKLCH so no two groups share
a hue family, neighbouring hue families take different tones, and legend
neighbours contrast; the drag state ends on a release off the canvas, every
listener has a `dispose()`, and the wheel zoom keeps the camera between 0.001
and 40 bounding radii of the zoom point.
