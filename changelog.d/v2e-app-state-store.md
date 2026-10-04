### CHANGED — apeGmshViewer: the D6 state store, BlobStore and effects; unmistakable selection; distinct group hues (ADR 0112, V2e)

The app's state is now one immutable record keyed by declaration path
(`<zone>/<family>/<name|#k>`, V0 decision 11) with a pure `reduce(state, event)`
over the twenty events of decision 17, a name index from `/opensees/names`,
`/labels` and `/physical_groups`, and a reserved, empty `overrides`. The file's
arrays live in a `BlobStore` outside the snapshot; the state holds `BlobRef`s
and the viewport's GPU cache is keyed by them. Every side effect (open, drop,
V2f's `onOpen` / `onFileChanged` / `requestOpen` / `goToSource`, each
feature-detected) is in `src/effects.ts`; the DOM panels under `src/panels/`
dispatch events and read selectors only. The chain the inspector shows is a
selector over the state and is held equal to the V1 resolver on the fixture
and on the synthetic fail-closed models. `scripts/check_app_wall.py` gains the
`panel-import` rule (W4), `npm run lint` runs dependency-cruiser over the same
graph, and a reducer-purity test freezes the state and forbids TypedArray,
ArrayBuffer, Map and Set in it. A zone a reader refuses is shown as "written
by an older apeGmsh (zone, version, window)", never a blank view.

From the V1 gate: a selected element gets a thick outline in the selection
colour over everything and the rest of the model dims (beams and shells, at
full-model zoom); group colours are generated in OKLCH so no two groups share
a hue family and legend neighbours contrast; the drag state ends on a release
off the canvas, every listener has a `dispose()`, and the wheel zoom keeps the
camera between 0.001 and 40 bounding radii of the zoom point.
