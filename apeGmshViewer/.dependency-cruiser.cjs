// dependency-cruiser rules for the app's module graph (ADR 0112 D6; the
// panel-import rule of scripts/check_app_wall.py, over the same graph).
//
//   npm run lint          (depcruise --config .dependency-cruiser.cjs src)
//
// A panel (src/panels/<a>) reads the store and the selectors and dispatches
// events. It never imports another panel, the renderer, the HDF5 reader or
// the BlobStore. test/lint.test.ts plants one violation of each rule and
// proves the run fails.

/** @type {import('dependency-cruiser').IConfiguration} */
module.exports = {
  forbidden: [
    {
      name: "no-panel-to-panel",
      severity: "error",
      comment: "D6: no panel talks to another panel; they share the store only",
      from: { path: "^src/panels/([^/.]+)" },
      to: { path: "^src/panels/", pathNot: "^src/panels/$1(\\.ts|/)" },
    },
    {
      name: "no-panel-to-render",
      severity: "error",
      comment: "D6: a panel never reaches the three.js viewport or the renderer entry",
      from: { path: "^src/panels/" },
      to: { path: "^src/render(er)?/" },
    },
    {
      name: "no-panel-to-reader",
      severity: "error",
      comment: "D6: a panel reads the state, never the file",
      from: { path: "^src/panels/" },
      to: { path: "^src/reader/" },
    },
    {
      name: "no-panel-to-blobs",
      severity: "error",
      comment: "D6: the BlobStore is the viewport's; a panel sees BlobRefs in the state only",
      from: { path: "^src/panels/" },
      to: { path: "^src/state/blobs" },
    },
    {
      name: "no-circular",
      severity: "error",
      from: {},
      to: { circular: true },
    },
  ],
  options: {
    doNotFollow: { path: "node_modules" },
    tsPreCompilationDeps: true,
    tsConfig: { fileName: "tsconfig.json" },
    enhancedResolveOptions: { extensions: [".ts", ".js", ".mjs", ".cjs"] },
  },
};
