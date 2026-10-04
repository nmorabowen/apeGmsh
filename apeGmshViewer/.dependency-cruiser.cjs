// dependency-cruiser rules for the app's module graph (ADR 0112 D6; the
// panel-import rule of scripts/check_app_wall.py, over the same graph).
//
//   npm run lint          (depcruise --config .dependency-cruiser.cjs src)
//
// A panel (src/panels/<a>) reads the store and the selectors and dispatches
// events. Its imports are an allow-list: src/state/store, src/state/selectors,
// src/state/types, src/ui/ and its own panel directory. Everything else
// (another panel, the renderer, the reader, the BlobStore, effects, the
// loader, a package) is a violation. test/lint.test.ts plants violations and
// proves the run fails.

const PANEL_ALLOW = ["^src/state/(store|selectors|types)\\.ts$", "^src/ui/"];

/** @type {import('dependency-cruiser').IConfiguration} */
module.exports = {
  forbidden: [
    {
      name: "panel-allow-list",
      severity: "error",
      comment: "D6: a panel imports the store, the selectors, the state types, ui/ and its own directory only",
      from: { path: "^src/panels/([^/.]+)" },
      to: { pathNot: [...PANEL_ALLOW, "^src/panels/$1/"] },
    },
    {
      name: "ui-allow-list",
      severity: "error",
      comment: "D6: ui/ imports ui/ and the state types only, so it cannot re-export what a panel may not reach",
      from: { path: "^src/ui/" },
      to: { pathNot: ["^src/ui/", "^src/state/types\\.ts$"] },
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
