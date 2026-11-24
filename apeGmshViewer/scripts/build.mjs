// Bundle the app into dist/: main (ESM, Node), the loader process main forks
// (ESM, Node; ADR 0137 D2), preload (CJS, sandboxed), renderer (browser IIFE
// with three.js), plus the static page.

import { build } from "esbuild";
import { copyFileSync, cpSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const dist = join(root, "dist");
mkdirSync(dist, { recursive: true });

const common = { bundle: true, sourcemap: true, logLevel: "warning", absWorkingDir: root };

await Promise.all([
  build({
    ...common,
    entryPoints: ["src/main/main.ts"],
    outfile: "dist/main.mjs",
    platform: "node",
    format: "esm",
    target: "node22",
    external: ["electron", "h5wasm"],
  }),
  build({
    ...common,
    entryPoints: ["src/main/loader/process.ts"],
    outfile: "dist/loader.mjs",
    platform: "node",
    format: "esm",
    target: "node22",
    external: ["h5wasm"],
  }),
  build({
    ...common,
    entryPoints: ["src/main/preload.ts"],
    outfile: "dist/preload.cjs",
    platform: "node",
    format: "cjs",
    target: "node22",
    external: ["electron"],
  }),
  // ADR 0137 H1: the tool window's preload (CJS, sandboxed) and page (browser IIFE, no three.js).
  build({
    ...common,
    entryPoints: ["src/main/toolPreload.ts"],
    outfile: "dist/toolPreload.cjs",
    platform: "node",
    format: "cjs",
    target: "node22",
    external: ["electron"],
  }),
  build({
    ...common,
    entryPoints: ["src/tools/window.ts"],
    outfile: "dist/tool.js",
    platform: "browser",
    format: "iife",
    target: "chrome130",
  }),
  build({
    ...common,
    entryPoints: ["src/renderer/app.ts"],
    outfile: "dist/renderer.js",
    platform: "browser",
    format: "iife",
    target: "chrome130",
  }),
]);
for (const f of ["index.html", "style.css"]) copyFileSync(join(root, "src/renderer", f), join(dist, f));
copyFileSync(join(root, "src/tools/tool.html"), join(dist, "tool.html"));
// The bundled Archivo Narrow files and their licence (src/theme/tokens.ts FONT_FILES).
cpSync(join(root, "src/renderer/fonts"), join(dist, "fonts"), { recursive: true });
