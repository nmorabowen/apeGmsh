// Bundle the app into dist/: main (ESM, Node), preload (CJS, sandboxed),
// renderer (browser IIFE with three.js), plus the static page.

import { build } from "esbuild";
import { copyFileSync, mkdirSync } from "node:fs";
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
    entryPoints: ["src/main/preload.ts"],
    outfile: "dist/preload.cjs",
    platform: "node",
    format: "cjs",
    target: "node22",
    external: ["electron"],
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
