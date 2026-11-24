#!/usr/bin/env node
// `apegmsh-viewer [file]`: open a file from any folder, from this checkout
// (package.json `bin`). It bundles dist/ and hands over to launch.mjs's view
// mode, as `npm start` does, but a relative path means the caller's folder:
// npm runs a script from the package folder, so `npm start -- x.h5` does not.
// A second call goes to the window already running and returns.

import { statSync } from "node:fs";

const USAGE = `usage: apegmsh-viewer [file]

Open a file in apeGmshViewer: model.h5, or any file of its set
(<stem>.geometry.h5, <stem>.ladruno, <stem>.results.h5, ...); the others beside
it open with it. Without a file, drop one on the window.`;

function refuse(why) {
  console.error(`apegmsh-viewer: ${why}\n${USAGE}`);
  process.exit(2);
}

const args = process.argv.slice(2);
if (args.includes("--help") || args.includes("-h")) {
  console.log(USAGE);
  process.exit(0);
}
const flag = args.find((a) => a.startsWith("-"));
if (flag) refuse(`unknown flag ${flag}`);
if (args.length > 1) refuse("one file at a time: the rest of its set opens with it");
if (args.length === 1) {
  const st = statSync(args[0], { throwIfNoEntry: false });
  if (!st) refuse(`no such file: ${args[0]}`);
  if (!st.isFile()) refuse(`not a file: ${args[0]}`);
}

process.argv.splice(2, process.argv.length, "view", ...args);
try {
  await import("./build.mjs");
  await (await import("esbuild")).stop(); // else its service process lives as long as the window
  await import("./launch.mjs");
} catch (e) {
  if (e?.code !== "ERR_MODULE_NOT_FOUND") throw e;
  console.error(`apegmsh-viewer: ${e.message}\nRun "npm ci" in apeGmshViewer/ first.`);
  process.exit(1);
}
