// VU12 (#2753), owner ruling "option A": the preload never lets the page name a
// path to open. `requestOpen` takes a dropped File, asks Electron for its path
// (`webUtils.getPathForFile`), tells main it was dropped (`viewer:dropped`,
// synchronously, before the open), and only then asks main to open it. A File
// the page made has no path on disk: nothing is sent and the answer refuses.
//
// The oracle is preload.ts itself, loaded with `electron` resolved to
// ./electronStub.ts, which records every message the preload sends.

import assert from "node:assert/strict";
import { registerHooks } from "node:module";
import { test } from "node:test";

const STUB = new URL("./electronStub.ts", import.meta.url).href;
registerHooks({
  resolve: (specifier, context, next) => (specifier === "electron" ? { url: STUB, shortCircuit: true } : next(specifier, context)),
});
const { exposed, filePaths, sent } = (await import(STUB)) as typeof import("./electronStub.ts");
await import(new URL("../../src/main/preload.ts", import.meta.url).href);
const viewer = exposed.get("viewer") as unknown as {
  requestOpen(f: object): Promise<{ ok: boolean; reason?: string }>;
  pathForFile(f: object): string;
  openModel(path: string): Promise<unknown>;
};

test("VU12: a dropped File is told to main as a drop, then opened by the path Electron names", async () => {
  const dropped = {};
  filePaths.set(dropped, "C:\\work\\frame.h5");
  sent.length = 0;
  await viewer.requestOpen(dropped);
  assert.deepEqual(sent, [
    { how: "sendSync", channel: "viewer:dropped", args: ["C:\\work\\frame.h5"] },
    { how: "invoke", channel: "viewer:requestOpen", args: ["C:\\work\\frame.h5"] },
  ]);
});

test("VU12: a File the page made (no path on disk) opens nothing and tells main nothing", async () => {
  sent.length = 0;
  const r = await viewer.requestOpen({});
  assert.equal(r.ok, false);
  assert.match(r.reason ?? "", /^requestOpen opens a file dropped from disk; this one has no path$/);
  assert.deepEqual(sent, []);
});

test("VU12: pathForFile (the drop fallback onto openModel) records the drop before the page has the path", () => {
  const dropped = {};
  filePaths.set(dropped, "C:\\work\\other.h5");
  sent.length = 0;
  assert.equal(viewer.pathForFile(dropped), "C:\\work\\other.h5");
  assert.deepEqual(sent, [{ how: "sendSync", channel: "viewer:dropped", args: ["C:\\work\\other.h5"] }]);
  sent.length = 0;
  assert.equal(viewer.pathForFile({}), "");
  assert.deepEqual(sent, [], "a File with no path is no drop");
});
