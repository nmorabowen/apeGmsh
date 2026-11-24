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

test("VU7b-2: readUserCase asks main's user:readCase with the request as given, and records no drop (option A: main confines the path, test/main/userDoors.test.ts)", async () => {
  const door = viewer as unknown as { readUserCase(req: object): Promise<unknown> };
  const req = { path: "C:\\work\\frame.user.h5", case: "Ex", target: "nodes", label: "DISPLACEMENT" };
  sent.length = 0;
  await door.readUserCase(req);
  assert.deepEqual(sent, [{ how: "invoke", channel: "user:readCase", args: [req] }], "one invoke, and no viewer:dropped that would let main open the path");
});

test("#3130: openModel posts main's answer to the window as one structured clone and hands the page only its id", async () => {
  const posted: { data: unknown; target: string }[] = [];
  const g = globalThis as { window?: unknown };
  const had = "window" in g;
  const before = g.window;
  g.window = { postMessage: (data: unknown, target: string) => posted.push({ data, target }) };
  try {
    sent.length = 0;
    const id = await viewer.openModel("C:\work\mat.h5");
    assert.equal(typeof id, "number");
    assert.deepEqual(sent, [{ how: "invoke", channel: "model:open", args: ["C:\work\mat.h5"] }]);
    // The stub's invoke answers { ok: true, channel }: that answer, under the id the bridge returned.
    assert.deepEqual(posted, [{ data: { apegmshViewerBulk: id, value: { ok: true, channel: "model:open" } }, target: "*" }]);
  } finally {
    if (had) g.window = before;
    else delete g.window;
  }
});
