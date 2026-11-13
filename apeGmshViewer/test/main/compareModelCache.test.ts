// VF3-e (#2963, #2902 item 4): `compare:model` reads a ~1M-element compare model
// in 1.5-1.8 s, so main answers a re-open of the same compare set from its last
// read. The oracle is main.ts itself, run in a child Node process with `electron`
// stubbed (as ./compareDoorsDriver.ts does; this file is its own driver when
// VF3E_DRIVER=1) and `../reader/node.ts`'s `openModel` counted, on the VU10
// study (test/fixtures/vu10: a 4 x 2 FourNodeQuad plate, so 15 nodes and 24
// elements of every dimension by construction):
//
// - the page's re-open (`compare:close`, then `variants:compare` on the same row,
//   renderer/contour.ts `VariantRunner.check`) reads the model file once, and the
//   second answer is the first;
// - the file touched since is read again;
// - a file rewritten under the held size and mtime with another model_hash is
//   refused as no longer paired: the lineage is checked before the held read;
// - a set change (another row paired, its model read or not) drops it: the first
//   set opened again is read again.

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { copyFileSync, cpSync, mkdtempSync, rmSync, statSync, utimesSync } from "node:fs";
import { registerHooks } from "node:module";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

const SELF = fileURLToPath(import.meta.url);
const MAIN = join(import.meta.dirname, "..", "..", "src", "main", "main.ts");
const READER = pathToFileURL(join(import.meta.dirname, "..", "..", "src", "reader", "node.ts")).href;
const COUNTED = "vf3e:counted-reader";
const opt = (name: string) => process.argv.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3);

/** The driver: main.ts with `electron` stubbed and every `openModel` it makes recorded by file name. */
async function drive(): Promise<void> {
  const root = opt("dir");
  if (!root) throw new Error("compareModelCache driver needs --dir=");
  const STUB = new URL("./electronStub.ts", import.meta.url).href;
  registerHooks({
    resolve: (specifier, context, next) => {
      if (specifier === "electron") return { url: STUB, shortCircuit: true };
      if (specifier === "../reader/node.ts" && pathToFileURL(MAIN).href === context.parentURL) return { url: COUNTED, shortCircuit: true };
      return next(specifier, context);
    },
    load: (url, context, next) =>
      url === COUNTED
        ? {
            format: "module",
            shortCircuit: true,
            source: `import { openModel as read } from ${JSON.stringify(READER)};
export const reads = [];
globalThis.__vf3eReads = reads;
export function openModel(path) { reads.push(path); return read(path); }`,
          }
        : next(url, context),
  });
  const { handlers } = (await import(STUB)) as typeof import("./electronStub.ts");
  await import(pathToFileURL(MAIN).href);
  const reads = (globalThis as { __vf3eReads?: string[] }).__vf3eReads;
  if (!reads) throw new Error("main.ts did not load ../reader/node.ts through the counting hook");
  const study = join(root, "study");
  const CRACKED = join(study, "cracked", "plate_cracked.h5");
  const event = { sender: { isDestroyed: () => false, send: () => undefined } };
  const call = async (channel: string, ...args: unknown[]): Promise<unknown> => {
    const handle = handlers.get(channel);
    if (!handle) throw new Error(`main.ts registered no ${channel}`);
    return handle(event, ...args);
  };
  const out: Record<string, unknown> = {};
  /** compare:model, with the model reads it made by file name. */
  const model = async (tag: string) => {
    const n = reads.length;
    out[tag] = { answer: await call("compare:model", {}), reads: reads.slice(n).map((p) => basename(p)) };
  };

  // 1. The page's re-open: compare, read, clear, compare the same row again, read.
  out["open"] = await call("variants:compare", 1, "Cracked");
  await model("first");
  await call("compare:close");
  out["reopen"] = await call("variants:compare", 1, "Cracked");
  await model("again");
  // 2. The file touched since the read is read again.
  const t = statSync(CRACKED);
  utimesSync(CRACKED, t.atime, new Date(t.mtimeMs + 5000));
  await model("touched");
  // 3. Another model_hash under the held size and mtime (Uncracked's file: the same mesh,
  // another material, the same length) is refused as no longer paired, never answered from the held read.
  const held = statSync(CRACKED);
  copyFileSync(CRACKED, `${CRACKED}.keep`);
  copyFileSync(join(study, "plate_uncracked.h5"), CRACKED);
  utimesSync(CRACKED, held.atime, held.mtime);
  out["sameStat"] = statSync(CRACKED).size === held.size && statSync(CRACKED).mtimeMs === held.mtimeMs;
  await model("swapped");
  copyFileSync(`${CRACKED}.keep`, CRACKED);
  rmSync(`${CRACKED}.keep`);
  utimesSync(CRACKED, held.atime, held.mtime);
  await model("restored");
  // 4. A set change: compare the other row (the window set moves to Cracked to allow it), then Cracked again.
  await call("variants:switch", 1, "Cracked");
  out["other"] = await call("variants:compare", 0, "Uncracked");
  await model("otherSet");
  await call("variants:switch", 0, "Uncracked");
  out["back"] = await call("variants:compare", 1, "Cracked");
  await model("afterSetChange");
  // 4b. A set change the page never read the model of drops it too.
  await call("variants:switch", 1, "Cracked");
  out["otherUnread"] = await call("variants:compare", 0, "Uncracked");
  await call("variants:switch", 0, "Uncracked");
  out["backUnread"] = await call("variants:compare", 1, "Cracked");
  await model("afterUnreadSetChange");
  process.stdout.write(`COMPARE_MODEL_CACHE ${JSON.stringify(out)}\n`);
  process.exit(0);
}

if (process.env["VF3E_DRIVER"] === "1") {
  await drive();
} else {
  const VU10 = fileURLToPath(new URL("../fixtures/vu10/", import.meta.url));
  const root = mkdtempSync(join(tmpdir(), "vf3e-cache-"));
  after(() => rmSync(root, { recursive: true, force: true }));
  cpSync(VU10, join(root, "study"), { recursive: true, filter: (src) => !src.endsWith(".py") });
  const stdout = execFileSync(process.execPath, [SELF, `--dir=${root}`, `--file=${join(root, "study", "study.variants.json")}`], {
    encoding: "utf8",
    env: { ...process.env, VF3E_DRIVER: "1" },
    stdio: ["ignore", "pipe", "pipe"],
    timeout: 60_000,
  });
  const line = stdout.split(/\r?\n/).find((l) => l.startsWith("COMPARE_MODEL_CACHE "));
  if (!line) throw new Error(`the driver printed no answer:\n${stdout}`);
  const got = JSON.parse(line.slice("COMPARE_MODEL_CACHE ".length)) as Record<string, unknown>;
  type Read = { answer: Record<string, unknown>; reads: string[] };
  const read = (k: string) => got[k] as Read;
  const opened = (k: string) => assert.equal((got[k] as { ok: boolean }).ok, true, `${k}: variants:compare ${JSON.stringify(got[k])}`);
  // The 4 x 2 quad plate: (4 + 1)(2 + 1) nodes; 4 * 2 quad4, 2 (4 + 2) boundary line2 and 4 corner point1 elements.
  const PLATE = { nodes: 15, elements: 8 + 12 + 4 };

  test("the page's re-open of the same compare set (compare:close, then variants:compare) reads its model file once", () => {
    opened("open");
    opened("reopen");
    const first = read("first");
    assert.equal(first.answer["ok"], true, String(first.answer["reason"]));
    assert.deepEqual({ nodes: first.answer["nodes"], elements: first.answer["elements"] }, PLATE);
    assert.deepEqual(first.reads, ["plate_cracked.h5"], "the first compare:model reads the model");
    const again = read("again");
    assert.deepEqual(again.reads, [], "the re-open answers from the last read");
    assert.deepEqual(again.answer, first.answer, "and its answer is the first's");
  });

  test("a model file touched since the read is read again", () => {
    const touched = read("touched");
    assert.equal(touched.answer["ok"], true, String(touched.answer["reason"]));
    assert.deepEqual(touched.reads, ["plate_cracked.h5"]);
  });

  test("another model_hash under the held size and mtime is refused as no longer paired, never answered from the held read", () => {
    assert.equal(got["sameStat"], true, "precondition: the swapped file has the held size and mtime");
    const swapped = read("swapped");
    assert.equal(swapped.answer["ok"], false);
    assert.match(String(swapped.answer["reason"]), /no longer pairs with the compare results/);
    assert.deepEqual(swapped.reads, []);
    // The file put back as it was answers from the held read again.
    const restored = read("restored");
    assert.equal(restored.answer["ok"], true, String(restored.answer["reason"]));
    assert.deepEqual(restored.reads, []);
  });

  test("a set change drops the held read: the other set is read, and the first set opened again is read again", () => {
    opened("other");
    opened("back");
    const other = read("otherSet");
    assert.equal(other.answer["ok"], true, String(other.answer["reason"]));
    assert.deepEqual(other.reads, ["plate_uncracked.h5"]);
    assert.deepEqual({ nodes: other.answer["nodes"], elements: other.answer["elements"] }, PLATE);
    const back = read("afterSetChange");
    assert.equal(back.answer["ok"], true, String(back.answer["reason"]));
    assert.deepEqual(back.reads, ["plate_cracked.h5"]);
    // Pairing the other row drops the held read even when its own model was never asked for.
    opened("otherUnread");
    opened("backUnread");
    const unread = read("afterUnreadSetChange");
    assert.equal(unread.answer["ok"], true, String(unread.answer["reason"]));
    assert.deepEqual(unread.reads, ["plate_cracked.h5"]);
  });
}
