// VU18 (#2856; ADR 0132 D8, D9; ADR 0128 D4): main's `render:variant` door,
// the one a render shot that names a variant row goes through. The oracle is
// main.ts itself, run in a child Node process with `electron` stubbed
// (./renderVariantDriver.ts), in render mode on a spec whose `files.variants`
// is the committed VU10 pair (test/fixtures/vu10), and judged on what the door
// answered and on which set the window's doors then read (`results:open`
// opens only the drawn set's results file):
//
// - a row named by index and label becomes the drawn set, and the window's
//   doors follow it; a compare slot holding that row empties;
// - a label that differs only in case, an index out of range or negative, an
//   empty label, a path, or a key beside index and label is refused, and the
//   drawn set stays as it was;
// - reload: the manifest rewritten between two shots is read again (D9.3);
// - D8: a row whose path leaves the study is refused with its code;
// - `null` draws the spec's set again; the page's switch door stays shut;
// - a spec with no `files.variants` refuses the door.

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { cpSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";

const FIX = fileURLToPath(new URL("../fixtures/vu10/", import.meta.url));
const DRIVER = fileURLToPath(new URL("./renderVariantDriver.ts", import.meta.url));

function battery(name: string, files: Record<string, unknown>, extra: string[] = []): Record<string, unknown> {
  const root = mkdtempSync(join(tmpdir(), `vu18-${name}-`));
  after(() => rmSync(root, { recursive: true, force: true }));
  const study = join(root, "study");
  cpSync(FIX, study, { recursive: true, filter: (src) => !src.endsWith(".py") });
  const spec = join(root, "spec.json");
  writeFileSync(spec, JSON.stringify({ spec: "apegmsh-view/1", files, shots: [{ name: "a" }] }));
  const args = [DRIVER, `--dir=${root}`, "--mode=render", `--spec=${spec}`, `--out=${join(root, "out")}`, ...extra];
  const stdout = execFileSync(process.execPath, args, { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], timeout: 60_000 });
  const line = stdout.split(/\r?\n/).find((l) => l.startsWith("RENDER_VARIANT "));
  if (!line) throw new Error(`the driver printed no answer:\n${stdout}`);
  return JSON.parse(line.slice("RENDER_VARIANT ".length)) as Record<string, unknown>;
}

type Seen = { ok: boolean; reason: string | null; set?: { model: string; geometry: string | null; results: string | null }; row?: { index: number; label: string } | null };
type Doors = { uncracked: boolean; cracked: boolean; reason: string | null };

const got = battery("study", { model: "study/plate_uncracked.h5", variants: "study/study.variants.json" });
const s = (k: string) => got[k] as Seen;
const doors = (k: string) => got[k] as Doors;
const tail = (p: string | undefined) => (p ?? "").replaceAll("\\", "/").toLowerCase();

test("the spec's set is drawn first, and the window's doors read only it", () => {
  assert.match(tail((got["job"] as { model: string }).model), /study\/plate_uncracked\.h5$/);
  assert.deepEqual([doors("doors:start").uncracked, doors("doors:start").cracked], [true, false]);
});

test("a row named by index and label becomes the drawn set: the window's doors follow it, and a compare slot holding it empties", () => {
  assert.equal(s("compare:1").ok, true, s("compare:1").reason ?? "");
  const r = s("row:1");
  assert.equal(r.ok, true, r.reason ?? "");
  assert.deepEqual(r.row, { index: 1, label: "Cracked" });
  assert.match(tail(r.set!.model), /study\/cracked\/plate_cracked\.h5$/);
  assert.match(tail(r.set!.results!), /study\/cracked\/plate_cracked\.ladruno$/);
  assert.equal(s("compare:step after row:1").ok, false);
  assert.match(s("compare:step after row:1").reason!, /no compare set is open/);
  assert.deepEqual([doors("doors:row 1").uncracked, doors("doors:row 1").cracked], [false, true]);
  assert.equal(s("model:row 1").ok, true, s("model:row 1").reason ?? "");
  assert.equal(s("model:spec set").ok, false, "the spec's model is not the drawn set's now");
});

test("degenerate names are refused, each with why, and the drawn set stays where it was", () => {
  assert.match(s("row:case").reason!, /variant row 1 is "Cracked", not "cracked"/);
  assert.match(s("row:out of range").reason!, /the manifest has no variant row 5/);
  assert.match(s("row:negative").reason!, /the manifest has no variant row -1/);
  assert.match(s("row:empty label").reason!, /variant row 1 is "Cracked", not ""/);
  assert.match(s("row:path").reason!, /named by \{ index, label \} and never a path/);
  assert.match(s("row:path key").reason!, /named by \{ index, label \} and never a path/);
  for (const k of ["row:case", "row:out of range", "row:negative", "row:empty label", "row:path", "row:path key"]) {
    assert.equal(s(k).ok, false, k);
    assert.equal(s(k).set, undefined, `${k} names no set`);
  }
  assert.deepEqual([doors("doors:after refusals").uncracked, doors("doors:after refusals").cracked], [false, true]);
});

test("reload: a manifest rewritten between two shots is read again, never the first shot's rows (D9.3)", () => {
  assert.equal(s("reload:old label").ok, false);
  assert.match(s("reload:old label").reason!, /variant row 1 is "Cracked x2", not "Cracked"/);
  assert.equal(s("reload:new label").ok, true, s("reload:new label").reason ?? "");
});

test("D8: a row whose path leaves the study is refused with its code, and nothing of it is drawn", () => {
  assert.equal(s("confine:row").ok, false);
  assert.match(s("confine:row").reason!, /variant row 1 is refused \(bad-path: .*could leave the manifest's folder/);
  // The drawn set is still the last one that answered (row 1 as read before the rewrite).
  assert.deepEqual([doors("doors:after confine").uncracked, doors("doors:after confine").cracked], [false, true]);
});

test("null draws the spec's set again, and the page's switch door stays shut in render mode", () => {
  assert.equal(s("row:null").ok, true, s("row:null").reason ?? "");
  assert.equal(s("row:null").row, null);
  assert.deepEqual([doors("doors:spec set").uncracked, doors("doors:spec set").cracked], [true, false]);
  assert.equal(s("switch").ok, false);
  assert.match(s("switch").reason!, /--mode=render has no window set to switch; a shot of the spec names the variant row it draws/);
});

test("a spec with no files.variants refuses the door, and the spec's set stays drawn", () => {
  const g = battery("plain", { model: "study/plate_uncracked.h5" }, ["--battery=no-variants"]);
  const r = g["row"] as Seen;
  assert.equal(r.ok, false);
  assert.match(r.reason!, /the spec names no files\.variants/);
  assert.deepEqual([(g["doors"] as Doors).uncracked, (g["doors"] as Doors).cracked], [true, false]);
});
