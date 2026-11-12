// VU19 (#2861; ADR 0132 D9.5, owner ruling on #2845): the compare set's model
// and `.user.h5` through compare-slot doors of their own. The oracle is main.ts
// itself, run in a child Node process with `electron` stubbed
// (./compareDoorsDriver.ts), opened in view mode on the VU10 study with a
// `.user.h5` beside each variant's model (test/fixtures/vu19), and judged
// against what Python's reader says of the same files (vu19.golden.json:
// `apeGmsh.results.user.staleness` and `UserFile.read_result`):
//
// - with no compare set, and after `compare:close`, a switch onto the row or a
//   page load, every compare door refuses;
// - `compare:model` reads the compare set's own model, and its fem_hash and
//   model_hash are the pairing main made at `variants:compare`;
// - `compare:userRead` lists the compare set's own file, each entry judged by
//   A6 against the compare set's OWN model and results: both current, where
//   against the window set's model and results both would be stale (golden);
// - `compare:userReadResult` reads the compare file's values, never the window
//   file's under the same name `footfall`;
// - a compare door takes no path and no row;
// - the shared namespace both ways: the window user doors refuse the compare
//   file, and with no compare `.user.h5` the compare read names the compare
//   set's missing file, never the window set's;
// - INV-6: no file of the study is written, created or removed by any of it.

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { copyFileSync, cpSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, relative } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";

const VU10 = fileURLToPath(new URL("../fixtures/vu10/", import.meta.url));
const VU19 = fileURLToPath(new URL("../fixtures/vu19/", import.meta.url));
const DRIVER = fileURLToPath(new URL("./compareDoorsDriver.ts", import.meta.url));
const GOLDEN = JSON.parse(readFileSync(join(VU19, "vu19.golden.json"), "utf8")) as {
  compare_own: { zones: Record<string, { stale: boolean; reason: string }>; results: Record<string, { stale: boolean; reason: string }> };
  compare_against_window: { results: Record<string, { stale: boolean; reason: string }> };
  window_own: { results: Record<string, { stale: boolean; reason: string }> };
  compare_footfall: { components: string[]; ids: number[]; data: (number | null)[][] };
  window_footfall: { components: string[]; ids: number[]; data: (number | null)[][] };
};

const root = mkdtempSync(join(tmpdir(), "vu19-doors-"));
after(() => rmSync(root, { recursive: true, force: true }));
const study = join(root, "study");
cpSync(VU10, study, { recursive: true, filter: (src) => !src.endsWith(".py") });
copyFileSync(join(VU19, "plate_cracked.user.h5"), join(study, "cracked", "plate_cracked.user.h5"));
copyFileSync(join(VU19, "plate_uncracked.user.h5"), join(study, "plate_uncracked.user.h5"));
const manifest = join(study, "study.variants.json");
const CRACKED_USER = join(study, "cracked", "plate_cracked.user.h5");

/** Every file under the study with its sha256: INV-6 is that the doors leave this as it was. */
function digest(dir: string): Record<string, string> {
  const out: Record<string, string> = {};
  const walk = (d: string) => {
    for (const n of readdirSync(d)) {
      const p = join(d, n);
      if (statSync(p).isDirectory()) walk(p);
      else out[relative(dir, p)] = createHash("sha256").update(readFileSync(p)).digest("hex");
    }
  };
  walk(dir);
  return out;
}

const before = digest(study);
const stdout = execFileSync(process.execPath, [DRIVER, `--dir=${root}`, `--file=${manifest}`], { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], timeout: 60_000 });
const after_ = digest(study);
const line = stdout.split(/\r?\n/).find((l) => l.startsWith("COMPARE_DOORS "));
if (!line) throw new Error(`the driver printed no answer:\n${stdout}`);
const got = JSON.parse(line.slice("COMPARE_DOORS ".length)) as Record<string, { resolved?: Record<string, unknown>; rejected?: string }>;

/** The answer, which must have resolved: a door never rejects. */
function answer(k: string): Record<string, unknown> {
  const a = got[k];
  assert.ok(a, `no answer for ${k}`);
  assert.ok(a.resolved !== undefined, `${k} rejected: ${a.rejected}`);
  return a.resolved as Record<string, unknown>;
}
type UserReadAnswer = { slot: { index: number; label: string } | null; path: string | null; read: { ok: boolean; error?: string; user?: { path: string; combinations: { ok: boolean; value?: { rows: unknown[]; generation: number | null } } }; staleness?: { results: Record<string, { stale: boolean; reason: string }> } } };
type ColumnAnswer = { slot: { index: number; label: string } | null; read: { ok: boolean; reason?: string; detail?: string; value?: { ids: number[]; values: number[]; component: string } } };
const same = (a: unknown, b: string) => typeof a === "string" && a.toLowerCase() === b.toLowerCase();
/** A golden column (null is NaN) as numbers, NaN kept. */
const column = (g: { data: (number | null)[][] }) => g.data.map((row) => (row[0] === null ? NaN : row[0]!));

/** Every compare door refused, each with why. */
function refusedAll(tag: string, why: RegExp): void {
  const m = answer(`${tag}:model`);
  assert.equal(m["ok"], false, `${tag}: compare:model`);
  assert.match(String(m["reason"]), why, `${tag}: compare:model`);
  const u = answer(`${tag}:userRead`) as unknown as UserReadAnswer;
  assert.equal(u.slot, null, `${tag}: compare:userRead names no slot`);
  assert.equal(u.read.ok, false, `${tag}: compare:userRead`);
  assert.match(String(u.read.error), why, `${tag}: compare:userRead`);
  const r = answer(`${tag}:userReadResult`) as unknown as ColumnAnswer;
  assert.equal(r.slot, null, `${tag}: compare:userReadResult names no slot`);
  assert.equal(r.read.ok, false, `${tag}: compare:userReadResult`);
  assert.match(String(r.read.detail), why, `${tag}: compare:userReadResult`);
}

test("with no compare set every compare door refuses, and nothing names the window set's files", () => {
  refusedAll("none", /no compare set is open/);
});

test("compare:model reads the compare set's own model: its mesh, with the fem_hash and model_hash main paired at variants:compare", () => {
  assert.equal((answer("compare") as { ok: boolean }).ok, true);
  const m = answer("open:model");
  assert.equal(m["ok"], true, String(m["reason"]));
  assert.deepEqual(m["slot"], { index: 1, label: "Cracked" });
  // The cracked model's /meta/lineage (the same mesh as Uncracked, another model_hash).
  assert.equal(m["femHash"], "d7bf21b6d7dd29ea55c0492eade9a5f5");
  assert.equal(m["modelHash"], "05337d99b871aed88bc2b3c7a727adb3");
  assert.equal(m["nodes"], GOLDEN.compare_footfall.ids.length);
  assert.ok(Number(m["elements"]) > 0);
  assert.ok(!("path" in m), "the answer names no path");
});

test("compare:userRead lists the compare set's own file, each entry judged by A6 against the compare set's own model and results (Python's verdicts)", () => {
  const u = answer("open:userRead") as unknown as UserReadAnswer;
  assert.deepEqual(u.slot, { index: 1, label: "Cracked" });
  assert.ok(same(u.path, CRACKED_USER), String(u.path));
  assert.equal(u.read.ok, true, u.read.error);
  assert.ok(same(u.read.user!.path, CRACKED_USER));
  // A6 against its OWN model and results: Python's `staleness(user, model_h5=cracked, results_file=cracked)`.
  for (const [name, want] of Object.entries(GOLDEN.compare_own.results)) {
    assert.deepEqual(u.read.staleness!.results[name], want, name);
  }
  // Against the window set's model and results the same entries would be stale (the golden's other read):
  // a door that judged the compare file against the window set would have said so.
  for (const [name, w] of Object.entries(GOLDEN.compare_against_window.results)) {
    assert.equal(w.stale, true, `precondition: ${name} is stale against the window set`);
    assert.equal(u.read.staleness!.results[name]!.stale, false, `${name}: judged against the compare set's own files`);
  }
  // The empty /combinations zone: a generation with no row.
  const c = u.read.user!.combinations;
  assert.equal(c.ok, true);
  assert.equal(c.value!.rows.length, 0);
  assert.notEqual(c.value!.generation, null);
});

test("compare:userReadResult reads the compare file's values, never the window file's under the same name", () => {
  const r = answer("open:userReadResult") as unknown as ColumnAnswer;
  assert.deepEqual(r.slot, { index: 1, label: "Cracked" });
  assert.equal(r.read.ok, true, r.read.detail);
  assert.deepEqual(r.read.value!.ids, GOLDEN.compare_footfall.ids);
  assert.deepEqual(r.read.value!.values, column(GOLDEN.compare_footfall));
  const w = answer("window:userReadResult own") as unknown as ColumnAnswer["read"];
  assert.equal(w.ok, true, w.detail);
  assert.deepEqual(w.value!.values, column(GOLDEN.window_footfall));
  assert.notDeepEqual(column(GOLDEN.window_footfall), column(GOLDEN.compare_footfall), "precondition: the two files differ");
  const d = answer("open:drift") as unknown as ColumnAnswer;
  assert.equal(d.read.ok, true, "drift (basis results) is current against the compare set's own results");
});

test("a compare door takes no path and no row: a request that carries one is refused before any read", () => {
  for (const k of ["path:model", "path:userRead", "path:userReadResult", "row:userRead"]) {
    const a = answer(k) as { reason?: string; read?: { error?: string; detail?: string }; slot: unknown };
    assert.equal(a.slot, null, k);
    assert.match(String(a.reason ?? a.read?.error ?? a.read?.detail), /never a path/, k);
  }
});

test("the shared namespace, both ways: the window doors refuse the compare file, and the compare read never falls back to the window's", () => {
  const own = answer("window:userRead own") as UserReadAnswer["read"];
  assert.equal(own.ok, true, own.error);
  assert.deepEqual(own.staleness!.results, GOLDEN.window_own.results);
  for (const k of ["window:userRead compare", "window:userReadResult compare"]) {
    const a = answer(k) as { ok: boolean; error?: string; detail?: string };
    assert.equal(a.ok, false, k);
    assert.match(String(a.error ?? a.detail), /reads only the open set's user file/, k);
  }
  assert.ok(same((answer("window:set") as { model: string }).model, join(study, "plate_uncracked.h5")), "the compare doors left the window set where it was");
  const gone = answer("gone:userRead") as unknown as UserReadAnswer;
  assert.deepEqual(gone.slot, { index: 1, label: "Cracked" });
  assert.equal(gone.read.ok, false);
  assert.ok(same(gone.read.error, `no such file: ${CRACKED_USER}`), String(gone.read.error));
  const goneCol = answer("gone:userReadResult") as unknown as ColumnAnswer;
  assert.equal(goneCol.read.ok, false);
  assert.ok(String(goneCol.read.detail).toLowerCase().includes(CRACKED_USER.toLowerCase()), String(goneCol.read.detail));
});

test("compare:close, a switch onto the compare row and a page load each empty the slot: every compare read is then refused", () => {
  refusedAll("closed", /no compare set is open/);
  assert.equal((answer("switch") as { ok: boolean }).ok, true);
  refusedAll("switched", /no compare set is open/);
  assert.equal((answer("reopen") as { ok: boolean }).ok, true);
  assert.equal(answer("before reload:model")["ok"], true, "precondition: the reopened slot reads");
  refusedAll("reloaded", /no compare set is open/);
});

test("INV-6: the compare doors write nothing: every file of the study is as it was, and none was added or removed", () => {
  assert.deepEqual(after_, before);
});
