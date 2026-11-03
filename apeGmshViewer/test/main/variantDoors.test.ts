// VU10 (#2694; ADR 0132 D9, owner ruling option A #2753): main's variant
// doors. The oracle is main.ts itself, run in a child Node process with
// `electron` stubbed (./variantDoorsDriver.ts), opened in view mode on the
// committed variant pair's manifest (test/fixtures/vu10), and judged on what
// each door answered and on the window set (`viewer:currentSet`):
//
// - a manifest the user gave opens the study, and its first row as the window set;
// - the page names a row by index and label: a mismatch, a refused row or a path is refused;
// - the compare door opens a row read-only and never adopts it: the window set
//   stays, and the window's doors still refuse the compare set's files;
// - the compare set has its own results host: `results:close` leaves it, and a
//   compare step names no path (a request that carries one is refused);
// - a switch opens the row as the window set, and empties a compare slot holding it;
// - at most one compare set, and `compare:close` empties the slot.
//
// The values are the fixture's closed form (make_vu10_fixture.py): at the last
// step, Uy = 0.5 * 0.03 * x^2 / r with r = 0.35 for Cracked.

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { cpSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";

const FIX = fileURLToPath(new URL("../fixtures/vu10/", import.meta.url));
const DRIVER = fileURLToPath(new URL("./variantDoorsDriver.ts", import.meta.url));

const root = mkdtempSync(join(tmpdir(), "vu10-doors-"));
after(() => rmSync(root, { recursive: true, force: true }));
const study = join(root, "study");
cpSync(FIX, study, { recursive: true, filter: (src) => !src.endsWith(".py") });
const manifest = join(study, "study.variants.json");
const UNCRACKED = join(study, "plate_uncracked.h5");
const CRACKED = join(study, "cracked", "plate_cracked.h5");

type Seen = { ok: boolean; reason: string | null; compare?: { index: number; label: string; results: string; femHash: string | null; stages: number }; uy?: number[] };
type Study = { manifest: string; rows: { index: number; label: string; path: string }[]; refused: unknown[]; active: number | null; compare: { index: number; label: string } | null };

const stdout = execFileSync(process.execPath, [DRIVER, `--dir=${root}`, `--file=${manifest}`], { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], timeout: 60_000 });
const line = stdout.split(/\r?\n/).find((l) => l.startsWith("VARIANT_DOORS "));
if (!line) throw new Error(`the driver printed no answer:\n${stdout}`);
const got = JSON.parse(line.slice("VARIANT_DOORS ".length)) as Record<string, unknown>;
const s = (k: string) => got[k] as Seen;
const st = (k: string) => got[k] as Study;
const same = (a: unknown, b: string) => typeof a === "string" && a.toLowerCase() === b.toLowerCase();

test("a manifest the user gave opens the study, and its first row as the window set; the page sees rows, never resolved paths", () => {
  assert.ok(same(got["open:model"], UNCRACKED), String(got["open:model"]));
  const v = st("open:study");
  assert.deepEqual(v.rows.map((r) => [r.index, r.label, r.path]), [[0, "Uncracked", "plate_uncracked.h5"], [1, "Cracked", "cracked/plate_cracked.h5"]]);
  assert.equal(v.active, 0);
  assert.equal(v.compare, null);
  assert.ok(!JSON.stringify(v).includes(study.replaceAll("\\", "\\\\")) || same(v.manifest, manifest), "no row carries an absolute path");
  for (const r of v.rows) assert.ok(!("resolved" in r), "a row never carries its resolved path");
});

test("the page names a row by index and label: a mismatch, a path and the window's own row are refused", () => {
  assert.equal(s("compare:mismatch").ok, false);
  assert.match(s("compare:mismatch").reason!, /variant row 1 is "Cracked", not "Uncracked"/);
  assert.equal(s("compare:path").ok, false);
  assert.match(s("compare:path").reason!, /named by its row index and label/);
  assert.equal(s("compare:window").ok, false);
  assert.match(s("compare:window").reason!, /is the window set; a compare set is another variant/);
  assert.equal(s("switch:mismatch").ok, false);
});

test("the compare door opens a row read-only and never adopts it: the window set and the window's doors stay as they were", () => {
  const c = s("compare:ok");
  assert.equal(c.ok, true, c.reason ?? "");
  assert.equal(c.compare!.label, "Cracked");
  assert.ok(c.compare!.femHash, "the compare results state their mesh");
  assert.ok(same(got["compare:model"], UNCRACKED), "the window set did not move");
  assert.deepEqual(st("compare:study").compare, { index: 1, label: "Cracked" });
  assert.equal(st("compare:study").active, 0);
  assert.equal(s("window:results compare").ok, false, "results:open refuses the compare set's results");
  assert.match(s("window:results compare").reason!, /reads only the open set's results file/);
  assert.equal(s("window:user compare").ok, false, "user:read refuses the compare set's user file");
  assert.equal(s("window:results own").ok, true, s("window:results own").reason ?? "");
});

test("the compare set has its own results host, and a compare step names no path", () => {
  const step = s("compare:step after results:close");
  assert.equal(step.ok, true, step.reason ?? "");
  // Cracked at the last step: Uy = 0.5 * 0.03 * x^2 / 0.35, so its largest value (x = 2) is 0.06 / 0.35.
  assert.ok(Math.abs(Math.max(...step.uy!) - 0.06 / 0.35) < 1e-12, String(Math.max(...step.uy!)));
  for (const k of ["compare:step with path", "compare:step with window path"]) {
    assert.equal(s(k).ok, false, k);
    assert.match(s(k).reason!, /never a path/, k);
  }
});

test("a switch opens the row as the window set, and empties a compare slot that held it", () => {
  assert.equal(s("switch:ok").ok, true, s("switch:ok").reason ?? "");
  assert.ok(same(got["switch:model"], CRACKED), String(got["switch:model"]));
  assert.equal(st("switch:study").active, 1);
  assert.equal(st("switch:study").compare, null);
  assert.equal(s("switch:compare step").ok, false);
  assert.match(s("switch:compare step").reason!, /no compare set is open/);
});

test("at most one compare set, and compare:close empties the slot", () => {
  assert.equal(s("compare2:ok").ok, true, s("compare2:ok").reason ?? "");
  // Uncracked at the last step: its largest Uy is 0.06.
  assert.ok(Math.abs(Math.max(...s("compare2:step").uy!) - 0.06) < 1e-12);
  assert.equal(s("close:step").ok, false);
  assert.equal(st("close:study").compare, null);
});
