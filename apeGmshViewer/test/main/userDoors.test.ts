// VU11 (#2750): main's `.user.h5` doors, `user:read` and `user:readCase`, read
// only the window's set's user file, judged against the set's own model and
// results; and a request their log line cannot JSON-encode (a BigInt, a cycle)
// still resolves with the reason instead of rejecting.
//
// The oracle is the handlers themselves: main.ts runs in a child Node process
// with `electron` stubbed (userDoorsDriver.ts), opens a set the way a
// double-click does (`--file=`), and answers a fixed battery. Every path it must
// refuse is a READABLE copy of the paired file, the VU7b-1 San Ramón case, so
// without the check each would read `ok: true`: a sibling `.user.h5`, a `..`
// path out of the set's folder, and an absolute path elsewhere.

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";

const USER = fileURLToPath(new URL("../fixtures/vu7b/sanramon_case.user.h5", import.meta.url));
const MODEL = fileURLToPath(new URL("../../fixtures/contour_staged.h5", import.meta.url));
const DRIVER = fileURLToPath(new URL("./userDoorsDriver.ts", import.meta.url));

const root = mkdtempSync(join(tmpdir(), "vu11-doors-"));
after(() => rmSync(root, { recursive: true, force: true }));
for (const d of ["set", "outside", "elsewhere"]) mkdirSync(join(root, d));
for (const p of ["set/s.user.h5", "set/t.user.h5", "outside/s.user.h5", "elsewhere/s.user.h5"]) copyFileSync(USER, join(root, p));
copyFileSync(MODEL, join(root, "set", "s.h5"));
// a readable model in another folder: once it opens on the window, the set is there
copyFileSync(MODEL, join(root, "elsewhere", "s.h5"));
// a model still being written: its first 512 bytes, with its set's user file beside it
mkdirSync(join(root, "bad"));
writeFileSync(join(root, "bad", "b.h5"), readFileSync(MODEL).subarray(0, 512));
copyFileSync(USER, join(root, "bad", "b.user.h5"));

type OnScreen = { user: string | null; model: string | null };
type Answer = { resolved: { ok: boolean; error?: string; reason?: string; detail?: string; values?: number } } | { rejected: string } | OnScreen | null;

/** `viewer:currentSet` as the driver recorded it: null when no set is open. */
const setAt = (answers: Record<string, Answer>, name: string) => answers[name] as OnScreen | null;

function run(withSet: boolean): Record<string, Answer> {
  const args = [DRIVER, ...(withSet ? [`--file=${join(root, "set", "s.h5")}`, "--steer"] : []), `--dir=${root}`];
  const stdout = execFileSync(process.execPath, args, { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], timeout: 60_000 });
  const line = stdout.split(/\r?\n/).find((l) => l.startsWith("USER_DOORS "));
  assert.ok(line, `the driver printed no answers:\n${stdout}`);
  return JSON.parse(line.slice("USER_DOORS ".length)) as Record<string, Answer>;
}

/** The answer, which must have resolved: a door never rejects. */
function resolved(answers: Record<string, Answer>, name: string) {
  const a = answers[name];
  assert.ok(a, `no answer for ${name}`);
  assert.ok("resolved" in a, `${name} rejected: ${"rejected" in a ? a.rejected : ""}`);
  return a.resolved;
}

const withSet = run(true);

test("VU11: both doors read the set's own user file", () => {
  assert.deepEqual(resolved(withSet, "read:paired"), { ok: true });
  const c = resolved(withSet, "readCase:paired");
  assert.equal(c.ok, true, c.detail);
  assert.ok((c.values ?? 0) > 0, "the case's per-mode DATA was read");
});

for (const name of ["sibling", "traversal", "elsewhere"]) {
  test(`VU11: both doors refuse a readable user file that is not the set's (${name})`, () => {
    const r = resolved(withSet, `read:${name}`);
    assert.equal(r.ok, false);
    assert.match(r.error ?? "", /^user:read reads only the open set's user file .*s\.user\.h5; refused /);
    const c = resolved(withSet, `readCase:${name}`);
    assert.equal(c.ok, false);
    assert.equal(c.reason, "refused");
    assert.match(c.detail ?? "", /^user:readCase reads only the open set's user file .*s\.user\.h5; refused /);
  });
}

test("VU11: user:read judges staleness only against the set's own model and results", () => {
  for (const [name, k] of [["otherModel", "model"], ["otherResults", "results"]] as const) {
    const r = resolved(withSet, `read:${name}`);
    assert.equal(r.ok, false);
    assert.match(r.error ?? "", new RegExp(`^user:read judges staleness only against the open set's ${k} `));
  }
});

test("VU11: a BigInt or circular request resolves with the shape it needed, never rejects", () => {
  for (const name of ["bigint", "circular"]) {
    const r = resolved(withSet, `read:${name}`);
    assert.equal(r.ok, false);
    assert.match(r.error ?? "", /^user:read needs \{ path, model, results \}; got /);
    const c = resolved(withSet, `readCase:${name}`);
    assert.equal(c.ok, false);
    assert.equal(c.reason, "refused");
    assert.match(c.detail ?? "", /^user:readCase needs \{ path, case, target: .*, label \}; got /);
  }
  assert.match(resolved(withSet, "read:bigint").error ?? "", /path: 5n/);
  assert.match(resolved(withSet, "read:circular").error ?? "", /\[Circular \*1\]/);
});

test("#2752 F1: a .user.h5 opened as a model never becomes the set the user doors read", () => {
  assert.equal(resolved(withSet, "steer:1 readCase").ok, false, "before: refused");
  assert.equal(resolved(withSet, "steer:2 model:open user file").ok, false, "a user file does not open as a model");
  // the old model left the screen (#1310 F2), so its set ended; the user file's set was not opened
  assert.equal(setAt(withSet, "steer:3 currentSet"), null, "no set is open, and none beside the user file");
  for (const step of ["steer:4 readCase", "steer:4 read"]) {
    const r = resolved(withSet, step);
    assert.equal(r.ok, false, `${step} is still refused`);
    assert.match(r.error ?? r.detail ?? "", /reads only the open set's user file, and no set is open; refused .*elsewhere/);
  }
});

test("#2752 F1: a model:open that reads moves the set, and the doors follow it", () => {
  assert.equal(resolved(withSet, "steer:5 model:open model").ok, true);
  assert.equal(setAt(withSet, "steer:6 currentSet")?.user, join(root, "elsewhere", "s.user.h5"));
  const c = resolved(withSet, "steer:7 readCase");
  assert.equal(c.ok, true, c.detail);
});

test("#2752 F1 delta: a model that fails to read (still being written) is the set, watched", () => {
  assert.equal(resolved(withSet, "steer:8 model:open truncated").ok, false, "a truncated model does not read");
  // currentSet's user slot is the newest watcher's: it names b's set, so b's watcher replaced the old one
  assert.deepEqual(setAt(withSet, "steer:9 currentSet"), { model: join(root, "bad", "b.h5"), user: join(root, "bad", "b.user.h5") });
  const c = resolved(withSet, "steer:10 readCase");
  assert.equal(c.ok, false);
  assert.match(c.detail ?? "", /reads only the open set's user file .*b\.user\.h5; refused .*elsewhere/);
});

test("VU11: with no set open, both doors refuse even the stem's user file", () => {
  const none = run(false);
  assert.match(resolved(none, "read:paired").error ?? "", /^user:read reads only the open set's user file, and no set is open; refused /);
  const c = resolved(none, "readCase:paired");
  assert.equal(c.reason, "refused");
  assert.match(c.detail ?? "", /^user:readCase reads only the open set's user file, and no set is open; refused /);
});

// ---- VU16 (#2800): `user:readResult`, footfall's state-free user result ---------
//
// The same battery over two more sets: the VU16 plate (test/fixtures/vu16, written
// by W3's own `FootfallResult.to_results`) with its own user file, and the plate's
// model beside W3's five-node user file, whose fem_hash is another fem's.

const VU16 = fileURLToPath(new URL("../fixtures/vu16/", import.meta.url));
type Column = { ids: number[]; values: (number | null)[] };

/** A root the driver reads, its set being `model` and `user` (copied as s.h5 and s.user.h5). */
function vu16Root(tag: string, model: string, user: string): string {
  const r = join(root, tag);
  for (const d of ["set", "outside", "elsewhere"]) mkdirSync(join(r, d), { recursive: true });
  copyFileSync(join(VU16, model), join(r, "set", "s.h5"));
  for (const p of ["set/s.user.h5", "set/t.user.h5", "outside/s.user.h5", "elsewhere/s.user.h5"]) copyFileSync(join(VU16, user), join(r, p));
  return r;
}

function runAt(r: string): Record<string, Answer> {
  const stdout = execFileSync(process.execPath, [DRIVER, `--file=${join(r, "set", "s.h5")}`, `--dir=${r}`], { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], timeout: 60_000 });
  const line = stdout.split(/\r?\n/).find((l) => l.startsWith("USER_DOORS "));
  assert.ok(line, `the driver printed no answers:\n${stdout}`);
  return JSON.parse(line.slice("USER_DOORS ".length)) as Record<string, Answer>;
}

test("VU16: user:readResult gives the set's own footfall_ratio column, equal to what W3 wrote", () => {
  const plate = runAt(vu16Root("vu16-plate", "plate.h5", "plate.user.h5"));
  const r = resolved(plate, "readResult:paired") as { ok: boolean; detail?: string; column?: Column };
  assert.equal(r.ok, true, r.detail);
  const golden = JSON.parse(readFileSync(join(VU16, "plate.golden.json"), "utf8")) as { ids: number[]; components: string[]; data: (number | null)[][] };
  const k = golden.components.indexOf("footfall_ratio");
  assert.deepEqual(r.column!.ids, golden.ids);
  r.column!.values.forEach((v, i) => {
    const want = golden.data[i]![k]!;
    if (want === null) assert.equal(v, null, `node ${golden.ids[i]}: NaN in the file, no value here`);
    else assert.ok(v !== null && Math.abs(v - want) <= 1e-12 * Math.max(1, Math.abs(want)), `node ${golden.ids[i]}: ${v} vs ${want}`);
  });
  assert.ok(r.column!.values.some((v) => v === null) && r.column!.values.some((v) => v !== null), "the fixture holds both values and NaN");
  // Confined as user:read is: a readable copy outside the set is refused, never read.
  for (const name of ["sibling", "traversal", "elsewhere"]) {
    const x = resolved(plate, `readResult:${name}`);
    assert.equal(x.ok, false);
    assert.equal(x.reason, "refused");
    assert.match(x.detail ?? "", /^user:readResult reads only the open set's user file .*s\.user\.h5; refused /);
  }
  const other = resolved(plate, "readResult:otherComponent");
  assert.equal(other.reason, "refused");
  assert.match(other.detail ?? "", /user result 'footfall' has no component 'footfall_nope'; its components are footfall_ap, footfall_ratio, footfall_fdom/);
  for (const name of ["bigint", "circular"]) {
    const x = resolved(plate, `readResult:${name}`);
    assert.equal(x.reason, "refused");
    assert.match(x.detail ?? "", /^user:readResult needs \{ path, name, component \}; got /);
  }
});

test("VU16: user:readResult refuses a stale entry with its A6 reason before reading it", () => {
  const stale = runAt(vu16Root("vu16-stale", "plate.h5", "w3_bay.user.h5"));
  const r = resolved(stale, "readResult:paired");
  assert.equal(r.ok, false);
  assert.equal(r.reason, "stale");
  assert.match(r.detail ?? "", /^user result 'footfall' is stale \(A6\): fem_hash differs: [0-9a-f]+ here, [0-9a-f]+ in the model$/);
});

test("VU16: user:readResult refuses a user file with no such result by name (the San Ramón set)", () => {
  const r = resolved(withSet, "readResult:paired");
  assert.equal(r.ok, false);
  assert.equal(r.reason, "refused");
  assert.match(r.detail ?? "", /holds no user result 'footfall'$/);
  const none = run(false);
  assert.match(resolved(none, "readResult:paired").detail ?? "", /^user:readResult reads only the open set's user file, and no set is open; refused /);
});
