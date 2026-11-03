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
import { copyFileSync, mkdirSync, mkdtempSync, rmSync } from "node:fs";
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

type Answer = { resolved: { ok: boolean; error?: string; reason?: string; detail?: string; values?: number } } | { rejected: string };

function run(withSet: boolean): Record<string, Answer> {
  const args = [DRIVER, ...(withSet ? [`--file=${join(root, "set", "s.h5")}`] : []), `--dir=${root}`];
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

test("VU11: with no set open, both doors refuse even the stem's user file", () => {
  const none = run(false);
  assert.match(resolved(none, "read:paired").error ?? "", /^user:read reads only the open set's user file, and no set is open; refused /);
  const c = resolved(none, "readCase:paired");
  assert.equal(c.reason, "refused");
  assert.match(c.detail ?? "", /^user:readCase reads only the open set's user file, and no set is open; refused /);
});
