// ADR 0135 D2 and D3: the runner, against a stand-in executable. CI has no fork;
// live.test.ts runs the real one where $APEGMSH_OPENSEES names it.

import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import { MaterialTestRunner, resolveSolver, type SolverAnswer } from "../../src/main/materialTest.ts";

const STAND_IN = fileURLToPath(new URL("./standInSolver.mjs", import.meta.url));
const root = mkdtempSync(join(tmpdir(), "material-runner-"));
after(() => rmSync(root, { recursive: true, force: true }));

const solver = (mode: string, extra: NodeJS.ProcessEnv = {}) => (): SolverAnswer => ({
  ok: true,
  launch: { command: process.execPath, args: [STAND_IN], env: { ...process.env, STAND_IN: mode, ...extra } },
});
const job = { target: 1, materials: [{ type: "Steel02", tag: 1, args: [420, 200000, 0.01] }], protocol: { kind: "monotonic", peak: 1, steps: 2 } };

test("a complete run returns its curves, from a working directory that is then gone", async () => {
  const where = join(root, "cwd.txt");
  const r = await new MaterialTestRunner(solver("ok", { STAND_IN_CWD: where })).run(job, 7);
  assert.ok(r.ok, JSON.stringify(r));
  assert.equal(r.seq, 7);
  assert.equal(r.build, "abc123");
  assert.deepEqual([...r.strain], [0, 0.5, 1]);
  assert.deepEqual([...r.stress], [0, 1, 2]);
  assert.equal(existsSync(readFileSync(where, "utf8")), false);
});

for (const [mode, why] of [
  ["material-error", /stopped after 0 of 3.*WARNING bad material/s],
  ["short", /stopped after 1 of 3/],
  ["no-end", /stopped after 3 of 3/],
  ["nan", /step 1/],
  ["nofork", /not a Ladruno fork build/],
] as const) {
  test(`exit 0 proves nothing: ${mode} gives no curve`, async () => {
    const r = await new MaterialTestRunner(solver(mode)).run(job, 1);
    assert.ok(!r.ok);
    assert.match(r.error, why);
  });
}

test("a hung solver is stopped at the time limit", async () => {
  const r = await new MaterialTestRunner(solver("hang"), 300).run(job, 1);
  assert.ok(!r.ok);
  assert.match(r.error, /longer than 0\.3 s/);
});

test("a flooding solver is stopped at the output limit", async () => {
  const r = await new MaterialTestRunner(solver("flood"), 10_000, 1024 * 1024).run(job, 1);
  assert.ok(!r.ok);
  assert.match(r.error, /more than 1 MB/);
});

test("a refused job starts nothing", async () => {
  let asked = false;
  const r = await new MaterialTestRunner(() => ((asked = true), solver("ok")())).run({ ...job, materials: [{ type: "Steel02", tag: 1, args: ["[exec calc]"] }] }, 1);
  assert.ok(!r.ok);
  assert.match(r.error, /refused/);
  assert.equal(asked, false);
});

test("a new run supersedes the one going, and cancel stops one", async () => {
  const runner = new MaterialTestRunner(solver("hang"));
  const first = runner.run(job, 1);
  await new Promise((f) => setTimeout(f, 200));
  const second = runner.run(job, 2);
  const a = await first;
  assert.ok(!a.ok && a.superseded);
  await new Promise((f) => setTimeout(f, 200));
  runner.cancel();
  const b = await second;
  assert.ok(!b.ok);
  assert.match(b.error, /cancelled|replaced/);
});

test("the executable comes from the environment or the install, and a broken choice is the answer", () => {
  const exe = join(root, "OpenSees.exe");
  writeFileSync(exe, "");
  const chosen = resolveSolver({ APEGMSH_OPENSEES: exe, PATH: "old" }, join(root, "none"));
  assert.ok(chosen.ok);
  assert.equal(chosen.launch.command, exe);
  assert.ok(chosen.launch.env["PATH"]!.startsWith(dirname(exe)) && chosen.launch.env["PATH"]!.endsWith("old"));
  assert.equal(resolveSolver({ APEGMSH_OPENSEES: "OpenSees.exe" }, exe).ok, false);
  assert.equal(resolveSolver({ APEGMSH_OPENSEES: join(root, "missing.exe") }, exe).ok, false);
  const fallback = resolveSolver({}, exe);
  assert.ok(fallback.ok);
  assert.equal(fallback.launch.command, exe);
  assert.equal(resolveSolver({}, join(root, "none")).ok, false);
});
