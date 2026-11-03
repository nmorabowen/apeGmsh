// VU12 (#2753): main's results and geometry doors (`results:open`,
// `results:readStep`, `results:scan`, `geometry:open`) read only the window's
// set's own files, and a request their log line cannot JSON-encode (a BigInt,
// a cycle) still resolves with the reason instead of rejecting.
//
// The oracle is the handlers themselves, as in VU11's userDoors.test.ts: main.ts
// runs in a child Node process with `electron` stubbed (setDoorsDriver.ts), opens
// its set the way each mode does (a double-click's `--file=`, a drop, a `--render`
// spec, a capture's `--file=`), and answers a fixed battery. Every path it must
// refuse is a READABLE results or geometry file, so without the check
// `results:open` and `geometry:open` would read each `ok: true`: a sibling, a
// `..` path out of the set's folder, and an absolute path elsewhere. The host
// already refuses a step or a scan of a file it has not indexed, so those two
// doors are also judged on the file the set has left after a drop, which the
// host still indexes and would read.

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { copyFileSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import { SPEC_VERSION } from "../../src/spec/spec.ts";

const FIX = fileURLToPath(new URL("../../fixtures/", import.meta.url));
const DRIVER = fileURLToPath(new URL("./setDoorsDriver.ts", import.meta.url));

const root = mkdtempSync(join(tmpdir(), "vu12-doors-"));
after(() => rmSync(root, { recursive: true, force: true }));
for (const d of ["set", "outside", "elsewhere"]) mkdirSync(join(root, d));
const copies: [string, string[]][] = [
  ["ladruno_frame3d.h5", ["set/s.h5", "elsewhere/s.h5"]],
  ["ladruno_frame3d.ladruno", ["set/s.ladruno", "outside/s.ladruno", "elsewhere/s.ladruno"]],
  // the native results file beside the `.ladruno`: the set's results slot only when a spec names it
  ["contour_staged.results.h5", ["set/s.results.h5"]],
  ["zones.geometry.h5", ["set/s.geometry.h5", "set/t.geometry.h5", "outside/s.geometry.h5", "elsewhere/s.geometry.h5"]],
];
for (const [from, tos] of copies) for (const to of tos) copyFileSync(join(FIX, from), join(root, to));
const spec = join(root, "spec.json");
writeFileSync(spec, JSON.stringify({ spec: SPEC_VERSION, files: { model: join(root, "set", "s.h5"), results: join(root, "set", "s.results.h5") }, shots: [{ name: "a" }] }));

type Seen = { ok: boolean; error?: string; seq?: number; scanId?: number; stages?: number; geometry?: boolean };
type Answers = Record<string, { resolved: Seen } | { rejected: string }>;

function run(which: string, mainArgs: string[]): Answers {
  const stdout = execFileSync(process.execPath, [DRIVER, `--case=${which}`, `--dir=${root}`, ...mainArgs], { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], timeout: 60_000 });
  const line = stdout.split(/\r?\n/).find((l) => l.startsWith("SET_DOORS "));
  assert.ok(line, `the driver printed no answers:\n${stdout}`);
  return JSON.parse(line.slice("SET_DOORS ".length)) as Answers;
}

/** The answer, which must have resolved: a door never rejects. */
function resolved(answers: Answers, name: string): Seen {
  const a = answers[name];
  assert.ok(a, `no answer for ${name}`);
  assert.ok("resolved" in a, `${name} rejected: ${"rejected" in a ? a.rejected : ""}`);
  return a.resolved;
}

const re = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const SET = join(root, "set");

/** The doors read the set's own files and refuse each readable file in `refused` by name. */
function judgeBattery(answers: Answers, ownResults: string, refused: string[], scanned: boolean): void {
  const own = resolved(answers, "results:open own");
  assert.equal(own.ok, true, own.error);
  assert.ok((own.stages ?? 0) > 0, "the set's results file was indexed");
  const g = resolved(answers, "geometry:open own");
  assert.equal(g.ok, true, g.error);
  assert.equal(g.geometry, true, "the set's /geometry zone was read");
  const st = resolved(answers, "results:readStep own");
  assert.equal(st.ok, true, st.error);
  if (scanned) assert.equal(resolved(answers, "results:scan own").ok, true, resolved(answers, "results:scan own").error);
  refused.forEach((k, i) => {
    const o = resolved(answers, `results:open ${k}`);
    assert.equal(o.ok, false, `results:open ${k} must be refused`);
    assert.match(o.error ?? "", new RegExp(`^results:open reads only the open set's results file ${re(ownResults)}; refused `));
    const s = resolved(answers, `results:readStep ${k}`);
    assert.equal(s.ok, false);
    assert.equal(s.seq, i + 1, "the refusal echoes the request's seq");
    assert.match(s.error ?? "", new RegExp(`^results:readStep reads only the open set's results file ${re(ownResults)}; refused `));
    const c = resolved(answers, `results:scan ${k}`);
    assert.equal(c.ok, false);
    assert.equal(c.scanId, i + 1, "the refusal echoes the request's scanId");
    assert.match(c.error ?? "", new RegExp(`^results:scan reads only the open set's results file ${re(ownResults)}; refused `));
  });
  for (const k of ["sibling", "traversal", "elsewhere"]) {
    const o = resolved(answers, `geometry:open ${k}`);
    assert.equal(o.ok, false, `geometry:open ${k} must be refused`);
    assert.match(o.error ?? "", new RegExp(`^geometry:open reads only the open set's geometry file ${re(join(SET, "s.geometry.h5"))}; refused `));
  }
}

const view = run("view", [`--file=${join(SET, "s.h5")}`]);

test("VU12: a set opened as a double-click opens it reads its own results and geometry, and refuses a readable sibling, a .. path and a file elsewhere", () => {
  judgeBattery(view, join(SET, "s.ladruno"), ["native", "traversal", "elsewhere"], true);
});

test("VU12: a BigInt or circular request resolves with the shape it needed, never rejects", () => {
  for (const door of ["results:open", "geometry:open"]) {
    assert.match(resolved(view, `${door} bigint`).error ?? "", new RegExp(`^${door} needs a path; got 5n$`));
    assert.match(resolved(view, `${door} circular`).error ?? "", new RegExp(`^${door} needs a path; got .*\\[Circular \\*1\\]`));
  }
  for (const door of ["results:readStep", "results:scan"]) {
    assert.match(resolved(view, `${door} bigint`).error ?? "", new RegExp(`^${door}: malformed request .*path: 5n`));
    assert.match(resolved(view, `${door} circular`).error ?? "", new RegExp(`^${door}: malformed request .*\\[Circular \\*1\\]`));
  }
  const ro = resolved(view, "viewer:requestOpen circular");
  assert.equal(ro.ok, false);
  assert.match(ro.error ?? "", /^requestOpen needs a path; got .*\[Circular \*1\]/);
  const oo = resolved(view, "viewer:offerOpen bigint");
  assert.equal(oo.ok, false);
  assert.match(oo.error ?? "", /^offerOpen needs a path; got 5n$/);
});

test("VU12: a drop moves the set; the set it left is refused, though the host still indexes its results", () => {
  assert.equal(resolved(view, "drop:1 requestOpen").ok, true);
  const newResults = join(root, "elsewhere", "s.ladruno");
  for (const name of ["drop:2 readStep old", "drop:2 scan old", "drop:3 open old"]) {
    const r = resolved(view, name);
    assert.equal(r.ok, false, `${name} is refused`);
    assert.match(r.error ?? "", new RegExp(`reads only the open set's results file ${re(newResults)}; refused ${re(join(SET, "s.ladruno"))}$`));
  }
  assert.match(resolved(view, "drop:3 geometry old").error ?? "", new RegExp(`reads only the open set's geometry file ${re(join(root, "elsewhere", "s.geometry.h5"))}; refused `));
  for (const name of ["drop:4 open new", "drop:4 readStep new", "drop:4 geometry new"]) assert.equal(resolved(view, name).ok, true, resolved(view, name).error);
});

test("#1310 F2 kept: a results file read as a model ends the set, and the doors then refuse even its files", () => {
  assert.equal(resolved(view, "model:1 open a results file").ok, false);
  for (const [name, slot] of [["model:2 open new", "results"], ["model:2 geometry new", "geometry"]] as const) {
    assert.match(resolved(view, name).error ?? "", new RegExp(`reads only the open set's ${slot} file, and no set is open; refused `));
  }
});

test("VU12: with no set open, the doors refuse even the stem's own files", () => {
  const none = run("none", []);
  for (const [name, slot] of [["results:open ladruno", "results"], ["geometry:open paired", "geometry"]] as const) {
    const r = resolved(none, name);
    assert.equal(r.ok, false);
    assert.match(r.error ?? "", new RegExp(`reads only the open set's ${slot} file, and no set is open; refused `));
  }
});

test("VU12: --render reads the job's set: the spec's native results file, and not the .ladruno beside it", () => {
  const out = join(root, "render-out");
  judgeBattery(run("render", ["--mode=render", `--spec=${spec}`, `--out=${out}`]), join(SET, "s.results.h5"), ["ladruno", "traversal", "elsewhere"], true);
});

test("VU12: a capture reads the set of its --file", () => {
  judgeBattery(run("capture", ["--mode=capture", `--file=${join(SET, "s.h5")}`, `--out=${join(root, "capture.png")}`]), join(SET, "s.ladruno"), ["native", "traversal", "elsewhere"], false);
});
