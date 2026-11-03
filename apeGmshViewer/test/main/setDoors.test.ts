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
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { SPEC_VERSION } from "../../src/spec/spec.ts";
import { tree, writeTree } from "../zones/tree.ts";

const FIX = fileURLToPath(new URL("../../fixtures/", import.meta.url));
const DRIVER = fileURLToPath(new URL("./setDoorsDriver.ts", import.meta.url));

const root = mkdtempSync(join(tmpdir(), "vu12-doors-"));
after(() => rmSync(root, { recursive: true, force: true }));
for (const d of ["set", "outside", "elsewhere"]) mkdirSync(join(root, d));
const copies: [string, string[]][] = [
  // outside/s.h5: a readable model the user never dropped (VU12's free string)
  // outside/loose.user.h5: a model read from a file that is no set's model (VU13's unpaired drop)
  ["ladruno_frame3d.h5", ["set/s.h5", "elsewhere/s.h5", "outside/s.h5", "outside/loose.user.h5"]],
  ["ladruno_frame3d.ladruno", ["set/s.ladruno", "outside/s.ladruno", "elsewhere/s.ladruno"]],
  // the native results file beside the `.ladruno`: the set's results slot only when a spec names it
  ["contour_staged.results.h5", ["set/s.results.h5"]],
  ["zones.geometry.h5", ["set/s.geometry.h5", "set/t.geometry.h5", "outside/s.geometry.h5", "elsewhere/s.geometry.h5"]],
];
for (const [from, tos] of copies) for (const to of tos) copyFileSync(join(FIX, from), join(root, to));
const spec = join(root, "spec.json");
writeFileSync(spec, JSON.stringify({ spec: SPEC_VERSION, files: { model: join(root, "set", "s.h5"), results: join(root, "set", "s.results.h5") }, shots: [{ name: "a" }] }));
// what a second instance's `--present` offers the view window: the elsewhere set
writeFileSync(join(root, "present.json"), JSON.stringify({ spec: SPEC_VERSION, files: { model: join(root, "elsewhere", "s.h5") }, shots: [{ name: "o" }] }));
// VU13: newer offers (the outside set), sent until the elsewhere offer falls off the page's list
writeFileSync(join(root, "flood.json"), JSON.stringify({ spec: SPEC_VERSION, files: { model: join(root, "outside", "s.h5") }, shots: [{ name: "f" }] }));

// VU13 (#2760): a set whose model's /provenance names a script inside the set's
// folder (a row relative to @base_dir, as the writer stores it) and a helper
// outside it (an absolute row, h5-schema.md "Files", Q8), plus a pseudo-file.
// Readable strangers sit in the set's folder and beside it.
await h5wasm.ready;
for (const d of ["src-set", "lib"]) mkdirSync(join(root, d));
const posix = (p: string) => p.replaceAll("\\", "/");
const SRC_SET = join(root, "src-set");
for (const f of [join(SRC_SET, "build.py"), join(root, "lib", "helper.py"), join(SRC_SET, "stranger.py"), join(root, "stranger.py")]) writeFileSync(f, "print('hi')\n");
writeTree(h5wasm, join(SRC_SET, "s.h5"), tree({
  meta: { attrs: { provenance_schema_version: "1.0.0" } },
  provenance: {
    attrs: { base_dir: posix(SRC_SET) },
    files: { path: ["build.py", posix(join(root, "lib", "helper.py")), "<string>"], sha256: ["a".repeat(64), "", ""], kind: ["script", "module", "module"] },
    sites: { file: new Int32Array([0, 1]), line: new Int32Array([3, 4]), function: ["<module>", "helper"] },
    records: { path: ["opensees/section/A", "opensees/section/B"], site: new Int32Array([0, 1]), script: new Int32Array([0, 0]), seq: new Int32Array([0, 1]) },
  },
}));

type Seen = { ok: boolean; error?: string; seq?: number; scanId?: number; stages?: number; geometry?: boolean };
type Answers = Record<string, { resolved: Seen } | { rejected: string } | { onScreen: string | null } | { sent: number }>;

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
  assert.match(oo.error ?? "", /^offerOpen needs a path and the files it needs; got 5n, \[\]$/);
});

/** The model on screen after a step, as the driver recorded it. */
const onScreen = (name: string) => {
  const a = view[name];
  assert.ok(a && "onScreen" in a, `no set recorded for ${name}`);
  return a.onScreen;
};

test("VU12: offerOpen opens only a set main offered, or the set an offer left; anything else is refused and the set stays", () => {
  const here = join(SET, "s.h5");
  const there = join(root, "elsewhere", "s.h5");
  const un = resolved(view, "offer:1 un-offered");
  assert.equal(un.ok, false);
  assert.match(un.error ?? "", /^offerOpen opens only the files of a set main offered or the set an offer left; refused .*outside/);
  assert.equal(onScreen("offer:1 model"), here, "a refused offer leaves the set where it was");
  assert.match(resolved(view, "offer:2 empty needs").error ?? "", /^offerOpen needs a path and the files it needs; got .*, \[\]$/);
  assert.match(resolved(view, "offer:3 un-offered need").error ?? "", /refused .*outside/);
  assert.equal(onScreen("offer:3 model"), here);
  assert.equal(resolved(view, "offer:4 offered").ok, true, resolved(view, "offer:4 offered").error);
  assert.equal(onScreen("offer:4 model"), there, "the offered set opened");
  assert.equal(resolved(view, "offer:5 back").ok, true, resolved(view, "offer:5 back").error);
  assert.equal(onScreen("offer:5 model"), here, "Back reopened the set the offer left");
});

test("VU12 (option A): requestOpen and model:open refuse a path the page named with no gesture, and the set stays", () => {
  const here = join(SET, "s.h5");
  for (const [name, door] of [["gesture:1 requestOpen free", "requestOpen"], ["gesture:2 model:open free", "model:open"]] as const) {
    const r = resolved(view, name);
    assert.equal(r.ok, false, `${name} is refused`);
    assert.match(r.error ?? "", new RegExp(`^${door} opens only a file the user dropped or a path main sent the page; refused .*outside`));
  }
  assert.equal(onScreen("gesture:1 model"), here, "a refused requestOpen leaves the set where it was");
  assert.equal(onScreen("gesture:2 model"), here, "a refused model:open leaves the set where it was (no re-pair, no close)");
});

test("VU12 (option A): the argv set's paths, which main sent the page, still open through both doors", () => {
  assert.equal(resolved(view, "gesture:3 requestOpen argv").ok, true, resolved(view, "gesture:3 requestOpen argv").error);
  assert.equal(resolved(view, "gesture:3 model:open argv").ok, true, resolved(view, "gesture:3 model:open argv").error);
  assert.equal(onScreen("gesture:3 model"), join(SET, "s.h5"));
});

test("VU12 (option A): a model the user dropped opens through model:open, and the set follows it", () => {
  assert.equal(resolved(view, "model:3 model:open dropped").ok, true, resolved(view, "model:3 model:open dropped").error);
  assert.equal(onScreen("model:3 model"), join(root, "outside", "s.h5"));
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

for (const m of ["describe", "measure"]) {
  test(`VU12: --mode=${m} reads the set of its --file`, () => {
    const outArg = m === "describe" ? join(root, "describe-out") : join(root, "measure.json");
    judgeBattery(run(m, [`--mode=${m}`, `--file=${join(SET, "s.h5")}`, `--out=${outArg}`]), join(SET, "s.ladruno"), ["native", "traversal", "elsewhere"], true);
  });
}

test("VU13 (#2759 review): only the latest drop is a gesture; a replaced drop's model:open is refused, the latest re-reads", () => {
  const r = resolved(view, "drop:5 model:open replaced drop");
  assert.equal(r.ok, false, "a drop the user has since replaced opens nothing");
  assert.match(r.error ?? "", new RegExp(`^model:open opens only a file the user dropped or a path main sent the page; refused ${re(join(root, "elsewhere", "s.h5"))}$`));
  assert.equal(onScreen("drop:5 model"), join(root, "outside", "s.h5"), "the set stays");
  assert.equal(resolved(view, "drop:6 model:open unpaired drop").ok, true, resolved(view, "drop:6 model:open unpaired drop").error);
  assert.equal(onScreen("drop:6 model"), null, "a model read from a .user.h5 is no set (#2752 review F1), so only the drop lets it re-read");
  assert.equal(resolved(view, "drop:7 model:open unpaired drop again").ok, true, `the latest drop is not spent by its open: ${resolved(view, "drop:7 model:open unpaired drop again").error}`);
});

test("VU13: an offer that falls off the page's list (MAX_OFFERS) is gone, and so is the set it left", async () => {
  const { MAX_OFFERS } = await import("../../src/state/reduce.ts");
  assert.equal(resolved(view, "prune:1 still listed").ok, true, `${MAX_OFFERS - 1} newer offers leave it listed: ${resolved(view, "prune:1 still listed").error}`);
  const gone = resolved(view, "prune:2 fallen off");
  assert.equal(gone.ok, false, `the ${MAX_OFFERS}th newer offer drops it`);
  assert.match(gone.error ?? "", /^offerOpen opens only the files of a set main offered or the set an offer left; refused '.*elsewhere.{1,2}s\.ladruno'$/);
  assert.equal(onScreen("prune:2 model"), join(root, "elsewhere", "s.h5"), "a refused offer leaves the set where it was");
  const back = resolved(view, "prune:3 back to the set it left");
  assert.equal(back.ok, false, "the set offer:4 left went with its offer");
  assert.match(back.error ?? "", /^offerOpen opens only the files of a set main offered or the set an offer left; refused /);
});

// VU14 (#2771): main.ts makes its (stub) window here, and the driver delivers the page's loads.
test("VU14: a page load leaves no offer openable that the new page is not sent; an offer still waiting stays", () => {
  const r = run("reload", [`--file=${join(SET, "s.h5")}`]);
  const shown = (name: string) => {
    const a = r[name];
    assert.ok(a && "onScreen" in a, `no set recorded for ${name}`);
    return a.onScreen;
  };
  const sent = (name: string) => {
    const a = r[name];
    assert.ok(a && "sent" in a, `no send count for ${name}`);
    return a.sent;
  };
  assert.equal(sent("reload:0 sent"), 1, "the page listened, so the offer was sent at once");
  assert.equal(resolved(r, "reload:1 listed offer").ok, true, resolved(r, "reload:1 listed offer").error);
  assert.equal(shown("reload:1 model"), join(root, "elsewhere", "s.h5"));
  const stale = resolved(r, "reload:2 offer the old page listed");
  assert.equal(stale.ok, false, "the reloaded page lists no offers, so none opens");
  assert.match(stale.error ?? "", /^offerOpen opens only the files of a set main offered or the set an offer left; refused '.*elsewhere.{1,2}s\.ladruno'$/);
  const back = resolved(r, "reload:3 the set that offer left");
  assert.equal(back.ok, false, "the set the offer left went with it");
  assert.match(back.error ?? "", /^offerOpen opens only the files of a set main offered or the set an offer left; refused /);
  assert.equal(shown("reload:3 model"), join(root, "elsewhere", "s.h5"), "a refused offer leaves the set where it was");
  assert.equal(sent("reload:4 sent"), 1, "an offer that arrives while the page loads waits");
  assert.equal(sent("reload:5 sent"), 2, "and is sent once the reloaded page listens");
  assert.equal(resolved(r, "reload:5 waiting offer").ok, true, `a waiting offer survives a page load: ${resolved(r, "reload:5 waiting offer").error}`);
  assert.equal(shown("reload:5 model"), join(root, "outside", "s.h5"));
});

// VU15 (#2776): the oracle is the count of `viewer:offers` sends. A stale offersReady that re-armed the page would
// send the waiting batch at once (and every later one), so each refused message leaves the count at 0.
test("VU15: offersReady from a page being replaced, another frame or another webContents re-arms nothing; the current load's page does", () => {
  const r = run("stale", [`--file=${join(SET, "s.h5")}`]);
  const sent = (name: string) => {
    const a = r[name];
    assert.ok(a && "sent" in a, `no send count for ${name}`);
    return a.sent;
  };
  assert.equal(sent("stale:1 replaced page, mid-load"), 0, "the page being replaced, before the new one commits");
  assert.equal(sent("stale:2 replaced page, after commit"), 0, "the page being replaced, after the new one commits in another frame");
  assert.equal(sent("stale:3 another frame"), 0, "a frame that is not the window's page");
  assert.equal(sent("stale:4 another webContents"), 0, "another webContents with the page's frame ids");
  assert.equal(sent("stale:5 no sender"), 0, "a message with no sender");
  assert.equal(sent("stale:6 a later offer"), 0, "no stale message armed the page, so a later offer waits too");
  assert.equal(sent("stale:7 current page"), 2, "the current load's page listens: both waiting batches are sent");
  assert.equal(sent("stale:9 same frame, mid-load"), 2, "a load that will commit in the same frame: before the commit, the message is the old page's");
  assert.equal(sent("stale:10 same frame, committed"), 3, "and after it, the new page's");
});

const source = run("source", [`--file=${join(SRC_SET, "s.h5")}`]);
const editorLog = () => (existsSync(join(root, "editor.log")) ? readFileSync(join(root, "editor.log"), "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l) as string[]) : []);

test("VU13 (#2760): source:goto opens a source file the set's /provenance names: a row relative to @base_dir, and one outside the set's folder", () => {
  for (const name of ["source:1 relative row", "source:2 absolute row outside the folder"]) assert.equal(resolved(source, name).ok, true, resolved(source, name).error);
  // the editor (a recorder) received each file as /provenance names it, and nothing else launched
  assert.deepEqual(editorLog().sort(), [["3", posix(join(SRC_SET, "build.py"))], ["4", posix(join(root, "lib", "helper.py"))]].sort());
});

test("VU13 (#2760): source:goto refuses a readable file the set's /provenance does not name, in the set's folder or beside it", () => {
  for (const [name, file] of [["source:3 stranger in the folder", posix(join(SRC_SET, "stranger.py"))], ["source:4 stranger elsewhere", join(root, "stranger.py")]] as const) {
    const r = resolved(source, name);
    assert.equal(r.ok, false, `${name} is refused`);
    assert.equal(r.error, `goToSource opens only a source file the open set's /provenance names; refused ${file}`);
  }
});

test("VU13 (#2760): a BigInt or circular source:goto resolves with the reason, never rejects", () => {
  assert.equal(resolved(source, "source:5 bigint path").error, "goToSource needs an absolute file path; got 5n");
  assert.match(resolved(source, "source:6 circular path").error ?? "", /^goToSource needs an absolute file path; got .*\[Circular \*1\]/);
  assert.equal(resolved(source, "source:7 bigint line").error, "goToSource needs a line number >= 1; got 5n");
});

test("VU13 (#2760): with no set open, source:goto opens nothing, even a file a set's /provenance names", () => {
  rmSync(join(root, "editor.log"), { force: true });
  const none = run("source-none", []);
  const r = resolved(none, "source:1 relative row");
  assert.equal(r.ok, false);
  assert.match(r.error ?? "", /^goToSource opens only a source file the open set's \/provenance names, and no set is open; refused /);
  assert.deepEqual(editorLog(), [], "no editor launched");
});

test("VU12: a capture reads the set of its --file", () => {
  judgeBattery(run("capture", ["--mode=capture", `--file=${join(SET, "s.h5")}`, `--out=${join(root, "capture.png")}`]), join(SET, "s.ladruno"), ["native", "traversal", "elsewhere"], false);
});
