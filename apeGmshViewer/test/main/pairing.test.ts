// File pairing (src/main/pairing.ts): which set opening one file opens, and
// which file a launch asks for. The disk is a Set of paths.

import assert from "node:assert/strict";
import { join, resolve } from "node:path";
import { test } from "node:test";
import { candidates, classify, fileFromArgv, inSet, pairSet, samePath, sameSet, stemSet, unopenedSibling } from "../../src/main/pairing.ts";

const dir = resolve("/runs/frame 1");
const p = (name: string) => join(dir, name);
const disk = (...names: string[]) => {
  const s = new Set(names.map(p));
  return (path: string) => s.has(path);
};

test("classify: the longest suffix wins, case-insensitively", () => {
  assert.deepEqual(classify(p("m.h5")), { kind: "model", stem: p("m") });
  assert.deepEqual(classify(p("m.geometry.h5")), { kind: "geometry", stem: p("m") });
  assert.deepEqual(classify(p("m.results.h5")), { kind: "results", stem: p("m") });
  assert.deepEqual(classify(p("M.Geometry.H5")), { kind: "geometry", stem: p("M") });
});

test("classify: a file the app does not open is refused by name; .mpco is not a sibling", () => {
  assert.throws(() => classify(p("m.txt")), /opens <stem>\.h5.*got .*m\.txt/);
  assert.throws(() => classify(p("m.mpco")), /<stem>\.results\.h5; got .*m\.mpco/);
  assert.throws(() => classify(".h5"), /got \.h5/);
});

test("candidates: the conventional names, and the opened file's own spelling", () => {
  assert.deepEqual(candidates(p("m.h5")), { model: p("m.h5"), geometry: p("m.geometry.h5"), results: p("m.results.h5") });
  assert.deepEqual(candidates(p("M.Geometry.H5")), { model: p("M.h5"), geometry: p("M.Geometry.H5"), results: p("M.results.h5") });
});

// Fable, #1310 finding 1: the set was built from lower-case names only, so a
// file opened under another case dropped out of its own set on a
// case-sensitive disk (and was not reported when rewritten on Windows).
test("an opened file stays in its set whatever the case of its suffix", () => {
  const set = pairSet(p("M.Geometry.H5"), disk("M.Geometry.H5", "M.h5"));
  assert.deepEqual(set, { model: p("M.h5"), geometry: p("M.Geometry.H5"), results: null });
  assert.equal(pairSet(p("m.RESULTS.H5"), disk("m.RESULTS.H5")).results, p("m.RESULTS.H5"));
});

test("samePath folds case only where asked; sameSet and inSet follow it", () => {
  assert.ok(samePath("C:/a/M.H5", "c:/a/m.h5", true));
  assert.ok(!samePath("C:/a/M.H5", "c:/a/m.h5", false));
  const a = { model: "C:/a/M.H5", geometry: null, results: null };
  assert.ok(sameSet(a, { ...a, model: "c:/a/m.h5" }, true));
  assert.ok(!sameSet(a, { ...a, model: "c:/a/m.h5" }, false));
  assert.ok(inSet(a, "c:/A/m.H5", true));
  assert.ok(!inSet(a, "c:/A/m.H5", false));
});

test("opening the model opens its geometry and results siblings", () => {
  const set = pairSet(p("m.h5"), disk("m.h5", "m.geometry.h5", "m.results.h5", "other.geometry.h5"));
  assert.deepEqual(set, { model: p("m.h5"), geometry: p("m.geometry.h5"), results: p("m.results.h5") });
});

test("absent siblings are null; an .mpco is not the results (maintainer, #1310)", () => {
  assert.deepEqual(pairSet(p("m.h5"), disk("m.h5")), { model: p("m.h5"), geometry: null, results: null });
  assert.equal(pairSet(p("m.h5"), disk("m.h5", "m.mpco")).results, null);
});

test("opening the geometry finds its model", () => {
  assert.deepEqual(pairSet(p("m.geometry.h5"), disk("m.h5", "m.geometry.h5")), {
    model: p("m.h5"),
    geometry: p("m.geometry.h5"),
    results: null,
  });
  // No model beside it: the set holds the geometry alone.
  assert.deepEqual(pairSet(p("m.geometry.h5"), disk("m.geometry.h5")), {
    model: null,
    geometry: p("m.geometry.h5"),
    results: null,
  });
});

test("opening the results finds its model and geometry", () => {
  assert.deepEqual(pairSet(p("m.results.h5"), disk("m.h5", "m.geometry.h5", "m.results.h5")), {
    model: p("m.h5"),
    geometry: p("m.geometry.h5"),
    results: p("m.results.h5"),
  });
});

test("pairSet refuses a missing or relative path", () => {
  assert.throws(() => pairSet(p("m.h5"), disk()), /no such file: .*m\.h5/);
  assert.throws(() => pairSet("m.h5", () => true), /absolute path/);
});

test("sameSet compares the three paths", () => {
  const a = { model: "a", geometry: null, results: null };
  assert.ok(sameSet(a, { ...a }));
  assert.ok(!sameSet(a, { ...a, geometry: "g" }));
  assert.ok(sameSet(null, null));
  assert.ok(!sameSet(a, null));
});

test("fileFromArgv: packaged double-click, unpackaged --file, second-instance switches", () => {
  const cwd = resolve("/work");
  // Packaged: `apeGmshViewer.exe "C:\...\m.h5"` (skip the exe).
  assert.equal(fileFromArgv(["app.exe", p("m.h5")], 1, cwd), p("m.h5"));
  // Unpackaged: `electron <app> --mode=view --file=...` (skip exe and app).
  assert.equal(fileFromArgv(["electron", "C:/app", "--mode=view", `--file=${p("m.h5")}`], 2, cwd), p("m.h5"));
  // Chromium appends switches to a second instance's argv.
  assert.equal(
    fileFromArgv(["app.exe", "--allow-file-access-from-files", "m.h5", "--original-process-start-time=1"], 1, cwd),
    join(cwd, "m.h5"),
  );
  assert.equal(fileFromArgv(["app.exe"], 1, cwd), null);
  assert.equal(fileFromArgv(["electron", "C:/app", "--mode=view"], 2, cwd), null);
});

// -- V4L-b1 (#2127): the .ladruno is a results file of the set (ADR 0122 D1) --

test("classify: <stem>.ladruno is the results kind, case-insensitively", () => {
  assert.deepEqual(classify(p("m.ladruno")), { kind: "results", stem: p("m") });
  assert.deepEqual(classify(p("M.LaDrUnO")), { kind: "results", stem: p("M") });
  assert.deepEqual(candidates(p("m.ladruno")), { model: p("m.h5"), geometry: p("m.geometry.h5"), results: p("m.ladruno") });
});

// -- V8-b (#2188): a part set is the results of its stem (ADR 0122 slice 8) --

test("classify: a <stem>.part-<N>.ladruno is the results kind of <stem>, case-insensitively", () => {
  assert.deepEqual(classify(p("m.part-0.ladruno")), { kind: "results", stem: p("m") });
  assert.deepEqual(classify(p("m.PART-12.LADRUNO")), { kind: "results", stem: p("m") });
  assert.deepEqual(classify(p("a.b.part-3.ladruno")), { kind: "results", stem: p("a.b") });
});

test("opening any part opens the set under the stem; the results slot keeps the opened part", () => {
  const d = disk("m.h5", "m.part-0.ladruno", "m.part-1.ladruno");
  assert.deepEqual(pairSet(p("m.part-1.ladruno"), d), { model: p("m.h5"), geometry: null, results: p("m.part-1.ladruno") });
  assert.deepEqual(pairSet(p("m.part-0.ladruno"), d), { model: p("m.h5"), geometry: null, results: p("m.part-0.ladruno") });
});

test("opening the model finds the part set by part 0; the .ladruno outranks it, and it outranks the .results.h5", () => {
  assert.equal(pairSet(p("m.h5"), disk("m.h5", "m.part-0.ladruno", "m.part-1.ladruno")).results, p("m.part-0.ladruno"));
  assert.equal(pairSet(p("m.h5"), disk("m.h5", "m.part-0.ladruno", "m.results.h5")).results, p("m.part-0.ladruno"));
  assert.equal(pairSet(p("m.h5"), disk("m.h5", "m.ladruno", "m.part-0.ladruno")).results, p("m.ladruno"));
  // An opened results file wins: the part set when a part was opened, the .ladruno when it was.
  assert.equal(pairSet(p("m.part-1.ladruno"), disk("m.ladruno", "m.part-0.ladruno", "m.part-1.ladruno")).results, p("m.part-1.ladruno"));
  assert.equal(pairSet(p("m.ladruno"), disk("m.ladruno", "m.part-0.ladruno")).results, p("m.ladruno"));
});

test("unopenedSibling: a part names the stem's .ladruno or .results.h5, never another part of its own set", () => {
  assert.equal(unopenedSibling(p("m.part-1.ladruno"), disk("m.part-0.ladruno", "m.part-1.ladruno")), null);
  assert.equal(unopenedSibling(p("m.part-1.ladruno"), disk("m.part-1.ladruno", "m.results.h5")), p("m.results.h5"));
  assert.equal(unopenedSibling(p("m.part-0.ladruno"), disk("m.part-0.ladruno", "m.ladruno", "m.results.h5")), p("m.ladruno"));
  assert.equal(unopenedSibling(p("m.ladruno"), disk("m.ladruno", "m.part-0.ladruno")), p("m.part-0.ladruno"));
  assert.equal(unopenedSibling(p("m.results.h5"), disk("m.results.h5", "m.part-0.ladruno")), p("m.part-0.ladruno"));
});

test("opening the model finds a .ladruno beside it when it is the only results file", () => {
  assert.deepEqual(pairSet(p("m.h5"), disk("m.h5", "m.ladruno")), { model: p("m.h5"), geometry: null, results: p("m.ladruno") });
  assert.deepEqual(pairSet(p("m.ladruno"), disk("m.h5", "m.geometry.h5", "m.ladruno")), {
    model: p("m.h5"),
    geometry: p("m.geometry.h5"),
    results: p("m.ladruno"),
  });
});

// The T ruling (2026-10-16): the opened file wins if it is a results file; otherwise the .ladruno.
test("both results files beside the model: one results slot; the opened one wins, else the .ladruno", () => {
  const both = disk("m.h5", "m.geometry.h5", "m.results.h5", "m.ladruno");
  assert.equal(pairSet(p("m.h5"), both).results, p("m.ladruno"), "from the model: the .ladruno (canonical)");
  assert.equal(pairSet(p("m.geometry.h5"), both).results, p("m.ladruno"), "from the geometry: the .ladruno");
  assert.equal(pairSet(p("m.results.h5"), both).results, p("m.results.h5"), "the opened native file wins");
  assert.equal(pairSet(p("m.ladruno"), both).results, p("m.ladruno"), "the opened .ladruno wins");
});

test("the watcher's re-evaluation keeps the rule: the opened results file wins while it exists", () => {
  const c = candidates(p("m.results.h5"));
  assert.equal(stemSet(c, disk("m.h5", "m.results.h5", "m.ladruno")).results, p("m.results.h5"));
  assert.equal(stemSet(c, disk("m.h5", "m.ladruno")).results, p("m.ladruno"), "the opened file gone: the other one");
  const fromModel = candidates(p("m.h5"));
  assert.equal(stemSet(fromModel, disk("m.h5", "m.results.h5")).results, p("m.results.h5"), "no .ladruno: the native file, as before");
  assert.equal(stemSet(fromModel, disk("m.h5", "m.results.h5", "m.ladruno")).results, p("m.ladruno"));
});

test("stemSet refuses Candidates candidates() did not make (it could not know which file was opened)", () => {
  assert.throws(() => stemSet({ model: p("m.h5"), geometry: p("m.geometry.h5"), results: p("m.results.h5") }, () => true), /candidates\(\) made/);
});

test("unopenedSibling: the stem's other results file when it exists, for the banner", () => {
  const both = disk("m.results.h5", "m.ladruno");
  assert.equal(unopenedSibling(p("m.ladruno"), both), p("m.results.h5"));
  assert.equal(unopenedSibling(p("m.results.h5"), both), p("m.ladruno"));
  assert.equal(unopenedSibling(p("m.ladruno"), disk("m.ladruno")), null);
  assert.equal(unopenedSibling(p("results_small.h5"), () => true), null, "a file read as results under another name has no pair");
});
