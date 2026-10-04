// File pairing (src/main/pairing.ts): which set opening one file opens, and
// which file a launch asks for. The disk is a Set of paths.

import assert from "node:assert/strict";
import { join, resolve } from "node:path";
import { test } from "node:test";
import { candidates, classify, fileFromArgv, pairSet, sameSet } from "../../src/main/pairing.ts";

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
  assert.deepEqual(classify(p("m.mpco")), { kind: "results", stem: p("m") });
  assert.deepEqual(classify(p("M.Geometry.H5")), { kind: "geometry", stem: p("M") });
});

test("classify: a file the app does not open is refused by name", () => {
  assert.throws(() => classify(p("m.txt")), /opens <stem>\.h5.*got .*m\.txt/);
  assert.throws(() => classify(".h5"), /got \.h5/);
});

test("candidates: the four names of a stem", () => {
  assert.deepEqual(candidates(p("m")), [p("m.h5"), p("m.geometry.h5"), p("m.results.h5"), p("m.mpco")]);
});

test("opening the model opens its geometry and results siblings", () => {
  const set = pairSet(p("m.h5"), disk("m.h5", "m.geometry.h5", "m.results.h5", "other.geometry.h5"));
  assert.deepEqual(set, { model: p("m.h5"), geometry: p("m.geometry.h5"), results: p("m.results.h5") });
});

test("absent siblings are null; .mpco is the results when .results.h5 is absent", () => {
  assert.deepEqual(pairSet(p("m.h5"), disk("m.h5")), { model: p("m.h5"), geometry: null, results: null });
  assert.deepEqual(pairSet(p("m.h5"), disk("m.h5", "m.mpco")).results, p("m.mpco"));
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

test("an opened results file is the results, even when the other results name exists", () => {
  const set = pairSet(p("m.mpco"), disk("m.h5", "m.mpco", "m.results.h5"));
  assert.equal(set.results, p("m.mpco"));
  assert.equal(set.model, p("m.h5"));
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
