// VU9a (ADR 0132 D2-D5; #2803): the app's workspace reader agrees with Python.
//
// The oracle is Python itself: every file and golden under
// test/fixtures/vu9a/ was written and judged by apeGmsh.results.user
// (`write_workspace`, `WorkspaceFile.verdicts`; the generator is in that
// folder's README). The app reads the same bytes and must state the same
// verdict for every set against every model of h5-schema.md's worked case, the
// same refusal for every row that refuses alone, and the same `partial` for a
// real gmsh re-mesh. The one documented difference is D5's: the app also
// validates a view as an `apegmsh-view/1` shot, where Python checks only that
// the body is a JSON object.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { bodyProblem, ModelSide, readWorkspaceFile, strictJson, UNREADABLE, verdictsOf, versionProblem, WorkspaceRefusal, type WorkspaceH5 } from "../../src/reader/workspace.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as WorkspaceH5;
const FIX = (p: string) => fileURLToPath(new URL(`../fixtures/vu9a/${p}`, import.meta.url));
const golden = (p: string) => JSON.parse(readFileSync(FIX(p), "utf8")) as Record<string, unknown>;

type Golden = Record<string, { verdict?: string; matched?: number; total?: number; members?: boolean[]; checked?: boolean; max_distance?: number | null; refused?: string }>;

/** The app's verdicts of `workspace` against `model`, in the golden's shape. */
function appVerdicts(workspace: string, model: string): Golden {
  const data = readWorkspaceFile(H5, FIX(workspace));
  const { verdicts, refused } = verdictsOf(data.sets, new ModelSide(H5, FIX(model)));
  const out: Golden = {};
  for (const [name, v] of verdicts) out[name] = { verdict: v.verdict, matched: v.matched, total: v.total, members: [...v.members], checked: v.checked, max_distance: v.maxDistance };
  for (const r of refused) out[r.name] = { refused: r.reason };
  return out;
}

const cases = golden("verdicts.golden.json") as Record<string, Golden>;
for (const [name, want] of Object.entries(cases)) {
  test(`the app's verdicts agree with Python's on the worked case: ${name}`, () => {
    assert.deepEqual(appVerdicts("frame.workspace.h5", `cases/${name}.h5`), want);
  });
}

test("the worked case states what h5-schema.md says (the golden is not vacuous)", () => {
  const v = (c: string, s: string) => cases[c]![s]!.verdict ?? cases[c]![s]!.refused;
  assert.equal(v("current", "springs"), "current");
  assert.equal(v("relabel", "roof"), "moved");
  assert.equal(v("base3_moved", "base"), "partial");
  assert.equal(v("column_up", "column line"), "lost");
  assert.equal(v("springs_swap", "springs"), "lost", "two zero-length springs on one pair are ambiguous, never moved (INV-3)");
  assert.equal(v("no_opensees", "springs"), "no-tag-map");
  assert.deepEqual(cases["tolerance"]!["roof"]!.members, [true, false, true]);
});

test("a re-meshed set reads partial: a real gmsh re-mesh, the app and Python agree", () => {
  const got = appVerdicts("plate/plate.workspace.h5", "plate/remeshed.h5");
  assert.deepEqual(got, golden("remesh.golden.json"));
  assert.equal(got["corners and inside"]!.verdict, "partial");
  assert.equal(got["corners and inside"]!.matched, 4);
});

test("rows that refuse alone: the same reason for every row as Python", () => {
  const data = readWorkspaceFile(H5, FIX("rows.workspace.h5"));
  const want = golden("rows.golden.json") as { revision: number; sets: string[]; plots: string[]; views: string[]; refused: [string, number, string, string][] };
  assert.equal(data.revision, want.revision);
  assert.deepEqual(data.sets.map((s) => s.name), want.sets);
  assert.deepEqual(data.plots.map((r) => r.name), want.plots);
  assert.deepEqual(data.views.map((r) => r.name), want.views);
  assert.deepEqual(data.refused.map((r) => [r.table, r.row, r.name, r.reason]), want.refused);
});

test("the saved file reads back as Python wrote it: members, positions, plot and view text", () => {
  const data = readWorkspaceFile(H5, FIX("frame.workspace.h5"));
  assert.equal(data.version, "1.0.0");
  assert.equal(data.writer, "make_vu9a");
  assert.deepEqual(data.sets.map((s) => s.name), ["roof", "base", "column line", "springs", "columns"]);
  const roof = data.sets[0]!;
  assert.deepEqual([...roof.ids], [31, 32, 33]);
  assert.deepEqual([...roof.xyz], [0, 0, 9, 6, 0, 9, 12, 0, 9]);
  assert.equal(roof.model_hash, "", "a node set stores no model_hash");
  assert.deepEqual([...data.sets[3]!.xyz], [12, 0, 0, 12, 0, 0]);
  assert.deepEqual(data.plots.map((r) => r.name), ["roof drift"]);
  assert.deepEqual(data.views.map((r) => r.name), ["iso"]);
  assert.deepEqual(data.refused, []);
});

test("file-level refusals: not HDF5, the version floor, a newer minor opens read-only", () => {
  assert.equal(versionProblem("1.0.0"), null);
  assert.equal(versionProblem("1.1.0"), "newer");
  assert.equal(versionProblem("2.0.0"), "major");
  assert.equal(versionProblem("1.0"), "bad-version");
  assert.equal(versionProblem("01.0.0"), "bad-version");
  assert.throws(() => readWorkspaceFile(H5, FIX("verdicts.golden.json")), (e: unknown) => e instanceof WorkspaceRefusal && e.reason === "not-hdf5");
  assert.throws(() => readWorkspaceFile(H5, FIX("frame.h5")), (e: unknown) => e instanceof WorkspaceRefusal && e.reason === "no-version");
});

test("a truncated workspace (the first half of a real one) reads as Python reads it: not-hdf5, which D6 renames aside", () => {
  const want = golden("truncated.golden.json") as { read: string; write_renamed_aside: boolean; bytes: number };
  assert.equal(readFileSync(FIX("truncated.workspace.h5")).length, want.bytes);
  assert.equal(want.write_renamed_aside, true, "Python's writer renamed it aside");
  assert.throws(() => readWorkspaceFile(H5, FIX("truncated.workspace.h5")), (e: unknown) => e instanceof WorkspaceRefusal && e.reason === want.read);
  assert.ok(UNREADABLE.has(want.read), "the app's writer treats that reason as unreadable, as Python's does");
});

test("strict JSON: Python's json with no NaN, no key twice, no overflowing float, and int tokens kept apart from float tokens", () => {
  for (const bad of ["{a: 1}", '{"a": NaN}', '{"a": Infinity}', '{"a": 1, "a": 2}', "[1,]", '{"a": 1e400}', "ï»¿{}", '{"a": "\u0001"}', "{} x"]) {
    assert.throws(() => strictJson(bad), Error, bad);
  }
  const r = strictJson('{"steps": [0, 1.0, 2e0, -1], "__proto__": {"x": 1}}');
  const v = r.value as { steps: number[] } & Record<string, unknown>;
  assert.deepEqual(v.steps.map((_, i) => r.floatAt(v.steps, i)), [false, true, true, false]);
  assert.ok(Object.hasOwn(v, "__proto__"), "__proto__ is an own key, never the prototype");
  assert.equal(Object.getPrototypeOf(v), Object.prototype);
  // An integer token past the largest double parses (Python's int); the plot grammar refuses it as not finite.
  assert.equal(strictJson("1" + "0".repeat(400)).value, Infinity);
});

test("D5: the app validates a view as an apegmsh-view/1 shot; a body that is no shot refuses bad-shot", () => {
  assert.deepEqual(bodyProblem("views", JSON.stringify({ name: "roof_view", camera: "iso" })).slice(1), ["", ""]);
  const [data, reason] = bodyProblem("views", JSON.stringify({ name: "roof_view", camera: "sideways" }));
  assert.equal(data, null);
  assert.equal(reason, "bad-shot");
});

test("a file the app wrote, read back by Python: Python's read of it (app_written.golden.json) is the app's", () => {
  // app_written/frame.workspace.h5 was written by main/workspace.ts `saveWorkspace` (four creates and a rename),
  // then read by apeGmsh.results.user.WorkspaceFile; the golden is what Python read and judged.
  const want = golden("app_written.golden.json") as { revision: number; writer: string; version: string; sets: Record<string, { kind: string; ids: number[]; xyz: number[][]; fem_hash: string; model_hash: string }>; refused: unknown[]; verdicts: Golden };
  const data = readWorkspaceFile(H5, FIX("app_written/frame.workspace.h5"));
  assert.equal(data.revision, want.revision);
  assert.equal(data.writer, want.writer);
  assert.equal(data.version, want.version);
  assert.deepEqual(want.refused, [], "Python refused no row the app wrote");
  const got = Object.fromEntries(data.sets.map((s) => [s.name, { kind: s.kind, ids: [...s.ids], xyz: Array.from({ length: s.ids.length }, (_, i) => [...s.xyz.slice(3 * i, 3 * i + 3)]), fem_hash: s.fem_hash, model_hash: s.model_hash }]));
  assert.deepEqual(got, want.sets);
  assert.deepEqual(Object.keys(got), ["roof line", "springs", "columns", "Ünïcode base"]);
  assert.deepEqual(appVerdicts("app_written/frame.workspace.h5", "frame.h5"), want.verdicts);
});
