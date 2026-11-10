// Stage activation (V4b, #1728): the app's activation per stage of the staged
// fixture equals Python's `build_from_model` on the same file (the oracle
// fixtures/staged_activation.oracle.json, written outside the app by
// scripts/make_viewer_results_fixtures.py), and each of the four states
// (filtered, unstaged, ambiguous, absent) has its case. The hide itself goes
// through renderer/visible.ts: the drawn primitives of an inactive element
// are not drawn.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { compileShot } from "../../src/spec/compile.ts";
import { describeSet } from "../../src/spec/describe.ts";
import { viewShotOf } from "../../src/spec/snapshot.ts";
import { validateSpec } from "../../src/spec/spec.ts";
import { activationOf } from "../../src/state/activation.ts";
import { activationNow, inactiveElements, noticesOf, resultsPairing } from "../../src/state/selectors.ts";
import { reduce, type State } from "../../src/state/store.ts";
import type { Definitions, SequenceZone, StageElements } from "../../src/state/types.ts";
import { BlobStore } from "../../src/state/blobs.ts";
import { elementMaskOf, visibleMaps } from "../../src/renderer/visible.ts";
import { FEM_ONLY, femIdsOf, ORACLE, stagedModelLoad, stagedResults, stagedState } from "./staged.ts";

const oracle = JSON.parse(readFileSync(ORACLE, "utf8")) as { elements: number; hidden_fem_ids: Record<string, number[]> };
const base = await stagedState();

const atStage = (s: State, stage: number): State => reduce(s, { type: "setPhase", at: { kind: "results", stage, step: 0 } });

test("oracle: the inactive elements of every program stage equal Python's build_from_model on the same file", () => {
  const names = base.results!.stages.map((st) => st.name);
  assert.deepEqual(names, ["base", "erect", "unprop", "mode_1"]);
  const compared: string[] = [];
  for (const [i, name] of names.entries()) {
    const a = activationNow(atStage(base, i))!.activation;
    if (!(name in oracle.hidden_fem_ids)) {
      assert.equal(a.state, "absent", `${name} names no program stage`);
      continue;
    }
    assert.equal(a.state, "filtered", name);
    if (a.state !== "filtered") continue;
    const app = femIdsOf(base, a.inactiveTags);
    assert.deepEqual(app, oracle.hidden_fem_ids[name], name);
    compared.push(`${name}: ${app.length}`);
  }
  // The pair differs between stages, so the comparison is not vacuous.
  assert.deepEqual(compared, ["base: 175", "erect: 0", "unprop: 220"]);
  assert.equal(Object.values(base.decls).filter((d) => d.kind === "element").length, oracle.elements);
});

test("filtered: the drawn primitives of an inactive element are not drawn; the next stage draws them again", async () => {
  const blobs = new BlobStore();
  // The state's blobs live in the loader's BlobStore; re-read the fixture into this one.
  const load = await stagedModelLoad(undefined, blobs);
  const r = await stagedResults();
  let s = reduce(base, { type: "fileLoaded", artifact: "model", load });
  s = reduce(s, { type: "resultsLoaded", ...r });
  const mesh = s.mesh!;
  const lineGroup = blobs.i32(mesh.lineGroup), triGroup = blobs.i32(mesh.triGroup), triElement = blobs.i32(mesh.triElement);
  const draw = (t: State) => visibleMaps(mesh.legend, t.visibility.hidden, lineGroup, triGroup, elementMaskOf(t, mesh, blobs));
  const all = draw(s);
  assert.equal(all.triMap.length, triGroup.length, "on the mesh phase every triangle is drawn");
  const topRow = mesh.legend.findIndex((e) => e.name === "Top");
  const propRow = mesh.legend.findIndex((e) => e.name === "Prop");
  const rowOf = (t: number) => triGroup[t]!;
  const atBase = draw(atStage(s, 0));
  const atErect = draw(atStage(s, 1));
  const atUnprop = draw(atStage(s, 2));
  const has = (m: Int32Array, row: number) => Array.from(m).some((t) => rowOf(t) === row);
  assert.equal(has(atBase.triMap, topRow), false, "base: Top (activated by erect) is not drawn");
  assert.equal(has(atBase.triMap, propRow), true);
  assert.equal(atErect.triMap.length, triGroup.length, "erect: everything is drawn");
  assert.equal(has(atUnprop.triMap, propRow), false, "unprop: Prop (removed) is not drawn");
  assert.equal(has(atUnprop.triMap, topRow), true);
  // Every hidden triangle belongs to an element the mask names.
  const mask = inactiveElements(atStage(s, 0))!.mask;
  const kept = new Set(atBase.triMap);
  for (let t = 0; t < triGroup.length; t++) assert.equal(kept.has(t), mask[triElement[t]!] === 0);
  assert.equal(inactiveElements(atStage(s, 0))!.inactive, 175);
});

// -- the states, on synthetic /sequence tables ------------------------------------

const STAGE_COLS = ["order", "name", "source"];
const CHANGE_COLS = ["stage", "op", "what", "name", "count", "source"];
const OWNED = "/opensees/stages/stage_001/owned_element_ids";
const REMOVED = "/opensees/stages/stage_002/remove_element";

function seqOf(names: string[], changes: unknown[][], staged = 1): SequenceZone {
  return {
    status: "ready",
    version: "1.0.0",
    attrs: { staged },
    stages: { path: "/sequence/stages", columns: STAGE_COLS, rows: names.map((n, i) => [i, n, `/opensees/stages/stage_${String(i).padStart(3, "0")}`]) },
    changes: { path: "/sequence/changes", columns: CHANGE_COLS, rows: changes },
  };
}
const CHANGES = [
  [1, "activate", "element", "", 2, OWNED],
  [1, "activate", "group", "Top", 1, "/opensees/stages/stage_001/activated_pgs"],
  [2, "remove", "element", "", 1, REMOVED],
];
const SETS: StageElements = { status: "ready", sets: { [OWNED]: [10, 11], [REMOVED]: [3] } };
const DEFS = (n: number) => ({ stages: Array.from({ length: n }) }) as unknown as Definitions;

test("filtered: active(K) = everything - not-yet-activated - removed so far", () => {
  const seq = seqOf(["a", "b", "c"], CHANGES);
  const tags = (name: string) => {
    const a = activationOf(seq, SETS, DEFS(3), name);
    assert.equal(a.state, "filtered", name);
    return a.state === "filtered" ? [...a.inactiveTags].sort((x, y) => x - y) : null;
  };
  assert.deepEqual(tags("a"), [10, 11]);
  assert.deepEqual(tags("b"), []);
  assert.deepEqual(tags("c"), [3]);
});

test("unstaged: @staged 0, or a model whose /opensees has no stages, draws everything without a banner", () => {
  assert.deepEqual(activationOf(seqOf([""], [], 0), { status: "none" }, DEFS(0), "history"), { state: "unstaged" });
  assert.deepEqual(activationOf({ status: "absent" }, { status: "none" }, DEFS(0), "history"), { state: "unstaged" });
});

test("ambiguous: two /sequence stages of that name, or a results stage with no name (never the first stage)", () => {
  const dup = activationOf(seqOf(["a", "a", "c"], CHANGES), SETS, DEFS(3), "a");
  assert.equal(dup.state, "ambiguous");
  assert.match((dup as { reason: string }).reason, /2 \/sequence stages are named "a"/);
  const empty = activationOf(seqOf(["", "b", "c"], CHANGES), SETS, DEFS(3), "");
  assert.equal(empty.state, "ambiguous", "an empty name never matches the unnamed first stage");
  assert.match((empty as { reason: string }).reason, /no name/);
});

test("absent: no /sequence, or no stage of that name; refused: element rows that cannot be read without guessing", () => {
  const stated = (want: string) => (seq: SequenceZone, sets: StageElements, name = "a") => {
    const a = activationOf(seq, sets, DEFS(3), name);
    assert.equal(a.state, want, JSON.stringify(a));
    return (a as { reason: string }).reason;
  };
  const absent = stated("absent");
  const reason = stated("refused");
  assert.match(absent({ status: "absent" }, { status: "none" }), /no \/sequence zone/);
  assert.match(absent(seqOf(["a", "b", "c"], CHANGES), SETS, "mode_1"), /no \/sequence stage is named "mode_1"/);
  assert.match(reason(seqOf(["a", "b", "c"], [...CHANGES, [2, "detach", "element", "", 1, REMOVED]]), SETS), /outside the \/sequence 1\.0\.0 vocabulary/);
  assert.match(reason(seqOf(["a", "b", "c"], CHANGES), { status: "ready", sets: { [OWNED]: [10], [REMOVED]: [3] } }), /holds 1 tags, and its \/sequence rows count 2/);
  assert.match(reason(seqOf(["a", "b", "c"], CHANGES), { status: "error", reason: "boom" }), /boom/);
  assert.match(reason(seqOf(["a", "b", "c"], [[1, "activate", "element", "", 2, "/opensees/stages/stage_002/owned_element_ids"]]), SETS), /names .*stage_002.*not .*stage_001\/owned_element_ids/);
});

test("refused: an element op this app does not apply ((add|set), element) refuses with the reason, never skipped", () => {
  for (const op of ["add", "set"]) {
    const seq = seqOf(["a", "b", "c"], [...CHANGES, [2, op, "element", "", 1, REMOVED]]);
    const a = activationOf(seq, SETS, DEFS(3), "c");
    assert.equal(a.state, "refused", op);
    assert.match((a as { reason: string }).reason, new RegExp(`row 3 is \\(${op}, element\\), which this app does not know how to apply`));
  }
  // The other read failures are refusals too, not "absent".
  assert.equal(activationOf(seqOf(["a", "b", "c"], CHANGES), { status: "error", reason: "boom" }, DEFS(3), "a").state, "refused");
});

test("unbound: results paired by fem_hash only are never joined to /sequence names (the review's reproducer)", async () => {
  // A fem-only NativeWriter file (no model_h5_src) of the same mesh, on a model whose /sequence repeats a name.
  const fem = await stagedResults(FEM_ONLY);
  assert.equal(fem.info.lineage!.modelHash, null);
  const dupLoad = await stagedModelLoad();
  const seq = dupLoad.sequence;
  assert.equal(seq.status, "ready");
  if (seq.status !== "ready") return;
  const nc = seq.stages.columns.indexOf("name");
  const dup = { ...seq, stages: { ...seq.stages, rows: seq.stages.rows.map((r) => r.map((c, i) => (i === nc ? "base" : c))) } };
  let s = reduce(base, { type: "fileLoaded", artifact: "model", load: { ...dupLoad, sequence: dup } });
  s = reduce(s, { type: "fileOpened", artifact: "results", path: fem.info.path });
  s = reduce(s, { type: "resultsLoaded", ...fem });
  assert.deepEqual(resultsPairing(s), { state: "paired", by: "fem_hash" });
  for (const i of [0, 1, 2]) {
    const t = atStage(s, i);
    assert.equal(activationNow(t)!.activation.state, "unbound", `stage ${i}`);
    assert.equal(inactiveElements(t), null, "unbound hides nothing");
    assert.match(noticesOf(t).join("\n"), /Stage activation unbound for results stage .*fem_hash only/);
  }
  // The undoubled model with the same fem-only file: still unbound, never filtered.
  let u = reduce(base, { type: "fileOpened", artifact: "results", path: fem.info.path });
  u = reduce(u, { type: "resultsLoaded", ...fem });
  assert.equal(activationNow(atStage(u, 0))!.activation.state, "unbound");
  // And the model_hash results on the doubled model: ambiguous, as before.
  const r = await stagedResults();
  let m = reduce(s, { type: "fileOpened", artifact: "results", path: r.info.path });
  m = reduce(m, { type: "resultsLoaded", ...r });
  assert.equal(activationNow(atStage(m, 0))!.activation.state, "ambiguous");
});

test("the ambiguous and absent states are a visible banner; filtered and unstaged are not", () => {
  const notices = (s: State) => noticesOf(s).filter((n) => /Stage activation/.test(n));
  assert.deepEqual(notices(atStage(base, 0)), []);
  assert.deepEqual(notices(atStage(base, 3)), [
    'Stage activation absent for results stage "mode_1": no /sequence stage is named "mode_1", so the results stage pairs with none. Every element is drawn.',
  ]);
  assert.equal(inactiveElements(atStage(base, 3)), null, "absent hides nothing");
  assert.deepEqual(notices(base), [], "off the results phases there is no activation notice");
});

// W4d #2843 (the W4c delta review's reproducer): a stage shown by a display name
// ("Base load", a .ladruno `stage_display_names` rename) still matches /sequence by
// its program name. Matched on `name`, the stage reads `absent` ("no /sequence
// stage is named \"Base load\"") and every element is drawn.
const renamed = (s: State, k: number, name: string, programName = s.results!.stages[k]!.programName): State => ({
  ...s,
  results: { ...s.results!, stages: s.results!.stages.map((st, i) => (i === k ? { ...st, name, programName } : st)) },
});

test('W4d: a stage renamed for display ("Base load") is still filtered by its program name, and labelled by its display name', () => {
  assert.equal(base.results!.stages[0]!.programName, "base", "a native file's program name is its stage name");
  const t = atStage(renamed(base, 0, "Base load"), 0);
  const now = activationNow(t)!;
  assert.equal(now.activation.state, "filtered", JSON.stringify(now.activation));
  if (now.activation.state !== "filtered") return;
  assert.deepEqual(femIdsOf(base, now.activation.inactiveTags), oracle.hidden_fem_ids["base"], "the same elements as the un-renamed stage");
  assert.equal(now.stageName, "Base load", "the banner and labels name the stage by its display name");
  assert.ok(inactiveElements(t), "the inactive elements are hidden");
  // The un-renamed stage again: the memo keys on both names, so nothing stale is served.
  assert.equal(activationNow(atStage(base, 0))!.stageName, "base");
});

test("W4d: a display name never takes part in matching, even when it is another stage's program name", () => {
  // Shown as "erect", but program stage "nope": absent by the program name, never erect's activation.
  const t = atStage(renamed(base, 0, "erect", "nope"), 0);
  const a = activationNow(t)!.activation;
  assert.equal(a.state, "absent");
  assert.match((a as { reason: string }).reason, /no \/sequence stage is named "nope"/);
  assert.equal(inactiveElements(t), null);
});

// W4d #2843, the spec side (Amendment 1): a spec names a stage by either of its names,
// so one written against the program name keeps resolving after a display rename;
// the two names hitting different stages refuses. A snapshot writes the program name.
const specShot = (stage: string | number) => {
  const v = validateSpec({ spec: "apegmsh-view/1", files: { model: "m.h5" }, shots: [{ name: "s", stage }] });
  if (!v.ok) throw new Error(`${v.key}: ${v.reason}`);
  return v.spec.shots[0]!;
};
const resolved = (s: State, stage: string) => {
  const c = compileShot(specShot(stage), describeSet(s, []));
  return c.ok ? c.plan.stage!.index : c.reason;
};

test("W4d: a spec resolves a renamed stage by its program name or its display name; the two names on different stages refuse", () => {
  const d = describeSet(renamed(base, 0, "Base load"), []);
  assert.deepEqual(
    d.stages.map((st) => [st.name, st.programName]),
    [["Base load", "base"], ["erect", "erect"], ["unprop", "unprop"], ["mode_1", "mode_1"]],
  );
  assert.equal(resolved(renamed(base, 0, "Base load"), "base"), 0, "the program name still resolves");
  assert.equal(resolved(renamed(base, 0, "Base load"), "Base load"), 0, "and so does the display name");
  assert.equal(resolved(base, "base"), 0);
  // Stage 0 shown as "erect" while stage 1's program name is "erect": refused, never a guess.
  assert.match(String(resolved(renamed(base, 0, "erect"), "erect")), /"erect" is the program name of stage 1 and the shown name of stage 0; name the stage by its index/);
  assert.match(String(resolved(renamed(base, 0, "Base load"), "gravity")), /no stage named "gravity"; their stages are "Base load" \(program stage "base"\), "erect"/);
});

test("W4d: a snapshot names a renamed stage by its program name, or by its index when that name is ambiguous", () => {
  const pose = { eye: [1, 1, 1] as [number, number, number], target: [0, 0, 0] as [number, number, number], up: [0, 0, 1] as [number, number, number] };
  assert.equal(viewShotOf(atStage(renamed(base, 0, "Base load"), 0), pose).shot.stage, "base");
  assert.equal(viewShotOf(atStage(base, 1), pose).shot.stage, "erect");
  // Stage 1 shown as "base": "base" would hit two stages, so the snapshot names stage 0 by its index.
  assert.equal(viewShotOf(atStage(renamed(base, 1, "base"), 0), pose).shot.stage, 0);
});
