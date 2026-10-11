// V3c1 (#1642): the stage diagram, read from the root /sequence zone (V3a).
//
// Oracles: fixtures/definitions.h5 (make_zone_fixtures.py `definitions`: two
// stages, gravity (static, 10 steps, pattern `gravity`, fix `supports`,
// recorder `crown`) and push (transient, 5 steps, dt 0.01, set_time 0,
// pattern `push`, one remove_sp); global fix `supports` on every bcs/fix row,
// pattern #1 and recorder #1). The fixture has no /sequence (V3c0 owns its
// script), so this test copies it and writes the /sequence zone V3a's
// deriver (src/apeGmsh/opensees/_internal/sequence_h5.py) gives that
// archive, by the rules of h5-schema.md `/sequence`: one stage row per
// stage_NNN; global rows with stage 0 and a source outside /opensees/stages;
// a pattern by its group path; `<store>[i]` for a recorder; one row per
// declaration of a grouped store. Every name, count and key asserted below is
// a literal of that script or of these rows, never a value read back from the
// app. The view runs on a minimal DOM: it proves what the diagram shows, and
// that a click dispatches `requestSource` through `act` and nothing else.

import assert from "node:assert/strict";
import { copyFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { WINDOWS } from "../../src/panels/windows/registry.ts";
import { NOT_RECORDED, stagesOf, stepsText, view } from "../../src/panels/windows/stages.ts";
import { openModel } from "../../src/reader/node.ts";
import { BlobStore } from "../../src/state/blobs.ts";
import { loadModel } from "../../src/state/load.ts";
import { initialState, reduce } from "../../src/state/reduce.ts";
import type { Event, State } from "../../src/state/types.ts";

// ---- a minimal DOM: elements with attributes, children, text and click listeners ----

class FakeText {
  textContent: string;
  constructor(text: string) {
    this.textContent = text;
  }
}
class FakeEl {
  children: (FakeEl | FakeText)[] = [];
  attrs: Record<string, string> = {};
  dataset: Record<string, string> = {};
  className = "";
  title = "";
  tag: string;
  private listeners: ((e: { stopPropagation(): void }) => void)[] = [];
  constructor(tag: string) {
    this.tag = tag;
  }
  append(...xs: (FakeEl | FakeText | string)[]): void {
    for (const x of xs) this.children.push(typeof x === "string" ? new FakeText(x) : x);
  }
  get textContent(): string {
    return this.children.map((c) => c.textContent).join("");
  }
  set textContent(v: string) {
    this.children = [new FakeText(v)];
  }
  setAttribute(k: string, v: string): void {
    this.attrs[k] = v;
  }
  addEventListener(type: string, fn: (e: { stopPropagation(): void }) => void): void {
    assert.equal(type, "click", "the window listens for clicks only");
    this.listeners.push(fn);
  }
  get clickable(): boolean {
    return this.listeners.length > 0;
  }
  click(): void {
    for (const f of this.listeners) f({ stopPropagation() {} });
  }
  all(pred: (e: FakeEl) => boolean): FakeEl[] {
    const out: FakeEl[] = pred(this) ? [this] : [];
    for (const c of this.children) if (c instanceof FakeEl) out.push(...c.all(pred));
    return out;
  }
}
(globalThis as { document?: unknown }).document = { createElement: (tag: string) => new FakeEl(tag) };

// ---- the fixtures ----

await h5wasm.ready;
const fixtures = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "fixtures");
const dir = mkdtempSync(join(tmpdir(), "agv-stages-"));
const loadPath = async (p: string): Promise<State> =>
  reduce(initialState, { type: "fileLoaded", artifact: "model", load: loadModel(await openModel(p), new BlobStore()) });

type Col = string[] | Int32Array | Float64Array;
/* eslint-disable @typescript-eslint/no-explicit-any */
function writeSequence(path: string, staged: number, stages: Record<string, Col>, changes: Record<string, Col>): void {
  const f = new h5wasm.File(path, "a");
  try {
    (f.get("meta") as any).create_attribute("sequence_schema_version", "1.0.0");
    const g = f.create_group("sequence") as any;
    g.create_attribute("derived_from", "/opensees");
    g.create_attribute("model_hash", "");
    g.create_attribute("staged", staged);
    for (const [name, cols] of [["stages", stages], ["changes", changes]] as const) {
      const t = g.create_group(name);
      for (const [c, data] of Object.entries(cols)) t.create_dataset({ name: c, data });
    }
  } finally {
    f.close();
  }
}

const STAGES = {
  order: new Int32Array([0, 1]),
  name: ["gravity", "push"],
  analysis: ["Static", "Transient"],
  integrator: ["LoadControl", "Newmark"],
  algorithm: ["Newton", "Newton"],
  test: ["NormDispIncr", "NormDispIncr"],
  system: ["UmfPack", "UmfPack"],
  numberer: ["RCM", "RCM"],
  handler: ["Plain", "Plain"],
  integrator_args: ["[0.1]", "[0.5, 0.25]"],
  algorithm_args: ["[]", "[]"],
  test_args: ["[1e-06, 10.0, 0.0, 2.0]", "[1e-06, 10.0, 0.0, 2.0]"],
  system_args: ["[]", "[]"],
  handler_args: ["[]", "[]"],
  steps: new Int32Array([10, 5]),
  dt: new Float64Array([Number.NaN, 0.01]),
  set_time: new Float64Array([Number.NaN, 0]),
  domain_change: new Int32Array([1, 0]),
  source: ["/opensees/stages/stage_000", "/opensees/stages/stage_001"],
};
const S0 = "/opensees/stages/stage_000", S1 = "/opensees/stages/stage_001";
// [stage, op, what, name, count, source]: the global rows first (stage 0), then each stage's, in the deriver's order.
const CHANGE_ROWS: [number, string, string, string, number, string][] = [
  [0, "add", "fix", "supports", 2, "/opensees/bcs/fix"],
  [0, "add", "pattern", "", 1, "/opensees/patterns/Plain_1"],
  [0, "add", "recorder", "", 1, "/opensees/recorders[0]"],
  [0, "add", "fix", "supports", 1, `${S0}/bcs/fix`],
  [0, "add", "pattern", "gravity", 1, `${S0}/patterns/Plain_2`],
  [0, "add", "recorder", "crown", 1, `${S0}/recorders[0]`],
  [1, "add", "pattern", "push", 1, `${S1}/patterns/Plain_3`],
  [1, "remove", "fix", "", 1, `${S1}/remove_sp`],
];
const CHANGES = {
  stage: new Int32Array(CHANGE_ROWS.map((r) => r[0])),
  op: CHANGE_ROWS.map((r) => r[1]),
  what: CHANGE_ROWS.map((r) => r[2]),
  name: CHANGE_ROWS.map((r) => r[3]),
  count: new Int32Array(CHANGE_ROWS.map((r) => r[4])),
  source: CHANGE_ROWS.map((r) => r[5]),
};

const SEQ_PATH = join(dir, "definitions.seq.h5");
copyFileSync(join(fixtures, "definitions.h5"), SEQ_PATH);
writeSequence(SEQ_PATH, 1, STAGES, CHANGES);
const staged = await loadPath(SEQ_PATH);
const bare = await loadPath(join(fixtures, "definitions.h5"));

// An unstaged deck whose loop is user Tcl: one implicit stage, name "", steps -1, dt NaN (kr3pb's shape).
const DECK_PATH = join(dir, "deck.seq.h5");
copyFileSync(join(fixtures, "shoebuckle.h5"), DECK_PATH);
writeSequence(
  DECK_PATH,
  0,
  {
    order: new Int32Array([0]), name: [""], analysis: ["Static"], integrator: ["LoadControl"], algorithm: ["Newton"], test: ["NormDispIncr"],
    steps: new Int32Array([-1]), dt: new Float64Array([Number.NaN]), set_time: new Float64Array([Number.NaN]), domain_change: new Int32Array([0]),
    source: ["/opensees/analysis"],
  },
  { stage: new Int32Array([0]), op: ["add"], what: ["pattern"], name: [""], count: new Int32Array([1]), source: ["/opensees/patterns/Plain_1"] },
);
const deck = await loadPath(DECK_PATH);

const render = (s: State): { root: FakeEl; events: Event[] } => {
  const events: Event[] = [];
  const root = view(s, (e) => events.push(e)) as unknown as FakeEl;
  return { root, events };
};
const cardsOf = (root: FakeEl) => root.all((e) => e.className === "stage-card");

test("the registry's stages entry is this window's view", () => {
  assert.equal(WINDOWS.stages.view, view);
});

test("every stage from /sequence in order: name, analysis, integrator / algorithm / test, steps, dt", () => {
  const m = stagesOf(staged);
  assert.equal(m.note, null);
  assert.equal(m.header, "/sequence 1.0.0 · staged, 2 stages");
  assert.deepEqual(
    m.stages.map((s) => [s.order, s.title, s.analysis, s.chain, s.steps, s.dt, s.setTime, s.domainChange, s.h5]),
    [
      [0, "gravity", "Static", "LoadControl · Newton · NormDispIncr", "10 steps", "static (no dt)", null, true, S0],
      [1, "push", "Transient", "Newmark · Newton · NormDispIncr", "5 steps", "dt 0.01", "set_time 0", false, S1],
    ],
  );
});

test("each stage's changes by (op, what), names as handles, counts, and the #k key of an unnamed declaration", () => {
  const m = stagesOf(staged);
  const flat = (gs: typeof m.global) => gs.map((g) => [`${g.op} ${g.what}`, g.items.map((i) => [i.label, i.count, i.decl?.key ?? null])]);
  // Global rows carry stage 0 in /sequence, but are listed once, outside the timeline (their source is outside /opensees/stages).
  assert.deepEqual(flat(m.global), [
    ["add fix", [["supports", 2, "opensees/fix/supports"]]],
    ["add pattern", [["#1", 1, "opensees/pattern/#1"]]],
    ["add recorder", [["#1", 1, "opensees/recorder/#1"]]],
  ]);
  assert.deepEqual(flat(m.stages[0]!.changes), [
    ["add fix", [["supports", 1, "opensees/fix/supports"]]],
    ["add pattern", [["gravity", 1, "opensees/pattern/gravity"]]],
    ["add recorder", [["crown", 1, "opensees/recorder/crown"]]],
  ]);
  // remove_sp has no decls row in the fixture: no key is invented, and the item says "unnamed".
  assert.deepEqual(flat(m.stages[1]!.changes), [
    ["add pattern", [["push", 1, "opensees/pattern/push"]]],
    ["remove fix", [["unnamed", 1, null]]],
  ]);
});

test("a name that disagrees with the joined declaration drops the key rather than naming the wrong one", () => {
  const seq = staged.sequence;
  assert.equal(seq.status, "ready");
  if (seq.status !== "ready") return;
  const ni = seq.changes.columns.indexOf("name");
  const rows = seq.changes.rows.map((r, i) => (i === 5 ? r.map((v, j) => (j === ni ? "not-crown" : v)) : r));
  const s: State = { ...staged, sequence: { ...seq, changes: { ...seq.changes, rows } } };
  const rec = stagesOf(s).stages[0]!.changes.find((g) => g.what === "recorder")!.items[0]!;
  assert.deepEqual([rec.label, rec.decl], ["not-crown", null]);
});

test("an unstaged deck shows its one implicit stage; steps = -1 is said in words, never as a count", () => {
  const m = stagesOf(deck);
  assert.equal(m.header, "/sequence 1.0.0 · unstaged deck: one implicit stage, from /opensees/analysis");
  assert.equal(m.stages.length, 1);
  const [st] = m.stages;
  assert.equal(st!.title, "implicit stage");
  assert.equal(st!.steps, "no analyze_steps archived (steps = -1): the deck runs its own analysis loop, so no step count is known");
  assert.doesNotMatch(st!.steps, /^-1 steps/);
  assert.equal(st!.dt, "static (no dt)");
  // The deck's pattern is global: listed outside the timeline, and the stage records no change of its own.
  assert.deepEqual(m.global.map((g) => `${g.op} ${g.what}`), ["add pattern"]);
  assert.deepEqual(st!.changes, []);
  assert.equal(stepsText(1), "1 step");
});

test("a file without /sequence says it predates readable stages, and never rebuilds a sequence from /opensees", () => {
  const m = stagesOf(bare);
  assert.match(m.note!, /^This file predates readable stages: it has no \/sequence zone/);
  assert.deepEqual([m.header, m.stages, m.global], [null, [], []]);
  const { root } = render(bare);
  assert.equal(cardsOf(root).length, 0);
  assert.match(root.textContent, /predates readable stages/);
});

test("empty, refused and malformed states are explicit", () => {
  assert.equal(stagesOf(initialState).note, "no model is open");
  const refused: State = { ...staged, sequence: { status: "refused", version: "2.0.0", accepted: "1.0 and any later 1.x", newer: true, reason: "this app reads major 1 only" } };
  assert.equal(stagesOf(refused).note, "/sequence 2.0.0 was not read: this app reads major 1 only (this app reads 1.0 and any later 1.x)");
  const seq = staged.sequence;
  if (seq.status !== "ready") throw new Error("fixture");
  const noSteps: State = { ...staged, sequence: { ...seq, stages: { ...seq.stages, columns: seq.stages.columns.map((c) => (c === "steps" ? "stepz" : c)) } } };
  assert.equal(stagesOf(noSteps).note, "/sequence/stages has no 'steps' column: the diagram cannot be drawn from this file");
  const none: State = { ...staged, sequence: { ...seq, stages: { ...seq.stages, rows: [] }, changes: { ...seq.changes, rows: [] } } };
  assert.equal(stagesOf(none).note, "the model declares no analysis: /sequence has no stage");
});

test("the diagram: a horizontal timeline of stage cards, the global block, and the #1646 note on what /sequence does not record", () => {
  const { root, events } = render(staged);
  const timeline = root.all((e) => e.className === "stage-timeline");
  assert.equal(timeline.length, 1);
  assert.match(timeline[0]!.attrs["style"]!, /display: flex/);
  const cards = cardsOf(root);
  assert.deepEqual(cards.map((c) => c.dataset["h5"]), [S0, S1]);
  assert.equal(root.all((e) => e.className === "stage-arrow").length, 1);
  assert.match(cards[0]!.textContent, /gravity.*Static.*LoadControl · Newton · NormDispIncr.*10 steps.*static \(no dt\) · domain change/);
  assert.match(cards[1]!.textContent, /push.*Transient.*5 steps.*dt 0\.01 · set_time 0/);
  assert.match(cards[1]!.textContent, /remove fix unnamed/);
  const sections = root.all((e) => e.className === "window-section").map((e) => e.textContent);
  assert.deepEqual(sections, ["in force from the first stage (outside any stage)", "stages"]);
  // The four verbs /sequence does not project are named on the diagram, and it never says a stage lacks them.
  const gap = root.all((e) => e.dataset["gap"] === "1646");
  assert.equal(gap.length, 1);
  for (const v of NOT_RECORDED) assert.match(gap[0]!.textContent, new RegExp(v));
  assert.deepEqual([...NOT_RECORDED], ["update_parameter", "zero_velocities", "set_creep_on", "pre_analyze_reset"]);
  assert.match(gap[0]!.textContent, /Not recorded in \/sequence.*#1646.*A stage may still make these actions/);
  assert.match(render(deck).root.textContent, /Not recorded in \/sequence/);
  // No source is recorded for these declarations: nothing is clickable, and each says why.
  const refs = root.all((e) => e.className === "window-ref");
  assert.ok(refs.length >= 7);
  assert.ok(refs.every((r) => !r.clickable));
  assert.match(refs.find((r) => r.textContent === "gravity")!.title, /no \/provenance record for opensees\/pattern\/gravity/);
  assert.match(refs.find((r) => r.textContent === "unnamed")!.title, /remove_sp: no \/opensees\/decls row names it/);
  assert.equal(cards.some((c) => c.clickable), false, "the store has no stage selection: a stage card dispatches nothing");
  assert.deepEqual(events, []);
});

test("a click on a change with a recorded source dispatches requestSource, which the store accepts", () => {
  const key = "opensees/pattern/gravity";
  const s: State = {
    ...staged,
    provenance: [{ key, origin: "user", seq: 0, source: { file: "/work/model.py", line: 7, function: "<module>", sha256: "ab", recorded: true, script: null } }],
  };
  const { root, events } = render(s);
  const refs = root.all((e) => e.className === "window-ref" && e.clickable);
  assert.deepEqual(refs.map((r) => r.textContent), ["gravity"]);
  refs[0]!.click();
  assert.deepEqual(events, [{ type: "requestSource", decl: key }]);
  assert.deepEqual(reduce(s, events[0]!).source.request, { decl: key, seq: 1 });
});

test("an op or what outside the /sequence 1.0.0 vocabulary is flagged as unknown, never drawn like a known change", () => {
  const seq = staged.sequence;
  if (seq.status !== "ready") throw new Error("fixture");
  const oi = seq.changes.columns.indexOf("op"), wi = seq.changes.columns.indexOf("what");
  const rows = seq.changes.rows.map((r, i) => (i === 6 ? r.map((v, j) => (j === oi ? "toggle" : j === wi ? "widget" : v)) : r));
  const s: State = { ...staged, sequence: { ...seq, changes: { ...seq.changes, rows } } };
  const groups = stagesOf(s).stages[1]!.changes;
  assert.deepEqual(groups.map((g) => [g.op, g.what, g.unknown]), [
    ["toggle", "widget", "unknown op 'toggle' and what 'widget': not in the /sequence 1.0.0 vocabulary"],
    ["remove", "fix", null],
  ]);
  assert.ok(stagesOf(staged).stages.every((st) => st.changes.every((g) => g.unknown === null)));
  const flagged = render(s).root.all((e) => e.dataset["unknown"] === "1");
  assert.equal(flagged.length, 1);
  assert.match(flagged[0]!.textContent, /toggle widget \[unknown op 'toggle' and what 'widget'/);
  assert.match(flagged[0]!.all((e) => e.className === "stage-unknown")[0]!.attrs["style"]!, /var\(--error\)/);
});
