// ADR 0137 D10, D11 (H1): the tool host's scheduling and the model window's side of the contract.
//
// - `ToolScheduler`: `update` runs at most once per frame, with the newest context, and never while
//   the tool is hidden; a tool shown again draws what arrived meanwhile.
// - `toolContextOf`: the context is small and plain (a file name, the step with `playing`, capped
//   targets, the theme), and `checkToolEvent` lets through only a step the state holds.

import assert from "node:assert/strict";
import { test } from "node:test";
import { TARGETS_SHOWN, type ToolContext } from "../../src/tools/contract.ts";
import { ToolScheduler } from "../../src/tools/host.ts";
import { TOOLS } from "../../src/tools/registry.ts";
import { checkToolEvent, toolContextOf } from "../../src/state/tools.ts";
import { initialState, reduce } from "../../src/state/store.ts";
import type { State } from "../../src/state/types.ts";

function clock() {
  const queued: (() => void)[] = [];
  return { request: (fn: () => void) => void queued.push(fn), frame: () => queued.splice(0).forEach((f) => f()), queued };
}
const ctx = (n: number): ToolContext => ({ model: `m${n}.h5`, step: null, targets: { elements: 0, nodes: 0, decls: [], nodeTags: [] }, params: {}, theme: "dark" });

test("many contexts between two frames make one update, with the newest", () => {
  const seen: string[] = [];
  const c = clock();
  const s = new ToolScheduler({ update: (x) => void seen.push(x.model!) }, c, () => true);
  for (let i = 1; i <= 5; i++) s.push(ctx(i));
  assert.equal(c.queued.length, 1, "one frame asked for");
  c.frame();
  assert.deepEqual(seen, ["m5.h5"]);
  c.frame();
  assert.deepEqual(seen, ["m5.h5"], "no new context, no update");
  s.push(ctx(6));
  c.frame();
  assert.deepEqual(seen, ["m5.h5", "m6.h5"]);
});

test("a hidden tool is not updated; shown again, it draws the newest context once", () => {
  const seen: string[] = [];
  const c = clock();
  let visible = false;
  const s = new ToolScheduler({ update: (x) => void seen.push(x.model!) }, c, () => visible);
  s.push(ctx(1));
  s.push(ctx(2));
  assert.equal(c.queued.length, 0, "hidden: no frame asked for");
  visible = true;
  s.shown();
  c.frame();
  assert.deepEqual(seen, ["m2.h5"]);
  // Hidden between the request and the frame: nothing is drawn, and the context waits.
  s.push(ctx(3));
  visible = false;
  c.frame();
  assert.deepEqual(seen, ["m2.h5"]);
  visible = true;
  s.shown();
  c.frame();
  assert.deepEqual(seen, ["m2.h5", "m3.h5"]);
});

test("every registered tool declares its cost, and its id is its key", () => {
  for (const [id, def] of Object.entries(TOOLS)) {
    assert.equal(def.id, id);
    assert.ok(def.cost === "light" || def.cost === "heavy");
    assert.ok(def.title.length > 0);
  }
  assert.equal(TOOLS["step"]?.cost, "light");
});

/** A state on results stage 0 of a 4-step stage, at `step`. */
function onResults(step: number): State {
  const s = reduce(initialState, { type: "fileOpened", artifact: "model", path: "C:\\work\\bridge.h5" });
  return {
    ...s,
    results: { stages: [{ id: "stage_0", name: "push", time: [0, 1, 2, 3], nodal: ["displacement_x", "displacement_y"] }] } as unknown as State["results"],
    phase: { axis: [{ kind: "results", stage: 0, step: 0 }] as unknown as State["phase"]["axis"], at: { kind: "results", stage: 0, step } },
  };
}

test("the context names the file, not its path, and carries the step with playback", () => {
  const s = { ...onResults(2), playing: true };
  const c = toolContextOf(s);
  assert.equal(c.model, "bridge.h5");
  assert.deepEqual(c.step, { stage: "stage_0", stageName: "push", step: 2, steps: 4, playing: true, components: ["displacement_x", "displacement_y"] });
  assert.equal(c.theme, "dark");
  assert.equal(toolContextOf(reduce(initialState, { type: "fileOpened", artifact: "model", path: "/m.h5" })).step, null, "no results stage, no step");
});

test("the context carries at most TARGETS_SHOWN targets of each kind, and the counts", () => {
  const base = onResults(0);
  const decls = Array.from({ length: 500 }, (_, i) => `element/${i}`);
  const nodes = Array.from({ length: 300 }, (_, i) => i + 1);
  const s = { ...base, selection: { ...base.selection, decls, nodes } } as State;
  const c = toolContextOf(s);
  assert.equal(c.targets.elements, 500);
  assert.equal(c.targets.nodes, 300);
  assert.equal(c.targets.decls.length, TARGETS_SHOWN);
  assert.equal(c.targets.nodeTags.length, TARGETS_SHOWN);
  assert.ok(JSON.stringify(c).length < 4096, "a context stays small");
});

test("a tool's event is let through only when the state allows it now", () => {
  const s = onResults(1);
  assert.deepEqual(checkToolEvent(s, { type: "setResultStep", step: 3 }), { type: "setResultStep", step: 3 });
  assert.equal(checkToolEvent(s, { type: "setResultStep", step: 4 }), null, "past the stage's last step");
  assert.equal(checkToolEvent(s, { type: "setResultStep", step: -1 }), null);
  assert.equal(checkToolEvent(s, { type: "setResultStep", step: 1.5 }), null);
  assert.equal(checkToolEvent(s, { type: "setResultStep", step: "2" }), null);
  assert.equal(checkToolEvent(s, { type: "fileOpened", artifact: "model", path: "/elsewhere.h5" }), null, "a tool may not open files");
  assert.equal(checkToolEvent(s, null), null);
  const noResults = reduce(initialState, { type: "fileOpened", artifact: "model", path: "/m.h5" });
  assert.equal(checkToolEvent(noResults, { type: "setResultStep", step: 0 }), null, "no results stage shown");
});
