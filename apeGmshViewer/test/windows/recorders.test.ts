// V3c3 (#1644): the recorders window. `recordersOf` is the window's content
// from the store alone; the view renders it (its DOM is proved by the
// screenshots in the PR, as the app tests run without a DOM).
//
// Oracles: fixtures/definitions.h5, written by hand by make_zone_fixtures.py
// (`definitions`): a global MPCO recorder `opensees/recorder/#1` ({"file":
// "shoebuckle.mpco"}) and a stage-0 (`gravity`) Node recorder named `crown`
// ({"file": "gravity.out", "nodes": [62], "dofs": [2], "response": "disp"}).
// Every expected value below is a literal of that script; shoebuckle.h5
// (opensees 2.22.0) is the file without /opensees/decls. The group-selection,
// /provenance and /sequence cases edit the loaded state by hand, each edit
// named in its test.

import assert from "node:assert/strict";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { recordersOf, view } from "../../src/panels/windows/recorders.ts";
import { markSelected, SELECTED_STYLE } from "../../src/panels/windows/frame.ts";
import { WINDOWS } from "../../src/panels/windows/registry.ts";
import { openModel } from "../../src/reader/node.ts";
import { BlobStore } from "../../src/state/blobs.ts";
import { loadModel } from "../../src/state/load.ts";
import { initialState, reduce } from "../../src/state/reduce.ts";
import type { DeclRecord, Definitions, State } from "../../src/state/types.ts";

const fixtures = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "fixtures");
const load = async (f: string): Promise<State> =>
  reduce(initialState, { type: "fileLoaded", artifact: "model", load: loadModel(await openModel(f), new BlobStore()) });
const state = await load(join(fixtures, "definitions.h5"));
const plain = await load(join(fixtures, "shoebuckle.h5"));

/** `state` with the `crown` recorder's decl_params replaced by `params`. */
function withCrown(params: Record<string, unknown>): State {
  const defs = state.definitions!;
  const records = defs.decls!.records.map((r): DeclRecord =>
    r.key === "opensees/recorder/crown" ? { ...r, params: [{ ...r.params[0]!, params: params as DeclRecord["params"][0]["params"] }] } : r,
  );
  return { ...state, definitions: { ...defs, decls: { ...defs.decls!, records } } };
}

test("the window is the registry's recorders view", () => {
  assert.equal(WINDOWS.recorders.view, view);
  assert.equal(WINDOWS.recorders.title, "Recorders");
});

test("every recorder, global then per stage, by handle: the user's name, or the #k of its decls key", () => {
  const c = recordersOf(state);
  assert.equal(c.note, "2 recorders · parameters by field name from /opensees/decl_params");
  assert.deepEqual(
    c.entries.map((e) => [e.section, e.handle, e.declKey, e.type, e.h5]),
    [
      ["global", "#1", "opensees/recorder/#1", "mpco (MPCO)", "/opensees/recorders/mpco_0"],
      ["stage 0 · gravity", "crown", "opensees/recorder/crown", "Node", "/opensees/stages/stage_000/recorders/Node_1"],
    ],
  );
});

test("what it records, on what, to which file, and the stage that adds it", () => {
  const [mpco, crown] = recordersOf(state).entries;
  assert.equal(crown!.records, "disp · dofs 2");
  assert.equal(crown!.targets, "1 node");
  assert.equal(crown!.output, "file gravity.out · archive kind typed");
  assert.equal(crown!.lifetime, "added by stage 0 · gravity; removal not in this file (no /sequence zone)");
  assert.equal(mpco!.records, "no response field is set");
  assert.equal(mpco!.targets, "no target field is set (nodes, elements, pg, region)");
  assert.equal(mpco!.output, "file shoebuckle.mpco · archive kind typed");
  assert.equal(mpco!.lifetime, "global, added before the first stage: every stage records it; removal not in this file (no /sequence zone)");
});

test("parameters are decl_params' fields by name, never the positional argv", () => {
  const crown = recordersOf(state).entries[1]!;
  assert.deepEqual(crown.params, [
    { field: "file", value: "gravity.out" },
    { field: "nodes", value: "[62]" },
    { field: "dofs", value: "[2]" },
    { field: "response", value: "disp" },
  ]);
  assert.deepEqual(crown.unset, []);
  assert.equal(crown.paramsNote, null);
  // The positional argv the file also carries (params_str: -file gravity.out -node "" -dof "" disp) is not read.
  const argv = state.definitions!.stages[0]!.recorders[0]!.params;
  assert.ok(argv.includes("-node"), "the fixture's argv carries the flag tokens");
  assert.ok(!crown.params.some((p) => p.field.startsWith("-") || p.value === "-node"));
});

test("null and empty fields are listed as unset, not as values", () => {
  const e = recordersOf(withCrown({ file: "a.out", response: "disp", nodes: null, pg: null, dofs: [], dT: null })).entries[1]!;
  assert.deepEqual(e.params, [{ field: "file", value: "a.out" }, { field: "response", value: "disp" }]);
  assert.deepEqual(e.unset, ["nodes", "pg", "dofs", "dT"]);
  assert.equal(e.targets, "no target field is set (nodes, elements, pg, region)");
});

test("a file without /opensees/decls shows its recorders by group name and says why there are no fields", () => {
  const c = recordersOf(plain);
  assert.equal(c.note, "1 recorder · this file has no /opensees/decl_params, so no parameters by field name");
  const [e] = c.entries;
  assert.equal(e!.handle, "mpco_0");
  assert.equal(e!.declKey, null);
  assert.deepEqual(e!.params, []);
  assert.equal(e!.records, "not in this file (no parameters by field name)");
  assert.equal(e!.paramsNote, "parameters by field name: not joined (the file has no /opensees/decls (opensees < 2.25.0))");
  assert.equal(e!.lifetime, "global, unstaged model: the whole analysis records it; removal not in this file (no /sequence zone)");
  assert.equal(e!.source, null);
});

test("a recorder on a group of the store selects that group; the store accepts the pick", () => {
  const s = withCrown({ file: "a.out", response: "disp", pg: "Arch", dofs: [2] });
  const e = recordersOf(s).entries[1]!;
  assert.equal(e.targets, "group Arch");
  assert.equal(e.select, "mesh/physical_group/Arch");
  assert.equal(e.selectNote, null);
  const after = reduce(s, { type: "select", pick: { decl: e.select!, at: null, nodes: null } });
  assert.deepEqual(after.selection.decls, ["mesh/physical_group/Arch"]);
});

test("a group the store lacks selects nothing, and the row says so; a node list selects nothing", () => {
  const e = recordersOf(withCrown({ nodes_pg: "Nope", elements: [2, 3, 4] })).entries[1]!;
  assert.equal(e.targets, "nodes of group Nope · 3 elements");
  assert.equal(e.select, null);
  assert.equal(e.selectNote, "group Nope is not a group of this file, so a click selects nothing");
  assert.equal(recordersOf(state).entries[1]!.select, null);
});

test("go to source: the recorder's decls key joins /provenance by key; off with the reason otherwise", () => {
  assert.deepEqual(recordersOf(state).entries[1]!.source, { off: "the model file has no /provenance zone" });
  const src = { file: "/work/model.py", line: 12, function: "<module>", sha256: "", recorded: true, script: null };
  const s: State = {
    ...state,
    provenance: [{ key: "opensees/recorder/crown", origin: "user", seq: 0, source: src }],
    artifacts: { ...state.artifacts, model: { ...state.artifacts.model!, zones: { ...state.artifacts.model!.zones, provenance: { status: "ready", version: "1.1.0" } } } },
  };
  const [mpco, crown] = recordersOf(s).entries;
  assert.deepEqual(crown!.source, { decl: "opensees/recorder/crown", label: "model.py:12", title: "open /work/model.py:12 in the editor" });
  assert.deepEqual(mpco!.source, { off: "no /provenance record for opensees/recorder/#1" });
  // The store takes the dispatch the button makes.
  assert.equal(reduce(s, { type: "requestSource", decl: "opensees/recorder/crown" }).source.request?.decl, "opensees/recorder/crown");
});

test("removal comes from /sequence when the file has it", () => {
  const changes = { path: "/sequence/changes", columns: ["stage", "op", "what", "name", "count", "source"], rows: [[1, "remove", "recorder", "crown", 1, ""], [1, "add", "pattern", "push", 1, ""]] };
  const stages = { path: "/sequence/stages", columns: ["order", "name"], rows: [[0, "gravity"], [1, "push"]] };
  const s: State = { ...state, sequence: { status: "ready", version: "1.0.0", attrs: {}, stages, changes } };
  const [mpco, crown] = recordersOf(s).entries;
  assert.equal(crown!.lifetime, "added by stage 0 · gravity; removed by stage 1");
  assert.equal(mpco!.lifetime, "global, added before the first stage: every stage records it; removal: unknown (unnamed; /sequence matches by name)");
  const refused: State = { ...state, sequence: { status: "refused", version: "9.0.0", accepted: "1.x", newer: true, reason: "newer" } };
  assert.match(recordersOf(refused).entries[1]!.lifetime, /removal: unknown \(\/sequence 9\.0\.0 was not read: newer\)$/);
});

test("an empty-name remove row in /sequence marks no unnamed recorder removed; a named one reads never removed", () => {
  // /sequence writes "" for every unnamed recorder, so the row cannot say which #k it removes.
  const changes = { path: "/sequence/changes", columns: ["stage", "op", "what", "name", "count", "source"], rows: [[1, "remove", "recorder", "", 1, ""]] };
  const stages = { path: "/sequence/stages", columns: ["order", "name"], rows: [[0, "gravity"], [1, "push"]] };
  const s: State = { ...state, sequence: { status: "ready", version: "1.0.0", attrs: {}, stages, changes } };
  const [mpco, crown] = recordersOf(s).entries;
  assert.equal(mpco!.handle, "#1");
  assert.equal(mpco!.lifetime, "global, added before the first stage: every stage records it; removal: unknown (unnamed; /sequence matches by name)");
  assert.doesNotMatch(mpco!.lifetime, /removed by stage/);
  assert.equal(crown!.lifetime, "added by stage 0 · gravity; never removed (/sequence)");
});

test("the empty states are explicit", () => {
  assert.deepEqual(recordersOf(initialState), { entries: [], note: "no model is open" });
  assert.deepEqual(recordersOf({ ...state, definitions: null }), { entries: [], note: "the model has no /opensees zone, so it declares no recorder" });
  const defs: Definitions = { ...state.definitions!, recorders: [], stages: state.definitions!.stages.map((st) => ({ ...st, recorders: [] })) };
  assert.deepEqual(recordersOf({ ...state, definitions: defs }), { entries: [], note: "the model declares no recorder (no /opensees/recorders, and no stage adds one)" });
});

test("a decls row count that differs from the recorders joins nothing, and says why", () => {
  const defs = state.definitions!;
  const s: State = { ...state, definitions: { ...defs, decls: { ...defs.decls!, rows: { ...defs.decls!.rows, recorders: [7, 7] } } } };
  const e = recordersOf(s).entries[0]!;
  assert.equal(e.handle, "mpco_0");
  assert.equal(e.declKey, null);
  assert.equal(e.paramsNote, "parameters by field name: not joined (/opensees/decls/rows/recorders has 2 rows for 1 recorders)");
});

test("a decls row past the records leaves that one recorder unjoined, with its own reason; the others still join", () => {
  const defs = state.definitions!;
  const crownGroup = { ...defs.stages[0]!.recorders[0]!, path: "/opensees/recorders/Node_1" };
  // Index order mpco_0 -> rows[0] = 7 (#1); Node_1 -> rows[1] = 999, past the 16 records.
  const s: State = { ...state, definitions: { ...defs, recorders: [crownGroup, defs.recorders[0]!], decls: { ...defs.decls!, rows: { ...defs.decls!.rows, recorders: [7, 999] } } } };
  const global = recordersOf(s).entries.filter((e) => e.section === "global");
  assert.deepEqual(global.map((e) => [e.handle, e.declKey]), [["#1", "opensees/recorder/#1"], ["Node_1", null]]);
  assert.equal(global[0]!.paramsNote, null);
  assert.equal(global[1]!.paramsNote, "parameters by field name: not joined (/opensees/decls/rows/recorders[1] = 999 is not a row of /opensees/decls (16 records))");
});

test("a group name without a trailing _<index> joins nothing, and the reason names it", () => {
  const defs = state.definitions!;
  const s: State = { ...state, definitions: { ...defs, recorders: [{ ...defs.recorders[0]!, groupName: "mpco" }] } };
  assert.equal(
    recordersOf(s).entries[0]!.paramsNote,
    "parameters by field name: not joined (the group name mpco has no trailing _<index>, so /opensees/decls/rows/recorders cannot be ordered)",
  );
});

test("within a section, recorders list in declaration order, not the file's group order (kr3pb: ladruno_0 is #1)", () => {
  const defs = state.definitions!;
  const crownGroup = { ...defs.stages[0]!.recorders[0]!, path: "/opensees/recorders/Node_1" };
  // File order Node_1, mpco_0; index order (rows/recorders) mpco_0 -> decl 7 (#1), Node_1 -> decl 11 (crown).
  const s: State = { ...state, definitions: { ...defs, recorders: [crownGroup, defs.recorders[0]!], decls: { ...defs.decls!, rows: { ...defs.decls!.rows, recorders: [7, 11] } } } };
  assert.deepEqual(recordersOf(s).entries.filter((e) => e.section === "global").map((e) => [e.handle, e.h5]), [
    ["#1", "/opensees/recorders/mpco_0"],
    ["crown", "/opensees/recorders/Node_1"],
  ]);
});

// ---- V4f: the selected row's highlight (a minimal DOM: only what the view and markSelected touch) ----

class RecNode {
  children: RecNode[] = [];
  className = "";
  title = "";
  textContent = "";
  disabled = false;
  dataset: Record<string, string> = {};
  style: Record<string, string> = {};
  private handlers: (() => void)[] = [];
  append(...xs: (RecNode | string)[]): void {
    for (const x of xs) if (typeof x !== "string") this.children.push(x);
  }
  addEventListener(_t: string, fn: () => void): void {
    this.handlers.push(fn);
  }
  click(): void {
    for (const h of this.handlers) h();
  }
  all(pred: (n: RecNode) => boolean): RecNode[] {
    return [...(pred(this) ? [this] : []), ...this.children.flatMap((c) => c.all(pred))];
  }
}

test("V4f: a row that selects a group carries data-decl, and the frame highlights it while the store selects that group", () => {
  const g = globalThis as { document?: unknown };
  const saved = g.document;
  g.document = { createElement: () => new RecNode() };
  try {
    const s = withCrown({ file: "a.out", response: "disp", pg: "Arch", dofs: [2] });
    const events: unknown[] = [];
    const rows = (view(s, (e) => events.push(e)) as unknown as RecNode).all((n) => n.className.split(" ").includes("window-row"));
    const arch = rows.filter((r) => r.dataset["decl"] === "mesh/physical_group/Arch");
    assert.equal(arch.length, 1, "the crown row names the group it selects");
    assert.ok(rows.filter((r) => r !== arch[0]).every((r) => r.dataset["decl"] === undefined), "a row that selects nothing carries no data-decl");
    arch[0]!.click();
    const selected = reduce(s, events[0] as Parameters<typeof reduce>[1]);
    const again = view(selected, () => {}) as unknown as RecNode;
    assert.equal(markSelected(again as unknown as HTMLElement, selected.selection.decls), 1);
    const hit = again.all((n) => n.className.split(" ").includes("selected"));
    assert.deepEqual(hit.map((r) => r.dataset["decl"]), ["mesh/physical_group/Arch"]);
    assert.equal(hit[0]!.style["background"], SELECTED_STYLE.background);
  } finally {
    g.document = saved;
  }
});
