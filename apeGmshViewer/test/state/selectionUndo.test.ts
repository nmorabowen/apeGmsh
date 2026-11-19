// ADR 0138 D6: selection changes have their own undo, so a click on empty
// space never loses a large selection for good.

import assert from "node:assert/strict";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { openModel } from "../../src/reader/node.ts";
import { BlobStore } from "../../src/state/blobs.ts";
import { loadModel } from "../../src/state/load.ts";
import { initialState, reduce, SELECTION_UNDO_LIMIT } from "../../src/state/reduce.ts";
import { selectionBarLines } from "../../src/panels/selectionBar.ts";
import type { Decl, Event, MeshInfo, State } from "../../src/state/types.ts";

/** `n` elements, all in group G, with a mesh so the selection bar shows. */
function grouped(n: number): State {
  const decls = Object.create(null) as State["decls"];
  for (let i = 0; i < n; i++) {
    const path = `mesh/element/${i}`;
    decls[path] = { path, kind: "element", name: path, type: "tri3", tag: i, h5: "", fields: [], refs: {}, element: { femId: i, cell: null, metas: [], groups: ["group/G"] } } as unknown as Decl;
  }
  decls["group/G"] = { path: "group/G", kind: "physical_group", name: "G", type: "", tag: 1, h5: "", fields: [], refs: {}, group: { dim: 2, tag: 1, count: n } } as unknown as Decl;
  return { ...initialState, decls, mesh: { elements: [] } as unknown as MeshInfo };
}

const run = (s: State, ...events: Event[]) => events.reduce(reduce, s);

test("undo steps back through edits of both kinds, making the edited kind active; redo replays them; a new edit clears redo", () => {
  let s = run(
    grouped(5),
    { type: "selectDecls", decls: ["group/G"] },
    { type: "setWorkspaceKind", kind: "nodes" },
    { type: "selectNodes", tags: [7, 8], mode: "replace" },
    { type: "setWorkspaceKind", kind: "elements" },
    { type: "clearSelection" },
  );
  assert.deepEqual([s.selection.decls, s.selection.nodes], [[], [7, 8]]);
  assert.equal(s.selectionUndo.past.length, 3, "three edits; switching the kind is not one");
  s = reduce(s, { type: "undoSelection" });
  assert.deepEqual(s.selection.decls, ["group/G"], "the click on empty space is undone: the group is back");
  s = reduce(s, { type: "undoSelection" });
  assert.deepEqual([s.selection.nodes, s.selection.active], [[], "node"], "undoing a node edit makes Nodes active");
  s = reduce(s, { type: "undoSelection" });
  assert.deepEqual([s.selection.decls, s.selection.active], [[], "element"]);
  assert.equal(reduce(s, { type: "undoSelection" }), s, "nothing left to undo: the state is unchanged");
  s = run(s, { type: "redoSelection" }, { type: "redoSelection" });
  assert.deepEqual([s.selection.decls, s.selection.nodes, s.selection.active], [["group/G"], [7, 8], "node"]);
  assert.equal(s.selectionUndo.future.length, 1);
  s = reduce(s, { type: "selectNodes", tags: [9], mode: "add" });
  assert.equal(s.selectionUndo.future.length, 0, "a new edit clears redo");
  assert.equal(reduce(s, { type: "redoSelection" }), s);
});

test("an edit that changes nothing is no step: Esc on an empty kind, the same click twice, a remove of what is not selected", () => {
  const pick = { decl: "mesh/element/1", at: null, nodes: null };
  let s = run(grouped(3), { type: "clearSelection" }, { type: "select", pick });
  assert.equal(s.selectionUndo.past.length, 1);
  s = run(s, { type: "select", pick }, { type: "selectDecls", decls: ["mesh/element/1"] }, { type: "selectNodes", tags: [4], mode: "remove" });
  assert.equal(s.selectionUndo.past.length, 1);
});

test(`undo keeps the last ${SELECTION_UNDO_LIMIT} edits`, () => {
  let s = grouped(1);
  for (let i = 0; i < SELECTION_UNDO_LIMIT + 20; i++) s = reduce(s, { type: "selectNodes", tags: [i], mode: "replace" });
  assert.equal(s.selectionUndo.past.length, SELECTION_UNDO_LIMIT);
  for (let i = 0; i < SELECTION_UNDO_LIMIT; i++) s = reduce(s, { type: "undoSelection" });
  assert.deepEqual(s.selection.nodes, [19], "the oldest steps kept are the ones after the first 20");
});

test("the selection bar's Undo is enabled once there is an edit to undo", () => {
  let s = grouped(2);
  assert.equal(selectionBarLines(s)!.canUndo, false);
  s = reduce(s, { type: "selectDecls", decls: ["mesh/element/0"] });
  assert.equal(selectionBarLines(s)!.canUndo, true);
  assert.equal(selectionBarLines(reduce(s, { type: "undoSelection" }))!.canUndo, false);
});

test("a model read empties the undo: a step could name a declaration the new read does not have", async () => {
  const model = await openModel(fileURLToPath(new URL("../../fixtures/shoebuckle.h5", import.meta.url)));
  const load = loadModel(model, new BlobStore());
  let s = reduce(initialState, { type: "fileLoaded", artifact: "model", load });
  const decl = Object.keys(s.decls).find((d) => s.decls[d]!.kind === "element")!;
  s = reduce(s, { type: "selectDecls", decls: [decl] });
  assert.equal(s.selectionUndo.past.length, 1);
  s = reduce(s, { type: "fileLoaded", artifact: "model", load });
  assert.deepEqual(s.selectionUndo, { past: [], future: [] });
});
