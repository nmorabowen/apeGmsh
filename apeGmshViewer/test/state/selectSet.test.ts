// ADR 0138 D12: a saved set into the selection, from the selection bar's
// list: select, add, subtract; and the list's rows (hide and isolate).

import assert from "node:assert/strict";
import { test } from "node:test";
import { setListLines } from "../../src/panels/selectionBar.ts";
import { initialState, reduce } from "../../src/state/reduce.ts";
import { selectedElementsOf } from "../../src/state/search.ts";
import type { Decl, MeshInfo, State, WorkspaceSetIn } from "../../src/state/types.ts";

/** Elements 0..n-1, element i carrying OpenSees tag 100 + i, and a workspace read holding `sets`. */
function model(n: number, sets: WorkspaceSetIn[]): State {
  const decls = Object.create(null) as State["decls"];
  for (let i = 0; i < n; i++) {
    const path = `mesh/element/${i}`;
    decls[path] = { path, kind: "element", name: path, type: "tri3", tag: i, h5: "", fields: [], refs: {}, element: { femId: i, cell: null, metas: [{ type: "ShellMITC4", tag: 100 + i, h5: "", row: i, args: [] }], groups: [] } } as unknown as Decl;
  }
  const read = { status: "ready", file: "m.workspace.h5", token: "t", version: "1", revision: 1, writer: null, saveId: null, readOnly: null, sets, plots: [], views: [], refused: [], broken: [] } as const;
  return { ...initialState, decls, mesh: { elements: [] } as unknown as MeshInfo, workspace: { ...initialState.workspace, read } };
}

const current = (n: number) => ({ verdict: "current" as const, matched: n, total: n, members: Array<boolean>(n).fill(true), checked: false, maxDistance: null }) as unknown as WorkspaceSetIn["verdict"];
const EDGE: WorkspaceSetIn = { row: 0, name: "Edge", kind: "elements", ids: [101, 103], verdict: current(2) };
const PARTIAL: WorkspaceSetIn = { row: 1, name: "Partial", kind: "elements", ids: [100, 104, 999], verdict: { verdict: "partial", matched: 1, total: 3, members: [true, false, true], checked: true, maxDistance: 0 } as unknown as WorkspaceSetIn["verdict"] };
const SUPPORTS: WorkspaceSetIn = { row: 2, name: "Supports", kind: "nodes", ids: [7, 8, 9], verdict: current(3) };

test("an element set selects the elements carrying its OpenSees tags: replace, add, subtract", () => {
  let s = model(5, [EDGE, PARTIAL]);
  s = reduce(s, { type: "selectSet", row: 0, mode: "replace" });
  assert.deepEqual(selectedElementsOf(s), ["mesh/element/1", "mesh/element/3"]);
  s = reduce(s, { type: "selectSet", row: 1, mode: "add" });
  assert.deepEqual(selectedElementsOf(s), ["mesh/element/1", "mesh/element/3", "mesh/element/0"], "the member that no longer matches (104) and the tag no element carries (999) are left out");
  s = reduce(s, { type: "selectSet", row: 0, mode: "remove" });
  assert.deepEqual(selectedElementsOf(s), ["mesh/element/0"]);
});

test("a node set selects its tags in the node kind and makes Nodes active; a set selection is one undo step", () => {
  let s = reduce(model(2, [SUPPORTS]), { type: "selectNodes", tags: [1], mode: "replace" });
  s = reduce(s, { type: "selectSet", row: 2, mode: "add" });
  assert.deepEqual([s.selection.nodes, s.selection.active], [[1, 7, 8, 9], "node"]);
  s = reduce(s, { type: "undoSelection" });
  assert.deepEqual(s.selection.nodes, [1]);
  assert.throws(() => reduce(s, { type: "selectSet", row: 9, mode: "replace" }), /no set at row 9/);
});

test("the bar's list holds the active kind's sets, with what selecting takes and whether Hide and Isolate apply", () => {
  let s = model(5, [EDGE, PARTIAL, SUPPORTS]);
  const el = setListLines(s)!;
  assert.deepEqual(el.rows.map((r) => [r.name, r.detail, r.decls.length, r.noHide]), [
    ["Edge", "2", 2, null],
    ["Partial", "1 of 3 · partial", 1, null],
  ]);
  s = reduce(s, { type: "setWorkspaceKind", kind: "nodes" });
  const nodes = setListLines(s)!;
  assert.deepEqual(nodes.rows.map((r) => [r.name, r.detail]), [["Supports", "3"]]);
  assert.match(nodes.rows[0]!.noHide!, /later slice/, "a node set's Hide waits for the elements of its nodes (ADR 0138 D4)");
  assert.match(setListLines(model(1, []))!.empty, /No element sets saved/);
});
