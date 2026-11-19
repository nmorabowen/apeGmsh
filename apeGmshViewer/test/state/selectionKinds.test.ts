// ADR 0138 SEL-1a: the element kind's two faults and the node kind.
//
// - Shift+click on an element selected through its group took the element in
//   beside the group, so it stayed selected; now it comes out, and the group
//   is replaced by its other elements.
// - Adding to a large selection looked up each held pick for each declaration;
//   now one map serves them all.

import assert from "node:assert/strict";
import { test } from "node:test";
import { editNodes, initialState, reduce } from "../../src/state/reduce.ts";
import { selectedElementsOf } from "../../src/state/search.ts";
import type { Decl, State } from "../../src/state/types.ts";

/** `n` elements, all in group G. */
function grouped(n: number): State {
  const decls = Object.create(null) as State["decls"];
  for (let i = 0; i < n; i++) {
    const path = `mesh/element/${i}`;
    decls[path] = { path, kind: "element", name: path, type: "tri3", tag: i, h5: "", fields: [], refs: {}, element: { femId: i, cell: null, metas: [], groups: ["group/G"] } } as unknown as Decl;
  }
  decls["group/G"] = { path: "group/G", kind: "physical_group", name: "G", type: "", tag: 1, h5: "", fields: [], refs: {}, group: { dim: 2, tag: 1, count: n } } as unknown as Decl;
  return { ...initialState, decls };
}

test("Shift+click on an element selected through its group takes it out, and the group gives way to its other elements", () => {
  let s = reduce(grouped(4), { type: "selectDecls", decls: ["group/G"] });
  assert.equal(selectedElementsOf(s).length, 4);
  s = reduce(s, { type: "selectAdd", pick: { decl: "mesh/element/2", at: [0, 0, 0], nodes: null } });
  assert.deepEqual(selectedElementsOf(s), ["mesh/element/0", "mesh/element/1", "mesh/element/3"]);
  assert.deepEqual(s.selection.decls, ["mesh/element/0", "mesh/element/1", "mesh/element/3"], "the group is no longer selected as a whole");
  assert.equal(s.selection.picks.length, s.selection.decls.length, "one pick per declaration");
  s = reduce(s, { type: "selectAdd", pick: { decl: "mesh/element/2", at: [0, 0, 0], nodes: null } });
  assert.equal(selectedElementsOf(s).length, 4, "a second Shift+click puts it back");
});

test("adding to a selection of every element is linear: 60,000 elements in well under a second", () => {
  const s0 = grouped(60000);
  const all = Object.keys(s0.decls).filter((p) => p !== "group/G");
  const s = reduce(s0, { type: "selectDecls", decls: all });
  const t = performance.now();
  const next = reduce(s, { type: "selectDecls", decls: [all[0]!], add: true });
  const ms = performance.now() - t;
  assert.equal(next.selection.picks.length, 60000);
  // The pairwise lookup this replaces took about 2.8 s at 26,970 elements (internal_docs/spike_viewer_selection.md).
  assert.ok(ms < 1000, `took ${ms.toFixed(0)} ms`);
});

test("Ctrl on a box or click: the elements come out, a selected group gives way to its other elements", () => {
  let s = reduce(grouped(5), { type: "selectDecls", decls: ["group/G"] });
  s = reduce(s, { type: "selectDecls", decls: ["mesh/element/1", "mesh/element/3"], remove: true });
  assert.deepEqual(selectedElementsOf(s), ["mesh/element/0", "mesh/element/2", "mesh/element/4"]);
  s = reduce(s, { type: "selectDecls", decls: ["mesh/element/0"], remove: true });
  assert.deepEqual(s.selection.decls, ["mesh/element/2", "mesh/element/4"]);
  s = reduce(s, { type: "selectDecls", decls: ["group/G"], remove: true });
  assert.deepEqual(s.selection.decls, [], "removing the group takes every element out");
});

test("node tags edit in pick order: replace, add, toggle, remove; a bad tag is refused", () => {
  assert.deepEqual(editNodes([1, 2, 3], [2, 9], "remove"), [1, 3]);
  assert.deepEqual(editNodes([1, 2], [3, 3, 4], "replace"), [3, 4]);
  assert.deepEqual(editNodes([1, 2], [2, 5], "add"), [1, 2, 5]);
  assert.deepEqual(editNodes([1, 2], [2, 5], "toggle"), [1, 5]);
  assert.throws(() => editNodes([], [1.5], "add"), RangeError);
});

test("switching the active kind keeps each kind's selection", () => {
  let s = reduce(grouped(2), { type: "setWorkspaceKind", kind: "nodes" });
  s = reduce(s, { type: "selectNodes", tags: [7], mode: "replace" });
  s = reduce(s, { type: "select", pick: { decl: "mesh/element/0", at: null, nodes: null } });
  assert.equal(s.selection.active, "node");
  assert.deepEqual(s.selection.nodes, [7]);
  assert.deepEqual(s.selection.decls, ["mesh/element/0"]);
});
