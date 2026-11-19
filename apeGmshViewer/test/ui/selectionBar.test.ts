// ADR 0138 D2: the selection bar's lines, read from the store.

import assert from "node:assert/strict";
import { test } from "node:test";
import { selectionBarLines } from "../../src/panels/selectionBar.ts";
import { initialState } from "../../src/state/store.ts";
import type { MeshInfo, State } from "../../src/state/types.ts";

test("the bar shows only with a mesh; it counts each kind and clears the active one", () => {
  assert.equal(selectionBarLines(initialState), null);
  const s: State = { ...initialState, mesh: { elements: [] } as unknown as MeshInfo, selection: { active: "node", decls: [], picks: [], nodes: [4, 5], through: false } };
  const lines = selectionBarLines(s)!;
  assert.deepEqual(lines.kinds.map((k) => [k.kind, k.count, k.active]), [["element", 0, false], ["node", 2, true]]);
  assert.equal(lines.canClear, true, "two nodes selected in the active kind");
  assert.equal(selectionBarLines({ ...s, selection: { ...s.selection, active: "element" } })!.canClear, false);
});
