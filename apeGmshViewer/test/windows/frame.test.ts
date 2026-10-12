// V4f (#1708): the window frame re-renders from the store on a selection, a
// stage selection and a go-to-source answer (D6: a click comes back through
// the store as a redraw, never as a panel-to-panel call).

import assert from "node:assert/strict";
import { test } from "node:test";
import { changed } from "../../src/panels/windows/frame.ts";
import { initialState } from "../../src/state/reduce.ts";
import type { State } from "../../src/state/types.ts";

test("the frame redraws on a selection, a stage selection and a go-to-source answer", () => {
  const s = initialState;
  const cases: [string, State][] = [
    ["selection", { ...s, selection: { decls: ["opensees/nDMaterial/#1"], picks: [] } }],
    ["stage", { ...s, stage: 0 }],
    ["source", { ...s, source: { ...s.source, last: { decl: "opensees/nDMaterial/#1", ok: false, reason: "missing" } } }],
    ["provenance", { ...s, provenance: [] }],
  ];
  for (const [what, next] of cases) assert.equal(changed(next, s), true, what);
});

test("the frame does not redraw on state no window reads (hover, the camera)", () => {
  const s = initialState;
  assert.equal(changed({ ...s, hover: "x" }, s), false);
  assert.equal(changed({ ...s, view: { frameSeq: 3 } }, s), false);
});
