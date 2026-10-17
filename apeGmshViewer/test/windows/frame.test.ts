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

class FakeRowNode {
  children: FakeRowNode[] = [];
  className = "";
  title = "";
  textContent = "";
  private handlers: ((e?: { shiftKey?: boolean; ctrlKey?: boolean; metaKey?: boolean }) => void)[] = [];
  append(...xs: (FakeRowNode | string)[]): void {
    for (const x of xs) if (typeof x !== "string") this.children.push(x);
  }
  addEventListener(_t: string, fn: (e?: { shiftKey?: boolean; ctrlKey?: boolean; metaKey?: boolean }) => void): void {
    this.handlers.push(fn);
  }
  click(mods?: { shiftKey?: boolean; ctrlKey?: boolean; metaKey?: boolean }): void {
    for (const h of this.handlers) h(mods);
  }
  all(pred: (n: FakeRowNode) => boolean): FakeRowNode[] {
    return [...(pred(this) ? [this] : []), ...this.children.flatMap((c) => c.all(pred))];
  }
}

test("placeholderList: clickable rows dispatch select on plain click and selectAdd on Shift/Ctrl/Meta click", async () => {
  const { placeholderList } = await import("../../src/panels/windows/view.ts");
  const g = globalThis as { document?: unknown };
  const saved = g.document;
  g.document = { createElement: () => new FakeRowNode() };
  try {
    const events: unknown[] = [];
    const root = placeholderList(
      {
        note: "note",
        rows: [
          { key: "r1", section: "s1", title: "r1", declKey: "k1", detail: "d1", h5: "/a", select: "opensees/nDMaterial/#1" },
          { key: "r2", section: "s1", title: "r2", declKey: null, detail: "", h5: "/b", select: null },
        ],
      },
      (e) => events.push(e),
    ) as unknown as FakeRowNode;
    const rows = root.all((n) => n.className.split(" ").includes("window-row"));
    assert.equal(rows.length, 2);
    rows[0]!.click();
    rows[0]!.click({ shiftKey: true });
    rows[0]!.click({ metaKey: true });
    rows[1]!.click({ shiftKey: true });
    assert.deepEqual(events, [
      { type: "select", pick: { decl: "opensees/nDMaterial/#1", at: null, nodes: null } },
      { type: "selectAdd", pick: { decl: "opensees/nDMaterial/#1", at: null, nodes: null } },
      { type: "selectAdd", pick: { decl: "opensees/nDMaterial/#1", at: null, nodes: null } },
    ]);
  } finally {
    g.document = saved;
  }
});

