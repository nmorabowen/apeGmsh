// V4f (#1708, review of #1714): `mountWindows` end to end on a fake DOM. A
// `select` dispatched to the store redraws the open windows with the selected
// row marked, and the redraw keeps the host's scroll and each stage
// timeline's horizontal scroll.
//
// The fake host models the worst case the frame guards against: replacing its
// children drops its scroll to 0 (a browser clamps scrollTop when the content
// is momentarily gone), and a new timeline element starts at scrollLeft 0.
//
// Oracles: fixtures/definitions.h5 (make_zone_fixtures.py `definitions`): its
// material `opensees/uniaxialMaterial/#1` is the first materials row. The
// /sequence zone is literal rows here (two stages, gravity and push).

import assert from "node:assert/strict";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { mountWindows } from "../../src/panels/windows/frame.ts";
import { openModel } from "../../src/reader/node.ts";
import { BlobStore } from "../../src/state/blobs.ts";
import { loadModel } from "../../src/state/load.ts";
import { Store } from "../../src/state/store.ts";
import type { SequenceZone } from "../../src/state/types.ts";

// ---- a fake DOM: what the frame and the materials / stages views touch ----

class FakeEl {
  children: FakeEl[] = [];
  parent: FakeEl | null = null;
  className = "";
  id = "";
  title = "";
  hidden = false;
  disabled = false;
  dataset: Record<string, string> = {};
  style: Record<string, string> = { cssText: "" };
  attrs: Record<string, string> = {};
  scrollTop = 0;
  scrollLeft = 0;
  private text = "";
  private listeners: (() => void)[] = [];
  readonly tagName: string;
  constructor(tagName: string) {
    this.tagName = tagName;
  }
  get textContent(): string {
    return this.text + this.children.map((c) => c.textContent).join("");
  }
  set textContent(t: string) {
    this.text = t;
    this.children = [];
  }
  classList = {
    toggle: (c: string, on: boolean) => {
      const set = new Set(this.className.split(" ").filter(Boolean));
      if (on) set.add(c);
      else set.delete(c);
      this.className = [...set].join(" ");
    },
  };
  setAttribute(k: string, v: string): void {
    this.attrs[k] = v;
  }
  append(...xs: (FakeEl | string)[]): void {
    for (const x of xs) {
      let n: FakeEl;
      if (typeof x === "string") {
        n = new FakeEl("#text");
        n.text = x;
      } else n = x;
      n.parent = this;
      this.children.push(n);
    }
  }
  /** Replacing the children drops the scroll, as a browser may when the content is momentarily gone. */
  replaceChildren(...xs: FakeEl[]): void {
    this.children = [];
    this.scrollTop = 0;
    this.append(...xs);
  }
  remove(): void {
    if (this.parent) this.parent.children = this.parent.children.filter((c) => c !== this);
  }
  addEventListener(_t: string, fn: () => void): void {
    this.listeners.push(fn);
  }
  removeEventListener(_t: string, fn: () => void): void {
    this.listeners = this.listeners.filter((f) => f !== fn);
  }
  click(): void {
    for (const f of this.listeners) f();
  }
  has(cls: string): boolean {
    return this.className.split(" ").includes(cls);
  }
  all(pred: (e: FakeEl) => boolean): FakeEl[] {
    const out: FakeEl[] = [];
    for (const c of this.children) {
      if (pred(c)) out.push(c);
      out.push(...c.all(pred));
    }
    return out;
  }
  closest(sel: string): FakeEl | null {
    if (sel !== "[data-window]") throw new Error(`fake closest: ${sel}`);
    for (let e: FakeEl | null = this; e; e = e.parent) if (e.dataset["window"] !== undefined) return e;
    return null;
  }
  querySelectorAll(sel: string): FakeEl[] {
    if (sel === "[data-window] .stage-timeline") return this.all((e) => e.has("stage-timeline") && e.parent?.closest("[data-window]") !== null);
    throw new Error(`fake querySelectorAll: ${sel}`);
  }
  querySelector(sel: string): FakeEl | null {
    if (sel === ".stage-timeline") return this.all((e) => e.has("stage-timeline"))[0] ?? null;
    throw new Error(`fake querySelector: ${sel}`);
  }
}

const body = new FakeEl("body");
(globalThis as { document?: unknown }).document = { createElement: (t: string) => new FakeEl(t.toUpperCase()), body };

// ---- the model: definitions.h5 plus a two-stage /sequence ----

const fixtures = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "fixtures");
const sequence: SequenceZone = {
  status: "ready",
  version: "1.0.0",
  attrs: { staged: 1 },
  stages: {
    path: "/sequence/stages",
    columns: ["order", "name", "analysis", "integrator", "algorithm", "test", "steps", "dt", "set_time", "domain_change", "source"],
    rows: [
      [0, "gravity", "Static", "LoadControl", "Newton", "NormDispIncr", 10, Number.NaN, Number.NaN, 1, "/opensees/stages/stage_000"],
      [1, "push", "Transient", "Newmark", "Newton", "NormDispIncr", 5, 0.01, 0, 0, "/opensees/stages/stage_001"],
    ],
  },
  changes: { path: "/sequence/changes", columns: ["stage", "op", "what", "name", "count", "source"], rows: [] },
};
const load = { ...loadModel(await openModel(join(fixtures, "definitions.h5")), new BlobStore()), sequence };
const M = "opensees/uniaxialMaterial/#1";

test("mountWindows: a select redraws the open window with the selected row marked, and keeps both scroll positions", () => {
  const store = new Store();
  store.dispatch({ type: "fileLoaded", artifact: "model", load });
  const unmount = mountWindows(store);
  try {
    const host = body.children.find((c) => c.id === "windows")!;
    store.dispatch({ type: "openWindow", window: "materials" });
    store.dispatch({ type: "openWindow", window: "stages" });
    const rowsOf = () => host.all((e) => e.has("window-row") && e.dataset["decl"] !== undefined);
    assert.ok(rowsOf().some((r) => r.dataset["decl"] === M), "the materials window lists the fixture's material");
    assert.equal(host.all((e) => e.has("selected")).length, 0, "nothing selected: nothing marked");
    host.scrollTop = 120;
    host.all((e) => e.has("stage-timeline"))[0]!.scrollLeft = 55;

    store.dispatch({ type: "select", pick: { decl: M, at: null, nodes: null } });

    const marked = host.all((e) => e.has("window-row") && e.has("selected"));
    assert.deepEqual(marked.map((r) => r.dataset["decl"]), [M], "the redraw marks the selected row, and only it");
    assert.equal(marked[0]!.style["background"], "var(--accent-wash)");
    assert.equal(host.scrollTop, 120, "the host keeps its scroll across the redraw");
    assert.equal(host.all((e) => e.has("stage-timeline"))[0]!.scrollLeft, 55, "the stage timeline keeps its horizontal scroll");

    // A stage card click comes back through the store as a highlighted card.
    host.all((e) => e.has("stage-card"))[1]!.click();
    assert.equal(store.get().stage, 1);
    assert.deepEqual(host.all((e) => e.has("stage-card")).map((c) => c.has("selected")), [false, true]);
  } finally {
    unmount();
  }
});
