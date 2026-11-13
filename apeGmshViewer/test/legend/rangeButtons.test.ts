// #2902 item 2: the colour-range buttons on the scale. A user result has one
// state (VU16: no history, no step), so "history" and "this step" both showed
// its own range under two names. The oracle is the user result's own read: its
// range is the column's min and max (test/state/userResult.test.ts holds that
// against W3's golden), and the scale offers it once, as "own range", beside
// "fixed". A User Combination (one state) and a difference (the step shown)
// range over one set of values too, so they offer it once as well. A Results
// field keeps RANGE_MODES as they are.

import assert from "node:assert/strict";
import { readdirSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { mountLegend, rangeButtonsOf } from "../../src/panels/legend.ts";
import { readUserResultColumn, readUserSet, type UserH5 } from "../../src/reader/userFile.ts";
import { UserResultFeed, type UserResultFetch } from "../../src/renderer/contour.ts";
import { RANGE_MODES } from "../../src/state/contour.ts";
import { userEventOf } from "../../src/state/reduce.ts";
import { contourView } from "../../src/state/selectors.ts";
import { harness } from "../contour/harness.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as UserH5;
const at = (name: string) => fileURLToPath(new URL(`../fixtures/vu16/${name}`, import.meta.url));
const PLATE = at("plate.h5");
const PLATE_USER = at("plate.user.h5");
const RATIO = "footfall_ratio";

test("a contour with history and steps offers RANGE_MODES as they are; a view with one range offers it once, and fixed", () => {
  // A Results field (nodal or element) scans its stage's history or reads the step shown, by the mode.
  for (const view of ["ready", "element"] as const) {
    for (const range of ["history", "step", "fixed"] as const) {
      assert.deepEqual(
        rangeButtonsOf(view, range).map((b) => [b.range, b.own, b.label, b.on]),
        RANGE_MODES.map((m) => [m.range, false, m.label, m.range === range]),
        `${view} at ${range}`,
      );
    }
  }
  // A user result and a User Combination: one state, so their own range whatever the mode (selectors.ts
  // userResultContour, combinationClim). A difference: the step shown whatever the mode (differenceContour).
  const own = { user: ["own range", "history"], combination: ["own range", "history"], difference: ["this step", "step"] } as const;
  for (const [view, [label, back]] of Object.entries(own) as [keyof typeof own, readonly [string, string]][]) {
    for (const range of ["history", "step"] as const) {
      assert.deepEqual(rangeButtonsOf(view, range).map((b) => [b.range, b.own, b.label, b.on]), [[back, true, label, true], ["fixed", false, "fixed", false]], `${view} at ${range}`);
    }
    assert.deepEqual(rangeButtonsOf(view, "fixed").map((b) => [b.range, b.own, b.label, b.on]), [[back, true, label, false], ["fixed", false, "fixed", true]], `${view} at fixed`);
    assert.ok(!rangeButtonsOf(view, "history").some((b) => b.label === "history"), `${view}: no history button`);
  }
  assert.ok(!rangeButtonsOf("user", "history").some((b) => /step/.test(b.label)), "a user result has no step either");
  assert.ok(!rangeButtonsOf("combination", "history").some((b) => /step/.test(b.label)), "nor has a combination");
});

/** Enough of the DOM for mountLegend (as test/contour/contour.test.ts's). */
class FakeEl {
  tagName: string;
  id = "";
  className = "";
  title = "";
  value = "";
  type = "";
  step = "";
  min = "";
  label = "";
  hidden = false;
  disabled = false;
  selected = false;
  open = false;
  dataset: Record<string, string | undefined> = {};
  style: Record<string, string> = {};
  children: (FakeEl | string)[] = [];
  listeners = new Map<string, ((ev?: unknown) => void)[]>();
  constructor(tag: string) {
    this.tagName = tag.toUpperCase();
  }
  append(...xs: (FakeEl | string)[]) {
    this.children.push(...xs);
  }
  replaceChildren(...xs: (FakeEl | string)[]) {
    this.children = [...xs];
  }
  remove() {}
  setCustomValidity() {}
  addEventListener(type: string, fn: (ev?: unknown) => void) {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), fn]);
  }
  removeEventListener(type: string, fn: (ev?: unknown) => void) {
    this.listeners.set(type, (this.listeners.get(type) ?? []).filter((f) => f !== fn));
  }
  trigger(type: string) {
    for (const fn of this.listeners.get(type) ?? []) fn();
  }
  get textContent(): string {
    return this.children.map((c) => (typeof c === "string" ? c : c.textContent)).join("");
  }
  set textContent(v: string) {
    this.children = [v];
  }
  all(pred: (e: FakeEl) => boolean): FakeEl[] {
    const out: FakeEl[] = pred(this) ? [this] : [];
    for (const c of this.children) if (typeof c !== "string") out.push(...c.all(pred));
    return out;
  }
}

test("mounted on the plate's footfall ratio, the scale offers 'own range' and 'fixed', never 'history' or 'this step'", async () => {
  const h = harness();
  assert.ok(await h.effects.open(PLATE), `plate did not load: ${h.store.get().artifacts.model?.error}`);
  const list = (d: string) => readdirSync(d);
  h.store.dispatch({ type: "userOpened", path: PLATE_USER });
  h.store.dispatch(userEventOf(PLATE_USER, readUserSet(H5, PLATE_USER, PLATE, null, list)));
  const fetch: UserResultFetch = async (req) => readUserResultColumn(H5, req.path, PLATE, null, list, req.name, req.component);
  const feed = new UserResultFeed(h.store, fetch);
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  h.store.dispatch({ type: "setContourField", field: "footfall", component: RATIO, source: "user" });
  await feed.check();
  const v = contourView(h.store.get());
  assert.equal(v.state, "user", JSON.stringify(v));
  if (v.state !== "user") return;
  assert.equal(h.store.get().contour.range, "history", "the default range mode stands under the user result");

  const els = new Map(["legend", "display", "deform", "viewport-scale", "contour"].map((id) => [id, new FakeEl(id === "viewport-scale" ? "div" : "aside")]));
  const prevDoc = (globalThis as { document?: unknown }).document;
  (globalThis as { document?: unknown }).document = {
    title: "",
    documentElement: { dataset: { theme: "dark" } },
    createElement: (t: string) => new FakeEl(t),
    getElementById: (id: string) => els.get(id) ?? null,
  };
  let unmount: (() => void) | null = null;
  try {
    unmount = mountLegend(h.store);
    const scale = els.get("viewport-scale")!;
    const buttons = () => scale.all((e) => e.className.includes("contour-modes"))[0]!.all((e) => e.tagName === "BUTTON");
    const shown = () => buttons().map((b) => [b.textContent, b.className === "on"]);
    assert.equal(scale.hidden, false, "the user result's scale is shown");
    assert.deepEqual(shown(), [["own range", true], ["fixed", false]]);
    assert.match(buttons()[0]!.title, /one state/);
    // Clicking the own range while it is on changes nothing.
    buttons()[0]!.trigger("click");
    assert.equal(h.store.get().contour.range, "history");
    // "fixed" starts from the range shown: the column's own min and max.
    buttons()[1]!.trigger("click");
    assert.deepEqual([h.store.get().contour.range, h.store.get().contour.fixed], ["fixed", { min: v.clim.min, max: v.clim.max }]);
    assert.deepEqual(shown(), [["own range", false], ["fixed", true]]);
    // And the own range is the way back from a fixed range.
    buttons()[0]!.trigger("click");
    assert.equal(h.store.get().contour.range, "history");
    assert.deepEqual(shown(), [["own range", true], ["fixed", false]]);
  } finally {
    unmount?.();
    (globalThis as { document?: unknown }).document = prevDoc;
    h.effects.dispose();
  }
});
