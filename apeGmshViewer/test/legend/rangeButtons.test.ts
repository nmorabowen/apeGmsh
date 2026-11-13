// #2902 item 2: the colour-range buttons on the scale. A user result has one
// state (VU16: no history, no step), so "history" and "this step" both showed
// its own range under two names. The oracle is the user result's own read: its
// range is the column's min and max (test/state/userResult.test.ts holds that
// against W3's golden), and the scale offers it once, as "own range", beside
// "fixed". A User Combination (one state) and a difference (the step shown)
// range over one set of values too, so they offer it once as well. A Results
// field keeps RANGE_MODES as they are.

import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { mountLegend } from "../../src/panels/legend.ts";
import { readUserResultColumn, readUserSet, type UserH5 } from "../../src/reader/userFile.ts";
import { UserResultFeed, type UserResultFetch } from "../../src/renderer/contour.ts";
import { RANGE_MODES } from "../../src/state/contour.ts";
import { reduce, userEventOf } from "../../src/state/reduce.ts";
import { contourView, rangeOptionsOf, spotlightViewOf } from "../../src/state/selectors.ts";
import { harness } from "../contour/harness.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as UserH5;
const at = (name: string) => fileURLToPath(new URL(`../fixtures/vu16/${name}`, import.meta.url));
const PLATE = at("plate.h5");
const PLATE_USER = at("plate.user.h5");
const RATIO = "footfall_ratio";

test("the one range rule: a contour with history and steps offers RANGE_MODES as they are; a kind with one range offers it once, and fixed", () => {
  // VF2-d (#2956): the rule is state/contour.ts rangeOptionsOf, read through selectors.ts by the legend and the palette alike.
  // A Results field (nodal or element) scans its stage's history or reads the step shown, by the mode.
  for (const range of ["history", "step", "fixed"] as const) {
    assert.deepEqual(
      rangeOptionsOf("steps", range).map((b) => [b.range, b.own, b.label, b.on]),
      RANGE_MODES.map((m) => [m.range, false, m.label, m.range === range]),
      `steps at ${range}`,
    );
  }
  // A user result and a User Combination: one state, so their own range whatever the mode (selectors.ts
  // userResultContour, combinationClim). A difference: the step shown whatever the mode (differenceContour).
  const own = { user: ["own range", "history"], combination: ["own range", "history"], difference: ["this step", "step"] } as const;
  for (const [kind, [label, back]] of Object.entries(own) as [keyof typeof own, readonly [string, string]][]) {
    for (const range of ["history", "step"] as const) {
      assert.deepEqual(rangeOptionsOf(kind, range).map((b) => [b.range, b.own, b.label, b.on]), [[back, true, label, true], ["fixed", false, "fixed", false]], `${kind} at ${range}`);
    }
    assert.deepEqual(rangeOptionsOf(kind, "fixed").map((b) => [b.range, b.own, b.label, b.on]), [[back, true, label, false], ["fixed", false, "fixed", true]], `${kind} at fixed`);
    assert.ok(!rangeOptionsOf(kind, "history").some((b) => b.label === "history"), `${kind}: no history button`);
  }
  assert.ok(!rangeOptionsOf("user", "history").some((b) => /step/.test(b.label)), "a user result has no step either");
  assert.ok(!rangeOptionsOf("combination", "history").some((b) => /step/.test(b.label)), "nor has a combination");
  assert.throws(() => rangeOptionsOf("history" as never, "history"), /unknown contour range kind "history"/);
});

test("VF2-d (#2956): the legend reads the one rule through state/selectors.ts and keeps no copy of it", () => {
  // VF2-b (#2953) left two copies held equal by a test; the legend's scale must now ask the rule the palette asks
  // (state/spotlight.ts), and the fixed range's start (`fixedRangeFrom`), so the two cannot drift apart again.
  const src = readFileSync(fileURLToPath(new URL("../../src/panels/legend.ts", import.meta.url)), "utf8");
  const imports = /import \{([^}]*)\} from "\.\.\/state\/selectors\.ts";/.exec(src)?.[1] ?? "";
  for (const name of ["rangeKindOf", "rangeOptionsOf", "fixedRangeFrom"]) assert.match(imports, new RegExp(`\\b${name}\\b`), `legend.ts imports ${name} from selectors.ts`);
  assert.match(src, /rangeOptionsOf\(rangeKindOf\(s\), s\.contour\.range\)/, "the scale's buttons are the rule's options at the state's kind");
  assert.doesNotMatch(src, /RANGE_MODES|label: "own range"|function rangeButtonsOf/, "no copy of the rule's modes or labels");
  assert.doesNotMatch(src, /c\.clim\.min < c\.clim\.max/, "no copy of the fixed range's start");
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
    // "fixed" starts from the range shown: the column's own min and max. VF2-d (#2956): the palette's "fixed",
    // offered with no stored range, picked from the same state, lands on the same range (selectors `fixedRangeFrom`).
    const before = h.store.get();
    assert.equal(before.contour.fixed, null, "precondition: no fixed range is stored");
    const palette = spotlightViewOf(reduce(before, { type: "openSpotlight", mode: "contourRange", query: "" })).items.find((i) => i.title === "Range: fixed");
    assert.ok(palette, "the palette offers fixed with no stored range");
    const picked = palette.action.events.reduce(reduce, before);
    buttons()[1]!.trigger("click");
    assert.deepEqual([h.store.get().contour.range, h.store.get().contour.fixed], ["fixed", { min: v.clim.min, max: v.clim.max }]);
    assert.deepEqual([picked.contour.range, picked.contour.fixed], [h.store.get().contour.range, h.store.get().contour.fixed], "the palette's pick is the legend's click");
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
