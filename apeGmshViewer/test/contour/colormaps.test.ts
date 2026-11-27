// Custom colormaps, levels and the colormap control (panels/legend/colormaps.ts).
// The oracles are closed forms: a banded ramp shows, in band k of N, the map's
// colour at k / (N - 1); a custom map is linear between its own stops; an id
// round-trips its name and stops. The control is driven through a stand-in
// DOM and a stand-in library storage.

import assert from "node:assert/strict";
import { test } from "node:test";
import { mountColormaps, parseLibrary, librarySave, reversedStops, withStopAdded, toHex, fromHex, LIBRARY_KEY } from "../../src/panels/legend/colormaps.ts";
import { contourLines, tickFractions } from "../../src/panels/legend.ts";
import { colormapTexture, MAX_TEXTURES, rampTexels, RAMP_TEXELS } from "../../src/renderer/colormapTexture.ts";
import {
  bandOf,
  CMAPS,
  customCmapId,
  customCmapProblem,
  isContourCmap,
  isLevels,
  MAX_LEVELS,
  parseCustomCmap,
  rampCss,
  rampParts,
  sampleCmap,
  withLevels,
  type Stops,
} from "../../src/state/contour.ts";
import { restoreView, viewMemoOf, viewShotOf } from "../../src/spec/snapshot.ts";
import { contourView } from "../../src/state/selectors.ts";
import { Store } from "../../src/state/store.ts";
import { opened, until } from "./harness.ts";

const close = (a: readonly number[], b: readonly number[], tol = 1e-12) => a.every((x, i) => Math.abs(x - b[i]!) <= tol);
const GREYS: Stops = [
  [0, 0, 0, 0],
  [0.25, 1, 0, 0],
  [1, 1, 1, 1],
];

test("a custom map's id round-trips its name and stops, and is linear between them", () => {
  const id = customCmapId("greys", GREYS);
  assert.ok(id.startsWith("custom:"));
  assert.deepEqual(parseCustomCmap(id), { name: "greys", stops: GREYS });
  assert.ok(isContourCmap(id) && isContourCmap("jet"));
  assert.ok(!isContourCmap("custom:{}") && !isContourCmap("jett") && !isContourCmap("custom:nope"));
  assert.deepEqual(sampleCmap(id, 0), [0, 0, 0]);
  assert.deepEqual(sampleCmap(id, 0.25), [1, 0, 0]);
  assert.ok(close(sampleCmap(id, 0.125), [0.5, 0, 0]));
  assert.ok(close(sampleCmap(id, 0.625), [1, 0.5, 0.5]));
  assert.deepEqual(sampleCmap(id, 2), [1, 1, 1], "clamped above 1");
  // Positions and colours are kept to four decimals, so the id is short and stable.
  assert.deepEqual(parseCustomCmap(customCmapId("x", [[0, 0.123456, 0, 0], [1, 1, 1, 1]]))!.stops[0], [0, 0.1235, 0, 0]);
});

test("a custom map is refused when it is not 2..MAX_STOPS stops from 0 to 1 in order, or its name is not a name", () => {
  const ok: Stops = [[0, 0, 0, 0], [1, 1, 1, 1]];
  assert.equal(customCmapProblem("a", ok), null);
  assert.match(customCmapProblem("", ok)!, /name/);
  assert.match(customCmapProblem(" a", ok)!, /name/);
  assert.match(customCmapProblem("a|b", ok)!, /\|/);
  assert.match(customCmapProblem("a", [[0, 0, 0, 0]])!, /2 to/);
  assert.match(customCmapProblem("a", [[0, 0, 0, 0], [0.5, 1, 1, 1]])!, /first stop sits at 0 and the last at 1/);
  assert.match(customCmapProblem("a", [[0, 0, 0, 0], [0.6, 0, 0, 0], [0.4, 0, 0, 0], [1, 1, 1, 1]])!, /stop 3 sits before stop 2/);
  assert.match(customCmapProblem("a", [[0, 0, 0, 0], [1, 1.5, 1, 1]])!, /stop 2/);
  assert.match(customCmapProblem("a", [[0, 0, 0, 0], [1, Number.NaN, 1, 1]])!, /stop 2/);
  assert.throws(() => customCmapId("a", [[0, 0, 0, 0]]), RangeError);
});

test("N levels: band k of N shows the map at k / (N - 1), the first band its first colour and the last its last", () => {
  for (const n of [2, 3, 5, 12]) {
    const ramp = withLevels("viridis", n);
    assert.deepEqual(rampParts(ramp), { cmap: "viridis", levels: n });
    for (let k = 0; k < n; k++) {
      const want = sampleCmap("viridis", k / (n - 1));
      for (const f of [0.001, 0.5, 0.999]) assert.deepEqual(sampleCmap(ramp, (k + f) / n), want, `band ${k} of ${n} at ${f}`);
    }
    assert.deepEqual(sampleCmap(ramp, 0), sampleCmap("viridis", 0));
    assert.deepEqual(sampleCmap(ramp, 1), sampleCmap("viridis", 1), "1 is in the last band");
  }
  assert.equal(withLevels("jet", 0), "jet", "0 levels is the map itself");
  assert.equal(bandOf(0.25, 4), 1, "a band edge belongs to the band above");
  const custom = customCmapId("greys", GREYS);
  assert.deepEqual(sampleCmap(withLevels(custom, 2), 0.4), [0, 0, 0]);
  assert.deepEqual(sampleCmap(withLevels(custom, 2), 0.6), [1, 1, 1]);
  for (const bad of [1, -1, 2.5, MAX_LEVELS + 1, Number.NaN]) {
    assert.ok(!isLevels(bad), String(bad));
    assert.throws(() => withLevels("jet", bad), RangeError);
  }
  assert.ok(isLevels(0) && isLevels(2) && isLevels(MAX_LEVELS));
  assert.throws(() => rampParts("jet|L1" as never), RangeError);
  assert.throws(() => rampParts("nope|L4" as never), RangeError);
});

test("the texture and the scale bar of a banded ramp hold exactly N colours, with hard edges", () => {
  const n = 7;
  const texels = rampTexels(withLevels("turbo", n));
  const colours = new Set<string>();
  for (let i = 0; i < RAMP_TEXELS; i++) colours.add(Array.from(texels.subarray(4 * i, 4 * i + 3)).join(","));
  assert.equal(colours.size, n);
  const css = rampCss(withLevels("turbo", n));
  // Each band is one colour at its two edges: 2N colour stops, each band's pair equal.
  const stops = [...css.matchAll(/rgb\(([^)]*)\) ([\d.]+)%/g)].map((m) => [m[1], Number(m[2])] as const);
  assert.equal(stops.length, 2 * n);
  for (let k = 0; k < n; k++) {
    assert.equal(stops[2 * k]![0], stops[2 * k + 1]![0], `band ${k} is one colour`);
    assert.ok(Math.abs(stops[2 * k]![1] - (100 * k) / n) < 0.01 && Math.abs(stops[2 * k + 1]![1] - (100 * (k + 1)) / n) < 0.01);
  }
  // A continuous ramp's bar still goes through the map's own stops.
  assert.equal([...rampCss("bwr").matchAll(/rgb\(/g)].length, CMAPS.bwr.length);
});

test("the textures kept are bounded: editing a map does not pile them up", () => {
  const made = Array.from({ length: MAX_TEXTURES + 3 }, (_, i) => colormapTexture(customCmapId(`m${i}`, [[0, 0, 0, i / 100], [1, 1, 1, 1]])));
  let disposed = 0;
  for (const t of made) t.addEventListener("dispose", () => disposed++);
  // Three more push the three oldest of the ones above out.
  for (let i = 0; i < 3; i++) colormapTexture(customCmapId(`n${i}`, [[0, 0, 0, 0], [1, 1, 1, i / 10]]));
  assert.equal(disposed, 3);
  const again = colormapTexture(customCmapId(`m${MAX_TEXTURES + 2}`, [[0, 0, 0, (MAX_TEXTURES + 2) / 100], [1, 1, 1, 1]]));
  assert.equal(again, made[MAX_TEXTURES + 2], "a held ramp is reused");
});

test("the reducer takes a custom map and the levels, and refuses what is neither", () => {
  const store = new Store();
  const id = customCmapId("greys", GREYS);
  store.dispatch({ type: "setContourCmap", cmap: id });
  assert.equal(store.get().contour.cmap, id);
  assert.throws(() => store.dispatch({ type: "setContourCmap", cmap: "custom:{\"name\":\"x\"}" as never }), RangeError);
  assert.equal(store.get().contour.levels, 0, "continuous by default");
  store.dispatch({ type: "setContourLevels", levels: 10 });
  assert.equal(store.get().contour.levels, 10);
  for (const bad of [1, 65, 2.5, -3]) assert.throws(() => store.dispatch({ type: "setContourLevels", levels: bad }), RangeError);
  assert.equal(store.get().contour.levels, 10);
});

test("the contour view draws the levels, and the scale ticks every band edge while they fit", async () => {
  const h = await opened();
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 2, step: 3 } });
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  h.store.dispatch({ type: "setContourRange", range: "fixed", fixed: { min: 0, max: 10 } });
  await until(() => contourView(h.store.get()).state === "ready", "the contour");
  const ticks = () => {
    const v = contourView(h.store.get());
    assert.ok(v.state === "ready");
    return { cmap: v.cmap, ticks: contourLines(h.store.get(), v).ticks };
  };
  assert.deepEqual(ticks(), { cmap: "jet", ticks: ["0", "2.5", "5", "7.5", "10"] });
  h.store.dispatch({ type: "setContourLevels", levels: 4 });
  assert.deepEqual(ticks(), { cmap: "jet|L4", ticks: ["0", "2.5", "5", "7.5", "10"] });
  h.store.dispatch({ type: "setContourLevels", levels: 5 });
  assert.deepEqual(ticks(), { cmap: "jet|L5", ticks: ["0", "2", "4", "6", "8", "10"] });
  h.store.dispatch({ type: "setContourLevels", levels: 20 });
  assert.deepEqual(ticks().ticks, ["0", "2", "4", "6", "8", "10"], "20 bands: every fourth edge");
  h.store.dispatch({ type: "setContourLevels", levels: 13 });
  assert.deepEqual(ticks().ticks, ["0", "2.5", "5", "7.5", "10"], "13 bands, which nothing divides: the quarters");
  const id = customCmapId("greys", GREYS);
  h.store.dispatch({ type: "setContourCmap", cmap: id });
  assert.equal(ticks().cmap, `${id}|L13`);
  // A render spec names a built-in map and draws it continuous: a snapshot of this view says what it leaves out.
  const pose = { eye: [1, 1, 1], target: [0, 0, 0], up: [0, 0, 1] } as const;
  const { shot, outside } = viewShotOf(h.store.get(), pose as never);
  assert.equal(shot.contour?.cmap, "jet");
  assert.ok(outside.includes("the contour colormap is the custom greys; the spec names jet"), outside.join("; "));
  assert.ok(outside.includes("the contour is drawn in 13 levels"));
  // Back (an offer's memo) puts the levels and the custom map back as they were.
  const memo = viewMemoOf(h.store.get(), null, "m.h5");
  h.store.dispatch({ type: "setContourLevels", levels: 0 });
  h.store.dispatch({ type: "setContourCmap", cmap: "jet" });
  restoreView(h.store, memo);
  assert.deepEqual([h.store.get().contour.cmap, h.store.get().contour.levels], [id, 13]);
  h.effects.dispose();
});

test("the scale ticks band edges, at most nine of them, and never between edges", () => {
  assert.deepEqual(tickFractions(0), [0, 0.25, 0.5, 0.75, 1]);
  assert.deepEqual(tickFractions(8).length, 9);
  assert.deepEqual(tickFractions(12), [0, 2, 4, 6, 8, 10, 12].map((k) => k / 12));
  assert.deepEqual(tickFractions(9), [0, 3, 6, 9].map((k) => k / 9));
  for (let n = 2; n <= MAX_LEVELS; n++) {
    const f = tickFractions(n);
    assert.ok(f.length <= 9, `${n}`);
    const onEdges = f.every((x) => Math.abs(x * n - Math.round(x * n)) < 1e-9);
    assert.ok(onEdges || f.join() === "0,0.25,0.5,0.75,1", `${n}: band edges, or the quarters`);
  }
});

test("the library keeps valid maps once by name, and the editor's helpers keep a map valid", () => {
  const ok = { name: "a", stops: GREYS };
  const raw = JSON.stringify([ok, { name: "a", stops: [[0, 0, 0, 0], [1, 0, 0, 0]] }, { name: "bad", stops: [[0, 0, 0, 0]] }, 5, null]);
  assert.deepEqual(parseLibrary(raw), [ok]);
  assert.deepEqual(parseLibrary("not json"), []);
  assert.deepEqual(parseLibrary(null), []);
  assert.deepEqual(parseLibrary("{}"), []);
  const b = { name: "b", stops: reversedStops(GREYS) };
  assert.deepEqual(librarySave([ok], b), [ok, b]);
  assert.deepEqual(librarySave([ok, b], { name: "a", stops: b.stops }), [{ name: "a", stops: b.stops }, b]);
  assert.deepEqual(reversedStops(GREYS), [[0, 1, 1, 1], [0.75, 1, 0, 0], [1, 0, 0, 0]]);
  assert.equal(customCmapProblem("r", reversedStops(GREYS)), null);
  const added = withStopAdded(GREYS);
  assert.equal(added.length, 4);
  assert.deepEqual(added[2], [0.625, 1, 0.5, 0.5], "in the widest gap, in the colour the map has there");
  assert.equal(customCmapProblem("r", added), null);
  assert.equal(toHex(1, 0.5, 0), "#ff8000");
  assert.ok(close(fromHex("#ff8000")!, [1, 128 / 255, 0]));
  assert.equal(fromHex("red"), null);
});

// -- the control, through a stand-in DOM ------------------------------------------------

class El {
  className = "";
  title = "";
  value = "";
  type = "";
  step = "";
  min = "";
  max = "";
  maxLength = 0;
  hidden = false;
  disabled = false;
  style: Record<string, string> = {};
  children: (El | string)[] = [];
  listeners = new Map<string, (() => void)[]>();
  validity = "";
  classList = {
    toggle: (c: string, on: boolean) => {
      const set = new Set(this.className.split(" ").filter(Boolean));
      if (on) set.add(c);
      else set.delete(c);
      this.className = [...set].join(" ");
    },
  };
  readonly tag: string;
  constructor(tag: string) {
    this.tag = tag;
  }
  append(...xs: (El | string)[]) {
    this.children.push(...xs);
  }
  replaceChildren(...xs: (El | string)[]) {
    this.children = [...xs];
  }
  setCustomValidity(v: string) {
    this.validity = v;
  }
  addEventListener(t: string, fn: () => void) {
    this.listeners.set(t, [...(this.listeners.get(t) ?? []), fn]);
  }
  removeEventListener(t: string, fn: () => void) {
    this.listeners.set(t, (this.listeners.get(t) ?? []).filter((f) => f !== fn));
  }
  set textContent(v: string) {
    this.children = [v];
  }
  get text(): string {
    return this.children.map((c) => (typeof c === "string" ? c : c.text)).join("");
  }
  fire(t: string) {
    for (const fn of this.listeners.get(t) ?? []) fn();
  }
  all(cls: string): El[] {
    const mine = this.className.split(" ").includes(cls) && !this.hidden ? [this] : [];
    return this.hidden ? [] : [...mine, ...this.children.flatMap((c) => (typeof c === "string" ? [] : c.all(cls)))];
  }
  one(cls: string): El {
    const f = this.all(cls);
    assert.equal(f.length, 1, cls);
    return f[0]!;
  }
}

function control() {
  (globalThis as { document?: unknown }).document = { createElement: (t: string) => new El(t), activeElement: null };
  let raw: string | null = null;
  const writes: string[] = [];
  const store = new Store();
  const c = mountColormaps(store, { read: () => raw, write: (v) => writes.push((raw = v)) });
  return { store, root: c.root as unknown as El, writes, dispose: c.dispose, setRaw: (v: string) => (raw = v) };
}

test("the control: each map is offered as its ramp, a click picks it, and the levels are typed in", () => {
  const { store, root, dispose } = control();
  try {
    const current = root.one("cmap-current");
    assert.equal(current.one("cmap-name").text, "jet");
    assert.equal(current.one("cmap-swatch").style["backgroundImage"], rampCss("jet"));
    assert.equal(root.all("cmap-option").length, 0, "the menu is closed");
    current.fire("click");
    const options = root.all("cmap-option");
    assert.equal(options.length, 13, "the built-in maps; none saved yet");
    const plasma = options.find((o) => o.one("cmap-name").text === "plasma")!;
    assert.equal(plasma.one("cmap-swatch").style["backgroundImage"], rampCss("plasma"), "the option shows its ramp");
    plasma.one("cmap-pick").fire("click");
    assert.equal(store.get().contour.cmap, "plasma");
    assert.equal(root.all("cmap-option").length, 0, "a pick closes the menu");
    assert.equal(current.one("cmap-name").text, "plasma");
    const levels = root.one("contour-levels");
    levels.value = "6";
    levels.fire("change");
    assert.equal(store.get().contour.levels, 6);
    assert.equal(current.one("cmap-swatch").style["backgroundImage"], rampCss(withLevels("plasma", 6)), "the swatch shows the bands");
    levels.value = "1";
    levels.fire("change");
    assert.equal(store.get().contour.levels, 0, "one band is continuous");
    levels.value = "99";
    levels.fire("change");
    assert.equal(store.get().contour.levels, 0);
    assert.match(levels.validity, /2 to 64/, "out of range is refused, and says why");
  } finally {
    dispose();
  }
});

test("the control's editor draws the map as it is edited, saves it to the library, and deletes it", () => {
  const { store, root, writes, dispose } = control();
  try {
    store.dispatch({ type: "setContourCmap", cmap: "bwr" });
    root.one("cmap-current").fire("click");
    root.one("cmap-new").fire("click");
    const editor = root.one("cmap-editor");
    assert.equal(editor.one("cmap-name-input").value, "bwr (edit)");
    const shown = () => parseCustomCmap(store.get().contour.cmap);
    assert.deepEqual(shown(), { name: "bwr (edit)", stops: CMAPS.bwr }, "the copy is drawn at once");
    // The middle stop's colour.
    const mid = root.all("cmap-stop")[1]!;
    mid.one("cmap-stop-colour").value = "#00ff00";
    mid.one("cmap-stop-colour").fire("change");
    assert.deepEqual(shown()!.stops[1], [0.5, 0, 1, 0]);
    // The ends cannot move or go.
    const first = root.all("cmap-stop")[0]!;
    assert.ok(first.one("cmap-stop-at").disabled && first.one("cmap-stop-drop").disabled);
    root.one("cmap-reverse").fire("click");
    assert.deepEqual(shown()!.stops.map((s) => s[0]), [0, 0.5, 1]);
    assert.deepEqual(shown()!.stops[0], [0, 1, 0, 0], "bwr ends in red: reversed, it starts there");
    // A stop moved past another is put back in order.
    const m = root.all("cmap-stop")[1]!.one("cmap-stop-at");
    m.value = "0.9";
    m.fire("change");
    assert.deepEqual(shown()!.stops.map((s) => s[0]), [0, 0.9, 1]);
    const name = root.one("cmap-name-input");
    name.value = "mine";
    name.fire("change");
    assert.equal(shown()!.name, "mine");
    assert.equal(writes.length, 0, "nothing is stored until saved");
    root.one("cmap-save").fire("click");
    assert.deepEqual(parseLibrary(writes.at(-1)!), [{ name: "mine", stops: shown()!.stops }]);
    root.one("cmap-close").fire("click");
    assert.equal(root.all("cmap-editor").length, 0);
    assert.equal(shown()!.name, "mine", "closing keeps the map on the contour");
    // The saved map is offered under Yours, with its ramp, and can be edited again and deleted.
    root.one("cmap-current").fire("click");
    const yours = root.all("cmap-option").filter((o) => o.all("cmap-edit").length === 1);
    assert.equal(yours.length, 1);
    assert.equal(yours[0]!.one("cmap-swatch").style["backgroundImage"], rampCss(store.get().contour.cmap));
    yours[0]!.one("cmap-edit").fire("click");
    root.one("cmap-delete").fire("click");
    assert.deepEqual(parseLibrary(writes.at(-1)!), []);
    assert.equal(shown()!.name, "mine", "the contour keeps the map until another is picked");
  } finally {
    dispose();
  }
});

test("the control: an invalid draft is said, never drawn", () => {
  const { store, root, dispose } = control();
  try {
    root.one("cmap-current").fire("click");
    root.one("cmap-new").fire("click");
    const before = store.get().contour.cmap;
    const name = root.one("cmap-name-input");
    name.value = "a|b";
    name.fire("change");
    assert.equal(store.get().contour.cmap, before, "the last valid map stays drawn");
    assert.match(root.one("cmap-problem").text, /Not drawn: .*\|/);
    assert.ok(root.one("cmap-save").disabled);
  } finally {
    dispose();
  }
});

test("the library is read from this profile's storage at mount", () => {
  (globalThis as { document?: unknown }).document = { createElement: (t: string) => new El(t), activeElement: null };
  const store = new Store();
  const raw = JSON.stringify([{ name: "greys", stops: GREYS }]);
  const c = mountColormaps(store, { read: () => raw, write: () => undefined });
  try {
    const root = c.root as unknown as El;
    root.one("cmap-current").fire("click");
    const greys = root.all("cmap-option").find((o) => o.one("cmap-name").text === "greys")!;
    greys.one("cmap-pick").fire("click");
    assert.equal(store.get().contour.cmap, customCmapId("greys", GREYS));
    assert.equal(LIBRARY_KEY, "apegmsh.viewer.colormaps");
  } finally {
    c.dispose();
  }
});
