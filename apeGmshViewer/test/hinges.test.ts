// The Ladruno beams' moment hinges (reader/ladruno.ts `readHingeStep`,
// `readHingeHistory`; state/hinges.ts; renderer/hinges.ts; panels/hinges.ts).
//
// The fixtures are real fork runs (fixtures/make_hinge_fixtures.py). Three beams
// and an elastic one meet at a tip pushed through their hinges' capacity. The
// oracle is OpenSees itself: `eleResponse(tag, "hingeSummary")` at every step of
// the run, and for the Steel01 hinge the bilinear law in closed form.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { ELEMENT_SYNTAX } from "../src/chain/signatures.ts";
import { hingeCardKeys, renderHinges } from "../src/panels/hinges.ts";
import { HINGE_COLUMNS, hingeColumnsOf, openLadruno } from "../src/reader/ladruno.ts";
import type { ResultsH5 } from "../src/reader/results.ts";
import { HingesLayer } from "../src/renderer/hinges.ts";
import { buildHinges3D, HINGE_COLUMN_NAMES, HINGE_NEUTRAL, HINGE_NO_DATA, hingeCapacityOf, hingeEndsOf, hingeSitesOf } from "../src/state/hinges.ts";
import { hingeJoinOf, hingeViewOf } from "../src/state/selectors.ts";
import { spotlightViewOf } from "../src/state/spotlight.ts";
import { reduce } from "../src/state/store.ts";
import { opened, until } from "./contour/harness.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as ResultsH5;
const fixture = (name: string) => fileURLToPath(new URL(`../fixtures/${name}`, import.meta.url));

interface Oracle {
  elements: Record<string, number>;
  femIds: Record<string, number>;
  steps: number;
  hingeSummary: Record<string, number[][]>;
  steel: { fy: number; k: number; b: number };
}
const oracle = (dim: string) => JSON.parse(readFileSync(fixture(`ladruno_hinges${dim}.oracle.json`), "utf8")) as Oracle;
const O3 = oracle("3d");
const O2 = oracle("2d");
const L = 4;

const close = (a: number, b: number, what: string, tol = 1e-12) => assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${what}: ${a}, expected ${b}`);

const h = await opened(fixture("ladruno_hinges3d.h5"), fixture("ladruno_hinges3d.ladruno"));
after(() => {
  h.effects.dispose();
  h.host.close();
});

async function atStep(step: number) {
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step } });
  const here = () => {
    const d = h.effects.currentStep();
    return d && d.stage === 0 && d.step === step && (!h.store.get().hinges.on || d.hinges !== undefined) ? d : null;
  };
  await until(() => here() !== null, `step ${step}`);
  return here()!;
}

for (const [dim, o, width] of [["3d", O3, 8], ["2d", O2, 4]] as const) {
  test(`${dim}: every step of hingeSummary reads as OpenSees reported it, by column name`, () => {
    const f = openLadruno(H5, fixture(`ladruno_hinges${dim}.ladruno`));
    const stage = f.index.stages[0]!.id;
    assert.equal(f.hingeCount(stage), Object.keys(o.hingeSummary).length);
    for (let k = 0; k < o.steps; k++) {
      const step = f.readHingeStep(stage, k);
      assert.equal(step.elementIds.length, Object.keys(o.hingeSummary).length);
      step.elementIds.forEach((id, r) => {
        const live = o.hingeSummary[String(id)]!;
        for (let c = 0; c < width; c++) close(step.values[r * 8 + c]!, live[k]![c]!, `element ${id} step ${k} ${HINGE_COLUMNS[c]}`);
        for (let c = width; c < 8; c++) assert.ok(Number.isNaN(step.values[r * 8 + c]!), "a 2-D file has no weak-axis column");
      });
    }
  });

  test(`${dim}: one element's history is its column of every step`, () => {
    const f = openLadruno(H5, fixture(`ladruno_hinges${dim}.ladruno`));
    const stage = f.index.stages[0]!.id;
    for (const [id, live] of Object.entries(o.hingeSummary)) {
      const hist = f.readHingeHistory(stage, Number(id));
      assert.equal(hist.time.length, o.steps);
      live.forEach((row, k) => row.forEach((v, c) => close(hist.values[k * 8 + c]!, v, `element ${id} step ${k} column ${c}`)));
    }
    assert.throws(() => f.readHingeHistory(stage, 99999), /records no hingeSummary/);
  });

  test(`${dim}: the Steel01 hinge follows its bilinear law`, () => {
    const { fy, k, b } = o.steel;
    const yieldRotation = fy / k;
    for (const row of o.hingeSummary[String(o.elements["imk"])]!) {
      const theta = Math.abs(row[0]!);
      const want = theta <= yieldRotation ? k * theta : fy + b * k * (theta - yieldRotation);
      close(Math.abs(row[1]!), want, `|Mz_i| at rotation ${theta}`, 1e-6);
    }
  });
}

test("a stage's colour range is the largest rotation and moment over all its steps", () => {
  const f = openLadruno(H5, fixture("ladruno_hinges3d.ladruno"));
  const rows = Object.values(O3.hingeSummary).flat();
  const top = (odd: number) => Math.max(...rows.flatMap((r) => r.filter((_, c) => c % 2 === odd).map(Math.abs)));
  const { range } = f.readHingeStep(f.index.stages[0]!.id, 0);
  close(range!.rotation, top(0), "rotation");
  close(range!.moment, top(1), "moment");
});

test("hinge columns are taken by name: another column set is refused, not read by position", () => {
  assert.deepEqual([...hingeColumnsOf(["theta_i", "M_i", "theta_j", "M_j"])!], [0, 1, 2, 3, -1, -1, -1, -1]);
  assert.deepEqual([...hingeColumnsOf(["Mz_i", "thetaZ_i", "Mz_j", "thetaZ_j"])!].slice(0, 4), [1, 0, 3, 2]);
  assert.equal(hingeColumnsOf(["a", "b", "c", "d"]), null);
  assert.equal(hingeColumnsOf(["thetaZ_i", "Mz_i"]), null);
  assert.deepEqual([...HINGE_COLUMNS], [...HINGE_COLUMN_NAMES], "the reader and the state agree on the order");
});

test("the hinge arguments decode as the fork's parsers read them", () => {
  const ends = (type: string, args: (number | string)[]) => hingeEndsOf(type, args).map((e) => [e.end, e.xi, e.offset, e.z?.family[0], e.z?.tag, e.y?.family[0], e.y?.tag, e.biaxial]);
  const F = "LadrunoForceBeamColumn", D = "LadrunoDispBeamColumn", I = "LadrunoIMKBeam";
  const imk = [0.04, 2e11, 1.6e-4, 1];
  assert.deepEqual(ends(F, [1, 1]), [], "no flag, no hinge");
  assert.deepEqual(ends(F, [1, 1, "-hinge", 5]), [["i", 0, 0, "u", 5, undefined, undefined, false]]);
  assert.deepEqual(ends(D, [1, 1, "-hinge", 5, 6]).map((e) => e.slice(0, 2).concat(e[4])), [["i", 0, 5], ["j", 1, 6]]);
  assert.deepEqual(ends(F, [1, 1, "-hingeJ", 5]).map((e) => e.slice(0, 2)), [["j", 1]]);
  assert.deepEqual(ends(F, [1, 1, "-hingeIJ", 5, "-hingeLoc", 0.2]).map((e) => [e[0], e[4]]), [["i", 5], ["j", 5]], "one tag serves both ends");
  const loc = hingeEndsOf(F, [1, 1, "-hingeIJ", 5, "-hingeLoc", 0.2]);
  close(loc[1]!.xi!, 0.8, "one -hingeLoc value mirrors to the other end");
  assert.deepEqual(ends(F, [1, 1, "-hingeI", 5, "-hingeYI", 6]), [["i", 0, 0, "u", 5, "u", 6, false]]);
  assert.deepEqual(ends(D, [1, 1, "-hingeBiaxialI", 3]), [["i", 0, 0, "n", 3, "n", 3, true]]);
  assert.deepEqual(ends(I, [...imk, "-matZ", 2]).map((e) => e[0]), ["i", "j"], "both ends by default");
  assert.deepEqual(ends(I, [...imk, "-hinge", "j", "-matZ", 2]).map((e) => e[0]), ["j"]);
  assert.deepEqual(ends(I, [...imk, "-hinge", "none", "-matZ", 2]), []);
  assert.deepEqual(ends(I, [...imk, "-hinge", "none", "-matZi", 2]).map((e) => e[0]), ["i"], "naming an end hinges it whatever -hinge says");
  assert.deepEqual(ends(I, [...imk, "-hingeBiaxial", 3, 4]).map((e) => [e[0], e[4]]), [["i", 3], ["j", 4]]);
  assert.deepEqual(ends(I, [...imk, "-hinge", "i", "-hingeBiaxial", 3]).map((e) => [e[0], e[3]]), [["i", "n"]]);
  assert.deepEqual(ends(I, [...imk, "-matZ", 2, "-endZones", 0.2, 0.4]).map((e) => [e[1], e[2]]), [[null, 0.2], [null, 0.4]]);
  assert.throws(() => hingeEndsOf(I, [...imk, "-hinge", "k"]), /not both, i, j or none/);
});

test("the model says which hinges exist and where, and an unhinged beam has none", () => {
  const s = h.store.get();
  const { sites, refused } = hingeSitesOf(s.decls);
  assert.deepEqual(refused, []);
  const row = (name: string) => sites.filter((x) => x.tag === O3.elements[name]).map((x) => [x.end, x.xi, x.offset, x.z !== null, x.y !== null, x.biaxial]);
  assert.deepEqual(row("imk"), [["i", null, 0.2, true, false, false], ["j", null, 0.4, true, true, false]]);
  assert.deepEqual(row("fbc"), [["i", 0.05, 0, true, false, false], ["j", 0.95, 0, true, false, false]]);
  assert.deepEqual(row("dbc"), [["i", 0.1, 0, true, true, true], ["j", 0.9, 0, true, true, true]]);
  assert.deepEqual(row("plain"), []);
});

test("the capacity comes from the hinge's material, and one the app does not read is named", () => {
  const s = h.store.get();
  const { sites } = hingeSitesOf(s.decls);
  const at = (name: string, end: string) => sites.find((x) => x.tag === O3.elements[name] && x.end === end)!;
  assert.deepEqual(hingeCapacityOf(s.decls, at("imk", "i"), "z"), { value: O3.steel.fy });
  assert.deepEqual(hingeCapacityOf(s.decls, at("fbc", "j"), "z"), { value: 50e3 });
  assert.deepEqual(hingeCapacityOf(s.decls, at("dbc", "i"), "y"), { value: 35e3 });
  assert.deepEqual(hingeCapacityOf(s.decls, at("fbc", "i"), "y"), { missing: "no My hinge at end i" });
});

test("markers sit on the chord at their hinge's position, neutral with no step", () => {
  const s = reduce(h.store.get(), { type: "setHinges", on: true });
  const g = buildHinges3D(s, h.blobs, null, null);
  assert.equal(g.count, 6);
  assert.equal(g.top, null);
  const at = (i: number) => [...g.positions.subarray(i * 3, i * 3 + 3)];
  const want = new Map<string, number>();
  hingeSitesOf(s.decls).sites.forEach((x, i) => {
    const t = x.xi ?? (x.end === "i" ? x.offset / Math.hypot(L, 1) : 1 - x.offset / Math.hypot(L, 1));
    want.set(String(i), t);
  });
  hingeSitesOf(s.decls).sites.forEach((x, i) => {
    const base = { [O3.elements["imk"]!]: -1, [O3.elements["fbc"]!]: 0, [O3.elements["dbc"]!]: 1 }[x.tag]!;
    const t = want.get(String(i))!;
    const p = at(i);
    close(p[0]!, t * L, `x of hinge ${i}`, 1e-6);
    close(p[2]!, base * (1 - t), `z of hinge ${i}`, 1e-6);
    assert.deepEqual([...g.colors.subarray(i * 3, i * 3 + 3)].map((c) => Number(c.toFixed(4))), HINGE_NEUTRAL.map((c) => Number(c.toFixed(4))));
  });
});

test("through the app: the switch reads the step, colours the markers, and plots the picked hinge against OpenSees", async () => {
  h.store.dispatch({ type: "setHinges", on: true });
  const last = O3.steps - 1;
  const step = await atStep(last);
  assert.ok(step.hinges, "the step carries hingeSummary");
  const s0 = h.store.get();
  assert.ok(!s0.results!.stages[0]!.unread.includes("ON_ELEMENTS/hingeSummary"), "read now, so not listed as unread");
  const join = hingeJoinOf(s0);
  assert.ok(join.ok, "the stamped file joins the model");
  const layer = new HingesLayer();
  assert.equal(layer.sync(s0, h.blobs, null, step as never), true);
  assert.equal(layer.drawn.count, 6);
  assert.equal(layer.drawn.noValue, 0);
  assert.equal(layer.sync(s0, h.blobs, null, step as never), false, "nothing changed");
  layer.dispose();

  const g = buildHinges3D(reduce(s0, { type: "setHinges", colorBy: "capacity" }), h.blobs, step.hinges!, (d) => (join.ok ? join.byDeclPath.get(d) : undefined));
  assert.equal(g.top, 1);
  assert.equal(g.noValue, 0, "every hinge here has a capacity");
  const none = buildHinges3D(s0, h.blobs, step.hinges!, null);
  assert.equal(none.noValue, 6, "unjoined results colour nothing");
  assert.deepEqual([...none.colors.subarray(0, 3)].map((c) => Number(c.toFixed(4))), HINGE_NO_DATA.map((c) => Number(c.toFixed(4))));

  h.store.dispatch({ type: "setHinges", element: `mesh/element/${O3.femIds["fbc"]}`, end: "j" });
  await until(() => hingeViewOf(h.store.get()).chart !== null, "the hinge history");
  const v = hingeViewOf(h.store.get());
  const live = O3.hingeSummary[String(O3.elements["fbc"])]!;
  assert.equal(v.chart!.curve.momentName, "Mz_j");
  assert.equal(v.chart!.step, last);
  assert.equal(v.chart!.capacity, 50e3);
  live.forEach((row, k) => {
    close(v.chart!.curve.rotation[k]!, row[2]!, `thetaZ_j step ${k}`);
    close(v.chart!.curve.moment[k]!, row[3]!, `Mz_j step ${k}`);
  });
  assert.deepEqual(v.axes, ["z"], "this hinge has no weak-axis law");

  const pick = `mesh/element/${O3.femIds["dbc"]}`;
  h.store.dispatch({ type: "select", pick: { decl: pick, at: null, nodes: null } });
  assert.equal(hingeViewOf(h.store.get()).focus, pick, "a viewport pick wins over the menu");
  await until(() => hingeViewOf(h.store.get()).chart?.curve.rotation[last] === O3.hingeSummary[String(O3.elements["dbc"])]![last]![2], "the picked hinge's history");
  assert.deepEqual(hingeViewOf(h.store.get()).axes, ["z", "y"], "a biaxial hinge plots both axes");
  h.store.dispatch({ type: "clearSelection" });
});

test("playback may skip steps: markers and the plot show the step on screen, whatever came before it", async () => {
  h.store.dispatch({ type: "setHinges", on: true, colorBy: "moment", element: `mesh/element/${O3.femIds["fbc"]}`, end: "i", axis: "z" });
  await atStep(O3.steps - 1);
  await until(() => hingeViewOf(h.store.get()).chart !== null, "the hinge history");
  const live = O3.hingeSummary[String(O3.elements["fbc"])]!;
  const layer = new HingesLayer();
  for (const k of [17, 3, 11]) {
    const step = await atStep(k);
    const s = h.store.get();
    const join = hingeJoinOf(s);
    assert.ok(join.ok);
    const row = [...step.hinges!.elementIds].indexOf(O3.elements["fbc"]!);
    close(step.hinges!.values[row * 8 + 1]!, live[k]![1]!, `Mz_i at step ${k}`);
    const v = hingeViewOf(s);
    assert.equal(v.chart!.step, k);
    close(v.chart!.curve.moment[k]!, live[k]![1]!, `the plot's Mz_i at step ${k}`);
    assert.equal(layer.sync(s, h.blobs, null, step as never), true, `step ${k} redraws`);
    assert.equal(layer.drawn.noValue, 0);
  }
  // A step still on its way keeps the last markers; it never paints another step's values as this one's.
  const waiting = reduce(h.store.get(), { type: "setPhase", at: { kind: "results", stage: 0, step: 5 } });
  assert.equal(layer.sync(waiting, h.blobs, null, null), false);
  layer.dispose();
  h.store.dispatch({ type: "setHinges", colorBy: "rotation", element: null });
});

test("with the switch off, stepping reads no hinge data at all", async () => {
  h.store.dispatch({ type: "setHinges", on: false });
  const readStep = h.host.readStep.bind(h.host), readHistory = h.host.readHingeHistory.bind(h.host);
  const asked: boolean[] = [];
  let histories = 0;
  h.host.readStep = (r) => (asked.push(r.hinges === true), readStep(r));
  h.host.readHingeHistory = (r) => (histories++, readHistory(r));
  try {
    for (const k of [2, 8, 5]) await atStep(k);
    assert.ok(asked.length >= 3, "the steps were read");
    assert.ok(asked.every((x) => !x), "no step asked for hingeSummary");
    assert.equal(histories, 0, "no history was read");
  } finally {
    h.host.readStep = readStep;
    h.host.readHingeHistory = readHistory;
  }
});

test("a step change redraws only the plot: the card and its menus are not rebuilt", () => {
  const s = reduce(h.store.get(), { type: "setPhase", at: { kind: "results", stage: 0, step: 4 } });
  const t = reduce(s, { type: "setPhase", at: { kind: "results", stage: 0, step: 9 } });
  assert.equal(hingeCardKeys(s).card, hingeCardKeys(t).card);
  assert.deepEqual([hingeCardKeys(s).step, hingeCardKeys(t).step], [4, 9]);
  assert.notEqual(hingeCardKeys(reduce(t, { type: "setHinges", axis: "y" })).card, hingeCardKeys(t).card);
});

test("the card: controls dispatch, Spotlight offers the switch, and a model without hinges has no card", () => {
  class El {
    className = "";
    title = "";
    value = "";
    selected = false;
    children: (El | string)[] = [];
    listeners: (() => void)[] = [];
    constructor(cls = "") {
      this.className = cls;
    }
    append(...xs: (El | string)[]) {
      this.children.push(...xs);
    }
    setAttribute() {}
    addEventListener(_: string, fn: () => void) {
      this.listeners.push(fn);
    }
    removeEventListener() {}
    set textContent(v: string) {
      this.children = [v];
    }
    all(pred: (e: El) => boolean): El[] {
      return [...(pred(this) ? [this] : []), ...this.children.flatMap((c) => (typeof c === "string" ? [] : c.all(pred)))];
    }
  }
  const prevDoc = (globalThis as { document?: unknown }).document;
  (globalThis as { document?: unknown }).document = { documentElement: { dataset: { theme: "dark" } }, createElement: () => new El() };
  try {
    const events: unknown[] = [];
    const s = reduce(h.store.get(), { type: "setHinges", on: false });
    const box = renderHinges(s, (e) => events.push(e)) as unknown as El;
    const one = (root: El, cls: string) => root.all((e) => e.className.split(" ")[0] === cls)[0]!;
    for (const fn of one(box, "hinges-toggle").listeners) fn();
    const menu = one(box, "hinges-color");
    menu.value = "moment";
    for (const fn of menu.listeners) fn();
    assert.deepEqual(events, [{ type: "setHinges", on: true }, { type: "setHinges", colorBy: "moment" }]);
    const cmd = spotlightViewOf(reduce(s, { type: "openSpotlight", query: "hinges" })).items.find((i) => i.id === "cmd:toggle-hinges");
    assert.deepEqual(cmd?.events, [{ type: "setHinges", on: true }]);
    assert.equal(renderHinges({ ...s, decls: {} }), null);
  } finally {
    (globalThis as { document?: unknown }).document = prevDoc;
  }
});

test("setHinges refuses bad values, and an unchanged value changes nothing", () => {
  const s = h.store.get();
  for (const bad of [{ on: "yes" }, { size: 0 }, { size: Number.NaN }, { colorBy: "x" }, { end: "k" }, { axis: "x" }, { notice: 1 }]) {
    assert.throws(() => reduce(s, { type: "setHinges", ...(bad as object) } as never), RangeError, JSON.stringify(bad));
  }
  assert.equal(reduce(s, { type: "setHinges", axis: s.hinges.axis }), s);
});

test("the card says which range the colours run to", () => {
  const s = reduce(h.store.get(), { type: "setPhase", at: { kind: "results", stage: 0, step: 1 } });
  const scale = (v: boolean) => hingeViewOf({ ...s, results: { ...s.results!, stages: [{ ...s.results!.stages[0]!, hingeStageRange: v }] } }).scale;
  assert.equal(s.results!.stages[0]!.hingeStageRange, true, "the fixture is small enough to scan");
  assert.match(scale(true)!, /of the stage$/);
  assert.match(scale(false)!, /step on screen/);
});

test("a stage without hingeSummary says so by name, and the markers still come from the model", async () => {
  const other = reduce(h.store.get(), { type: "setHinges", on: true });
  const stage = other.results!.stages[0]!;
  const s = { ...other, results: { ...other.results!, stages: [{ ...stage, hinges: 0 }] } };
  assert.ok(hingeViewOf(s).notes.some((n) => n.includes("does not record hingeSummary")));
  assert.equal(hingeViewOf(s).chart, null);
  assert.equal(buildHinges3D(s, h.blobs, null, null).count, 6);
});

test("the material chain follows the hinge flags to the right family", () => {
  const slots = (type: string, args: (number | string)[]) => {
    const d = ELEMENT_SYNTAX[type]!(args);
    assert.ok("slots" in d, type);
    return d.slots.filter((x) => x.family.endsWith("Material")).map((x) => [x.slot, x.family, x.label]);
  };
  assert.deepEqual(slots("LadrunoForceBeamColumn", [1, 1, "-hingeIJ", 2, 3, "-hingeLoc", 0.05, 0.95]), [[3, "uniaxialMaterial", "hingeIJ"], [4, "uniaxialMaterial", "hingeIJ#2"]]);
  assert.deepEqual(slots("LadrunoDispBeamColumn", [1, 1, "-hingeBiaxialI", 2]), [[3, "nDMaterial", "hingeBiaxialI"]]);
  assert.deepEqual(slots("LadrunoIMKBeam", [0.04, 2e11, 1.6e-4, 1, "-hingeBiaxialIJ", 7, 8]), [[5, "nDMaterial", "hingeBiaxialI"], [6, "nDMaterial", "hingeBiaxialJ"]]);
  assert.deepEqual(slots("LadrunoIMKBeam", [0.04, 2e11, 1.6e-4, 1, "-hingeBiaxial", 7]), [[5, "nDMaterial", "hingeBiaxial"]]);
});
