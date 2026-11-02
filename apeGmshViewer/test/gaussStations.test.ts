// Beam stations as Gauss markers on a real file: the element's own I to J order
// for joint offsets, the model's connectivity when the step has none, the
// stations the line diagram would refuse, stations that never cross files, and
// the Spotlight command's Alt+Enter.

import assert from "node:assert/strict";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { opened, STAGED, STAGED_RESULTS, until } from "./contour/harness.ts";
import { gaussBasePositions, gaussLayoutOf } from "../src/state/gaussPoints.ts";
import { spotlightViewOf } from "../src/state/spotlight.ts";
import { reduce } from "../src/state/store.ts";

const fixture = (n: string) => fileURLToPath(new URL(`../fixtures/${n}`, import.meta.url));
const CANTILEVER = fixture("line_oracle_cantilever3d.h5");
const CANTILEVER_RESULTS = fixture("line_oracle_cantilever3d.ladruno");

/** The cantilever on its first step with the markers on, and its stations. Element 2 is mesh/element/4, on nodes 1 (x = 0) and 2 (x = 2). */
async function cantilever() {
  const h = await opened(CANTILEVER, CANTILEVER_RESULTS);
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 0 } });
  h.store.dispatch({ type: "setGaussPoints", on: true });
  await until(() => h.effects.stationsOfStep() !== null, "cantilever stations");
  return { h, stations: h.effects.stationsOfStep()! };
}

const xs = (h: Awaited<ReturnType<typeof opened>>, s: ReturnType<Awaited<ReturnType<typeof opened>>["store"]["get"]>, layout: NonNullable<ReturnType<typeof gaussLayoutOf>>) =>
  [...gaussBasePositions(layout, h.blobs.get(s.mesh!.nodes!.ids) as Float64Array, h.blobs.get(s.mesh!.nodes!.xyz) as Float64Array)].filter((_, i) => i % 3 === 0);

test("joint offsets go to the element's own I and J, also on a cell stored J to I", async () => {
  const { h, stations } = await cantilever();
  try {
    const s0 = h.store.get();
    const tr = s0.decls["opensees/geomTransf/#1"]!;
    const s = { ...s0, decls: { ...s0.decls, ["opensees/geomTransf/#1"]: { ...tr, jntOffset: [{ tag: 1, dI: [0.5, 0, 0], dJ: [-0.25, 0, 0] }] } } as typeof s0.decls };
    const at = (en: number[]) => xs(h, s, gaussLayoutOf(s, h.blobs, { ...stations, elementNodes: Float64Array.from(en) })!).slice(0, 2);
    // Recorded I = node 1, J = node 2: xi = -1 at 0 + 0.5, xi = 1 at 2 - 0.25.
    assert.deepEqual(at([1, 2, 1, 2, 2, 3, 2, 3]), [0.5, 1.75]);
    // Recorded I = node 2, J = node 1, as the line diagram draws it: xi = -1 at 2 + 0.5, xi = 1 at 0 - 0.25.
    assert.deepEqual(at([2, 1, 2, 1, 2, 3, 2, 3]), [2.5, -0.25]);
  } finally {
    h.effects.dispose();
  }
});

test("-endZones are measured from the element's own I to J, also on a cell stored J to I", async () => {
  const { h, stations } = await cantilever();
  try {
    const s0 = h.store.get();
    const e4 = s0.decls["mesh/element/4"]!;
    const m0 = e4.element!.metas[0]!;
    // Unequal zones: 0.3 at I and 0.1 at J, so swapped ends would be seen.
    const metas = [{ ...m0, type: "LadrunoIMKBeam", args: [...m0.args, "-endZones", 0.3, 0.1] }, ...e4.element!.metas.slice(1)];
    const s = { ...s0, decls: { ...s0.decls, ["mesh/element/4"]: { ...e4, element: { ...e4.element!, metas } } } as typeof s0.decls };
    const at = (en: number[]) => xs(h, s, gaussLayoutOf(s, h.blobs, { ...stations, elementNodes: Float64Array.from(en) })!).slice(0, 2).map((x) => Math.round(x * 1e6) / 1e6);
    assert.deepEqual(at([1, 2, 1, 2, 2, 3, 2, 3]), [0.3, 1.9]);
    assert.deepEqual(at([2, 1, 2, 1, 2, 3, 2, 3]), [1.7, 0.1]);
  } finally {
    h.effects.dispose();
  }
});

test("stations with NaN element nodes keep one key: a new key per call would rebuild the markers without end", async () => {
  const { h } = await cantilever();
  try {
    const step = (h.effects as unknown as { step: { lineDiagram: { elementNodes: Float64Array } } }).step;
    step.lineDiagram.elementNodes[0] = NaN;
    const keys = [1, 2, 3].map(() => h.effects.stationsOfStep()!.key);
    assert.equal(new Set(keys).size, 1, `keys ${keys.join(" ")}`);
  } finally {
    h.effects.dispose();
  }
});

test("with no element nodes in the step, the model's connectivity orders the stations", async () => {
  const { h, stations } = await cantilever();
  try {
    const s0 = h.store.get();
    const e4 = s0.decls["mesh/element/4"]!;
    const s = { ...s0, decls: { ...s0.decls, ["mesh/element/4"]: { ...e4, element: { ...e4.element!, inlineNodes: [2, 1] } } } as typeof s0.decls };
    assert.deepEqual(xs(h, s, gaussLayoutOf(s, h.blobs, { ...stations, elementNodes: null })!).slice(0, 2), [2, 0]);
  } finally {
    h.effects.dispose();
  }
});

test("a station whose xi is not finite or is outside [-1, 1] refuses its element, as the line diagram does", async () => {
  const { h } = await cantilever();
  try {
    const s = h.store.get();
    const layout = gaussLayoutOf(s, h.blobs, { elementIndex: Float64Array.from([2, 2, 3, 3]), stationNaturalCoord: Float64Array.from([NaN, NaN, -1, 3]), elementNodes: null })!;
    assert.equal(layout.count, 0);
    assert.deepEqual(layout.refused, [{ reason: "beam stations with a natural coordinate that is not finite or is outside [-1, 1]", elements: 2 }]);
  } finally {
    h.effects.dispose();
  }
});

test("another file's stations are never placed: a stage read without stations has none", async () => {
  const { h } = await cantilever();
  try {
    if (!(await h.effects.open(STAGED))) throw new Error("staged model");
    if (!(await h.effects.openResults(STAGED_RESULTS))) throw new Error("staged results");
    h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 0 } });
    if (!h.store.get().gaussPoints.on) h.store.dispatch({ type: "setGaussPoints", on: true });
    await until(() => (h.effects as unknown as { step: { stage: number } | null }).step?.stage === 0, "staged step");
    assert.equal(h.effects.stationsOfStep(), null);
  } finally {
    h.effects.dispose();
  }
});

test("the Spotlight command's Alt+Enter toggles the markers too", async () => {
  const { h } = await cantilever();
  try {
    const s = reduce(reduce(h.store.get(), { type: "openSpotlight" }), { type: "setSpotlightQuery", query: "gauss" });
    const item = spotlightViewOf(s).items.find((i) => i.id === "cmd:toggle-gauss-points");
    const alt = (item as { action?: { altEvents?: readonly Parameters<typeof reduce>[1][] } } | undefined)?.action?.altEvents;
    assert.ok(alt, "no Alt+Enter events");
    assert.equal(alt.reduce(reduce, s).gaussPoints.on, !s.gaussPoints.on);
  } finally {
    h.effects.dispose();
  }
});
