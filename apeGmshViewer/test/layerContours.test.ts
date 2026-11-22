// Layered-shell results as element contour fields (reader/layerVariants.ts): each layer group offered as its
// + face, - face and max over layers, read and reduced by the reader, drawn by the Contour card as any element field.
// The oracle is the raw layer read (S1's, itself checked against DATA by the spec's column rule) and the face rule
// (the + face is the outer layer facing the positive global axis the normal is most along).
//
// ladruno_shells3d: a wall in the x-z plane whose MODEL/LOCAL_AXES gives e3 = -y (its diagonals' cross product;
// test/shellCracks.test.ts checks the frame against the geometry), so its + face (+y) is the first layer (the -e3
// side) and its - face the last.

import assert from "node:assert/strict";
import { readdirSync } from "node:fs";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { openLadruno, type LadrunoFile } from "../src/reader/ladruno.ts";
import { openLadrunoSet } from "../src/reader/ladrunoParts.ts";
import { plusLayersOf, reduceLayers } from "../src/reader/layerVariants.ts";
import type { ResultsH5 } from "../src/reader/results.ts";
import { gaussFields } from "../src/state/contour.ts";
import { contourView } from "../src/state/selectors.ts";
import { opened, until } from "./contour/harness.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as ResultsH5;
const fixture = (name: string) => fileURLToPath(new URL(`../fixtures/${name}`, import.meta.url));
const SHELLS = fixture("ladruno_shells3d.ladruno");
const DMG6 = "section.fiber.damage/203-ASDShellQ4[201:0:0]";

function withFile<T>(path: string, body: (r: LadrunoFile) => T): T {
  const r = openLadruno(H5, path);
  try {
    return body(r);
  } finally {
    r.close();
  }
}

test("reduceLayers: + face, - face and max over layers, per element; NaN for an element with no frame", () => {
  // T = 1, E = 2, G = 1, L = 3: element 0 layers [1, 5, 2], element 1 [7, 0, 3].
  const v = [1, 5, 2, 7, 0, 3];
  const plus = Int32Array.of(2, -1);
  assert.deepEqual(Array.from(reduceLayers(v, 1, 2, 1, 3, "max", null)), [5, 7]);
  assert.deepEqual(Array.from(reduceLayers(v, 1, 2, 1, 3, "+face", plus)), [2, NaN]);
  assert.deepEqual(Array.from(reduceLayers(v, 1, 2, 1, 3, "-face", plus)), [1, NaN]);
});

test("plusLayersOf: the last layer when e3 points to the positive axis it is most along, the first when it points away", () => {
  const s = Math.SQRT1_2;
  // e3 is row 3 of R(q): identity (e3 = +z) -> last; a half turn about x (e3 = -z) -> first; a +90 deg turn about x
  // (e3 = +y) -> last; a -90 deg turn about x (e3 = -y) -> first; unrecorded -> -1.
  const q = [1, 0, 0, 0, 0, 1, 0, 0, s, s, 0, 0, s, -s, 0, 0, 1, 0, 0, 0];
  assert.deepEqual(Array.from(plusLayersOf(q, [1, 1, 1, 1, 0], 6)), [5, 0, 5, 0, -1]);
});

test("each layer group is offered as three element fields, read as the raw layers reduced", () => {
  withFile(SHELLS, (r) => {
    const st = r.index.stages[0]!;
    const ids = st.gauss.map((g) => g.id);
    for (const v of ["+face", "-face", "max"]) assert.ok(ids.includes(`${DMG6}@${v}`), `${v} in ${ids.join(", ")}`);
    const g = st.gauss.find((x) => x.id === `${DMG6}@max`)!;
    assert.deepEqual(g.components, ["section.fiber.damage@max/d+", "section.fiber.damage@max/d-"]);
    assert.deepEqual(g.layerVariant, { layerGroupId: DMG6, variant: "max" });
    const lg = st.layers!.find((x) => x.id === DMG6)!;
    const step = st.nSteps - 1;
    // Raw: (E, G, L, 1) of d+.
    const raw = r.readLayerStep(st.id, DMG6, ["section.fiber.damage/d+"], null, step);
    const { nElements: E, nGauss: G, nLayers: L } = lg;
    const max = r.readGaussStep(st.id, `${DMG6}@max`, "section.fiber.damage@max/d+", step);
    const plus = r.readGaussStep(st.id, `${DMG6}@+face`, "section.fiber.damage@+face/d+", step);
    const minus = r.readGaussStep(st.id, `${DMG6}@-face`, "section.fiber.damage@-face/d+", step);
    assert.equal(max.length, E * G);
    for (let e = 0; e < E; e++)
      for (let p = 0; p < G; p++) {
        const layers = Array.from({ length: L }, (_, k) => raw[(e * G + p) * L + k]!);
        assert.equal(max[e * G + p], Math.max(...layers));
        // The wall's e3 is -y: its + face (+y) is the first layer, its - face the last.
        assert.equal(plus[e * G + p], layers[0]);
        assert.equal(minus[e * G + p], layers[L - 1]);
      }
    assert.ok(Array.from(max).some((x) => x > 0), "the wall has damage at the last step");
    // The rows are the layer group's.
    assert.deepEqual(r.elementIds(st.id, `${DMG6}@max`), r.layerElementIds(st.id, DMG6));
    // Steps stacked as one read: readGaussRows is readGaussStep step-major.
    const rows = r.readGaussRows(st.id, `${DMG6}@max`, "section.fiber.damage@max/d+", step - 1, step + 1);
    assert.deepEqual(Array.from(rows.subarray(E * G)), Array.from(max));
  });
});

test("a file with no shell frames offers max over layers only; the face variants are refused by name", () => {
  withFile(fixture("ladruno_shells3d.noframes.ladruno"), (r) => {
    const st = r.index.stages[0]!;
    const face = st.gauss.find((x) => x.id === `${DMG6}@+face`)!;
    assert.deepEqual(face.components, []);
    assert.match(face.refused[0]?.reason ?? "", /no frame.*LOCAL_AXES.*#1133/);
    assert.equal(st.gauss.find((x) => x.id === `${DMG6}@max`)!.components.length, 2);
  });
});

test("a part set offers the stitched layer groups' variants; a multi-step read is each part's, end to end within each step", () => {
  const parts = [fixture("ladruno_shellparts.part-0.ladruno"), fixture("ladruno_shellparts.part-1.ladruno")];
  const s = openLadrunoSet(H5, parts[0]!, (d) => readdirSync(d));
  const own = parts.map((p) => openLadruno(H5, p, { part: true }));
  try {
    const st = s.index.stages[0]!;
    const CR = "section.fiber.crackPattern/203-ASDShellQ4[201:0:0]@max";
    assert.ok(st.gauss.some((g) => g.id === CR && g.components.length === 9), st.gauss.map((g) => g.id).join(", "));
    assert.ok(!st.unread.some((u) => u.includes("@")), "a variant is never listed unread");
    const comp = "section.fiber.crackPattern@max/C1x";
    const k1 = st.nSteps, k0 = k1 - 2;
    const got = s.readGaussRows(st.id, CR, comp, k0, k1);
    const per = own.map((f) => (f.index.stages[0]!.gauss.some((g) => g.id === CR) ? f.readGaussRows(st.id, CR, comp, k0, k1) : new Float64Array(0)));
    const want: number[] = [];
    for (let t = 0; t < 2; t++) for (const r of per) want.push(...r.subarray((t * r.length) / 2, ((t + 1) * r.length) / 2));
    assert.deepEqual(Array.from(got), want);
    const e = s.elementIds(st.id, CR);
    assert.deepEqual(e.mapping === "fem_eid" ? Array.from(e.ids) : [], [1, 4, 2, 5]);
  } finally {
    s.close();
    for (const f of own) f.close();
  }
});

// -- the Contour card ---------------------------------------------------------------------------

const h = await opened(fixture("ladruno_shells3d.h5"), SHELLS);
after(() => {
  h.effects.dispose();
  h.host.close();
});

test("the Contour card lists the variants as element fields, and draws the max of d+ over the layers at the last step", async () => {
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 0 } });
  const st = h.store.get().results!.stages[0]!;
  const fields = gaussFields(st.gauss).map((f) => f.name);
  for (const v of ["+face", "-face", "max"]) assert.ok(fields.includes(`section.fiber.damage@${v}`), fields.join(", "));
  const f = gaussFields(st.gauss).find((x) => x.name === "section.fiber.damage@max")!;
  assert.equal(f.refused, null);
  assert.equal(f.groups.length, 2, "both header buckets carry it");
  h.store.dispatch({ type: "setResultStep", step: 5 });
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  h.store.dispatch({ type: "setContourRange", range: "step" });
  h.store.dispatch({ type: "setContourField", field: f.name, component: f.components[0]!.key, source: "element" });
  await until(() => {
    const v = contourView(h.store.get());
    return v.state === "element" && v.stepMeasured && h.effects.currentStep()?.element === f.components[0]!.key;
  }, "the element field");
  const v = contourView(h.store.get());
  assert.equal(v.state, "element");
  if (v.state === "element") assert.ok(v.clim.max! > 0, `the colour range tops at ${v.clim.max}`);
});
