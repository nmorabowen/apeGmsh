// The Cracks card on layered shells end to end over fixtures/ladruno_shells3d.ladruno (a fork run of a cracked
// 3 x 2 wall of ASDShellQ4 in the x-z plane, `-local 1 0 0`, with MODEL/LOCAL_AXES from nmorabowen/OpenSees#1133):
// the field (state/shellCracks.ts), main's layer read (main/results.ts `readStep` `layers`), the step fetch
// (effects), the ticks (renderer/cracks.ts) and the card (panels/cracks.ts). The pure tick builder is checked on
// hand-made frames and openings, where the right answer is known without the app.
//
// The wall's frame, as the file's MODEL/LOCAL_AXES records it: e1 = +x, e3 = -y (the normal, most along y; checked
// against the element's geometry below), so the face toward +y (red) is the first concrete layer (the -e3 side) and
// the face toward -y (blue) the last.

import assert from "node:assert/strict";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import { renderCracks } from "../src/panels/cracks.ts";
import { CracksLayer } from "../src/renderer/cracks.ts";
import * as h5wasm from "h5wasm/node";
import { quaternionToZRec } from "../src/reader/ladrunoLocalAxes.ts";
import { buildShellTicks, FACE_RGB, frameRows, shellCrackFieldOf, type ShellTickGroup } from "../src/state/shellCracks.ts";
import { shellCrackFieldOf as selectShellField } from "../src/state/selectors.ts";
import type { LayerGroup } from "../src/state/types.ts";
import { opened, until } from "./contour/harness.ts";

const fixture = (name: string) => fileURLToPath(new URL(`../fixtures/${name}`, import.meta.url));
const h = await opened(fixture("ladruno_shells3d.h5"), fixture("ladruno_shells3d.ladruno"));
after(() => {
  h.effects.dispose();
  h.host.close();
});

const CRACK = [1, 2, 3].flatMap((k) => ["x", "y", "z"].map((a) => `section.fiber.crackPattern/C${k}${a}`));
const G6 = "section.fiber.crackPattern/203-ASDShellQ4[201:0:0]";
const G4 = "section.fiber.crackPattern/203-ASDShellQ4[201:0:1]";
const LAST = 5;
const RED = FACE_RGB.plus, BLUE = FACE_RGB.minus;

async function atStep(step: number, withLayers: boolean) {
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step } });
  const here = () => {
    const d = h.effects.currentStep();
    return d && d.stage === 0 && d.step === step && (!withLayers || d.layers) ? d : null;
  };
  await until(() => here() !== null, `step ${step}`);
  return here()!;
}

type Geo = { geometry: { getAttribute(n: string): { array: Float32Array } } };
const ticksOf = (layer: CracksLayer) => {
  const lines = layer.root.children[1] as unknown as Geo;
  return { pos: lines.geometry.getAttribute("position").array, col: lines.geometry.getAttribute("color").array };
};
const toLin = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const isColour = (col: Float32Array, v: number, rgb: readonly number[]) => rgb.every((c, k) => Math.abs(col[3 * v + k]! - toLin(c)) < 1e-6);

// -- the pure builder on hand-made frames ---------------------------------------------

/** One group of one element and one Gauss point, its two outer layers read, in the frame `q`. */
function oneGroup(q: [number, number, number, number], first: number[], last: number[], thickness: number | null = null): ShellTickGroup {
  return {
    values: Float64Array.from([...first, ...last]),
    layers: [0, 5],
    nLayers: 6,
    nGauss: 1,
    quaternions: Float64Array.from(q),
    recorded: Uint8Array.from([1]),
    thickness: thickness === null ? null : Float64Array.from([thickness]),
  };
}
const ONE = { group: Int32Array.of(0), element: Int32Array.of(0), gauss: Int32Array.of(0) };
const crack = (x: number, y: number, z: number) => [x, y, z, 0, 0, 0, 0, 0, 0];
const OPTS = { face: "both" as const, minOpening: 0, halfLength: 10, eps: 1 };

test("frameRows: the rows of the quaternion's rotation matrix (identity, a quarter turn about z); row 3 is quaternionToZRec", () => {
  assert.deepEqual(frameRows(1, 0, 0, 0), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
  const s = Math.SQRT1_2;
  // R(q) of a +90 deg turn about z is [[0, -1, 0], [1, 0, 0], [0, 0, 1]]: e1 = -y, e2 = +x, e3 = +z.
  const r = frameRows(s, 0, 0, s).map((row) => row.map((v) => Math.round(v * 1e12) / 1e12 + 0));
  assert.deepEqual(r, [[0, -1, 0], [1, 0, 0], [0, 0, 1]]);
  // The same matrix the reader decodes (reader/ladrunoLocalAxes.ts, as Python LocalAxes.matrices), for a generic q.
  const q = [0.3, -0.5, 0.7, 0.4].map((v, _, a) => v / Math.hypot(...a)) as [number, number, number, number];
  frameRows(...q)[2]!.forEach((v, k) => assert.ok(Math.abs(v - quaternionToZRec(...q)[k]!) < 1e-12));
});

test("frameRows on the recorded frames is each shell's own frame: e1 along its first edge, e3 along its diagonals' cross product", () => {
  // The independent oracle: ASDShellQ4 with no -local takes e1 along node 1 -> 2 and e3 = (x3 - x1) x (x4 - x2),
  // here from the fixture's own MODEL; the wall in the x-z plane gives e1 = +x, e2 = +z, e3 = -y.
  const f = new h5wasm.File(fixture("ladruno_shells3d.ladruno"), "r");
  try {
    const stage = f.keys().filter((k) => k.startsWith("MODEL_STAGE")).pop()!;
    const ids = Array.from((f.get(`${stage}/MODEL/NODES/ID`) as h5wasm.Dataset).value as Int32Array);
    const xyz = (f.get(`${stage}/MODEL/NODES/COORDINATES`) as h5wasm.Dataset).value as Float64Array;
    const at = (n: number) => Array.from(xyz.subarray(3 * ids.indexOf(n), 3 * ids.indexOf(n) + 3));
    const conn = (f.get(`${stage}/MODEL/ELEMENTS/203-ASDShellQ4[201:0]/CONNECTIVITY`) as h5wasm.Dataset).value as Int32Array;
    const la = `${stage}/MODEL/LOCAL_AXES/203-ASDShellQ4`;
    const eid = Array.from((f.get(`${la}/ID`) as h5wasm.Dataset).value as Int32Array);
    const q = (f.get(`${la}/FRAME`) as h5wasm.Dataset).value as Float64Array;
    const unit = (v: number[]) => v.map((x) => x / Math.hypot(...v));
    const sub = (a: number[], b: number[]) => a.map((x, k) => x - b[k]!);
    const cross = (a: number[], b: number[]) => [a[1]! * b[2]! - a[2]! * b[1]!, a[2]! * b[0]! - a[0]! * b[2]!, a[0]! * b[1]! - a[1]! * b[0]!];
    let checked = 0;
    for (let e = 0; e < conn.length / 5; e++) {
      const [id, n1, n2, n3, n4] = Array.from(conn.subarray(5 * e, 5 * e + 5));
      const row = eid.indexOf(id!);
      if (row < 0) continue;
      const e3 = unit(cross(sub(at(n3!), at(n1!)), sub(at(n4!), at(n2!))));
      const e1 = unit(sub(at(n2!), at(n1!)));
      const [r1, , r3] = frameRows(q[4 * row]!, q[4 * row + 1]!, q[4 * row + 2]!, q[4 * row + 3]!);
      for (const [got, want] of [[r1!, e1], [r3!, e3]] as const) got.forEach((v, k) => assert.ok(Math.abs(v - want[k]!) < 1e-9, `element ${id}: ${got} vs ${want}`));
      assert.deepEqual(e3.map((v) => Math.round(v) + 0), [0, -1, 0]);
      checked++;
    }
    assert.ok(checked > 0);
  } finally {
    f.close();
  }
});

test("a tick lies along the crack's trace e3 x n, centred on its Gauss point, a hair off its face along e3", () => {
  // Identity frame (a slab, e3 = +z): a crack whose normal is e1 = x traces along e3 x e1 = +y.
  const t = buildShellTicks(Float32Array.of(0, 0, 0), ONE, [oneGroup([1, 0, 0, 0], crack(0.5, 0, 0), crack(0, 0.3, 0))], OPTS);
  assert.equal(t.ticks, 2);
  // The +z face is the last layer (e3 = +z): red there, at z = +eps, its crack normal y tracing along x.
  const ticks = [0, 1].map((k) => ({ a: Array.from(t.positions.subarray(6 * k, 6 * k + 3)), b: Array.from(t.positions.subarray(6 * k + 3, 6 * k + 6)), rgb: Array.from(t.colours.subarray(6 * k, 6 * k + 3)) }));
  const red = ticks.find((x) => x.rgb.every((c, k) => Math.abs(c - RED[k]!) < 1e-6))!;
  const blue = ticks.find((x) => x.rgb.every((c, k) => Math.abs(c - BLUE[k]!) < 1e-6))!;
  assert.deepEqual([red.a[2], red.b[2]], [1, 1], "red: the +z face, eps above");
  assert.deepEqual([Math.abs(red.a[0]!), red.a[1]], [10, 0], "the last layer's crack normal is y: its trace is along x");
  assert.deepEqual([blue.a[2], blue.b[2]], [-1, -1], "blue: the -z face, eps below");
  assert.deepEqual([blue.a[0], Math.abs(blue.a[1]!)], [0, 10], "the first layer's crack normal is x: its trace is along y");
  assert.equal(t.maxOpening, 0.5);
});

test("red is the face toward the positive axis the normal is most along, whichever way e3 points", () => {
  // A half turn about x: e3 = -z. The +z face is now the first layer (the -e3 side): red takes the first layer's crack.
  const t = buildShellTicks(Float32Array.of(0, 0, 0), ONE, [oneGroup([0, 1, 0, 0], crack(0.5, 0, 0), crack(0, 0.3, 0))], { ...OPTS, face: "plus" });
  assert.equal(t.ticks, 1);
  assert.ok(Math.abs(t.positions[2]! - 1) < 1e-12, "on the +z side");
  assert.ok(Math.abs(Math.abs(t.positions[1]!) - 10) < 1e-12, "the first layer's crack (normal along e1 = x) traces along y");
  assert.ok(Array.from(t.colours.subarray(0, 3)).every((c, k) => Math.abs(c - RED[k]!) < 1e-6));
});

test("with the section's thickness, a tick sits at its layer's face, t/2 + eps off the mid-surface; an opening under the threshold, or an element with no frame, is not drawn", () => {
  const g = oneGroup([1, 0, 0, 0], crack(0.5, 0, 0), crack(0.02, 0, 0), 200);
  const t = buildShellTicks(Float32Array.of(0, 0, 0), ONE, [g], { ...OPTS, minOpening: 0.05 });
  assert.equal(t.ticks, 1, "the 0.02 crack is under 0.05");
  assert.equal(t.positions[2], -101, "the -z face of a 200 thick slab, plus eps");
  const unframed = buildShellTicks(Float32Array.of(0, 0, 0), ONE, [{ ...g, recorded: Uint8Array.of(0) }], OPTS);
  assert.deepEqual([unframed.ticks, unframed.unframed], [0, 1]);
});

test("the field: refused by name with no frames, with partial frames named, and with element rows not verified", () => {
  const base: LayerGroup = { id: G6, components: CRACK, refused: [], nElements: 4, nGauss: 4, nLayers: 6, naturalCoords: null, layerZ: [], geometryReason: null };
  const refs = { elementIds: { id: "e" }, quaternions: { id: "q" }, recorded: { id: "r" } } as never as Pick<LayerGroup, "elementIds" | "quaternions" | "recorded">;
  assert.deepEqual(shellCrackFieldOf([{ ...base, ...refs, unframed: 0 }])?.refused, []);
  assert.match(shellCrackFieldOf([{ ...base, ...refs, unframed: 4 }])!.refused[0]!, /records no frame.*#1133/);
  const partial = shellCrackFieldOf([{ ...base, ...refs, unframed: 1 }])!;
  assert.equal(partial.groups.length, 1);
  assert.match(partial.refused[0]!, /^1 of the 4 elements/);
  assert.match(shellCrackFieldOf([base])!.refused[0]!, /not verified/);
  assert.equal(shellCrackFieldOf([{ ...base, components: ["section.fiber.damage/d+"] }]), null);
});

// -- end to end on the fork run --------------------------------------------------------

test("the stage's shell crack field is both header buckets; the step reads their two outer layers only while the card is on", async () => {
  await atStep(LAST, false);
  const f = selectShellField(h.store.get())!;
  assert.deepEqual(f.groups.map((g) => g.id).sort(), [G6, G4].sort());
  assert.deepEqual(f.components, CRACK);
  assert.deepEqual(f.refused, []);
  assert.equal(h.effects.currentStep()!.layers, undefined, "off: no layer read");
  h.store.dispatch({ type: "setCracks", on: true });
  const d = await atStep(LAST, true);
  assert.deepEqual(d.layers![G6]!.layers, [0, 5]);
  assert.deepEqual(d.layers![G4]!.layers, [0, 3]);
  assert.equal(d.layers![G6]!.values.length, 4 * 4 * 2 * 9, "4 elements x 4 Gauss points x 2 layers x 9 components");
  h.store.dispatch({ type: "setCracks", on: false });
});

test("the ticks: in the wall's plane, along a trace normal to the wall's normal (y); red on the +y face, blue on the -y face; one face at a time", async () => {
  h.store.dispatch({ type: "setCracks", on: true, minOpening: 0.01, style: "ticks" });
  const d = await atStep(LAST, true);
  const layer = new CracksLayer();
  try {
    assert.equal(layer.sync(h.store.get(), h.blobs, null, d), true);
    assert.equal(layer.root.visible, true);
    assert.ok(layer.drawnShell.ticks > 0, "the wall has cracked by the last step");
    assert.equal(layer.drawnShell.unframed, 0);
    const { pos, col } = ticksOf(layer);
    let red = 0, blue = 0;
    for (let v = 0; v < pos.length / 3; v += 2) {
      const ya = pos[3 * v + 1]!, yb = pos[3 * (v + 1) + 1]!;
      assert.ok(Math.abs(ya - yb) < 1e-4, "a tick lies in its face: no extent along the normal (y)");
      if (isColour(col, v, RED)) {
        red++;
        assert.ok(ya > 0, `red on the +y face, y = ${ya}`);
      } else if (isColour(col, v, BLUE)) {
        blue++;
        assert.ok(ya < 0, `blue on the -y face, y = ${ya}`);
      } else assert.fail(`tick ${v / 2} is neither face's colour`);
    }
    assert.ok(red > 0 && blue > 0, `${red} red, ${blue} blue`);
    h.store.dispatch({ type: "setCracks", face: "plus" });
    assert.equal(layer.sync(h.store.get(), h.blobs, null, d), true);
    assert.equal(layer.drawnShell.ticks, red, "the red face alone");
    h.store.dispatch({ type: "setCracks", face: "minus" });
    layer.sync(h.store.get(), h.blobs, null, d);
    assert.equal(layer.drawnShell.ticks, blue, "the blue face alone");
    h.store.dispatch({ type: "setCracks", on: false });
    layer.sync(h.store.get(), h.blobs, null, d);
    assert.equal(layer.root.visible, false);
  } finally {
    layer.dispose();
    h.store.dispatch({ type: "setCracks", on: false, minOpening: 0, face: "both", style: "lines" });
  }
});

test("the card: a shells row with the face picker and its colours, no colour map without discs; the picker dispatches setCracks", async () => {
  await atStep(LAST, false);
  const prevDoc = (globalThis as { document?: unknown }).document;
  class El {
    className = "";
    title = "";
    value = "";
    type = "";
    step = "";
    min = "";
    selected = false;
    dataset: Record<string, string> = {};
    style: Record<string, string> = {};
    children: (El | string)[] = [];
    listeners: (() => void)[] = [];
    append(...xs: (El | string)[]) {
      this.children.push(...xs);
    }
    setAttribute() {}
    setCustomValidity() {}
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
  (globalThis as { document?: unknown }).document = { documentElement: { dataset: { theme: "dark" } }, createElement: () => new El(), createElementNS: () => new El() };
  try {
    const events: unknown[] = [];
    const box = renderCracks(h.store.get(), (e) => events.push(e)) as unknown as El | null;
    assert.ok(box, "the card shows: the stage records a shell crack pattern");
    const of = (cls: string) => box.all((e) => e.className.split(" ").includes(cls));
    assert.equal(of("cracks-cmap").length, 0, "no disc field: no colour map");
    assert.deepEqual([of("cracks-face-plus")[0]!.style.background, of("cracks-face-minus")[0]!.style.background], ["#B2182B", "#2166AC"]);
    const face = of("cracks-face")[0]!;
    assert.deepEqual(face.children.map((o) => (o as El).value), ["both", "plus", "minus"]);
    face.value = "minus";
    for (const fn of face.listeners) fn();
    assert.deepEqual(events, [{ type: "setCracks", face: "minus" }]);
  } finally {
    (globalThis as { document?: unknown }).document = prevDoc;
  }
});
