// renderer/entities.ts (ADR 0130 D8, slice VU8): entities draw where they are.
//
// The oracle is closed form: a storey's outline is the plan hull of its nodes
// at its level; a point set's markers are its coordinates; a cut with a
// polygon draws that polygon, and one without draws its plane where it crosses
// the model's box (every vertex on the plane and on the box), with a tick to
// the kept side. Each drawn coordinate is compared exactly.

import assert from "node:assert/strict";
import { test } from "node:test";
import { boxOf, drawnEntities, entityDrawing, planeInBox, planHull, type Box, type EntitiesIn } from "../src/renderer/entities.ts";
import { initialState } from "../src/state/reduce.ts";
import type { MeshInfo, State, UserCut, Vec3 } from "../src/state/types.ts";

const CURRENT = { stale: false, reason: "" };
/** frame_04b's plan: a 6 m by 4 m bay, levels 0, 4000, 7500 and 11000 mm. */
const NODES = new Map<number, Vec3>([
  [5, [0, 0, 4000]],
  [6, [6000, 0, 4000]],
  [7, [6000, 4000, 4000]],
  [8, [0, 4000, 4000]],
  [99, [3000, 2000, 3990]],
]);
const BOX: Box = [
  [0, 0, 0],
  [6000, 4000, 11000],
];

const cut = (o: Partial<UserCut>): UserCut => ({ row: 0, name: "c", side: "positive", point: [0, 0, 2000], normal: [0, 0, 1], reference: [0, 0, 2000], elementIds: [17], polygon: [], ...o });
const zone = (o: Partial<EntitiesIn> = {}): EntitiesIn => ({ ok: true, cuts: [], storeys: [], pointSets: [], refused: [], stale: CURRENT, ...o });
const triples = (xs: readonly number[]) => Array.from({ length: xs.length / 3 }, (_, i) => [xs[3 * i]!, xs[3 * i + 1]!, xs[3 * i + 2]!]);
const segments = (xs: readonly number[]) => Array.from({ length: xs.length / 6 }, (_, i) => [xs.slice(6 * i, 6 * i + 3), xs.slice(6 * i + 3, 6 * i + 6)]);

test("a storey draws the plan hull of its nodes at its level, and a marker at its reference", () => {
  const d = entityDrawing(zone({ storeys: [{ row: 0, name: "L1", level: 4000, reference: [3000, 2000, 4000], nodeIds: [5, 6, 7, 8, 99] }] }), NODES, BOX);
  // The interior node is not on the outline; node 99 sits 10 mm low and the outline is still at the level.
  assert.deepEqual(segments(d.storeyLines), [
    [[0, 0, 4000], [6000, 0, 4000]],
    [[6000, 0, 4000], [6000, 4000, 4000]],
    [[6000, 4000, 4000], [0, 4000, 4000]],
    [[0, 4000, 4000], [0, 0, 4000]],
  ]);
  assert.deepEqual(d.storeyReferences, [3000, 2000, 4000]);
  assert.deepEqual(d.notes, []);
});

test("a storey naming a node the model does not hold is not drawn, and says why", () => {
  const d = entityDrawing(zone({ storeys: [{ row: 3, name: "L9", level: 1, reference: [0, 0, 1], nodeIds: [5, 404] }] }), NODES, BOX);
  assert.deepEqual([d.storeyLines, d.storeyReferences], [[], []]);
  assert.deepEqual(d.notes, ['storey "L9" names node 404, which the model does not hold, so it is not drawn']);
  const none = entityDrawing(zone({ storeys: [{ row: 0, name: "L1", level: 1, reference: [0, 0, 1], nodeIds: [5] }] }), null, BOX);
  assert.match(none.notes[0]!, /the model's nodes are not read/);
});

test("a point set draws a marker at each of its points", () => {
  const xyz: Vec3[] = [
    [0, 0, 11000],
    [6000, 4000, 11000],
    [1234.5, -0.25, 7500],
  ];
  const d = entityDrawing(zone({ pointSets: [{ row: 0, name: "roof", xyz, hostElement: [-1, -1, 33] }] }), NODES, BOX);
  assert.deepEqual(triples(d.points), xyz);
});

test("a cut with a polygon draws that polygon, filled, with a tick to the kept side and a marker at its reference", () => {
  const polygon: Vec3[] = [
    [-500, 2000, 5000],
    [6500, 2000, 5000],
    [6500, 2000, 6500],
    [-500, 2000, 6500],
  ];
  const d = entityDrawing(zone({ cuts: [cut({ point: [3000, 2000, 5750], normal: [0, 1, 0], side: "negative", polygon, reference: [3000, 2000, 5750] })] }), NODES, BOX);
  const segs = segments(d.cutLines);
  assert.deepEqual(segs.slice(0, 4), polygon.map((a, i) => [a, polygon[(i + 1) % 4]]));
  // The tick: from the centroid, against the normal (the kept side is negative), a tenth of the box diagonal.
  const diag = Math.hypot(6000, 4000, 11000);
  assert.deepEqual(segs[4], [[3000, 2000, 5750], [3000, 2000 - 0.1 * diag, 5750]]);
  assert.deepEqual(triples(d.cutTris), [polygon[0], polygon[1], polygon[2], polygon[0], polygon[2], polygon[3]]);
  assert.deepEqual(d.cutReferences, [3000, 2000, 5750]);
});

test("a cut with no polygon draws its plane where it crosses the model's box", () => {
  const flat = entityDrawing(zone({ cuts: [cut({})] }), NODES, BOX);
  const ring = segments(flat.cutLines).slice(0, 4).map(([a]) => a!);
  assert.deepEqual(new Set(ring.map((p) => p.join(","))), new Set(["0,0,2000", "6000,0,2000", "6000,4000,2000", "0,4000,2000"]));
  // Each consecutive pair is a box edge in the plane: the ring is in order, never a bow tie.
  for (const [a, b] of segments(flat.cutLines).slice(0, 4)) assert.equal([0, 1].filter((k) => a![k] !== b![k]).length, 1);

  const n: Vec3 = [0.2822162605150792, -0.18814417367671948, 0.9407208683835974];
  const p: Vec3 = [2500, 0, 1500];
  const skew = planeInBox(p, n, BOX);
  assert.ok(skew.length >= 3 && skew.length <= 6);
  const diag = Math.hypot(6000, 4000, 11000);
  for (const v of skew) {
    assert.ok(Math.abs((v[0] - p[0]) * n[0] + (v[1] - p[1]) * n[1] + (v[2] - p[2]) * n[2]) <= 1e-9 * diag, `${v} is on the plane`);
    assert.ok(v.every((x, k) => x >= BOX[0][k]! - 1e-9 * diag && x <= BOX[1][k]! + 1e-9 * diag), `${v} is on the box`);
  }
  // Convex and in order: every turn has the same sign about the normal.
  const turns = skew.map((a, i) => {
    const b = skew[(i + 1) % skew.length]!;
    const c = skew[(i + 2) % skew.length]!;
    const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const w = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
    return Math.sign((u[1]! * w[2]! - u[2]! * w[1]!) * n[0] + (u[2]! * w[0]! - u[0]! * w[2]!) * n[1] + (u[0]! * w[1]! - u[1]! * w[0]!) * n[2]);
  });
  assert.equal(new Set(turns).size, 1);

  const miss = entityDrawing(zone({ cuts: [cut({ name: "high", point: [0, 0, 20000] })] }), NODES, BOX);
  assert.deepEqual([miss.cutLines, miss.notes], [[], ['cut "high": its plane misses the model, so it is not drawn']]);
});

test("planHull and boxOf: the closed forms the layer stands on", () => {
  assert.deepEqual(planHull([[0, 0], [2, 0], [1, 0], [2, 2], [0, 2], [1, 1]]), [[0, 0], [2, 0], [2, 2], [0, 2]]);
  assert.deepEqual(planHull([[1, 1], [1, 1]]), [[1, 1]]);
  const b = boxOf([0, 0, 0, 3, 4, 0], 0.2)!;
  assert.deepEqual(b, [[-1, -1, -1], [4, 5, 1]]);
});

test("drawnEntities: only a zone that reads and is current, with the model drawn and not on the geometry phase", () => {
  const mesh = { center: [0, 0, 0], radius: 1 } as unknown as MeshInfo;
  const ready = (entities: unknown): State =>
    ({
      ...initialState,
      mesh,
      user: { status: "ready", path: "C:/w/run.user.h5", info: { warnings: [], combinations: null, results: null, spectra: null, modes: null, entities } },
    }) as State;
  const e = zone();
  assert.equal(drawnEntities(ready(e)), e);
  assert.equal(drawnEntities(ready({ ...e, stale: { stale: true, reason: "fem_hash differs" } })), null, "A6: stale is drawn only on request");
  assert.equal(drawnEntities(ready({ ok: false, reason: "refused", stale: CURRENT })), null);
  assert.equal(drawnEntities(ready(null)), null);
  assert.equal(drawnEntities({ ...ready(e), mesh: null }), null);
  assert.equal(drawnEntities({ ...ready(e), phase: { ...initialState.phase, at: { kind: "geometry" } } as State["phase"] }), null);
});
