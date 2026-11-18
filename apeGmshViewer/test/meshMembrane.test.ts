// #2667: a bridge-minted LadrunoMembrane (fem_eid -1, no inline connectivity)
// is drawn as the polygon its archived `-nodes` / `-hole` command names. The
// args layout is `LadrunoMembrane._emit`'s.

import assert from "node:assert/strict";
import { test } from "node:test";
import { buildMesh } from "../src/mesh/build.ts";
import { membraneLoops, opsOnlyNodes } from "../src/mesh/membrane.ts";
import type { DeclRecord, DeclTable, Definitions, ElementMeta, ModelFile, Param } from "../src/model/types.ts";

/** An L-shaped outer loop (area 3) and a square hole node set, in the z = 2 plane. */
const NODES: Record<number, [number, number, number]> = {
  1: [0, 0, 2], 2: [2, 0, 2], 3: [2, 1, 2], 4: [1, 1, 2], 5: [1, 2, 2], 6: [0, 2, 2],
  11: [0.25, 0.25, 2], 12: [0.75, 0.25, 2], 13: [0.75, 0.75, 2], 14: [0.25, 0.75, 2],
};
const L_LOOP = [1, 2, 3, 4, 5, 6];
const HOLE = [11, 12, 13, 14];

function meta(type: string, rows: Param[][]): ElementMeta {
  return {
    type,
    path: `/opensees/element_meta/${type}`,
    ids: Float64Array.from(rows, (_, i) => 100 + i),
    femEids: new Float64Array(rows.length).fill(-1),
    args: rows,
    inlineConnectivity: null,
  };
}

function model(elementMeta: ElementMeta[], extra: Partial<ModelFile> = {}, decls: DeclTable | null = null): ModelFile {
  const ids = Object.keys(NODES).map(Number);
  return {
    path: "synthetic.h5",
    sizeBytes: 0,
    neutralVersion: "2.33.0",
    meta: {},
    nodeIds: Float64Array.from(ids),
    nodeCoords: Float64Array.from(ids.flatMap((id) => NODES[id]!)),
    blocks: [],
    physicalGroups: [],
    labels: [],
    opensees: { version: "2.38.0", objects: [], elementMeta, names: [], definitions: { decls } as unknown as Definitions },
    sequence: { status: "absent" },
    warnings: [],
    readMs: 0,
    ...extra,
  };
}

/** The summed triangle area, and the z component of each triangle's normal. */
function triangles(p: Float32Array): { area: number; nz: number[] } {
  let area = 0;
  const nz: number[] = [];
  for (let i = 0; i < p.length; i += 9) {
    const ux = p[i + 3]! - p[i]!, uy = p[i + 4]! - p[i + 1]!, vx = p[i + 6]! - p[i]!, vy = p[i + 7]! - p[i + 1]!;
    const z = ux * vy - uy * vx;
    area += Math.abs(z) / 2;
    nz.push(z);
  }
  return { area, nz };
}

test("a minted membrane is drawn as its polygon, non-convex outline included", () => {
  const mesh = buildMesh(model([meta("LadrunoMembrane", [["-nodes", ...L_LOOP, "-method", "fan", "-stiff", 3e7, 0.2, 0.15]])]));
  assert.equal(mesh.counts.opsOnly, 1);
  assert.deepEqual(mesh.warnings, []);
  const { area, nz } = triangles(mesh.triPositions);
  assert.ok(Math.abs(area - 3) < 1e-6, `area ${area}`); // a fan from node 1 would overlap and not sum to 3
  assert.ok(nz.every((z) => z > 0), "every triangle winds as the -nodes loop does");
  assert.ok(mesh.triRefs.every((r) => r.kind === "ops" && r.row === 0));
  assert.equal(mesh.edgePositions.length / 6, L_LOOP.length, "only the perimeter is outlined");
  assert.deepEqual(mesh.legend, [{ name: "(OpenSees-only, no cell)", elements: 1 }]);
});

test("a -hole is cut out and outlined, whichever way its loop runs", () => {
  for (const hole of [HOLE, [...HOLE].reverse()]) {
    const mesh = buildMesh(model([meta("LadrunoMembrane", [["-nodes", ...L_LOOP, "-hole", ...hole, "-method", "fan"]])]));
    assert.ok(Math.abs(triangles(mesh.triPositions).area - 2.75) < 1e-6);
    assert.equal(mesh.edgePositions.length / 6, L_LOOP.length + HOLE.length);
  }
});

test("a clockwise -nodes loop and a tilted plane still triangulate", () => {
  const m = model([meta("LadrunoMembrane", [["-nodes", ...[...L_LOOP].reverse()]])]);
  // rotate the plane: (x, y, z) -> (x, z, y)
  const c = m.nodeCoords;
  for (let i = 0; i < c.length; i += 3) [c[i + 1], c[i + 2]] = [c[i + 2]!, c[i + 1]!];
  const p = buildMesh(m).triPositions;
  let area = 0;
  for (let i = 0; i < p.length; i += 9) {
    const u = [p[i + 3]! - p[i]!, p[i + 5]! - p[i + 2]!], v = [p[i + 6]! - p[i]!, p[i + 8]! - p[i + 2]!];
    area += Math.abs(u[0]! * v[1]! - u[1]! * v[0]!) / 2;
  }
  assert.ok(Math.abs(area - 3) < 1e-6, `area ${area}`);
});

test("a pg= panel is not drawn twice: its mesh cells are the drawing", () => {
  const rec = { params: [{ row: 0, type: "LadrunoMembrane", params: { pg: "Slab" }, paramsNames: null, refs: { transf: [], integration: [], section: [] } }] } as unknown as DeclRecord;
  const decls = { hasParams: true, records: [rec], tags: [], rows: {} } as unknown as DeclTable;
  const m = model([meta("LadrunoMembrane", [["-nodes", 1, 2, 3, 6]])], {
    blocks: [{ alias: "quad4", code: 3, dim: 2, npe: 4, ids: new Float64Array([9]), connectivity: new Float64Array([1, 2, 3, 6]) }],
    physicalGroups: [{ name: "Slab", dim: 2, tag: 1, path: "/physical_groups/Slab", elementIds: new Float64Array([9]) }],
  }, decls);
  const mesh = buildMesh(m);
  assert.equal(mesh.counts.opsOnly, 0);
  assert.equal(mesh.triRefs.length, 2);
  assert.ok(mesh.triRefs.every((r) => r.kind === "fem"));
});

test("a minted row with nothing to draw is named, not dropped silently", () => {
  const mesh = buildMesh(model([meta("LadrunoKinematicCoupling", [[1, 2], [3, 4]]), meta("LadrunoMembrane", [["-method", "fan"]])]));
  assert.equal(mesh.counts.opsOnly, 0);
  assert.deepEqual(mesh.warnings, [
    "/opensees/element_meta/LadrunoKinematicCoupling: 2 bridge-minted elements have no connectivity to draw; not drawn",
    "/opensees/element_meta/LadrunoMembrane: 1 bridge-minted element has no connectivity to draw; not drawn",
  ]);
});

test("a node-pair row keeps its inline connectivity, and the inspector gets a membrane's boundary", () => {
  const pair: ElementMeta = { ...meta("zeroLength", [[]]), inlineConnectivity: [[1, 2]] };
  const mesh = buildMesh(model([pair]));
  assert.equal(mesh.counts.opsOnly, 1);
  assert.equal(mesh.lineRefs.length, 1);
  assert.deepEqual(opsOnlyNodes(pair, 0), [1, 2]);
  assert.deepEqual(opsOnlyNodes(meta("LadrunoMembrane", [["-nodes", 1, 2, 3, "-rho", 2.4]]), 0), [1, 2, 3]);
});

test("membraneLoops refuses a loop too short to be a polygon", () => {
  assert.equal(membraneLoops(["-nodes", 1, 2, "-method", "fan"]), null);
  assert.deepEqual(membraneLoops(["-nodes", 1, 2, 3, "-frames", 7, 8, "-hole", 4, 5, 6]), { outer: [1, 2, 3], holes: [[4, 5, 6]] });
});
