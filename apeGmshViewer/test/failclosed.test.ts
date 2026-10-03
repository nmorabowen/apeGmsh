// Fail-closed behaviour on synthetic models: an unknown type, a missing or
// ambiguous tag, a dangling material_ref and a bad schema version are each
// reported by name; none is skipped silently.

import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import * as h5wasm from "h5wasm/node";
import { resolveChain } from "../src/chain/resolve.ts";
import { buildMesh } from "../src/mesh/build.ts";
import type { ElementMeta, ModelFile, OpsObject, Param } from "../src/model/types.ts";
import { readModel, SchemaError, type H5Module } from "../src/reader/read.ts";

function obj(family: OpsObject["family"], dir: string, type: string, tag: number, params: Param[] = [], tables: OpsObject["tables"] = {}): OpsObject {
  const groupName = `${type}_${tag}`;
  return { family, path: `${dir}/${groupName}`, groupName, type, tag, params, attrs: {}, tables };
}

function meta(type: string, args: Param[]): ElementMeta {
  return {
    type,
    path: `/opensees/element_meta/${type}`,
    ids: new Float64Array([7]),
    femEids: new Float64Array([1]),
    args: [args],
    inlineConnectivity: null,
  };
}

function model(objects: OpsObject[], elementMeta: ElementMeta[]): ModelFile {
  return {
    path: "synthetic.h5",
    sizeBytes: 0,
    neutralVersion: "2.33.0",
    meta: {},
    nodeIds: new Float64Array([1, 2]),
    nodeCoords: new Float64Array([0, 0, 0, 1, 0, 0]),
    blocks: [{ alias: "line2", code: 1, dim: 1, npe: 2, ids: new Float64Array([1]), connectivity: new Float64Array([1, 2]) }],
    physicalGroups: [],
    labels: [],
    opensees: { version: "2.21.0", objects, elementMeta, names: [] },
    warnings: [],
    readMs: 0,
  };
}

const T = "/opensees/transforms", BI = "/opensees/beam_integration", S = "/opensees/sections", U = "/opensees/materials/uniaxial";
const ref = { kind: "fem", blockIndex: 0, row: 0 } as const;

const fiber = (matPath: string) =>
  obj("section", S, "Fiber", 1, [], {
    patches: { path: `${S}/Fiber_1/patches`, columns: ["kind", "material_ref"], rows: [["rect", matPath]] },
  });

const complete = () => [
  obj("geomTransf", T, "Linear", 1),
  obj("beamIntegration", BI, "Lobatto", 1, [1, 5]),
  fiber(`${U}/Steel02_1`),
  obj("uniaxialMaterial", U, "Steel02", 1, [420e6, 2e11, 0.01]),
];

test("a complete synthetic chain resolves with no problems", () => {
  const c = resolveChain(model(complete(), [meta("forceBeamColumn", [1, 1])]), ref);
  assert.deepEqual(c.problems, []);
  assert.equal(c.root.children.length, 2);
});

test("an element type outside the syntax table is reported, not guessed", () => {
  const c = resolveChain(model(complete(), [meta("FooBeam", [1, 1])]), ref);
  assert.deepEqual(c.root.children, []);
  assert.deepEqual(c.problems, [
    "/opensees/element_meta/FooBeam[0]: FooBeam args not decoded: element type FooBeam is not in the syntax table",
  ]);
});

test("a tag that matches no object, or two, is reported", () => {
  const none = resolveChain(model(complete(), [meta("forceBeamColumn", [9, 1])]), ref);
  assert.ok(none.problems.some((p) => p.includes("transfTag 9 matches 0 objects in /opensees/transforms")));
  const two = resolveChain(model([...complete(), obj("geomTransf", T, "PDelta", 1)], [meta("forceBeamColumn", [1, 1])]), ref);
  assert.ok(two.problems.some((p) => p.includes("transfTag 1 matches 2 objects")));
});

test("an unknown beamIntegration type is reported", () => {
  const objs = complete();
  objs[1] = obj("beamIntegration", BI, "UserHinge", 1, [1, 2, 3]);
  const c = resolveChain(model(objs, [meta("forceBeamColumn", [1, 1])]), ref);
  assert.ok(c.problems.some((p) => p.includes("beamIntegration type UserHinge is not in the syntax table")));
});

test("a dangling material_ref is reported", () => {
  const objs = complete();
  objs[2] = fiber(`${U}/Missing_9`);
  const c = resolveChain(model(objs, [meta("forceBeamColumn", [1, 1])]), ref);
  assert.ok(c.problems.some((p) => p.includes(`material_ref ${U}/Missing_9 does not exist`)));
});

test("an element without an element_meta row says so", () => {
  const c = resolveChain(model(complete(), []), ref);
  assert.deepEqual(c.problems, ["FEM element 1 has no row in /opensees/element_meta/*/fem_eids"]);
});

test("mesh: two hexes sharing a face draw 10 boundary quads and 20 outline edges", () => {
  // unit cubes [0,1]x[0,1]x[0,1] and [1,2]x[0,1]x[0,1], gmsh hex8 ordering
  const xs = [0, 1, 2];
  const ids: number[] = [], xyz: number[] = [];
  const nid = (i: number, j: number, k: number) => 1 + i + 3 * j + 6 * k;
  for (let k = 0; k < 2; k++) for (let j = 0; j < 2; j++) for (let i = 0; i < 3; i++) {
    ids.push(nid(i, j, k));
    xyz.push(xs[i]!, j, k);
  }
  const hex = (i: number) => [nid(i, 0, 0), nid(i + 1, 0, 0), nid(i + 1, 1, 0), nid(i, 1, 0), nid(i, 0, 1), nid(i + 1, 0, 1), nid(i + 1, 1, 1), nid(i, 1, 1)];
  const m = model([], []);
  m.opensees = null;
  m.nodeIds = new Float64Array(ids);
  m.nodeCoords = new Float64Array(xyz);
  m.blocks = [
    { alias: "hex8", code: 5, dim: 3, npe: 8, ids: new Float64Array([1, 2]), connectivity: new Float64Array([...hex(0), ...hex(1)]) },
    { alias: "weird", code: 999, dim: 3, npe: 4, ids: new Float64Array([3]), connectivity: new Float64Array([1, 2, 3, 4]) },
  ];
  const mesh = buildMesh(m);
  assert.equal(mesh.triRefs.length, 2 * 10);
  // Euler on the closed surface: V - E + F = 2 with V = 12, F = 10, so E = 20.
  assert.equal(mesh.edgePositions.length / 6, 12 + 10 - 2);
  assert.deepEqual(mesh.warnings, ["/elements/weird: gmsh type code 999 is not drawn (1 cells)"]);
});

// --- reader: schema versions and required children, on files written by h5wasm

await h5wasm.ready;
const dir = mkdtempSync(join(tmpdir(), "agv-test-"));

function writeFile(name: string, neutral: string, withNodes = true): string {
  const path = join(dir, name);
  const f = new h5wasm.File(path, "w");
  const meta = f.create_group("meta");
  meta.create_attribute("neutral_schema_version", neutral);
  if (withNodes) {
    const nodes = f.create_group("nodes");
    nodes.create_dataset({ name: "ids", data: new BigInt64Array([1n, 2n]) });
    nodes.create_dataset({ name: "coords", data: new Float64Array([0, 0, 0, 1, 0, 0]), shape: [2, 3] });
  }
  f.close();
  return path;
}

const read = (p: string) => readModel(h5wasm as unknown as H5Module, p, 0);

test("reader: a file inside the window reads with no warning", () => {
  const m = read(writeFile("ok.h5", "2.33.0"));
  assert.equal(m.nodeIds.length, 2);
  assert.deepEqual(m.warnings, []);
});

test("reader: another major version is refused", () => {
  assert.throws(() => read(writeFile("major.h5", "3.0.0")), SchemaError);
});

test("reader: a pre-2.10 layout is refused", () => {
  assert.throws(() => read(writeFile("old.h5", "2.9.0")), /layouts before 2.10/);
});

test("reader: a minor outside the window is a loud warning", () => {
  const m = read(writeFile("minor.h5", "2.30.0"));
  assert.equal(m.warnings.length, 1);
  assert.match(m.warnings[0]!, /neutral_schema_version 2.30.0 is outside the reader window 2.32-2.33/);
});

test("reader: a missing required group raises, naming it", () => {
  assert.throws(() => read(writeFile("nonodes.h5", "2.33.0", false)), /\/nodes is missing/);
});
