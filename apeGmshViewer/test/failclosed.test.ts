// Fail-closed behaviour on synthetic models: an unknown type, a missing or
// ambiguous tag, a dangling material_ref and a bad schema version are each
// reported by name; none is skipped silently. Schema versions follow ADR 0113
// (a floor per zone; the app opens a newer same-major minor with one banner).

import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import * as h5wasm from "h5wasm/node";
import { resolveChain } from "../src/chain/resolve.ts";
import { buildMesh } from "../src/mesh/build.ts";
import type { ElementMeta, ModelFile, OpsObject, Param } from "../src/model/types.ts";
import { parseRefusal } from "../src/effects.ts";
import { readGeometry } from "../src/reader/geometry.ts";
import { readProvenance } from "../src/reader/provenance.ts";
import {
  GEOMETRY_TARGET,
  NEUTRAL_TARGET,
  OPENSEES_TARGET,
  PROVENANCE_TARGET,
  readModel,
  SchemaError,
  ZONE_FLOOR,
  type H5Module,
} from "../src/reader/read.ts";

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

function writeFile(name: string, neutral: string, withNodes = true, opensees: string | null = null): string {
  const path = join(dir, name);
  const f = new h5wasm.File(path, "w");
  const meta = f.create_group("meta");
  meta.create_attribute("neutral_schema_version", neutral);
  if (opensees !== null) {
    meta.create_attribute("opensees_schema_version", opensees);
    f.create_group("opensees");
  }
  if (withNodes) {
    const nodes = f.create_group("nodes");
    nodes.create_dataset({ name: "ids", data: new BigInt64Array([1n, 2n]) });
    nodes.create_dataset({ name: "coords", data: new Float64Array([0, 0, 0, 1, 0, 0]), shape: [2, 3] });
  }
  f.close();
  return path;
}

const read = (p: string) => readModel(h5wasm as unknown as H5Module, p, 0);
const v = (t: { major: number }, minor: number) => `${t.major}.${minor}.0`;

// ADR 0113 D7, per zone: another major is refused; a minor below the floor is
// refused, naming the floor, and reads as "written by an older apeGmsh"; from
// the floor to the target opens with no banner; a newer minor of the same
// major opens with exactly one banner naming the file's stamp and the target.

test("ADR 0113 floors: the app's table (neutral 2.10, opensees 2.11, geometry 1.0, provenance 1.0)", () => {
  assert.deepEqual(ZONE_FLOOR, { neutral: 10, opensees: 11, geometry: 0, provenance: 0 });
});

test("neutral: the floor and an older minor above it open with no banner", () => {
  for (const minor of [ZONE_FLOOR.neutral, NEUTRAL_TARGET.minor - 5, NEUTRAL_TARGET.minor]) {
    const m = read(writeFile(`n${minor}.h5`, v(NEUTRAL_TARGET, minor)));
    assert.equal(m.nodeIds.length, 2);
    assert.deepEqual(m.warnings, [], `neutral ${minor}`);
  }
});

test("neutral: below the floor is refused, naming it, as an older apeGmsh", () => {
  const stamp = v(NEUTRAL_TARGET, ZONE_FLOOR.neutral - 1);
  assert.throws(() => read(writeFile("n-old.h5", stamp)), (e: unknown) => {
    const r = parseRefusal((e as Error).message);
    return e instanceof SchemaError && /layouts before 2\.10 are not supported/.test((e as Error).message) && r?.newer === false && r.version === stamp;
  });
});

test("neutral: another major is refused", () => {
  assert.throws(() => read(writeFile("n-major.h5", "3.0.0")), /neutral_schema_version 3\.0\.0: this app reads major 2 only/);
});

test("neutral: a newer minor opens with exactly one banner naming the stamp and the target", () => {
  const m = read(writeFile("n-new.h5", v(NEUTRAL_TARGET, NEUTRAL_TARGET.minor + 2)));
  assert.deepEqual(m.warnings, [
    `neutral_schema_version ${v(NEUTRAL_TARGET, NEUTRAL_TARGET.minor + 2)} is newer than this app (${NEUTRAL_TARGET.major}.${NEUTRAL_TARGET.minor}.x): the file opens, and what that apeGmsh added is not shown`,
  ]);
});

test("opensees: below the floor (2.10, before the rank flip) is refused, naming it", () => {
  const stamp = v(OPENSEES_TARGET, ZONE_FLOOR.opensees - 1);
  assert.throws(() => read(writeFile("o-old.h5", v(NEUTRAL_TARGET, NEUTRAL_TARGET.minor), true, stamp)), (e: unknown) => {
    const r = parseRefusal((e as Error).message);
    return /opensees_schema_version 2\.10\.0: layouts before 2\.11 are not supported/.test((e as Error).message) && r?.zone === "opensees" && r.newer === false;
  });
});

test("opensees: another major is refused through a real read", () => {
  const n = v(NEUTRAL_TARGET, NEUTRAL_TARGET.minor);
  assert.throws(() => read(writeFile("o-major.h5", n, true, "3.0.0")), (e: unknown) => {
    const r = parseRefusal((e as Error).message);
    return e instanceof SchemaError && /opensees_schema_version 3\.0\.0: this app reads major 2 only/.test((e as Error).message) && r?.zone === "opensees" && r.newer === true;
  });
});

test("opensees: the floor opens silently; a newer minor opens with one banner", () => {
  const n = v(NEUTRAL_TARGET, NEUTRAL_TARGET.minor);
  assert.deepEqual(read(writeFile("o-floor.h5", n, true, v(OPENSEES_TARGET, ZONE_FLOOR.opensees))).warnings, []);
  const w = read(writeFile("o-new.h5", n, true, v(OPENSEES_TARGET, OPENSEES_TARGET.minor + 1))).warnings;
  assert.equal(w.length, 1);
  assert.match(w[0]!, /^opensees_schema_version 2\.22\.0 is newer than this app \(2\.21\.x\)/);
});

/** A file holding only the zone's stamp and an empty zone group: the version is checked first. */
function zoneFile(name: string, key: string, group: string, stamp: string): string {
  const path = join(dir, name);
  const f = new h5wasm.File(path, "w");
  f.create_group("meta").create_attribute(key, stamp);
  f.create_group(group);
  f.close();
  return path;
}

for (const [zone, target, readZone] of [
  ["geometry", GEOMETRY_TARGET, (p: string) => readGeometry(h5wasm as unknown as H5Module, p)],
  ["provenance", PROVENANCE_TARGET, (p: string) => readProvenance(h5wasm as unknown as H5Module, p)],
] as const) {
  test(`${zone}: below the floor (the previous major, the floor being ${target.major}.0) is refused as an older apeGmsh`, () => {
    const stamp = `${target.major - 1}.9.0`;
    assert.throws(() => readZone(zoneFile(`${zone}-old.h5`, `${zone}_schema_version`, zone, stamp)), (e: unknown) => {
      const r = parseRefusal((e as Error).message);
      return e instanceof SchemaError && r?.zone === zone && r.newer === false && r.version === stamp;
    });
  });
  test(`${zone}: another, newer major is refused`, () => {
    assert.throws(
      () => readZone(zoneFile(`${zone}-major.h5`, `${zone}_schema_version`, zone, `${target.major + 1}.0.0`)),
      new RegExp(`${zone}_schema_version ${target.major + 1}\\.0\\.0: this app reads major ${target.major} only`),
    );
  });
}

test("reader: a missing required group raises, naming it", () => {
  assert.throws(() => read(writeFile("nonodes.h5", "2.33.0", false)), /\/nodes is missing/);
});
