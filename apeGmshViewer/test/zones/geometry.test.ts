// The /geometry reader (src/reader/geometry.ts).
//
// Oracles:
// - fixtures/zones.geometry.h5, written by h5py to h5-schema.md (a unit cube):
//   closed forms of the cube from the arrays the reader returns (total face
//   area 6, enclosed volume 1 by the divergence theorem, edge length 12);
// - session_id pairing against fixtures/zones.h5 (same session), and against
//   synthetic files with another session or none;
// - fail-closed cases on synthetic files written here with h5wasm, each
//   raising and naming the broken path.

import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { GEOMETRY_TARGET, pairGeometry, readGeometry, type GeometryZone } from "../../src/reader/geometry.ts";
import { readModel, SchemaError, type H5Module } from "../../src/reader/read.ts";
import { tree, writeTree, type Tree } from "./tree.ts";

await h5wasm.ready;
const h5 = h5wasm as unknown as H5Module;
const fixtures = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "fixtures");
const tmp = mkdtempSync(join(tmpdir(), "agv-geometry-"));
after(() => rmSync(tmp, { recursive: true, force: true }));

const xyz = (v: Float64Array, i: number) => [v[3 * i]!, v[3 * i + 1]!, v[3 * i + 2]!] as const;
const sub = (a: readonly number[], b: readonly number[]) => [a[0]! - b[0]!, a[1]! - b[1]!, a[2]! - b[2]!];
const cross = (a: number[], b: number[]) => [a[1]! * b[2]! - a[2]! * b[1]!, a[2]! * b[0]! - a[0]! * b[2]!, a[0]! * b[1]! - a[1]! * b[0]!];
const dot = (a: readonly number[], b: readonly number[]) => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;

/** Each triangle of surface row `s`, as three points. */
type P = readonly [number, number, number];
function* triangles(g: GeometryZone, s: number): Generator<[P, P, P]> {
  const { offsets, vertices, triangleOffsets, triangles: tri } = g.surfaces;
  const base = offsets[s]!;
  for (let t = triangleOffsets[s]!; t < triangleOffsets[s + 1]!; t++) {
    const at = (k: number) => xyz(vertices, base + tri[3 * t + k]!);
    yield [at(0), at(1), at(2)];
  }
}

test("the h5py cube fixture reads, and its closed forms hold", () => {
  const g = readGeometry(h5, join(fixtures, "zones.geometry.h5"))!;
  assert.ok(g);
  assert.equal(g.version, "1.0.0");
  assert.equal(g.sessionId, "0f8e5c1a2b3d4e5f8a9b0c1d2e3f4a5b");
  assert.equal(g.source, "mesh");
  assert.equal(g.status, "ok");
  assert.equal(g.curveSamples, 32);
  assert.deepEqual(g.bbox, [0, 0, 0, 1, 1, 1]);
  assert.deepEqual(g.warnings, []);
  assert.equal(g.entities.tag.length, 27);
  assert.deepEqual([g.points.entity.length, g.curves.entity.length, g.surfaces.entity.length, g.volumes.entity.length], [8, 12, 6, 1]);

  // Total area of the six faces: 6.
  let area = 0;
  for (let s = 0; s < 6; s++) for (const [a, b, c] of triangles(g, s)) area += Math.hypot(...cross(sub(b, a), sub(c, a))) / 2;
  assert.ok(Math.abs(area - 6) < 1e-12, `area ${area}`);

  // Volume of the cube from its faces (the volume's face rows): 1, positive,
  // so the faces point outward.
  let vol = 0;
  const v = g.volumes;
  for (let k = v.faceOffsets[0]!; k < v.faceOffsets[1]!; k++) {
    for (const [a, b, c] of triangles(g, v.faces[k]!)) vol += dot(a, cross([...b], [...c])) / 6;
  }
  assert.ok(Math.abs(vol - 1) < 1e-12, `volume ${vol}`);

  // Edge lengths: 12 unit edges, 32 samples each.
  let len = 0;
  const c = g.curves;
  for (let r = 0; r < c.entity.length; r++) {
    assert.equal(c.offsets[r + 1]! - c.offsets[r]!, 32);
    for (let i = c.offsets[r]! + 1; i < c.offsets[r + 1]!; i++) len += Math.hypot(...sub(xyz(c.vertices, i), xyz(c.vertices, i - 1)));
  }
  assert.ok(Math.abs(len - 12) < 1e-12, `length ${len}`);

  assert.deepEqual(g.memberships.kind, ["label", "physical_group", "physical_group"]);
  assert.deepEqual(g.memberships.name, ["Box", "Bottom", "Top"]);
  assert.deepEqual([...g.memberships.pg], [-1, 5, 6]);
});

test("pairing: the fixture pair shares a session_id", () => {
  const model = readModel(h5, join(fixtures, "zones.h5"), 0);
  const g = readGeometry(h5, join(fixtures, "zones.geometry.h5"))!;
  assert.deepEqual(pairGeometry(model.meta, g), { paired: true });
});

test("pairing: another session, or none, is stale and says why", () => {
  const g = readGeometry(h5, join(fixtures, "zones.geometry.h5"))!;
  const other = pairGeometry({ session_id: "ffff" }, g);
  assert.equal(other.paired, false);
  assert.match((other as { reason: string }).reason, /stale or foreign.*0f8e5c1a.*ffff/);
  assert.match((pairGeometry({}, g) as { reason: string }).reason, /model file has no session_id/);
  const noSid = write("nosid", (t) => {
    delete t.meta!.attrs!["session_id"];
  });
  assert.match((pairGeometry({ session_id: "ffff" }, readGeometry(h5, noSid)!) as { reason: string }).reason, /has no session_id/);
});

test("a file without the zone is ignored", () => {
  assert.equal(readGeometry(h5, join(fixtures, "shoebuckle.h5")), null);
});

// --- synthetic files -------------------------------------------------------

/** A small valid geometry: 1 point, 1 curve, 1 two-triangle surface, 1 volume. */
function valid(): Tree {
  return tree({
    meta: { attrs: { geometry_schema_version: "1.0.0", session_id: "abc" } },
    geometry: {
      attrs: { source: "temp_mesh", gmsh_version: "4.13.1", curve_samples: 32, lod_size: 0.1, bbox: new Float64Array([0, 0, 0, 1, 1, 0]), status: "ok" },
      entities: {
        dim: new Int8Array([0, 1, 2, 3]),
        tag: new Int32Array([1, 1, 1, 1]),
        bbox: { data: new Float64Array(24), shape: [4, 6] },
        ok: new Int8Array([1, 1, 1, 1]),
      },
      points: { entity: new Int32Array([0]), xyz: { data: new Float64Array([0, 0, 0]), shape: [1, 3] } },
      curves: {
        entity: new Int32Array([1]),
        vertex_offsets: new Int32Array([0, 2]),
        vertices: { data: new Float64Array([0, 0, 0, 1, 0, 0]), shape: [2, 3] },
      },
      surfaces: {
        entity: new Int32Array([2]),
        vertex_offsets: new Int32Array([0, 4]),
        vertices: { data: new Float64Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]), shape: [4, 3] },
        triangle_offsets: new Int32Array([0, 2]),
        triangles: { data: new Int32Array([0, 1, 2, 0, 2, 3]), shape: [2, 3] },
      },
      volumes: { entity: new Int32Array([3]), face_offsets: new Int32Array([0, 1]), faces: new Int32Array([0]) },
      memberships: {
        dim: new Int8Array([2]),
        tag: new Int32Array([1]),
        kind: ["physical_group"],
        name: ["Floor"],
        pg: new Int32Array([7]),
      },
    },
  });
}

let n = 0;
function write(name: string, mutate: (t: Tree) => void): string {
  const t = valid();
  mutate(t);
  const path = join(tmp, `${name}-${n++}.geometry.h5`);
  writeTree(h5wasm, path, t);
  return path;
}

test("the synthetic file is valid (the cases below change one thing each)", () => {
  const g = readGeometry(h5, write("valid", () => {}))!;
  assert.equal(g.source, "temp_mesh");
  assert.equal(g.surfaces.triangles.length, 6);
});

const refuse = (mutate: (t: Tree) => void, pattern: RegExp) => () => {
  assert.throws(() => readGeometry(h5, write("bad", mutate)), (e: unknown) => e instanceof SchemaError && pattern.test(e.message));
};

test("another major is refused", refuse((t) => (t.meta!.attrs!["geometry_schema_version"] = `${GEOMETRY_TARGET.major + 1}.0.0`), /geometry_schema_version 2\.0\.0: this app reads major 1 only/));
test("a version that is not X.Y.Z is refused", refuse((t) => (t.meta!.attrs!["geometry_schema_version"] = "1.0"), /is not X\.Y\.Z/));
test("the key without the group is refused", refuse((t) => delete t.geometry, /geometry_schema_version is 1\.0\.0 but \/geometry is missing/));
test("the group without the key is refused", refuse((t) => delete t.meta!.attrs!["geometry_schema_version"], /\/geometry is present but \/meta has no geometry_schema_version/));
test("a missing table is refused", refuse((t) => delete t.geometry!["volumes"], /\/geometry\/volumes is missing/));
test("an int64 column is refused", refuse((t) => ((t.geometry!["entities"] as Tree)["tag"] = new BigInt64Array([1n, 1n, 1n, 1n])), /entities\/tag is int64/));
test("offsets that do not end at the data are refused", refuse((t) => ((t.geometry!["curves"] as Tree)["vertex_offsets"] = new Int32Array([0, 3])), /curves\/vertex_offsets ends at 3; the data has 2 rows/));
test("a triangle outside its surface's vertices is refused", refuse(
  (t) => ((t.geometry!["surfaces"] as Tree)["triangles"] = { data: new Int32Array([0, 1, 2, 0, 2, 4]), shape: [2, 3] }),
  /triangles row 1 indexes vertex 4 of surface row 0, which has 4/,
));
test("an entity row of the wrong dimension is refused", refuse((t) => ((t.geometry!["curves"] as Tree)["entity"] = new Int32Array([2])), /curves\/entity\[0\] = 2 is an entity of dim 2; expected 1/));
test("a volume face outside the surfaces is refused", refuse((t) => ((t.geometry!["volumes"] as Tree)["faces"] = new Int32Array([1])), /volumes\/faces\[0\] = 1 is not a row of surfaces/));
test("an unknown membership kind is refused", refuse((t) => ((t.geometry!["memberships"] as Tree)["kind"] = ["group"]), /memberships\/kind\[0\] = "group"; expected one of label, physical_group/));
test("a label with a pg is refused", refuse((t) => ((t.geometry!["memberships"] as Tree)["kind"] = ["label"]), /pg\[0\] = 7 for a label; expected -1/));
test("an unknown @source is refused", refuse((t) => (t.geometry!.attrs!["source"] = "brep"), /\/geometry@source = "brep"/));
test("status ok with a failed entity is refused", refuse((t) => ((t.geometry!["entities"] as Tree)["ok"] = new Int8Array([1, 0, 1, 1])), /status is ok but 1 of 4 entities have ok = 0/));

test("a newer minor is read with a warning", () => {
  const g = readGeometry(h5, write("newer", (t) => (t.meta!.attrs!["geometry_schema_version"] = "1.4.0")))!;
  assert.equal(g.version, "1.4.0");
  assert.match(g.warnings.join("\n"), /1\.4\.0 is newer than this reader \(1\.0\); fields added since are ignored/);
});
