// Read the /geometry zone of a `<stem>.geometry.h5` sibling (ADR 0112 D2a),
// and decide whether it pairs with a model file.
//
// Layout and rules: architecture/h5-schema.md, "/geometry" and
// "/meta/session_id and the geometry sibling". This module keeps the same
// rules as read.ts, and adds the zone's own:
// - an absent zone (no version key and no group) is ignored: `null`;
// - a version key without its group, or a group without its key, is a broken
//   file and raises;
// - a known zone outside its version window is refused (ADR 0023 INV-2);
// - the zone stores no int64 (the integer policy), so an int64 column raises
//   instead of arriving as BigInt;
// - every offset, index and enumeration is checked; a bad one raises,
//   naming its HDF5 path.
//
// Browser-safe: no Node import, so the renderer can use it too.

import { checkZoneVersion, GEOMETRY_TARGET, readAttrs, SchemaError, ZONE_FLOOR, type H5Dataset, type H5File, type H5Group, type H5Module } from "./read.ts";
import type { Param } from "../model/types.ts";

// The target and the floor live in read.ts's one version table (#1303).
export { GEOMETRY_TARGET } from "./read.ts";

// ---------------------------------------------------------------------------
// Zone helpers, shared with provenance.ts
// ---------------------------------------------------------------------------

/**
 * The zone's version from `/meta/<key>`, or `null` when the key is absent
 * (an absent zone is ignored). A present stamp follows ADR 0113 D7, the one
 * rule every zone shares (read.ts checkZoneVersion).
 */
export function zoneVersion(
  meta: Record<string, Param | Param[]>,
  key: string,
  target: { major: number; minor: number },
  floor: number,
  warnings: string[],
): string | null {
  const raw = meta[key];
  if (raw === undefined) return null;
  return checkZoneVersion(raw, key, target, floor, warnings);
}

export function has(g: H5Group, name: string): boolean {
  return g.keys().includes(name);
}

function joinPath(parent: string, name: string): string {
  return parent === "/" ? `/${name}` : `${parent}/${name}`;
}

export function group(h5: H5Module, parent: H5Group, name: string): H5Group {
  if (!has(parent, name)) throw new SchemaError(`${joinPath(parent.path, name)} is missing`);
  const node = parent.get(name);
  if (!(node instanceof h5.Group)) {
    throw new SchemaError(`${joinPath(parent.path, name)} is listed but is not a readable group`);
  }
  return node as H5Group;
}

export function dataset(h5: H5Module, parent: H5Group, name: string): H5Dataset {
  if (!has(parent, name)) throw new SchemaError(`${joinPath(parent.path, name)} is missing`);
  const node = parent.get(name);
  if (!(node instanceof h5.Dataset)) {
    throw new SchemaError(`${joinPath(parent.path, name)} is listed but is not a readable dataset`);
  }
  return node as H5Dataset;
}

function shaped(ds: H5Dataset, cols: number | null): number {
  const shape = ds.shape ?? [];
  if (cols === null) {
    if (shape.length !== 1) throw new SchemaError(`${ds.path} has shape (${shape.join(",")}); expected (N,)`);
  } else if (shape.length !== 2 || shape[1] !== cols) {
    throw new SchemaError(`${ds.path} has shape (${shape.join(",")}); expected (N, ${cols})`);
  }
  return shape[0]!;
}

function typed<T>(ds: H5Dataset, ctor: { new (n: number): T; name: string }, what: string): T {
  const v = ds.value;
  if (v instanceof BigInt64Array || v instanceof BigUint64Array) {
    throw new SchemaError(`${ds.path} is int64; the ADR 0112 zones store no int64 (expected ${what})`);
  }
  if (!(v instanceof (ctor as unknown as abstract new (...a: never[]) => unknown))) {
    throw new SchemaError(`${ds.path} is ${(v as object | null)?.constructor?.name ?? typeof v}; expected ${what}`);
  }
  return v as T;
}

/** A 1-D int32 column. */
export function int32s(ds: H5Dataset): Int32Array {
  shaped(ds, null);
  return typed(ds, Int32Array, "int32");
}

/** A 1-D int8 column. */
export function int8s(ds: H5Dataset): Int8Array {
  shaped(ds, null);
  return typed(ds, Int8Array, "int8");
}

/** A float64 table of `cols` columns, flat row-major. */
export function float64s(ds: H5Dataset, cols: number): Float64Array {
  shaped(ds, cols);
  return typed(ds, Float64Array, "float64");
}

/** A 1-D variable-length UTF-8 string column. */
export function strs(ds: H5Dataset): string[] {
  const n = shaped(ds, null);
  const v = ds.json_value;
  if (n === 0) return [];
  if (Array.isArray(v) && v.every((x) => typeof x === "string")) return v as string[];
  throw new SchemaError(`${ds.path} is not a string column`);
}

export function sameLength(where: string, cols: Record<string, { length: number }>): number {
  const lens = Object.entries(cols).map(([k, c]) => `${k}=${c.length}`);
  const n = Object.values(cols)[0]!.length;
  if (Object.values(cols).some((c) => c.length !== n)) {
    throw new SchemaError(`${where}: column lengths differ (${lens.join(", ")})`);
  }
  return n;
}

function oneOf<T extends string>(v: Param | Param[] | undefined, allowed: readonly T[], where: string): T {
  if (typeof v !== "string" || !(allowed as readonly string[]).includes(v)) {
    throw new SchemaError(`${where} = ${JSON.stringify(v)}; expected one of ${allowed.join(", ")}`);
  }
  return v as T;
}

function numAttr(v: Param | Param[] | undefined, where: string): number {
  if (typeof v !== "number") throw new SchemaError(`${where} is not a number`);
  return v;
}

// ---------------------------------------------------------------------------
// The /geometry zone
// ---------------------------------------------------------------------------

/** Concatenated rows plus offsets: row i is `[offsets[i], offsets[i+1])`. */
export interface Csr {
  /** Row of `entities` for each row. */
  entity: Int32Array;
  offsets: Int32Array;
  /** (V, 3) flat. */
  vertices: Float64Array;
}

export interface GeometryZone {
  /** The file read. */
  path: string;
  version: string;
  /** `/meta/session_id`, or null when the file has none. */
  sessionId: string | null;
  source: "mesh" | "temp_mesh";
  status: "ok" | "partial";
  gmshVersion: string;
  curveSamples: number;
  lodSize: number;
  bbox: number[];
  entities: { dim: Int8Array; tag: Int32Array; bbox: Float64Array; ok: Int8Array };
  points: { entity: Int32Array; xyz: Float64Array };
  curves: Csr;
  surfaces: Csr & { triangleOffsets: Int32Array; triangles: Int32Array };
  volumes: { entity: Int32Array; faceOffsets: Int32Array; faces: Int32Array };
  memberships: { dim: Int8Array; tag: Int32Array; kind: ("label" | "physical_group")[]; name: string[]; pg: Int32Array };
  warnings: string[];
}

/** Read `/geometry` from an open file; `null` when the file has no such zone. */
export function readGeometryZone(h5: H5Module, f: H5File, path: string): GeometryZone | null {
  const warnings: string[] = [];
  const meta = has(f, "meta") ? readAttrs(group(h5, f, "meta")) : {};
  const version = zoneVersion(meta, "geometry_schema_version", GEOMETRY_TARGET, ZONE_FLOOR.geometry, warnings);
  const present = has(f, "geometry");
  if (version === null && !present) return null;
  if (version === null) throw new SchemaError(`/geometry is present but /meta has no geometry_schema_version`);
  if (!present) throw new SchemaError(`/meta/geometry_schema_version is ${version} but /geometry is missing`);

  const sid = meta["session_id"];
  if (sid !== undefined && typeof sid !== "string") throw new SchemaError(`/meta/session_id is not a string`);

  const g = group(h5, f, "geometry");
  const a = readAttrs(g);
  const bbox = a["bbox"];
  if (!Array.isArray(bbox) || bbox.length !== 6 || !bbox.every((x) => typeof x === "number")) {
    throw new SchemaError(`/geometry@bbox is not 6 numbers`);
  }
  const gmshVersion = a["gmsh_version"];
  if (typeof gmshVersion !== "string") throw new SchemaError(`/geometry@gmsh_version is not a string`);

  const sub = (name: string) => group(h5, g, name);
  const ds = (parent: H5Group, name: string) => dataset(h5, parent, name);

  const eg = sub("entities");
  const entities = {
    dim: int8s(ds(eg, "dim")),
    tag: int32s(ds(eg, "tag")),
    bbox: float64s(ds(eg, "bbox"), 6),
    ok: int8s(ds(eg, "ok")),
  };
  const K = sameLength(eg.path, { dim: entities.dim, tag: entities.tag, ok: entities.ok, bbox: { length: entities.bbox.length / 6 } });
  // Strict where the spec implies an invariant: dims are 0..3, `ok` is a
  // flag, and every model entity is listed once.
  const listed = new Set<string>();
  for (let k = 0; k < K; k++) {
    const d = entities.dim[k]!;
    if (d < 0 || d > 3) throw new SchemaError(`${eg.path}/dim[${k}] = ${d}; expected 0..3`);
    const ok = entities.ok[k]!;
    if (ok !== 0 && ok !== 1) throw new SchemaError(`${eg.path}/ok[${k}] = ${ok}; expected 0 or 1`);
    const key = `${d}:${entities.tag[k]}`;
    if (listed.has(key)) throw new SchemaError(`${eg.path} lists entity (dim ${d}, tag ${entities.tag[k]}) twice`);
    listed.add(key);
  }

  const pg = sub("points");
  const points = { entity: int32s(ds(pg, "entity")), xyz: float64s(ds(pg, "xyz"), 3) };
  sameLength(pg.path, { entity: points.entity, xyz: { length: points.xyz.length / 3 } });
  checkEntities(pg.path, points.entity, entities.dim, 0);

  const cg = sub("curves");
  const curves: Csr = {
    entity: int32s(ds(cg, "entity")),
    offsets: int32s(ds(cg, "vertex_offsets")),
    vertices: float64s(ds(cg, "vertices"), 3),
  };
  checkEntities(cg.path, curves.entity, entities.dim, 1);
  checkOffsets(`${cg.path}/vertex_offsets`, curves.offsets, curves.entity.length, curves.vertices.length / 3);

  const sg = sub("surfaces");
  const surfaces = {
    entity: int32s(ds(sg, "entity")),
    offsets: int32s(ds(sg, "vertex_offsets")),
    vertices: float64s(ds(sg, "vertices"), 3),
    triangleOffsets: int32s(ds(sg, "triangle_offsets")),
    triangles: typed(ds(sg, "triangles"), Int32Array, "int32"),
  };
  shaped(ds(sg, "triangles"), 3);
  const S = surfaces.entity.length;
  checkEntities(sg.path, surfaces.entity, entities.dim, 2);
  checkOffsets(`${sg.path}/vertex_offsets`, surfaces.offsets, S, surfaces.vertices.length / 3);
  checkOffsets(`${sg.path}/triangle_offsets`, surfaces.triangleOffsets, S, surfaces.triangles.length / 3);
  for (let s = 0; s < S; s++) {
    const nv = surfaces.offsets[s + 1]! - surfaces.offsets[s]!;
    for (let t = surfaces.triangleOffsets[s]! * 3; t < surfaces.triangleOffsets[s + 1]! * 3; t++) {
      const i = surfaces.triangles[t]!;
      if (i < 0 || i >= nv) {
        throw new SchemaError(
          `${sg.path}/triangles row ${Math.floor(t / 3)} indexes vertex ${i} of surface row ${s}, which has ${nv}`,
        );
      }
    }
  }

  const vg = sub("volumes");
  const volumes = {
    entity: int32s(ds(vg, "entity")),
    faceOffsets: int32s(ds(vg, "face_offsets")),
    faces: int32s(ds(vg, "faces")),
  };
  checkEntities(vg.path, volumes.entity, entities.dim, 3);
  checkOffsets(`${vg.path}/face_offsets`, volumes.faceOffsets, volumes.entity.length, volumes.faces.length);
  checkRange(`${vg.path}/faces`, volumes.faces, S, "a row of surfaces");

  const mg = sub("memberships");
  const kinds = strs(ds(mg, "kind"));
  const memberships = {
    dim: int8s(ds(mg, "dim")),
    tag: int32s(ds(mg, "tag")),
    kind: kinds.map((k, i) => oneOf(k, ["label", "physical_group"] as const, `${mg.path}/kind[${i}]`)),
    name: strs(ds(mg, "name")),
    pg: int32s(ds(mg, "pg")),
  };
  sameLength(mg.path, memberships);
  for (let i = 0; i < kinds.length; i++) {
    const pg = memberships.pg[i]!;
    if (memberships.kind[i] === "label" && pg !== -1) {
      throw new SchemaError(`${mg.path}/pg[${i}] = ${pg} for a label; expected -1`);
    }
    // A physical group's tag is a gmsh physical tag: positive.
    if (memberships.kind[i] === "physical_group" && pg < 1) {
      throw new SchemaError(`${mg.path}/pg[${i}] = ${pg} for a physical_group; expected its tag (>= 1)`);
    }
    // One row per (entity, label or group): the entity must be listed.
    const key = `${memberships.dim[i]}:${memberships.tag[i]}`;
    if (!listed.has(key)) {
      throw new SchemaError(`${mg.path} row ${i} names entity (dim ${memberships.dim[i]}, tag ${memberships.tag[i]}), which /geometry/entities does not list`);
    }
  }

  const status = oneOf(a["status"], ["ok", "partial"] as const, "/geometry@status");
  const failed = entities.ok.reduce((n, v) => n + (v === 0 ? 1 : 0), 0);
  if (failed > 0 && status === "ok") {
    throw new SchemaError(`/geometry@status is ok but ${failed} of ${K} entities have ok = 0`);
  }
  if (failed === 0 && status === "partial") warnings.push(`/geometry@status is partial but every entity has ok = 1`);

  return {
    path,
    version,
    sessionId: sid ?? null,
    source: oneOf(a["source"], ["mesh", "temp_mesh"] as const, "/geometry@source"),
    status,
    gmshVersion,
    curveSamples: numAttr(a["curve_samples"], "/geometry@curve_samples"),
    lodSize: numAttr(a["lod_size"], "/geometry@lod_size"),
    bbox: bbox as number[],
    entities,
    points,
    curves,
    surfaces,
    volumes,
    memberships,
    warnings,
  };
}

/** Open `path` and read its `/geometry` zone (see readGeometryZone). */
export function readGeometry(h5: H5Module, path: string): GeometryZone | null {
  const f = new h5.File(path, "r");
  try {
    return readGeometryZone(h5, f, path);
  } finally {
    f.close();
  }
}

function checkRange(where: string, v: Int32Array, n: number, what: string): void {
  for (let i = 0; i < v.length; i++) {
    if (v[i]! < 0 || v[i]! >= n) throw new SchemaError(`${where}[${i}] = ${v[i]} is not ${what} (0..${n - 1})`);
  }
}

function checkEntities(where: string, entity: Int32Array, dims: Int8Array, dim: number): void {
  checkRange(`${where}/entity`, entity, dims.length, "a row of entities");
  for (let i = 0; i < entity.length; i++) {
    if (dims[entity[i]!] !== dim) {
      throw new SchemaError(`${where}/entity[${i}] = ${entity[i]} is an entity of dim ${dims[entity[i]!]}; expected ${dim}`);
    }
  }
}

/** `rows + 1` offsets from 0, non-decreasing, ending at `total`. */
function checkOffsets(where: string, offsets: Int32Array, rows: number, total: number): void {
  if (offsets.length !== rows + 1) throw new SchemaError(`${where} has ${offsets.length} values for ${rows} rows; expected ${rows + 1}`);
  if (offsets[0] !== 0) throw new SchemaError(`${where}[0] = ${offsets[0]}; expected 0`);
  for (let i = 1; i < offsets.length; i++) {
    if (offsets[i]! < offsets[i - 1]!) throw new SchemaError(`${where} decreases at ${i}`);
  }
  if (offsets[rows] !== total) throw new SchemaError(`${where} ends at ${offsets[rows]}; the data has ${total} rows`);
}

// ---------------------------------------------------------------------------
// Pairing with the model
// ---------------------------------------------------------------------------

export type GeometryPairing = { paired: true } | { paired: false; reason: string };

/**
 * Whether `geometry` is this model's geometry: both carry `/meta/session_id`
 * and the two are equal. Otherwise the reason, for the stale-geometry notice;
 * such a sibling must not be drawn as this model's geometry.
 */
export function pairGeometry(modelMeta: Record<string, Param | Param[]>, geometry: GeometryZone): GeometryPairing {
  return pairSessions(modelMeta["session_id"], geometry.sessionId, geometry.path);
}

/** The pairing rule on the two ids alone (the store keeps the ids, not the files). */
export function pairSessions(modelSid: unknown, geometrySid: string | null, geometryPath: string): GeometryPairing {
  if (modelSid === undefined || modelSid === null) {
    return { paired: false, reason: "the model file has no session_id (written before apeGmsh stamped one), so no geometry pairs with it" };
  }
  if (typeof modelSid !== "string") return { paired: false, reason: "the model file's session_id is not a string" };
  if (geometrySid === null) return { paired: false, reason: `${geometryPath} has no session_id` };
  if (geometrySid !== modelSid) {
    return {
      paired: false,
      reason: `${geometryPath} is stale or foreign: it was written by session ${geometrySid}, the model by session ${modelSid}`,
    };
  }
  return { paired: true };
}
