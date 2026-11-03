// reader/userEntities.ts and the `/entities` zone of reader/userFile.ts (ADR 0130 slice VU8).
//
// The oracles:
// - cross-language: fixtures/entities/*.user.h5 were written by Python's
//   writer (tests/results/user/make_app_entities_fixtures.py), and
//   goldens.json is what Python's reader reads from them;
//   tests/results/user/test_user_entities_app_fixtures.py holds the JSON to
//   that reader, so the app here equals Python on the same bytes;
// - cross-engine: the cut kernel (STKO_to_python.cuts) on the STKO elastic
//   frame, fed the original SectionCutDef, gives goldens.json's F and M; the
//   app reads the cut and the resultant on it from the file and must equal
//   them bit for bit (the card's "a cut's resultant equals the cuts module's");
// - the layout of h5-schema.md "`/entities`", for each way a zone refuses,
//   written here with h5wasm and named with Python's verdict (`read_entities`).

import assert from "node:assert/strict";
import { mkdtempSync, readdirSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { EntityValuesRefused, readEntityValues, readUserFile, readUserSet, type UserH5, type UserIndex } from "../../src/reader/userFile.ts";
import { bindingProblem, entityRow, rowRefusals, type CutEntity, type Entities } from "../../src/reader/userEntities.ts";
import { userInfoOf } from "../../src/state/reduce.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as UserH5;
const FIXTURES = fileURLToPath(new URL("./fixtures/entities/", import.meta.url));
const FRAME_CUT = join(FIXTURES, "frame_cut.user.h5");
const ROWS = join(FIXTURES, "rows.user.h5");
const FEM = "fedcba9876543210fedcba9876543210";
const MODEL = "0123456789abcdef0123456789abcdef";

interface GoldenSide {
  entities: {
    cuts: Omit<CutEntity, never>[];
    storeys: Entities["storeys"];
    pointSets: Entities["pointSets"];
    refused: [string, number, string][];
  };
  entries: string[];
  refused: [string, string][];
}
const GOLDEN = JSON.parse(readFileSync(join(FIXTURES, "goldens.json"), "utf8")) as {
  frame_cut: GoldenSide & {
    kernel: { def: { point: number[]; normal: number[]; elementIds: number[]; side: string }; reference: number[]; components: string[]; states: [string, string, number, number][]; F: number[][]; M: number[][] };
  };
  rows: GoldenSide;
};

const bits = (xs: readonly number[]) => Buffer.from(Float64Array.from(xs).buffer).toString("hex");
function ok<T>(z: { ok: true; value: T } | { ok: false; reason: string } | null): T {
  assert.ok(z !== null, "the zone is absent");
  assert.ok(z.ok, z.ok ? "" : z.reason);
  return z.value;
}

// ---- cross-language: the app reads what Python reads ----------------------

for (const stem of ["frame_cut", "rows"] as const) {
  test(`${stem}.user.h5: the app reads the rows, refusals and index Python reads`, () => {
    const u = readUserFile(H5, join(FIXTURES, `${stem}.user.h5`), null);
    const g = GOLDEN[stem];
    const e = ok(u.entities);
    assert.deepEqual(e.cuts, g.entities.cuts);
    assert.deepEqual(e.storeys, g.entities.storeys);
    assert.deepEqual(e.pointSets, g.entities.pointSets);
    assert.deepEqual(e.refused.map((r) => [r.kind, r.row, r.reason]), g.entities.refused);
    // Bit for bit, not just to the JSON's printing: every coordinate.
    assert.equal(bits(e.cuts.flatMap((c) => [...c.point, ...c.normal, ...c.reference])), bits(g.entities.cuts.flatMap((c) => [...c.point, ...c.normal, ...c.reference])));
    const r = ok(u.results);
    assert.deepEqual(r.entries.map((x) => x.name), g.entries);
    assert.deepEqual(r.refused.map((x) => [x.name, x.reason]), g.refused);
    assert.ok(u.zones.includes("entities"));
  });
}

test("a bad row refuses alone and keeps its position: the rows after it keep their ENTITY_ROW", () => {
  const e = ok(readUserFile(H5, ROWS, null).entities);
  assert.equal(e.cuts.length, 3);
  assert.equal(entityRow(e, "cut", 2).name, "Delta_after (L/480)");
  assert.throws(() => entityRow(e, "cut", 1), /cuts row 1 \(''\) is refused: empty-name/);
  assert.throws(() => entityRow(e, "cut", 0), /refused: bad-param \(cut 'pier P1 base': normal \[0\.0, 0\.0, 2\.0\] is not of unit length\)/);
  assert.throws(() => entityRow(e, "point_set", 0), /a host_element is below -1/);
  assert.throws(() => entityRow(e, "storey", 1), /holds 1 rows, not row 1/);
  const r = ok(readUserFile(H5, ROWS, null).results);
  assert.match(r.refused[0]!.detail, /names cut row 1, which is refused: empty-name/);
  assert.match(r.refused[1]!.detail, /names point_set row 7, and \/entities\/point_sets holds 2/);
});

// ---- cross-engine: the cut and its resultant equal the cut kernel's ------

test("a cut's resultant, read by the app, equals the cut kernel's on the STKO frame", () => {
  const k = GOLDEN.frame_cut.kernel;
  const v = readEntityValues(H5, FRAME_CUT, "frame cut resultant");
  assert.deepEqual(v.components, k.components);
  assert.deepEqual(v.states, k.states.map(([kind, source, step, mode]) => ({ kind, source, step, mode })));
  assert.equal(v.rows.length, 1);
  const [{ kind, row, entity }] = v.rows as [(typeof v.rows)[number]];
  assert.deepEqual([kind, row, entity.name], ["cut", 0, "frame cut"]);
  // The cut is the def the kernel was fed, field for field, and bit for bit.
  const cut = entity as CutEntity;
  assert.equal(bits(cut.point), bits(k.def.point));
  assert.equal(bits(cut.normal), bits(k.def.normal));
  assert.deepEqual(cut.elementIds, k.def.elementIds);
  assert.equal(cut.side, k.def.side);
  assert.equal(bits(cut.reference), bits(k.reference));
  // The resultant: F, and M about the cut's reference, per state.
  const [s, n, c] = v.shape;
  assert.deepEqual([s, n, c], [k.F.length, 1, 6]);
  const at = (i: number, j: number) => v.data[i * n * c + j]!;
  for (let i = 0; i < s; i++) {
    assert.equal(bits([0, 1, 2].map((j) => at(i, j))), bits(k.F[i]!), `F at state ${i}`);
    assert.equal(bits([3, 4, 5].map((j) => at(i, j))), bits(k.M[i]!), `M at state ${i}`);
  }
  // Not a vacuous match: the frame carries load.
  assert.ok(Math.max(...k.F.map((f) => Math.abs(f[2]!))) > 1);
});

test("readEntityValues refuses a result that does not stand, is absent, or is not on entities", () => {
  assert.throws(() => readEntityValues(H5, ROWS, "on the unnamed cut"), (e: unknown) => e instanceof EntityValuesRefused && /is refused: unresolved/.test((e as Error).message));
  assert.throws(() => readEntityValues(H5, ROWS, "nothing"), (e: unknown) => e instanceof EntityValuesRefused && /holds no user result 'nothing'/.test((e as Error).message));
  const v = readEntityValues(H5, ROWS, "on the second point set");
  assert.deepEqual([v.rows[0]!.kind, v.rows[0]!.row, v.rows[0]!.entity.name, v.data], ["point_set", 1, "drift points", [0.25]]);
});

// ---- the rules, alone -------------------------------------------------------

const cutRow = (o: Partial<CutEntity> = {}): CutEntity => ({ name: "c", side: "positive", point: [0, 0, 0], normal: [0, 0, 1], reference: [0, 0, 0], elementIds: [1], polygon: [], ...o });

test("rowRefusals: the name rule, then duplicates (each row), then the kind's rule, as Python's row_refusals", () => {
  const rows = rowRefusals({
    cuts: [
      cutRow({ name: " c" }),
      cutRow({ name: "d" }),
      cutRow({ name: "d" }),
      cutRow({ name: "e", side: "both" }),
      cutRow({ name: "f", elementIds: [] }),
      cutRow({ name: "g", polygon: [[0, 0, 0], [1, 0, 0]] }),
      cutRow({ name: "h", point: [NaN, 0, 0] }),
      cutRow({ name: "i", normal: [0, 0, 1 + 1e-10] }),
    ],
    storeys: [{ name: "s", level: 0, reference: [0, 0, 0], nodeIds: [] }],
    pointSets: [{ name: "p", xyz: [], hostElement: [] }],
  });
  assert.deepEqual(rows.map((r) => [r.kind, r.row, r.reason]), [
    ["cut", 0, "empty-name"],
    ["cut", 1, "duplicate"],
    ["cut", 2, "duplicate"],
    ["cut", 3, "bad-param"],
    ["cut", 4, "bad-param"],
    ["cut", 5, "bad-param"],
    ["cut", 6, "bad-param"],
    ["storey", 0, "bad-param"],
    ["point_set", 0, "bad-param"],
  ]);
  assert.equal(rows[3]!.detail, "cut 'e': side 'both' is not one of ('positive', 'negative')");
  assert.equal(rows[1]!.detail, "2 cuts rows are named 'd'");
});

test("bindingProblem names the row, the table's size, or the refusal", () => {
  const e: Entities = { cuts: [cutRow()], storeys: [], pointSets: [], refused: [{ kind: "cut", row: 0, name: "c", reason: "bad-param", detail: "x" }] };
  assert.equal(bindingProblem(e, "r", ["storey"], [0]), "user result 'r' names storey row 0, and /entities/storeys holds 0");
  assert.equal(bindingProblem(e, "r", ["cut"], [0]), "user result 'r' names cut row 0, which is refused: bad-param");
  assert.equal(bindingProblem({ ...e, refused: [] }, "r", ["cut"], [0]), null);
  assert.match(bindingProblem(e, "r", ["planet"], [0])!, /entity kind 'planet' is not one of \['cut', 'point_set', 'storey'\]/);
});

// ---- a zone that refuses, and a file with none ---------------------------

type G = h5wasm.Group;
const str = (g: G, name: string, data: string[]) => g.create_dataset({ name, data, dtype: "S" });
const i4 = (g: G, name: string, data: number[], shape?: number[]) => g.create_dataset({ name, data: Int32Array.from(data), shape: shape ?? [data.length] });
const f8 = (g: G, name: string, data: number[], shape?: number[]) => g.create_dataset({ name, data: Float64Array.from(data), shape: shape ?? [data.length] });

interface ZoneOpts {
  version?: string | null;
  space?: string | null;
  /** leave out `<table>/<column>`, or a whole table */
  drop?: string;
  /** write `<table>/<column>` this way instead */
  override?: Record<string, (g: G, name: string) => void>;
  /** leave the group out, the key in */
  noGroup?: boolean;
}

/** A user file with one valid `/entities`: a cut, a storey and a point set, unless `o` breaks it. */
function writeZone(o: ZoneOpts = {}): string {
  const d = mkdtempSync(join(tmpdir(), "agv-entities-"));
  const path = join(d, "run.user.h5");
  const f = new h5wasm.File(path, "w");
  try {
    const meta = f.create_group("meta") as G;
    meta.create_attribute("user_meta_schema_version", "1.0.0");
    if (o.version !== null) meta.create_attribute("entities_schema_version", o.version ?? "1.0.0");
    meta.create_attribute("fem_hash", FEM);
    meta.create_attribute("model_hash", MODEL);
    meta.create_attribute("revision", Int32Array.of(1));
    if (o.noGroup) return path;
    const z = f.create_group("entities") as G;
    if (o.space !== null) z.create_attribute("element_index_space", o.space ?? "opensees");
    const cols: Record<string, Record<string, (g: G, n: string) => unknown>> = {
      cuts: {
        name: (g, n) => str(g, n, ["c"]),
        side: (g, n) => str(g, n, ["positive"]),
        point: (g, n) => f8(g, n, [0, 0, 1], [1, 3]),
        normal: (g, n) => f8(g, n, [0, 0, 1], [1, 3]),
        reference: (g, n) => f8(g, n, [0, 0, 1], [1, 3]),
        element_offsets: (g, n) => i4(g, n, [0, 2]),
        element_ids: (g, n) => i4(g, n, [1, 2]),
        polygon_offsets: (g, n) => i4(g, n, [0, 0]),
        polygon: (g, n) => f8(g, n, [], [0, 3]),
      },
      storeys: {
        name: (g, n) => str(g, n, ["L1"]),
        level: (g, n) => f8(g, n, [3]),
        reference: (g, n) => f8(g, n, [1, 1, 3], [1, 3]),
        node_offsets: (g, n) => i4(g, n, [0, 2]),
        node_ids: (g, n) => i4(g, n, [5, 6]),
      },
      point_sets: {
        name: (g, n) => str(g, n, ["pts"]),
        point_offsets: (g, n) => i4(g, n, [0, 1]),
        xyz: (g, n) => f8(g, n, [1, 2, 3], [1, 3]),
        host_element: (g, n) => i4(g, n, [-1]),
      },
    };
    for (const [table, columns] of Object.entries(cols)) {
      if (o.drop === table) continue;
      const g = z.create_group(table) as G;
      for (const [name, write] of Object.entries(columns)) {
        const key = `${table}/${name}`;
        if (o.drop === key) continue;
        (o.override?.[key] ?? write)(g, name);
      }
    }
  } finally {
    f.close();
  }
  return path;
}

test("a valid zone written as h5-schema.md lays it out reads, empty polygon table included", () => {
  const e = ok(readUserFile(H5, writeZone(), null).entities);
  assert.deepEqual(e.cuts, [{ name: "c", side: "positive", point: [0, 0, 1], normal: [0, 0, 1], reference: [0, 0, 1], elementIds: [1, 2], polygon: [] }]);
  assert.deepEqual(e.storeys, [{ name: "L1", level: 3, reference: [1, 1, 3], nodeIds: [5, 6] }]);
  assert.deepEqual(e.pointSets, [{ name: "pts", xyz: [[1, 2, 3]], hostElement: [-1] }]);
  assert.deepEqual(e.refused, []);
});

const refusedWith = (path: string, re: RegExp) => {
  const u = readUserFile(H5, path, null);
  assert.ok(u.entities !== null && !u.entities.ok, "the zone reads");
  assert.match(u.entities.reason, re);
  // A zone that refuses does not refuse the file.
  assert.ok(u.combinations.ok && u.results.ok);
};

test("a missing table, a malformed column or bad offsets refuse the zone, as Python's read_entities", () => {
  refusedWith(writeZone({ drop: "storeys" }), /^refused: \/entities: it has no storeys table$/);
  refusedWith(writeZone({ drop: "cuts/side" }), /\/entities\/cuts has no side column/);
  refusedWith(writeZone({ override: { "cuts/point": (g, n) => f8(g, n, [0, 0, 1]) } }), /\/entities\/cuts\/point is float64 \(3,\), expected f8 \(1, 3\)/);
  refusedWith(writeZone({ override: { "cuts/element_ids": (g, n) => g.create_dataset({ name: n, data: BigInt64Array.of(1n, 2n), shape: [2] }) } }), /element_ids is int64 \(2,\), expected i4 \(None,\)/);
  refusedWith(writeZone({ override: { "storeys/node_offsets": (g, n) => i4(g, n, [0, 3]) } }), /node_offsets \[0, 3\] does not run from 0 to 2/);
  refusedWith(writeZone({ override: { "cuts/element_offsets": (g, n) => i4(g, n, [1, 2]) } }), /does not run from 0 to 2/);
  refusedWith(writeZone({ override: { "point_sets/host_element": (g, n) => i4(g, n, [-1, -1]) } }), /host_element is int32 \(2,\), expected i4 \(1,\)/);
  refusedWith(writeZone({ override: { "cuts/name": (g, n) => i4(g, n, [1]) } }), /name is int32 \(1,\), expected str/);
});

/** A point set of `n` points and a cut polygon of `m` vertices, each coordinate distinct; `lastHost` is the last point's host. */
function largeZone(n: number, m: number, lastHost = -1): { path: string; xyz: number[]; poly: number[] } {
  const xyz = Array.from({ length: 3 * n }, (_, k) => k * 0.5 + 0.125);
  const poly = Array.from({ length: 3 * m }, (_, k) => k % 3 === 2 ? 1 : Math.sin(k) * 10);
  const host = Array.from({ length: n }, (_, k) => (k === n - 1 ? lastHost : k % 7 === 0 ? -1 : k + 1));
  const path = writeZone({
    override: {
      "point_sets/point_offsets": (g, nm) => i4(g, nm, [0, n]),
      "point_sets/xyz": (g, nm) => f8(g, nm, xyz, [n, 3]),
      "point_sets/host_element": (g, nm) => i4(g, nm, host),
      "cuts/polygon_offsets": (g, nm) => i4(g, nm, [0, m]),
      "cuts/polygon": (g, nm) => f8(g, nm, poly, [m, 3]),
    },
  });
  return { path, xyz, poly };
}

test("a large legal zone reads, as Python reads it: a 50,000-point set and a 200,000-vertex polygon (no spread into a call, #2749 review)", () => {
  const { path, xyz, poly } = largeZone(50_000, 200_000);
  const e = ok(readUserFile(H5, path, null).entities);
  assert.deepEqual(e.refused, []);
  assert.equal(e.pointSets[0]!.xyz.length, 50_000);
  assert.equal(bits(e.pointSets[0]!.xyz.flat()), bits(xyz));
  assert.equal(e.cuts[0]!.polygon.length, 200_000);
  assert.equal(bits(e.cuts[0]!.polygon.flat()), bits(poly));
  // The row rules still run over every point: the last host below -1 refuses the row, as Python's `_point_set_problem`.
  const bad = ok(readUserFile(H5, largeZone(50_000, 3, -2).path, null).entities);
  assert.deepEqual(bad.refused.map((r) => [r.kind, r.row, r.reason, r.detail]), [["point_set", 0, "bad-param", "point set 'pts': a host_element is below -1 (-1 = none)"]]);
});

test("an element index space other than OpenSees tags refuses the zone, and so does none", () => {
  refusedWith(writeZone({ space: "fem" }), /@element_index_space is 'fem'; this reader knows 'opensees' only/);
  refusedWith(writeZone({ space: null }), /@element_index_space is None/);
});

test("the version: a newer minor opens with a banner, a newer major refuses the zone, a key with no group refuses it", () => {
  const newer = readUserFile(H5, writeZone({ version: "1.1.0" }), null);
  assert.ok(newer.entities?.ok);
  assert.equal(newer.warnings.length, 1);
  assert.match(newer.warnings[0]!, /entities_schema_version 1\.1\.0/);
  refusedWith(writeZone({ version: "2.0.0" }), /entities_schema_version 2\.0\.0/);
  const keyOnly = readUserFile(H5, writeZone({ noGroup: true }), null);
  assert.equal(keyOnly.entities, null, "the index lists no zone the file does not hold");
  // A group with no key refuses (Python's `_zone_version_checked`).
  refusedWith(writeZone({ version: null }), /\/meta has no version key for zone 'entities'/);
});

test("a file with no /entities lists none (an ON_ENTITIES result there is unresolved: test/reader/userFile.test.ts)", () => {
  const u: UserIndex = readUserFile(H5, writeZone({ version: null, noGroup: true }), null);
  assert.equal(u.entities, null);
  assert.ok(!u.zones.includes("entities"));
});

// ---- staleness and the store --------------------------------------------

function writeModel(fem: string): string {
  const d = mkdtempSync(join(tmpdir(), "agv-entities-model-"));
  const path = join(d, "run.h5");
  const f = new h5wasm.File(path, "w");
  try {
    const lineage = (f.create_group("meta") as G).create_group("lineage") as G;
    lineage.create_attribute("fem_hash", fem);
    lineage.create_attribute("model_hash", MODEL);
  } finally {
    f.close();
  }
  return path;
}

test("/entities goes stale with the model hashes (A6), and the store keeps only the standing rows, by position", () => {
  const list = (d: string) => readdirSync(d);
  const cur = readUserSet(H5, ROWS, writeModel(FEM), null, list);
  assert.ok(cur.ok);
  assert.deepEqual(cur.staleness.zones["entities"], { stale: false, reason: "" });
  const stale = readUserSet(H5, ROWS, writeModel("0".repeat(32)), null, list);
  assert.ok(stale.ok && stale.staleness.zones["entities"]!.stale);

  const info = userInfoOf(cur.user, cur.staleness);
  const e = info.entities;
  assert.ok(e && e.ok);
  assert.deepEqual(e.cuts.map((c) => [c.row, c.name]), [[2, "Delta_after (L/480)"]]);
  assert.deepEqual(e.storeys.map((s) => [s.row, s.name]), [[0, "L1"]]);
  assert.deepEqual(e.pointSets.map((p) => [p.row, p.name]), [[1, "drift points"]]);
  assert.deepEqual(e.refused.map((r) => [r.kind, r.row, r.reason]), GOLDEN.rows.entities.refused);
  assert.deepEqual(e.stale, { stale: false, reason: "" });
  assert.ok(userInfoOf(stale.user, stale.staleness).entities!.stale.stale);
});

test("userInfoOf fails closed on an entity kind it does not know, and reads an index with no entities as none", () => {
  const cur = readUserSet(H5, ROWS, null, null, (d) => readdirSync(d));
  assert.ok(cur.ok);
  const e = ok(cur.user.entities);
  const odd = { ...cur.user, entities: { ok: true as const, value: { ...e, refused: [{ ...e.refused[0]!, kind: "planet" }] } } };
  assert.throws(() => userInfoOf(odd, cur.staleness), (x: unknown) => x instanceof RangeError && /the kind of the refused entity/.test((x as Error).message));
  const { entities: _, ...older } = cur.user;
  void _;
  assert.equal(userInfoOf(older, cur.staleness).entities, null);
  const refused = userInfoOf({ ...cur.user, entities: { ok: false, reason: "refused: /entities: it has no cuts table" } }, cur.staleness).entities;
  assert.deepEqual(refused, { ok: false, reason: "refused: /entities: it has no cuts table", stale: cur.staleness.zones["entities"] });
});
