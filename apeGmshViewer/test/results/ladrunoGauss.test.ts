// Gauss points from a fork-written .ladruno (ADR 0122 slice 8a, #2187): reader/ladrunoElements.ts
// through reader/ladruno.ts, and main/results.ts serving them.
//
// ladruno_quad2d.ladruno (fixtures/make_ladruno_fixtures.py --quad2d, fork build 79e0623670): a 3 x 2
// grid of unit FourNodeQuad elements 1..6, plane stress, two LoadControl 0.5 steps of a linear model, with
// `-E stress strain`. Its stress and strain buckets are 2 x 2 Gauss buckets (GAUSS_ID 0..3) over
// QUADRATURE/GP_PARAM (+-1/sqrt 3, PARAM_DOMAIN "[-1,1]"); it is stamped with element_index_space "fem".
//
// The oracles name the right answer without the app:
// - values: ladruno_quad2d.oracle.json, written by tests/fixtures/ladruno/make_viewer_gauss_oracle.py with apeGmsh's Python
//   reader (LadrunoReader.read_gauss) on the same file: each Gauss point's element, natural coordinates
//   and value. The step-0 values are half the step-1 ones (a linear model at load factors 0.5 and 1);
// - positions: the recorder's own MODEL/ELEMENTS/.../GLOBAL_GP_COORDS. The natural coordinates the reader
//   offers, put through gmsh's quad4 shape functions over the model's corner nodes, must land on them;
// - refusals: copies of the fixture with one bucket made inconsistent (h5wasm cannot delete a dataset, so
//   each copy adds a bucket built from the original one's values under another result name).

import assert from "node:assert/strict";
import { copyFileSync, mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { ResultsHost } from "../../src/main/results.ts";
import { LadrunoError, openLadruno, type LadrunoFile } from "../../src/reader/ladruno.ts";
import { uniqueElementIds, type IdList } from "../../src/reader/ladrunoChecks.ts";
import { gatherGauss } from "../../src/reader/ladrunoElements.ts";
import type { ResultsH5 } from "../../src/reader/results.ts";
import { applyPlan, buildPlan, extrapolationVerdict, shapeOf, type CellOf } from "../../src/state/extrapolation.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as ResultsH5;
const fixture = (name: string) => fileURLToPath(new URL(`../../fixtures/${name}`, import.meta.url));
const QUAD = fixture("ladruno_quad2d.ladruno");
const PORTAL = fixture("ladruno_portal3d.ladruno");
const FRAME = fixture("ladruno_frame3d.ladruno");
const TRUSS = fixture("ladruno_truss2d.ladruno");
const ORACLE = JSON.parse(readFileSync(fixture("ladruno_quad2d.oracle.json"), "utf8")) as {
  reads: { component: string; stage: string; step: number; points: { element: number; natural: number[]; value: number }[] }[];
};
const STAGE = "/MODEL_STAGE[1]";
const ELEMS = `${STAGE}/RESULTS/ON_ELEMENTS`;
const KEY = "31-FourNodeQuad[201:0:0]";
const MODEL_KEY = "31-FourNodeQuad[201:0]";
const STRESS = `stress/${KEY}`;
const STRAIN = `strain/${KEY}`;
const R = 1 / Math.sqrt(3);
/** The model's nodes (row by row from the origin, unit spacing) and elements, as the fork run builds them. */
const NODE_XY = (n: number): [number, number] => [(n - 1) % 4, Math.floor((n - 1) / 4)];
const CONN: Record<number, number[]> = { 1: [1, 2, 6, 5], 2: [2, 3, 7, 6], 3: [3, 4, 8, 7], 4: [5, 6, 10, 9], 5: [6, 7, 11, 10], 6: [7, 8, 12, 11] };

function edited(from: string, edit: (f: h5wasm.File) => void): string {
  const p = join(mkdtempSync(join(tmpdir(), "agv-ladruno-gauss-")), "edited.ladruno");
  copyFileSync(from, p);
  const f = new h5wasm.File(p, "a");
  try {
    edit(f);
  } finally {
    f.close();
  }
  return p;
}

function withFile<T>(path: string, body: (r: LadrunoFile) => T): T {
  const r = openLadruno(H5, path);
  try {
    return body(r);
  } finally {
    r.close();
  }
}

type Attrs = { attrs: Record<string, unknown>; delete_attribute(n: string): number; create_attribute(n: string, v: string | number): void };
function setAttr(node: Attrs, key: string, value: string | number | null): void {
  if (Object.keys(node.attrs).includes(key)) node.delete_attribute(key);
  if (value !== null) node.create_attribute(key, value);
}

/** The stress bucket's own datasets, read from the fixture. */
const SRC = (() => {
  const f = new h5wasm.File(QUAD, "r");
  try {
    const ds = (n: string) => f.get(`${ELEMS}/${STRESS}/${n}`) as h5wasm.Dataset;
    return { data: ds("DATA").value as Float64Array, time: ds("TIME").value as Float64Array };
  } finally {
    f.close();
  }
})();

interface BucketSpec {
  gaussId?: number[];
  compNames?: string;
  numComp?: number[];
  multiplicity?: number[];
  fiberId?: number[];
  numColumns?: number | null;
  /** DATA as (T, 6, width); the default is the original stress DATA (2, 6, 12) */
  data?: { values: Float64Array; shape: [number, number, number] };
  time?: Float64Array;
  step?: Int32Array;
  /** the bucket's ID rows; the default is elements 1..6 */
  ids?: number[];
}

/**
 * Add `ON_ELEMENTS/<result>/<key>` to a copy: by default a byte-for-byte rebuild of the stress
 * bucket (4 Gauss points, `sigma11,sigma22,sigma12`), with `spec` overriding one part of it.
 */
function addBucket(f: h5wasm.File, result: string, spec: BucketSpec = {}, key = KEY): void {
  const g = (f.get(ELEMS) as h5wasm.Group).create_group(result).create_group(key);
  const gaussId = spec.gaussId ?? [0, 1, 2, 3];
  const n = gaussId.length;
  const col = (v: number[]) => ({ data: new Int32Array(v), shape: [v.length, 1] });
  const data = spec.data ?? { values: SRC.data, shape: [2, 6, 12] as [number, number, number] };
  const ids = spec.ids ?? [1, 2, 3, 4, 5, 6];
  g.create_dataset({ name: "ID", data: new Int32Array(ids), shape: [ids.length, 1] });
  g.create_dataset({ name: "TIME", data: spec.time ?? SRC.time });
  g.create_dataset({ name: "STEP", data: spec.step ?? new Int32Array([0, 1]) });
  g.create_dataset({ name: "DATA", data: data.values, shape: data.shape });
  const nc = spec.numColumns === undefined ? data.shape[2] : spec.numColumns;
  if (nc !== null) g.create_attribute("NUM_COLUMNS", nc);
  const cm = g.create_group("COLUMN_MAP");
  cm.create_attribute("COMP_NAMES", spec.compNames ?? Array(n).fill("sigma11,sigma22,sigma12").join("\n"));
  cm.create_dataset({ name: "LEVELS", ...col(Array(n).fill(4)) });
  cm.create_dataset({ name: "GAUSS_ID", ...col(gaussId) });
  cm.create_dataset({ name: "NUM_COMP", ...col(spec.numComp ?? Array(n).fill(3)) });
  cm.create_dataset({ name: "MULTIPLICITY", ...col(spec.multiplicity ?? Array(n).fill(1)) });
  cm.create_dataset({ name: "FIBER_ID", ...col(spec.fiberId ?? Array(n).fill(-1)) });
}

/** The refusal of the group `id` on stage_0, or a failed assertion when it was read. */
function refusalOf(path: string, id: string): string[] {
  return withFile(path, (r) => {
    const g = r.index.stages[0]!.gauss.find((x) => x.id === id);
    assert.ok(g, `no gauss group ${id}; groups: ${r.index.stages[0]!.gauss.map((x) => x.id).join(", ")}`);
    assert.deepEqual(g.components, [], `${id} must be refused, never read`);
    assert.throws(() => r.readGaussStep("stage_0", id, `${id.split("/")[0]}/sigma11`, 0), /has no component/);
    return g.refused.map((x) => `${x.component}: ${x.reason}`);
  });
}

/** Each Gauss point's (element, natural coords) -> value at one step, from the reader. */
function readPoints(r: LadrunoFile, component: string, step: number): Map<string, number> {
  const st = r.index.stages[0]!;
  const out = new Map<string, number>();
  for (const g of st.gauss.filter((x) => x.components.includes(component))) {
    const v = r.readGaussStep(st.id, g.id, component, step);
    const e = r.elementIds(st.id, g.id);
    assert.equal(e.mapping, "fem_eid");
    const ids = e.mapping === "fem_eid" ? e.ids : new Float64Array(0);
    assert.equal(v.length, g.nElements * g.nGauss);
    ids.forEach((eid, i) => g.naturalCoords!.forEach((nat, k) => out.set(`${eid}@${nat.join(",")}`, v[i * g.nGauss + k]!)));
  }
  return out;
}

// -- the index -------------------------------------------------------------------

test("a fork-written quad .ladruno indexes its stress and strain buckets as 2 x 2 Gauss groups, named as the file names them", () => {
  withFile(QUAD, (r) => {
    assert.deepEqual(r.index.warnings, []);
    const s = r.index.stages[0]!;
    assert.deepEqual(s.unread, [], "every ON_ELEMENTS bucket of this file is read");
    assert.deepEqual(s.nodes!.components, ["DISPLACEMENT/Ux", "DISPLACEMENT/Uy"]);
    assert.deepEqual(
      s.gauss.map((g) => [g.id, g.path, g.classTag, g.intRule, g.nElements, g.nGauss, g.localAxes, g.refused]),
      // In the file's own order (h5wasm's), as every other list of this reader.
      [
        [STRESS, `${ELEMS}/${STRESS}`, 31, 201, 6, 4, false, []],
        [STRAIN, `${ELEMS}/${STRAIN}`, 31, 201, 6, 4, false, []],
      ],
    );
    assert.deepEqual(s.gauss[0]!.components, ["stress/sigma11", "stress/sigma22", "stress/sigma12"]);
    assert.deepEqual(s.gauss[1]!.components, ["strain/eta11", "strain/eta22", "strain/eta12"]);
    // GP_PARAM's rows, in GAUSS_ID order: gmsh's quad4 corner order, +-1/sqrt 3.
    const want = [[-R, -R], [R, -R], [R, R], [-R, R]];
    for (const g of s.gauss) g.naturalCoords!.forEach((row, k) => row.forEach((x, d) => assert.ok(Math.abs(x - want[k]![d]!) < 1e-15)));
    assert.deepEqual(r.elementIndexVerdict(), { verified: true });
    const e = r.elementIds("stage_0", STRESS);
    assert.equal(e.mapping, "fem_eid");
    assert.deepEqual(e.mapping === "fem_eid" ? Array.from(e.ids) : [], [1, 2, 3, 4, 5, 6]);
  });
});

test("oracle: each Gauss value equals Python's LadrunoReader.read_gauss on the same file (committed ladruno_quad2d.oracle.json)", () => {
  withFile(QUAD, (r) => {
    for (const read of ORACLE.reads) {
      const got = readPoints(r, read.component, read.step);
      assert.equal(got.size, read.points.length, `${read.component}: one value per (element, Gauss point)`);
      for (const p of read.points) {
        const v = got.get(`${p.element}@${p.natural.join(",")}`);
        assert.equal(v, p.value, `${read.component} step ${read.step}, element ${p.element} at (${p.natural.join(", ")})`);
      }
    }
    // A linear model: step 0 (load factor 0.5) is half of step 1, so the history reads the same Gauss point.
    const one = ORACLE.reads.find((x) => x.component === "stress/sigma11" && x.step === 1)!;
    const half = readPoints(r, "stress/sigma11", 0);
    for (const p of one.points) assert.ok(Math.abs(half.get(`${p.element}@${p.natural.join(",")}`)! - p.value / 2) < 1e-9 * Math.max(1, Math.abs(p.value)));
  });
});

test("readGaussRows is readGaussStep stacked step-major; a bad range or component is an argument error", () => {
  withFile(QUAD, (r) => {
    const rows = r.readGaussRows("stage_0", STRESS, "stress/sigma22", 0, 2);
    const s0 = r.readGaussStep("stage_0", STRESS, "stress/sigma22", 0);
    const s1 = r.readGaussStep("stage_0", STRESS, "stress/sigma22", 1);
    assert.deepEqual(Array.from(rows), [...s0, ...s1]);
    assert.throws(() => r.readGaussRows("stage_0", STRESS, "stress/sigma22", 1, 3), /not a range inside stage stage_0/);
    assert.throws(() => r.readGaussStep("stage_0", STRESS, "stress/sigma22", 2), /outside stage stage_0/);
    assert.throws(() => r.readGaussStep("stage_0", STRESS, "DISPLACEMENT/Ux", 0), /has no component "DISPLACEMENT\/Ux"/);
    assert.throws(() => r.readGaussStep("stage_0", "stress/nope", "stress/sigma11", 0), /no gauss group "stress\/nope"/);
  });
});

test("gatherGauss maps DATA columns to element-major Gauss order, and refuses a short read", () => {
  // 1 step, 2 elements, columns 4..8 (width 4); Gauss point g lives in column cols[g].
  const raw = new Float64Array([10, 11, 12, 13, 20, 21, 22, 23]);
  assert.deepEqual(Array.from(gatherGauss(raw, 1, 2, 4, 8, new Int32Array([7, 4]))), [13, 10, 23, 20]);
  assert.throws(() => gatherGauss(raw.subarray(0, 7), 1, 2, 4, 8, new Int32Array([7, 4])), RangeError);
});

// -- positions ---------------------------------------------------------------------

test("positions: GP_PARAM through gmsh's quad4 shape functions lands on the recorder's GLOBAL_GP_COORDS", () => {
  const f = new h5wasm.File(QUAD, "r");
  let global: Float64Array;
  try {
    global = (f.get(`${STAGE}/MODEL/ELEMENTS/${MODEL_KEY}/GLOBAL_GP_COORDS`) as h5wasm.Dataset).value as Float64Array;
  } finally {
    f.close();
  }
  withFile(QUAD, (r) => {
    const g = r.index.stages[0]!.gauss.find((x) => x.id === STRESS)!;
    const N = shapeOf(3)!.N;
    for (let e = 0; e < 6; e++) {
      const corners = CONN[e + 1]!.map(NODE_XY);
      g.naturalCoords!.forEach((nat, k) => {
        const w = N(nat);
        const x = w.reduce((a, wi, c) => a + wi * corners[c]![0], 0);
        const y = w.reduce((a, wi, c) => a + wi * corners[c]![1], 0);
        assert.ok(Math.abs(x - global[e * 8 + 2 * k]!) < 1e-12 && Math.abs(y - global[e * 8 + 2 * k + 1]!) < 1e-12, `element ${e + 1}, Gauss point ${k}`);
      });
    }
  });
});

test("extrapolation to nodes works unchanged: the plan carries the recorded x of each Gauss point to each node's x", () => {
  withFile(QUAD, (r) => {
    const st = r.index.stages[0]!;
    const g = st.gauss.find((x) => x.id === STRESS)!;
    const e = r.elementIds(st.id, g.id);
    const eids = e.mapping === "fem_eid" ? e.ids : new Float64Array(0);
    const cellOf: CellOf = (id) => (CONN[id] ? { code: 3, nodes: CONN[id]! } : null);
    const input = [{ id: g.id, nElements: g.nElements, nGauss: g.nGauss, eids, naturalCoords: g.naturalCoords }];
    assert.deepEqual(extrapolationVerdict(input, cellOf), { corners: null, matrix: null });
    const plan = buildPlan("gauss_node", input, cellOf);
    // A Gauss field whose value is the point's x (linear, so least squares is exact at the corners).
    const xs = new Float64Array(6 * 4);
    for (let i = 0; i < 6; i++) {
      const corners = CONN[eids[i]!]!.map(NODE_XY);
      g.naturalCoords!.forEach((nat, k) => (xs[i * 4 + k] = shapeOf(3)!.N(nat).reduce((a, w, c) => a + w * corners[c]![0], 0)));
    }
    const out = applyPlan(plan, [xs]);
    plan.nodeIds.forEach((n, i) => assert.ok(Math.abs(out.nodal![i]! - NODE_XY(n)[0]) < 1e-12, `node ${n}`));
  });
});

test("a PARAM_DOMAIN other than [-1,1] still reads the Gauss values, offers no natural coordinates, and says why", () => {
  const p = edited(QUAD, (f) => setAttr(f.get(`${STAGE}/MODEL/ELEMENTS/${MODEL_KEY}`) as unknown as Attrs, "PARAM_DOMAIN", "bary"));
  withFile(p, (r) => {
    const s = r.index.stages[0]!;
    assert.deepEqual(s.gauss.map((g) => [g.id, g.components.length, g.naturalCoords]), [
      [STRESS, 3, null],
      [STRAIN, 3, null],
    ]);
    assert.equal(r.index.warnings.length, 2);
    // The at-nodes refusal names the domain cause, never "not floating point" alone.
    const g = s.gauss[0]!;
    const v = extrapolationVerdict([{ id: g.id, nElements: 6, nGauss: 4, eids: new Float64Array([1, 2, 3, 4, 5, 6]), naturalCoords: g.naturalCoords }], (id) => ({ code: 3, nodes: CONN[id]! }));
    assert.match(v.matrix!, /stress\/31-FourNodeQuad\[201:0:0\] has no natural coordinates .*PARAM_DOMAIN is not "\[-1,1\]"/);
    assert.match(r.index.warnings[0]!, /stress\/31-FourNodeQuad\[201:0:0\]: .*PARAM_DOMAIN is "bary", not "\[-1,1\]"; its Gauss points are read, but not extrapolated to nodes/);
  });
});

// -- element ids --------------------------------------------------------------------

test("element rows are FEM ids only under element_index_space = \"fem\"; otherwise unverified, with the reason", () => {
  for (const [value, reason] of [
    // reader/ladruno.ts reports "ops_tag" as malformed (schema §7b.1 allows "fem" or no attribute).
    ["ops_tag", /element_index_space = "ops_tag".*not known to be FEM element ids/],
    [null, /has no @element_index_space/],
  ] as const) {
    const p = edited(QUAD, (f) => setAttr(f.get("INFO/PROVENANCE") as unknown as Attrs, "element_index_space", value));
    withFile(p, (r) => {
      const v = r.elementIndexVerdict();
      assert.equal(v.verified, false);
      const e = r.elementIds("stage_0", STRESS);
      assert.equal(e.mapping, "unverified");
      if (e.mapping === "unverified") {
        assert.match(e.reason, reason);
        assert.deepEqual(Array.from(e.rawIndex), [1, 2, 3, 4, 5, 6]);
      }
    });
  }
});

// -- refusals by name (mutations) ---------------------------------------------------

test("mutation: a GAUSS_ID shortened to 3 rows refuses the bucket by name, never read", () => {
  // Only GAUSS_ID is short: COMP_NAMES and NUM_COMP keep their 4 rows, as in the file.
  const four = { compNames: Array(4).fill("sigma11,sigma22,sigma12").join("\n"), numComp: [3, 3, 3, 3] };
  const p = edited(QUAD, (f) => addBucket(f, "short", { gaussId: [0, 1, 2], ...four }));
  const reasons = refusalOf(p, `short/${KEY}`);
  assert.ok(reasons.length > 0 && reasons.every((x) => /GAUSS_ID/.test(x)), reasons.join("\n"));
  // The file's own buckets still read.
  withFile(p, (r) => assert.deepEqual(r.index.stages[0]!.gauss.find((g) => g.id === STRESS)!.refused, []));
});

test("mutation: a consistent 3-point bucket against a 4-row GP_PARAM is refused: the counts disagree", () => {
  const keep = (k: number) => k % 12 < 9;
  const values = SRC.data.filter((_, i) => keep(i));
  const p = edited(QUAD, (f) =>
    addBucket(f, "three", { gaussId: [0, 1, 2], data: { values: Float64Array.from(values), shape: [2, 6, 9] } }),
  );
  assert.deepEqual(refusalOf(p, `three/${KEY}`), [
    "three/sigma11: /MODEL_STAGE[1]/RESULTS/ON_ELEMENTS/three/31-FourNodeQuad[201:0:0]/COLUMN_MAP/GAUSS_ID lists 3 Gauss points and /MODEL_STAGE[1]/MODEL/ELEMENTS/31-FourNodeQuad[201:0]/QUADRATURE/GP_PARAM 4: the counts disagree",
    "three/sigma22: /MODEL_STAGE[1]/RESULTS/ON_ELEMENTS/three/31-FourNodeQuad[201:0:0]/COLUMN_MAP/GAUSS_ID lists 3 Gauss points and /MODEL_STAGE[1]/MODEL/ELEMENTS/31-FourNodeQuad[201:0]/QUADRATURE/GP_PARAM 4: the counts disagree",
    "three/sigma12: /MODEL_STAGE[1]/RESULTS/ON_ELEMENTS/three/31-FourNodeQuad[201:0:0]/COLUMN_MAP/GAUSS_ID lists 3 Gauss points and /MODEL_STAGE[1]/MODEL/ELEMENTS/31-FourNodeQuad[201:0]/QUADRATURE/GP_PARAM 4: the counts disagree",
  ]);
});

test("mutation: NUM_COLUMNS that is not the sum of MULTIPLICITY * NUM_COMP is refused; so is a missing one", () => {
  const p = edited(QUAD, (f) => {
    addBucket(f, "wide", { numColumns: 13 });
    addBucket(f, "nocols", { numColumns: null });
    addBucket(f, "numcomp", { numComp: [3, 3, 3, 2] });
  });
  assert.match(refusalOf(p, `wide/${KEY}`)[0]!, /NUM_COLUMNS = 13, the column map sums MULTIPLICITY \* NUM_COMP to 12, and DATA has 12 columns: they must agree/);
  assert.match(refusalOf(p, `nocols/${KEY}`)[0]!, /has no integer NUM_COLUMNS attribute/);
  assert.match(refusalOf(p, `numcomp/${KEY}`)[0]!, /NUM_COMP\[3\] = 2 but COMP_NAMES line 3 names 3 components/);
});

test("seam partial writes: a DATA step axis that disagrees with TIME and STEP is refused, never read short", () => {
  const p = edited(QUAD, (f) => addBucket(f, "short_steps", { data: { values: SRC.data.slice(0, 72), shape: [1, 6, 12] } }));
  assert.match(refusalOf(p, `short_steps/${KEY}`)[0]!, /holds 1 steps, TIME 2 and STEP 2: the step axes disagree/);
  const q = edited(QUAD, (f) => addBucket(f, "other_time", { time: new Float64Array([0.25, 0.5]) }));
  assert.match(refusalOf(q, `other_time/${KEY}`)[0]!, /TIME is not the stage's time/);
});

test("seam partial writes: a DATA element axis that disagrees with ID is refused", () => {
  const p = edited(QUAD, (f) => addBucket(f, "short_ids", { ids: [1, 2, 3, 4, 5] }));
  assert.match(refusalOf(p, `short_ids/${KEY}`)[0]!, /DATA has 6 element rows and .*\/ID 5 ids \(a partial write\)/);
});

/** A copy with a second quad element group `31-FourNodeQuad[201:1]` (GP_PARAM copied, GLOBAL_GP_COORDS `width` wide) and a stress bucket of it, elements 7..12. */
function secondGroup(width: number): string {
  // h5wasm cannot rewrite the fixture's model group, so the copy adds a second one and a bucket of it.
  return edited(QUAD, (f) => {
    const src = f.get(`${STAGE}/MODEL/ELEMENTS/${MODEL_KEY}/QUADRATURE/GP_PARAM`) as h5wasm.Dataset;
    const grp = (f.get(`${STAGE}/MODEL/ELEMENTS`) as h5wasm.Group).create_group("31-FourNodeQuad[201:1]");
    grp.create_attribute("PARAM_DOMAIN", "[-1,1]");
    grp.create_group("QUADRATURE").create_dataset({ name: "GP_PARAM", data: src.value as Float64Array, shape: [4, 2] });
    grp.create_dataset({ name: "GLOBAL_GP_COORDS", data: new Float64Array(6 * width), shape: [6, width] });
    // Other elements than the first bucket's 1..6: one element in two buckets of a token is refused (#2316).
    addBucket(f, "stress", { ids: [7, 8, 9, 10, 11, 12] }, "31-FourNodeQuad[201:1:0]");
  });
}

test("positions: a GLOBAL_GP_COORDS whose width is no multiple of GP_PARAM's rows refuses the bucket", () => {
  assert.match(refusalOf(secondGroup(7), "stress/31-FourNodeQuad[201:1:0]")[0]!, /GLOBAL_GP_COORDS has shape \(6, 7\), not \(elements, 4 x dimension\)/);
  // The same group with an 8-wide GLOBAL_GP_COORDS reads: the refusal is the width, nothing else.
  withFile(secondGroup(8), (r) => {
    const g = r.index.stages[0]!.gauss.find((x) => x.id === "stress/31-FourNodeQuad[201:1:0]")!;
    assert.deepEqual(g.refused, []);
    assert.equal(g.components.length, 3);
  });
});

test("seam degenerate: a bucket with no Gauss point (G == 0) is refused alone", () => {
  const p = edited(QUAD, (f) => addBucket(f, "empty", { gaussId: [], compNames: "", data: { values: new Float64Array(0), shape: [2, 6, 0] } }));
  withFile(p, (r) => {
    const g = r.index.stages[0]!.gauss.find((x) => x.id === `empty/${KEY}`)!;
    assert.equal(g.nGauss, 0);
    assert.deepEqual(g.components, []);
    assert.equal(g.refused.length, 1);
    assert.match(g.refused[0]!.reason, /lists no Gauss point \(G == 0\)/);
    assert.deepEqual(r.index.stages[0]!.gauss.filter((x) => x.refused.length === 0).map((x) => x.id), [STRESS, STRAIN]);
  });
});

test("COMP_NAMES that differ between Gauss points refuse the bucket", () => {
  const p = edited(QUAD, (f) => addBucket(f, "mixed", { compNames: "sigma11,sigma22,sigma12\nsigma11,sigma22,sigma12\nsigma11,sigma22,tau12\nsigma11,sigma22,sigma12" }));
  assert.match(refusalOf(p, `mixed/${KEY}`)[0]!, /the Gauss points name different components/);
});

// -- what stays unread ----------------------------------------------------------------

test("fibers, element-level rows and buckets with no structured column map stay in unread, by name", () => {
  const p = edited(QUAD, (f) => {
    addBucket(f, "fiber_mult", { multiplicity: [4, 4, 4, 4], numColumns: 48 });
    addBucket(f, "fiber_id", { fiberId: [0, 1, 2, 3] });
    addBucket(f, "element_level", { gaussId: [-1, -1, -1, -1] });
  });
  withFile(p, (r) => {
    const s = r.index.stages[0]!;
    assert.deepEqual(s.gauss.map((g) => g.id), [STRESS, STRAIN]);
    assert.deepEqual(s.unread, ["ON_ELEMENTS/fiber_mult", "ON_ELEMENTS/fiber_id", "ON_ELEMENTS/element_level"]);
  });
  // One result, two buckets: the read one leaves the result; the other stays, by its bucket name.
  const q = edited(QUAD, (f) => addBucket(f, "stress", { gaussId: [-1, -1, -1, -1], ids: [7, 8, 9, 10, 11, 12] }, "31-FourNodeQuad[201:0:1]"));
  withFile(q, (r) => assert.deepEqual(r.index.stages[0]!.unread, ["ON_ELEMENTS/stress/31-FourNodeQuad[201:0:1]"]));
});

test("beams keep their line-station path: a 1-D GP_PARAM is never a Gauss group (fork portal3d, frame3d), and the truss's element rows stay unread", () => {
  withFile(PORTAL, (r) => {
    for (const s of r.index.stages) assert.deepEqual(s.gauss, []);
    assert.deepEqual(r.index.stages[0]!.unread, ["ON_ELEMENTS/section.force", "ON_ELEMENTS/localForce"]);
    assert.ok(r.availableLineStations("stage_0").includes("Mz"));
  });
  withFile(FRAME, (r) => {
    for (const s of r.index.stages) assert.deepEqual(s.gauss, []);
  });
  withFile(TRUSS, (r) => {
    assert.deepEqual(r.index.stages[0]!.gauss, []);
    assert.deepEqual(r.index.stages[0]!.unread, ["ON_ELEMENTS/basicForce"]);
  });
});

// -- an element listed twice (#2316) ---------------------------------------------------

const esc = (s: string) => s.replace(/[\\^$.*+?()[\]{}|]/g, "\\$&");
/** Opening `path` throws a LadrunoError of kind layout whose message matches `named`. */
function refusedAtOpen(path: string, named: RegExp): void {
  assert.throws(() => openLadruno(H5, path), (e: unknown) => e instanceof LadrunoError && e.kind === "layout" && named.test(e.message), `expected a refusal at open matching ${named}`);
}

test("#2316 uniqueElementIds: interleaved buckets with no repeat pass; ascending buckets that share an id, and an id twice in one, are named", () => {
  const err = (m: string) => new Error(m);
  const check = (lists: IdList[]) => uniqueElementIds(lists, "W", "Y", err);
  // Each list ascends and the ranges overlap: the sort path, and nothing repeats.
  check([{ at: "A", ids: [1, 3, 5] }, { at: "B", ids: [2, 4, 6] }]);
  check([{ at: "A", ids: [5] }, { at: "B", ids: [1] }, { at: "C", ids: [] }, { at: "D", ids: [2, 3] }]);
  assert.throws(() => check([{ at: "A", ids: [1, 3, 5] }, { at: "B", ids: [2, 5, 6] }]), /^Error: W lists element id\(s\) more than once \(id 5 at A row 2 and B row 1\); Y$/);
  assert.throws(() => check([{ at: "A", ids: [4, 4, 1, 4] }]), /^Error: W lists element id\(s\) more than once \(id 4 at A rows 0, 1, 3\); Y$/);
  // Past REPEATED_SHOWN repeated ids, the rest are counted, not listed.
  const many = Array.from({ length: 12 }, (_, i) => i + 1);
  assert.throws(() => check([{ at: "A", ids: many }, { at: "B", ids: many }]), /id 10 at A row 9 and B row 9; and 2 more\); Y$/);
});

test("#2316: a stress bucket whose ID lists element 2 twice refuses the file at open, by name, rather than reading element 3's Gauss rows as element 2", () => {
  // ID 1..6 -> 1, 2, 2, 4, 5, 6. Unrefused, the group reads 6 rows over 5 elements: element 2 twice, element 3 gone.
  const p = edited(QUAD, (f) => (f.get(`${ELEMS}/${STRESS}/ID`) as h5wasm.Dataset).write_slice([[2, 3], [0, 1]], new Int32Array([2])));
  refusedAtOpen(p, new RegExp(`^${esc(p)}: ${esc(`${ELEMS}/stress`)} lists element id\\(s\\) more than once \\(id 2 at ${esc(`${ELEMS}/${STRESS}/ID`)} rows 1, 2\\); the recorder writes each element once per token`));
});

test("#2316: one element in two stress buckets refuses the file at open, naming both buckets; two tokens holding one element, and disjoint buckets, still open", () => {
  const second = "31-FourNodeQuad[201:1:0]";
  // The added bucket holds elements 7, 8, 3, 10, 11, 12: element 3 is also row 2 of the fixture's own bucket.
  const p = edited(QUAD, (f) => addBucket(f, "stress", { ids: [7, 8, 3, 10, 11, 12] }, second));
  refusedAtOpen(p, new RegExp(`^${esc(p)}: ${esc(`${ELEMS}/stress`)} lists element id\\(s\\) more than once \\(id 3 at ${esc(`${ELEMS}/${STRESS}/ID`)} row 2 and ${esc(`${ELEMS}/stress/${second}/ID`)} row 2\\)`));
  // Controls: the same bucket over elements 7..12 opens, and stress and strain (one element in two tokens) are the fixture itself.
  withFile(edited(QUAD, (f) => addBucket(f, "stress", { ids: [7, 8, 9, 10, 11, 12] }, second)), (r) => assert.ok(r.index.stages[0]!.gauss.some((g) => g.id === `stress/${second}`)));
  withFile(QUAD, (r) => assert.deepEqual(r.index.stages[0]!.gauss.map((g) => g.nElements), [6, 6]));
});

test("#2316: a line-station bucket (not a Gauss one) whose ID lists an element twice refuses the file at open too (fork portal3d localForce)", () => {
  const token = `${STAGE}/RESULTS/ON_ELEMENTS/localForce`;
  let key = "";
  const p = edited(PORTAL, (f) => {
    key = (f.get(token) as h5wasm.Group).keys()[0]!;
    const id = f.get(`${token}/${key}/ID`) as h5wasm.Dataset;
    const ids = Array.from(id.value as Int32Array);
    assert.equal(new Set(ids).size, 3, "the committed bucket lists its 3 elements once each");
    id.write_slice([[1, 2], [0, 1]], new Int32Array([ids[0]!]));
  });
  withFile(PORTAL, (r) => assert.ok(r.availableLineStations("stage_0").includes("Mz"), "the committed portal opens and serves its line stations"));
  refusedAtOpen(p, new RegExp(`^${esc(p)}: ${esc(token)} lists element id\\(s\\) more than once \\(id \\d+ at ${esc(`${token}/${key}/ID`)} rows 0, 1\\)`));
});

// -- main serves them -------------------------------------------------------------------

const host = () => new ResultsHost(H5, h5wasm.ready);

test("main: open carries the Gauss groups and the element ids; readStep serves an element component; a nodal and a Gauss component of one label stay apart", async () => {
  // Seam "shared namespace": an element result named DISPLACEMENT with a `Ux` column.
  const p = edited(QUAD, (f) => addBucket(f, "DISPLACEMENT", { compNames: Array(4).fill("Ux,Uy,Uz").join("\n") }));
  const h = host();
  const a = await h.open(p);
  assert.ok(a.ok, a.ok ? "" : a.error);
  if (!a.ok) return;
  const st = a.meta.stages[0]!;
  assert.deepEqual(a.meta.elementIndex, { verified: true });
  assert.deepEqual(st.gauss.map((g) => [g.id, g.nElements, g.nGauss, g.components]), [
    [STRESS, 6, 4, ["stress/sigma11", "stress/sigma22", "stress/sigma12"]],
    [STRAIN, 6, 4, ["strain/eta11", "strain/eta22", "strain/eta12"]],
    [`DISPLACEMENT/${KEY}`, 6, 4, ["DISPLACEMENT/Ux", "DISPLACEMENT/Uy", "DISPLACEMENT/Uz"]],
  ]);
  assert.deepEqual(Array.from(a.elementIds["stage_0"]![STRESS]!), [1, 2, 3, 4, 5, 6]);
  const s = h.readStep({ path: p, stage: "stage_0", step: 1, seq: 1, element: "DISPLACEMENT/Ux" });
  assert.ok(s.ok, s.ok ? "" : s.error);
  if (!s.ok) return;
  // The nodal Ux is the displacement of 12 nodes; the element one is the bucket's sigma11 columns (24 values).
  assert.equal(s.nodal["DISPLACEMENT/Ux"]!.length, 12);
  assert.equal(s.gauss[`DISPLACEMENT/${KEY}`]!.length, 24);
  withFile(QUAD, (r) => assert.deepEqual(Array.from(s.gauss[`DISPLACEMENT/${KEY}`]!), Array.from(r.readGaussStep("stage_0", STRESS, "stress/sigma11", 1))));
  assert.notDeepEqual(Array.from(s.nodal["DISPLACEMENT/Ux"]!), Array.from(s.gauss[`DISPLACEMENT/${KEY}`]!).slice(0, 12));
  h.close();
});

test("main: an element scan of a .ladruno reads every step (oracle: the element means at load factors 0.5 and 1)", async () => {
  const h = host();
  const a = await h.open(QUAD);
  assert.ok(a.ok);
  const scan = await h.scan({ path: QUAD, stage: "stage_0", components: ["stress/sigma11"], magnitude: false, scanId: 1, source: "element" }, () => {});
  assert.ok(scan.ok, scan.ok ? "" : scan.error);
  if (!scan.ok) return;
  const pts = ORACLE.reads.find((x) => x.component === "stress/sigma11" && x.step === 1)!.points;
  const means = [1, 2, 3, 4, 5, 6].map((e) => pts.filter((p) => p.element === e).reduce((acc, p) => acc + p.value, 0) / 4);
  const all = [...means, ...means.map((m) => m / 2)];
  assert.ok(Math.abs(scan.min! - Math.min(...all)) < 1e-9 && Math.abs(scan.max! - Math.max(...all)) < 1e-9, `${scan.min}..${scan.max}`);
  h.close();
});

test("main: an unverified .ladruno never serves element values", async () => {
  const p = edited(QUAD, (f) => setAttr(f.get("INFO/PROVENANCE") as unknown as Attrs, "element_index_space", "ops_tag"));
  const h = host();
  const a = await h.open(p);
  assert.ok(a.ok);
  if (!a.ok) return;
  assert.equal(a.meta.elementIndex.verified, false);
  assert.deepEqual(a.elementIds, {}, "an unverified file's rows never leave main");
  const s = h.readStep({ path: p, stage: "stage_0", step: 0, seq: 1, element: "stress/sigma11" });
  assert.ok(!s.ok && /element results of .* are refused: .*"ops_tag"/.test(s.error));
  h.close();
});
