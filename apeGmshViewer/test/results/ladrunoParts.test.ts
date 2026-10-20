// reader/ladrunoParts.ts and its dispatch in main/results.ts against a fork-written
// 2-part set (ADR 0122 slice 8, #2188). fixtures/make_ladruno_fixtures.py (`--parts`)
// says how it was made: an OpenSeesMP run at fork build 80dd3f13d, two trusses that
// meet at node 2, one per rank, so node 2 is in both parts. The fork stamps
// DISPLACEMENT and basicForce PARTITION_REDUCTION = NONE and REACTION_FORCE = SUM.
//
// The oracle is Python's stitched read of the same files
// (`LadrunoMultiPartitionReader.read_nodes`, written to
// fixtures/ladruno_parts2d.oracle.json by tests/fixtures/ladruno/make_viewer_parts_oracle.py),
// and the closed form of the model: Uy2 = -10 t / k, k = (EA/L1)/2 + (EA/L2)/5,
// L1 = sqrt 2, L2 = sqrt 5, and node 2's reaction is the sum of the parts' rows
// (500 Uy2 / sqrt 2) (1, 1) + (0, 10 t) and (200 Uy2 / sqrt 5) (-2, 1).
//
// Each mutation is a scratch copy of the set: a part removed, a PARTITION_REDUCTION
// absent, unknown or UNSUPPORTED, a NUM_PARTITIONS that disagrees, a part past the
// count, a part still being written, parts that disagree on the spatial dimension,
// an unstamped or differently stamped part, and a part re-stamped while the set is open.

import assert from "node:assert/strict";
import { copyFileSync, existsSync, mkdtempSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { pairSet } from "../../src/main/pairing.ts";
import { ResultsHost } from "../../src/main/results.ts";
import { LadrunoError, openLadruno, type LadrunoFile } from "../../src/reader/ladruno.ts";
import { LadrunoPartSet, openLadrunoSet, partOf, partSiblings, PART_SET_ELEMENTS_UNREAD } from "../../src/reader/ladrunoParts.ts";
import type { ResultsH5 } from "../../src/reader/results.ts";
import { pairResults } from "../../src/state/results.ts";
import type { ResultsInfo } from "../../src/state/types.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as ResultsH5;
const fixture = (name: string) => fileURLToPath(new URL(`../../fixtures/${name}`, import.meta.url));
const PART = [fixture("ladruno_parts2d.part-0.ladruno"), fixture("ladruno_parts2d.part-1.ladruno")];
const SHORT = fixture("ladruno_parts2d.short.part-1.ladruno");
const ORACLE = JSON.parse(readFileSync(fixture("ladruno_parts2d.oracle.json"), "utf8")) as {
  reads: { python_component: string; group: string; column: number; node_ids: number[]; time: number[]; values: number[][] }[];
};
const LABELS: Record<string, string[]> = { DISPLACEMENT: ["Ux", "Uy"], REACTION_FORCE: ["RFx", "RFy"] };
const ON_NODES = "/MODEL_STAGE[1]/RESULTS/ON_NODES";
const MODEL_HASH = "0123456789abcdef0123456789abcdef";

const list = (dir: string) => readdirSync(dir);
const close = (a: number, b: number, tol = 1e-12) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));

/** A scratch directory holding the set under `stem`, part k from `from[k]` (null: no part k), each edited by `edit[k]`. */
function scratchSet(
  from: (string | null)[] = PART,
  edit: Record<number, (f: h5wasm.File) => void> = {},
  stem = "run",
): { dir: string; part: (k: number) => string } {
  const dir = mkdtempSync(join(tmpdir(), "agv-parts-"));
  const part = (k: number) => join(dir, `${stem}.part-${k}.ladruno`);
  from.forEach((src, k) => {
    if (src === null) return;
    copyFileSync(src, part(k));
    const e = edit[k];
    if (!e) return;
    const f = new h5wasm.File(part(k), "a");
    try {
      e(f);
    } finally {
      f.close();
    }
  });
  return { dir, part };
}

type AttrNode = { attrs: Record<string, unknown>; delete_attribute(n: string): number; create_attribute(n: string, v: string | number): void };
function setAttr(node: AttrNode, key: string, value: string | number | null): void {
  if (Object.keys(node.attrs).includes(key)) node.delete_attribute(key);
  if (value !== null) node.create_attribute(key, value);
}
const reduction = (value: string | null) => (f: h5wasm.File) => setAttr(f.get(`${ON_NODES}/REACTION_FORCE`) as unknown as AttrNode, "PARTITION_REDUCTION", value);

function withSet<T>(path: string, body: (s: LadrunoPartSet) => T): T {
  const s = openLadrunoSet(H5, path, list);
  try {
    return body(s);
  } finally {
    s.close();
  }
}

function refuses(path: string, pattern: RegExp): void {
  assert.throws(
    () => openLadrunoSet(H5, path, list),
    (e: unknown) => e instanceof LadrunoError && pattern.test(e.message),
    `expected a refusal matching ${pattern}`,
  );
}

// -- the oracle -----------------------------------------------------------------

test("the stitched set equals Python's stitched read, step by step (NONE and SUM), opened from either part", () => {
  for (const opened of PART) {
    withSet(opened, (s) => {
      const st = s.index.stages[0]!;
      assert.equal(s.index.stages.length, 1);
      assert.deepEqual(Array.from(st.nodes!.ids), [1, 2, 3], "node 2, on the partition interface, is one node of the set");
      assert.deepEqual(st.nodes!.components, ["DISPLACEMENT/Ux", "DISPLACEMENT/Uy", "REACTION_FORCE/RFx", "REACTION_FORCE/RFy"]);
      assert.deepEqual(st.nodes!.refused, []);
      for (const read of ORACLE.reads) {
        const comp = `${read.group}/${LABELS[read.group]![read.column]!}`;
        assert.deepEqual(read.node_ids, [1, 2, 3]);
        assert.deepEqual(Array.from(st.time), read.time);
        read.values.forEach((row, k) => {
          const got = Array.from(s.readNodalStep(st.id, [comp], k)[0]!);
          row.forEach((v, i) => assert.ok(close(got[i]!, v), `${comp} step ${k} node ${read.node_ids[i]}: app ${got[i]}, Python ${v}`));
        });
        // The history read (a scan) agrees with the step reads.
        const rows = s.readNodalRows(st.id, [comp], 0, read.values.length)[0]!;
        assert.equal(rows.length, read.values.flat().length);
        read.values.flat().forEach((v, i) => assert.ok(close(rows[i]!, v)));
      }
    });
  }
});

test("the closed form: node 2's displacement is one copy, its reaction the sum of the parts' partials", () => {
  const k = 1000 / Math.SQRT2 / 2 + 1000 / Math.sqrt(5) / 5;
  withSet(PART[0]!, (s) => {
    const st = s.index.stages[0]!;
    st.time.forEach((t, step) => {
      const [uy, rx, ry] = s.readNodalStep(st.id, ["DISPLACEMENT/Uy", "REACTION_FORCE/RFx", "REACTION_FORCE/RFy"], step);
      const u = (-10 * t) / k;
      assert.ok(close(uy![1]!, u, 1e-9), `Uy2 ${uy![1]} vs ${u}`);
      const p0 = [(500 * u) / Math.SQRT2, (500 * u) / Math.SQRT2 + 10 * t];
      const p1 = [(-400 * u) / Math.sqrt(5), (200 * u) / Math.sqrt(5)];
      assert.ok(close(rx![1]!, p0[0]! + p1[0]!, 1e-9), `RFx2 ${rx![1]} vs ${p0[0]! + p1[0]!}`);
      assert.ok(Math.abs(ry![1]!) < 1e-9, `RFy2 ${ry![1]} sums to the free DOF's zero residual`);
      // Neither part's copy alone is the answer: the stitch summed them.
      assert.ok(!close(rx![1]!, p0[0]!, 1e-3) && !close(rx![1]!, p1[0]!, 1e-3));
    });
  });
});

// -- one read per part for every component (#2297) --------------------------------

function sameBits(a: Float64Array, b: Float64Array): boolean {
  return Buffer.from(a.buffer, a.byteOffset, a.byteLength).equals(Buffer.from(b.buffer, b.byteOffset, b.byteLength));
}

/**
 * Before #2297: one component per call. Each part's column is read on its own
 * (`readNodalRows(stage, [c], k0, k1)`), over the sorted union of the parts' node ids;
 * NONE writes the parts last to first, SUM adds them in part order from 0.
 */
function perComponentStitch(paths: readonly string[], stageId: string, comp: string, k0: number, k1: number): Float64Array {
  const parts = paths.map((p) => openLadruno(H5, p, { part: true }));
  try {
    const ids = parts.map((p) => Array.from(p.stage(stageId).nodes!.ids));
    const union = [...new Set(ids.flat())].sort((a, b) => a - b);
    const at = new Map(union.map((n, i) => [n, i]));
    const kinds = [...new Set(parts.map((p) => p.nodalReduction(stageId, comp)))];
    assert.equal(kinds.length, 1, `${comp}: one PARTITION_REDUCTION`);
    const sum = kinds[0] === "SUM";
    const T = k1 - k0;
    const N = union.length;
    const out = new Float64Array(T * N);
    const order = sum ? parts.map((_, j) => j) : parts.map((_, j) => parts.length - 1 - j);
    for (const j of order) {
      const rows = parts[j]!.readNodalRows(stageId, [comp], k0, k1)[0]!;
      const nj = ids[j]!.length;
      for (let k = 0; k < T; k++) {
        for (let i = 0; i < nj; i++) {
          const a = k * N + at.get(ids[j]![i]!)!;
          out[a] = sum ? out[a]! + rows[k * nj + i]! : rows[k * nj + i]!;
        }
      }
    }
    return out;
  } finally {
    for (const p of parts) p.close();
  }
}

/** The whole history, each single step, and an inner range. */
function rangesOf(nSteps: number): [number, number][] {
  const out: [number, number][] = [[0, nSteps]];
  for (let k = 0; k < nSteps; k++) out.push([k, k + 1]);
  if (nSteps >= 3) out.push([1, nSteps - 1]);
  return out;
}

/** The set's rows, in several selections, against the one-component-per-call stitch, bit for bit; the count of arrays checked. */
function setMatchesPerComponent(opened: string): number {
  let checked = 0;
  withSet(opened, (s) => {
    for (const st of s.index.stages) {
      const comps = st.nodes?.components ?? [];
      if (comps.length === 0) continue;
      const selections = [comps, [...comps].reverse(), comps.slice(1), [comps[comps.length - 1]!, comps[0]!]];
      for (const [k0, k1] of rangesOf(st.nSteps)) {
        const want = comps.map((c) => perComponentStitch(s.paths, st.id, c, k0, k1));
        for (const order of selections) {
          const got = s.readNodalRows(st.id, order, k0, k1);
          assert.equal(got.length, order.length);
          order.forEach((c, i) => {
            assert.ok(sameBits(got[i]!, want[comps.indexOf(c)]!), `${opened} ${st.id} steps ${k0}..${k1} ${c} in [${order.join(", ")}]`);
            checked++;
          });
          if (k1 === k0 + 1) {
            const step = s.readNodalStep(st.id, order, k0);
            order.forEach((c, i) => assert.ok(sameBits(step[i]!, want[comps.indexOf(c)]!), `${opened} ${st.id} step ${k0} ${c}`));
          }
        }
        const twice = s.readNodalRows(st.id, [comps[0]!, comps[0]!], k0, k1);
        assert.notEqual(twice[0], twice[1], "a component asked for twice gets its own array");
        assert.ok(sameBits(twice[0]!, want[0]!) && sameBits(twice[1]!, want[0]!));
        checked += 2;
      }
    }
  });
  return checked;
}

/** Every nodal DATA of the part rewritten as `factor` times itself plus `offset` times each value's position, so each part's copy differs. */
const perturb = (factor: number, offset: number) => (f: h5wasm.File) => {
  for (const r of ["DISPLACEMENT", "REACTION_FORCE"]) {
    const ds = f.get(`${ON_NODES}/${r}/DATA`) as h5wasm.Dataset;
    const [T, n, c] = ds.shape as [number, number, number];
    const v = ds.value as Float64Array;
    assert.ok(v instanceof Float64Array, `${ds.path} is float64`);
    ds.write_slice([[0, T], [0, n], [0, c]], Float64Array.from(v, (x, i) => x * factor + (i + 1) * offset));
  }
};
const both = (...edits: ((f: h5wasm.File) => void)[]) => (f: h5wasm.File) => edits.forEach((e) => e(f));
const partition = (id: number | null, count: number) => (f: h5wasm.File) => {
  const info = f.get("INFO") as unknown as AttrNode;
  if (id !== null) setAttr(info, "PARTITION_ID", id);
  setAttr(info, "NUM_PARTITIONS", count);
};

/** A 3-part set: the committed parts and a third, a copy of part 1; parts 1 and 2 perturbed so that no two copies of node 2 agree. */
const threeParts = () =>
  scratchSet([PART[0]!, PART[1]!, PART[1]!], { 0: partition(null, 3), 1: both(partition(null, 3), perturb(1.5, 1 / 7)), 2: both(partition(2, 3), perturb(-2, 1 / 3)) });

test("#2297: readNodalRows and readNodalStep return the one-component-per-call bits, every component, every range, several selections", () => {
  // The committed set, opened from either part (its interface copies agree), then sets whose copies differ.
  for (const opened of PART) assert.equal(setMatchesPerComponent(opened), 3 * (4 + 4 + 3 + 2 + 2), `${opened}`);
  const two = scratchSet(PART, { 1: perturb(1.5, 1 / 7) });
  const three = threeParts();
  try {
    assert.ok(setMatchesPerComponent(two.part(1)) > 0);
    assert.ok(setMatchesPerComponent(three.part(2)) > 0);
  } finally {
    rmSync(two.dir, { recursive: true });
    rmSync(three.dir, { recursive: true });
  }
});

test("#2297: in a 3-part set node 2's NONE value is part 0's copy and its SUM the three copies added in part order", () => {
  const three = threeParts();
  try {
    const own = (k: number, comp: string) => {
      const p = openLadruno(H5, three.part(k), { part: true });
      try {
        const st = p.index.stages[0]!;
        const i = Array.from(st.nodes!.ids).indexOf(2);
        const rows = p.readNodalRows(st.id, [comp], 0, st.nSteps)[0]!;
        return Array.from({ length: st.nSteps }, (_, step) => rows[step * st.nodes!.ids.length + i]!);
      } finally {
        p.close();
      }
    };
    withSet(three.part(0), (s) => {
      const st = s.index.stages[0]!;
      const [ux, rfx] = s.readNodalRows(st.id, ["DISPLACEMENT/Ux", "REACTION_FORCE/RFx"], 0, st.nSteps);
      const copies = [0, 1, 2].map((k) => own(k, "DISPLACEMENT/Ux"));
      const partials = [0, 1, 2].map((k) => own(k, "REACTION_FORCE/RFx"));
      for (let step = 0; step < st.nSteps; step++) {
        assert.equal(new Set(copies.map((c) => c[step])).size, 3, "the three copies of node 2 differ");
        assert.equal(ux![step * 3 + 1], copies[0]![step], `step ${step}: Ux2 is part 0's copy`);
        assert.equal(rfx![step * 3 + 1], 0 + partials[0]![step]! + partials[1]![step]! + partials[2]![step]!, `step ${step}: RFx2 is the parts' sum`);
      }
    });
  } finally {
    rmSync(three.dir, { recursive: true });
  }
});

test("#2297: N components of P parts are P hyperslabs per DATA, not N x P", () => {
  const proto = h5wasm.Dataset.prototype;
  const slice = proto.slice;
  const seen: string[] = [];
  proto.slice = function (this: h5wasm.Dataset, ranges: Parameters<typeof slice>[0]) {
    seen.push(this.path);
    return slice.call(this, ranges);
  };
  const three = threeParts();
  try {
    for (const [opened, P] of [[PART[0]!, 2], [three.part(0), 3]] as const) {
      withSet(opened, (s) => {
        const st = s.index.stages[0]!;
        const comps = st.nodes!.components;
        assert.equal(comps.length, 4);
        const perData = () => {
          const n = new Map<string, number>();
          for (const p of seen) n.set(p, (n.get(p) ?? 0) + 1);
          return n;
        };
        const reads: (() => unknown)[] = [
          () => s.readNodalRows(st.id, comps, 0, st.nSteps),
          () => s.readNodalStep(st.id, [...comps].reverse(), 1),
          () => s.readNodalRows(st.id, [comps[0]!, comps[0]!, comps[3]!], 0, 1),
        ];
        for (const read of reads) {
          seen.length = 0;
          read();
          const n = perData();
          assert.deepEqual([...n.keys()].sort(), [`${ON_NODES}/DISPLACEMENT/DATA`, `${ON_NODES}/REACTION_FORCE/DATA`]);
          for (const [path, count] of n) assert.equal(count, P, `${path}: ${count} slices for ${P} parts`);
        }
      });
    }
  } finally {
    proto.slice = slice;
    rmSync(three.dir, { recursive: true });
  }
});

test("#2297: every component is checked before any part is read; the range check comes first", () => {
  const proto = h5wasm.Dataset.prototype;
  const slice = proto.slice;
  let slices = 0;
  proto.slice = function (this: h5wasm.Dataset, ranges: Parameters<typeof slice>[0]) {
    slices++;
    return slice.call(this, ranges);
  };
  try {
    withSet(PART[0]!, (s) => {
      const st = s.index.stages[0]!;
      slices = 0;
      assert.throws(() => s.readNodalRows(st.id, ["DISPLACEMENT/Ux", "DISPLACEMENT/Uz"], 0, 1), /stage stage_0 of the part set has no component "DISPLACEMENT\/Uz"/);
      assert.throws(() => s.readNodalRows(st.id, ["DISPLACEMENT/Uz"], 0, st.nSteps + 1), /steps 0\.\.3 are not a range inside stage stage_0 \(0\.\.2\)/);
      assert.throws(() => s.readNodalRows("stage_9", ["DISPLACEMENT/Ux"], 0, 1), /no stage "stage_9"/);
      assert.equal(slices, 0, "a refused read reads nothing");
      assert.deepEqual(s.readNodalRows(st.id, [], 0, 1), []);
      assert.equal(slices, 0, "no component, no read");
    });
  } finally {
    proto.slice = slice;
  }
});

test("element results of a part set are listed unread, and the element verdict is unverified", () => {
  withSet(PART[1]!, (s) => {
    const st = s.index.stages[0]!;
    assert.deepEqual(st.gauss, []);
    // A one-file truss serves basicForce as the line station N; the set says why it does not.
    assert.ok(st.unread.includes(`line stations N (${PART_SET_ELEMENTS_UNREAD})`), st.unread.join("; "));
    assert.deepEqual(s.elementIndexVerdict(), { verified: false, reason: PART_SET_ELEMENTS_UNREAD });
    assert.deepEqual(s.availableLineStations(st.id), []);
    assert.throws(() => s.readLineStationStep(st.id, "N"), /not stitched/);
  });
});

// -- the open set -----------------------------------------------------------------

test("discovery: the siblings of the opened part's stem, by index; another stem is not a sibling", () => {
  const { dir, part } = scratchSet();
  copyFileSync(PART[0]!, join(dir, "other.part-0.ladruno"));
  assert.deepEqual(partOf(part(1)), { dir: `${dir}${dir.includes("\\") ? "\\" : "/"}`, stem: "run", index: 1 });
  assert.deepEqual(partSiblings(part(1), list).map((x) => x.index), [0, 1]);
  assert.equal(partOf(join(dir, "run.ladruno")), null);
  rmSync(dir, { recursive: true });
});

test("mutation: part 1 deleted -> the set is refused, naming index 1, from part 0", () => {
  const { dir, part } = scratchSet([PART[0]!, null]);
  refuses(part(0), /NUM_PARTITIONS = 2, and the directory holds part\(s\) 0: part\(s\) 1 missing/);
  rmSync(dir, { recursive: true });
});

test("a part past NUM_PARTITIONS, or a PARTITION_ID that is not the file's index, refuses the set", () => {
  const extra = scratchSet([PART[0]!, PART[1]!, PART[1]!]);
  refuses(extra.part(0), /PARTITION_ID is not the file's own index \(run\.part-2\.ladruno = 1\)/);
  rmSync(extra.dir, { recursive: true });
  const past = scratchSet([PART[0]!, PART[1]!, PART[1]!], { 2: (f) => setAttr(f.get("INFO") as unknown as AttrNode, "PARTITION_ID", 2) });
  refuses(past.part(1), /part\(s\) 2 past NUM_PARTITIONS/);
  rmSync(past.dir, { recursive: true });
});

test("parts whose NUM_PARTITIONS disagree are refused, naming each part's value", () => {
  const { dir, part } = scratchSet(PART, { 1: (f) => setAttr(f.get("INFO") as unknown as AttrNode, "NUM_PARTITIONS", 3) });
  refuses(part(0), /disagree on INFO@NUM_PARTITIONS \(run\.part-0\.ladruno = 2, run\.part-1\.ladruno = 3\)/);
  rmSync(dir, { recursive: true });
});

test("partial write: a part still being written (a shorter step axis) refuses the set; the common prefix is never read", () => {
  const { dir, part } = scratchSet([PART[0]!, SHORT]);
  refuses(part(0), /run\.part-1\.ladruno holds 1 step\(s\) and run\.part-0\.ladruno 2: the step axes disagree/);
  rmSync(dir, { recursive: true });
});

test("parts whose TIME values differ (same length) are refused as not one run", () => {
  const shift = (f: h5wasm.File) => {
    for (const r of ["DISPLACEMENT", "REACTION_FORCE"]) (f.get(`${ON_NODES}/${r}/TIME`) as h5wasm.Dataset).write_slice([[1, 2]], new Float64Array([0.75]));
  };
  const { dir, part } = scratchSet(PART, { 1: shift });
  refuses(part(0), /stage stage_0: the TIME axes of run\.part-0\.ladruno and run\.part-1\.ladruno differ; the parts are not one run/);
  rmSync(dir, { recursive: true });
});

// Review of 78d3f3ae, finding 2: Python `_validate_part_set` compares INFO@RUN_ID (fork WP-165).
const runId = (id: string | null, scope: string | null) => (f: h5wasm.File) => {
  const info = f.get("INFO") as unknown as AttrNode;
  setAttr(info, "RUN_ID", id);
  setAttr(info, "RUN_ID_SCOPE", scope);
};

test("RUN_ID: parts of run A and run B are refused, naming each part's RUN_ID", () => {
  const { dir, part } = scratchSet(PART, { 0: runId("run-A", "broadcast"), 1: runId("run-B", "broadcast") });
  refuses(part(1), /the parts come from different runs \(INFO@RUN_ID differs: run\.part-0\.ladruno = run-A, run\.part-1\.ladruno = run-B\)/);
  rmSync(dir, { recursive: true });
});

test("RUN_ID is not compared when a part has none, when a part's RUN_ID_SCOPE is absent, or when one is \"process\" (Python's rule)", () => {
  const cases: Record<number, (f: h5wasm.File) => void>[] = [
    { 0: runId("run-A", "broadcast"), 1: runId(null, "broadcast") },
    { 0: runId("run-A", "broadcast"), 1: runId("run-B", null) },
    { 0: runId("run-A", "process"), 1: runId("run-B", "broadcast") },
    { 0: runId("run-A", "launcher"), 1: runId("run-A", "user") },
  ];
  for (const edit of cases) {
    const { dir, part } = scratchSet(PART, edit);
    withSet(part(0), (s) => assert.deepEqual(Array.from(s.index.stages[0]!.nodes!.ids), [1, 2, 3]));
    rmSync(dir, { recursive: true });
  }
});

// #2245: the set takes part 0's index, so its parts must agree on the spatial dimension,
// resolved per part as the single-file reader does (INFO@SPATIAL_DIM, else the one
// coordinate width). The committed parts both say 2.
const spatialDim = (dim: number | null) => (f: h5wasm.File) => setAttr(f.get("INFO") as unknown as AttrNode, "SPATIAL_DIM", dim);

test("SPATIAL_DIM: part 1 rewritten as 3 against part 0's 2 is refused, naming each part's value, from either part", () => {
  const { dir, part } = scratchSet(PART, { 1: spatialDim(3) });
  for (const k of [0, 1]) {
    refuses(part(k), /run\.part-<N>\.ladruno: the parts disagree on the spatial dimension \(run\.part-0\.ladruno = 2 \(INFO@SPATIAL_DIM\), run\.part-1\.ladruno = 3 \(INFO@SPATIAL_DIM\)\)/);
  }
  rmSync(dir, { recursive: true });
});

test("SPATIAL_DIM absent from part 1: its dimension is its coordinate width (2), which agrees, so the set reads", () => {
  const { dir, part } = scratchSet(PART, { 1: spatialDim(null) });
  withSet(part(0), (s) => assert.deepEqual(Array.from(s.index.stages[0]!.nodes!.ids), [1, 2, 3]));
  rmSync(dir, { recursive: true });
});

// h5wasm 0.10 cannot delete or reshape a dataset, so a part whose coordinate width is 3,
// or that has no coordinates, is a stand-in: the check reads only `index` and
// `inferSpatialDim()`, and it runs right after the manifest check.
function standIns(dims: { attr: number | null; inferred: number | null }[]): LadrunoFile[] {
  return dims.map((d, j) => ({
    index: { path: `run.part-${j}.ladruno`, spatialDim: d.attr, partition: { count: dims.length, id: j, runId: null, runIdScope: null } },
    inferSpatialDim: () => d.inferred,
    close: () => undefined,
  })) as unknown as LadrunoFile[];
}

test("SPATIAL_DIM absent: a coordinate width of 3, or no single width, beside part 0's 2 is refused, naming where each value came from", () => {
  const open = (dims: Parameters<typeof standIns>[0]) => () => new LadrunoPartSet("run.part-0.ladruno", "run", [0, 1], standIns(dims));
  const refusal = (pattern: RegExp) => (e: unknown) => e instanceof LadrunoError && pattern.test(e.message);
  assert.throws(
    open([{ attr: 2, inferred: 2 }, { attr: null, inferred: 3 }]),
    refusal(/\(run\.part-0\.ladruno = 2 \(INFO@SPATIAL_DIM\), run\.part-1\.ladruno = 3 \(the coordinate width\)\)/),
  );
  // Unknown beside known is a disagreement: the set is never read as part 0's 2.
  assert.throws(
    open([{ attr: 2, inferred: 2 }, { attr: null, inferred: null }]),
    refusal(/run\.part-1\.ladruno = unknown \(no INFO@SPATIAL_DIM and no single coordinate width\)\)/),
  );
});

// -- PARTITION_REDUCTION ----------------------------------------------------------

test("mutation: REACTION_FORCE@PARTITION_REDUCTION = MAX is refused by name; the displacement still reads", () => {
  const { dir, part } = scratchSet(PART, { 1: reduction("MAX") });
  withSet(part(0), (s) => {
    const n = s.index.stages[0]!.nodes!;
    assert.deepEqual(n.components, ["DISPLACEMENT/Ux", "DISPLACEMENT/Uy"]);
    assert.deepEqual(n.refused.map((r) => r.component), ["REACTION_FORCE/RFx", "REACTION_FORCE/RFy"]);
    assert.match(n.refused[0]!.reason, /REACTION_FORCE@PARTITION_REDUCTION is "MAX" in run\.part-1\.ladruno; expected one of NONE, SUM, UNSUPPORTED/);
    assert.throws(() => s.readNodalStep("stage_0", ["REACTION_FORCE/RFx"], 0), /has no component "REACTION_FORCE\/RFx"/);
  });
  rmSync(dir, { recursive: true });
});

test("an absent PARTITION_REDUCTION is refused, never assumed NONE; UNSUPPORTED and disagreeing parts are refused too", () => {
  const cases: [Record<number, (f: h5wasm.File) => void>, RegExp][] = [
    [{ 0: reduction(null) }, /REACTION_FORCE has no PARTITION_REDUCTION in run\.part-0\.ladruno \(schema §7\.1\); a part set never assumes NONE/],
    [{ 0: reduction("UNSUPPORTED"), 1: reduction("UNSUPPORTED") }, /REACTION_FORCE is PARTITION_REDUCTION = UNSUPPORTED/],
    [{ 1: reduction("NONE") }, /the parts disagree on REACTION_FORCE@PARTITION_REDUCTION \(run\.part-0\.ladruno = SUM, run\.part-1\.ladruno = NONE\)/],
  ];
  for (const [edit, pattern] of cases) {
    const { dir, part } = scratchSet(PART, edit);
    withSet(part(1), (s) => {
      const n = s.index.stages[0]!.nodes!;
      assert.ok(!n.components.some((c) => c.startsWith("REACTION_FORCE/")), n.components.join(", "));
      assert.ok(n.refused.some((r) => pattern.test(r.reason)), n.refused.map((r) => r.reason).join("; "));
    });
    rmSync(dir, { recursive: true });
  }
});

test("#2299: a part whose node ID lists node 2 twice refuses the set at open, by name, rather than summing its two rows into node 2", () => {
  // Part 1's ids 2, 3 -> 2, 2 in both results. Unrefused, the union is 1, 2 and REACTION_FORCE (SUM)
  // adds part 0's row of node 2 to both rows of part 1: node 2's reaction counted twice, node 3's folded in.
  const repeat = (f: h5wasm.File) => {
    for (const r of ["DISPLACEMENT", "REACTION_FORCE"]) (f.get(`${ON_NODES}/${r}/ID`) as h5wasm.Dataset).write_slice([[1, 2], [0, 1]], new Int32Array([2]));
  };
  const { dir, part } = scratchSet(PART, { 1: repeat });
  withSet(PART[0]!, (s) => assert.ok(s.index.stages[0]!.nodes!.components.includes("REACTION_FORCE/RFx"), "the committed set sums REACTION_FORCE"));
  const named = new RegExp(`^${part(1).replace(/[\\^$.*+?()[\]{}|]/g, "\\$&")}: ${ON_NODES.replace(/[[\]]/g, "\\$&")}/DISPLACEMENT/ID lists node id\\(s\\) more than once \\(id 2 at rows 0, 1\\)`);
  for (const opened of [part(0), part(1)]) refuses(opened, named);
  // The part alone is refused at its own open too: the set never sees it.
  assert.throws(() => openLadruno(H5, part(1), { part: true }), (e: unknown) => e instanceof LadrunoError && e.kind === "layout" && named.test(e.message));
  rmSync(dir, { recursive: true });
});

// -- pairing ------------------------------------------------------------------------

const MODEL = { lineage: { modelHash: MODEL_HASH, femHash: "fedcba9876543210fedcba9876543210", resultsHash: null }, openseesVersion: null, programDecl: null };
const asInfo = (s: LadrunoPartSet): ResultsInfo =>
  ({
    path: s.index.path,
    version: "1.0.0",
    sourceType: "ladruno",
    lineage: s.index.lineage ? { ...s.index.lineage, resultsHash: null } : null,
    lineageReason: s.index.lineageReason,
  }) as unknown as ResultsInfo;

test("pairing: a set every part of which carries the same stamp pairs by model_hash", () => {
  withSet(PART[0]!, (s) => {
    assert.deepEqual(s.index.lineage, { modelHash: MODEL_HASH, femHash: "fedcba9876543210fedcba9876543210" });
    assert.deepEqual(pairResults(MODEL, asInfo(s)), { state: "paired", by: "model_hash" });
  });
});

test("pairing: an unstamped part makes the set unpairable, naming the part", () => {
  const unstamp = (f: h5wasm.File) => {
    const prov = f.get("INFO/PROVENANCE") as unknown as AttrNode;
    for (const k of Object.keys(prov.attrs)) prov.delete_attribute(k);
  };
  const { dir, part } = scratchSet(PART, { 1: unstamp });
  withSet(part(0), (s) => {
    assert.equal(s.index.lineage, null);
    assert.match(s.index.lineageReason!, /part\(s\) 1 of the set \(run\.part-1\.ladruno\) carry no pairing stamp/);
    const p = pairResults(MODEL, asInfo(s));
    assert.equal(p.state, "unpairable");
  });
  rmSync(dir, { recursive: true });
});

test("pairing: parts stamped against different models pair with none, naming the part indices", () => {
  const other = (f: h5wasm.File) => {
    const prov = f.get("INFO/PROVENANCE") as unknown as AttrNode & { attrs: Record<string, unknown> };
    prov.delete_attribute("model_hash");
    (prov as unknown as h5wasm.Group).create_attribute("model_hash", "ffffffffffffffffffffffffffffffff");
  };
  const { dir, part } = scratchSet(PART, { 1: other });
  withSet(part(0), (s) => {
    assert.equal(s.index.lineage, null);
    assert.match(s.index.lineageReason!, /part\(s\) 0 model_hash 01234567…, .*; part\(s\) 1 model_hash ffffffff…/);
    assert.notEqual(pairResults(MODEL, asInfo(s)).state, "paired");
  });
  rmSync(dir, { recursive: true });
});

// -- main's dispatch: open, read, reload ---------------------------------------------

test("ResultsHost opens a part as the set, serves stitched steps, and refuses a set that loses a part", async () => {
  const { dir, part } = scratchSet();
  const host = new ResultsHost(H5, h5wasm.ready, { idleMs: 20 });
  const opened = await host.open(part(1));
  assert.ok(opened.ok, opened.ok ? "" : opened.error);
  if (!opened.ok) return;
  assert.equal(opened.meta.path, part(1));
  assert.deepEqual(opened.meta.lineage, { femHash: "fedcba9876543210fedcba9876543210", modelHash: MODEL_HASH, resultsHash: null });
  assert.deepEqual(Array.from(opened.nodeIds["stage_0"]!), [1, 2, 3]);
  assert.deepEqual(opened.elementIds, {}, "no element row of a part set leaves main");
  assert.equal(opened.meta.elementIndex.verified, false);
  const step = host.readStep({ path: part(1), stage: "stage_0", step: 1, seq: 1 });
  assert.ok(step.ok);
  if (!step.ok) return;
  const rx = ORACLE.reads.find((r) => r.python_component === "reaction_force_x")!;
  rx.values[1]!.forEach((v, i) => assert.ok(close(step.nodal["REACTION_FORCE/RFx"]![i]!, v)));
  host.close();
  rmSync(part(1));
  const reopened = await host.open(part(0));
  assert.ok(!reopened.ok);
  if (!reopened.ok) assert.match(reopened.error, /part\(s\) 1 missing/);
  host.close();
  rmSync(dir, { recursive: true });
});

test("opened from the model, a set that lost part 0 is refused naming index 0, never shown empty", async () => {
  const { dir, part } = scratchSet([null, PART[1]!]);
  copyFileSync(fixture("ladruno_parts2d.h5"), join(dir, "run.h5"));
  const slot = pairSet(join(dir, "run.h5"), existsSync).results;
  assert.equal(slot, part(1));
  const host = new ResultsHost(H5, h5wasm.ready, { idleMs: 20 });
  const opened = await host.open(slot!);
  assert.ok(!opened.ok);
  if (!opened.ok) assert.match(opened.error, /holds part\(s\) 1: part\(s\) 0 missing/);
  host.close();
  rmSync(dir, { recursive: true });
});

test("reload: one part re-stamped while the set is open answers changed, and the re-index reads the new stamp", async () => {
  const { dir, part } = scratchSet();
  const host = new ResultsHost(H5, h5wasm.ready, { idleMs: 20 });
  const opened = await host.open(part(0));
  assert.ok(opened.ok);
  host.close();
  // Part 1 (not the opened file) is re-stamped against another model.
  await new Promise((r) => setTimeout(r, 30));
  const f = new h5wasm.File(part(1), "a");
  try {
    const prov = f.get("INFO/PROVENANCE") as unknown as h5wasm.Group;
    prov.delete_attribute("model_hash");
    prov.create_attribute("model_hash", "ffffffffffffffffffffffffffffffff");
    prov.delete_attribute("writer");
    prov.create_attribute("writer", "a longer writer string, so the part file's size changes too");
  } finally {
    f.close();
  }
  const step = host.readStep({ path: part(0), stage: "stage_0", step: 0, seq: 2 });
  assert.equal(step.ok, false);
  if (!step.ok) assert.equal(step.changed, true);
  const again = await host.open(part(0));
  assert.ok(again.ok);
  if (again.ok) {
    assert.equal(again.meta.lineage, null);
    assert.match(again.meta.lineageReason ?? "", /different stamps/);
  }
  host.close();
  rmSync(dir, { recursive: true });
});
