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
import { LadrunoError, type LadrunoFile } from "../../src/reader/ladruno.ts";
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
