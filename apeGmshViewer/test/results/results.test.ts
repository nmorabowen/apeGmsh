// reader/results.ts against fixtures/results_small.h5, written by apeGmsh's
// own NativeWriter (the repository's scripts/make_viewer_results_fixtures.py). The expected
// values are the generator's closed forms, recomputed here:
//   time[k]                 = 0.01 k
//   nodes/<comp c>[k, j]    = id_j + 0.001 (c + 1) k
//   gauss <comp c>[k, e, g] = eid_e + 0.1 g + 0.001 (c + 1) k
// Each refusal (an unopenable file, version below the floor, newer major,
// missing dataset, non-float data, unknown layout, several partitions) is a
// typed error, and
// the element-id mapping is "fem_eid" for domain_capture only (#1711).

import assert from "node:assert/strict";
import { copyFileSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { openResults, ResultsError, sortStageIds, type ResultsH5 } from "../../src/reader/results.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as ResultsH5;
const FIXTURE = fileURLToPath(new URL("../../fixtures/results_small.h5", import.meta.url));
const NODAL = ["displacement_x", "displacement_y", "displacement_z"];
const STAGE = "stage_0";
const T = 5;

const close = (a: number, b: number) => Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(b));

/** A scratch copy of the fixture, edited through h5wasm in read-write mode. */
function edited(edit: (f: h5wasm.File) => void): string {
  const p = join(mkdtempSync(join(tmpdir(), "agv-results-")), "edited.results.h5");
  copyFileSync(FIXTURE, p);
  const f = new h5wasm.File(p, "a");
  try {
    edit(f);
  } finally {
    f.close();
  }
  return p;
}

function setRootAttr(f: h5wasm.File, key: string, value: string | null): void {
  if (Object.keys(f.attrs).includes(key)) f.delete_attribute(key);
  if (value !== null) f.create_attribute(key, value);
}

function refuses(path: string, kind: string, pattern: RegExp): void {
  assert.throws(
    () => openResults(H5, path),
    (e: unknown) => e instanceof ResultsError && e.kind === kind && pattern.test(e.message),
  );
}

test("the index lists the stage, its time vector, the nodal block and the gauss group", () => {
  const r = openResults(H5, FIXTURE);
  try {
    const { index } = r;
    assert.equal(index.version, "1.1.0");
    assert.equal(index.sourceType, "domain_capture");
    assert.deepEqual(index.warnings, []);
    assert.equal(index.stages.length, 1);
    const s = index.stages[0]!;
    assert.equal(s.id, STAGE);
    assert.equal(s.name, "history");
    assert.equal(s.kind, "transient");
    assert.equal(s.nSteps, T);
    assert.equal(s.partition, "partition_0");
    for (let k = 0; k < T; k++) assert.ok(close(s.time[k]!, 0.01 * k), `time[${k}]`);
    // 2 x 2 cells extruded 1 layer: 3 x 3 x 2 nodes, 2 x 2 x 2 wedges.
    assert.equal(s.nodes!.ids.length, 18);
    assert.deepEqual(s.nodes!.components, NODAL);
    assert.equal(s.gauss.length, 1);
    const g = s.gauss[0]!;
    assert.deepEqual(
      { id: g.id, classTag: g.classTag, intRule: g.intRule, nElements: g.nElements, nGauss: g.nGauss, components: g.components },
      { id: "group_0", classTag: 0, intRule: 0, nElements: 8, nGauss: 6, components: ["stress_xx", "stress_yy"] },
    );
    assert.deepEqual(s.unread, []);
  } finally {
    r.close();
  }
});

test("a nodal step, a gauss step and a node history equal the generator's closed forms", () => {
  const r = openResults(H5, FIXTURE);
  try {
    const s = r.stage(STAGE);
    const ids = s.nodes!.ids;
    for (let k = 0; k < T; k++) {
      const cols = r.readNodalStep(STAGE, NODAL, k);
      cols.forEach((col, c) => {
        assert.ok(col instanceof Float64Array);
        assert.equal(col.length, ids.length);
        ids.forEach((id, j) => assert.ok(close(col[j]!, id + 0.001 * (c + 1) * k), `${NODAL[c]}[${k}, ${j}]`));
      });
    }
    const map = r.elementIds(STAGE, "group_0");
    assert.equal(map.mapping, "fem_eid");
    const eids = map.mapping === "fem_eid" ? map.ids : new Float64Array();
    assert.equal(eids.length, 8);
    for (const [c, comp] of ["stress_xx", "stress_yy"].entries()) {
      const v = r.readGaussStep(STAGE, "group_0", comp, 3);
      assert.equal(v.length, 8 * 6);
      for (let e = 0; e < 8; e++) {
        for (let g = 0; g < 6; g++) assert.ok(close(v[e * 6 + g]!, eids[e]! + 0.1 * g + 0.001 * (c + 1) * 3), `${comp}[3, ${e}, ${g}]`);
      }
    }
    const node = ids[7]!;
    const h = r.readNodeHistory(STAGE, "displacement_z", node);
    assert.equal(h.length, T);
    for (let k = 0; k < T; k++) assert.ok(close(h[k]!, node + 0.003 * k), `history[${k}]`);
  } finally {
    r.close();
  }
});

test("element ids of the fixture are the /model/elements ids (domain_capture)", () => {
  const f = new h5wasm.File(FIXTURE, "r");
  const model = Array.from((f.get("model/elements/prism6/ids") as h5wasm.Dataset).value as BigInt64Array, Number).sort((a, b) => a - b);
  f.close();
  const r = openResults(H5, FIXTURE);
  const map = r.elementIds(STAGE, "group_0");
  r.close();
  assert.equal(map.mapping, "fem_eid");
  if (map.mapping === "fem_eid") assert.deepEqual(Array.from(map.ids).sort((a, b) => a - b), model);
});

test("reads outside the file are refused by name", () => {
  const r = openResults(H5, FIXTURE);
  try {
    assert.throws(() => r.readNodalStep(STAGE, ["displacement_x"], T), (e: unknown) => e instanceof ResultsError && e.kind === "argument");
    assert.throws(() => r.readNodalStep(STAGE, ["rotation_x"], 0), /no component "rotation_x"/);
    assert.throws(() => r.readNodeHistory(STAGE, "displacement_x", 999), /records no node 999/);
    assert.throws(() => r.stage("stage_9"), /no stage "stage_9"/);
  } finally {
    r.close();
  }
});

// -- #1711: the element mapping is an allow-list -------------------------------

for (const [sourceType, label] of [
  ["tcl_recorders", "the recorder transcoder"],
  ["parallel_modal", "another writer the doc names"],
  ["made_up_writer", "an unknown value"],
  [null, "no source_type at all"],
] as const) {
  test(`element rows are "unverified", never FEM ids, for ${label}`, () => {
    const p = edited((f) => setRootAttr(f, "source_type", sourceType));
    const r = openResults(H5, p);
    try {
      const map = r.elementIds(STAGE, "group_0");
      assert.equal(map.mapping, "unverified");
      assert.equal(map.sourceType, sourceType);
      assert.ok(!("ids" in map), "an unverified mapping has no `ids` field to join on");
      if (map.mapping === "unverified") {
        assert.equal(map.rawIndex.length, 8);
        assert.match(map.reason, sourceType === null ? /no source_type/ : /only for domain_capture/);
      }
      // Nodal reads are unaffected.
      assert.equal(r.readNodalStep(STAGE, ["displacement_x"], 0)[0]!.length, 18);
    } finally {
      r.close();
    }
  });
}

// -- dtype: float32 is cast, non-float is refused --------------------------------

test("a float32 component is read and cast to float64", () => {
  const vals = new Float32Array(T * 18).map((_, i) => 0.1 * i);
  const p = edited((f) => {
    (f.get(`stages/${STAGE}/partitions/partition_0/nodes`) as h5wasm.Group).create_dataset({ name: "temperature", data: vals, shape: [T, 18], dtype: "<f" });
  });
  const r = openResults(H5, p);
  try {
    assert.ok(r.stage(STAGE).nodes!.components.includes("temperature"));
    const col = r.readNodalStep(STAGE, ["temperature"], 2)[0]!;
    assert.ok(col instanceof Float64Array);
    for (let j = 0; j < 18; j++) assert.equal(col[j], vals[2 * 18 + j]);
    const h = r.readNodeHistory(STAGE, "temperature", r.stage(STAGE).nodes!.ids[4]!);
    for (let k = 0; k < T; k++) assert.equal(h[k], vals[k * 18 + 4]);
  } finally {
    r.close();
  }
});

test("an integer component is refused at open (dtype)", () => {
  const p = edited((f) => {
    (f.get(`stages/${STAGE}/partitions/partition_0/nodes`) as h5wasm.Group).create_dataset({
      name: "flag",
      data: new Int32Array(T * 18),
      shape: [T, 18],
      dtype: "<i",
    });
  });
  refuses(p, "dtype", /nodes\/flag is not a floating-point dataset/);
});

// -- the version gate -----------------------------------------------------------

test("a results zone below the floor is refused", () => {
  refuses(edited((f) => setRootAttr(f, "results_schema_version", "0.9.0")), "version", /0\.9\.0: below the results floor 1\.0\.0/);
});

test("a newer results major is refused", () => {
  refuses(edited((f) => setRootAttr(f, "results_schema_version", "2.0.0")), "version", /2\.0\.0: a newer major/);
});

test("a newer results minor opens with one warning; the floor itself opens", () => {
  const r = openResults(H5, edited((f) => setRootAttr(f, "results_schema_version", "1.4.0")));
  assert.equal(r.index.warnings.length, 1);
  assert.match(r.index.warnings[0]!, /1\.4\.0 is newer than this app \(1\.1\.x\)/);
  r.close();
  const floor = openResults(H5, edited((f) => setRootAttr(f, "results_schema_version", "1.0.0")));
  assert.deepEqual(floor.index.warnings, []);
  floor.close();
});

test("the envelope schema_version is the fallback; neither stamp is not a results file", () => {
  const r = openResults(H5, edited((f) => setRootAttr(f, "results_schema_version", null)));
  assert.equal(r.index.version, "1.1.0");
  r.close();
  refuses(
    edited((f) => {
      setRootAttr(f, "results_schema_version", null);
      setRootAttr(f, "schema_version", null);
    }),
    "version",
    /not a native apeGmsh results\.h5/,
  );
});

// -- layout ---------------------------------------------------------------------

/** A minimal results file written from scratch, for the layouts the writer never produces. */
function minimal(build: (stage: h5wasm.Group) => void): string {
  const p = join(mkdtempSync(join(tmpdir(), "agv-results-")), "minimal.results.h5");
  const f = new h5wasm.File(p, "w");
  f.create_attribute("results_schema_version", "1.1.0");
  f.create_attribute("source_type", "domain_capture");
  const s = f.create_group("stages").create_group(STAGE);
  s.create_attribute("name", "s");
  s.create_attribute("kind", "static");
  build(s);
  f.close();
  return p;
}

test("a stage without its time vector is refused (missing dataset)", () => {
  const p = minimal((s) => {
    s.create_group("partitions").create_group("partition_0").create_group("nodes").create_dataset({ name: "_ids", data: new BigInt64Array([1n]) });
  });
  refuses(p, "layout", /missing dataset \/stages\/stage_0\/time/);
});

test("a nodal block without _ids is refused (missing dataset)", () => {
  const p = minimal((s) => {
    s.create_dataset({ name: "time", data: new Float64Array([0]) });
    s.create_group("partitions").create_group("partition_0").create_group("nodes").create_dataset({ name: "displacement_x", data: new Float64Array([0]), shape: [1, 1] });
  });
  refuses(p, "layout", /missing dataset \/stages\/stage_0\/partitions\/partition_0\/nodes\/_ids/);
});

test("a component whose step axis disagrees with time is refused", () => {
  const p = minimal((s) => {
    s.create_dataset({ name: "time", data: new Float64Array([0, 1]) });
    const n = s.create_group("partitions").create_group("partition_0").create_group("nodes");
    n.create_dataset({ name: "_ids", data: new BigInt64Array([1n]) });
    n.create_dataset({ name: "displacement_x", data: new Float64Array([0, 0, 0]), shape: [3, 1] });
  });
  refuses(p, "layout", /has shape \(3, 1\); expected \(2, 1\)/);
});

test("an unknown partition child and several partitions are refused", () => {
  const unknown = minimal((s) => {
    s.create_dataset({ name: "time", data: new Float64Array([0]) });
    s.create_group("partitions").create_group("partition_0").create_group("springs");
  });
  refuses(unknown, "layout", /partition_0\/springs: not a partition child/);
  const two = minimal((s) => {
    s.create_dataset({ name: "time", data: new Float64Array([0]) });
    const ps = s.create_group("partitions");
    ps.create_group("partition_0");
    ps.create_group("partition_1");
  });
  refuses(two, "unsupported", /has 2 partitions/);
});

test("a truncated file is refused at open, not opened as an empty file", () => {
  // h5wasm returns a File with a negative id instead of throwing; the reader checks it.
  const p = join(mkdtempSync(join(tmpdir(), "agv-results-")), "truncated.results.h5");
  const bytes = readFileSync(FIXTURE);
  writeFileSync(p, bytes.subarray(0, Math.floor(bytes.length / 2)));
  refuses(p, "open", /HDF5 could not open/);
});

test("stage ids sort by their integer suffix, then lexically", () => {
  assert.deepEqual(sortStageIds(["stage_10", "b", "stage_2", "a", "stage_0"]), ["stage_0", "stage_2", "stage_10", "a", "b"]);
});
