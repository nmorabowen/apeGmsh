// reader/results.ts against fixtures/results_small.h5, written by apeGmsh's
// own NativeWriter (the repository's scripts/make_viewer_results_fixtures.py). The expected
// values are the generator's closed forms, recomputed here:
//   time[k]                 = 0.01 k
//   nodes/<comp c>[k, j]    = id_j + 0.001 (c + 1) k
//   gauss <comp c>[k, e, g] = eid_e + 0.1 g + 0.001 (c + 1) k
// Each refusal (an unopenable file, version below the floor, newer major,
// missing dataset, non-float data, unknown layout, several partitions) is a
// typed error, and
// the element-id mapping is "fem_eid" if and only if the root attribute
// element_index_space is "fem" (stamp-only, #1768; h5-schema.md
// §element_index_space). fixtures/results_stamped.h5 is the same file written
// at results 1.2.0 with NativeWriter.stamp_element_index_space("fem");
// results_small.h5 is an unstamped domain_capture file.

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
const STAMPED = fileURLToPath(new URL("../../fixtures/results_stamped.h5", import.meta.url));
const NODAL = ["displacement_x", "displacement_y", "displacement_z"];
const STAGE = "stage_0";
const T = 5;

const close = (a: number, b: number) => Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(b));

/** A scratch copy of a fixture, edited through h5wasm in read-write mode. */
function edited(edit: (f: h5wasm.File) => void, from: string = FIXTURE): string {
  const p = join(mkdtempSync(join(tmpdir(), "agv-results-")), "edited.results.h5");
  copyFileSync(from, p);
  const f = new h5wasm.File(p, "a");
  try {
    edit(f);
  } finally {
    f.close();
  }
  return p;
}

function setRootAttr(f: h5wasm.File, key: string, value: string | number | null): void {
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
    // The file is unstamped, so its rows are unverified; the closed form is in the raw rows.
    const map = r.elementIds(STAGE, "group_0");
    assert.equal(map.mapping, "unverified");
    const eids = map.mapping === "unverified" ? map.rawIndex : new Float64Array();
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

test("a file the writer stamped element_index_space = \"fem\" has verified ids, the /model/elements ids", () => {
  const f = new h5wasm.File(STAMPED, "r");
  const model = Array.from((f.get("model/elements/prism6/ids") as h5wasm.Dataset).value as BigInt64Array, Number).sort((a, b) => a - b);
  f.close();
  const r = openResults(H5, STAMPED);
  try {
    assert.equal(r.index.version, "1.2.0");
    assert.equal(r.index.elementIndexSpace, "fem");
    assert.deepEqual(r.index.warnings, [], "a 1.2.0 file opens with no version warning");
    const map = r.elementIds(STAGE, "group_0");
    assert.equal(map.mapping, "fem_eid");
    assert.equal(map.sourceType, "domain_capture");
    if (map.mapping === "fem_eid") assert.deepEqual(Array.from(map.ids).sort((a, b) => a - b), model);
    // The same closed form as the unstamped file: the stamp changes the verdict, not the values.
    const ids = map.mapping === "fem_eid" ? map.ids : new Float64Array();
    const v = r.readGaussStep(STAGE, "group_0", "stress_yy", 2);
    for (let e = 0; e < 8; e++) for (let g = 0; g < 6; g++) assert.ok(close(v[e * 6 + g]!, ids[e]! + 0.1 * g + 0.002 * 2));
  } finally {
    r.close();
  }
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

// -- #1768: the element mapping is stamp-only ------------------------------------

/** Open `path`, read group_0's mapping, and check it is "unverified" with no `ids` to join on. */
function assertUnverified(path: string, sourceType: string | null, reason: RegExp): void {
  const r = openResults(H5, path);
  try {
    const map = r.elementIds(STAGE, "group_0");
    assert.equal(map.mapping, "unverified");
    assert.equal(map.sourceType, sourceType);
    assert.ok(!("ids" in map), "an unverified mapping has no `ids` field to join on");
    if (map.mapping === "unverified") {
      assert.equal(map.rawIndex.length, 8);
      assert.match(map.reason, reason);
    }
    // Nodal reads are unaffected.
    assert.equal(r.readNodalStep(STAGE, ["displacement_x"], 0)[0]!.length, 18);
  } finally {
    r.close();
  }
}

test("a domain_capture file without the stamp is unverified (no source_type is an exception)", () => {
  assertUnverified(FIXTURE, "domain_capture", /has no @element_index_space/);
  // The writer's own stamped file, with the stamp removed, is unverified too: the stamp is the whole verdict.
  assertUnverified(edited((f) => setRootAttr(f, "element_index_space", null), STAMPED), "domain_capture", /has no @element_index_space/);
});

for (const [sourceType, label] of [
  ["tcl_recorders", "the recorder transcoder"],
  ["parallel_modal", "another writer the doc names"],
  ["made_up_writer", "an unknown source_type"],
  [null, "no source_type at all"],
] as const) {
  test(`element rows are "unverified" without the stamp for ${label}`, () => {
    assertUnverified(edited((f) => setRootAttr(f, "source_type", sourceType)), sourceType, /has no @element_index_space/);
  });
  test(`the "fem" stamp verifies element rows for ${label}`, () => {
    const r = openResults(H5, edited((f) => setRootAttr(f, "source_type", sourceType), STAMPED));
    try {
      assert.equal(r.elementIds(STAGE, "group_0").mapping, "fem_eid");
    } finally {
      r.close();
    }
  });
}

for (const [value, label, reason] of [
  ["ops", "an unknown value", /@element_index_space = "ops" is not "fem"/],
  ["FEM", "another case of the value", /@element_index_space = "FEM" is not "fem"/],
  ["", "an empty string", /@element_index_space = "" is not "fem"/],
  [1, "a non-string attribute", /@element_index_space is not a string/],
] as const) {
  test(`element_index_space set to ${label} is unverified`, () => {
    assertUnverified(edited((f) => setRootAttr(f, "element_index_space", value), STAMPED), "domain_capture", reason);
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

test("a newer results minor opens with one warning; the target and the floor open without", () => {
  const r = openResults(H5, edited((f) => setRootAttr(f, "results_schema_version", "1.3.0")));
  assert.equal(r.index.warnings.length, 1);
  assert.match(r.index.warnings[0]!, /1\.3\.0 is newer than this app \(1\.2\.x\)/);
  r.close();
  const target = openResults(H5, edited((f) => setRootAttr(f, "results_schema_version", "1.2.0")));
  assert.deepEqual(target.index.warnings, []);
  target.close();
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

// -- review round 1 (#1717): each guard below has a test that fails when it is reverted --

test("the envelope's two-part form opens as Python's SchemaVersion.parse reads it", () => {
  for (const [stamp, warns] of [
    ["1.0", 0],
    ["1.1", 0],
  ] as const) {
    const p = edited((f) => {
      setRootAttr(f, "results_schema_version", null);
      setRootAttr(f, "schema_version", stamp);
    });
    const r = openResults(H5, p);
    assert.equal(r.index.version, stamp);
    assert.equal(r.index.warnings.length, warns);
    r.close();
  }
  for (const bad of ["1", "1.0.0.0", "1.x", "", "1..0", "v1.0"]) {
    refuses(edited((f) => setRootAttr(f, "results_schema_version", bad)), "version", /is empty|is not X\.Y or X\.Y\.Z|non-integer parts/);
  }
});

function setModelMeta(f: h5wasm.File, key: string, value: string): void {
  const m = f.get("model/meta") as h5wasm.Group;
  if (Object.keys(m.attrs).includes(key)) m.delete_attribute(key);
  m.create_attribute(key, value);
}

test("an embedded neutral zone of a newer major refuses the whole file (ADR 0113 D9)", () => {
  refuses(edited((f) => setModelMeta(f, "neutral_schema_version", "3.0.0")), "version", /embedded neutral zone is a newer major/);
});

test("an embedded opensees zone of a newer major refuses the whole file", () => {
  const p = edited((f) => {
    f.create_group("opensees");
    setRootAttr(f, "opensees_schema_version", "3.1.0");
  });
  refuses(p, "version", /@opensees_schema_version 3\.1\.0: the embedded opensees zone is a newer major/);
});

test("an embedded zone below its floor leaves /stages readable and is reported unavailable", () => {
  const p = edited((f) => {
    setModelMeta(f, "neutral_schema_version", "2.5.0");
    f.create_group("opensees");
    setRootAttr(f, "opensees_schema_version", "2.12.0");
  });
  const r = openResults(H5, p);
  try {
    assert.deepEqual(r.index.unavailableZones, ["neutral"]);
    assert.equal(r.index.warnings.length, 1);
    assert.match(r.index.warnings[0]!, /neutral zone is below its floor 2\.10\.0/);
    assert.equal(r.readNodalStep(STAGE, ["displacement_x"], 1)[0]!.length, 18);
  } finally {
    r.close();
  }
  assert.deepEqual(openResults(H5, FIXTURE).index.unavailableZones, []);
});

/** A minimal gauss group: E elements, G points, components {name: shape}. */
function gaussGroup(parent: h5wasm.Group, id: string, E: number, G: number, comps: Record<string, number[]>, extra: string[] = []): void {
  const g = parent.create_group(id);
  g.create_attribute("class_tag", 4);
  g.create_attribute("int_rule", 0);
  g.create_dataset({ name: "_element_index", data: new BigInt64Array(E).map((_, i) => BigInt(i + 1)) });
  g.create_dataset({ name: "_natural_coords", data: new Float64Array(G * 2), shape: [G, 2] });
  for (const [name, shape] of Object.entries(comps)) {
    g.create_dataset({ name, data: new Float64Array(shape.reduce((a, b) => a * b, 1)), shape });
  }
  for (const name of extra) g.create_dataset({ name, data: new Float64Array([0]) });
}

function oneStageWithElements(build: (elements: h5wasm.Group) => void): string {
  return minimal((s) => {
    s.create_dataset({ name: "time", data: new Float64Array([0, 1]) });
    build(s.create_group("partitions").create_group("partition_0").create_group("elements"));
  });
}

test("stages come back in stage_<int> order, not HDF5's listing order", () => {
  const p = join(mkdtempSync(join(tmpdir(), "agv-results-")), "stages.results.h5");
  const f = new h5wasm.File(p, "w");
  f.create_attribute("results_schema_version", "1.1.0");
  const st = f.create_group("stages");
  for (const id of ["stage_10", "stage_2", "stage_0"]) {
    const s = st.create_group(id);
    s.create_attribute("name", id);
    s.create_attribute("kind", "static");
    s.create_dataset({ name: "time", data: new Float64Array([0]) });
    s.create_group("partitions").create_group("partition_0");
  }
  f.close();
  const r = openResults(H5, p);
  assert.deepEqual(r.index.stages.map((s) => s.id), ["stage_0", "stage_2", "stage_10"]);
  r.close();
});

test("an unknown element category is refused", () => {
  refuses(oneStageWithElements((e) => e.create_group("springs")), "layout", /elements\/springs: not an element category/);
});

test("an element category the spike does not read is listed in unread, not dropped", () => {
  const p = oneStageWithElements((e) => {
    e.create_group("fibers").create_group("group_0");
    gaussGroup(e.create_group("gauss_points"), "group_0", 3, 2, { stress_xx: [2, 3, 2] });
  });
  const r = openResults(H5, p);
  try {
    assert.deepEqual(r.stage(STAGE).unread, ["fibers (1 groups)"]);
    assert.equal(r.stage(STAGE).gauss.length, 1);
  } finally {
    r.close();
  }
});

test("a gauss component whose shape is not (T, E, G) is refused", () => {
  refuses(
    oneStageWithElements((e) => gaussGroup(e.create_group("gauss_points"), "group_0", 3, 2, { stress_xx: [2, 3, 1] })),
    "layout",
    /stress_xx has shape \(2, 3, 1\); expected \(2, 3, 2\)/,
  );
});

test("an unknown _ index dataset in a gauss group is refused", () => {
  refuses(
    oneStageWithElements((e) => gaussGroup(e.create_group("gauss_points"), "group_0", 3, 2, { stress_xx: [2, 3, 2] }, ["_weights"])),
    "layout",
    /group_0\/_weights: not a gauss index dataset/,
  );
});

test("a stage kind outside static, transient and mode is refused", () => {
  const p = edited((f) => {
    const s = f.get(`stages/${STAGE}`) as h5wasm.Group;
    s.delete_attribute("kind");
    s.create_attribute("kind", "dynamic");
  });
  refuses(p, "layout", /@kind = "dynamic"/);
});

test("an unknown stage child is refused", () => {
  refuses(edited((f) => (f.get(`stages/${STAGE}`) as h5wasm.Group).create_group("notes")), "layout", /stage_0\/notes: not a stage child/);
});

test("a negative or fractional step is refused", () => {
  const r = openResults(H5, FIXTURE);
  try {
    for (const k of [-1, 1.5]) {
      assert.throws(() => r.readNodalStep(STAGE, ["displacement_x"], k), (e: unknown) => e instanceof ResultsError && e.kind === "argument");
      assert.throws(() => r.readGaussStep(STAGE, "group_0", "stress_xx", k), (e: unknown) => e instanceof ResultsError && e.kind === "argument");
    }
  } finally {
    r.close();
  }
});

test("stage ids sort by their integer suffix, then lexically", () => {
  assert.deepEqual(sortStageIds(["stage_10", "b", "stage_2", "a", "stage_0"]), ["stage_0", "stage_2", "stage_10", "a", "b"]);
});
