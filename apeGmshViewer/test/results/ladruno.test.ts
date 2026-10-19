// reader/ladruno.ts against fork-written .ladruno fixtures (ADR 0122 slice 4a, #2126).
// fixtures/make_ladruno_fixtures.py says how each was made. The expected values are
// closed forms of the models, recomputed here, not values copied from the files:
//
// ladruno_truss2d*.ladruno: two trusses (EA = 1000, L = 1) in a line, a load of 10
//   at node 3, LoadControl 0.25 for 4 steps, so time[k] = 0.25 (k + 1) and
//   Ux(node j, t) = 10 (j - 1) t / 1000, Uy = 0. The `.stamped` copy carries the
//   §7b.1 stamp below; the `.badstamp` copy carries one in the wrong encodings.
// ladruno_modes.ladruno: two trusses (k = EA / L = 1000) along x, node 1 fixed,
//   unit masses in x at nodes 2 and 3: lambda = k (3 -/+ sqrt 5) / 2, with
//   Ux3 / Ux2 = 2 - lambda / k and unit mass-normalised shapes (m = 1).
//
// Each refusal (GENERATOR, FORMAT_VERSION, a part file, an unknown KIND, an
// inconsistent modal record) is a typed LadrunoError, and each degenerate nodal
// result (COMPONENTS empty or the wrong count, step axes that disagree) is refused
// by name in NodalInfo.refused while the rest of the file reads.

import assert from "node:assert/strict";
import { copyFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { LadrunoError, openLadruno, type LadrunoFile } from "../../src/reader/ladruno.ts";
import type { ResultsH5 } from "../../src/reader/results.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as ResultsH5;
const fixture = (name: string) => fileURLToPath(new URL(`../../fixtures/${name}`, import.meta.url));
const TRUSS = fixture("ladruno_truss2d.ladruno");
const STAMPED = fixture("ladruno_truss2d.stamped.ladruno");
const BADSTAMP = fixture("ladruno_truss2d.badstamp.ladruno");
const MODES = fixture("ladruno_modes.ladruno");
const NATIVE = fixture("results_small.h5");
const DISP = "/MODEL_STAGE[1]/RESULTS/ON_NODES/DISPLACEMENT";

const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));
const ux = (node: number, t: number) => (10 * (node - 1) * t) / 1000;

/** A scratch copy of a fixture under `name`, edited through h5wasm in read-write mode. */
function edited(from: string, edit: (f: h5wasm.File) => void, name = "edited.ladruno"): string {
  const p = join(mkdtempSync(join(tmpdir(), "agv-ladruno-")), name);
  copyFileSync(from, p);
  const f = new h5wasm.File(p, "a");
  try {
    edit(f);
  } finally {
    f.close();
  }
  return p;
}

function setAttr(node: { attrs: Record<string, unknown>; delete_attribute(n: string): number; create_attribute(n: string, v: string | number): void }, key: string, value: string | number | null): void {
  if (Object.keys(node.attrs).includes(key)) node.delete_attribute(key);
  if (value !== null) node.create_attribute(key, value);
}

function refuses(path: string, kind: string, pattern: RegExp): void {
  assert.throws(
    () => openLadruno(H5, path),
    (e: unknown) => e instanceof LadrunoError && e.kind === kind && pattern.test(e.message),
  );
}

function withFile<T>(path: string, body: (r: LadrunoFile) => T): T {
  const r = openLadruno(H5, path);
  try {
    return body(r);
  } finally {
    r.close();
  }
}

// -- the gate ------------------------------------------------------------------

test("the gate refuses another GENERATOR, before any other read", () => {
  refuses(edited(TRUSS, (f) => setAttr(f.get("INFO") as h5wasm.Group, "GENERATOR", "MPCO")), "generator", /GENERATOR is "MPCO"/);
  // A broken stage behind a wrong GENERATOR: the gate speaks first.
  const both = edited(TRUSS, (f) => {
    setAttr(f.get("INFO") as h5wasm.Group, "GENERATOR", "STKO");
    setAttr(f.get("MODEL_STAGE[1]") as h5wasm.Group, "KIND", "dynamic");
  });
  refuses(both, "generator", /GENERATOR is "STKO"/);
  // A native results.h5 has no INFO group at all.
  refuses(NATIVE, "generator", /no INFO group/);
  refuses(edited(TRUSS, (f) => setAttr(f.get("INFO") as h5wasm.Group, "GENERATOR", null)), "generator", /no INFO@GENERATOR/);
});

test("the gate refuses a FORMAT_VERSION outside the supported set", () => {
  refuses(edited(TRUSS, (f) => setAttr(f.get("INFO") as h5wasm.Group, "FORMAT_VERSION", 2)), "format_version", /FORMAT_VERSION is "2"; this app reads 1/);
  refuses(edited(TRUSS, (f) => setAttr(f.get("INFO") as h5wasm.Group, "FORMAT_VERSION", 0)), "format_version", /FORMAT_VERSION is "0"/);
  refuses(edited(TRUSS, (f) => setAttr(f.get("INFO") as h5wasm.Group, "FORMAT_VERSION", "1")), "format_version", /FORMAT_VERSION is "1"/);
  refuses(edited(TRUSS, (f) => setAttr(f.get("INFO") as h5wasm.Group, "FORMAT_VERSION", null)), "format_version", /FORMAT_VERSION is absent/);
  // and a format_version refusal also precedes the stage checks
  const both = edited(TRUSS, (f) => {
    setAttr(f.get("INFO") as h5wasm.Group, "FORMAT_VERSION", 2);
    setAttr(f.get("MODEL_STAGE[1]") as h5wasm.Group, "KIND", "dynamic");
  });
  refuses(both, "format_version", /FORMAT_VERSION is "2"/);
});

test("a part file is refused by its name, and a file whose INFO says it is a part is refused too", () => {
  refuses(edited(TRUSS, () => {}, "run.part-0.ladruno"), "unsupported", /part 0 of the partitioned set run\.part-<N>\.ladruno/);
  refuses(edited(TRUSS, () => {}, "run.part-12.ladruno"), "unsupported", /part 12/);
  refuses(edited(TRUSS, (f) => setAttr(f.get("INFO") as h5wasm.Group, "PARTITIONED", 1)), "unsupported", /one part of a partitioned run/);
  refuses(edited(TRUSS, (f) => setAttr(f.get("INFO") as h5wasm.Group, "NUM_PARTITIONS", 2)), "unsupported", /NUM_PARTITIONS = 2/);
});

test("an unreadable file is an open error", () => {
  const p = join(mkdtempSync(join(tmpdir(), "agv-ladruno-")), "junk.ladruno");
  copyFileSync(fileURLToPath(new URL("../../fixtures/README.md", import.meta.url)), p);
  refuses(p, "open", /could not open/);
});

// -- nodes ---------------------------------------------------------------------

test("the truss indexes one static stage, its time and its nodal components as the file spells them", () => {
  withFile(TRUSS, (r) => {
    const { index } = r;
    assert.equal(index.generator, "Ladruno");
    assert.equal(index.formatVersion, 1);
    assert.equal(index.formatMinorVersion, 0);
    assert.equal(index.spatialDim, 2);
    assert.deepEqual(index.warnings, []);
    assert.deepEqual(index.modes, []);
    assert.equal(index.stages.length, 1);
    const s = index.stages[0]!;
    assert.equal(s.id, "stage_0");
    assert.equal(s.name, "MODEL_STAGE[1]");
    assert.equal(s.kind, "static");
    assert.equal(index.fileKinds["stage_0"], "static");
    assert.equal(s.nSteps, 4);
    for (let k = 0; k < 4; k++) assert.ok(close(s.time[k]!, 0.25 * (k + 1)), `time[${k}]`);
    assert.deepEqual(Array.from(s.nodes!.ids), [1, 2, 3]);
    assert.deepEqual(s.nodes!.components, ["DISPLACEMENT/Ux", "DISPLACEMENT/Uy"]);
    assert.deepEqual(s.nodes!.refused, []);
    // the element result is listed, not skipped silently
    assert.deepEqual(s.unread, ["ON_ELEMENTS/basicForce"]);
    assert.deepEqual(s.gauss, []);
  });
});

test("one step by hyperslab, a row range, and one node's history match the closed form", () => {
  withFile(TRUSS, (r) => {
    for (let k = 0; k < 4; k++) {
      const t = 0.25 * (k + 1);
      const [x, y] = r.readNodalStep("stage_0", ["DISPLACEMENT/Ux", "DISPLACEMENT/Uy"], k);
      assert.equal(x!.length, 3);
      [1, 2, 3].forEach((node, j) => {
        assert.ok(close(x![j]!, ux(node, t)), `Ux node ${node} step ${k}: ${x![j]}`);
        assert.equal(y![j], 0);
      });
    }
    const [rows] = r.readNodalRows("stage_0", ["DISPLACEMENT/Ux"], 1, 3);
    assert.equal(rows!.length, 2 * 3);
    for (let k = 1; k < 3; k++) for (let j = 0; j < 3; j++) assert.ok(close(rows![(k - 1) * 3 + j]!, ux(j + 1, 0.25 * (k + 1))));
    const h = r.readNodeHistory("stage_0", "DISPLACEMENT/Ux", 3);
    assert.deepEqual(Array.from(h).map((v) => +v.toFixed(12)), [0.005, 0.01, 0.015, 0.02]);
  });
});

test("reads outside the index are argument errors", () => {
  withFile(TRUSS, (r) => {
    const arg = (fn: () => unknown, pattern: RegExp) =>
      assert.throws(fn, (e: unknown) => e instanceof LadrunoError && e.kind === "argument" && pattern.test(e.message));
    arg(() => r.readNodalStep("stage_0", ["DISPLACEMENT/Ux"], 4), /step 4 is outside/);
    arg(() => r.readNodalStep("stage_0", ["displacement_x"], 0), /no component "displacement_x"; it has DISPLACEMENT\/Ux, DISPLACEMENT\/Uy/);
    arg(() => r.readNodeHistory("stage_0", "DISPLACEMENT/Ux", 99), /records no node 99/);
    arg(() => r.readNodalRows("stage_0", ["DISPLACEMENT/Ux"], 2, 2), /not a range/);
    arg(() => r.stage("stage_9"), /no stage "stage_9"/);
  });
});

// -- seams: degenerate names and partial writes -------------------------------

test("a COMPONENTS count that disagrees with DATA, or an empty label, refuses that result by name", () => {
  const short = edited(TRUSS, (f) => setAttr(f.get(DISP) as h5wasm.Group, "COMPONENTS", "Ux"));
  withFile(short, (r) => {
    const n = r.index.stages[0]!.nodes!;
    assert.deepEqual(n.components, []);
    assert.equal(n.refused.length, 1);
    assert.equal(n.refused[0]!.component, "DISPLACEMENT");
    assert.match(n.refused[0]!.reason, /2 component columns and COMPONENTS names 1 \(Ux\)/);
    assert.throws(() => r.readNodalStep("stage_0", ["DISPLACEMENT/Ux"], 0), (e: unknown) => e instanceof LadrunoError && e.kind === "argument");
  });
  for (const labels of ["", "Ux,"]) {
    withFile(edited(TRUSS, (f) => setAttr(f.get(DISP) as h5wasm.Group, "COMPONENTS", labels)), (r) => {
      const n = r.index.stages[0]!.nodes!;
      assert.deepEqual(n.components, []);
      assert.match(n.refused[0]!.reason, /has an empty label/);
    });
  }
  withFile(edited(TRUSS, (f) => setAttr(f.get(DISP) as h5wasm.Group, "COMPONENTS", null)), (r) => {
    assert.match(r.index.stages[0]!.nodes!.refused[0]!.reason, /has no COMPONENTS attribute/);
  });
});

test("DATA's step axis against TIME and STEP (a recorder killed mid-run) refuses that result and says why", () => {
  for (const axis of ["TIME", "STEP"]) {
    const p = edited(TRUSS, (f) => (f.get(`${DISP}/${axis}`) as h5wasm.Dataset).resize([3]));
    withFile(p, (r) => {
      const n = r.index.stages[0]!.nodes!;
      assert.deepEqual(n.components, []);
      assert.equal(n.refused[0]!.component, "DISPLACEMENT");
      assert.match(n.refused[0]!.reason, new RegExp(`holds 4 steps, TIME ${axis === "TIME" ? 3 : 4} and STEP ${axis === "STEP" ? 3 : 4}: the step axes disagree`));
    });
  }
});

test("a second nodal result on another time axis is refused alone; the first still reads", () => {
  const p = edited(TRUSS, (f) => {
    const g = f.create_group("MODEL_STAGE[1]/RESULTS/ON_NODES/VELOCITY");
    g.create_attribute("COMPONENTS", "Vx,Vy");
    g.create_dataset({ name: "ID", data: new Int32Array([1, 2, 3]), shape: [3, 1] });
    g.create_dataset({ name: "DATA", data: new Float64Array(3 * 3 * 2), shape: [3, 3, 2] });
    g.create_dataset({ name: "TIME", data: new Float64Array([0.25, 0.5, 0.75]) });
    g.create_dataset({ name: "STEP", data: new Int32Array([0, 1, 2]) });
  });
  withFile(p, (r) => {
    const n = r.index.stages[0]!.nodes!;
    assert.deepEqual(n.components, ["DISPLACEMENT/Ux", "DISPLACEMENT/Uy"]);
    assert.equal(n.refused.length, 1);
    assert.equal(n.refused[0]!.component, "VELOCITY");
    assert.match(n.refused[0]!.reason, /TIME is not the stage's time/);
  });
});

/** The truss plus a second ON_NODES result VELOCITY on the stage's time axis, built by `fill`. */
function withVelocity(fill: (g: h5wasm.Group) => void): string {
  return edited(TRUSS, (f) => {
    const g = f.create_group("MODEL_STAGE[1]/RESULTS/ON_NODES/VELOCITY");
    g.create_attribute("COMPONENTS", "Vx,Vy");
    g.create_dataset({ name: "TIME", data: new Float64Array([0.25, 0.5, 0.75, 1.0]) });
    g.create_dataset({ name: "STEP", data: new Int32Array([0, 1, 2, 3]) });
    fill(g);
  });
}

/** The one refusal of VELOCITY, with DISPLACEMENT still read. */
function velocityRefused(p: string): string {
  return withFile(p, (r) => {
    const n = r.index.stages[0]!.nodes!;
    assert.deepEqual(n.components, ["DISPLACEMENT/Ux", "DISPLACEMENT/Uy"]);
    assert.equal(n.refused.length, 1, JSON.stringify(n.refused));
    assert.equal(n.refused[0]!.component, "VELOCITY");
    return n.refused[0]!.reason;
  });
}

test("a second result is refused alone when it records another node set, or node rows that disagree with its ID", () => {
  // same count, other ids
  const otherSet = withVelocity((g) => {
    g.create_dataset({ name: "ID", data: new Int32Array([1, 2, 4]), shape: [3, 1] });
    g.create_dataset({ name: "DATA", data: new Float64Array(4 * 3 * 2), shape: [4, 3, 2] });
  });
  assert.match(velocityRefused(otherSet), /VELOCITY\/ID is not the node set of .*DISPLACEMENT\/ID \(3 and 3 nodes\)/);
  // 2 ids, 3 node rows
  const rows = withVelocity((g) => {
    g.create_dataset({ name: "ID", data: new Int32Array([1, 2]), shape: [2, 1] });
    g.create_dataset({ name: "DATA", data: new Float64Array(4 * 3 * 2), shape: [4, 3, 2] });
  });
  assert.match(velocityRefused(rows), /VELOCITY\/DATA has 3 node rows and .*VELOCITY\/ID 2 node ids \(a partial write\)/);
  // the same file with matching rows reads both results: the refusals above are the seams, not the fixture
  const fine = withVelocity((g) => {
    g.create_dataset({ name: "ID", data: new Int32Array([1, 2, 3]), shape: [3, 1] });
    g.create_dataset({ name: "DATA", data: new Float64Array(4 * 3 * 2), shape: [4, 3, 2] });
  });
  withFile(fine, (r) => assert.deepEqual(r.index.stages[0]!.nodes!.components, ["DISPLACEMENT/Ux", "DISPLACEMENT/Uy", "VELOCITY/Vx", "VELOCITY/Vy"]));
});

test("a per-step DATA group (the superseded schema §7.0 layout) refuses that result by name", () => {
  const p = withVelocity((g) => {
    g.create_dataset({ name: "ID", data: new Int32Array([1, 2, 3]), shape: [3, 1] });
    g.create_group("DATA").create_dataset({ name: "STEP_0", data: new Float64Array(3 * 2), shape: [3, 2] });
  });
  assert.match(velocityRefused(p), /VELOCITY\/DATA is a group of per-step datasets/);
});

test("a COMPONENTS label repeated refuses that result by name", () => {
  withFile(edited(TRUSS, (f) => setAttr(f.get(DISP) as h5wasm.Group, "COMPONENTS", "Ux,Ux")), (r) => {
    const n = r.index.stages[0]!.nodes!;
    assert.deepEqual(n.components, []);
    assert.equal(n.refused[0]!.component, "DISPLACEMENT");
    assert.match(n.refused[0]!.reason, /COMPONENTS = "Ux,Ux" repeats a label/);
  });
});

test("stages sort by their integer id; one with no ON_NODES has no nodal block", () => {
  const p = edited(TRUSS, (f) => {
    const ten = f.create_group("MODEL_STAGE[10]");
    ten.create_attribute("KIND", "static");
    ten.create_group("RESULTS");
    const two = f.create_group("MODEL_STAGE[2]");
    two.create_attribute("KIND", "transient");
  });
  withFile(p, (r) => {
    assert.deepEqual(r.index.stages.map((s) => s.name), ["MODEL_STAGE[1]", "MODEL_STAGE[2]", "MODEL_STAGE[10]"]);
    assert.deepEqual(r.index.stages.map((s) => s.id), ["stage_0", "stage_1", "stage_2"]);
    const [, two, ten] = r.index.stages;
    assert.equal(two!.kind, "transient");
    assert.equal(two!.nodes, null);
    assert.equal(two!.nSteps, 0);
    assert.equal(ten!.nodes, null);
    assert.throws(() => r.readNodeHistory("stage_2", "DISPLACEMENT/Ux", 1), (e: unknown) => e instanceof LadrunoError && e.kind === "argument");
  });
});

test("an unknown or absent stage KIND is refused, never read as static", () => {
  refuses(edited(TRUSS, (f) => setAttr(f.get("MODEL_STAGE[1]") as h5wasm.Group, "KIND", "dynamic")), "layout", /KIND = "dynamic"; expected static, transient or eigen/);
  refuses(edited(TRUSS, (f) => setAttr(f.get("MODEL_STAGE[1]") as h5wasm.Group, "KIND", null)), "layout", /KIND = null/);
  refuses(edited(TRUSS, (f) => f.create_group("MODEL_STAGE[x]")), "layout", /MODEL_STAGE\[x\]: a MODEL_STAGE whose id is not an integer/);
});

// -- modes ---------------------------------------------------------------------

const K = 1000;
const LAMBDAS = [(K * (3 - Math.sqrt(5))) / 2, (K * (3 + Math.sqrt(5))) / 2];

test("KIND=eigen is a static host stage; each recorded mode is its own stage of kind mode, after its host", () => {
  withFile(MODES, (r) => {
    const { index } = r;
    assert.deepEqual(index.warnings, []);
    assert.deepEqual(index.stages.map((s) => [s.id, s.name, s.kind, s.nSteps]), [
      ["stage_0", "MODEL_STAGE[1]", "static", 1],
      ["stage_0_mode_1", "mode_1", "mode", 1],
      ["stage_0_mode_2", "mode_2", "mode", 1],
    ]);
    assert.equal(index.fileKinds["stage_0"], "eigen");
    assert.deepEqual(index.stages[0]!.nodes!.components, ["DISPLACEMENT/Ux", "DISPLACEMENT/Uy"]);
    const mode = index.stages[1]!;
    assert.deepEqual(Array.from(mode.time), [0]);
    assert.deepEqual(Array.from(mode.nodes!.ids), [1, 2, 3]);
    assert.deepEqual(mode.nodes!.components, ["MODES_OF_VIBRATION(U)/Ux", "MODES_OF_VIBRATION(U)/Uy", "MODES_OF_VIBRATION(R)/Rz"]);
    assert.deepEqual(mode.nodes!.refused, []);
    assert.equal(r.mode("stage_0"), null);
    index.modes.forEach((m, i) => {
      assert.equal(m.stageId, `stage_0_mode_${i + 1}`);
      assert.equal(m.hostStageId, "stage_0");
      assert.equal(m.modeIndex, i + 1);
      assert.equal(m.stepKey, "STEP_0");
      assert.equal(m.modeKey, `MODE_${i}`);
      assert.ok(close(m.eigenvalue, LAMBDAS[i]!), `lambda ${i + 1}: ${m.eigenvalue}`);
      const omega = Math.sqrt(LAMBDAS[i]!);
      assert.ok(close(m.frequencyHz, omega / (2 * Math.PI)));
      assert.ok(close(m.periodS, (2 * Math.PI) / omega));
      assert.deepEqual(r.mode(m.stageId), m);
    });
  });
});

test("a mode shape reads as one step: Ux3 / Ux2 = 2 - lambda / k, unit mass-normalised, and no rotation", () => {
  withFile(MODES, (r) => {
    LAMBDAS.forEach((lam, i) => {
      const id = `stage_0_mode_${i + 1}`;
      const [x, y, rz] = r.readNodalStep(id, ["MODES_OF_VIBRATION(U)/Ux", "MODES_OF_VIBRATION(U)/Uy", "MODES_OF_VIBRATION(R)/Rz"], 0);
      assert.equal(x![0], 0);
      assert.ok(close(x![2]! / x![1]!, 2 - lam / K, 1e-9), `mode ${i + 1}: ${x![2]! / x![1]!}`);
      assert.ok(close(x![1]! ** 2 + x![2]! ** 2, 1, 1e-9));
      assert.deepEqual(Array.from(y!), [0, 0, 0]);
      assert.deepEqual(Array.from(rz!), [0, 0, 0]);
      const h = r.readNodeHistory(id, "MODES_OF_VIBRATION(U)/Ux", 3);
      assert.deepEqual(Array.from(h), [x![2]]);
      assert.deepEqual(r.readNodalRows(id, ["MODES_OF_VIBRATION(U)/Ux"], 0, 1)[0], x);
      assert.throws(() => r.readNodalStep(id, ["MODES_OF_VIBRATION(U)/Ux"], 1), (e: unknown) => e instanceof LadrunoError && e.kind === "argument");
    });
  });
});

const MODAL = "/MODEL_STAGE[1]/RESULTS/ON_NODES";

test("a modal group whose MODE shape disagrees with its COMPONENTS is refused in each mode stage; the other group reads", () => {
  withFile(edited(MODES, (f) => setAttr(f.get(`${MODAL}/MODES_OF_VIBRATION(R)`) as h5wasm.Group, "COMPONENTS", "Rz,Rx")), (r) => {
    for (const id of ["stage_0_mode_1", "stage_0_mode_2"]) {
      const n = r.stage(id).nodes!;
      assert.deepEqual(n.components, ["MODES_OF_VIBRATION(U)/Ux", "MODES_OF_VIBRATION(U)/Uy"]);
      assert.equal(n.refused.length, 1);
      assert.equal(n.refused[0]!.component, "MODES_OF_VIBRATION(R)");
      assert.match(n.refused[0]!.reason, /MODE_\d has shape \(3, 1\); ID has 3 nodes and COMPONENTS names 2 \(a partial write\)/);
    }
  });
});

test("an inconsistent modal record refuses the file: a mode without LAMBDA, or (R) disagreeing with (U)", () => {
  refuses(
    edited(MODES, (f) => (f.get(`${MODAL}/MODES_OF_VIBRATION(U)/DATA/STEP_0/MODE_0`) as h5wasm.Dataset).delete_attribute("LAMBDA")),
    "layout",
    /MODE_0 has no numeric LAMBDA attribute/,
  );
  refuses(
    edited(MODES, (f) => setAttr(f.get(`${MODAL}/MODES_OF_VIBRATION(R)/DATA/STEP_0/MODE_1`) as h5wasm.Dataset, "LAMBDA", 1.5)),
    "layout",
    /MODES_OF_VIBRATION\(R\) lists different modes or eigenvalues than .*MODES_OF_VIBRATION\(U\)/,
  );
});

// -- INFO/PROVENANCE -------------------------------------------------------------

const STAMP = {
  model_hash: "0123456789abcdef0123456789abcdef",
  fem_hash: "fedcba9876543210fedcba9876543210",
  stage_names: '["push"]',
  element_index_space: "fem",
  unsupported: "[]",
  writer: "apeGmsh fixture (make_ladruno_fixtures.py)",
};

test("an unstamped file has an empty PROVENANCE: every attribute absent, no lineage", () => {
  withFile(TRUSS, (r) => {
    const p = r.index.provenance;
    assert.equal(p.state, "present");
    if (p.state !== "present") return;
    for (const v of Object.values(p.attrs)) assert.deepEqual(v, { state: "absent" });
    assert.equal(p.stageNames, null);
    assert.equal(p.unsupported, null);
    assert.deepEqual(p.other, []);
    assert.equal(r.index.lineage, null);
    assert.match(r.index.lineageReason!, /no @model_hash; INFO\/PROVENANCE has no @fem_hash: the file is not stamped/);
  });
});

test("a §7b.1 stamp reads attribute by attribute, and gives the lineage", () => {
  withFile(STAMPED, (r) => {
    const p = r.index.provenance;
    assert.equal(p.state, "present");
    if (p.state !== "present") return;
    for (const [k, v] of Object.entries(STAMP)) assert.deepEqual(p.attrs[k as keyof typeof STAMP], { state: "present", value: v }, k);
    assert.deepEqual(p.stageNames, ["push"]);
    assert.deepEqual(p.unsupported, []);
    assert.deepEqual(r.index.lineage, { modelHash: STAMP.model_hash, femHash: STAMP.fem_hash });
    assert.equal(r.index.lineageReason, null);
  });
});

test("a stamp in the wrong encodings is reported as such, never coerced, and gives no lineage", () => {
  withFile(BADSTAMP, (r) => {
    const p = r.index.provenance;
    assert.equal(p.state, "present");
    if (p.state !== "present") return;
    const reason = (k: keyof typeof STAMP) => {
      const f = p.attrs[k];
      assert.equal(f.state, "malformed", k);
      return f.state === "malformed" ? f.reason : "";
    };
    assert.match(reason("model_hash"), /fixed-length string/);
    assert.match(reason("fem_hash"), /ASCII-charset string/);
    assert.match(reason("stage_names"), /not a JSON array of strings/);
    assert.match(reason("unsupported"), /not a string/);
    assert.deepEqual(p.attrs.element_index_space, { state: "present", value: "fem" });
    // vlen UTF-8 at shape (1,): the shape is what is wrong, and the reason says so
    assert.match(reason("writer"), /string array of shape \(1\), not a scalar/);
    assert.equal(p.stageNames, null);
    assert.equal(r.index.lineage, null);
    assert.match(r.index.lineageReason!, /fixed-length string.*; .*ASCII-charset/);
  });
});

test("a value outside §7b.1, an unknown attribute, and a missing group are reported", () => {
  const p = edited(STAMPED, (f) => {
    const g = f.get("INFO/PROVENANCE") as h5wasm.Group;
    setAttr(g, "element_index_space", "ops");
    setAttr(g, "fem_hash", null);
    setAttr(g, "MODEL_HASH", "x");
  });
  withFile(edited(STAMPED, (f) => setAttr(f.get("INFO/PROVENANCE") as h5wasm.Group, "model_hash", "")), (r) => {
    const prov = r.index.provenance;
    assert.equal(prov.state, "present");
    if (prov.state !== "present") return;
    assert.deepEqual(prov.attrs.model_hash, { state: "malformed", reason: "@model_hash is an empty string" });
    assert.equal(r.index.lineage, null);
  });
  withFile(p, (r) => {
    const prov = r.index.provenance;
    assert.equal(prov.state, "present");
    if (prov.state !== "present") return;
    assert.equal(prov.attrs.element_index_space.state, "malformed");
    assert.deepEqual(prov.attrs.fem_hash, { state: "absent" });
    assert.deepEqual(prov.other, ["MODEL_HASH"]);
    assert.equal(r.index.lineage, null);
    assert.match(r.index.lineageReason!, /^INFO\/PROVENANCE has no @fem_hash/);
  });
  // h5wasm cannot delete a group, so the absent-group branch reads a minimal file built without one.
  const noGroup = join(mkdtempSync(join(tmpdir(), "agv-ladruno-")), "noprov.ladruno");
  const w = new h5wasm.File(noGroup, "w");
  try {
    const info = w.create_group("INFO");
    info.create_attribute("GENERATOR", "Ladruno");
    info.create_attribute("FORMAT_VERSION", 1);
  } finally {
    w.close();
  }
  withFile(noGroup, (r) => {
    assert.equal(r.index.provenance.state, "absent");
    assert.deepEqual(r.index.stages, []);
    assert.equal(r.index.lineage, null);
    assert.match(r.index.lineageReason!, /no INFO\/PROVENANCE group/);
  });
});

test("INFO/PROVENANCE@units reads a known UnitSystemId into index.units and marks an unknown id malformed", () => {
  const withUnits = edited(STAMPED, (f) => {
    setAttr(f.get("INFO/PROVENANCE") as h5wasm.Group, "units", "N_mm_s");
  });
  withFile(withUnits, (r) => {
    assert.equal(r.index.units, "N_mm_s");
    const prov = r.index.provenance;
    assert.equal(prov.state, "present");
    if (prov.state === "present") {
      assert.deepEqual(prov.attrs.units, { state: "present", value: "N_mm_s" });
      assert.equal(prov.units, "N_mm_s");
    }
  });
  const badUnits = edited(STAMPED, (f) => {
    setAttr(f.get("INFO/PROVENANCE") as h5wasm.Group, "units", "slug_furlong_fortnight");
  });
  withFile(badUnits, (r) => {
    assert.equal(r.index.units, null);
    const prov = r.index.provenance;
    assert.equal(prov.state, "present");
    if (prov.state === "present") {
      assert.equal(prov.attrs.units.state, "malformed");
      assert.equal(prov.units, null);
    }
  });
});

// -- line stations & local axes ------------------------------------------------

test("TRUSS basicForce reads single-station axial force N across steps", () => {
  withFile(TRUSS, (r) => {
    assert.deepEqual(r.availableLineStations("stage_0"), ["N"]);
    const slab = r.readLineStations("stage_0", "N");
    assert.equal(slab.component, "N");
    assert.equal(slab.nSteps, 4);
    assert.equal(slab.nStationsTotal, 2);
    assert.deepEqual(Array.from(slab.elementIndex), [1, 2]);
    assert.deepEqual(Array.from(slab.stationNaturalCoord), [0, 0]);
    // EA = 1000, L = 1, F = 10 * t => N = 2.5, 5.0, 7.5, 10.0 in both trusses
    for (let k = 0; k < 4; k++) {
      const expected = 2.5 * (k + 1);
      assert.ok(close(slab.values[k * 2 + 0]!, expected));
      assert.ok(close(slab.values[k * 2 + 1]!, expected));
      const step = r.readLineStationStep("stage_0", "N", k);
      assert.ok(close(step.values[0]!, expected));
      assert.ok(close(step.values[1]!, expected));
    }
    const emptyVy = r.readLineStations("stage_0", "Vy");
    assert.equal(emptyVy.nStationsTotal, 0);
    assert.equal(emptyVy.values.length, 0);
  });
});

test("line station precedence (section.force > localForce > basicForce > axialForce), GP_PARAM, and 2D/3D localForce gate", () => {
  const p2d = edited(TRUSS, (f) => {
    const stage = f.get("MODEL_STAGE[1]") as h5wasm.Group;
    const elems = stage.get("RESULTS/ON_ELEMENTS") as h5wasm.Group;
    const timeArr = new Float64Array([0.25, 0.5, 0.75, 1.0]);
    const stepArr = new Int32Array([0, 1, 2, 3]);

    // Add localForce for element 1 & 2 (2D: N_1, V_1, M_1, N_2, V_2, M_2)
    const lf = elems.create_group("localForce");
    const lfBlock = lf.create_group("10-ElasticBeam2d[0]");
    lfBlock.create_dataset({ name: "ID", data: new Int32Array([1, 2]), shape: [2] });
    lfBlock.create_dataset({ name: "TIME", data: timeArr });
    lfBlock.create_dataset({ name: "STEP", data: stepArr });
    // Step 0..3: elem 1 has end_1=(N_1=-100, V_1=20, M_1=-50), end_2=(N_2=100, V_2=-20, M_2=80)
    // elem 2 has end_1=(N_1=-200, V_1=30, M_1=-60), end_2=(N_2=200, V_2=-30, M_2=90)
    const lfData = new Float64Array(4 * 2 * 6);
    for (let k = 0; k < 4; k++) {
      const s = k + 1;
      lfData.set([-100 * s, 20 * s, -50 * s, 100 * s, -20 * s, 80 * s], k * 12 + 0);
      lfData.set([-200 * s, 30 * s, -60 * s, 200 * s, -30 * s, 90 * s], k * 12 + 6);
    }
    lfBlock.create_dataset({ name: "DATA", data: lfData, shape: [4, 2, 6] });
    const lfMap = lfBlock.create_group("COLUMN_MAP");
    lfMap.create_attribute("COMP_NAMES", "N_1,V_1,M_1,N_2,V_2,M_2");
    lfMap.create_dataset({ name: "LEVELS", data: new Int32Array([0]), shape: [1] });
    lfMap.create_dataset({ name: "GAUSS_ID", data: new Int32Array([-1]), shape: [1] });

    // Add section.force for element 1 only (3 Gauss stations: xi = [-0.7746, 0, 0.7746])
    const sf = elems.create_group("section.force");
    const sfBlock = sf.create_group("20-ForceBeamColumn2d[0]");
    sfBlock.create_dataset({ name: "ID", data: new Int32Array([1]), shape: [1] });
    sfBlock.create_dataset({ name: "TIME", data: timeArr });
    sfBlock.create_dataset({ name: "STEP", data: stepArr });
    const sfData = new Float64Array(4 * 1 * 6);
    for (let k = 0; k < 4; k++) {
      const s = k + 1;
      // 3 stations x (P, Mz): (111*s, 11*s), (112*s, 12*s), (113*s, 13*s)
      sfData.set([111 * s, 11 * s, 112 * s, 12 * s, 113 * s, 13 * s], k * 6);
    }
    sfBlock.create_dataset({ name: "DATA", data: sfData, shape: [4, 1, 6] });
    const sfMap = sfBlock.create_group("COLUMN_MAP");
    sfMap.create_attribute("COMP_NAMES", "P,Mz\nP,Mz\nP,Mz");
    sfMap.create_dataset({ name: "LEVELS", data: new Int32Array([2, 2, 2]), shape: [3] });
    sfMap.create_dataset({ name: "GAUSS_ID", data: new Int32Array([0, 1, 2]), shape: [3] });

    const modelElements = stage.get("MODEL/ELEMENTS") as h5wasm.Group;
    const fbcModel = modelElements.create_group("20-ForceBeamColumn2d[0]");
    const quad = fbcModel.create_group("QUADRATURE");
    quad.create_dataset({ name: "GP_PARAM", data: new Float64Array([-0.7746, 0.0, 0.7746]), shape: [3, 1] });
  });

  withFile(p2d, (r) => {
    // 2D file exposes N, Vy, Mz (V_1/V_2 -> Vy, M_1/M_2 -> Mz; no Vz, My, T)
    assert.deepEqual(r.availableLineStations("stage_0"), ["N", "Vy", "Mz"]);
    const nSlab = r.readLineStations("stage_0", "N");
    // Element 1 comes from section.force (Rank 0, 3 stations); Element 2 comes from localForce (Rank 1, 2 stations)
    assert.deepEqual(Array.from(nSlab.elementIndex), [1, 1, 1, 2, 2]);
    assert.deepEqual(
      Array.from(nSlab.stationNaturalCoord).map((v: number) => +v.toFixed(4)),
      [-0.7746, 0, 0.7746, -1, 1],
    );
    const step0N = r.readLineStationStep("stage_0", "N", 0);
    // Element 1 stations: 111, 112, 113; Element 2 stations: end_1 (-1 * -200 = +200), end_2 (+1 * 200 = +200)
    assert.deepEqual(Array.from(step0N.values), [111, 112, 113, 200, 200]);

    const mzSlab = r.readLineStations("stage_0", "Mz");
    const step0Mz = r.readLineStationStep("stage_0", "Mz", 0);
    // Element 1 from section.force: [11, 12, 13]; Element 2 from localForce: end_1 (-1 * -60 = +60), end_2 (+1 * 90 = +90)
    assert.deepEqual(Array.from(mzSlab.elementIndex), [1, 1, 1, 2, 2]);
    assert.deepEqual(Array.from(step0Mz.values), [11, 12, 13, 60, 90]);

    // Vy is only in localForce, so both Element 1 and Element 2 fall back to Rank 1 localForce!
    const step0Vy = r.readLineStationStep("stage_0", "Vy", 0);
    // end_1: +1 * V_1, end_2: -1 * V_2 => elem 1: [20, 20], elem 2: [30, 30]
    assert.deepEqual(Array.from(step0Vy.elementIndex), [1, 1, 2, 2]);
    assert.deepEqual(Array.from(step0Vy.values), [20, 20, 30, 30]);
  });
});

test("readLocalAxes prioritises per-step RESULTS/ON_ELEMENTS/LOCAL_AXES over reference MODEL/LOCAL_AXES", () => {
  // 90 deg rotation around X axis: q = [w=cos(45)=sqrt(1/2), x=sin(45)=sqrt(1/2), y=0, z=0]
  // Row 2 of R(q) (LocalAxes.matrices in _slabs.py): [2(xz - yw), 2(yz + xw), 1 - 2(x^2 + y^2)] = [0, +1, 0]
  const s = Math.SQRT1_2;
  const p = edited(TRUSS, (f) => {
    const model = f.get("MODEL_STAGE[1]/MODEL") as h5wasm.Group;
    const la = model.create_group("LOCAL_AXES");
    const beam = la.create_group("ElasticBeam3d");
    beam.create_dataset({ name: "ID", data: new Int32Array([1, 2]), shape: [2] });
    // elem 1: identity q=[1,0,0,0] -> z_rec=[0,0,1]; elem 2: 90 deg about X -> z_rec=[0,+1,0]
    beam.create_dataset({
      name: "FRAME",
      data: new Float64Array([1, 0, 0, 0, s, s, 0, 0]),
      shape: [2, 4],
    });
  });

  withFile(p, (r) => {
    const axes = r.readLocalAxes("stage_0", { step: 0 });
    assert.deepEqual(Array.from(axes.elementIds), [1, 2]);
    assert.deepEqual(Array.from(axes.hasRecorded), [1, 1]);
    const expected = [0, 0, 1, 0, 1, 0];
    for (let i = 0; i < 6; i++) assert.ok(close(axes.zRec[i]!, expected[i]!, 1e-12));
  });

  // Add per-step RESULTS/ON_ELEMENTS/LOCAL_AXES and verify it takes priority
  const pStep = edited(p, (f) => {
    const elems = f.get("MODEL_STAGE[1]/RESULTS/ON_ELEMENTS") as h5wasm.Group;
    const la = elems.create_group("LOCAL_AXES");
    const block = la.create_group("CorotBeam3d[0]");
    block.create_dataset({ name: "ID", data: new Int32Array([1]), shape: [1] });
    // 4 steps x 1 elem x 3 components (z_x, z_y, z_z)
    const data = new Float64Array([
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
      -1, 0, 0,
    ]);
    block.create_dataset({ name: "DATA", data, shape: [4, 1, 3] });
  });

  withFile(pStep, (r) => {
    // F1a: calling readLocalAxes without options.step returns reference MODEL/LOCAL_AXES, not step 0 of RESULTS/ON_ELEMENTS/LOCAL_AXES!
    const axesRef = r.readLocalAxes("stage_0", { elementIds: [1] });
    assert.deepEqual(Array.from(axesRef.elementIds), [1]);
    assert.deepEqual(Array.from(axesRef.zRec), [0, 0, 1]);
    assert.deepEqual(axesRef.sources, ["model_local_axes"]);

    // All-step readLineStations (no step) also attaches reference MODEL/LOCAL_AXES
    const allStepSlab = r.readLineStations("stage_0", "N", { elementIds: [1] });
    assert.deepEqual(Array.from(allStepSlab.localAxesZRec!), [0, 0, 1]);
    assert.deepEqual(allStepSlab.localAxesSources, ["model_local_axes"]);

    // Passing step: 0 or step: 1 reads RESULTS/ON_ELEMENTS/LOCAL_AXES and marks provenance "step_local_axes"
    const axes0 = r.readLocalAxes("stage_0", { elementIds: [1], step: 0 });
    assert.deepEqual(Array.from(axes0.zRec), [1, 0, 0]);
    assert.deepEqual(axes0.sources, ["step_local_axes"]);

    const axes1 = r.readLocalAxes("stage_0", { elementIds: [1], step: 1 });
    assert.deepEqual(Array.from(axes1.elementIds), [1]);
    assert.deepEqual(Array.from(axes1.zRec), [0, 1, 0]);
    assert.deepEqual(axes1.sources, ["step_local_axes"]);

    const step1 = r.readLineStationStep("stage_0", "N", 1, { elementIds: [1] });
    assert.deepEqual(Array.from(step1.localAxesZRec!), [0, 1, 0]);
    assert.deepEqual(step1.localAxesSources, ["step_local_axes"]);
  });

  // Planar 3D model without INFO@SPATIAL_DIM (COORDINATES shape (3, 3) with z = 0)
  // must infer ndm = 3 from column width (matching Python _file_spatial_dim) and
  // refuse 2D localForce (V_1/M_1).
  const pPlanar3d = edited(TRUSS, (f) => {
    const info = f.get("INFO") as h5wasm.Group;
    setAttr(info, "SPATIAL_DIM", null);
    const nodesGrp = f.get("MODEL_STAGE[1]/MODEL/NODES") as h5wasm.Group & { delete(name: string): void };
    if (typeof nodesGrp.delete === "function") {
      nodesGrp.delete("COORDINATES");
      nodesGrp.create_dataset({
        name: "COORDINATES",
        data: new Float64Array([0, 0, 0, 1, 0, 0, 2, 0, 0]),
        shape: [3, 3],
      });
    } else {
      setAttr(info, "SPATIAL_DIM", 3);
    }
    const elems = f.get("MODEL_STAGE[1]/RESULTS/ON_ELEMENTS") as h5wasm.Group;
    const lf = elems.create_group("localForce");
    const lfBlock = lf.create_group("10-ElasticBeam2d[0]");
    lfBlock.create_dataset({ name: "ID", data: new Int32Array([1]), shape: [1] });
    lfBlock.create_dataset({ name: "TIME", data: new Float64Array([0.25, 0.5, 0.75, 1.0]) });
    lfBlock.create_dataset({ name: "STEP", data: new Int32Array([0, 1, 2, 3]) });
    lfBlock.create_dataset({ name: "DATA", data: new Float64Array(4 * 1 * 6), shape: [4, 1, 6] });
    const lfMap = lfBlock.create_group("COLUMN_MAP");
    lfMap.create_attribute("COMP_NAMES", "N_1,V_1,M_1,N_2,V_2,M_2");
    lfMap.create_dataset({ name: "LEVELS", data: new Int32Array([0]), shape: [1] });
    lfMap.create_dataset({ name: "GAUSS_ID", data: new Int32Array([-1]), shape: [1] });
  });
  withFile(pPlanar3d, (r) => {
    assert.equal(r.inferSpatialDim(), 3);
    const warnings: string[] = [];
    const avail = r.availableLineStations("stage_0", warnings);
    // V_1/M_1 are refused on a 3D file; only N (from basicForce/localForce) is available.
    assert.deepEqual(avail, ["N"]);
    assert.ok(warnings.some((w) => /2-D names/.test(w)));
  });
});

test("F2/F3/F4/F6: sectionServed scoping, whole-element GAUSS_ID validation (1-based, out-of-bounds, negative), 2D section alias gating on 3D, and LOCAL_AXES shape guard", () => {
  const timeArr = new Float64Array([0.25, 0.5, 0.75, 1.0]);
  const stepArr = new Int32Array([0, 1, 2, 3]);

  // 1. F2 + F4: 3-D file with `section.deformation` (LEVELS=2, eps/kappaZ) + 2-D `localForce` (V_1/M_1)
  //    + `section.force` with 2-D shorthand `M,V,P` on a 3-D model:
  //    - 2-D shorthand `M`, `V` in `section.force` must NOT alias to `Mz`, `Vy` on `ndm === 3` (F4)
  //    - `section.deformation` (or `section.force` serving `P -> N`) must NOT mark `Mz` as sectionServed,
  //      so reading `Mz` must still trigger the 2-D `localForce` layout error gate (F2)!
  const pF2F4 = edited(TRUSS, (f) => {
    const info = f.get("INFO") as h5wasm.Group;
    setAttr(info, "SPATIAL_DIM", 3);
    const elems = f.get("MODEL_STAGE[1]/RESULTS/ON_ELEMENTS") as h5wasm.Group;

    // 2D localForce on a 3D model (invalid for element 1)
    const lf = elems.create_group("localForce");
    const lfBlock = lf.create_group("10-ElasticBeam2d[0]");
    lfBlock.create_dataset({ name: "ID", data: new Int32Array([1]), shape: [1] });
    lfBlock.create_dataset({ name: "TIME", data: timeArr });
    lfBlock.create_dataset({ name: "STEP", data: stepArr });
    lfBlock.create_dataset({ name: "DATA", data: new Float64Array(4 * 1 * 6), shape: [4, 1, 6] });
    const lfMap = lfBlock.create_group("COLUMN_MAP");
    lfMap.create_attribute("COMP_NAMES", "N_1,V_1,M_1,N_2,V_2,M_2");
    lfMap.create_dataset({ name: "LEVELS", data: new Int32Array([0]), shape: [1] });
    lfMap.create_dataset({ name: "GAUSS_ID", data: new Int32Array([-1]), shape: [1] });

    // section.deformation with 3D-valid eps,kappaZ and 2D-only gamma,alpha,kappa
    const sd = elems.create_group("section.deformation");
    const sdBlock = sd.create_group("20-ForceBeamColumn3d[0]");
    sdBlock.create_dataset({ name: "ID", data: new Int32Array([1]), shape: [1] });
    sdBlock.create_dataset({ name: "TIME", data: timeArr });
    sdBlock.create_dataset({ name: "STEP", data: stepArr });
    sdBlock.create_dataset({ name: "DATA", data: new Float64Array(4 * 1 * 4), shape: [4, 1, 4] });
    const sdMap = sdBlock.create_group("COLUMN_MAP");
    sdMap.create_attribute("COMP_NAMES", "eps,gamma,alpha,kappa");
    sdMap.create_dataset({ name: "LEVELS", data: new Int32Array([2]), shape: [1] });
    sdMap.create_dataset({ name: "GAUSS_ID", data: new Int32Array([0]), shape: [1] });

    // section.force with P and 2D-only V, M on a 3D model
    const sf = elems.create_group("section.force");
    const sfBlock = sf.create_group("20-ForceBeamColumn3d[0]");
    sfBlock.create_dataset({ name: "ID", data: new Int32Array([1]), shape: [1] });
    sfBlock.create_dataset({ name: "TIME", data: timeArr });
    sfBlock.create_dataset({ name: "STEP", data: stepArr });
    sfBlock.create_dataset({ name: "DATA", data: new Float64Array(4 * 1 * 3).fill(42), shape: [4, 1, 3] });
    const sfMap = sfBlock.create_group("COLUMN_MAP");
    sfMap.create_attribute("COMP_NAMES", "P,V,M");
    sfMap.create_dataset({ name: "LEVELS", data: new Int32Array([2]), shape: [1] });
    sfMap.create_dataset({ name: "GAUSS_ID", data: new Int32Array([0]), shape: [1] });
  });

  withFile(pF2F4, (r) => {
    // F4: On a 3-D model, 2-D shorthand aliases V, M, gamma, alpha, kappa are NOT exposed; only N (from P) and eps are available.
    const warnings: string[] = [];
    assert.deepEqual(r.availableLineStations("stage_0", warnings), ["N", "eps"]);
    // N is served by section.force (P -> N) for element 1 (42) and basicForce for element 2 (2.5).
    const nStep = r.readLineStationStep("stage_0", "N", 0);
    assert.deepEqual(Array.from(nStep.values), [42, 2.5]);
    // F2: Mz is NOT served by section.force (since M is refused on 3-D) nor by section.deformation,
    // so reading Mz must throw the 2-D localForce layout error instead of being suppressed by sectionServed!
    assert.throws(
      () => r.readLineStations("stage_0", "Mz"),
      (e: unknown) => e instanceof LadrunoError && e.kind === "layout" && /2-D names/.test(e.message),
    );
  });

  // 2. F3 + F6: 1-based GAUSS_ID [1, 2, 3], partially out-of-bounds GAUSS_ID [0, 1, 99],
  //    multiple negative GAUSS_ID [-1, -1, -1], and malformed LOCAL_AXES shape
  const pF3F6 = edited(TRUSS, (f) => {
    const stage = f.get("MODEL_STAGE[1]") as h5wasm.Group;
    const elems = stage.get("RESULTS/ON_ELEMENTS") as h5wasm.Group;
    const modelElements = stage.get("MODEL/ELEMENTS") as h5wasm.Group;

    // Element 1: 1-based GAUSS_ID = [1, 2, 3] with 3-row GP_PARAM = [-0.7746, 0, 0.7746]
    const sf = elems.create_group("section.force");
    const b1 = sf.create_group("20-ForceBeamColumn2d[0]");
    b1.create_dataset({ name: "ID", data: new Int32Array([1]), shape: [1] });
    b1.create_dataset({ name: "TIME", data: timeArr });
    b1.create_dataset({ name: "STEP", data: stepArr });
    b1.create_dataset({ name: "DATA", data: new Float64Array(4 * 1 * 3).fill(10), shape: [4, 1, 3] });
    const m1 = b1.create_group("COLUMN_MAP");
    m1.create_attribute("COMP_NAMES", "V\nV\nV");
    m1.create_dataset({ name: "LEVELS", data: new Int32Array([2, 2, 2]), shape: [3] });
    m1.create_dataset({ name: "GAUSS_ID", data: new Int32Array([1, 2, 3]), shape: [3] });
    const fbcModel1 = modelElements.create_group("20-ForceBeamColumn2d[0]");
    fbcModel1.create_group("QUADRATURE").create_dataset({
      name: "GP_PARAM",
      data: new Float64Array([-0.7746, 0.0, 0.7746]),
      shape: [3, 1],
    });

    // Element 2: out-of-bounds GAUSS_ID = [0, 1, 99] with 3-row GP_PARAM = [-0.7746, 0, 0.7746] -> whole-element fallback [-1, 0, 1]
    const b2 = sf.create_group("21-DispBeamColumn2d[0]");
    b2.create_dataset({ name: "ID", data: new Int32Array([2]), shape: [1] });
    b2.create_dataset({ name: "TIME", data: timeArr });
    b2.create_dataset({ name: "STEP", data: stepArr });
    b2.create_dataset({ name: "DATA", data: new Float64Array(4 * 1 * 3).fill(20), shape: [4, 1, 3] });
    const m2 = b2.create_group("COLUMN_MAP");
    m2.create_attribute("COMP_NAMES", "M\nM\nM");
    m2.create_dataset({ name: "LEVELS", data: new Int32Array([2, 2, 2]), shape: [3] });
    m2.create_dataset({ name: "GAUSS_ID", data: new Int32Array([0, 1, 99]), shape: [3] });
    const fbcModel2 = modelElements.create_group("21-DispBeamColumn2d[0]");
    fbcModel2.create_group("QUADRATURE").create_dataset({
      name: "GP_PARAM",
      data: new Float64Array([-0.7746, 0.0, 0.7746]),
      shape: [3, 1],
    });

    // Element 3: multiple single-GP blocks with GAUSS_ID = [-1, -1, -1] AND a 5-point QUADRATURE/GP_PARAM
    // (must not collide on key -1, and must NOT falsely treat [-1, -1, -1] as 0-based [0, 1, 2] into GP_PARAM!)
    const sd = elems.create_group("section.deformation");
    const b3 = sd.create_group("22-ElasticBeam2d[0]");
    b3.create_dataset({ name: "ID", data: new Int32Array([3]), shape: [1] });
    b3.create_dataset({ name: "TIME", data: timeArr });
    b3.create_dataset({ name: "STEP", data: stepArr });
    const d3 = new Float64Array(4 * 1 * 3);
    for (let k = 0; k < 4; k++) d3.set([1, 2, 3], k * 3);
    b3.create_dataset({ name: "DATA", data: d3, shape: [4, 1, 3] });
    const m3 = b3.create_group("COLUMN_MAP");
    m3.create_attribute("COMP_NAMES", "kappa\nkappa\nkappa");
    m3.create_dataset({ name: "LEVELS", data: new Int32Array([2, 2, 2]), shape: [3] });
    m3.create_dataset({ name: "GAUSS_ID", data: new Int32Array([-1, -1, -1]), shape: [3] });
    const fbcModel3 = modelElements.create_group("22-ElasticBeam2d[0]");
    fbcModel3.create_group("QUADRATURE").create_dataset({
      name: "GP_PARAM",
      data: new Float64Array([-0.90618, -0.538469, 0.0, 0.538469, 0.90618]),
      shape: [5, 1],
    });

    // F6: Malformed MODEL/LOCAL_AXES with ID length 2 and FRAME length 5 (not divisible by 2)
    const la = (stage.get("MODEL") as h5wasm.Group).create_group("LOCAL_AXES");
    const badSub = la.create_group("BadBeam");
    badSub.create_dataset({ name: "ID", data: new Int32Array([1, 2]), shape: [2] });
    badSub.create_dataset({ name: "FRAME", data: new Float64Array([1, 0, 0, 0, 1]), shape: [5] });
  });

  withFile(pF3F6, (r) => {
    // 1-based GAUSS_ID [1, 2, 3] maps to GP_PARAM [-0.7746, 0, 0.7746]
    const vy = r.readLineStationStep("stage_0", "Vy", 0);
    assert.deepEqual(
      Array.from(vy.stationNaturalCoord).map((v) => +v.toFixed(4)),
      [-0.7746, 0, 0.7746],
    );
    // Out-of-bounds GAUSS_ID [0, 1, 99] falls back to [-1, 0, 1] for the ENTIRE element (never mixing [-0.7746, 0, 1])
    const mz = r.readLineStationStep("stage_0", "Mz", 0);
    assert.deepEqual(Array.from(mz.stationNaturalCoord), [-1, 0, 1]);
    // GAUSS_ID [-1, -1, -1] keeps all 3 stations in block order with values [1, 2, 3] at [-1, 0, 1], ignoring the 5-row GP_PARAM
    const kz = r.readLineStationStep("stage_0", "kappaZ", 0);
    assert.equal(kz.values.length, 3);
    assert.deepEqual(Array.from(kz.stationNaturalCoord), [-1, 0, 1]);
    assert.deepEqual(Array.from(kz.values), [1, 2, 3]);
    // Malformed MODEL/LOCAL_AXES is safely skipped
    const axes = r.readLocalAxes("stage_0");
    assert.equal(axes.elementIds.length, 0);
  });
});

test("#2187: COMP_NAMES has one rule for line stations and Gauss points: an empty label refuses the bucket, never dropped", () => {
  const withNames = (names: string) =>
    edited(TRUSS, (f) => {
      const lf = (f.get("MODEL_STAGE[1]/RESULTS/ON_ELEMENTS") as h5wasm.Group).create_group("localForce");
      const b = lf.create_group("10-ElasticBeam2d[0]");
      b.create_dataset({ name: "ID", data: new Int32Array([1]), shape: [1] });
      b.create_dataset({ name: "TIME", data: new Float64Array([0.25, 0.5, 0.75, 1.0]) });
      b.create_dataset({ name: "STEP", data: new Int32Array([0, 1, 2, 3]) });
      b.create_dataset({ name: "DATA", data: new Float64Array(4 * 6), shape: [4, 1, 6] });
      const cm = b.create_group("COLUMN_MAP");
      cm.create_attribute("COMP_NAMES", names);
      cm.create_dataset({ name: "LEVELS", data: new Int32Array([0]), shape: [1] });
      cm.create_dataset({ name: "GAUSS_ID", data: new Int32Array([-1]), shape: [1] });
    });
  withFile(withNames("N_1,V_1,M_1,N_2,V_2,M_2"), (r) => {
    const warnings: string[] = [];
    r.availableLineStations("stage_0", warnings);
    assert.deepEqual(warnings, []);
  });
  // Before #2187 the line-station parser dropped the trailing empty label and read the bucket.
  withFile(withNames("N_1,V_1,M_1,N_2,V_2,M_2,"), (r) => {
    const warnings: string[] = [];
    r.availableLineStations("stage_0", warnings);
    assert.match(warnings.join("\n"), /localForce bucket "10-ElasticBeam2d\[0\]" .* is unreadable: .*COMP_NAMES line "N_1,V_1,M_1,N_2,V_2,M_2," has an empty label/);
  });
});
