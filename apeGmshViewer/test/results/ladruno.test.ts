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

test("an inconsistent modal record refuses the file: a mode without LAMBDA, or (R) disagreeing with (U)", () => {
  const MODAL = "/MODEL_STAGE[1]/RESULTS/ON_NODES";
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
    assert.deepEqual(p.attrs.writer, { state: "absent" });
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
