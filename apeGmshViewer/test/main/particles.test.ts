// main/results.ts's particle doors (ADR 0134 D5; chain M, M2-S4 #2983): the index of the
// indexed `.ladruno`, one output by hyperslab, and the seeded set read whole.
//
// The oracle is the fork's fixtures as h5py reads them (tests/fixtures/mpm/README.md): one
// particle stage, outputs at steps 0, 10, 20 holding 12, 12 and 11 particles, POSITION
// float64, STATE_eqPlasticStrain NaN on the block (particles 8..11); the seeded set
// `particle_set_v1.h5` holds the 12 particles at the run's output-0 positions.

import assert from "node:assert/strict";
import { appendFileSync, copyFileSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { readParticleSet, ResultsHost } from "../../src/main/results.ts";
import type { ResultsH5 } from "../../src/reader/results.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as ResultsH5;
const repo = (rel: string) => fileURLToPath(new URL(`../../../${rel}`, import.meta.url));
const tmp = mkdtempSync(join(tmpdir(), "particle-doors-"));
after(() => rmSync(tmp, { recursive: true, force: true }));
const copy = (rel: string, name: string): string => {
  const p = join(tmp, name);
  copyFileSync(repo(rel), p);
  return p;
};
const ALL = copy("tests/fixtures/mpm/particles_v1m2.ladruno", "run.ladruno");
const MIXED = copy("tests/fixtures/mpm/particles_mixed_v1m2.ladruno", "mixed.ladruno");
const SET = copy("tests/fixtures/mpm/particle_set_v1.h5", "case1.particles.h5");

async function hosted<T>(path: string, body: (h: ResultsHost) => T | Promise<T>): Promise<T> {
  const h = new ResultsHost(H5, h5wasm.ready, { idleMs: 5 });
  try {
    const o = await h.open(path);
    assert.ok(o.ok, o.ok ? "" : o.error);
    return await body(h);
  } finally {
    h.close();
  }
}

test("particleIndex: the stage as the store keeps it (plain arrays), each output's particle count from OFFSETS", async () => {
  await hosted(ALL, (h) => {
    const a = h.particleIndex(ALL);
    assert.ok(a.ok, a.ok ? "" : a.error);
    assert.equal(a.stages.length, 1);
    const s = a.stages[0]!;
    assert.equal(s.id, "stage_0");
    assert.equal(s.ndm, 2);
    assert.deepEqual(s.steps, [0, 10, 20]);
    assert.deepEqual(s.time, [0, 0.001, 0.002]);
    assert.deepEqual(s.counts, [12, 12, 11]);
    assert.deepEqual(s.grid, { origin: [0, 0], cellSize: [0.5, 0.5], numCells: [4, 4], basis: "gimp" });
    assert.deepEqual(s.fields.find((f) => f.name === "STRESS"), { name: "STRESS", displayName: "Stress", components: ["sigma11", "sigma22", "sigma12"], dimension: "F/L^2" });
    assert.deepEqual(a.unread, []);
    assert.ok(Array.isArray(s.steps) && Array.isArray(s.counts), "plain arrays: the store holds no typed array");
  });
});

test("readParticleOutput: POSITION and the colour field of one output, each particle's row; NaN read as stored", async () => {
  await hosted(MIXED, (h) => {
    const a = h.readParticleOutput({ path: MIXED, stage: "stage_1", output: 2, field: "STATE_eqPlasticStrain", seq: 7 });
    assert.ok(a.ok, a.ok ? "" : a.error);
    assert.equal(a.seq, 7);
    assert.equal(a.step, 20);
    assert.deepEqual(Array.from(a.ids), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    assert.equal(a.positions.length, 22);
    assert.deepEqual(Array.from(a.positions.subarray(0, 4)), [0.625, 0.625, 0.875, 0.625]);
    assert.equal(a.nComp, 1);
    assert.deepEqual(Array.from(a.values!.subarray(0, 8)), Array(8).fill(2e-4));
    assert.ok(Array.from(a.values!.subarray(8)).every(Number.isNaN), "the block's STATE is NaN");
    const plain = h.readParticleOutput({ path: MIXED, stage: "stage_1", output: 0, field: null, seq: 8 });
    assert.ok(plain.ok && plain.values === null && plain.positions.length === 24);
    const pos = h.readParticleOutput({ path: MIXED, stage: "stage_1", output: 0, field: "POSITION", seq: 9 });
    assert.ok(pos.ok && pos.nComp === 2 && pos.values === pos.positions);
  });
});

test("the particle doors refuse by name: another path, a stage with no particles, an output outside the stage, a field not recorded, a file that is not a .ladruno", async () => {
  await hosted(MIXED, (h) => {
    const other = h.readParticleOutput({ path: ALL, stage: "stage_0", output: 0, field: null, seq: 1 });
    assert.ok(!other.ok && /is not the open results file/.test(other.error));
    const fe = h.readParticleOutput({ path: MIXED, stage: "stage_0", output: 0, field: null, seq: 2 });
    assert.ok(!fe.ok && /stage stage_0 holds no particles; particle stages: stage_1/.test(fe.error), fe.ok ? "" : fe.error);
    const out = h.readParticleOutput({ path: MIXED, stage: "stage_1", output: 3, field: null, seq: 3 });
    assert.ok(!out.ok && /output 3 is outside stage stage_1 \(3 outputs\)/.test(out.error));
    const field = h.readParticleOutput({ path: MIXED, stage: "stage_1", output: 0, field: "PRESSURE", seq: 4 });
    assert.ok(!field.ok && /records no particle field "PRESSURE"/.test(field.error));
  });
  await hosted(repo("apeGmshViewer/fixtures/contour_staged.results.h5"), (h) => {
    const p = repo("apeGmshViewer/fixtures/contour_staged.results.h5");
    const a = h.particleIndex(p);
    assert.ok(!a.ok && /contour_staged\.results\.h5 holds no particle results this app reads: particles are read from one \.ladruno file/.test(a.error), a.ok ? "" : a.error);
  });
});

test("a .ladruno rewritten since it was indexed answers changed, never the old index's particles", async () => {
  const p = copy("tests/fixtures/mpm/particles_v1m2.ladruno", "changing.ladruno");
  await hosted(p, (h) => {
    assert.ok(h.particleIndex(p).ok);
    appendFileSync(p, Buffer.alloc(16));
    const a = h.particleIndex(p);
    assert.ok(!a.ok && a.changed === true, JSON.stringify(a));
    const r = h.readParticleOutput({ path: p, stage: "stage_0", output: 0, field: null, seq: 5 });
    assert.ok(!r.ok && r.changed === true);
  });
});

test("readParticleSet: the seeded set as one stage of one output, with every per-particle array to colour by", async () => {
  const a = await readParticleSet(H5, h5wasm.ready, SET);
  assert.ok(a.ok, a.ok ? "" : a.error);
  assert.equal(a.stage.id, "set");
  assert.equal(a.stage.name, "case1.particles.h5");
  assert.deepEqual([a.stage.nOutputs, a.stage.counts, a.stage.steps], [1, [12], [0]]);
  assert.deepEqual(a.stage.fields.map((f) => f.name), ["POSITION", "VOLUME", "DENSITY", "GIMP_HALF_LENGTH", "BODY_ID", "MATERIAL_TAG"], "the fixture's MASS_SOURCE is density");
  assert.equal(a.positions.length, 24);
  assert.deepEqual(Array.from(a.values["BODY_ID"]!), [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]);
  assert.equal(a.values["POSITION"], a.positions);
  const bad = join(tmp, "bad.particles.h5");
  writeFileSync(bad, "not hdf5");
  const b = await readParticleSet(H5, h5wasm.ready, bad);
  assert.ok(!b.ok && /\(C1-REQ\)/.test(b.error), b.ok ? "" : b.error);
});
