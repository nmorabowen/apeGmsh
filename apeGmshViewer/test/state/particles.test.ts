// The particle slice (ADR 0134 D5; chain M, M2-S4 #2983): the reducer, the banner's
// particle lines, and the effects that read the index and push each frame.
//
// The oracle is the fork's fixtures as h5py reads them (tests/fixtures/mpm/, the README
// records the fork commit): one transient particle stage, outputs at steps 0, 10 and 20,
// 12 particles in two bodies (soil 0..7, block 8..11), particle 11 removed after output
// 1, and STATE_eqPlasticStrain NaN on the block. The seeded set `particle_set_v1.h5`
// holds the same 12 particles. The effects read them through the real main-side host
// (main/results.ts), with a bridge that calls it directly, as main's doors do.

import assert from "node:assert/strict";
import { copyFileSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { Effects, type Bridge, type FrameTarget } from "../../src/effects.ts";
import { readParticleSet, ResultsHost } from "../../src/main/results.ts";
import type { ResultsH5 } from "../../src/reader/results.ts";
import type { ParticleFrame } from "../../src/renderer/particles.ts";
import { BlobStore } from "../../src/state/blobs.ts";
import { reduce, initialState } from "../../src/state/reduce.ts";
import { particleNotices, particleUnit } from "../../src/state/results.ts";
import { failureOf, noticesOf } from "../../src/state/selectors.ts";
import { Store } from "../../src/state/store.ts";
import type { ParticleStageInfo, State } from "../../src/state/types.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as ResultsH5;
const repo = (rel: string) => fileURLToPath(new URL(`../../../${rel}`, import.meta.url));
const ALL = repo("tests/fixtures/mpm/particles_v1m2.ladruno");
const tmp = mkdtempSync(join(tmpdir(), "particles-slice-"));
after(() => rmSync(tmp, { recursive: true, force: true }));
// The seeded set under its set name (main/pairing.ts pairs `<stem>.particles.h5`).
const SET = join(tmp, "case1.particles.h5");
copyFileSync(repo("tests/fixtures/mpm/particle_set_v1.h5"), SET);

const stage = (over: Partial<ParticleStageInfo> = {}): ParticleStageInfo => ({
  id: "stage_0",
  name: "/MODEL_STAGE[1]",
  ndm: 2,
  units: { length: "m", force: "kN", time: "s", system: "kN_m_s" },
  grid: { origin: [0, 0], cellSize: [0.5, 0.5], numCells: [4, 4], basis: "gimp" },
  bodies: [{ id: 0, name: "soil" }],
  nOutputs: 3,
  steps: [0, 10, 20],
  time: [0, 0.001, 0.002],
  counts: [12, 12, 11],
  fields: [
    { name: "POSITION", displayName: "Position", components: ["x", "y"], dimension: "L" },
    { name: "STRESS", displayName: "Stress", components: ["sigma11", "sigma22", "sigma12"], dimension: "F/L^2" },
  ],
  refusedFields: [],
  torn: null,
  ...over,
});

const loaded = (stages: ParticleStageInfo[], source: "results" | "set" = "results", path = "x.ladruno"): State =>
  reduce(reduce(initialState, { type: "particlesOpened", source, path }), { type: "particlesLoaded", source, path, stages, unread: [] });

// -- the reducer -------------------------------------------------------------------

test("particlesLoaded opens on the stage's last output, coloured by POSITION's vertical component", () => {
  const s = loaded([stage()]);
  assert.equal(s.particles!.status, "ready");
  assert.equal(s.particles!.output, 2);
  assert.deepEqual(s.particles!.colour, { field: "POSITION", component: 1 });
  const s3 = loaded([stage({ ndm: 3, fields: [{ name: "POSITION", displayName: "Position", components: ["x", "y", "z"], dimension: "L" }] })]);
  assert.deepEqual(s3.particles!.colour, { field: "POSITION", component: 2 }, "z in 3-D");
});

test("a request the stage cannot honour throws by name, never clamps", () => {
  const s = loaded([stage()]);
  assert.throws(() => reduce(s, { type: "setParticleOutput", output: 3 }), /output 3 is outside the stage's 3 outputs/);
  assert.throws(() => reduce(s, { type: "setParticleOutput", output: 1.5 }), /outside/);
  assert.throws(() => reduce(s, { type: "setParticleColour", colour: { field: "PRESSURE", component: 0 } }), /not a field component of the stage; it offers POSITION\/x,y; STRESS/);
  assert.throws(() => reduce(s, { type: "setParticleColour", colour: { field: "STRESS", component: 3 } }), /not a field component/);
  assert.throws(() => reduce(s, { type: "setParticleStage", stage: 1 }), /1 is not a particle stage/);
  assert.throws(() => reduce(initialState, { type: "setParticleOutput", output: 0 }), /no particles are read/);
  assert.throws(() => reduce(initialState, { type: "particlesLoaded", source: "set", path: "x.particles.h5", stages: [], unread: [] }), /is not the particles being read/);
  assert.throws(() => reduce(reduce(initialState, { type: "particlesOpened", source: "results", path: "x" }), { type: "particlesLoaded", source: "results", path: "x", stages: [stage({ counts: [12] })], unread: [] }), /says 3 outputs and carries 3 steps, 3 times, 1 counts/);
});

test("a frame for an output no longer shown is dropped; one with impossible counts throws", () => {
  const s = reduce(loaded([stage()]), { type: "setParticleOutput", output: 0 });
  const late = reduce(s, { type: "particleFrame", frame: { stage: 0, output: 2, status: "ready", error: null, count: 11, noValue: 0, range: null } });
  assert.equal(late, s);
  assert.throws(() => reduce(s, { type: "particleFrame", frame: { stage: 0, output: 0, status: "ready", error: null, count: 3, noValue: 4, range: null } }), /4 without a value of 3/);
});

test("a re-read keeps the stage, output and colour while they still hold; the model's close keeps the particles (they are the set's)", () => {
  let s = reduce(loaded([stage()]), { type: "setParticleOutput", output: 1 });
  s = reduce(s, { type: "setParticleColour", colour: { field: "STRESS", component: 2 } });
  s = reduce(reduce(s, { type: "particlesOpened", source: "results", path: "x.ladruno" }), { type: "particlesLoaded", source: "results", path: "x.ladruno", stages: [stage()], unread: [] });
  assert.equal(s.particles!.output, 1);
  assert.deepEqual(s.particles!.colour, { field: "STRESS", component: 2 });
  // The stage shrank to one output, and lost STRESS: the defaults come back.
  s = reduce(reduce(s, { type: "particlesOpened", source: "results", path: "x.ladruno" }), {
    type: "particlesLoaded",
    source: "results",
    path: "x.ladruno",
    stages: [stage({ nOutputs: 1, steps: [0], time: [0], counts: [12], fields: [stage().fields[0]!] })],
    unread: [],
  });
  assert.equal(s.particles!.output, 0);
  assert.deepEqual(s.particles!.colour, { field: "POSITION", component: 1 });
  const closed = reduce(reduce(s, { type: "fileOpened", artifact: "model", path: "m.h5" }), { type: "fileClosed", artifact: "model" });
  assert.equal(closed.particles, s.particles);
  assert.equal(reduce(s, { type: "particlesClosed" }).particles, null);
});

// -- the banner's particle lines ------------------------------------------------------

test("units: a field's DIMENSION in the stage's units; POSITION's is the length unit", () => {
  const u = stage().units;
  assert.equal(particleUnit({ name: "STRESS", displayName: "", components: [], dimension: "F/L^2" }, u), "kN/m^2");
  assert.equal(particleUnit({ name: "VELOCITY", displayName: "", components: [], dimension: "L/t" }, u), "m/s");
  assert.equal(particleUnit({ name: "STATE_x", displayName: "", components: [], dimension: "" }, u), "");
  assert.equal(particleUnit({ name: "POSITION", displayName: "", components: [], dimension: "L" }, u), "m");
});

test("the lines say what is drawn, the particles with no value, an all-NaN output, a stage with no output, and a failed read", () => {
  let s = loaded([stage()]);
  s = reduce(s, { type: "particleFrame", frame: { stage: 0, output: 2, status: "ready", error: null, count: 11, noValue: 3, range: { min: 0.625, max: 1.375 } } });
  const lines = particleNotices(s);
  assert.ok(lines.includes("Particles: output 3 of 3 (step 20, t = 0.002 s), 11 drawn, coloured by POSITION/y (m): 0.625 to 1.375."), lines.join("\n"));
  assert.ok(lines.some((l) => /^Particles: 3 of 11 have no value of POSITION\/y \(m\) \(NaN: their material does not answer it\); they are drawn in the no-value colour\.$/.test(l)), lines.join("\n"));
  s = reduce(s, { type: "particleFrame", frame: { stage: 0, output: 2, status: "ready", error: null, count: 11, noValue: 11, range: null } });
  assert.ok(particleNotices(s).some((l) => /no particle has a value of POSITION\/y \(m\) at this output/.test(l)));
  const none = loaded([stage({ nOutputs: 0, steps: [], time: [], counts: [], torn: "a torn tail (§2.5 crash rule): STEP holds 0" })]);
  assert.deepEqual(particleNotices(none), [
    "Particles of /MODEL_STAGE[1]: 0 outputs read; a torn tail (§2.5 crash rule): STEP holds 0.",
    "Particles of /MODEL_STAGE[1]: no output was committed, so none is drawn.",
  ]);
  const failed = reduce(reduce(initialState, { type: "particlesOpened", source: "set", path: "C:/r/case1.particles.h5" }), { type: "particlesFailed", path: "C:/r/case1.particles.h5", error: "NUM_PARTICLES is 0; expected >= 1 (C1-N)" });
  assert.deepEqual(particleNotices(failed), ["Particles not drawn: case1.particles.h5: NUM_PARTICLES is 0; expected >= 1 (C1-N)."]);
  const unread = reduce(reduce(initialState, { type: "particlesOpened", source: "results", path: "x.ladruno" }), {
    type: "particlesLoaded",
    source: "results",
    path: "x.ladruno",
    stages: [],
    unread: [{ stage: "/MODEL_STAGE[1]", reason: "refused: the registry columns differ in length (C2-REG)" }],
  });
  assert.deepEqual(particleNotices(unread), ["Particles of /MODEL_STAGE[1] not drawn: refused: the registry columns differ in length (C2-REG)."]);
});

// -- the effects, through the real main-side host ---------------------------------------

/** A bridge whose particle doors call main/results.ts directly, as main's handlers do; a frame target that records. */
function rig(): { store: Store; effects: Effects; frames: (ParticleFrame | null)[]; host: ResultsHost } {
  const store = new Store();
  const host = new ResultsHost(H5, h5wasm.ready, { idleMs: 5 });
  const bridge = {
    openModel: async () => ({ ok: false, error: "no model in this test" }),
    openResults: (path: string) => host.open(path),
    particleIndex: async (path: string) => host.particleIndex(path),
    readParticleOutput: async (req: Parameters<NonNullable<Bridge["readParticleOutput"]>>[0]) => host.readParticleOutput(req),
    particleSet: () => readParticleSet(H5, h5wasm.ready, SET),
  } as unknown as Bridge;
  const effects = new Effects(store, new BlobStore(), bridge);
  const frames: (ParticleFrame | null)[] = [];
  const target: FrameTarget = { frameTo: () => {}, showParticles: (f) => frames.push(f) };
  effects.setFrameTarget(target);
  return { store, effects, frames, host };
}

async function settled(store: Store, ok: (s: State) => boolean, what: string): Promise<State> {
  for (let i = 0; i < 400; i++) {
    const s = store.get();
    if (ok(s)) return s;
    await new Promise((r) => setTimeout(r, 5));
  }
  throw new Error(`waited for ${what}; particles: ${JSON.stringify(store.get().particles)}`);
}
const frameReady = (s: State) => s.particles?.frame?.status === "ready" && s.particles.frame.output === s.particles.output;

test("#2985's refusal is gone: a particle set opened alone draws its 12 particles, with its notice and no failure", async () => {
  const { store, effects, frames, host } = rig();
  try {
    effects.openSet({ model: null, geometry: null, results: null, particles: SET });
    const s = await settled(store, frameReady, "the set's frame");
    assert.equal(failureOf(s), null, "no load failure: the window is not refused");
    assert.equal(s.particles!.source, "set");
    assert.equal(s.particles!.frame!.count, 12);
    assert.deepEqual(s.particles!.frame!.range, { min: 0.625, max: 1.375 });
    const notices = noticesOf(s);
    assert.ok(notices.some((n) => /^Particle set open without a model: case1\.particles\.h5 draws its 12 seeded particles alone \(ADR 0134\)/.test(n)), notices.join("\n"));
    const f = frames.at(-1)!;
    assert.equal(f.positions.length, 24);
    assert.deepEqual(Array.from(f.scalar!.subarray(0, 4)), [0.625, 0.625, 0.625, 0.625], "POSITION/y of the soil's bottom row");
    assert.equal(f.fit, true, "a set with no mesh frames the camera on its particles");
    // Another set naming nothing closes the particles and draws none.
    effects.openSet({ model: null, geometry: null, results: null, particles: null });
    await settled(store, (x) => x.particles === null, "the particles closed");
    assert.equal(frames.at(-1), null);
  } finally {
    effects.dispose();
    host.close();
  }
});

test("a .ladruno's particles: the last output by default; STATE_eqPlasticStrain's NaN on the block is counted, not drawn as a value; a scrub reads another output", async () => {
  const { store, effects, frames, host } = rig();
  try {
    // The results win over a seeded set of the same set: the run, not its seed.
    effects.openSet({ model: null, geometry: null, results: ALL, particles: SET });
    let s = await settled(store, frameReady, "the first frame");
    assert.equal(s.particles!.source, "results");
    assert.equal(s.particles!.output, 2);
    assert.equal(s.particles!.frame!.count, 11, "particle 11 left after output 1");
    store.dispatch({ type: "setParticleColour", colour: { field: "STATE_eqPlasticStrain", component: 0 } });
    s = await settled(store, (x) => frameReady(x) && frames.at(-1)?.scalar !== null && Number.isNaN(frames.at(-1)!.scalar![8]!), "the eqPlasticStrain frame");
    assert.equal(s.particles!.frame!.noValue, 3, "the block's 3 particles at output 2");
    assert.deepEqual(s.particles!.frame!.range, { min: 2e-4, max: 2e-4 }, "the soil's value, NaN left out");
    store.dispatch({ type: "setParticleOutput", output: 0 });
    s = await settled(store, (x) => frameReady(x) && x.particles!.frame!.count === 12, "output 0");
    assert.equal(s.particles!.frame!.noValue, 4);
    assert.equal(frames.at(-1)!.positions.length, 24);
    assert.equal(frames.at(-1)!.fit, false, "the camera is framed once per set, not per output");
  } finally {
    effects.dispose();
    host.close();
  }
});
