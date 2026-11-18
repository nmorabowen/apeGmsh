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
import { copyFileSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
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
import { dimensionExponents, particleNotices, particleUnit } from "../../src/state/results.ts";
import { mountParticleScale } from "../../src/panels/particleScale.ts";
import { NO_DATA_RGB, rampCss } from "../../src/state/contour.ts";
import { failureOf, noticesOf, particleScaleOf } from "../../src/state/selectors.ts";
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

// -- #2992 review: the header's Units choice ------------------------------------------

test("dimensionExponents reads the fork's DIMENSION products; anything else is null, never guessed", () => {
  assert.deepEqual(dimensionExponents(""), { f: 0, l: 0, t: 0 });
  assert.deepEqual(dimensionExponents("L"), { f: 0, l: 1, t: 0 });
  assert.deepEqual(dimensionExponents("F/L^2"), { f: 1, l: -2, t: 0 });
  assert.deepEqual(dimensionExponents("L/t"), { f: 0, l: 1, t: -1 });
  assert.deepEqual(dimensionExponents("F*t^2/L^4"), { f: 1, l: -4, t: 2 });
  assert.equal(dimensionExponents("kPa"), null);
  assert.equal(dimensionExponents("F/L/t"), null);
});

test("the stage's UNIT_SYSTEM is the header's Native units with no model; a model's own stay first; a model close keeps the particles'", () => {
  const s = loaded([stage()]);
  assert.equal(s.contour.declaredUnits, "kN_m_s", "the fixtures' kN_m_s shows as Native, not unknown");
  assert.equal(s.contour.targetUnits, "kN_m_s");
  const none = loaded([stage({ units: { length: "m", force: "kN", time: "s", system: "" } })]);
  assert.equal(none.contour.declaredUnits, null, "a stage naming no known system declares none");
  const modelFirst = { ...loaded([stage()]), units: "N_mm_s" as const };
  const again = reduce(reduce(modelFirst, { type: "particlesOpened", source: "results", path: "x.ladruno" }), { type: "particlesLoaded", source: "results", path: "x.ladruno", stages: [stage()], unread: [] });
  assert.equal(again.contour.declaredUnits, "N_mm_s", "the model's declared units come first");
  const closed = reduce(reduce(s, { type: "fileOpened", artifact: "model", path: "m.h5" }), { type: "fileClosed", artifact: "model" });
  assert.equal(closed.contour.declaredUnits, "kN_m_s");
});

test("the particle lines follow the header's Units choice: labels and values converted, the colours unchanged; as stored shows the stored values", () => {
  let s = loaded([stage()]);
  s = reduce(s, { type: "particleFrame", frame: { stage: 0, output: 2, status: "ready", error: null, count: 11, noValue: 0, range: { min: 0.625, max: 1.375 } } });
  assert.ok(particleNotices(s).includes("Particles: output 3 of 3 (step 20, t = 0.002 s), 11 drawn, coloured by POSITION/y (m): 0.625 to 1.375."));
  s = reduce(s, { type: "setContourUnits", target: "N_mm_s" });
  assert.ok(particleNotices(s).includes("Particles: output 3 of 3 (step 20, t = 0.002 s), 11 drawn, coloured by POSITION/y (mm): 625 to 1375."), particleNotices(s).join("\n"));
  s = reduce(s, { type: "setParticleColour", colour: { field: "STRESS", component: 1 } });
  s = reduce(s, { type: "particleFrame", frame: { stage: 0, output: 2, status: "ready", error: null, count: 11, noValue: 0, range: { min: -7.3575, max: 0 } } });
  // kN/m^2 to N/mm^2: 1e3 / 1e6 = 1e-3.
  assert.ok(particleNotices(s).some((l) => /coloured by STRESS\/sigma22 \(N\/mm\^2\): -0\.007358 to 0\.$/.test(l)), particleNotices(s).join("\n"));
  s = reduce(s, { type: "setContourUnits", target: "as_stored" });
  assert.ok(particleNotices(s).some((l) => /coloured by STRESS\/sigma22 \(kN\/m\^2\): -7\.357 to 0\.$/.test(l)), particleNotices(s).join("\n"));
});

// -- the effects, through the real main-side host ---------------------------------------

/** A bridge whose particle doors call main/results.ts directly, as main's handlers do; a frame target that records. */
/** `delayMs(output)` holds main's answer for that output (a slow read), so a later request can overtake it. */
function rig(delayMs: (output: number) => number = () => 0): { store: Store; effects: Effects; frames: (ParticleFrame | null)[]; host: ResultsHost } {
  const store = new Store();
  const host = new ResultsHost(H5, h5wasm.ready, { idleMs: 5 });
  const bridge = {
    openModel: async () => ({ ok: false, error: "no model in this test" }),
    openResults: (path: string) => host.open(path),
    particleIndex: async (path: string) => host.particleIndex(path),
    readParticleOutput: async (req: Parameters<NonNullable<Bridge["readParticleOutput"]>>[0]) => {
      const answer = host.readParticleOutput(req);
      const ms = delayMs(req.output);
      if (ms > 0) await new Promise((r) => setTimeout(r, ms));
      return answer;
    },
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

test("#2992 review: the latest output wins: a slow answer for an output the user has left never paints over the newer one", async () => {
  // Output 0's read is slow; the user scrubs on to output 1 before it lands. The soil's STATE_eqPlasticStrain
  // is 0 at output 0 and 1e-4 at output 1 (the fixture), so the frame on screen says which answer painted last.
  const { store, effects, frames, host } = rig((output) => (output === 0 ? 120 : 0));
  try {
    effects.openSet({ model: null, geometry: null, results: ALL, particles: null });
    await settled(store, frameReady, "the first frame");
    store.dispatch({ type: "setParticleColour", colour: { field: "STATE_eqPlasticStrain", component: 0 } });
    await settled(store, (x) => frameReady(x) && frames.at(-1)?.scalar?.[0] === 2e-4, "the eqPlasticStrain frame at output 2");
    store.dispatch({ type: "setParticleOutput", output: 0 });
    await new Promise((r) => setTimeout(r, 10));
    store.dispatch({ type: "setParticleOutput", output: 1 });
    await settled(store, (x) => frameReady(x) && x.particles!.output === 1, "output 1");
    // Wait past output 0's late answer.
    await new Promise((r) => setTimeout(r, 250));
    const last = frames.at(-1)!;
    assert.equal(last.scalar![0], 1e-4, "the frame on screen is output 1's, not output 0's late answer");
    assert.ok(!frames.some((f) => f?.scalar?.[0] === 0), "output 0's late answer was never pushed");
    const s = store.get();
    assert.equal(s.particles!.output, 1);
    assert.equal(s.particles!.frame!.output, 1);
    assert.equal(s.particles!.frame!.status, "ready");
  } finally {
    effects.dispose();
    host.close();
  }
});

// -- owner D7 on #2992: the particles' colour scale ----------------------------------------
//
// The scale is the contour's own (ui/scaleChrome.ts `fillScale`), filled by `particleScaleOf`.
// Oracles: the ticks rule of the contour scale (panels/legend.ts `contourLines`: five values,
// or the one value and "(constant)", or "no finite value"), the banner's unit conversion
// (`particleDisplay`, pinned above), the colormap the effects hand the layer (`frame.cmap`),
// and the fixture's values as h5py reads them (POSITION/y 0.625 .. 1.375 m; eqPlasticStrain
// 2e-4 on the soil and NaN on the block's 3 particles at the last output).

const EPS = { name: "STATE_eqPlasticStrain", displayName: "Equivalent plastic strain", components: ["epsPeq"], dimension: "" };
const withFrame = (s: State, frame: Partial<NonNullable<NonNullable<State["particles"]>["frame"]>>): State =>
  reduce(s, { type: "particleFrame", frame: { stage: 0, output: 2, status: "ready", error: null, count: 11, noValue: 0, range: null, ...frame } });

test("the scale names the field, component and unit, ticks the range as the contour scale does, and follows the header's Units", () => {
  let s = loaded([stage()]);
  // Before the first frame lands the scale is up, saying the output is being read (no flicker during a scrub).
  assert.deepEqual(particleScaleOf(s), { title: "POSITION · y [m]", tooltip: "as the stage stores it, in m (kN_m_s)", cmap: "jet", ticks: ["reading the output"], range: "range of output 3 of 3", noValue: null });
  s = withFrame(s, { range: { min: 0.625, max: 1.375 } });
  assert.deepEqual(particleScaleOf(s)!.ticks, ["0.625", "0.8125", "1", "1.188", "1.375"]);
  s = reduce(s, { type: "setContourUnits", target: "N_mm_s" });
  const mm = particleScaleOf(s)!;
  assert.equal(mm.title, "POSITION · y [mm]");
  assert.deepEqual(mm.ticks, ["625", "812.5", "1000", "1188", "1375"]);
  assert.match(mm.tooltip, /^stored in m \(kN_m_s\); shown in the header's Units$/);
  // The colormap is the one the points are drawn with: effects.ts hands `contour.cmap` to the layer.
  s = reduce(s, { type: "setContourCmap", cmap: "viridis" });
  assert.equal(particleScaleOf(s)!.cmap, "viridis");
  // The seeded set says so; another output says which.
  assert.equal(particleScaleOf(reduce(withFrame(s, {}), { type: "setParticleOutput", output: 0 }))!.range, "range of output 1 of 3");
  const set = loaded([stage({ id: "set", nOutputs: 1, steps: [0], time: [0], counts: [12] })], "set", "c.particles.h5");
  assert.equal(particleScaleOf(set)!.range, "range of the seeded set");
});

test("the scale with no value (NaN): a constant field keeps its one value; the particles with none are counted beside the no-value chip", () => {
  let s = reduce(loaded([stage({ fields: [...stage().fields, EPS] })]), { type: "setParticleColour", colour: { field: "STATE_eqPlasticStrain", component: 0 } });
  s = withFrame(s, { noValue: 3, range: { min: 2e-4, max: 2e-4 } });
  const v = particleScaleOf(s)!;
  assert.equal(v.title, "STATE_eqPlasticStrain · epsPeq", "dimensionless: no unit");
  assert.deepEqual(v.ticks, ["2.000e-4", "(constant)"]);
  assert.equal(v.noValue, "no value (NaN): 3 of 11 particles");
  const all = particleScaleOf(withFrame(s, { noValue: 11, range: null }))!;
  assert.deepEqual([all.ticks, all.noValue], [["no finite value"], "no value (NaN): all 11 particles"]);
});

test("no scale when the particles are not coloured by a field, not drawn, or their output failed to read", () => {
  const s = withFrame(loaded([stage()]), { range: { min: 0.625, max: 1.375 } });
  assert.notEqual(particleScaleOf(s), null);
  assert.equal(particleScaleOf(reduce(s, { type: "setParticleColour", colour: null })), null, "one plain colour");
  assert.equal(particleScaleOf(reduce(s, { type: "setParticlesOn", on: false })), null, "particles off");
  assert.equal(particleScaleOf(withFrame(s, { status: "failed", error: "read failed" })), null, "the banner says why");
  assert.equal(particleScaleOf(loaded([stage({ nOutputs: 0, steps: [], time: [], counts: [] })])), null, "no output");
  assert.equal(particleScaleOf(initialState), null);
  // A colour the stage does not offer is a broken state: refused by name, never a scale skipped.
  const broken = { ...s, particles: { ...s.particles!, colour: { field: "PRESSURE", component: 0 } } };
  assert.throws(() => particleScaleOf(broken), /\{"field":"PRESSURE","component":0\} is not a field component of \/MODEL_STAGE\[1\]/);
});

/** Enough of an element for the scale panel (ui/dom.ts `el`, ui/scaleChrome.ts `fillScale`). */
class FakeEl {
  tagName: string;
  className = "";
  title = "";
  type = "";
  hidden = false;
  dataset: Record<string, string | undefined> = {};
  style: Record<string, string> = {};
  children: (FakeEl | string)[] = [];
  listeners = new Map<string, (() => void)[]>();
  constructor(tag: string) {
    this.tagName = tag.toUpperCase();
  }
  append(...xs: (FakeEl | string)[]) {
    this.children.push(...xs);
  }
  replaceChildren(...xs: (FakeEl | string)[]) {
    this.children = [...xs];
  }
  addEventListener(type: string, fn: () => void) {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), fn]);
  }
  removeEventListener(type: string, fn: () => void) {
    this.listeners.set(type, (this.listeners.get(type) ?? []).filter((f) => f !== fn));
  }
  get textContent(): string {
    return this.children.map((c) => (typeof c === "string" ? c : c.textContent)).join("");
  }
  set textContent(v: string) {
    this.children = [v];
  }
  all(pred: (e: FakeEl) => boolean): FakeEl[] {
    const out: FakeEl[] = pred(this) ? [this] : [];
    for (const c of this.children) if (typeof c !== "string") out.push(...c.all(pred));
    return out;
  }
}

test("mounted on the fixture: the scale shows the eqPlasticStrain field in the points' colormap, its constant value, and the block's 3 particles with no value", async () => {
  const { store, effects, frames, host } = rig();
  const scale = new FakeEl("div");
  const prevDoc = (globalThis as { document?: unknown }).document;
  (globalThis as { document?: unknown }).document = {
    createElement: (t: string) => new FakeEl(t),
    getElementById: (id: string) => (id === "particle-scale" ? scale : null),
  };
  let unmount: (() => void) | null = null;
  try {
    unmount = mountParticleScale(store);
    assert.equal(scale.hidden, true, "nothing is open: no scale");
    effects.openSet({ model: null, geometry: null, results: ALL, particles: null });
    await settled(store, frameReady, "the first frame");
    const text = (cls: string) => scale.all((e) => e.className === cls).map((e) => e.textContent);
    assert.equal(scale.hidden, false, "coloured by POSITION/y: the scale shows");
    assert.deepEqual(text("vs-title"), ["POSITION · y [m]"]);
    assert.deepEqual(scale.all((e) => e.className === "vs-ticks")[0]!.children.map((c) => (c as FakeEl).textContent), ["0.625", "0.8125", "1", "1.187", "1.375"], "the fixture's own values put the 3/4 tick just under 1.1875 (the exact ends give 1.188, above)");
    store.dispatch({ type: "setParticleColour", colour: { field: "STATE_eqPlasticStrain", component: 0 } });
    await settled(store, (x) => frameReady(x) && frames.at(-1)?.scalar !== null && Number.isNaN(frames.at(-1)!.scalar![8]!), "the eqPlasticStrain frame");
    assert.deepEqual(text("vs-title"), ["STATE_eqPlasticStrain · epsPeq"]);
    assert.deepEqual(scale.all((e) => e.className === "vs-ticks")[0]!.children.map((c) => (c as FakeEl).textContent), ["2.000e-4", "(constant)"]);
    // The bar is the colormap the points were drawn with: the layer's own frame names it.
    assert.equal(scale.all((e) => e.className === "vs-ramp")[0]!.style["backgroundImage"], rampCss(frames.at(-1)!.cmap));
    // The no-value row: the contour's no-value chip (#6B7280, the grey the still's block particles read) and the count.
    const row = scale.all((e) => e.className === "legend-row particle-novalue")[0]!;
    assert.equal(row.all((e) => e.className === "chip")[0]!.style["backgroundColor"], `rgb(${NO_DATA_RGB.map((c) => Math.round(c * 255)).join(",")})`);
    assert.equal(row.all((e) => e.className === "chip")[0]!.style["backgroundColor"], "rgb(107,114,128)");
    assert.deepEqual(text("contour-nodata"), ["no value (NaN): 3 of 11 particles"]);
    // One plain colour: no scale.
    store.dispatch({ type: "setParticleColour", colour: null });
    assert.equal(scale.hidden, true);
  } finally {
    unmount?.();
    (globalThis as { document?: unknown }).document = prevDoc;
    effects.dispose();
    host.close();
  }
});

test("the page carries the particle scale in the scale stack, after the line diagram's, and app.ts mounts it", () => {
  const html = readFileSync(new URL("../../src/renderer/index.html", import.meta.url), "utf8");
  const at = (id: string) => html.indexOf(`id="${id}"`);
  assert.ok(at("scale-stack") > 0 && at("line-diagram-scale") < at("particle-scale"), "inside the stack, after the line diagram's scale");
  assert.match(html, /<div id="particle-scale" class="viewport-scale" hidden data-orient="horizontal"><\/div>\s*<\/div>/, "the stack's last child");
  const app = readFileSync(new URL("../../src/renderer/app.ts", import.meta.url), "utf8");
  assert.match(app, /^\s*mountParticleScale\(store\),$/m);
});
