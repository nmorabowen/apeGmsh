// reader/ladrunoParticles.ts (through reader/ladruno.ts) against the fork's MPM fixtures
// (ADR 0134 D3; chain M, M2-S3 #2982).
//
// The oracle is the fixture as h5py reads it (tests/fixtures/mpm/, byte-identical to the
// fork at the commit its README records), and the fork's contract
// `Ladruno_implementation/232a_mpm_contracts.md` §2 and §4: one transient particle stage,
// outputs at steps 0, 10 and 20, 12 particles in two bodies (soil 0..7, block 8..11),
// particle 11 removed after output 1, and `STATE_eqPlasticStrain` NaN on the block.
//
// The torn-tail and chunk cases are files this test writes with h5wasm into a temp
// folder: a fixture copy whose datasets are resized as a killed writer leaves them
// (§2.5 crash rule), and a rewrite of the particle stage with contiguous datasets,
// 5-row chunks and variable-length string attributes (§0 "Storage", "Attribute
// encodings"): the reads must not depend on the chunk shape or the string encoding.

import assert from "node:assert/strict";
import { copyFileSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { LadrunoError, openLadruno, type LadrunoFile } from "../../src/reader/ladruno.ts";
import type { LadrunoParticles, ParticleStage } from "../../src/reader/ladrunoParticles.ts";
import type { ResultsH5 } from "../../src/reader/results.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as ResultsH5;
const repo = (rel: string) => fileURLToPath(new URL(`../../../${rel}`, import.meta.url));
const ALL = repo("tests/fixtures/mpm/particles_v1m2.ladruno");
const MIXED = repo("tests/fixtures/mpm/particles_mixed_v1m2.ladruno");
const tmp = mkdtempSync(join(tmpdir(), "ladruno-particles-"));
after(() => rmSync(tmp, { recursive: true, force: true }));

function withFile<T>(path: string, body: (f: LadrunoFile, p: LadrunoParticles) => T, opts: { part?: boolean } = {}): T {
  const f = openLadruno(H5, path, opts);
  try {
    assert.ok(f.particles, `${path}: particles are read at minor 2`);
    return body(f, f.particles);
  } finally {
    f.close();
  }
}

const IDS_FULL = Array.from({ length: 12 }, (_, i) => i);
const POSITION_2 = [
  [0.625, 0.625], [0.875, 0.625], [1.125, 0.625], [1.375, 0.625],
  [0.625, 0.875], [0.875, 0.875], [1.125, 0.875], [1.375, 0.875],
  [1.125, 1.12498038], [1.375, 1.12498038], [1.125, 1.37498038],
];

/** The fixture's particle stage as h5py reads it, whatever the file it sits in. */
function assertFixtureStage(s: ParticleStage, p: LadrunoParticles): void {
  assert.equal(s.kind, "transient");
  assert.equal(s.mode, "explicit");
  assert.equal(s.epoch, 0);
  assert.equal(s.patternTag, 1);
  assert.equal(s.sourceSetId, "3f6c1d2e-8a4b-4c5d-9e7f-0a1b2c3d4e5f");
  assert.equal(s.ndm, 2);
  assert.equal(s.thickness, 1);
  assert.deepEqual(s.units, { length: "m", force: "kN", time: "s", system: "kN_m_s" });
  assert.deepEqual(s.grid, { origin: [0, 0], cellSize: [0.5, 0.5], numCells: [4, 4], basis: "gimp" });
  assert.deepEqual(s.bodies, [{ id: 0, name: "soil" }, { id: 1, name: "block" }]);
  assert.deepEqual(Array.from(s.registry.ids), IDS_FULL);
  assert.deepEqual(Array.from(s.registry.bodyIds), [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]);
  assert.deepEqual(Array.from(s.registry.materialTags), [2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1]);
  assert.deepEqual(Array.from(s.registry.mass), [...Array(8).fill(0.125), ...Array(4).fill(0.490625)]);
  assert.equal(s.nOutputs, 3);
  assert.deepEqual(Array.from(s.steps), [0, 10, 20]);
  assert.deepEqual(Array.from(s.time), [0, 0.001, 0.002]);
  assert.deepEqual(Array.from(s.offsets), [0, 12, 24, 35]);
  assert.deepEqual(s.fields.map((f) => f.name).sort(), ["DEFORMATION_GRADIENT", "POSITION", "STATE_eqPlasticStrain", "STRAIN", "STRESS", "VELOCITY", "VOLUME"]);
  assert.deepEqual(s.refusedFields, []);
  assert.equal(s.torn, null);
  const fg = s.fields.find((f) => f.name === "DEFORMATION_GRADIENT")!;
  assert.equal(fg.dimension, "", "an absent DIMENSION reads as \"\" (the fork does not write an empty string)");
  assert.deepEqual(fg.components, ["F11", "F12", "F21", "F22"]);
  const pos = s.fields.find((f) => f.name === "POSITION")!;
  assert.deepEqual({ displayName: pos.displayName, components: pos.components, dimension: pos.dimension, dataType: pos.dataType, dtype: pos.dtype }, { displayName: "Position", components: ["x", "y"], dimension: "L", dataType: 1, dtype: "f64" });

  // ids per output: particle 11 is gone after output 1
  assert.deepEqual(Array.from(p.outputIds(s.stageId, 0)), IDS_FULL);
  assert.deepEqual(Array.from(p.outputIds(s.stageId, 1)), IDS_FULL);
  assert.deepEqual(Array.from(p.outputIds(s.stageId, 2)), IDS_FULL.slice(0, 11));

  const out2 = p.readOutput(s.stageId, 2, "POSITION");
  assert.equal(out2.nComp, 2);
  assert.equal(out2.step, 20);
  assert.equal(out2.time, 0.002);
  assert.deepEqual(Array.from(out2.rows), IDS_FULL.slice(0, 11), "each id's registry row");
  assert.deepEqual(Array.from(out2.values), POSITION_2.flat());

  // STATE_eqPlasticStrain: 0, 1e-4, 2e-4 on the soil, NaN on the block (whose material does not answer)
  for (const [k, soil] of [[0, 0], [1, 0.0001], [2, 0.0002]] as const) {
    const o = p.readOutput(s.stageId, k, "STATE_eqPlasticStrain");
    assert.equal(o.nComp, 1);
    o.ids.forEach((id, i) => {
      const v = o.values[i]!;
      if (s.registry.bodyIds[o.rows[i]!] === 1) assert.ok(Number.isNaN(v), `output ${k} particle ${id} (block): NaN, read ${v}`);
      else assert.equal(v, soil, `output ${k} particle ${id} (soil)`);
    });
    assert.equal(o.ids.length, k === 2 ? 11 : 12);
  }
  const st = p.readOutput(s.stageId, 0, "STRESS");
  assert.deepEqual(Array.from(st.values.subarray(0, 3)), [-3.67875, -7.3575, 0]);
  assert.deepEqual(Array.from(st.values.subarray(33, 36)), [0, 0, 0]);
}

test("particles_v1m2 (all particles): the one particle stage reads as the fixture holds it", () => {
  withFile(ALL, (f, p) => {
    assert.deepEqual(p.refused, []);
    assert.equal(p.stages.length, 1);
    const s = p.stages[0]!;
    assert.equal(s.stageId, "stage_0", "the id reader/ladruno.ts gives MODEL_STAGE[1], the file's only stage");
    assert.equal(s.group, "/MODEL_STAGE[1]");
    assert.equal(f.stage(s.stageId).name, "MODEL_STAGE[1]");
    assertFixtureStage(s, p);
    assert.deepEqual(f.index.warnings, []);
  });
});

test("particles_mixed_v1m2: the FE stage reads as before, and the particle stage reads as the fixture holds it", () => {
  withFile(MIXED, (f, p) => {
    assert.equal(p.stages.length, 1, "the FE stage holds no particles");
    const s = p.stages[0]!;
    assert.equal(s.stageId, "stage_1");
    assert.equal(s.group, "/MODEL_STAGE[1]");
    assertFixtureStage(s, p);
    const fe = f.stage("stage_0");
    assert.ok(!fe.unread.some((u) => /PARTICLES/.test(u)), "the FE stage lists no particle bucket");
    assert.ok((fe.nodes?.components.length ?? 0) > 0, "the FE stage still reads its nodal results");
  });
});

test("a read outside the stage, the outputs or the fields says why", () => {
  withFile(ALL, (_f, p) => {
    assert.throws(() => p.outputIds("stage_0", 3), /output 3 is outside stage stage_0 \(3 outputs\)/);
    assert.throws(() => p.readOutput("stage_0", 0, "PRESSURE"), /records no particle field "PRESSURE"/);
    assert.throws(() => p.stage("stage_9"), /stage stage_9 holds no particles/);
  });
});

// -- a part file -----------------------------------------------------------------

test("a part file that holds particles is refused by name (C2-PART)", () => {
  const part = join(tmp, "run.part-0.ladruno");
  copyFileSync(ALL, part);
  const w = new h5wasm.File(part, "a");
  try {
    const info = w.get("INFO") as h5wasm.Group;
    for (const [k, v] of [["PARTITIONED", 1], ["NUM_PARTITIONS", 2]] as const) {
      info.delete_attribute(k);
      info.create_attribute(k, new Int32Array([v]), [1]);
    }
  } finally {
    w.close();
  }
  assert.throws(
    () => openLadruno(H5, part, { part: true }),
    (e: unknown) => e instanceof LadrunoError && e.kind === "unsupported" && /run\.part-0\.ladruno is one part of a partitioned run and \/MODEL_STAGE\[1\] holds particles.*C2-PART/.test(e.message),
  );
  assert.throws(() => openLadruno(H5, part), /is part 0 of the partitioned set/, "opened alone, the name refuses it first, as for any part");
});

// -- the crash rule (§2.5) -------------------------------------------------------

/** A copy of the all-particle fixture whose ON_PARTICLES datasets are resized to `rows` (a killed writer's file). */
function tornCopy(name: string, rows: Record<string, number>): string {
  const path = join(tmp, name);
  copyFileSync(ALL, path);
  const w = new h5wasm.File(path, "a");
  try {
    for (const [ds, n] of Object.entries(rows)) {
      const d = w.get(`MODEL_STAGE[1]/RESULTS/ON_PARTICLES/${ds}`) as h5wasm.Dataset;
      assert.ok(d, ds);
      const shape = d.shape!;
      d.resize([n, ...shape.slice(1)]);
    }
  } finally {
    w.close();
  }
  return path;
}

test("a torn tail: the output being written (ID and part of DATA appended, OFFSETS not yet) is not read; every committed one is", () => {
  // Output 2 in flight: its 11 ids written, POSITION's rows half written, STEP written, TIME and OFFSETS not yet.
  const path = tornCopy("torn_inflight.ladruno", { OFFSETS: 3, STEP: 3, TIME: 2, "POSITION/DATA": 30 });
  withFile(path, (f, p) => {
    const s = p.stage("stage_0");
    assert.equal(s.nOutputs, 2);
    assert.deepEqual(Array.from(s.offsets), [0, 12, 24]);
    assert.deepEqual(Array.from(s.steps), [0, 10]);
    assert.match(s.torn!, /STEP holds 3, TIME 2 and OFFSETS 3 entries, so 2 outputs are committed/);
    assert.match(s.torn!, /rows past 24 .*ID: 35.*are a partial output and are not read/);
    assert.deepEqual(Array.from(p.outputIds("stage_0", 1)), IDS_FULL);
    assert.deepEqual(Array.from(p.readOutput("stage_0", 1, "STATE_eqPlasticStrain").values.subarray(0, 8)), Array(8).fill(0.0001));
    assert.throws(() => p.readOutput("stage_0", 2, "POSITION"), /output 2 is outside stage stage_0 \(2 outputs; a torn tail/);
    assert.ok(f.index.warnings.some((w) => /ON_PARTICLES: 2 outputs read; a torn tail/.test(w)), "the index warns");
  });
});

test("a torn tail: ID longer than a field's DATA, with STEP, TIME and OFFSETS already written, drops that output by OFFSETS[k+1] against each length", () => {
  const path = tornCopy("torn_data.ladruno", { "STRESS/DATA": 30 });
  withFile(path, (_f, p) => {
    const s = p.stage("stage_0");
    assert.equal(s.nOutputs, 2, "OFFSETS[3] = 35 exceeds STRESS's 30 rows");
    assert.match(s.torn!, /STRESS\/DATA holds 30 rows, short of output 2's end 35, so 1 committed output is dropped/);
    assert.deepEqual(Array.from(p.readOutput("stage_0", 1, "POSITION").ids), IDS_FULL);
  });
  const shortId = tornCopy("torn_id.ladruno", { ID: 30 });
  withFile(shortId, (_f, p) => {
    assert.equal(p.stage("stage_0").nOutputs, 2, "OFFSETS[3] = 35 exceeds ID's 30 rows");
  });
});

test("a run killed before its first output commits reads as a stage with no outputs, not a refusal", () => {
  const path = tornCopy("torn_none.ladruno", { OFFSETS: 1, STEP: 0, TIME: 0 });
  withFile(path, (_f, p) => {
    const s = p.stage("stage_0");
    assert.equal(s.nOutputs, 0);
    assert.deepEqual(Array.from(s.offsets), [0]);
    assert.deepEqual(Array.from(s.registry.ids), IDS_FULL, "the registry still reads");
    assert.throws(() => p.outputIds("stage_0", 0), /outside stage stage_0 \(0 outputs/);
  });
});

test("a decreasing OFFSETS refuses the particle stage by name (C2-OFF), and the file still opens", () => {
  const path = join(tmp, "bad_offsets.ladruno");
  copyFileSync(ALL, path);
  const w = new h5wasm.File(path, "a");
  try {
    (w.get("MODEL_STAGE[1]/RESULTS/ON_PARTICLES/OFFSETS") as h5wasm.Dataset).write_slice([[1, 2]], new BigInt64Array([30n]));
  } finally {
    w.close();
  }
  const f = openLadruno(H5, path);
  try {
    assert.equal(f.particles!.stages.length, 0);
    assert.equal(f.particles!.refused.length, 1);
    assert.match(f.particles!.refused[0]!.reason, /OFFSETS decreases at 2 \(C2-OFF\)/);
    assert.ok(f.index.warnings.some((x) => /MODEL_STAGE\[1\]: its particles are refused and not read: .*C2-OFF/.test(x)));
    assert.throws(() => f.particles!.stage("stage_0"), /stage stage_0's particles are refused/);
  } finally {
    f.close();
  }
});

// -- chunk shape and string encoding (§0) ------------------------------------------

/** h5wasm's `.value` of an attribute as one JS value (a 1-element array unwrapped, NULs trimmed). */
function attrValue(a: h5wasm.Attribute): unknown {
  const v = a.value as unknown;
  const one = Array.isArray(v) || (ArrayBuffer.isView(v) && !(v instanceof DataView)) ? ((v as ArrayLike<unknown>).length === 1 ? (v as ArrayLike<unknown>)[0] : v) : v;
  return typeof one === "string" ? one.replace(/\0+$/, "") : one;
}

/** Copy a group's attributes, strings as scalar variable-length (h5py's encoding), numbers as they are. */
function copyAttrs(from: h5wasm.Group | h5wasm.Dataset, to: h5wasm.Group | h5wasm.Dataset): void {
  for (const [k, a] of Object.entries(from.attrs)) {
    const v = attrValue(a);
    if (typeof v === "string") to.create_attribute(k, v);
    else to.create_attribute(k, a.value as never, a.shape, a.dtype as never);
  }
}

/** The fixture's particle stage rewritten into a fresh file, with `chunks` rows per chunk (null: contiguous). */
function rewritten(name: string, chunks: number | null): string {
  const path = join(tmp, name);
  const src = new h5wasm.File(ALL, "r");
  const out = new h5wasm.File(path, "w");
  try {
    const info = out.create_group("INFO");
    copyAttrs(src.get("INFO") as h5wasm.Group, info);
    const stage = out.create_group("MODEL_STAGE[0]");
    copyAttrs(src.get("MODEL_STAGE[1]") as h5wasm.Group, stage);
    const copyTree = (from: h5wasm.Group, to: h5wasm.Group) => {
      copyAttrs(from, to);
      for (const k of from.keys()) {
        const item = from.get(k);
        if (item instanceof h5wasm.Group) copyTree(item, to.create_group(k));
        else if (item instanceof h5wasm.Dataset) {
          const shape = item.shape!;
          const strings = item.metadata.type === 3;
          const data = strings ? (item.value as string[]).map((x) => x.replace(/\0+$/, "")) : (item.value as never);
          const ds = to.create_dataset({
            name: k,
            data,
            shape,
            ...(strings ? {} : { dtype: item.dtype as never }),
            ...(chunks === null || strings || shape[0] === 0 ? {} : { chunks: [Math.min(chunks, shape[0]!), ...shape.slice(1)], maxshape: [null, ...shape.slice(1)] }),
          });
          copyAttrs(item, ds);
        }
      }
    };
    const model = stage.create_group("MODEL");
    copyTree(src.get("MODEL_STAGE[1]/MODEL/PARTICLES") as h5wasm.Group, model.create_group("PARTICLES"));
    const results = stage.create_group("RESULTS");
    copyTree(src.get("MODEL_STAGE[1]/RESULTS/ON_PARTICLES") as h5wasm.Group, results.create_group("ON_PARTICLES"));
  } finally {
    out.close();
    src.close();
  }
  return path;
}

for (const [label, chunks] of [["contiguous datasets", null], ["5-row chunks", 5]] as const) {
  test(`${label} and variable-length string attributes read the same particles (the reader does not depend on the chunk shape)`, () => {
    const path = rewritten(`rewritten_${chunks ?? "contiguous"}.ladruno`, chunks);
    const check = new h5wasm.File(path, "r");
    try {
      const pos = check.get("MODEL_STAGE[0]/RESULTS/ON_PARTICLES/POSITION/DATA") as h5wasm.Dataset;
      assert.deepEqual(pos.metadata.chunks, chunks === null ? null : [5, 2], "the file has the layout this case names");
      assert.equal((check.get("MODEL_STAGE[0]") as h5wasm.Group).attrs["KIND"]!.metadata.vlen, true, "strings are variable-length here");
    } finally {
      check.close();
    }
    withFile(path, (_f, p) => {
      assert.deepEqual(p.refused, []);
      const s = p.stage("stage_0");
      assert.equal(s.group, "/MODEL_STAGE[0]");
      assertFixtureStage(s, p);
    });
  });
}
