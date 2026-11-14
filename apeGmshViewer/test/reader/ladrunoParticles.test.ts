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
import { metaOf } from "../../src/main/results.ts";
import { pairingNotice, pairResults, particlesOnly, PARTICLES_UNREAD } from "../../src/state/results.ts";

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

/**
 * The fixture's particle stage rewritten into a fresh file, with `chunks` rows per chunk
 * (null: contiguous). `skip` leaves out the links with these paths (relative to the stage),
 * and `dtype` writes a dataset (same relative path) with another dtype: the review's repros.
 */
function rewritten(name: string, chunks: number | null, opts: { skip?: string[]; dtype?: Record<string, string>; stages?: number } = {}): string {
  const path = join(tmp, name);
  const src = new h5wasm.File(ALL, "r");
  const out = new h5wasm.File(path, "w");
  try {
    const info = out.create_group("INFO");
    copyAttrs(src.get("INFO") as h5wasm.Group, info);
    // `stages` copies of the fixture's particle stage, MODEL_STAGE[0] .. [stages - 1].
    for (let k = 0; k < (opts.stages ?? 1); k++) writeStage(out.create_group(`MODEL_STAGE[${k}]`));
  } finally {
    out.close();
    src.close();
  }
  return path;

  function writeStage(stage: h5wasm.Group): void {
    copyAttrs(src.get("MODEL_STAGE[1]") as h5wasm.Group, stage);
    const rel = (path: string) => path.replace(/^\/MODEL_STAGE\[1\]\//, "");
    const copyTree = (from: h5wasm.Group, to: h5wasm.Group) => {
      copyAttrs(from, to);
      for (const k of from.keys()) {
        const item = from.get(k);
        const at = rel(`${from.path}/${k}`);
        if (opts.skip?.includes(at)) continue;
        const retyped = opts.dtype?.[at];
        if (item instanceof h5wasm.Group) copyTree(item, to.create_group(k));
        else if (item instanceof h5wasm.Dataset) {
          const shape = item.shape!;
          const strings = item.metadata.type === 3;
          const value = item.value as unknown;
          const data = strings
            ? (value as string[]).map((x) => x.replace(/\0+$/, ""))
            : retyped === "<Q"
              ? BigUint64Array.from(value as BigInt64Array, (x) => BigInt.asUintN(64, x))
              : (value as never);
          const ds = to.create_dataset({
            name: k,
            data: data as never,
            shape,
            ...(strings ? {} : { dtype: (retyped ?? item.dtype) as never }),
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
  }
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

// -- #2985 review: findings 2, 3 and 4 ---------------------------------------------

/** The review's refused stage: the file still opens, the refusal is said, and nothing claims the particles read. */
function assertRefusedStage(path: string, code: RegExp): void {
  const f = openLadruno(H5, path);
  try {
    assert.equal(f.particles!.stages.length, 0);
    assert.equal(f.particles!.refused.length, 1);
    assert.match(f.particles!.refused[0]!.reason, code);
    const unread = f.index.stages[0]!.unread;
    assert.ok(!unread.includes(PARTICLES_UNREAD), "a refused stage never lists its particles as read");
    assert.ok(unread.some((u) => /^ON_PARTICLES \(particles refused, not read: /.test(u)), unread.join("\n"));
    assert.ok(f.index.warnings.some((w) => /its particles are refused and not read/.test(w)), "the index warns");
    const meta = metaOf(f);
    assert.equal(particlesOnly(meta), false);
    assert.deepEqual(pairResults(null, meta), { state: "noModel" }, "the banner does not say the particles read");
    assert.doesNotMatch(pairingNotice(pairResults(null, meta), "x.ladruno")!, /particles only/);
  } finally {
    f.close();
  }
}

test("finding 2: an all-particle file whose POSITION is missing refuses its stage (C2-POS), and the banner does not claim the particles read", () => {
  assertRefusedStage(rewritten("no_position.ladruno", null, { skip: ["RESULTS/ON_PARTICLES/POSITION"] }), /has no POSITION field \(C2-POS\)/);
});

test("finding 3: a registry ID stored as uint64 is refused (C2-REG), never read as int64", () => {
  assertRefusedStage(rewritten("uint64_registry.ladruno", null, { dtype: { "MODEL/PARTICLES/ID": "<Q" } }), /MODEL\/PARTICLES\/ID is not int64 .*unsigned.*\(C2-REG\)/);
});

test("finding 3: ON_PARTICLES@PARTITION_REDUCTION other than NONE is refused (C2-RED)", () => {
  const path = join(tmp, "reduction.ladruno");
  copyFileSync(ALL, path);
  const w = new h5wasm.File(path, "a");
  try {
    const on = w.get("MODEL_STAGE[1]/RESULTS/ON_PARTICLES") as h5wasm.Group;
    on.delete_attribute("PARTITION_REDUCTION");
    on.create_attribute("PARTITION_REDUCTION", "SUM");
  } finally {
    w.close();
  }
  assertRefusedStage(path, /ON_PARTICLES@PARTITION_REDUCTION = "SUM"; expected "NONE".*\(C2-RED\)/);
});

test("finding 3: a STEP that does not strictly increase is refused (C2-AXES)", () => {
  const path = join(tmp, "step.ladruno");
  copyFileSync(ALL, path);
  const w = new h5wasm.File(path, "a");
  try {
    (w.get("MODEL_STAGE[1]/RESULTS/ON_PARTICLES/STEP") as h5wasm.Dataset).write_slice([[2, 3]], new Int32Array([10]));
  } finally {
    w.close();
  }
  assertRefusedStage(path, /STEP is not strictly increasing at output 2 \(10, then 10\) \(C2-AXES\)/);
});

test("finding 4: an OFFSETS entry extended but not yet filled (0, below the one before it) is the output in flight; the committed outputs read", () => {
  // The writer appended output 3's DATA, ID, STEP and TIME, then extended OFFSETS, and was killed before it wrote the entry.
  const path = tornCopy("torn_offsets.ladruno", { OFFSETS: 5, STEP: 4, TIME: 4 });
  withFile(path, (f, p) => {
    const s = p.stage("stage_0");
    assert.equal(s.nOutputs, 3, "the three committed outputs");
    assert.deepEqual(Array.from(s.offsets), [0, 12, 24, 35]);
    assert.match(s.torn!, /OFFSETS\[4\] = 0 is below the entry before it, so it is the output being written and is not counted/);
    assert.deepEqual(Array.from(p.outputIds("stage_0", 2)), IDS_FULL.slice(0, 11));
    assert.ok(f.index.warnings.some((w) => /ON_PARTICLES: 3 outputs read; a torn tail/.test(w)));
  });
});

test("the fork's edge case: when every output so far is empty, a torn OFFSETS entry equals the one before it and reads as one more empty output", () => {
  const empty: Record<string, number> = { ID: 0, OFFSETS: 3, STEP: 2, TIME: 2 };
  for (const field of ["POSITION", "VELOCITY", "STRESS", "STRAIN", "VOLUME", "DEFORMATION_GRADIENT", "STATE_eqPlasticStrain"]) empty[`${field}/DATA`] = 0;
  const path = tornCopy("all_empty.ladruno", empty);
  const w = new h5wasm.File(path, "a");
  try {
    (w.get("MODEL_STAGE[1]/RESULTS/ON_PARTICLES/OFFSETS") as h5wasm.Dataset).write_slice([[1, 3]], new BigInt64Array([0n, 0n]));
  } finally {
    w.close();
  }
  withFile(path, (_f, p) => {
    assert.deepEqual(p.refused, []);
    const s = p.stage("stage_0");
    assert.equal(s.nOutputs, 2, "the committed empty output and the torn one, read as one more empty output");
    assert.deepEqual(Array.from(s.offsets), [0, 0, 0]);
    assert.equal(s.torn, null);
    for (const k of [0, 1]) {
      assert.equal(p.outputIds("stage_0", k).length, 0);
      const o = p.readOutput("stage_0", k, "POSITION");
      assert.equal(o.ids.length, 0);
      assert.equal(o.values.length, 0);
    }
  });
});

// -- §2.5's crash rule, step by step (the fork's final text, fork commit 7b949e58d) --------

/** A fixture copy whose OFFSETS rows `[from, from + values.length)` are `values` (after the resizes `rows` names). */
function offsetsCopy(name: string, rows: Record<string, number>, from: number, values: bigint[]): string {
  const path = tornCopy(name, rows);
  const w = new h5wasm.File(path, "a");
  try {
    (w.get("MODEL_STAGE[1]/RESULTS/ON_PARTICLES/OFFSETS") as h5wasm.Dataset).write_slice([[from, from + values.length]], new BigInt64Array(values));
  } finally {
    w.close();
  }
  return path;
}

test("crash rule step 1: OFFSETS[0] other than 0 refuses the stage (C2-OFF)", () => {
  assertRefusedStage(offsetsCopy("step1.ladruno", {}, 0, [5n]), /OFFSETS\[0\] = 5; expected 0 \(C2-OFF\)/);
});

test("crash rule step 3: a decrease before the torn last entry still refuses the stage, though step 2 dropped that entry", () => {
  // OFFSETS [0, 30, 24, 0]: step 2 drops the torn last entry, and [0, 30, 24] still decreases.
  assertRefusedStage(offsetsCopy("step3.ladruno", {}, 1, [30n, 24n, 0n]), /OFFSETS decreases at 2 \(C2-OFF\)/);
});

test("crash rule: T' is bounded by a short STEP as by a short TIME", () => {
  withFile(tornCopy("short_step.ladruno", { STEP: 2 }), (_f, p) => {
    const s = p.stage("stage_0");
    assert.equal(s.nOutputs, 2);
    assert.match(s.torn!, /STEP holds 2, TIME 3 and OFFSETS 4 entries, so 2 outputs are committed/);
  });
  withFile(tornCopy("short_time.ladruno", { TIME: 1 }), (_f, p) => {
    const s = p.stage("stage_0");
    assert.equal(s.nOutputs, 1);
    assert.match(s.torn!, /STEP holds 3, TIME 1 and OFFSETS 4 entries, so 1 output is committed/);
  });
});

// -- #2985 / #2987 review follow-ups (M2-S4 #2983) -------------------------------------

/** A fixture copy resized as `rows` names, then edited in place by `edit` (h5wasm, append mode). */
function editedCopy(name: string, edit: (w: h5wasm.File) => void, rows: Record<string, number> = {}): string {
  const path = tornCopy(name, rows);
  const w = new h5wasm.File(path, "a");
  try {
    edit(w);
  } finally {
    w.close();
  }
  return path;
}

const ON = "MODEL_STAGE[1]/RESULTS/ON_PARTICLES";

test("#2985 follow-up: a refused field's DATA still bounds the committed outputs, as the fork's committed_outputs reads it", () => {
  // STRESS loses its TYPE (refused, C2-ATTR) and was cut short at 30 rows, below output 2's end 35.
  const path = editedCopy("refused_short.ladruno", (w) => (w.get(`${ON}/STRESS`) as h5wasm.Group).delete_attribute("TYPE"), { "STRESS/DATA": 30 });
  withFile(path, (_f, p) => {
    const s = p.stage("stage_0");
    assert.ok(s.refusedFields.some((x) => x.field === "STRESS" && /has no @TYPE \(C2-ATTR\)/.test(x.reason)), "STRESS is refused");
    assert.equal(s.nOutputs, 2, "OFFSETS[3] = 35 exceeds the refused STRESS's 30 rows");
    assert.match(s.torn!, /STRESS\/DATA holds 30 rows, short of output 2's end 35, so 1 committed output is dropped/);
  });
});

test("#2987 follow-up: a field group with no TYPE, or with PARTITION_REDUCTION other than NONE, is refused by name; the stage's other fields read", () => {
  const noType = editedCopy("no_type.ladruno", (w) => (w.get(`${ON}/VELOCITY`) as h5wasm.Group).delete_attribute("TYPE"));
  const sum = editedCopy("field_sum.ladruno", (w) => {
    const g = w.get(`${ON}/VOLUME`) as h5wasm.Group;
    g.delete_attribute("PARTITION_REDUCTION");
    g.create_attribute("PARTITION_REDUCTION", "SUM");
  });
  for (const [path, field, why] of [
    [noType, "VELOCITY", /VELOCITY has no @TYPE \(C2-ATTR\)/],
    [sum, "VOLUME", /VOLUME@PARTITION_REDUCTION = "SUM"; expected "NONE" \(C2-RED\)/],
  ] as const) {
    withFile(path, (f, p) => {
      const s = p.stage("stage_0");
      assert.deepEqual(s.refusedFields.map((x) => x.field), [field]);
      assert.match(s.refusedFields[0]!.reason, why);
      assert.ok(!s.fields.some((x) => x.name === field), `${field} is not offered`);
      assert.throws(() => p.readOutput("stage_0", 0, field), new RegExp(`${field} is refused`));
      assert.equal(s.nOutputs, 3, "the other fields read every output");
      assert.ok(f.index.warnings.some((x) => new RegExp(`ON_PARTICLES/${field}: refused and not read`).test(x)), "the index warns");
    });
  }
});

test("#2987 follow-up: registry columns of unequal length beside a committed output refuse the stage (C2-REG): no writer leaves that", () => {
  assertRefusedStage(
    editedCopy("short_mass.ladruno", (w) => (w.get("MODEL_STAGE[1]/MODEL/PARTICLES/MASS") as h5wasm.Dataset).resize([11])),
    /registry columns differ in length \(ID 12, BODY_ID 12, MATERIAL_TAG 12, MASS 11\), and the stage holds 3 committed outputs, so it is no torn opening \(C2-REG\)/,
  );
  assertRefusedStage(
    editedCopy("short_body.ladruno", (w) => (w.get("MODEL_STAGE[1]/MODEL/PARTICLES/BODY_ID") as h5wasm.Dataset).resize([11])),
    /registry columns differ in length \(ID 12, BODY_ID 11, MATERIAL_TAG 12, MASS 12\), and the stage holds 3 committed outputs.*\(C2-REG\)/,
  );
});

test("#2983 contract follow-up: a stage torn at opening (registry columns of unequal length, no committed output) is skipped by name; the file opens and its other stages read", () => {
  // Two copies of the fixture's particle stage; the second one's writer died inside the registry write at opening,
  // so its MASS is short and nothing of ON_PARTICLES was appended.
  const path = rewritten("torn_opening.ladruno", 5, { stages: 2 });
  const w = new h5wasm.File(path, "a");
  try {
    (w.get("MODEL_STAGE[1]/MODEL/PARTICLES/MASS") as h5wasm.Dataset).resize([7]);
    const on = "MODEL_STAGE[1]/RESULTS/ON_PARTICLES";
    for (const ds of ["STEP", "TIME", "OFFSETS", "ID"]) (w.get(`${on}/${ds}`) as h5wasm.Dataset).resize([0]);
    for (const g of (w.get(on) as h5wasm.Group).keys()) {
      const d = w.get(`${on}/${g}/DATA`);
      if (d instanceof h5wasm.Dataset) d.resize([0, d.shape![1]!]);
    }
  } finally {
    w.close();
  }
  const f = openLadruno(H5, path);
  try {
    const p = f.particles!;
    assert.deepEqual(p.refused, [], "a torn opening is no contract break");
    assert.deepEqual(p.stages.map((s) => s.stageId), ["stage_0"], "the other stage reads");
    assertFixtureStage(p.stages[0]!, p);
    assert.equal(p.skipped.length, 1, "the torn stage is named, not dropped");
    assert.equal(p.skipped[0]!.stageId, "stage_1");
    assert.match(p.skipped[0]!.reason, /MODEL_STAGE\[1\]\/MODEL\/PARTICLES: torn at opening: the registry columns differ in length \(ID 12, BODY_ID 12, MATERIAL_TAG 12, MASS 7\) and no output was committed/);
    assert.throws(() => p.stage("stage_1"), /stage stage_1's particles are skipped: .*torn at opening/);
    assert.ok(f.index.warnings.some((x) => /MODEL_STAGE\[1\]: its particles are skipped: .*torn at opening/.test(x)), "the index warns");
    const unread = f.stage("stage_1").unread;
    assert.ok(!unread.includes(PARTICLES_UNREAD), "a skipped stage never lists its particles as read");
    assert.ok(unread.some((u) => /^ON_PARTICLES \(particles skipped, not read: .*torn at opening/.test(u)), unread.join("\n"));
    assert.equal(particlesOnly(metaOf(f)), false, "the banner does not claim every stage's particles read");
  } finally {
    f.close();
  }
});

test("#2987 follow-up: outputIds checks each output's ids as readOutput does: sorted and unique within the output, and in the registry (C2-ID)", () => {
  // Output 0 holds ids 0..11 at rows 0..11: repeat id 0 at row 1, then name id 99, which the registry lacks.
  const repeated = editedCopy("ids_repeat.ladruno", (w) => (w.get(`${ON}/ID`) as h5wasm.Dataset).write_slice([[1, 2]], new BigInt64Array([0n])));
  withFile(repeated, (_f, p) => {
    assert.throws(() => p.outputIds("stage_0", 0), /output 0's ids are not strictly increasing at row 1 \(C2-ID\)/);
    assert.throws(() => p.readOutput("stage_0", 0, "POSITION"), /output 0's ids are not strictly increasing at row 1 \(C2-ID\)/);
    assert.deepEqual(Array.from(p.outputIds("stage_0", 1)), IDS_FULL, "the other outputs still read");
  });
  const stranger = editedCopy("ids_stranger.ladruno", (w) => (w.get(`${ON}/ID`) as h5wasm.Dataset).write_slice([[11, 12]], new BigInt64Array([99n])));
  withFile(stranger, (_f, p) => {
    assert.throws(() => p.outputIds("stage_0", 0), /particle 99 is not in the stage registry \(C2-ID\)/);
    assert.throws(() => p.readOutput("stage_0", 0, "POSITION"), /particle 99 is not in the stage registry \(C2-ID\)/);
  });
});
