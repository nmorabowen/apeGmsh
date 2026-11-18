// reader/ladruno.ts on a killed recorder run (M2-S8 #3003): the fork's schema §7b.2 at the
// release ladruno-v20261116 (the FORK_PIN).
//
// The oracle is tests/fixtures/ladruno_killed/killed_swmr.ladruno, a run of the release build
// killed while it recorded (its README), as h5wasm reads it raw, and the committed steps the
// fork's own `committed_steps` names for it: 20 and 15 (tests/results/test_ladruno_killed_run.py
// checks those against the vendored reference, so Python and the viewer agree on one table).
// The ragged and torn cases are copies this test writes into a temp folder: a stage's datasets
// resized as a kill inside a flush leaves them, and a stage whose object header signature is
// flipped (a stage torn while laid out), at the offsets Python's test finds with h5py.

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { copyFileSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { H5BusyError, lookup3, repairCopy, superblockIo, SWMR_TAIL_READ } from "../../src/reader/h5open.ts";
import { committedSteps, LadrunoError, openLadruno, type LadrunoFile } from "../../src/reader/ladruno.ts";
import type { ResultsH5 } from "../../src/reader/results.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as ResultsH5;
const repo = (rel: string) => fileURLToPath(new URL(`../../../${rel}`, import.meta.url));
const KILLED = repo("tests/fixtures/ladruno_killed/killed_swmr.ladruno");
const tmp = mkdtempSync(join(tmpdir(), "ladruno-killed-"));
after(() => rmSync(tmp, { recursive: true, force: true }));

const S2 = "MODEL_STAGE[2]";
const DISP = "RESULTS/ON_NODES/DISPLACEMENT";
const FORCE = "RESULTS/ON_ELEMENTS/force/3-ElasticBeam2d[1:0:0]";
/** The object header of each stage group in KILLED (h5py's `h5o.get_info(...).addr`; asserted below). */
const OHDR = { "MODEL_STAGE[1]": 45736, "MODEL_STAGE[2]": 263035 } as const;

const sha = (path: string) => createHash("sha256").update(readFileSync(path)).digest("hex");

/** A copy of `src` with superblock flags `flags` (and EOA = file size when `eoaToSize`), re-checksummed. */
function flagged(src: string, flags: number, name: string, eoaToSize = false): string {
  const path = join(tmp, name);
  copyFileSync(src, path);
  const b = readFileSync(path);
  b[11] = flags;
  if (eoaToSize) b.writeBigUInt64LE(BigInt(b.length), 28);
  b.writeUInt32LE(lookup3(b.subarray(0, 44)), 44);
  writeFileSync(path, b);
  return path;
}

/** KILLED's MODEL_STAGE[2] ragged as Python's test makes it: DISPLACEMENT/DATA 14 rows, force/STEP 13, the rest 15. */
function raggedBase(): string {
  const path = join(tmp, "ragged_base.ladruno");
  repairCopy(KILLED, path);
  const w = new h5wasm.File(path, "a");
  try {
    (w.get(`${S2}/${DISP}/DATA`) as h5wasm.Dataset).resize([14, 5, 2]);
    (w.get(`${S2}/${FORCE}/STEP`) as h5wasm.Dataset).resize([13]);
  } finally {
    w.close();
  }
  return path;
}

function torn(stage: keyof typeof OHDR, flags: number, name: string): string {
  const path = flagged(KILLED, flags, name, flags === 0x00);
  const b = readFileSync(path);
  assert.equal(b.subarray(OHDR[stage], OHDR[stage] + 4).toString("latin1"), "OHDR", `${stage}'s object header`);
  b[OHDR[stage]] ^= 0xff;
  writeFileSync(path, b);
  return path;
}

function raw(path: string, ds: string, rows: number): Float64Array {
  const f = new h5wasm.File(path, "Sr");
  try {
    return (f.get(ds) as h5wasm.Dataset).slice([[0, rows]]) as Float64Array;
  } finally {
    f.close();
  }
}

function withFile<T>(path: string, body: (f: LadrunoFile) => T): T {
  const f = openLadruno(H5, path);
  try {
    return body(f);
  } finally {
    f.close();
  }
}

const steps = (f: LadrunoFile) => f.index.stages.map((s) => [s.id, s.nSteps] as const);

test("the fixture: a run killed with SWMR armed, read to the committed steps the fork's committed_steps names", () => {
  const before = sha(KILLED);
  withFile(KILLED, (f) => {
    assert.deepEqual(f.opened, { step: 2, repaired: false, flags: 0x05 });
    assert.equal(f.killed, true);
    // The empty MODES_OF_VIBRATION(U) of MODEL_STAGE[2] (OpenSees#1084) gives no mode stage.
    assert.deepEqual(steps(f), [["stage_0", 20], ["stage_0_mode_1", 1], ["stage_0_mode_2", 1], ["stage_1", 15]]);
    const s1 = f.stage("stage_1");
    assert.deepEqual(Array.from(s1.time), Array.from(raw(KILLED, `${S2}/${DISP}/TIME`, 15)));
    // Node 5 is the fifth ID row; DATA is [T, 5, 2].
    const d = raw(KILLED, `${S2}/${DISP}/DATA`, 15);
    assert.deepEqual(Array.from(f.readNodeHistory("stage_1", "DISPLACEMENT/Ux", 5)), Array.from({ length: 15 }, (_, t) => d[t * 10 + 4 * 2]!));
    const rows = f.readNodalRows("stage_1", ["DISPLACEMENT/Uy"], 0, 15)[0]!;
    assert.deepEqual(Array.from(rows), Array.from({ length: 75 }, (_, k) => d[k * 2 + 1]!));
  });
  assert.equal(sha(KILLED), before, "the user's file is never written");
});

test("committedSteps is the fork's T' per stage", () => {
  const f = new h5wasm.File(KILLED, "Sr");
  try {
    const at = (name: string) => committedSteps(f.get(name) as never, H5);
    assert.deepEqual(at("MODEL_STAGE[1]"), { n: 20 });
    assert.deepEqual(at(S2), { n: 15 });
  } finally {
    f.close();
  }
});

for (const [flags, step, repaired] of [[0x05, 2, false], [0x01, 3, true]] as const) {
  test(`a ragged stage of a killed run (flags 0x0${flags}) is cut to T' = 13`, () => {
    const path = flagged(raggedBase(), flags, `ragged_${flags}.ladruno`);
    const before = sha(path);
    withFile(path, (f) => {
      assert.deepEqual(f.opened, { step, repaired, flags });
      assert.equal(f.stage("stage_0").nSteps, 20);
      const s1 = f.stage("stage_1");
      assert.equal(s1.nSteps, 13);
      assert.deepEqual(s1.nodes!.refused, []);
      assert.deepEqual(Array.from(s1.time), Array.from(raw(KILLED, `${S2}/${DISP}/TIME`, 13)));
      const d = raw(KILLED, `${S2}/${DISP}/DATA`, 13);
      assert.deepEqual(Array.from(f.readNodeHistory("stage_1", "DISPLACEMENT/Uy", 5)), Array.from({ length: 13 }, (_, t) => d[t * 10 + 4 * 2 + 1]!));
      assert.throws(() => f.readNodalRows("stage_1", ["DISPLACEMENT/Ux"], 0, 14), LadrunoError, "no step past T'");
    });
    assert.equal(sha(path), before);
  });
}

test("a cleanly closed ragged file reads as it is: no cut, and its disagreeing result is refused as before", () => {
  const path = flagged(raggedBase(), 0x00, "ragged_0.ladruno");
  withFile(path, (f) => {
    assert.equal(f.killed, false);
    const s1 = f.stage("stage_1");
    // Its time is the next TIME (the force bucket's, 15 rows), as Python's reader keeps 15.
    assert.equal(s1.nSteps, 15);
    assert.deepEqual(s1.nodes!.refused.map((r) => r.component), ["DISPLACEMENT"]);
    assert.match(s1.nodes!.refused[0]!.reason, /holds 14 steps, TIME 15 and STEP 15: the step axes disagree/);
  });
});

for (const flags of [0x00, 0x01]) {
  test(`a torn last stage is skipped after ${flags === 0 ? "a plain open" : "a repair of a 0x01 file"}`, () => {
    const path = torn(S2, flags, `torn_last_${flags}.ladruno`);
    const before = sha(path);
    withFile(path, (f) => {
      assert.deepEqual(steps(f), [["stage_0", 20], ["stage_0_mode_1", 1], ["stage_0_mode_2", 1]]);
      assert.ok(f.index.warnings.some((w) => /MODEL_STAGE\[2\] was torn while it was laid out .* it is skipped/.test(w)), f.index.warnings.join("\n"));
    });
    assert.equal(sha(path), before);
  });
}

test("a torn last stage refuses a file killed with SWMR armed", () => {
  const path = torn(S2, 0x05, "torn_last_5.ladruno");
  assert.throws(() => openLadruno(H5, path), (e: unknown) => e instanceof LadrunoError && /MODEL_STAGE\[2\] does not read in a file killed with SWMR armed/.test(e.message));
});

for (const flags of [0x00, 0x01, 0x05]) {
  test(`an earlier stage that does not read refuses the file (flags 0x0${flags})`, () => {
    const path = torn("MODEL_STAGE[1]", flags, `torn_first_${flags}.ladruno`);
    assert.throws(() => openLadruno(H5, path), (e: unknown) => e instanceof LadrunoError && /MODEL_STAGE\[1\] does not read .*corruption/.test(e.message));
  });
}

test("a live writer's lock is H5BusyError, naming the SWMR tail read", () => {
  const real = superblockIo.readHead;
  superblockIo.readHead = () => {
    throw Object.assign(new Error("EBUSY: resource busy or locked"), { code: "EBUSY" });
  };
  try {
    assert.throws(() => openLadruno(H5, KILLED), (e: unknown) => e instanceof H5BusyError && /a live writer holds it/.test(e.message) && e.message.includes(SWMR_TAIL_READ));
  } finally {
    superblockIo.readHead = real;
  }
});
