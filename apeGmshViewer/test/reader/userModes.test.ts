// reader/userModes.ts (ADR 0130 slice VU7a): the `/modes` table's rules and
// A6's row rule against the paired `.ladruno`.
//
// The oracle is Python's src/apeGmsh/results/user/_modes.py. `recorded_lambdas`
// on fixtures/ladruno_modes.ladruno returns {("stage_0", 0): {1:
// 381.9660112501052, 2: 2618.0339887498953}}, and `_one_row` returns the
// verdicts below, word for word. Both were printed by that module and are
// copied here; the PR body names the probe that ran them.

import assert from "node:assert/strict";
import { copyFileSync, mkdtempSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import type { FpH5 } from "../../src/reader/rf1.ts";
import { LambdasRefused, modeTableProblem, recordedLambdas, rowVerdicts, type ModeRow, type ModeTable, type RecordedLambdas } from "../../src/reader/userModes.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as FpH5;
const MODES = fileURLToPath(new URL("../../fixtures/ladruno_modes.ladruno", import.meta.url));
const TRUSS = fileURLToPath(new URL("../../../tests/fixtures/ladruno/truss2d.ladruno", import.meta.url));
const list = (d: string) => readdirSync(d);
const L1 = 381.9660112501052;
const L2 = 2618.0339887498953;

const row = (host: string, step: number, mode: number, lambda: number): ModeRow => ({
  host, kind: "eigen", step, mode, lambda, period: (2 * Math.PI) / Math.sqrt(lambda), generalizedMass: 1,
  participation: [0.5, 0.1], massRatio: [0.6, 0], cumulativeRatio: [0.6, 0],
});
const table = (rows: ModeRow[]): ModeTable => ({ components: ["MX", "MY"], rows });

test("the .ladruno's eigenvalues: stage_<i> by MODEL_STAGE order, STEP id, 1-based mode, bit-equal to Python's", () => {
  const r = recordedLambdas(H5, MODES, list);
  assert.deepEqual([...r.steps], [["stage_0", [0]]]);
  assert.deepEqual([...r.sets.values()].map((m) => [...m]), [[[1, L1], [2, L2]]]);
  // A file that records no mode holds no set.
  assert.equal(recordedLambdas(H5, TRUSS, list).sets.size, 0);
});

test("A6's row rule, verdict for verdict as Python's _one_row", () => {
  const lam: RecordedLambdas = {
    sets: new Map([["stage_0\u00000", new Map([[1, L1], [2, L2]])], ["stage_1\u00002", new Map([[1, 5]])], ["stage_1\u00004", new Map([[1, 6]])]]),
    steps: new Map([["stage_0", [0]], ["stage_1", [2, 4]]]),
  };
  const cases: [ModeRow, boolean, string][] = [
    [row("stage_0", 0, 1, L1), false, ""],
    // Inside 1e-9 relative: current; step -1 names the only set.
    [row("stage_0", -1, 2, L2 * (1 + 5e-10)), false, ""],
    [row("stage_0", 0, 1, L1 * (1 + 2e-9)), true, "lambda 381.96601201403723 here, 381.9660112501052 in the results file (more than 1e-09 relative)"],
    [row("MODEL_STAGE[1]", 0, 1, 1), true, "host 'MODEL_STAGE[1]' is not a stage_<i> name; a stamped stage name is not resolved against the results file here"],
    [row("stage_7", 0, 1, 1), true, "the results file records no modes on stage_7"],
    [row("stage_1", -1, 1, 5), true, "step -1 names the only set, and stage_1 holds 2"],
    [row("stage_0", 3, 1, 1), true, "the results file records no STEP_3 on stage_0"],
    [row("stage_0", 0, 3, 1), true, "the results file records no mode 3 on stage_0 STEP_0"],
    [row("stage_1", 4, 1, 6), false, ""],
  ];
  const got = rowVerdicts(table(cases.map((c) => c[0])), lam);
  cases.forEach(([r, stale, reason], i) => assert.deepEqual(got[i], { stale, reason }, `${r.host} ${r.step} ${r.mode}`));
  // A NaN lambda never passes the gap test.
  assert.equal(rowVerdicts(table([row("stage_0", 0, 1, NaN)]), lam)[0]!.stale, true);
});

test("a .ladruno whose modal record cannot be read refuses, naming why (Python raises ValueError)", () => {
  const d = mkdtempSync(join(tmpdir(), "agv-modes-"));
  const edit = (name: string, fn: (f: h5wasm.File) => void) => {
    const p = join(d, name);
    copyFileSync(MODES, p);
    const f = new h5wasm.File(p, "a");
    try {
      fn(f);
    } finally {
      f.close();
    }
    return p;
  };
  const noLambda = edit("nolambda.ladruno", (f) => (f.get("/MODEL_STAGE[1]/RESULTS/ON_NODES/MODES_OF_VIBRATION(U)/DATA/STEP_0/MODE_1") as h5wasm.Dataset).delete_attribute("LAMBDA"));
  assert.throws(() => recordedLambdas(H5, noLambda, list), (e: unknown) => e instanceof LambdasRefused && /^nolambda\.ladruno MODEL_STAGE\[1\]: ON_NODES\/MODES_OF_VIBRATION\(U\)\/DATA\/STEP_0\/MODE_1 has no LAMBDA attribute; the eigenvalue of this mode is not recorded\.$/.test((e as Error).message));
  const differ = edit("differ.ladruno", (f) => {
    const ds = f.get("/MODEL_STAGE[1]/RESULTS/ON_NODES/MODES_OF_VIBRATION(R)/DATA/STEP_0/MODE_0") as h5wasm.Dataset;
    ds.delete_attribute("LAMBDA");
    ds.create_attribute("LAMBDA", 1.5);
  });
  assert.throws(() => recordedLambdas(H5, differ, list), /MODES_OF_VIBRATION\(R\) lists different modes or eigenvalues than ON_NODES\/MODES_OF_VIBRATION\(U\)/);
  assert.throws(() => recordedLambdas(H5, join(d, "absent.ladruno"), list), LambdasRefused);
});

test("the table's rules are Python's table_problem", () => {
  const good = table([row("stage_0", 0, 1, 4), row("stage_0", 0, 2, 9)]);
  assert.equal(modeTableProblem(good), null);
  const bad: [string, ModeTable][] = [
    ["the table has no rows", table([])],
    ["components MX,MX must be distinct tokens of MX, MY, MZ, RMX, RMY, RMZ", { ...good, components: ["MX", "MX"] }],
    ["two rows name the same (host, step, mode)", table([row("stage_0", 0, 1, 4), row("stage_0", 0, 1, 9)])],
    ["modes are 1-based", table([row("stage_0", 0, 0, 4)])],
    ["a step is below -1", table([row("stage_0", -2, 1, 4)])],
    ["every lambda must be finite and > 0 (lambda <= 0 is refused)", table([row("stage_0", 0, 1, -1)])],
    ["kinds modal are not one of eigen, residual, ritz", table([{ ...row("stage_0", 0, 1, 4), kind: "modal" }])],
    ["mass_ratio holds an infinite value", table([{ ...row("stage_0", 0, 1, 4), massRatio: [Infinity, 0] }])],
  ];
  for (const [why, t] of bad) assert.equal(modeTableProblem(t), why);
});

// #2744 review, F2: Python's `discover_partition_files` refuses a part set that is not numbered 0..n-1.
test("a part set is read from part 0; a set whose parts are not numbered from 0 refuses, in Python's words", () => {
  const d = mkdtempSync(join(tmpdir(), "agv-parts-"));
  copyFileSync(MODES, join(d, "run.part-1.ladruno"));
  assert.throws(
    () => recordedLambdas(H5, join(d, "run.part-1.ladruno"), list),
    (e: unknown) => e instanceof LambdasRefused && (e as Error).message === "Partition files for stem 'run' are not contiguous from 0 — found indices [1]. Expected [0, 1].",
  );
  copyFileSync(MODES, join(d, "run.part-0.ladruno"));
  assert.deepEqual([...recordedLambdas(H5, join(d, "run.part-1.ladruno"), list).sets.values()].map((m) => [...m]), [[[1, L1], [2, L2]]]);
  copyFileSync(MODES, join(d, "run.part-3.ladruno"));
  assert.throws(() => recordedLambdas(H5, join(d, "run.part-0.ladruno"), list), /found indices \[0, 1, 3\]\. Expected \[0, 1, 2, 3\]\.$/);
});
