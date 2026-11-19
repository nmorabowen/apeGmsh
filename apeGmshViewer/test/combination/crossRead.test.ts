// VU9b (#2693, ADR 0130 A3, A4): the generations the app writes, read by Python.
//
// test/fixtures/vu9b/app_written/plate.user.h5 is the Python-written fixture
// (test/fixtures/vu9b/plate.user.h5, g0 of three rows) after three saves through
// main/combinations.ts `saveCombination` (test/main/combinationsDriver.ts: the
// default row, `--preset=cqc`, `--preset=linear`). app_written.golden.json is what
// Python read of it: `UserFile` (/meta, @current, every g<N>, each row) and the
// evaluator over `UserSpectra` for every row (test/fixtures/vu9b/README.md has both
// scripts). The oracles:
//   - Python reads exactly the rows the app wrote, with no refusal;
//   - VU4's evaluator gives, on the same file, Python's values to 1e-12 relative,
//     and refuses where Python refuses (a stage term, with no results file);
//   - every column of an app-written g<N> has the dtype of Python's g0 (vlen
//     UTF-8 text, i4, f8): the file Python wrote and the file the app wrote are
//     the same layout.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { CombinationRefused, CombinationTable, evaluate } from "../../src/combination/evaluate.ts";
import { readUserFile, type UserH5 } from "../../src/reader/userFile.ts";
import { readCaseValues, spectrumSource, type CaseRead } from "../../src/reader/userSpectra.ts";
import { spectrumCasesOf } from "../../src/state/selectors.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as UserH5;
const FIX = (p: string) => fileURLToPath(new URL(`../fixtures/vu9b/${p}`, import.meta.url));
const USER = FIX("app_written/plate.user.h5");
const MODEL = FIX("plate.h5");
type Row = { name: string; rule: string; envelope: string; fraction: number | null; damping: number | null; terms: [string, string, number, number, number][] };
const GOLDEN = JSON.parse(readFileSync(FIX("app_written.golden.json"), "utf8")) as {
  revision: number;
  writer: string;
  current: number;
  generations: number[];
  rows: Row[];
  refused: unknown[];
  refusals: Record<string, string>;
  values: Record<string, Record<string, { ids: number[]; values: number[]; signed: boolean }>>;
};
const RTOL = 1e-12;
const QUANTITY: Record<string, { label: string; component: string }> = {
  displacement_x: { label: "DISPLACEMENT", component: "Ux" },
  displacement_y: { label: "DISPLACEMENT", component: "Uy" },
  displacement_z: { label: "DISPLACEMENT", component: "Uz" },
};

const index = readUserFile(H5, USER, MODEL);
if (!index.combinations.ok) throw new Error(`the app-written /combinations is refused: ${index.combinations.reason}`);
const C = index.combinations.value;
const TABLE = new CombinationTable(C);

test("Python reads the generations the app wrote: the revision, @current, every g<N> and every row, none refused", () => {
  assert.equal(index.revision, GOLDEN.revision);
  assert.equal(index.writer, GOLDEN.writer);
  assert.equal(C.generation, GOLDEN.current);
  assert.deepEqual(C.generations, GOLDEN.generations);
  assert.deepEqual(GOLDEN.refused, []);
  assert.deepEqual(C.refused, []);
  const app = C.rows.map((r) => ({ name: r.name, rule: r.rule, envelope: r.envelope, fraction: r.fraction, damping: r.damping, terms: r.terms.map((t) => [t.kind, t.source, t.step, t.mode, t.factor]) }));
  assert.deepEqual(app, GOLDEN.rows);
  // The three the app added are the last three, after the three Python wrote in g0.
  assert.deepEqual(GOLDEN.rows.slice(3).map((r) => r.name), ["SRSS Ex modes 1-2", "CQC Ex modes 1-3 3%", "1.3 SRSS Ex"]);
});

const reads = new Map<string, CaseRead>();
for (const name of new Set(C.rows.flatMap((r) => spectrumCasesOf(TABLE, r.name)))) {
  reads.set(name, readCaseValues(H5, USER, readUserFile(H5, USER, null).spectra, name, "nodes", "DISPLACEMENT"));
}

for (const [name, byComponent] of Object.entries(GOLDEN.values)) {
  for (const [component, want] of Object.entries(byComponent)) {
    test(`${name}, ${component}: VU4's evaluator gives Python's value on the app-written file, to 1e-12 relative`, () => {
      const got = evaluate(TABLE, name, spectrumSource(reads, QUANTITY[component]!, { stale: false, reason: "" }, false));
      assert.deepEqual(got.field.keys.map((k) => k[0]), want.ids);
      assert.equal(got.signed, want.signed);
      const v = got.field.values;
      assert.equal(v.length, want.values.length);
      want.values.forEach((w, i) => assert.ok(Math.abs(v[i]! - w) <= RTOL * Math.abs(w), `${name} ${component} value ${i}: ${v[i]} vs ${w}`));
    });
  }
}

test("a row of stage terms, which neither side can evaluate without a results file, is refused by both, never given a value", () => {
  // Python's /spectra provider refuses a stage address (`bad-term`); the app's /spectra source refuses it too (`unresolved`: it reads no results file).
  assert.deepEqual(GOLDEN.refusals, { "1.2D + 1.6L": "bad-term" });
  assert.throws(() => evaluate(TABLE, "1.2D + 1.6L", spectrumSource(reads, QUANTITY["displacement_x"]!, { stale: false, reason: "" }, false)), (e: unknown) => e instanceof CombinationRefused && e.reason === "unresolved");
});

test("an app-written g<N> has the layout of Python's g0: every column's dtype, shape rank and string encoding", () => {
  const f = new (h5wasm as unknown as { File: new (p: string, m: "r") => { get(p: string): unknown; close(): void } }).File(USER, "r");
  try {
    const meta = (p: string) => {
      const d = f.get(p) as { metadata: Record<string, unknown>; shape: number[] };
      const m = d.metadata;
      return { type: m["type"], size: m["size"], signed: m["signed"], vlen: m["vlen"], cset: m["cset"], littleEndian: m["littleEndian"], rank: d.shape.length };
    };
    for (const col of ["name", "rule", "envelope", "fraction", "damping", "term_offsets", "terms/kind", "terms/source", "terms/step", "terms/mode", "terms/factor"]) {
      assert.deepEqual(meta(`combinations/g3/${col}`), meta(`combinations/g0/${col}`), `g3/${col} is laid out as Python's g0/${col}`);
    }
  } finally {
    f.close();
  }
});
