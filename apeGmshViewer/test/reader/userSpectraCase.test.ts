// VU7b-1 (#2740, ADR 0130 D4, D5): a `/spectra` case's CQC, through VU4's evaluator,
// as a nodal contour column.
//
// The oracle is Python. test/fixtures/vu7b/make_sanramon_case.py writes a SYNTHETIC
// nodal case on San Ramon (its mode shapes and eigenvalues are the .ladruno's, its
// participation factors are seeded random numbers) with apeGmsh's own writer, and
// records what `Results.combinations().evaluate(name, component, level="nodes")`
// answers for four rows: a whole-case `cqc` at the row's damping and at a factor of
// -1.5, a whole-case `srss`, and a `cqc` over four of its increments. Modes 1 and 2
// share one `sa_period`, so a whole case sums them first. The app reads the same
// file with reader/userFile.ts, reads each case's DATA as main's `user:readCase`
// does (reader/userSpectra.ts `readCaseValues`), and evaluates with
// combination/evaluate.ts through `spectrumSource`. Equal means the same node ids in
// the same order and every value to 1e-12 relative.
//
// The app names a column as the file spells it (`DISPLACEMENT/Ux`, ADR 0122 D7);
// Python names it `displacement_x`. The pairing below is this test's, never the app's.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { CombinationRefused, CombinationTable, deformedShape, evaluate, NotADeformedShape } from "../../src/combination/evaluate.ts";
import { userCombinationColumn, type Bridge } from "../../src/effects.ts";
import { readUserFile, type UserH5 } from "../../src/reader/userFile.ts";
import { readCaseValues, spectrumSource, type CaseRead, type SpectraZone } from "../../src/reader/userSpectra.ts";
import type { Combination } from "../../src/reader/userRules.ts";
import { columnOf, combinedColumn, vertexValues } from "../../src/state/contour.ts";
import { spectrumCasesOf } from "../../src/state/selectors.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as UserH5;
const USER = fileURLToPath(new URL("../fixtures/vu7b/sanramon_case.user.h5", import.meta.url));
const GOLDEN = JSON.parse(readFileSync(fileURLToPath(new URL("../fixtures/vu7b/sanramon_case.golden.json", import.meta.url)), "utf8")) as {
  case: string;
  nodes: number;
  modes: number;
  combinations: Record<string, Record<string, { ids: number[]; values: number[]; signed: boolean }>>;
};
const RTOL = 1e-12;
const CASE = GOLDEN.case;
/** Python's canonical name -> the file's spelling (this test's pairing only). */
const QUANTITY: Record<string, { label: string; component: string }> = {
  displacement_x: { label: "DISPLACEMENT", component: "Ux" },
  displacement_y: { label: "DISPLACEMENT", component: "Uy" },
  displacement_z: { label: "DISPLACEMENT", component: "Uz" },
};
const CURRENT = { stale: false, reason: "" };

const index = readUserFile(H5, USER, null);
const spectra = index.spectra as SpectraZone;
if (!index.combinations.ok) throw new Error(`the fixture's /combinations is refused: ${index.combinations.reason}`);
const TABLE = new CombinationTable(index.combinations.value);

/** main's `user:readCase`, without Electron: the same two calls on the same file. */
const bridge: Pick<Bridge, "readUserCase"> = {
  readUserCase: async (req) => readCaseValues(H5, req.path, readUserFile(H5, req.path, null).spectra, req.case, req.target, req.label),
};

function close(got: Float64Array, want: readonly number[], what: string): void {
  assert.equal(got.length, want.length, `${what}: length`);
  want.forEach((w, i) => assert.ok(Math.abs(got[i]! - w) <= RTOL * Math.abs(w), `${what}: value ${i} is ${got[i]}, want ${w} to ${RTOL} relative`));
}

test("the fixture is what its generator says: four rows over one synthetic nodal case, every node of it current", () => {
  assert.equal(index.combinations.ok && index.combinations.value.refused.length, 0);
  assert.deepEqual(TABLE.rows.map((r) => r.name).sort(), Object.keys(GOLDEN.combinations).sort());
  assert.ok(spectra !== null && spectra.ok);
  const c = spectra.value.cases.find((x) => x.name === CASE)!;
  assert.equal(c.modes.length, GOLDEN.modes);
  assert.equal(c.results[0]!.ids!.length, GOLDEN.nodes);
  // modes 1 and 2 share one sa_period > 0: a whole case groups them
  assert.ok(c.modes[0]!.saPeriod > 0 && c.modes[0]!.saPeriod === c.modes[1]!.saPeriod);
});

for (const [name, byComponent] of Object.entries(GOLDEN.combinations)) {
  for (const [component, want] of Object.entries(byComponent)) {
    test(`${name}, ${component}: the contour column equals Python's Results to 1e-12 relative`, async () => {
      const read = await userCombinationColumn(bridge, { path: USER, table: TABLE, name, quantity: QUANTITY[component]!, staleness: CURRENT });
      assert.ok(read.ok, read.ok ? "" : read.refusal.message);
      const col = read.column;
      assert.equal(col.label, `DISPLACEMENT/${QUANTITY[component]!.component}`);
      assert.deepEqual([...col.ids], want.ids);
      close(col.values, want.values, `${name} ${component}`);
      assert.equal(col.signed, want.signed);
      assert.equal(col.signed, false, "a modal rule is unsigned");
      const finite = want.values.filter(Number.isFinite);
      assert.deepEqual(col.range, { min: Math.min(...finite), max: Math.max(...finite) });
    });
  }
}

test("an unsigned CQC is a contour only: the deformed shape refuses it (ADR 0130 D4, A3)", async () => {
  const reads = new Map<string, CaseRead>([[CASE, await bridge.readUserCase!({ path: USER, case: CASE, target: "nodes", label: "DISPLACEMENT" })]]);
  const c = evaluate(TABLE, "CQC Ex", spectrumSource(reads, QUANTITY["displacement_x"]!, CURRENT, false));
  assert.throws(() => deformedShape(c), NotADeformedShape);
});

test("the column joins the vertices by node id: a vertex takes its node's value, an unknown node is no data", async () => {
  const read = await userCombinationColumn(bridge, { path: USER, table: TABLE, name: "CQC Ex", quantity: QUANTITY["displacement_y"]!, staleness: CURRENT });
  assert.ok(read.ok);
  const want = GOLDEN.combinations["CQC Ex"]!["displacement_y"]!;
  const vertexNodes = Float64Array.from([want.ids[3]!, want.ids[0]!, -7, want.ids[3]!]);
  const v = vertexValues(vertexNodes, columnOf(read.column.ids), [read.column.values], false);
  close(Float64Array.from([v[0]!, v[1]!, v[3]!]), [want.values[3]!, want.values[0]!, want.values[3]!], "vertices");
  assert.ok(Number.isNaN(v[2]!));
});

const refusal = async (req: Partial<Parameters<typeof userCombinationColumn>[1]>, b: Pick<Bridge, "readUserCase"> = bridge) => {
  const read = await userCombinationColumn(b, { path: USER, table: TABLE, name: "CQC Ex", quantity: QUANTITY["displacement_x"]!, staleness: CURRENT, ...req });
  assert.ok(!read.ok, "it must refuse");
  return read.refusal;
};

test("a stale /spectra refuses stale, and reads with allowStale (A6)", async () => {
  const stale = { stale: true, reason: "fem_hash differs" };
  const r = await refusal({ staleness: stale });
  assert.equal(r.reason, "stale");
  assert.match(r.detail, /fem_hash differs/);
  const read = await userCombinationColumn(bridge, { path: USER, table: TABLE, name: "CQC Ex", quantity: QUANTITY["displacement_x"]!, staleness: stale, allowStale: true });
  assert.ok(read.ok);
});

test("a quantity the case does not hold refuses unresolved, and so does a result it does not hold", async () => {
  assert.equal((await refusal({ quantity: { label: "DISPLACEMENT", component: "Rx" } })).reason, "unresolved");
  const r = await refusal({ quantity: { label: "ROTATION", component: "Rx" } });
  assert.equal(r.reason, "unresolved");
  assert.match(r.detail, /holds no nodes result 'ROTATION'/);
});

test("a combination the table does not hold refuses unresolved; one over a missing case too", async () => {
  assert.equal((await refusal({ name: "nope" })).reason, "unresolved");
  const t = CombinationTable.resolve([row("ghost", "cqc", [{ kind: "spectrum", source: "no such case", step: -1, mode: 0, factor: 1 }], 0.05)]);
  const r = await refusal({ table: t, name: "ghost" });
  assert.equal(r.reason, "unresolved");
  assert.match(r.detail, /holds no case 'no such case'/);
});

test("a stage term is read from the results file, which this source does not read: it refuses, never sums", async () => {
  const t = CombinationTable.resolve([
    ...index.combinations.ok ? index.combinations.value.rows : [],
    row("D + E", "linear", [{ kind: "stage", source: "stage_0", step: -1, mode: 0, factor: 1 }, { kind: "spectrum", source: CASE, step: -1, mode: 3, factor: 1 }], null),
  ]);
  const r = await refusal({ table: t, name: "D + E" });
  assert.equal(r.reason, "unresolved");
  assert.match(r.detail, /results file/);
});

test("a refused or absent /spectra, and a refused case, answer the evaluator's reason", () => {
  assert.deepEqual(readCaseValues(H5, USER, null, CASE, "nodes", "DISPLACEMENT").ok, false);
  assert.equal((readCaseValues(H5, USER, null, CASE, "nodes", "DISPLACEMENT") as { reason: string }).reason, "unresolved");
  assert.equal((readCaseValues(H5, USER, { ok: false, reason: "refused: bad" }, CASE, "nodes", "DISPLACEMENT") as { reason: string }).reason, "refused");
  const refused: SpectraZone = { ok: true, value: { cases: [], refused: [{ name: CASE, reason: "bad-param", detail: "x", origin: "user", row: 0 }] } };
  const r = readCaseValues(H5, USER, refused, CASE, "nodes", "DISPLACEMENT");
  assert.ok(!r.ok && r.reason === "refused" && /is refused: bad-param/.test(r.detail));
});

test("a case whose DATA no longer has the shape the index read refuses, never reads a wrong slab", () => {
  assert.ok(spectra !== null && spectra.ok);
  const c = spectra.value.cases[0]!;
  const moved = { ...c, results: c.results.map((q) => ({ ...q, shape: [q.shape[0], q.shape[1] + 1, q.shape[2]] as [number, number, number] })) };
  const r = readCaseValues(H5, USER, { ok: true, value: { cases: [moved], refused: [] } }, CASE, "nodes", "DISPLACEMENT");
  assert.ok(!r.ok && r.reason === "refused" && /has shape/.test(r.detail), JSON.stringify(r));
});

test("spectrumCasesOf: the cases a combination reaches, through the combinations it names", () => {
  const t = CombinationTable.resolve([
    row("A", "cqc", [{ kind: "spectrum", source: "Ex", step: -1, mode: 0, factor: 1 }], 0.05),
    row("B", "cqc", [{ kind: "spectrum", source: "Ey", step: -1, mode: 0, factor: 1 }], 0.05),
    row("AB", "directional", [{ kind: "combination", source: "A", step: -1, mode: 0, factor: 1 }, { kind: "combination", source: "B", step: -1, mode: 0, factor: 1 }], null, 0.3),
  ]);
  assert.deepEqual(spectrumCasesOf(t, "AB").sort(), ["Ex", "Ey"]);
  assert.deepEqual(spectrumCasesOf(t, "A"), ["Ex"]);
  assert.throws(() => spectrumCasesOf(t, "C"), CombinationRefused);
});

test("a bridge without readUserCase throws, rather than answering a refusal", async () => {
  await assert.rejects(userCombinationColumn({}, { path: USER, table: TABLE, name: "CQC Ex", quantity: QUANTITY["displacement_x"]!, staleness: CURRENT }), /no readUserCase/);
});

test("combinedColumn refuses a field that is not one nodal label", () => {
  const two = { name: "x", signed: false, badges: [], field: { keyNames: ["id"], keys: [[1]], components: ["a", "b"], values: Float64Array.from([1, 2]) } };
  assert.throws(() => combinedColumn(two), /2 labels/);
  const gauss = { name: "x", signed: false, badges: [], field: { keyNames: ["id", "sub"], keys: [[1, 0]], components: ["a"], values: Float64Array.from([1]) } };
  assert.throws(() => combinedColumn(gauss), /not by node id/);
});

function row(name: string, rule: string, terms: Combination["terms"], damping: number | null, fraction: number | null = null): Combination {
  return { name, rule, terms, envelope: "", fraction, damping };
}
