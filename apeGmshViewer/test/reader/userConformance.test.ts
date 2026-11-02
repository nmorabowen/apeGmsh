// reader/userRules.ts `resolve` against the ADR 0130 D4 conformance fixture
// (review of #2729, finding 3).
//
// The oracle is tests/fixtures/user_results/conformance.json: VU2's Python
// writer (make_conformance.py beside it) records the rows of `combinations`
// (the .user.h5 table) and `model_combinations` (model.h5's), and in `table`
// every row Python's `resolve` refuses, with its reason. This file reads the
// JSON only; it imports nothing of apeGmsh (app wall W1).

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { resolve, type Combination } from "../../src/reader/userRules.ts";

const FIXTURE = fileURLToPath(new URL("../../../tests/fixtures/user_results/conformance.json", import.meta.url));

interface JsonTerm {
  kind: string;
  source: string;
  step: number;
  mode: number;
  factor: number | null;
}
interface JsonRow {
  name: string;
  rule: string;
  terms: JsonTerm[];
  envelope?: string | null;
  fraction?: number | null;
  damping?: number | null;
}
interface Verdict {
  origin: "user" | "model";
  row: number;
  name: string;
  reason: string;
}
const doc = JSON.parse(readFileSync(FIXTURE, "utf8")) as { combinations: JsonRow[]; model_combinations: JsonRow[]; table: Verdict[] };

/** The fixture's JSON row as the reader's `Combination` ("null is NaN"). */
const rows = (rs: JsonRow[]): Combination[] =>
  rs.map((c) => ({
    name: c.name,
    rule: c.rule,
    envelope: c.envelope ?? "",
    fraction: c.fraction ?? null,
    damping: c.damping ?? null,
    terms: c.terms.map((t) => ({ kind: t.kind, source: t.source, step: t.step, mode: t.mode, factor: t.factor ?? NaN })),
  }));

test("resolve on the conformance rows refuses exactly the rows of its table goldens, with their reasons", () => {
  assert.ok(doc.table.length > 0, "the fixture holds refusals to compare");
  const key = (v: Verdict) => [v.origin, v.row, v.name, v.reason].join("|");
  const got = resolve(rows(doc.combinations), rows(doc.model_combinations)).refused.map(key).sort();
  assert.deepEqual(got, doc.table.map(key).sort());
});

test("the rows the table does not refuse stand, in their file's order", () => {
  const out = resolve(rows(doc.combinations), rows(doc.model_combinations));
  const refused = (origin: string) => new Set(doc.table.filter((v) => v.origin === origin).map((v) => v.row));
  assert.deepEqual(out.rows.map((r) => r.name), doc.combinations.filter((_, i) => !refused("user").has(i)).map((r) => r.name));
  assert.deepEqual(out.modelRows.map((r) => r.name), doc.model_combinations.filter((_, i) => !refused("model").has(i)).map((r) => r.name));
});
