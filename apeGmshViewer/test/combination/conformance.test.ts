// combination/evaluate.ts against the ADR 0130 D4 conformance fixture (VU4).
//
// The oracle is tests/fixtures/user_results/conformance.json: its goldens are
// written by a reference in make_conformance.py that shares no code with
// either evaluator, and VU2's Python evaluator meets the same goldens
// (tests/results/user/test_user_conformance.py). This file reads the JSON only;
// it imports nothing of apeGmsh (app wall W1). Equal means: the same sign, the
// same badges, the same key names, keys and components, values to the file's
// `rtol` (1e-12) with NaN equal to NaN, and for a refusal the same reason.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { CombinationRefused, CombinationTable, deformedShape, evaluate, EVAL_REASONS, isSigned, makeCase, NotADeformedShape, readState, SUPERPOSITION_BADGE, type Case, type Combined, type Source } from "../../src/combination/evaluate.ts";
import { makeField, type Field, type Key } from "../../src/combination/field.ts";
import { userResultSource } from "../../src/combination/userSource.ts";
import type { Combination, StateAddress } from "../../src/reader/userRules.ts";

const FIXTURE = fileURLToPath(new URL("../../../tests/fixtures/user_results/conformance.json", import.meta.url));

type Cell = number | null;
interface JsonField {
  key_names: string[];
  keys: Key[][];
  components: string[];
  values: Cell[][];
}
interface JsonTerm {
  kind: string;
  source: string;
  step: number;
  mode: number;
  factor: number;
}
interface JsonRow {
  name: string;
  rule: string;
  terms: JsonTerm[];
  envelope: string;
  fraction: number | null;
  damping: number | null;
}
interface JsonCase {
  mode: number[];
  lambda: number[];
  sa_period: number[];
  damping: Cell[];
  increments: JsonField[];
}
type Golden = { refused: string } | (JsonField & { signed: boolean; badges: string[] });
interface Doc {
  rtol: number;
  states: Record<string, JsonField>;
  other_states: Record<string, JsonField>;
  eigenvalues: Record<string, number>;
  cases: Record<string, JsonCase>;
  combinations: JsonRow[];
  model_combinations: JsonRow[];
  quantities: Record<string, { linear: boolean; scale: number; stored: Record<string, JsonField> }>;
  goldens: Record<string, Record<string, Golden>>;
  lone: Record<string, Record<string, Golden>>;
}
const DOC = JSON.parse(readFileSync(FIXTURE, "utf8")) as Doc;
const RTOL = DOC.rtol;

const nan = (v: Cell) => (v === null ? NaN : v);
const field = (f: JsonField, scale = 1): Field => makeField(f.key_names, f.keys, f.components, f.values.flat().map((v) => scale * nan(v)));
const rows = (rs: JsonRow[]): Combination[] => rs.map((c) => ({ ...c, terms: c.terms.map((t) => ({ ...t })) }));
const table = () => CombinationTable.resolve(rows(DOC.combinations), rows(DOC.model_combinations));
const addressText = (a: StateAddress) => `${a.kind}|${a.source}|${a.step}|${a.mode}`;
function address(text: string): StateAddress {
  const [kind, source, step, mode] = text.split("|") as [string, string, string, string];
  return { kind, source, step: Number(step), mode: Number(mode) };
}
const caseOf = (c: JsonCase): Case =>
  makeCase({ mode: c.mode, lam: c.lambda, saPeriod: c.sa_period, damping: c.damping.map(nan), increments: c.increments.map((f) => field(f)) });

/** The recorded quantity of the fixture: every state, case and eigenvalue (Python's `FixtureSource`). */
class FixtureSource implements Source {
  readonly linear = true;
  stored(): Field | null {
    return null;
  }
  field(a: StateAddress): Field {
    const key = addressText(a);
    const found = DOC.states[key] ?? DOC.other_states[key];
    if (found === undefined) throw new CombinationRefused("", "unresolved", `no state ${key}`);
    return field(found);
  }
  eigenvalue(a: StateAddress): number {
    const lam = DOC.eigenvalues[addressText(a)];
    if (lam === undefined) throw new CombinationRefused("", "unresolved", `no eigenvalue for ${addressText(a)}`);
    return lam;
  }
  case(name: string): Case {
    const c = DOC.cases[name];
    if (c === undefined) throw new CombinationRefused("", "unresolved", `no case ${name}`);
    return caseOf(c);
  }
}

/**
 * A nodal user result holding every state of the fixture, scaled, plus its
 * stored rows: Python's `_user_result`, dense on nodes 1-3 over ux and uy, so a
 * state that lacks a node or a label (W) holds NaN there.
 */
function userSource(quantity: string): Source {
  const q = DOC.quantities[quantity]!;
  const states: StateAddress[] = [];
  const fields: Field[] = [];
  const dense = (f: JsonField, scale: number): Field => {
    const out = new Array<number>(6).fill(NaN);
    f.keys.forEach((k, r) => f.components.forEach((label, c) => (out[(Number(k[0]) - 1) * 2 + ["ux", "uy"].indexOf(label)] = scale * nan(f.values[r]![c]!))));
    return makeField(["id"], [[1], [2], [3]], ["ux", "uy"], out);
  };
  for (const [a, f] of Object.entries(DOC.states)) {
    states.push(address(a));
    fields.push(dense(f, q.scale));
  }
  for (const [combo, f] of Object.entries(q.stored)) {
    states.push({ kind: "combination", source: combo, step: -1, mode: 0 });
    fields.push(dense(f, 1));
  }
  const recorded = { check: () => undefined, eigenvalue: (a: StateAddress) => new FixtureSource().eigenvalue(a) };
  return userResultSource({ name: quantity, linear: q.linear, states, fields }, recorded, quantity);
}

const sourceOf = (quantity: string): Source => (quantity === "recorded" ? new FixtureSource() : userSource(quantity));

function assertGolden(got: Combined, want: Exclude<Golden, { refused: string }>, what: string): void {
  assert.equal(got.signed, want.signed, `${what}: signed`);
  assert.deepEqual([...got.badges], want.badges, `${what}: badges`);
  assert.deepEqual([...got.field.components], want.components, `${what}: components`);
  assert.deepEqual([...got.field.keyNames], want.key_names, `${what}: key names`);
  assert.deepEqual(got.field.keys.map((k) => [...k]), want.keys, `${what}: keys`);
  const expected = want.values.flat().map(nan);
  assert.equal(got.field.values.length, expected.length, `${what}: value count`);
  expected.forEach((w, i) => {
    const g = got.field.values[i]!;
    if (Number.isNaN(w)) assert.ok(Number.isNaN(g), `${what}: value ${i} is ${g}, want NaN`);
    else assert.ok(Math.abs(g - w) <= RTOL * Math.abs(w), `${what}: value ${i} is ${g}, want ${w} to ${RTOL} relative`);
  });
}

function check(want: Golden, run: () => Combined, what: string): void {
  if ("refused" in want) {
    assert.throws(run, (err: unknown) => err instanceof CombinationRefused && err.reason === want.refused, `${what}: refused ${want.refused}`);
    return;
  }
  assertGolden(run(), want, what);
}

const GOLDENS = Object.entries(DOC.goldens).flatMap(([q, g]) => Object.entries(g).map(([n, want]) => [q, n, want] as const));
const LONE = Object.entries(DOC.lone).flatMap(([q, g]) => Object.entries(g).map(([a, want]) => [q, a, want] as const));

test("the fixture holds goldens for every quantity, both signs and refusals", () => {
  assert.deepEqual(Object.keys(DOC.goldens).sort(), ["m_xx", "recorded", "wood_armer_top"]);
  assert.ok(GOLDENS.length >= 50 && LONE.length >= 10, `${GOLDENS.length} goldens, ${LONE.length} lone`);
  const signed = new Set(GOLDENS.flatMap(([, , w]) => ("signed" in w ? [w.signed] : [])));
  assert.deepEqual([...signed].sort(), [false, true]);
  const reasons = new Set(GOLDENS.flatMap(([, , w]) => ("refused" in w ? [w.refused] : [])));
  for (const r of reasons) assert.ok((EVAL_REASONS as readonly string[]).includes(r), r);
  assert.deepEqual([...reasons].sort(), ["bad-param", "not-linear", "refused", "unresolved"]);
});

for (const [quantity, name, want] of GOLDENS) {
  test(`golden ${quantity} / ${JSON.stringify(name)}`, () => {
    check(want, () => evaluate(table(), name, sourceOf(quantity)), `${quantity} ${name}`);
  });
}

for (const [quantity, a, want] of LONE) {
  test(`lone ${quantity} / ${a}`, () => {
    check(want, () => readState(address(a), sourceOf(quantity)), `${quantity} ${a}`);
  });
}

test("sign is derived from the rule and the terms, as the goldens record it", () => {
  const t = table();
  for (const [quantity, name, want] of GOLDENS) {
    if (quantity === "recorded" && "signed" in want) assert.equal(isSigned(t, name), want.signed, name);
  }
  assert.equal(isSigned(t, "plus"), false, "linear over an unsigned term is unsigned");
});

test("an unsigned result is refused as a deformed shape; a signed one gives its field", () => {
  let unsigned = 0;
  for (const [quantity, name, want] of GOLDENS) {
    if ("refused" in want) continue;
    const got = evaluate(table(), name, sourceOf(quantity));
    if (got.signed) assert.equal(deformedShape(got), got.field, name);
    else {
      unsigned += 1;
      assert.throws(() => deformedShape(got), NotADeformedShape, name);
    }
  }
  assert.ok(unsigned >= 20, `${unsigned} unsigned goldens`);
  // A stored row keeps its descriptor's sign: an unsigned envelope stays a contour.
  const env = evaluate(table(), "env_unit", sourceOf("wood_armer_top"));
  assert.throws(() => deformedShape(env), NotADeformedShape);
});

test("the superposition badge is on a linear over a stage only", () => {
  const src = new FixtureSource();
  assert.deepEqual(evaluate(table(), "ULS", src).badges, [SUPERPOSITION_BADGE]);
  assert.deepEqual(evaluate(table(), "k_linear", src).badges, []);
  assert.deepEqual(evaluate(table(), "twice_ULS", src).badges, []);
});

// ---- hand values: literals, not the golden file ---------------------------------

const value = (name: string) => evaluate(table(), name, new FixtureSource()).field.values;

test("hand values", () => {
  // ULS = 1.2 D + 1.6 L at node 1: 1.2 + 0.8 and 2.4 - 1.6.
  const uls = value("ULS");
  assert.ok(Math.abs(uls[0]! - 2.0) < 1e-15 && Math.abs(uls[1]! - 0.8) < 1e-15);
  // srss of D and -2 L at node 2: sqrt(3² + 4²) and sqrt(4² + 0²).
  assert.deepEqual([...value("srss_list").slice(2, 4)], [5, 4]);
  // The join: W has no node 3 and no uy, so D + W is NaN there.
  const dw = value("D+W");
  assert.equal(dw[0], 11);
  assert.equal(dw[2], 23);
  assert.ok(Number.isNaN(dw[4]!) && [1, 3, 5].every((i) => Number.isNaN(dw[i]!)));
  // E's NaN at node 1, uy: nanmax would read 2.0 there.
  const env = value("env_max");
  assert.ok(Number.isNaN(env[1]!));
  assert.equal(env[0], 1);
});

test("a stored row wins over the rule for a user result", () => {
  assert.deepEqual([...evaluate(table(), "ULS", sourceOf("wood_armer_top")).field.values], [9, 8, 7, 6, 5, 4]);
});

// ---- refusals at the seams ---------------------------------------------------------

test("a state-free user result is never combined", () => {
  assert.throws(
    () => userResultSource({ name: "R", linear: true, states: null, fields: [] }, null, "x"),
    (err: unknown) => err instanceof CombinationRefused && err.reason === "state-free",
  );
});

test("a refused row, and a name no file holds, refuse on evaluation", () => {
  const t = CombinationTable.resolve([{ name: "bad", rule: "median", terms: [{ kind: "stage", source: "D", step: -1, mode: 0, factor: 1 }], envelope: "", fraction: null, damping: null }]);
  assert.throws(() => evaluate(t, "bad", new FixtureSource()), (err: unknown) => err instanceof CombinationRefused && err.reason === "refused");
  assert.throws(() => evaluate(t, "nobody", new FixtureSource()), (err: unknown) => err instanceof CombinationRefused && err.reason === "unresolved");
});

test("terms on different targets are refused", () => {
  class Mixed extends FixtureSource {
    override field(a: StateAddress): Field {
      return a.source === "L" ? makeField(["id", "sub"], [[1, 0]], ["ux"], [1]) : super.field(a);
    }
  }
  assert.throws(() => evaluate(table(), "ULS", new Mixed()), (err: unknown) => err instanceof CombinationRefused && err.reason === "bad-term");
});

test("entity terms join on kind and row, text by code point", () => {
  class Entities extends FixtureSource {
    override field(a: StateAddress): Field {
      if (a.source === "D") return makeField(["kind", "row"], [["cut", 0]], ["V"], [2]);
      return makeField(["kind", "row"], [["cut", 0], ["a_long_storey_kind", 3]], ["V"], [1, 5]);
    }
  }
  const got = evaluate(table(), "ULS", new Entities()).field;
  assert.deepEqual(got.keys, [["a_long_storey_kind", 3], ["cut", 0]]);
  assert.ok(Number.isNaN(got.values[0]!));
  assert.ok(Math.abs(got.values[1]! - (1.2 * 2 + 1.6 * 1)) < 1e-15);
});

test("an unknown evaluation reason is a bug, not a refusal", () => {
  assert.throws(() => new CombinationRefused("x", "nope" as never, ""), RangeError);
});

test("a malformed field or case fails loudly at the seam", () => {
  assert.throws(() => makeField(["id"], [[1], [1]], ["ux"], [1, 2]), /repeats the row/);
  assert.throws(() => makeField(["id"], [[1]], ["ux", "ux"], [1, 2]), /repeats a label/);
  assert.throws(() => makeField(["id"], [[1]], ["ux"], [1, 2]), /values/);
  assert.throws(() => makeCase({ mode: [1], lam: [], saPeriod: [0], damping: [0.05], increments: [makeField(["id"], [[1]], ["ux"], [1])] }), RangeError);
});
