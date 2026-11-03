// Evaluate a combination per quantity, on read (ADR 0130 D4, Amendment 1 A3, A5).
//
// The app's twin of src/apeGmsh/results/user/_evaluate.py (VU2): the same rules,
// the same order of checks and the same refusal reasons, so both evaluators
// agree on tests/fixtures/user_results/conformance.json to 1e-12 relative. The
// rules table of architecture/h5-schema.md (`<stem>.user.h5`, "Rules"), with
// `E = factor · r` for each term and `p` the row's fraction:
//
//   linear       list                          ΣE
//   srss         list, or one whole case       √ΣE²
//   abs_sum      list, or one whole case       Σ|E|
//   cqc          one whole case, or the modes  √max(0, EᵀρE)
//                of one set
//   directional  list                          max_a(|E_a| + p Σ_{b≠a} |E_b|)
//   envelope     list                          max, min or absmax
//
// A whole case first sums the modes that share a `sa_period` (0 is the mode's
// own period) into the slot of the group's first mode, at ω = 2π/T and with
// that mode's damping. ρ is Der Kiureghian's with ω = √λ; λ ≤ 0 is refused. A
// finite row damping overrides the case's column. NaN propagates through every
// rule (`max` is numpy's `max`, never `nanmax`).
//
// Only `linear` can be signed, and only if all its terms are; sign is derived
// from the descriptors and never stored. Only a signed result draws as a
// deformed shape: `deformedShape` refuses any other (A3). A `linear` with a
// `stage` term carries the superposition badge (D4).
//
// A user result (A5): a stored row for the combination wins. Otherwise
// `envelope` reads the terms' stored rows, and every other rule, and any term
// whose factor is not 1, needs `@linear=1`. A rule that cannot be applied
// refuses; it never sums.
//
// The evaluator reads values only through a `Source`, so one code path serves a
// set's recorded results, its `.user.h5` user results and the conformance fixture.

import { addressProblem, resolve, type Combination, type Refusal, type StateAddress, type Term } from "../reader/userRules.ts";
import { align, scaled, targetProblem, type Field } from "./field.ts";

/** Why an evaluation refuses (Python's `EVAL_REASONS`). A row the table refused reads `refused`. */
export const EVAL_REASONS = ["refused", "unresolved", "bad-param", "bad-term", "not-linear", "state-free", "stale", "foreign-handle"] as const;
export type EvalReason = (typeof EVAL_REASONS)[number];
export const SUPERPOSITION_BADGE = "superposition assumes linearity";

/** One combination cannot be evaluated for one quantity, and why. */
export class CombinationRefused extends Error {
  readonly combination: string;
  readonly reason: EvalReason;
  readonly detail: string;
  constructor(combination: string, reason: EvalReason, detail: string) {
    if (!(EVAL_REASONS as readonly string[]).includes(reason)) throw new RangeError(`unknown evaluation reason ${JSON.stringify(reason)}`);
    super(`combination ${JSON.stringify(combination)} refused (${reason}): ${detail}`);
    this.combination = combination;
    this.reason = reason;
    this.detail = detail;
  }
}

/**
 * One whole spectrum case for one quantity: per mode, the signed increment.
 * Row `i` of every column, and `increments[i]`, is one mode of the case
 * (`/spectra/s<k>/modes`). `saPeriod` 0 reads the mode at its own period; NaN
 * in `damping` means the case gives none for that mode.
 */
export interface Case {
  readonly mode: readonly number[];
  readonly lam: readonly number[];
  readonly saPeriod: readonly number[];
  readonly damping: readonly number[];
  readonly increments: readonly Field[];
}

/** A checked case (Python's `Case.__post_init__`): one row per increment, at least one. */
export function makeCase(c: Case): Case {
  const m = c.increments.length;
  if (m === 0) throw new RangeError("a case has at least one mode");
  for (const what of ["mode", "lam", "saPeriod", "damping"] as const) {
    if (c[what].length !== m) throw new RangeError(`Case.${what} has ${c[what].length} rows for ${m} increments`);
  }
  return c;
}

/** A combination's value for one quantity. `signed` is whether it may draw as a deformed shape. */
export interface Combined {
  readonly name: string;
  readonly field: Field;
  readonly signed: boolean;
  readonly badges: readonly string[];
}

/**
 * Where an evaluation reads one quantity's values. `linear` says whether a
 * summing rule may combine it (A5): true for a recorded response, a user
 * result's `@linear` flag. `stored` is a user result's stored row for a
 * combination, else null. `field`, `eigenvalue` and `case` throw
 * `CombinationRefused` for what they cannot read.
 */
export interface Source {
  readonly linear: boolean;
  stored(name: string): Field | null;
  field(address: StateAddress): Field;
  eigenvalue(address: StateAddress): number;
  case(name: string): Case;
}

/** The rows in force across both files, by name, and the first refusal of each refused name. */
export class CombinationTable {
  readonly rows: readonly Combination[];
  readonly modelRows: readonly Combination[];
  readonly refused: readonly Refusal[];
  readonly #rows = new Map<string, Combination>();
  readonly #refusals = new Map<string, Refusal>();

  constructor(table: { rows: readonly Combination[]; modelRows: readonly Combination[]; refused: readonly Refusal[] }) {
    this.rows = table.rows;
    this.modelRows = table.modelRows;
    this.refused = table.refused;
    for (const row of [...table.rows, ...table.modelRows]) if (!this.#rows.has(row.name)) this.#rows.set(row.name, row);
    for (const ref of table.refused) if (!this.#refusals.has(ref.name)) this.#refusals.set(ref.name, ref);
  }

  /** Judge both files' rows (reader/userRules.ts `resolve`) and index what stands. */
  static resolve(user: readonly Combination[], model: readonly Combination[] = []): CombinationTable {
    return new CombinationTable(resolve(user, model));
  }

  refusal(name: string): Refusal | null {
    return this.#refusals.get(name) ?? null;
  }

  /** The row named `name`; a missing one refuses `refused` when the table refused it, else `unresolved`. */
  row(name: string, caller: string): Combination {
    const row = this.#rows.get(name);
    if (row !== undefined) return row;
    const ref = this.refusal(name);
    if (ref !== null) throw new CombinationRefused(caller, "refused", `combination ${JSON.stringify(name)} is refused: ${ref.reason} (${ref.detail})`);
    throw new CombinationRefused(caller, "unresolved", `no combination named ${JSON.stringify(name)}`);
  }
}

// ---------------------------------------------------------------------------
// Sign, from the descriptors alone
// ---------------------------------------------------------------------------

/**
 * `name` and every combination it reaches, each after the ones it names.
 * Iterative, so depth is not bounded by the stack; `leaf` stops the walk at a
 * combination whose terms are not needed (a stored row). The table is acyclic:
 * `resolve` refused every row in or reaching a cycle.
 */
function postOrder(table: CombinationTable, name: string, leaf: (n: string) => boolean = () => false): string[] {
  const order: string[] = [];
  const done = new Set<string>();
  const stack: [string, boolean][] = [[name, false]];
  while (stack.length > 0) {
    const [current, expanded] = stack.pop()!;
    if (done.has(current)) continue;
    if (expanded) {
      done.add(current);
      order.push(current);
      continue;
    }
    const row = table.row(current, name);
    stack.push([current, true]);
    if (!leaf(current)) for (const t of row.terms) if (t.kind === "combination" && !done.has(t.source)) stack.push([t.source, false]);
  }
  return order;
}

function signs(table: CombinationTable, order: readonly string[], memo: Map<string, boolean>): void {
  for (const current of order) {
    if (memo.has(current)) continue;
    const row = table.row(current, current);
    memo.set(current, row.rule === "linear" && row.terms.every((t) => (t.kind === "combination" ? memo.get(t.source)! : true)));
  }
}

/** Whether combination `name` is signed: a `linear` whose terms all are. Only a signed result draws as a deformed shape (A3). */
export function isSigned(table: CombinationTable, name: string): boolean {
  const memo = new Map<string, boolean>();
  signs(table, postOrder(table, name), memo);
  return memo.get(name)!;
}

/** A value that may not draw as a deformed shape: an unsigned combination (A3, D4). */
export class NotADeformedShape extends Error {}

/**
 * The displacement field a deformed shape may draw from `c`. An unsigned
 * result (a modal rule, an envelope, or a `linear` over one) is shown as a
 * contour only: it has lost the signs a shape needs, so it is refused here.
 */
export function deformedShape(c: Combined): Field {
  if (!c.signed) {
    throw new NotADeformedShape(`${JSON.stringify(c.name)} is unsigned, so it is shown as a contour only, never as a deformed shape (ADR 0130 D4, A3)`);
  }
  return c.field;
}

// ---------------------------------------------------------------------------
// The kernels
// ---------------------------------------------------------------------------

/** Der Kiureghian's modal correlation, `r = ω_j/ω_i`, `ρ_ii = 1` (Python's `cqc_rho`). */
export function cqcRho(omega: readonly number[], damping: readonly number[]): number[][] {
  return omega.map((wi, i) =>
    omega.map((wj, j) => {
      if (i === j) return 1;
      const r = wj / wi;
      const xi = damping[i]!;
      const xj = damping[j]!;
      const num = 8 * Math.sqrt(xi * xj) * (xi + r * xj) * r ** 1.5;
      const den = (1 - r * r) ** 2 + 4 * xi * xj * r * (1 + r * r) + 4 * (xi * xi + xj * xj) * r * r;
      return den > 0 ? num / den : 1;
    }),
  );
}

/** One rule over `stack` (k blocks of n·c values), cell by cell. */
function reduceCells(stack: readonly Float64Array[], f: (v: number[]) => number): Float64Array {
  const size = stack[0]!.length;
  const out = new Float64Array(size);
  const v: number[] = new Array<number>(stack.length);
  for (let at = 0; at < size; at++) {
    for (let k = 0; k < stack.length; k++) v[k] = stack[k]![at]!;
    out[at] = f(v);
  }
  return out;
}

const sum = (v: number[]) => v.reduce((a, x) => a + x);
const srss = (v: number[]) => Math.sqrt(v.reduce((a, x) => a + x * x, 0));
const absSum = (v: number[]) => v.reduce((a, x) => a + Math.abs(x), 0);
/** numpy's `max`/`min`: a NaN anywhere is NaN (Math.max and Math.min agree). */
const max = (v: number[]) => Math.max(...v);
const min = (v: number[]) => Math.min(...v);

function cqcCells(stack: readonly Float64Array[], omega: readonly number[], damping: readonly number[]): Float64Array {
  const rho = cqcRho(omega, damping);
  return reduceCells(stack, (v) => {
    let total = 0;
    for (let i = 0; i < v.length; i++) for (let j = 0; j < v.length; j++) total += v[i]! * rho[i]![j]! * v[j]!;
    return Math.sqrt(Math.max(total, 0)); // Math.max keeps a NaN, as np.maximum
  });
}

function checkLambda(lam: readonly number[], name: string, what: string): number[] {
  if (lam.some((l) => !Number.isFinite(l) || l <= 0)) throw new CombinationRefused(name, "bad-param", `${what} has an eigenvalue <= 0 or not finite: ${JSON.stringify(lam)}`);
  return lam.map(Math.sqrt);
}

function checkDamping(xi: readonly number[], name: string, what: string): number[] {
  if (xi.some((x) => !Number.isFinite(x) || x < 0 || x >= 1)) {
    throw new CombinationRefused(name, "bad-param", `${what} has a damping ratio outside [0, 1): ${JSON.stringify(xi.map((x) => (Number.isNaN(x) ? "NaN" : x)))}`);
  }
  return [...xi];
}

/**
 * Each mode's slot, and the first mode of each slot (A3, "a whole case"):
 * modes sharing one `sa_period > 0` are one term, in the slot of the first of
 * them; a mode at its own period (`0`) is a term alone.
 */
function saGroups(saPeriod: readonly number[]): { slot: number[]; keep: number[] } {
  const slot: number[] = [];
  const keep: number[] = [];
  const group = new Map<number, number>();
  saPeriod.forEach((period, a) => {
    if (!Number.isFinite(period) || period < 0) throw new RangeError(`sa_period ${period} is not 0 or a period > 0`);
    const hit = period > 0 ? group.get(period) : undefined;
    if (hit !== undefined) {
      slot.push(hit);
      return;
    }
    slot.push(keep.length);
    if (period > 0) group.set(period, keep.length);
    keep.push(a);
  });
  return { slot, keep };
}

function caseRow(c: Case, mode: number, name: string): number {
  const hits = c.mode.flatMap((m, i) => (m === mode ? [i] : []));
  if (hits.length !== 1) throw new CombinationRefused(name, "unresolved", `the case has ${hits.length} rows for mode ${mode}`);
  return hits[0]!;
}

const addressKey = (a: StateAddress) => JSON.stringify([a.kind, a.source, a.step, a.mode]);

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

/** One evaluation: the table, the source, and what was read already. */
class Run {
  readonly memo = new Map<string, Combined>();
  readonly signs = new Map<string, boolean>();
  readonly fields = new Map<string, Field>();
  readonly table: CombinationTable;
  readonly source: Source;

  constructor(table: CombinationTable, source: Source) {
    this.table = table;
    this.source = source;
  }

  field(address: StateAddress): Field {
    const key = addressKey(address);
    let f = this.fields.get(key);
    if (f === undefined) {
      f = this.source.field(address);
      this.fields.set(key, f);
    }
    return f;
  }

  /** Evaluate `name`: the combinations it needs first, in dependency order. */
  combination(name: string): Combined {
    const stored = new Map<string, Field | null>();
    const leaf = (current: string): boolean => {
      if (!stored.has(current)) stored.set(current, this.source.stored(current));
      return stored.get(current) !== null;
    };
    const order = postOrder(this.table, name, leaf);
    // A stored row is a leaf, but its sign still follows its descriptor.
    signs(this.table, postOrder(this.table, name), this.signs);
    if (!this.source.linear) {
      // A5 over every combination the value needs, before any value is read,
      // so the refusal does not depend on the order the terms are read in.
      for (const current of order) if ((stored.get(current) ?? null) === null) this.checkA5(this.table.row(current, current));
    }
    for (const current of order) if (!this.memo.has(current)) this.memo.set(current, this.one(current, stored.get(current) ?? null));
    return this.memo.get(name)!;
  }

  /** A rule or a factor that would sum a quantity that is not linear refuses (A5). */
  checkA5(row: Combination): void {
    if (this.source.linear) return;
    if (row.rule !== "envelope") {
      throw new CombinationRefused(
        row.name,
        "not-linear",
        `${row.rule} sums its terms, and this quantity is not linear (a user result without @linear=1): a sum of it is not its value under the sum. Store a row for this combination, or use envelope (ADR 0130 A5)`,
      );
    }
    for (const t of row.terms) {
      if (t.factor !== 1) {
        throw new CombinationRefused(row.name, "not-linear", `a term of factor ${t.factor} scales this quantity, which is not linear; an envelope of it takes factor-1 terms (ADR 0130 A5)`);
      }
    }
  }

  one(name: string, stored: Field | null): Combined {
    const row = this.table.row(name, name);
    const badges = row.rule === "linear" && row.terms.some((t) => t.kind === "stage") ? [SUPERPOSITION_BADGE] : [];
    const signed = this.signs.get(name)!;
    if (stored !== null) return { name, field: stored, signed, badges };
    this.checkA5(row);
    return { name, field: this.apply(row), signed, badges };
  }

  term(t: Term): Field {
    if (t.kind === "combination") return scaled(this.memo.get(t.source)!.field, t.factor);
    return scaled(this.field(t), t.factor);
  }

  apply(row: Combination): Field {
    const whole = row.terms.filter((t) => t.kind === "spectrum" && t.mode === 0);
    if (whole.length > 0) return this.wholeCase(row, whole[0]!);
    const fields = row.terms.map((t) => this.term(t));
    const problem = targetProblem(fields);
    if (problem !== null) throw new CombinationRefused(row.name, "bad-term", problem);
    const { keyNames, keys, components, stack } = align(fields);
    let values: Float64Array;
    if (row.rule === "linear") values = reduceCells(stack, sum);
    else if (row.rule === "srss") values = reduceCells(stack, srss);
    else if (row.rule === "abs_sum") values = reduceCells(stack, absSum);
    else if (row.rule === "cqc") {
      const { omega, damping } = this.termModes(row);
      values = cqcCells(stack, omega, damping);
    } else if (row.rule === "directional") {
      const p = row.fraction!;
      values = reduceCells(stack, (v) => {
        const e = v.map(Math.abs);
        const total = e.reduce((a, x) => a + x);
        return Math.max(...e.map((x) => x + p * (total - x)));
      });
    } else if (row.rule === "envelope" && row.envelope === "max") values = reduceCells(stack, max);
    else if (row.rule === "envelope" && row.envelope === "min") values = reduceCells(stack, min);
    else if (row.rule === "envelope" && row.envelope === "absmax") values = reduceCells(stack, (v) => max(v.map(Math.abs)));
    else throw new CombinationRefused(row.name, "refused", `unknown rule ${JSON.stringify(row.rule)} (envelope ${JSON.stringify(row.envelope)})`);
    return { keyNames, keys, components, values };
  }

  /** `ω` and `ξ` of `cqc` over the modes of one set, or one case's increments. */
  termModes(row: Combination): { omega: number[]; damping: number[] } {
    const first = row.terms[0]!;
    let lam: number[];
    let xi: number[];
    if (first.kind === "mode") {
      lam = row.terms.map((t) => this.source.eigenvalue(t));
      if (row.damping === null) throw new CombinationRefused(row.name, "bad-param", "cqc over modes needs a damping");
      xi = lam.map(() => row.damping!);
    } else {
      const c = this.source.case(first.source);
      const at = row.terms.map((t) => caseRow(c, t.mode, row.name));
      lam = at.map((i) => c.lam[i]!);
      xi = at.map((i) => (row.damping !== null ? row.damping : c.damping[i]!));
    }
    const omega = checkLambda(lam, row.name, "a cqc term");
    return { omega, damping: checkDamping(xi, row.name, "a cqc term") };
  }

  wholeCase(row: Combination, term: Term): Field {
    const c = this.source.case(term.source);
    const problem = targetProblem(c.increments);
    if (problem !== null) throw new CombinationRefused(row.name, "bad-term", problem);
    const { keyNames, keys, components, stack } = align(c.increments);
    let groups: { slot: number[]; keep: number[] };
    try {
      groups = saGroups(c.saPeriod);
    } catch (err) {
      if (!(err instanceof RangeError)) throw err;
      throw new CombinationRefused(row.name, "bad-param", `case ${JSON.stringify(term.source)}: ${err.message}`);
    }
    const { slot, keep } = groups;
    const size = stack[0]!.length;
    const grouped = keep.map(() => new Float64Array(size));
    stack.forEach((block, a) => {
      const into = grouped[slot[a]!]!;
      for (let at = 0; at < size; at++) into[at]! += block[at]!;
    });
    let values: Float64Array;
    if (row.rule === "srss") values = reduceCells(grouped, srss);
    else if (row.rule === "abs_sum") values = reduceCells(grouped, absSum);
    else if (row.rule === "cqc") {
      const lam = keep.map((i) => c.lam[i]!);
      const own = checkLambda(lam, row.name, `case ${JSON.stringify(term.source)}`);
      const omega = keep.map((i, s) => (c.saPeriod[i]! > 0 ? (2 * Math.PI) / c.saPeriod[i]! : own[s]!));
      const xi = keep.map((i) => (row.damping !== null ? row.damping : c.damping[i]!));
      values = cqcCells(grouped, omega, checkDamping(xi, row.name, `case ${JSON.stringify(term.source)}`));
    } else {
      // The table refuses any other rule over a whole case (A3, "a list excludes whole cases").
      throw new CombinationRefused(row.name, "refused", `${row.rule} does not take a whole case`);
    }
    const factor = Math.abs(term.factor);
    return { keyNames, keys, components, values: values.map((v) => factor * v) };
  }
}

/** Evaluate combination `name` of `table` for the quantity `source` reads. Throws `CombinationRefused`. */
export function evaluate(table: CombinationTable, name: string, source: Source): Combined {
  return new Run(table, source).combination(name);
}

/**
 * One state alone, read as it is: no rule, so no `@linear` check and no badge.
 * A recorded or user result at one stage, mode or increment is signed. A whole
 * case (`spectrum`, mode 0) is many states, so it is refused here.
 */
export function readState(address: StateAddress, source: Source): Combined {
  const name = `<${address.kind} ${address.source}>`;
  let problem = addressProblem(address);
  if (problem === null && address.kind === "spectrum" && address.mode === 0) problem = "a whole case is not one state; combine it with a rule";
  if (problem === null && address.kind === "combination") problem = "a combination is evaluated by name, not read as a state";
  if (problem !== null) throw new CombinationRefused(name, "bad-term", problem);
  return { name, field: source.field(address), signed: true, badges: [] };
}
