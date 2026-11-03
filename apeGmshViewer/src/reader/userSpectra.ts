// The `/spectra` cases of a `<stem>.user.h5` (ADR 0130 D5, Amendment 1): one
// spectrum along one direction per case, with its per-mode increments. The
// app's twin of src/apeGmsh/results/user/_spectra.py `case_problem`.
//
// reader/userFile.ts reads a case's attributes, its spectrum table, its modes
// columns and the index of its results (label, family, columns, ids and the
// shape of DATA); this module judges them. The index never reads DATA.
//
// VU7b-1 (#2740): one result of one case is read with its values on request
// (`readCaseValues`, main's `user:readCase`), and `spectrumSource` hands the
// cases read to VU4's evaluator (combination/evaluate.ts) as its `Source`, the
// app's twin of Python's `UserSpectra` (results/user/_spectra_source.py). A
// rule over a case is evaluated there and nowhere else.
//
// A refusal's detail is Python's text, character for character: the printing
// helpers of reader/userModes.ts print a string, a list, a shape and a dtype as
// Python and numpy do.

import { CombinationRefused, makeCase, type Case, type EvalReason, type Source } from "../combination/evaluate.ts";
import { makeField, type Field } from "../combination/field.ts";
import { openH5 } from "./h5open.ts";
import { pyFloat, type FpDataset, type FpGroup, type FpH5 } from "./rf1.ts";
import { MODE_KINDS, pyList, pyShape, pyStr } from "./userModes.ts";
import { nameProblem, type Refusal, type StateAddress } from "./userRules.ts";

export { npDtype, pyList, pyShape, pyStr } from "./userModes.ts";

/** The families a 1.0.0 case may hold, by target (Python's `_ON`). */
export const SPECTRA_FAMILIES = { domain: "ON_DOMAIN", nodes: "ON_NODES" } as const;
export type SpectrumTarget = keyof typeof SPECTRA_FAMILIES;

/** One result of a case, without its values: `DATA` is `(S, n, c)`. */
export interface SpectrumQuantity {
  label: string;
  target: SpectrumTarget;
  components: string[];
  /** `DATA`'s shape */
  shape: [number, number, number];
  /** `ID` (nodes only), else null */
  ids: number[] | null;
}

/** A case's `modes/` columns as stored: their lengths are judged by `caseProblem`. */
export interface CaseColumns {
  mode: number[];
  kind: string[];
  lambda: number[];
  saPeriod: number[];
  gamma: number[];
  sa: number[];
  damping: number[];
}

/** A case as read, before `caseProblem` judges it. */
export interface CaseDraft {
  name: string;
  link: string;
  direction: number;
  scale: number;
  periods: number[];
  sa: number[];
  columns: CaseColumns;
  results: SpectrumQuantity[];
  reference: SpectrumQuantity[];
}

/** `/spectra/s<k>`: one case that stands. */
export interface SpectrumCase {
  name: string;
  /** the `s<k>` link under `/spectra` */
  link: string;
  /** 1-based */
  direction: number;
  scale: number;
  /** the spectrum table, unscaled */
  periods: number[];
  sa: number[];
  /** one per mode the case uses, in `modes/` row order (row i is DATA axis 0) */
  modes: SpectrumMode[];
  results: SpectrumQuantity[];
  reference: SpectrumQuantity[];
}

export interface SpectrumMode {
  mode: number;
  kind: string;
  lambda: number;
  /** 0 = the mode's own period */
  saPeriod: number;
  gamma: number;
  /** `@scale` times `Sa` at `saPeriod` (h5-schema.md) */
  sa: number;
  /** NaN = the case gives none for this mode */
  damping: number;
}

/** A draft `caseProblem` passed, as its rows. */
export function caseOf(d: CaseDraft): SpectrumCase {
  const c = d.columns;
  const modes = c.mode.map((mode, i) => ({ mode, kind: c.kind[i]!, lambda: c.lambda[i]!, saPeriod: c.saPeriod[i]!, gamma: c.gamma[i]!, sa: c.sa[i]!, damping: c.damping[i]! }));
  return { name: d.name, link: d.link, direction: d.direction, scale: d.scale, periods: d.periods, sa: d.sa, modes, results: d.results, reference: d.reference };
}

const linkProblem = (name: string): string | null => {
  const problem = nameProblem(name);
  if (problem === null && (name.includes("/") || name.replace(/^\.+|\.+$/g, "") === "")) return "cannot be an HDF5 link";
  return problem;
};

/** Python's `_token_problem`. */
function tokensProblem(tokens: readonly string[], what: string): string | null {
  if (tokens.length === 0) return `${what} is empty`;
  for (const token of tokens) {
    let problem = nameProblem(token);
    if (problem === null && (token.includes(",") || token.includes("\n"))) problem = "has a comma or a newline";
    if (problem !== null) return `${what} ${pyStr(token)}: ${problem}`;
  }
  return null;
}

/** Python's `_quantities_problem`. */
function quantitiesProblem(qs: readonly SpectrumQuantity[], s: number, what: string): string | null {
  if (qs.length === 0) return `${what} holds no result`;
  const seen = new Set<string>();
  for (const q of qs) {
    const label = pyStr(q.label);
    if (!Object.hasOwn(SPECTRA_FAMILIES, q.target)) return `${what} ${label}: target ${pyStr(q.target)} is not one of ${pyList(Object.keys(SPECTRA_FAMILIES).sort())}`;
    const link = linkProblem(q.label);
    if (link !== null) return `${what} label ${label}: ${link}`;
    const key = JSON.stringify([q.target, q.label]);
    if (seen.has(key)) return `${what} holds ${label} twice`;
    seen.add(key);
    const tokens = tokensProblem(q.components, `${what} ${q.label} component`);
    if (tokens !== null) return tokens;
    const [s0, n, c] = q.shape;
    if (s0 !== s || c !== q.components.length) return `${what} ${label} has shape ${pyShape(q.shape)}, expected (${s}, n, ${q.components.length})`;
    if (q.target === "domain" && n !== 1) return `${what} ${label}: a domain value has n = 1, got ${n}`;
    if (q.target === "nodes" && (q.ids === null || q.ids.length !== n)) return `${what} ${label}: ids must be ${n} integers`;
  }
  return null;
}

/** What `reference/RESULTS` must mirror of `RESULTS`: per (target, label), the shape past S, the columns and the ids. */
const layoutOf = (qs: readonly SpectrumQuantity[]): string =>
  JSON.stringify(
    qs
      .map((q) => [q.target, q.label, q.shape.slice(1), q.components, q.ids] as const)
      .sort((a, b) => (a[0] + "\u0000" + a[1] < b[0] + "\u0000" + b[1] ? -1 : 1)),
  );

const allFinite = (xs: readonly number[]) => xs.every((x) => Number.isFinite(x));

/** Why `c` cannot be a `/spectra` case, or null (Python's `case_problem`, check for check and word for word). */
export function caseProblem(c: CaseDraft): string | null {
  const name = nameProblem(c.name);
  if (name !== null) return `name ${pyStr(c.name)}: ${name}`;
  if (c.direction < 1) return `direction ${c.direction} is 1-based`;
  if (!(Number.isFinite(c.scale) && c.scale > 0)) return `scale ${pyFloat(c.scale)} must be finite and > 0`;
  const t = c.periods;
  if (t.length === 0 || c.sa.length !== t.length) return `spectrum T ${pyShape([t.length])} and Sa ${pyShape([c.sa.length])} must be equal, non-empty columns`;
  if (t.some((x) => x < 0) || t.some((x, i) => i > 0 && !(x - t[i - 1]! > 0)) || !allFinite(t)) return "spectrum T must be non-negative and strictly increasing";
  if (!allFinite(c.sa) || c.sa.some((x) => x < 0)) return "spectrum Sa must be finite and non-negative";
  const col = c.columns;
  const m = col.mode.length;
  if (m === 0) return `modes/mode must be integers, got int32 ${pyShape([0])}`;
  if (col.mode.some((k) => k < 1) || new Set(col.mode).size !== m) return "modes/mode must be distinct, 1-based and fit i4";
  if (col.kind.length !== m || col.kind.some((k) => !(MODE_KINDS as readonly string[]).includes(k))) return `modes/kind must be one of ${pyList([...MODE_KINDS].sort())} per mode`;
  for (const [what, xs] of [["lambda", col.lambda], ["sa_period", col.saPeriod], ["gamma", col.gamma], ["sa", col.sa], ["damping", col.damping]] as const) {
    if (xs.length !== m) return `modes/${what} has shape ${pyShape([xs.length])}, expected (${m},)`;
  }
  if (!allFinite(col.lambda) || col.lambda.some((x) => x <= 0)) return "every lambda must be finite and > 0 (lambda <= 0 is refused)";
  if (!allFinite(col.saPeriod) || col.saPeriod.some((x) => x < 0)) return "sa_period must be finite and >= 0 (0 = the mode's own period)";
  if (!allFinite(col.gamma)) return "gamma must be finite";
  if (!allFinite(col.sa) || col.sa.some((x) => x < 0)) return "sa must be finite and non-negative";
  if (col.damping.some((x) => !Number.isNaN(x) && !(x > 0 && x < 1))) return "damping must be NaN or lie in (0, 1)";
  for (const [qs, s, what] of [[c.results, m, "RESULTS"], [c.reference, 1, "reference/RESULTS"]] as const) {
    const problem = quantitiesProblem(qs, s, what);
    if (problem !== null) return problem;
  }
  if (layoutOf(c.results) !== layoutOf(c.reference)) return "reference/RESULTS must hold the results of RESULTS, with their ids and columns";
  return null;
}

// ---------------------------------------------------------------------------
// One result of a case, with its values (VU7b-1)
// ---------------------------------------------------------------------------

/** One result of one case and its values, as `user:readCase` answers. */
export interface CaseValues {
  case: SpectrumCase;
  /** the `RESULTS` entry read */
  quantity: SpectrumQuantity;
  /** `RESULTS/<family>/<label>/DATA`, row-major `(S, n, c)`: row `i` is `case.modes[i]` */
  data: Float64Array;
  /** `reference/RESULTS/<family>/<label>/DATA`, row-major `(1, n, c)` */
  reference: Float64Array;
}

/** What `user:readCase` answers. A refusal carries the evaluator's reason, so a term over it refuses with it. */
export type CaseRead = { ok: true; value: CaseValues } | { ok: false; reason: Extract<EvalReason, "refused" | "unresolved">; detail: string };

/** `/spectra` as reader/userFile.ts `UserIndex.spectra` holds it (structural: that module imports this one). */
export type SpectraZone = { ok: true; value: { cases: readonly SpectrumCase[]; refused: readonly Refusal[] } } | { ok: false; reason: string } | null;

const H5T_FLOAT = 1;

/** The node at `path` under `g`, h5py's `in` link by link, or null where a link is missing. */
function walk(g: FpGroup, path: string): unknown {
  let at: unknown = g;
  for (const part of path.split("/")) {
    const grp = at as Partial<FpGroup> | null;
    if (grp === null || typeof grp !== "object" || typeof grp.keys !== "function" || !grp.keys().includes(part)) return null;
    at = grp.get!(part);
  }
  return at;
}

/** `g`'s dataset at `path`, f8 and of the shape the index judged, as values; anything else is why not (the file moved under the index). */
function dataOf(g: FpGroup, path: string, shape: readonly number[]): Float64Array | string {
  const d = walk(g, path) as Partial<FpDataset> | null;
  if (d === null) return `${g.path}/${path} is missing`;
  if (typeof d !== "object" || d.metadata === undefined || !Array.isArray(d.shape)) return `${g.path}/${path} is not a dataset`;
  if (d.metadata.type !== H5T_FLOAT || d.metadata.size !== 8) return `${d.path} is not f8`;
  if (d.shape.length !== shape.length || d.shape.some((x, i) => x !== shape[i])) return `${d.path} has shape ${pyShape(d.shape)}, the index read ${pyShape(shape)}`;
  const v = d.value;
  if (!(v instanceof Float64Array)) return `${d.path} did not read as f8 values`;
  return v;
}

/**
 * Case `name`'s result `label` on `target`, with its values. `spectra` is the
 * `/spectra` reader/userFile.ts read from `path`, which judged the case; its
 * DATA is then read from `path`. A refused zone or case answers `refused`; a
 * case or a result the zone does not hold answers `unresolved`, and so does a
 * file with no `/spectra`. It raises only for a file h5wasm cannot open.
 */
export function readCaseValues(h5: FpH5, path: string, spectra: SpectraZone, name: string, target: SpectrumTarget, label: string): CaseRead {
  if (spectra === null) return { ok: false, reason: "unresolved", detail: `${path} holds no /spectra, so no case ${pyStr(name)}` };
  if (!spectra.ok) return { ok: false, reason: "refused", detail: `/spectra cannot be read: ${spectra.reason}` };
  const found = spectra.value.cases.find((c) => c.name === name);
  if (found === undefined) {
    const ref = spectra.value.refused.find((r) => r.name === name);
    if (ref !== undefined) return { ok: false, reason: "refused", detail: `case ${pyStr(name)} is refused: ${ref.reason} (${ref.detail})` };
    return { ok: false, reason: "unresolved", detail: `/spectra holds no case ${pyStr(name)}` };
  }
  const quantity = found.results.find((q) => q.target === target && q.label === label);
  if (quantity === undefined) {
    const held = found.results.map((q) => `${q.target}/${q.label}`);
    return { ok: false, reason: "unresolved", detail: `case ${pyStr(name)} holds no ${target} result ${pyStr(label)}; it holds ${pyList(held)}` };
  }
  // caseProblem holds reference/RESULTS to RESULTS's layout, so the entry is there.
  const reference = found.reference.find((q) => q.target === target && q.label === label)!;
  const family = SPECTRA_FAMILIES[target];
  const f = openH5(h5, path);
  try {
    const where = `spectra/${found.link}`;
    const g = walk(f, where) as FpGroup | null;
    if (g === null || typeof g.keys !== "function") return { ok: false, reason: "refused", detail: `/${where} is not a group` };
    const data = dataOf(g, `RESULTS/${family}/${label}/DATA`, quantity.shape);
    if (typeof data === "string") return { ok: false, reason: "refused", detail: data };
    const ref = dataOf(g, `reference/RESULTS/${family}/${label}/DATA`, reference.shape);
    if (typeof ref === "string") return { ok: false, reason: "refused", detail: ref };
    return { ok: true, value: { case: found, quantity, data, reference: ref } };
  } finally {
    f.close();
  }
}

// ---------------------------------------------------------------------------
// The cases as the evaluator's spectrum source (Python's `UserSpectra`)
// ---------------------------------------------------------------------------

/**
 * One quantity of a case: the `ON_NODES` result `label` and its column
 * `component`, both as the file spells them (`DISPLACEMENT`, `Ux`). ADR 0122
 * D7: the app keeps no token table, so the field is labelled
 * `<label>/<component>`, the app's name for the same `.ladruno` column, and
 * joins a stage term's under that name.
 */
export interface CaseQuantityName {
  label: string;
  component: string;
}

/** Whether `/spectra` is stale (A6: the model hashes differ), as `user:read`'s staleness says. */
export interface SpectraStaleness {
  stale: boolean;
  reason: string;
}

const termName = (a: StateAddress) => `<${a.kind} ${a.source}>`;

/**
 * The cases in `reads` (by case name, each as `user:readCase` answered it) as
 * the evaluator's `Source` for quantity `q`. A whole case gives one signed
 * increment per mode, `spectrum` mode `k` one of them, and `reference` the
 * reference state. A stale `/spectra` refuses `stale` unless `allowStale` (A6).
 * A `stage` or `mode` term and an eigenvalue refuse `unresolved`: this source
 * reads `/spectra` alone.
 */
export function spectrumSource(reads: ReadonlyMap<string, CaseRead>, q: CaseQuantityName, staleness: SpectraStaleness, allowStale: boolean): Source {
  const fieldLabel = `${q.label}/${q.component}`;
  const located = (name: string, who: string): { values: CaseValues; column: number } => {
    if (staleness.stale && !allowStale) {
      throw new CombinationRefused(who, "stale", `/spectra is stale (${staleness.reason || "no reason given"}); pass allowStale to read it (ADR 0130 A6)`);
    }
    const read = reads.get(name);
    if (read === undefined) throw new CombinationRefused(who, "unresolved", `case ${pyStr(name)} was not read`);
    if (!read.ok) throw new CombinationRefused(who, read.reason, read.detail);
    const v = read.value;
    if (v.quantity.target !== "nodes" || v.quantity.label !== q.label) {
      throw new CombinationRefused(who, "unresolved", `case ${pyStr(name)} was read for ${v.quantity.target}/${v.quantity.label}, not nodes/${q.label}`);
    }
    const hits = v.quantity.components.flatMap((c, i) => (c === q.component ? [i] : []));
    if (hits.length !== 1) {
      const what = hits.length === 0 ? "no" : `${hits.length} columns for`;
      const held = v.case.results.map((r) => `${r.target}/${r.label}(${r.components.join(",")})`);
      throw new CombinationRefused(who, "unresolved", `case ${pyStr(name)} holds ${what} nodes quantity ${pyStr(fieldLabel)}; it holds ${pyList(held)}`);
    }
    return { values: v, column: hits[0]! };
  };
  const field = (v: CaseValues, data: Float64Array, row: number, column: number): Field => {
    const [, n, c] = v.quantity.shape;
    const ids = v.quantity.ids!;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = data[(row * n + i) * c + column]!;
    return makeField(["id"], ids.map((id) => [id]), [fieldLabel], out);
  };
  return {
    linear: true,
    stored: () => null,
    field(address: StateAddress): Field {
      const who = termName(address);
      if (address.kind !== "spectrum" && address.kind !== "reference") {
        throw new CombinationRefused(who, "unresolved", `a ${address.kind} term is read from the results file; this evaluation reads the /spectra cases only`);
      }
      if (address.kind === "spectrum" && address.mode === 0) throw new CombinationRefused(who, "bad-term", "a whole case is not one state; combine it with a rule");
      const { values, column } = located(address.source, who);
      if (address.kind === "reference") return field(values, values.reference, 0, column);
      const rows = values.case.modes.flatMap((m, i) => (m.mode === address.mode ? [i] : []));
      if (rows.length !== 1) {
        throw new CombinationRefused(who, "unresolved", `case ${pyStr(values.case.name)} has no mode ${address.mode}: its modes are [${values.case.modes.map((m) => m.mode).join(", ")}]`);
      }
      return field(values, values.data, rows[0]!, column);
    },
    eigenvalue(address: StateAddress): number {
      throw new CombinationRefused(termName(address), "unresolved", "an eigenvalue is read from the results file; this evaluation reads the /spectra cases only");
    },
    case(name: string): Case {
      const { values, column } = located(name, `<spectrum ${name}>`);
      const m = values.case.modes;
      return makeCase({
        mode: m.map((x) => x.mode),
        lam: m.map((x) => x.lambda),
        saPeriod: m.map((x) => x.saPeriod),
        damping: m.map((x) => x.damping),
        increments: m.map((_, i) => field(values, values.data, i, column)),
      });
    },
  };
}
