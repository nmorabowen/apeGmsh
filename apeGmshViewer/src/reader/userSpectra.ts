// The `/spectra` cases of a `<stem>.user.h5` (ADR 0130 D5, Amendment 1): one
// spectrum along one direction per case, with its per-mode increments. The
// app's twin of src/apeGmsh/results/user/_spectra.py `case_problem`.
//
// reader/userFile.ts reads a case's attributes, its spectrum table, its modes
// columns and the index of its results (label, family, columns, ids and the
// shape of DATA); this module judges them. DATA itself is never read here: a
// case is listed, and no CQC or other rule is evaluated over it (VU7b, after
// the evaluator of VU4).
//
// A refusal's detail is Python's text, character for character: the printing
// helpers of reader/userModes.ts print a string, a list, a shape and a dtype as
// Python and numpy do.

import { pyFloat } from "./rf1.ts";
import { MODE_KINDS, pyList, pyShape, pyStr } from "./userModes.ts";
import { nameProblem } from "./userRules.ts";

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
