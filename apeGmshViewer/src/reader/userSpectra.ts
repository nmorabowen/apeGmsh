// The `/spectra` cases of a `<stem>.user.h5` (ADR 0130 D5, Amendment 1): one
// spectrum along one direction per case, with its per-mode increments. The
// app's twin of src/apeGmsh/results/user/_spectra.py `case_problem`.
//
// reader/userFile.ts reads a case's attributes, its spectrum table, its modes
// table and the index of its results (label, family, columns, ids and the
// shape of DATA); this module judges them. DATA itself is never read here: a
// case is listed, and no CQC or other rule is evaluated over it (VU7b, after
// the evaluator of VU4).

import { MODE_KINDS } from "./userModes.ts";
import { nameProblem } from "./userRules.ts";

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

/** `/spectra/s<k>`: one case. */
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

const linkProblem = (name: string): string | null => {
  const problem = nameProblem(name);
  if (problem === null && (name.includes("/") || name.replace(/^\.+|\.+$/g, "") === "")) return "cannot be an HDF5 link";
  return problem;
};

function tokensProblem(tokens: readonly string[], what: string): string | null {
  if (tokens.length === 0) return `${what} is empty`;
  for (const token of tokens) {
    let problem = nameProblem(token);
    if (problem === null && (token.includes(",") || token.includes("\n"))) problem = "has a comma or a newline";
    if (problem !== null) return `${what} ${JSON.stringify(token)}: ${problem}`;
  }
  return null;
}

/** Python's `_quantities_problem`. */
function quantitiesProblem(qs: readonly SpectrumQuantity[], s: number, what: string): string | null {
  if (qs.length === 0) return `${what} holds no result`;
  const seen = new Set<string>();
  for (const q of qs) {
    if (!Object.hasOwn(SPECTRA_FAMILIES, q.target)) return `${what} ${JSON.stringify(q.label)}: target ${JSON.stringify(q.target)} is not one of domain, nodes`;
    const link = linkProblem(q.label);
    if (link !== null) return `${what} label ${JSON.stringify(q.label)}: ${link}`;
    const key = JSON.stringify([q.target, q.label]);
    if (seen.has(key)) return `${what} holds ${JSON.stringify(q.label)} twice`;
    seen.add(key);
    const tokens = tokensProblem(q.components, `${what} ${q.label} component`);
    if (tokens !== null) return tokens;
    const [s0, n, c] = q.shape;
    if (s0 !== s || c !== q.components.length) return `${what} ${JSON.stringify(q.label)} has shape (${q.shape.join(", ")}), expected (${s}, n, ${q.components.length})`;
    if (q.target === "domain" && n !== 1) return `${what} ${JSON.stringify(q.label)}: a domain value has n = 1, got ${n}`;
    if (q.target === "nodes" && (q.ids === null || q.ids.length !== n)) return `${what} ${JSON.stringify(q.label)}: ids must be ${n} integers`;
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

/** Why `c` cannot be a `/spectra` case, or null (Python's `case_problem`). */
export function caseProblem(c: SpectrumCase): string | null {
  const name = nameProblem(c.name);
  if (name !== null) return `name ${JSON.stringify(c.name)}: ${name}`;
  if (c.direction < 1) return `direction ${c.direction} is 1-based`;
  if (!(Number.isFinite(c.scale) && c.scale > 0)) return `scale ${c.scale} must be finite and > 0`;
  const t = c.periods;
  if (t.length === 0 || c.sa.length !== t.length) return `spectrum T (${t.length}) and Sa (${c.sa.length}) must be equal, non-empty columns`;
  if (!allFinite(t) || t.some((x) => x < 0) || t.some((x, i) => i > 0 && x - t[i - 1]! <= 0)) return "spectrum T must be non-negative and strictly increasing";
  if (!allFinite(c.sa) || c.sa.some((x) => x < 0)) return "spectrum Sa must be finite and non-negative";
  const m = c.modes.length;
  if (m === 0) return "modes/mode must be integers, got an empty column";
  if (c.modes.some((r) => r.mode < 1) || new Set(c.modes.map((r) => r.mode)).size !== m) return "modes/mode must be distinct, 1-based and fit i4";
  if (c.modes.some((r) => !(MODE_KINDS as readonly string[]).includes(r.kind))) return `modes/kind must be one of ${[...MODE_KINDS].sort().join(", ")} per mode`;
  const lam = c.modes.map((r) => r.lambda);
  if (!allFinite(lam) || lam.some((x) => x <= 0)) return "every lambda must be finite and > 0 (lambda <= 0 is refused)";
  const saPeriod = c.modes.map((r) => r.saPeriod);
  if (!allFinite(saPeriod) || saPeriod.some((x) => x < 0)) return "sa_period must be finite and >= 0 (0 = the mode's own period)";
  if (!allFinite(c.modes.map((r) => r.gamma))) return "gamma must be finite";
  const sa = c.modes.map((r) => r.sa);
  if (!allFinite(sa) || sa.some((x) => x < 0)) return "sa must be finite and non-negative";
  if (c.modes.some((r) => !Number.isNaN(r.damping) && !(r.damping > 0 && r.damping < 1))) return "damping must be NaN or lie in (0, 1)";
  for (const [qs, s, what] of [[c.results, m, "RESULTS"], [c.reference, 1, "reference/RESULTS"]] as const) {
    const problem = quantitiesProblem(qs, s, what);
    if (problem !== null) return problem;
  }
  if (layoutOf(c.results) !== layoutOf(c.reference)) return "reference/RESULTS must hold the results of RESULTS, with their ids and columns";
  return null;
}
