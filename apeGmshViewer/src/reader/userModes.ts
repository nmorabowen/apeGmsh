// The `/modes` table of a `<stem>.user.h5` (ADR 0130 D6, Amendment 1 A6): its
// rules and A6's row rule. The app's twin of src/apeGmsh/results/user/_modes.py
// (`table_problem`, `recorded_lambdas`, `row_staleness`).
//
// reader/userFile.ts reads the columns; this module judges them. A row is
// stale when its `lambda` differs from the paired `.ladruno`'s `LAMBDA` of the
// same mode by more than `1e-9` relative. A row the `.ladruno` does not hold,
// or that cannot be matched to it, is stale too: nothing is shown as current
// on no proof.

import { openH5 } from "./h5open.ts";
import { partOf, partSiblings } from "./ladrunoParts.ts";
import { pyFloat, type FpAttr, type FpDataset, type FpFile, type FpGroup, type FpH5 } from "./rf1.ts";
import { nameProblem } from "./userRules.ts";

/** `modalProperties` component tokens, in the order a table lists them (Python's `COMPONENTS`). */
export const MODE_COMPONENTS = ["MX", "MY", "MZ", "RMX", "RMY", "RMZ"] as const;
/** `modeKind` names (Python's `KINDS`). */
export const MODE_KINDS = ["eigen", "residual", "ritz"] as const;
/** A6: the relative `lambda` gap past which a row is stale (Python's `LAMBDA_RTOL`). */
export const LAMBDA_RTOL = 1e-9;

/** One `/modes` row. The three tables hold one value per component, in `components` order. */
export interface ModeRow {
  host: string;
  kind: string;
  /** the `STEP_<s>` id; -1 names the host's only set */
  step: number;
  /** 1-based */
  mode: number;
  lambda: number;
  period: number;
  /** NaN where every participation factor is zero (A1) */
  generalizedMass: number;
  participation: number[];
  massRatio: number[];
  cumulativeRatio: number[];
}

/** `/modes`: `rows` over `components`. */
export interface ModeTable {
  components: string[];
  rows: ModeRow[];
}

/** One row's A6 verdict ("" when current). */
export interface RowVerdict {
  stale: boolean;
  reason: string;
}

/** Why `t` cannot be `/modes`, or null (Python's `table_problem`). */
export function modeTableProblem(t: ModeTable): string | null {
  const m = t.rows.length;
  const c = t.components.length;
  if (m === 0) return "the table has no rows";
  if (c === 0 || new Set(t.components).size !== c || t.components.some((x) => !(MODE_COMPONENTS as readonly string[]).includes(x))) {
    return `components ${t.components.join(",")} must be distinct tokens of ${MODE_COMPONENTS.join(", ")}`;
  }
  for (const r of t.rows) {
    const problem = nameProblem(r.host);
    if (problem !== null) return `host ${JSON.stringify(r.host)}: ${problem}`;
  }
  const bad = [...new Set(t.rows.map((r) => r.kind).filter((k) => !(MODE_KINDS as readonly string[]).includes(k)))].sort();
  if (bad.length > 0) return `kinds ${bad.join(", ")} are not one of ${MODE_KINDS.join(", ")}`;
  // A loop, not Math.min(...col): a spread of a long column overflows the call stack.
  for (const r of t.rows) if (r.step < -1) return "a step is below -1";
  for (const r of t.rows) if (r.mode < 1) return "modes are 1-based";
  if (new Set(t.rows.map((r) => JSON.stringify([r.host, r.step, r.mode]))).size !== m) return "two rows name the same (host, step, mode)";
  if (t.rows.some((r) => !Number.isFinite(r.lambda) || r.lambda <= 0)) return "every lambda must be finite and > 0 (lambda <= 0 is refused)";
  if (t.rows.some((r) => !Number.isFinite(r.period) || r.period <= 0)) return "every period must be finite and > 0";
  if (t.rows.some((r) => r.generalizedMass === Infinity || r.generalizedMass === -Infinity)) return "a generalized_mass is infinite";
  const tables = [["participation", "participation"], ["mass_ratio", "massRatio"], ["cumulative_ratio", "cumulativeRatio"]] as const;
  for (const [what, key] of tables) {
    for (const r of t.rows) {
      if (r[key].length !== c) return `${what} has a row of ${r[key].length} values, expected ${c}`;
      if (r[key].some((x) => x === Infinity || x === -Infinity)) return `${what} holds an infinite value`;
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// The `.ladruno`'s eigenvalues (Python's `recorded_lambdas`)
// ---------------------------------------------------------------------------

/** The results file's modes cannot be read (Python's `ValueError` / `KeyError` / `OSError`). */
export class LambdasRefused extends Error {}

/** `{host, step}` as one map key. */
const setKey = (host: string, step: number) => `${host}\u0000${step}`;

/** `(host, STEP id)` to `{mode: LAMBDA}`, with the hosts that hold a set. */
export interface RecordedLambdas {
  sets: Map<string, Map<number, number>>;
  /** the STEP ids recorded on each host, ascending */
  steps: Map<string, number[]>;
}

/** The modal groups under `ON_NODES`, translational first (Python's `discover_modes` order). */
const MODAL_GROUPS = ["MODES_OF_VIBRATION(U)", "MODES_OF_VIBRATION(R)"];
const STAGE = "MODEL_STAGE[";
const H5T_INTEGER = 0;
const H5T_FLOAT = 1;

/** The file to read: `path` alone, or the first part of its set (Python's `discover_partition_files(path)[0]`). */
function firstPart(path: string, list: (dir: string) => readonly string[]): string {
  if (partOf(path) === null) return path;
  let parts: { index: number; path: string }[];
  try {
    parts = partSiblings(path, list).sort((a, b) => a.index - b.index);
  } catch (err) {
    throw new LambdasRefused(`the part files of ${path} cannot be listed: ${err instanceof Error ? err.message : String(err)}`);
  }
  if (parts.length === 0) throw new LambdasRefused(`no part file of ${path} is found`);
  return parts[0]!.path;
}

/** Keys with `prefix`, sorted by their integer suffix; a suffix that is not one raises (Python's `_keyed`). */
function keyed(g: FpGroup, prefix: string): string[] {
  const keys = g.keys().filter((k) => k.startsWith(prefix));
  for (const k of keys) {
    if (!/^-?\d+$/.test(k.slice(prefix.length))) throw new LambdasRefused(`${g.path}/${k}: ${JSON.stringify(k.slice(prefix.length))} is not an integer`);
  }
  return keys.sort((a, b) => Number(a.slice(prefix.length)) - Number(b.slice(prefix.length)));
}

/** `float(_attr_scalar(attr))`: one numeric value. */
function scalar(attr: FpAttr, where: string): number {
  const { metadata, value } = attr;
  if (metadata.type !== H5T_FLOAT && metadata.type !== H5T_INTEGER) throw new LambdasRefused(`${where} is not numeric`);
  const list = ArrayBuffer.isView(value) ? Array.from(value as unknown as ArrayLike<number | bigint>, Number) : Array.isArray(value) ? value.map(Number) : [Number(value)];
  if (list.length !== 1) throw new LambdasRefused(`${where} holds ${list.length} values`);
  return list[0]!;
}

/**
 * `{(host, s): {mode: LAMBDA}}` of every modal set a `.ladruno` records.
 * `host` is `stage_<i>`, the `i`-th `MODEL_STAGE` by its index (the name of a
 * stage the run stamped none for), `s` the `STEP_<s>` id. A part set is read
 * from its first part: every part records the same modes. Raises
 * `LambdasRefused` on a file whose modal record cannot be read or disagrees.
 */
export function recordedLambdas(h5: FpH5, path: string, list: (dir: string) => readonly string[]): RecordedLambdas {
  const first = firstPart(path, list);
  let f: FpFile;
  try {
    f = openH5(h5, first);
  } catch (err) {
    throw new LambdasRefused(`${first} cannot be opened: ${err instanceof Error ? err.message : String(err)}`);
  }
  const isGroup = (x: unknown): x is FpGroup => x instanceof (h5.Group as unknown as abstract new () => unknown);
  const isDataset = (x: unknown): x is FpDataset => x instanceof (h5.Dataset as unknown as abstract new () => unknown);
  const sub = (g: FpGroup, name: string): FpGroup | null => {
    if (!g.keys().includes(name)) return null;
    const x = g.get(name);
    return isGroup(x) ? x : null;
  };
  const out: RecordedLambdas = { sets: new Map(), steps: new Map() };
  try {
    const stages = f
      .keys()
      .filter((k) => k.startsWith(STAGE))
      .sort((a, b) => Number(a.slice(STAGE.length, -1)) - Number(b.slice(STAGE.length, -1)));
    stages.forEach((stage, i) => {
      const results = sub(f, stage) === null ? null : sub(sub(f, stage)!, "RESULTS");
      const onNodes = results === null ? null : sub(results, "ON_NODES");
      if (onNodes === null) return;
      const names = MODAL_GROUPS.filter((n) => onNodes.keys().includes(n));
      if (names.length === 0) return;
      const perGroup = names.map((name) => {
        const g = sub(onNodes, name);
        const data = g === null ? null : sub(g, "DATA");
        if (data === null) throw new LambdasRefused(`${first} ${stage}: ON_NODES/${name} has no DATA group; the modal record is incomplete`);
        const sets: [number, [number, number][]][] = [];
        for (const stepKey of keyed(data, "STEP_")) {
          const sg = sub(data, stepKey);
          if (sg === null) throw new LambdasRefused(`${data.path}/${stepKey} is not a group`);
          const modes: [number, number][] = [];
          for (const modeKey of keyed(sg, "MODE_")) {
            const ds = sg.get(modeKey);
            const where = `${first} ${stage}: ON_NODES/${name}/DATA/${stepKey}/${modeKey}`;
            if (!isDataset(ds) && !isGroup(ds)) throw new LambdasRefused(`${where} is not a dataset`);
            const lam = (ds as FpDataset).attrs["LAMBDA"];
            if (lam === undefined) throw new LambdasRefused(`${where} has no LAMBDA attribute; the eigenvalue of this mode is not recorded`);
            modes.push([Number(modeKey.slice("MODE_".length)) + 1, scalar(lam, `${where}@LAMBDA`)]);
          }
          if (modes.length > 0) sets.push([Number(stepKey.slice("STEP_".length)), modes]);
        }
        return sets;
      });
      const ref = JSON.stringify(perGroup[0]);
      names.slice(1).forEach((name, k) => {
        if (JSON.stringify(perGroup[k + 1]) !== ref) {
          throw new LambdasRefused(`${first} ${stage}: ON_NODES/${name} lists different modes or eigenvalues than ON_NODES/${names[0]}; the modal record is inconsistent`);
        }
      });
      const host = `stage_${i}`;
      for (const [step, modes] of perGroup[0]!) {
        out.sets.set(setKey(host, step), new Map(modes));
        out.steps.set(host, [...(out.steps.get(host) ?? []), step].sort((a, b) => a - b));
      }
    });
  } catch (err) {
    // What h5wasm raises on a damaged file (a broken link) is the record not reading, as Python's OSError and KeyError.
    if (err instanceof LambdasRefused) throw err;
    throw new LambdasRefused(`${first}: ${err instanceof Error ? err.message : String(err)}`);
  } finally {
    f.close();
  }
  return out;
}

// ---------------------------------------------------------------------------
// The row rule of A6 (Python's `row_staleness`)
// ---------------------------------------------------------------------------

const HOST = /^stage_(\d+)$/;

/** One verdict per row of `t` against `recordedLambdas`. */
export function rowVerdicts(t: ModeTable, lambdas: RecordedLambdas): RowVerdict[] {
  return t.rows.map((r) => oneRow(r.host, r.step, r.mode, r.lambda, lambdas));
}

function oneRow(host: string, step: number, mode: number, lam: number, lambdas: RecordedLambdas): RowVerdict {
  if (!HOST.test(host)) {
    return { stale: true, reason: `host '${host}' is not a stage_<i> name; a stamped stage name is not resolved against the results file here` };
  }
  const sets = lambdas.steps.get(host) ?? [];
  if (sets.length === 0) return { stale: true, reason: `the results file records no modes on ${host}` };
  let s = step;
  if (s === -1) {
    if (sets.length !== 1) return { stale: true, reason: `step -1 names the only set, and ${host} holds ${sets.length}` };
    s = sets[0]!;
  }
  const set = lambdas.sets.get(setKey(host, s));
  if (set === undefined) return { stale: true, reason: `the results file records no STEP_${s} on ${host}` };
  const recorded = set.get(mode);
  if (recorded === undefined) return { stale: true, reason: `the results file records no mode ${mode} on ${host} STEP_${s}` };
  const gap = Math.abs(lam - recorded);
  if (!Number.isFinite(gap) || gap > LAMBDA_RTOL * Math.abs(recorded)) {
    return { stale: true, reason: `lambda ${pyFloat(lam)} here, ${pyFloat(recorded)} in the results file (more than 1e-09 relative)` };
  }
  return { stale: false, reason: "" };
}
