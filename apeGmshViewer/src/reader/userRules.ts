// The row rules of a combination table (ADR 0130 A2, A3): the app's port of
// src/apeGmsh/opensees/_internal/_combination_rules.py, the one implementation
// both Python readers share. The checks, their order and the reason each one
// names follow that module; its docstrings are the specification.
//
// A bad row refuses alone, with one of REASONS, and the table still opens.
// `rowProblem` judges a row by itself; `judge` adds what needs the whole table:
// `duplicate` and `clash`, `cycle`, then `unresolved`. A `.user.h5` row may name
// a `model.h5` row, never the reverse.
//
// One difference is stated, not hidden: Python refuses a name whose bytes are not
// UTF-8 before they become text. h5wasm has already decoded them (with U+FFFD for
// a bad byte), so this port cannot see that case.

export const RULES = ["linear", "srss", "cqc", "abs_sum", "directional", "envelope"] as const;
export const ENVELOPES = ["max", "min", "absmax"] as const;
export const KINDS = ["stage", "mode", "spectrum", "reference", "combination"] as const;
export const REASONS = ["empty-name", "duplicate", "clash", "unknown-rule", "no-terms", "bad-param", "bad-term", "unresolved", "cycle"] as const;
export type Reason = (typeof REASONS)[number];
export type Origin = "user" | "model";

/** One source address of `terms` and `STATES` (h5-schema.md "Source address"). */
export interface StateAddress {
  kind: string;
  source: string;
  step: number;
  mode: number;
}
export interface Term extends StateAddress {
  factor: number;
}
/** One combination row. `null` is NaN in the file (`""` for `envelope`). */
export interface Combination {
  name: string;
  rule: string;
  terms: Term[];
  envelope: string;
  fraction: number | null;
  damping: number | null;
}
/** A row that does not stand, and why. `row` is its position in its own table. */
export interface Refusal {
  name: string;
  reason: Reason;
  detail: string;
  origin: Origin;
  row: number;
}

const I4_MIN = -(2 ** 31);
const I4_MAX = 2 ** 31 - 1;
const repr = (x: unknown) => JSON.stringify(x);
/** A surrogate with no partner: a string with no UTF-8 form. */
const LONE_SURROGATE = /[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]/;

/** Why `name` breaks the name rule of A2, or null when it holds. */
export function nameProblem(name: unknown): string | null {
  if (typeof name !== "string") return `not a string: ${repr(name)}`;
  if (name === "") return "empty";
  if (LONE_SURROGATE.test(name)) return "not valid UTF-8";
  if (name.normalize("NFC") !== name) return "not NFC";
  // Unicode category Cc: C0, DEL and C1.
  if (/[\u0000-\u001f\u007f-\u009f]/.test(name)) return "has a control character";
  if (name !== pyStrip(name)) return "has leading or trailing whitespace";
  return null;
}

/** Python's `str.strip()`: the characters `str.isspace()` holds for. */
const PY_SPACE = "\\t\\n\\v\\f\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const STRIP = new RegExp(`^[${PY_SPACE}]+|[${PY_SPACE}]+$`, "gu");
const pyStrip = (s: string) => s.replace(STRIP, "");

const isI4 = (v: number) => Number.isInteger(v) && v >= I4_MIN && v <= I4_MAX;

/** Why an address breaks the source-address table, or null. */
export function addressProblem(a: StateAddress): string | null {
  if (!(KINDS as readonly string[]).includes(a.kind)) return `unknown kind ${repr(a.kind)}`;
  const problem = nameProblem(a.source);
  if (problem !== null) return `source ${repr(a.source)}: ${problem}`;
  if (!(isI4(a.step) && isI4(a.mode))) return `step ${a.step} and mode ${a.mode} must be 32-bit integers`;
  const s = a.step;
  const m = a.mode;
  let ok: boolean;
  if (a.kind === "stage") ok = s >= -1 && m === 0;
  else if (a.kind === "mode") ok = s >= -1 && m >= 1;
  else if (a.kind === "spectrum") ok = s === -1 && m >= 0;
  else ok = s === -1 && m === 0; // reference, combination
  return ok ? null : `a ${a.kind} term with step ${s}, mode ${m}`;
}

function termProblem(t: Term): string | null {
  const problem = addressProblem(t);
  if (problem !== null) return problem;
  if (!Number.isFinite(t.factor)) return `factor ${t.factor} must be a finite number`;
  return null;
}

/** The `Terms` column of the rules table, or null when it holds. */
function shapeProblem(row: Combination): string | null {
  const terms = row.terms;
  const whole = terms.filter((t) => t.kind === "spectrum" && t.mode === 0);
  if (row.rule === "srss" || row.rule === "abs_sum") {
    if (whole.length > 0 && terms.length !== 1) return `${row.rule} takes a list or one whole case, not both`;
  } else if (row.rule === "cqc") {
    const first = terms[0]!;
    if (whole.length > 0) {
      if (terms.length !== 1) return "cqc takes one whole case, not a list with one";
    } else if (
      (first.kind !== "mode" && first.kind !== "spectrum") ||
      terms.some((t) => t.kind !== first.kind || t.source !== first.source || t.step !== first.step)
    ) {
      return "cqc takes one whole case, or the modes of one set";
    }
  } else if (whole.length > 0) {
    return `${row.rule} takes a list, and a list excludes whole cases`;
  }
  return null;
}

function paramProblem(row: Combination, origin: Origin): string | null {
  if (row.rule === "envelope") {
    if (!(ENVELOPES as readonly string[]).includes(row.envelope)) return `envelope ${repr(row.envelope)} is not one of ${ENVELOPES.join(", ")}`;
  } else if (row.envelope) {
    return `${row.rule} takes no envelope, got ${repr(row.envelope)}`;
  }
  if (row.rule === "directional") {
    if (row.fraction === null || !(row.fraction >= 0 && row.fraction <= 1)) return `directional fraction ${row.fraction} is not in [0, 1]`;
  } else if (row.fraction !== null) {
    return `${row.rule} takes no fraction, got ${row.fraction}`;
  }
  if (row.damping !== null) {
    if (row.rule !== "cqc") return `${row.rule} takes no damping, got ${row.damping}`;
    if (!(row.damping > 0 && row.damping < 1)) return `cqc damping ${row.damping} is not in (0, 1)`;
  } else if (origin === "model" && row.rule === "cqc") {
    return "cqc in model.h5 needs a damping ratio in (0, 1)";
  }
  return null;
}

/** `[reason, detail]` for a row that is bad by itself, or null; the first failing check names it. */
export function rowProblem(row: Combination, origin: Origin): [Reason, string] | null {
  const name = nameProblem(row.name);
  if (name !== null) return ["empty-name", `name ${repr(row.name)}: ${name}`];
  if (!(RULES as readonly string[]).includes(row.rule)) return ["unknown-rule", `rule ${repr(row.rule)} is not one of ${RULES.join(", ")}`];
  if (row.terms.length === 0) return ["no-terms", "a combination needs at least one term"];
  if (origin === "model") {
    for (const t of row.terms) {
      if (t.kind === "spectrum" || t.kind === "reference") {
        return ["bad-term", `a model.h5 combination cannot reference a ${repr(t.kind)} source: its case lives in .user.h5. Declare this combination in .user.h5 after the run (ADR 0130 A7)`];
      }
    }
  }
  const param = paramProblem(row, origin);
  if (param !== null) return ["bad-param", param];
  for (const t of row.terms) {
    const problem = termProblem(t);
    if (problem !== null) return ["bad-term", problem];
  }
  const shape = shapeProblem(row);
  if (shape !== null) return ["bad-term", shape];
  if (row.rule === "cqc" && row.damping === null && row.terms[0]!.kind === "mode") {
    return ["bad-param", "cqc over mode terms needs a damping: modes have no damping source"];
  }
  return null;
}

/** The verdict on every refused row of a table, keyed by position in `rows`. */
export function judge(rows: readonly (readonly [Origin, Combination])[]): Map<number, [Reason, string]> {
  const verdict = new Map<number, [Reason, string]>();
  rows.forEach(([origin, row], at) => {
    const problem = rowProblem(row, origin);
    if (problem !== null) verdict.set(at, problem);
  });

  const holders = new Map<string, number[]>();
  rows.forEach(([, row], at) => {
    if (nameProblem(row.name) === null) holders.set(row.name, [...(holders.get(row.name) ?? []), at]);
  });
  for (const [name, ats] of holders) {
    if (ats.length < 2) continue;
    const reason: Reason = new Set(ats.map((at) => rows[at]![0])).size > 1 ? "clash" : "duplicate";
    const detail = `${ats.length} combinations are named ${repr(name)}`;
    for (const at of ats) if (!verdict.has(at)) verdict.set(at, [reason, detail]);
  }

  // Names are unique among the rows still standing, so a name is one row.
  const standing = new Map<string, number>();
  rows.forEach(([, row], at) => {
    if (!verdict.has(at)) standing.set(row.name, at);
  });
  const target = (at: number, source: string): number | null => {
    const hit = standing.get(source);
    if (hit === undefined || (rows[at]![0] === "model" && rows[hit]![0] !== "model")) return null;
    return hit;
  };

  const refs = new Map<number, number[]>();
  const dangling = new Set<number>();
  for (const at of standing.values()) {
    const found: number[] = [];
    for (const t of rows[at]![1].terms) {
      if (t.kind !== "combination") continue;
      const hit = target(at, t.source);
      if (hit === null) dangling.add(at);
      else found.push(hit);
    }
    refs.set(at, found);
  }
  const users = new Map<number, number[]>([...refs.keys()].map((at) => [at, []]));
  for (const [at, found] of refs) for (const hit of found) users.get(hit)!.push(at);

  // Kahn's peel: what never settles is in a cycle or reaches one.
  const waiting = new Map([...refs].map(([at, found]) => [at, found.length]));
  const queue = [...waiting].filter(([, n]) => n === 0).map(([at]) => at);
  const settled = new Set<number>();
  while (queue.length > 0) {
    const at = queue.shift()!;
    settled.add(at);
    for (const user of users.get(at)!) {
      const n = waiting.get(user)! - 1;
      waiting.set(user, n);
      if (n === 0) queue.push(user);
    }
  }
  for (const at of refs.keys()) {
    if (!settled.has(at)) verdict.set(at, ["cycle", "this combination is in a reference cycle or reaches one"]);
  }

  // `unresolved` from each dangling reference, carried to every row reaching it.
  const spread = [...dangling].filter((at) => settled.has(at));
  while (spread.length > 0) {
    const at = spread.shift()!;
    if (verdict.has(at)) continue;
    const missing = rows[at]![1].terms.filter((t) => {
      if (t.kind !== "combination") return false;
      const hit = target(at, t.source);
      return hit === null || verdict.has(hit);
    });
    verdict.set(at, ["unresolved", `combination ${repr(missing[0]!.source)} does not stand`]);
    for (const u of users.get(at)!) if (!verdict.has(u)) spread.push(u);
  }
  return verdict;
}

/** Python's `resolve`: which rows of both files stand. */
export function resolve(user: readonly Combination[], model: readonly Combination[] = []): { rows: Combination[]; modelRows: Combination[]; refused: Refusal[] } {
  const rows: [Origin, number, Combination][] = [...user.map((r, i) => ["user", i, r] as [Origin, number, Combination]), ...model.map((r, i) => ["model", i, r] as [Origin, number, Combination])];
  const verdict = judge(rows.map(([o, , r]) => [o, r] as const));
  const refused = [...verdict.keys()]
    .sort((a, b) => a - b)
    .map((at) => {
      const [origin, row, c] = rows[at]!;
      const [reason, detail] = verdict.get(at)!;
      return { name: c.name, reason, detail, origin, row };
    });
  const standing = rows.filter((_, at) => !verdict.has(at));
  return {
    rows: standing.filter(([o]) => o === "user").map(([, , r]) => r),
    modelRows: standing.filter(([o]) => o === "model").map(([, , r]) => r),
    refused,
  };
}
