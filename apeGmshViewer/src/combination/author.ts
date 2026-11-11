// VU9b (ADR 0130 A3, A4; HG9 on #2693): a User Combination the human writes in the app.
//
// Before a draft is offered for save it is judged twice. First by the readers'
// rules (reader/userRules.ts `resolve`, the check Python's writer runs as
// `_check_combinations`), together with every row the file holds and the model's:
// a row the readers would refuse is never written. Then by VU4's evaluator
// (./evaluate.ts), for one quantity, through the app's `Source`. The app reads
// `/spectra` cases only; a `stage` or `mode` term is read from the results file,
// which this app does not combine yet. `authoringSource` turns that read into
// `NotReadHere`, which is no refusal of the descriptor: the draft stands, and the
// panel says it was not evaluated here. Every other refusal of the evaluator
// blocks the save.
//
// Pure: no Electron, no node:fs, so the renderer and main share it.

import type { CombinationDraft, CombinationSaveIn } from "../state/types/user.ts";
import { resolve, type Combination, type StateAddress } from "../reader/userRules.ts";
import { CombinationRefused, CombinationTable, evaluate, type Case, type Combined, type Source } from "./evaluate.ts";

/** A replace another program blocks: the typed refusal of HG9 rule 3 (main/combinations.ts answers it; the panel shows it). */
export function combinationHeldRefusal(file: string): Extract<CombinationSaveIn, { ok: false }> {
  return {
    ok: false,
    reason: "held",
    detail: `${file} is open in another program (for example a Python session), which keeps this app from replacing it. Close it there and save again. Nothing was replaced.`,
    file,
    held: true,
  };
}

/** The row a draft saves: A3's descriptor, every term copied. */
export function draftRow(d: CombinationDraft): Combination {
  return { name: d.name, rule: d.rule, terms: d.terms.map((t) => ({ kind: t.kind, source: t.source, step: t.step, mode: t.mode, factor: t.factor })), envelope: d.envelope, fraction: d.fraction, damping: d.damping };
}

/** The draft judged by the readers' rules beside the file's rows: the table it would make, or the refusal (the draft's own first). */
export function judgeDraft(rows: readonly Combination[], modelRows: readonly Combination[], draft: CombinationDraft): { ok: true; table: CombinationTable; name: string } | { ok: false; reason: string; detail: string } {
  const row = draftRow(draft);
  const user = [...rows, row];
  const judged = resolve(user, modelRows);
  const at = user.length - 1;
  const own = judged.refused.find((r) => r.origin === "user" && r.row === at);
  if (own) return { ok: false, reason: own.reason, detail: own.detail };
  const other = judged.refused.find((r) => r.origin === "user");
  if (other) return { ok: false, reason: other.reason, detail: `with this row, ${JSON.stringify(other.name)} is refused: ${other.detail}` };
  return { ok: true, table: new CombinationTable(judged), name: row.name };
}

/** A read the app cannot make: a `stage` or `mode` term (or an eigenvalue) comes from the results file. Not a refusal of the descriptor. */
export class NotReadHere extends Error {}

const NOT_HERE = "a stage or mode term is read from the results file, which this app does not combine yet (Python's Results evaluates it)";

/** `inner` (the `/spectra` cases, or null when none was read), with stage and mode reads raising `NotReadHere` before they reach it. */
export function authoringSource(inner: Source | null): Source {
  const spectra = (what: string): never => {
    throw new CombinationRefused(what, "unresolved", "the file holds no /spectra case to read it from");
  };
  return {
    linear: true,
    stored: () => null,
    field(address: StateAddress) {
      if (address.kind === "stage" || address.kind === "mode") throw new NotReadHere(NOT_HERE);
      return inner === null ? spectra(`<${address.kind} ${address.source}>`) : inner.field(address);
    },
    eigenvalue(address: StateAddress): number {
      if (address.kind === "mode") throw new NotReadHere(NOT_HERE);
      return inner === null ? spectra(`<${address.kind} ${address.source}>`) : inner.eigenvalue(address);
    },
    case(name: string): Case {
      return inner === null ? spectra(`<spectrum ${name}>`) : inner.case(name);
    },
  };
}

/** VU4's evaluator on the draft: its value, its refusal, or that it reads what this app does not. */
export type DraftValue = { ok: true; combined: Combined } | { ok: false; refused: CombinationRefused } | { ok: "not-here"; note: string };

export function evaluateDraft(table: CombinationTable, name: string, inner: Source | null): DraftValue {
  try {
    return { ok: true, combined: evaluate(table, name, authoringSource(inner)) };
  } catch (err) {
    if (err instanceof NotReadHere) return { ok: "not-here", note: `not evaluated here: ${err.message}` };
    if (err instanceof CombinationRefused) return { ok: false, refused: err };
    throw err;
  }
}
