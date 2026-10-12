// Pairing a results file with the open model (V4b, #1728), by content only.
//
// The rule is architecture/h5-schema.md, "`results.h5`: the native results
// file", "how to pair a file with its `model.h5`": the results'
// `/meta/lineage/model_hash` equals the model's; when the results carry no
// `/opensees` (no `model_hash`), their `fem_hash` equals the model's, which
// matches the mesh only. Never by name, path or session id: the open set
// finds the files by stem (src/main/pairing.ts), this decides whether they
// belong together. Pure: tested in Node (test/results/pairing.test.ts).

import type { Lineage, ResultsInfo } from "./types.ts";

export type ResultsPairing =
  /** no results file has been read */
  | { state: "none" }
  | { state: "paired"; by: "model_hash" | "fem_hash" }
  /** the results belong to another model: nothing of them is drawn on this one */
  | { state: "mismatch"; reason: string }
  /** a results file is open with no model loaded; its embedded /model is not drawn (V4b) */
  | { state: "noModel" }
  /** the results file carries no hash to pair by */
  | { state: "unpairable"; reason: string };

const short = (h: string): string => h.slice(0, 8);

/**
 * Whether `results` pairs with the loaded model. `model` is null when no
 * model is loaded (none open, still reading, or failed).
 */
export function pairResults(model: { lineage: Lineage | null } | null, results: ResultsInfo | null): ResultsPairing {
  if (!results) return { state: "none" };
  if (!model) return { state: "noModel" };
  const r = results.lineage;
  if (!r) {
    return {
      state: "unpairable",
      reason: `the results file has no /meta/lineage (results ${results.version}), so it cannot be paired with a model by content`,
    };
  }
  const m = model.lineage;
  if (r.modelHash !== null) {
    if (!m?.modelHash) return { state: "mismatch", reason: "the results name a model_hash, and the open model has none to compare (its /meta/lineage has no model_hash)" };
    if (m.modelHash === r.modelHash) return { state: "paired", by: "model_hash" };
    return { state: "mismatch", reason: `model_hash differs: the results were written from another model (results ${short(r.modelHash)}…, model ${short(m.modelHash)}…)` };
  }
  if (r.femHash !== null) {
    if (!m?.femHash) return { state: "mismatch", reason: "the results name a fem_hash, and the open model has none to compare (its /meta/lineage has no fem_hash)" };
    if (m.femHash === r.femHash) return { state: "paired", by: "fem_hash" };
    return { state: "mismatch", reason: `fem_hash differs: the results were written on another mesh (results ${short(r.femHash)}…, model ${short(m.femHash)}…)` };
  }
  return { state: "unpairable", reason: "the results' /meta/lineage has neither model_hash nor fem_hash, so it cannot be paired with a model by content" };
}

/** The banner sentence for a pairing that does not draw results, or null. */
export function pairingNotice(p: ResultsPairing, resultsName: string): string | null {
  switch (p.state) {
    case "none":
      return null;
    case "paired":
      return p.by === "fem_hash" ? `${resultsName} pairs with this model by fem_hash: the mesh matches; the results carry no OpenSees model to compare.` : null;
    case "mismatch":
      return `Results not drawn: ${resultsName} does not belong to this model (${p.reason}).`;
    case "noModel":
      return `Results not drawn: no model is open for ${resultsName}. Its embedded /model is not drawn by this version; open the model.h5 it was written from.`;
    case "unpairable":
      return `Results not drawn: ${p.reason}.`;
    default: {
      const unknown: never = p;
      throw new Error(`pairingNotice: unknown pairing ${JSON.stringify(unknown)}`);
    }
  }
}
