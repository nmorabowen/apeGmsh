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
 * The first opensees minor whose hashed `/opensees/program` `decl` column
 * holds each call's declaration (A7, #1839), so every archive the bridge
 * writes changes its `model_hash` once here. Named after Python's
 * `h5_reader.PROGRAM_DECL_FROM`; architecture/h5-schema.md, the opensees
 * zone's `2.31.0` history entry, says a tool comparing hashes across it can
 * read `/meta@opensees_schema_version` on both sides to tell the one-time
 * change from a different model. The one place the viewer names it (V4h, #1856).
 */
export const PROGRAM_DECL_FROM = { major: 2, minor: 31 } as const;
const BOUNDARY = `${PROGRAM_DECL_FROM.major}.${PROGRAM_DECL_FROM.minor}.0`;

/** Whether an opensees stamp is at or above the boundary; null when there is no stamp or it is not `X.Y[.Z]`. */
function declEra(version: string | null): boolean | null {
  if (version === null) return null;
  const m = /^(\d+)\.(\d+)(?:\.\d+)?$/.exec(version.trim());
  if (!m) return null;
  const major = Number(m[1]);
  const minor = Number(m[2]);
  return major > PROGRAM_DECL_FROM.major || (major === PROGRAM_DECL_FROM.major && minor >= PROGRAM_DECL_FROM.minor);
}

/**
 * The cause of a `model_hash` mismatch whose two sides straddle
 * PROGRAM_DECL_FROM, or null (either stamp absent, or both on one side of
 * it). It names the cause and keeps the refusal: the hashes still differ.
 */
function hashEraCause(modelVersion: string | null, resultsVersion: string | null): string | null {
  const m = declEra(modelVersion);
  const r = declEra(resultsVersion);
  if (m === null || r === null || m === r) return null;
  const remedy = m
    ? "re-run the analysis against the re-saved model, or open the results with the model.h5 they were run on"
    : "open the results with the model.h5 they were run on, or re-run the analysis against this model";
  return (
    `the opensees schema ${BOUNDARY} hash change (declaration ids entered the hash; ` +
    `results opensees ${resultsVersion}, model opensees ${modelVersion}); ${remedy}`
  );
}

/**
 * Whether `results` pairs with the loaded model. `model` is null when no
 * model is loaded (none open, still reading, or failed); its
 * `openseesVersion` is `/meta@opensees_schema_version` of the model.h5 (null
 * when the model has no opensees zone).
 */
export function pairResults(model: { lineage: Lineage | null; openseesVersion: string | null } | null, results: ResultsInfo | null): ResultsPairing {
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
    const cause = hashEraCause(model.openseesVersion, results.openseesVersion);
    if (cause) return { state: "mismatch", reason: `model_hash differs: ${cause} (results ${short(r.modelHash)}…, model ${short(m.modelHash)}…)` };
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
