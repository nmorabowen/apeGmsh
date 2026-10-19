// Pairing a results file with the open model (V4b, #1728), by content only.
//
// The rule is architecture/h5-schema.md, "`results.h5`: the native results
// file", "how to pair a file with its `model.h5`": the results'
// `/meta/lineage/model_hash` equals the model's; when the results carry no
// `/opensees` (no `model_hash`), their `fem_hash` equals the model's, which
// matches the mesh only. Never by name, path or session id: the open set
// finds the files by stem (src/main/pairing.ts), this decides whether they
// belong together. Pure: tested in Node (test/results/pairing.test.ts).
//
// A `.ladruno` (ADR 0122 D3; #2127) pairs by the same rule, its lineage the
// INFO/PROVENANCE `model_hash` and `fem_hash` that `stamp_provenance` copies
// from the model.h5. An unstamped file (a recorder's output before the stamp,
// or one still being written) is `unpairable`, and the reason names the
// remedy. Which reader read the file is main's dispatch, by the suffix.

import type { Lineage, ProgramDecls, ResultsInfo } from "./types.ts";

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
/** main/results.ts reads a path with this suffix with reader/ladruno.ts. */
const LADRUNO = /\.ladruno$/i;

/**
 * The first opensees minor whose hashed `/opensees/program` `decl` column
 * holds each call's declaration (A7, #1839), so every archive the bridge
 * writes changes its `model_hash` once here. Named after Python's
 * `h5_reader.PROGRAM_DECL_FROM`; architecture/h5-schema.md, the opensees
 * zone's `2.31.0` history entry. The one place the viewer names it (V4h, #1856).
 * It names the change in the message; it does not decide a side's era: a
 * `to_h5` rewrite of an older file is stamped 2.31 but keeps its old hash, so
 * the era is the completeness marker (`recordsProgramDecl`), as in Python #1861.
 */
export const PROGRAM_DECL_FROM = { major: 2, minor: 31 } as const;
const BOUNDARY = `${PROGRAM_DECL_FROM.major}.${PROGRAM_DECL_FROM.minor}.0`;

/**
 * Whether a side's `model_hash` includes declaration ids, from its
 * `/opensees/decls@program_decl` marker (Python `h5_reader.records_program_decl`):
 * "complete" yes, "absent" no (the readers report a missing marker, or no
 * decls group, as "absent"); null with no opensees zone (null) or another
 * marker value: the era is unknown and no cause is named.
 */
function recordsProgramDecl(marker: string | null): boolean | null {
  if (marker === "complete") return true;
  if (marker === "absent") return false;
  return null;
}

/**
 * The cause of a `model_hash` mismatch where exactly one side's hash includes
 * declaration ids, or null (either side unknown, or both alike). It names the
 * cause and keeps the refusal: the hashes still differ.
 */
function hashEraCause(
  model: { openseesVersion: string | null; programDecl: string | null },
  results: { openseesVersion: string | null; programDecl: string | null },
): string | null {
  const m = recordsProgramDecl(model.programDecl);
  const r = recordsProgramDecl(results.programDecl);
  if (m === null || r === null || m === r) return null;
  const modelVersion = model.openseesVersion;
  const resultsVersion = results.openseesVersion;
  const remedy = m
    ? "re-run the analysis against the re-saved model, or open the results with the model.h5 they were run on"
    : "open the results with the model.h5 they were run on, or re-run the analysis against this model";
  return (
    `the opensees schema ${BOUNDARY} hash change (declaration ids entered the hash; ` +
    `results opensees ${resultsVersion}, model opensees ${modelVersion}); ${remedy}`
  );
}

/**
 * The opensees schema step at which the side-channel calls' declaration ids
 * entered the hashed `/opensees/program` `decl` column (#1853). The
 * `program_decl` marker reads "complete" on both sides of it, so
 * `hashEraCause` cannot see it. Python's `Results.SIDE_CHANNEL_DECL_FROM`
 * (#1911, K #2198); tests/test_app_hash_232_drift.py pins this to it. It names
 * the step in the message only: no rule here reads a version stamp.
 */
export const SIDE_CHANNEL_DECL_FROM = { major: 2, minor: 32 } as const;
const SIDE_CHANNEL_BOUNDARY = `${SIDE_CHANNEL_DECL_FROM.major}.${SIDE_CHANNEL_DECL_FROM.minor}.0`;

/**
 * The six verbs whose calls first carried a proved declaration at
 * `SIDE_CHANNEL_DECL_FROM`: the content of that one step (h5-schema.md, the
 * opensees zone's 2.32.0 entry), so it does not grow with the bridge. A copy of
 * Python's `Results.SIDE_CHANNEL_DECL_VERBS` (T's ruling on #2189), pinned by
 * tests/test_app_hash_232_drift.py: any difference fails there.
 */
export const SIDE_CHANNEL_DECL_VERBS: readonly string[] = [
  "step_hook_ramp",
  "addToParameter",
  "flip_element_stage",
  "update_parameter",
  "set_node_vel",
  "set_node_accel",
] as const;

/** Python's `Results.SIDE_CHANNEL_HASH_CAUSE`, verbatim (the drift test pins it). */
export const SIDE_CHANNEL_HASH_CAUSE = "hash change (side-channel calls' declaration ids entered model_hash)";

/**
 * Whether a side's side-channel calls carry a proved declaration (Python
 * `_side_channel_proof`): null when the program calls none of
 * `SIDE_CHANNEL_DECL_VERBS` (or was not read), true when one of their runs names a
 * declaration, so the ids are in that side's model_hash, false when every one
 * is `-1` (an emit below the step, or a `to_h5` rewrite of one).
 */
export function sideChannelProof(program: ProgramDecls | null | undefined): boolean | null {
  if (!program) return null;
  let proof: boolean | null = null;
  for (const verb of SIDE_CHANNEL_DECL_VERBS) {
    if (!Object.prototype.hasOwnProperty.call(program, verb)) continue;
    if (program[verb]) return true;
    proof = false;
  }
  return proof;
}

/**
 * The `SIDE_CHANNEL_DECL_FROM` cause of a model_hash mismatch, or null: Python's
 * `_side_channel_hash_cause`, with its sentence. Called only when both markers
 * read "complete". Named when both sides carry side-channel calls and exactly
 * one side's hash includes their declaration ids; it reads the hashed column,
 * not the version stamps (a `to_h5` rewrite is stamped current but keeps the
 * old column and hash).
 */
function sideChannelHashCause(resultsProof: boolean | null, modelProof: boolean | null): string | null {
  if (resultsProof === null || modelProof === null || resultsProof === modelProof) return null;
  const remedy = modelProof
    ? "re-run the analysis against the re-saved model, or open the results with the model.h5 they were run on"
    : "open the results with the model.h5 they were run on, or re-run the analysis against this model";
  return `likely cause: the opensees schema ${SIDE_CHANNEL_BOUNDARY} ${SIDE_CHANNEL_HASH_CAUSE}; ${remedy}`;
}

/**
 * Whether `results` pairs with the loaded model. `model` is null when no
 * model is loaded (none open, still reading, or failed); its
 * `openseesVersion` is `/meta@opensees_schema_version` of the model.h5 and
 * `programDecl` its `/opensees/decls@program_decl` marker ("absent" when
 * missing); both null when the model has no opensees zone. `program` is its
 * `/opensees/program` as read.ts `programDecls` reads it; null or absent names
 * no 2.32.0 cause.
 */
export function pairResults(
  model: { lineage: Lineage | null; openseesVersion: string | null; programDecl: string | null; program?: ProgramDecls | null } | null,
  results: ResultsInfo | null,
): ResultsPairing {
  if (!results) return { state: "none" };
  if (!model) return { state: "noModel" };
  const r = results.lineage;
  if (!r && LADRUNO.test(results.path)) {
    const why = results.lineageReason ?? "INFO/PROVENANCE carries no model_hash and fem_hash";
    return {
      state: "unpairable",
      reason: `${why}. Stamp it with stamp_provenance(path, model_h5) against the model.h5 the run was emitted from (ADR 0122 D3), then reopen it`,
    };
  }
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
    let cause = hashEraCause(model, results);
    // Both markers "complete": the 2.32.0 side-channel step, as Python's
    // Results.lineage names it (#2198). A .ladruno embeds no /opensees
    // (programDecl null), so it never gets here: no era cause, as Python.
    if (cause === null && recordsProgramDecl(model.programDecl) === true && recordsProgramDecl(results.programDecl) === true) {
      cause = sideChannelHashCause(sideChannelProof(results.program), sideChannelProof(model.program));
    }
    if (cause)return { state: "mismatch", reason: `model_hash differs: ${cause} (results ${short(r.modelHash)}…, model ${short(m.modelHash)}…)` };
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
      if (LADRUNO.test(resultsName)) return `Results not drawn: no model is open for ${resultsName}; open the model.h5 the run was emitted from.`;
      return `Results not drawn: no model is open for ${resultsName}. Its embedded /model is not drawn by this version; open the model.h5 it was written from.`;
    case "unpairable":
      return `Results not drawn: ${p.reason}.`;
    default: {
      const unknown: never = p;
      throw new Error(`pairingNotice: unknown pairing ${JSON.stringify(unknown)}`);
    }
  }
}
