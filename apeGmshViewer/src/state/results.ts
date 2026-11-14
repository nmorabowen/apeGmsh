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
//
// Particles (ADR 0134 D1, D3) read without a model: a `<stem>.particles.h5` (the
// seeded set), or a `.ladruno` whose every stage holds particles only, opened with
// no model is `noModel` with `particlesOnly`, and the banner says the particles read
// without one, not "Results not drawn". A `.ladruno` with an FE stage keeps today's
// sentence: its FE results need the model.
//
// The particle slice (ADR 0134 D5, M2-S4 #2983) is reduced here too: the store keeps
// the particle index and what is asked (stage, output, colour); the arrays go from the
// effects to the viewport and never enter the store. A set that holds only its
// `<stem>.particles.h5` draws it, with the "particle set open without a model" notice.

import type { Event, Lineage, ParticleFieldInfo, ParticlesState, ParticleStageInfo, ProgramDecls, ResultsInfo, State } from "./types.ts";

export type ResultsPairing =
  /** no results file has been read */
  | { state: "none" }
  | { state: "paired"; by: "model_hash" | "fem_hash" }
  /** the results belong to another model: nothing of them is drawn on this one */
  | { state: "mismatch"; reason: string }
  /**
   * a results file is open with no model loaded; its embedded /model is not drawn (V4b).
   * `particlesOnly`: the file holds particles only (`particlesOnly(info)`), which read
   * without a model (ADR 0134 D3); absent for every other file
   */
  | { state: "noModel"; particlesOnly?: true }
  /** the results file carries no hash to pair by */
  | { state: "unpairable"; reason: string };

/**
 * A `.ladruno` stage's `unread` entry for its RESULTS/ON_PARTICLES bucket when the
 * file is read at FORMAT_MINOR_VERSION >= 2 and the stage's particles were read
 * (reader/ladruno.ts writes it): the FE stage does not read them,
 * reader/ladrunoParticles.ts does (ADR 0134 D3). A stage whose particles were
 * refused lists `ON_PARTICLES (particles refused, not read: <why>)` instead, so
 * `particlesOnly` never claims particles that did not read (#2985 review).
 */
export const PARTICLES_UNREAD = "ON_PARTICLES (particle results: read by the particle reader, not as FE results)";

const short = (h: string): string => h.slice(0, 8);
/** main/results.ts reads a path with this suffix with reader/ladruno.ts. */
const LADRUNO = /\.ladruno$/i;
/** The seeded particle set (ADR 0134 D1; main/pairing.ts's `particles` slot). */
const PARTICLE_SET = /\.particles\.h5$/i;

/**
 * Whether a results info holds particles only: a `.ladruno` with at least one stage,
 * every stage listing its ON_PARTICLES bucket as PARTICLES_UNREAD and holding no FE
 * result (nodal, Gauss, line-station or spring). Such a file reads without a model.
 */
export function particlesOnly(info: Pick<ResultsInfo, "path" | "stages">): boolean {
  if (!LADRUNO.test(info.path) || info.stages.length === 0) return false;
  return info.stages.every(
    (s) => s.unread.includes(PARTICLES_UNREAD) && s.nodal.length === 0 && s.gauss.length === 0 && (s.lineStations?.length ?? 0) === 0 && (s.springs ?? 0) === 0,
  );
}

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
 * Python's whole line (`_side_channel_hash_cause`), one per direction, as
 * literals so tests/test_app_hash_232_drift.py compares each with Python's
 * output: frame, step, phrase and remedy. FORWARD: the model's hash includes
 * the side-channel ids and the results' does not; REVERSE: the other way.
 * test/results/pairing.test.ts checks they carry `SIDE_CHANNEL_DECL_FROM` and
 * `SIDE_CHANNEL_HASH_CAUSE`.
 */
export const SIDE_CHANNEL_LINE_FORWARD =
  "likely cause: the opensees schema 2.32.0 hash change (side-channel calls' declaration ids entered model_hash); re-run the analysis against the re-saved model, or open the results with the model.h5 they were run on";
export const SIDE_CHANNEL_LINE_REVERSE =
  "likely cause: the opensees schema 2.32.0 hash change (side-channel calls' declaration ids entered model_hash); open the results with the model.h5 they were run on, or re-run the analysis against this model";

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
  return modelProof ? SIDE_CHANNEL_LINE_FORWARD : SIDE_CHANNEL_LINE_REVERSE;
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
  if (!model) return particlesOnly(results) ? { state: "noModel", particlesOnly: true } : { state: "noModel" };
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

// -- ADR 0134 D5: the particles the points layer draws ------------------------------

type ParticleEvent = Extract<
  Event,
  {
    type:
      | "particlesOpened"
      | "particlesLoaded"
      | "particlesFailed"
      | "particlesClosed"
      | "setParticleStage"
      | "setParticleOutput"
      | "setParticleColour"
      | "setParticleGrid"
      | "setParticlesOn"
      | "particleFrame";
  }
>;

/** The colour a stage opens with: POSITION's last component (the vertical: y in 2-D, z in 3-D). */
export function defaultParticleColour(st: ParticleStageInfo): { field: string; component: number } | null {
  const pos = st.fields.find((f) => f.name === "POSITION");
  return pos ? { field: "POSITION", component: Math.min(st.ndm, pos.components.length) - 1 } : null;
}

/** Whether `colour` names a field the stage offers and one of its components. */
function colourOffered(st: ParticleStageInfo | undefined, colour: { field: string; component: number }): boolean {
  const f = st?.fields.find((x) => x.name === colour.field);
  return f !== undefined && Number.isInteger(colour.component) && colour.component >= 0 && colour.component < f.components.length;
}

/** The output a stage opens on: its last committed one (0 for a stage with none). */
const lastOutput = (st: ParticleStageInfo | undefined): number => Math.max(0, (st?.nOutputs ?? 0) - 1);

/** A stage index, output and colour that hold on `stages`: the ones asked while they still do, else the stage's defaults. */
function settle(p: ParticlesState, stages: ParticleStageInfo[], keepId: string | null): Pick<ParticlesState, "stage" | "output" | "colour"> {
  const at = keepId === null ? -1 : stages.findIndex((x) => x.id === keepId);
  const stage = at >= 0 ? at : 0;
  const st = stages[stage];
  const output = at >= 0 && p.output < (st?.nOutputs ?? 0) ? p.output : lastOutput(st);
  const colour = p.colour !== null && colourOffered(st, p.colour) ? p.colour : st ? defaultParticleColour(st) : null;
  return { stage, output, colour };
}

/** The particle slice's reducer (state/reduce.ts calls it for every particle event); a request the state cannot honour throws. */
export function reduceParticles(p: ParticlesState | null, e: ParticleEvent): ParticlesState | null {
  switch (e.type) {
    case "particlesOpened": {
      // The same file again (a re-read after a rewrite) keeps what was read until the new read lands.
      if (p && p.path === e.path && p.source === e.source) return { ...p, status: "loading", error: null };
      return { source: e.source, path: e.path, status: "loading", error: null, stages: [], unread: [], stage: 0, output: 0, colour: null, grid: p?.grid ?? false, on: p?.on ?? true, frame: null };
    }
    case "particlesLoaded": {
      if (!p || p.path !== e.path || p.source !== e.source) throw new Error(`particlesLoaded: ${e.path} is not the particles being read (${p ? p.path : "none"})`);
      for (const st of e.stages) {
        const n = st.nOutputs;
        if (!Number.isInteger(n) || n < 0 || st.steps.length !== n || st.time.length !== n || st.counts.length !== n) {
          throw new RangeError(`particlesLoaded: stage ${st.id} says ${n} outputs and carries ${st.steps.length} steps, ${st.time.length} times, ${st.counts.length} counts`);
        }
      }
      const keepId = p.status === "ready" || p.stages.length > 0 ? (p.stages[p.stage]?.id ?? null) : null;
      return { ...p, status: "ready", error: null, stages: e.stages, unread: e.unread, ...settle(p, e.stages, keepId), frame: null };
    }
    case "particlesFailed":
      if (!p || p.path !== e.path) throw new Error(`particlesFailed: ${e.path} is not the particles being read (${p ? p.path : "none"})`);
      return { ...p, status: "failed", error: e.error, stages: [], unread: [], frame: null };
    case "particlesClosed":
      return null;
    case "setParticleStage": {
      if (!p || p.status !== "ready") throw new Error("setParticleStage: no particles are read");
      if (!Number.isInteger(e.stage) || e.stage < 0 || e.stage >= p.stages.length) throw new RangeError(`setParticleStage: ${e.stage} is not a particle stage (0..${p.stages.length - 1})`);
      if (e.stage === p.stage) return p;
      const st = p.stages[e.stage]!;
      const colour = p.colour !== null && colourOffered(st, p.colour) ? p.colour : defaultParticleColour(st);
      return { ...p, stage: e.stage, output: lastOutput(st), colour, frame: null };
    }
    case "setParticleOutput": {
      if (!p || p.status !== "ready") throw new Error("setParticleOutput: no particles are read");
      const n = p.stages[p.stage]?.nOutputs ?? 0;
      if (!Number.isInteger(e.output) || e.output < 0 || e.output >= n) throw new RangeError(`setParticleOutput: output ${e.output} is outside the stage's ${n} outputs`);
      return e.output === p.output ? p : { ...p, output: e.output };
    }
    case "setParticleColour": {
      if (!p || p.status !== "ready") throw new Error("setParticleColour: no particles are read");
      if (e.colour !== null && !colourOffered(p.stages[p.stage], e.colour)) {
        const st = p.stages[p.stage];
        throw new RangeError(`setParticleColour: ${JSON.stringify(e.colour)} is not a field component of the stage; it offers ${st ? st.fields.map((f) => `${f.name}/${f.components.join(",")}`).join("; ") : "none"}`);
      }
      const same = e.colour === null ? p.colour === null : p.colour !== null && p.colour.field === e.colour.field && p.colour.component === e.colour.component;
      return same ? p : { ...p, colour: e.colour };
    }
    case "setParticleGrid":
      if (typeof e.on !== "boolean") throw new TypeError(`setParticleGrid: on is ${JSON.stringify(e.on)}`);
      if (!p) throw new Error("setParticleGrid: no particles are open");
      return e.on === p.grid ? p : { ...p, grid: e.on };
    case "setParticlesOn":
      if (typeof e.on !== "boolean") throw new TypeError(`setParticlesOn: on is ${JSON.stringify(e.on)}`);
      if (!p) throw new Error("setParticlesOn: no particles are open");
      return e.on === p.on ? p : { ...p, on: e.on };
    case "particleFrame": {
      // A frame for an output no longer shown is dropped (the effects also drop replies that are not their latest).
      if (!p || p.status !== "ready" || e.frame.stage !== p.stage || e.frame.output !== p.output) return p;
      const f = e.frame;
      if (!Number.isInteger(f.count) || f.count < 0 || !Number.isInteger(f.noValue) || f.noValue < 0 || f.noValue > f.count) {
        throw new RangeError(`particleFrame: ${f.noValue} without a value of ${f.count} particles`);
      }
      return { ...p, frame: f };
    }
    default: {
      const unknown: never = e;
      throw new Error(`reduceParticles: unknown event ${JSON.stringify(unknown)}`);
    }
  }
}

/** A field's DIMENSION in the stage's units: `F/L^2` with kN and m is `kN/m^2`; "" stays "" (dimensionless). */
export function particleUnit(field: ParticleFieldInfo, units: ParticleStageInfo["units"]): string {
  if (field.name === "POSITION") return units.length;
  return field.dimension.replace(/\b[LFt]\b/g, (s) => (s === "L" ? units.length : s === "F" ? units.force : units.time));
}

const fmt = (x: number): string => (x === 0 ? "0" : Math.abs(x) >= 1e4 || Math.abs(x) < 1e-3 ? x.toExponential(3) : String(Number(x.toPrecision(4))));

/** The banner's particle lines: what is drawn and how it is coloured, and every particle stage, field or value that is not. */
export function particleNotices(s: Pick<State, "particles" | "artifacts">): string[] {
  const p = s.particles;
  if (p === null) return [];
  const name = baseName(p.path);
  const out: string[] = [];
  if (p.status === "failed") return [`Particles not drawn: ${name}: ${p.error ?? "the read failed"}.`];
  if (p.status === "loading" && p.stages.length === 0) return [];
  if (p.source === "set" && s.artifacts.model?.status !== "ready") {
    const n = p.stages[0]?.counts[0];
    out.push(`Particle set open without a model: ${name} draws its ${n === undefined ? "" : `${n.toLocaleString()} `}seeded particles alone (ADR 0134); open the model.h5 or the run's .ladruno to see more.`);
  }
  for (const u of p.unread) out.push(`Particles of ${u.stage} not drawn: ${u.reason}.`);
  if (p.stages.length === 0) {
    if (p.unread.length === 0) out.push(`${name} holds no particles to draw.`);
    return out;
  }
  const st = p.stages[p.stage]!;
  for (const r of st.refusedFields) out.push(`Particle field ${r.field} of ${st.name} not offered: ${r.reason}.`);
  if (st.torn) out.push(`Particles of ${st.name}: ${st.nOutputs} output${st.nOutputs === 1 ? "" : "s"} read; ${st.torn}.`);
  if (st.nOutputs === 0) {
    out.push(`Particles of ${st.name}: no output was committed, so none is drawn.`);
    return out;
  }
  if (!p.on) return out;
  const f = p.frame;
  if (f?.status === "failed") {
    out.push(`Particle output ${p.output} of ${st.name} not drawn: ${f.error ?? "the read failed"}.`);
    return out;
  }
  const where = p.source === "set" ? "the seeded set" : `output ${p.output + 1} of ${st.nOutputs} (step ${st.steps[p.output]}, t = ${fmt(st.time[p.output]!)} ${st.units.time})`;
  if (p.colour === null) {
    out.push(`Particles: ${where}, one colour.`);
    return out;
  }
  const field = st.fields.find((x) => x.name === p.colour!.field)!;
  const unit = particleUnit(field, st.units);
  const label = `${field.name}/${field.components[p.colour.component]}${unit ? ` (${unit})` : ""}`;
  if (f?.status !== "ready") {
    out.push(`Particles: ${where}, coloured by ${label}.`);
    return out;
  }
  const range = f.range ? `${fmt(f.range.min)} to ${fmt(f.range.max)}` : "no finite value";
  out.push(`Particles: ${where}, ${f.count.toLocaleString()} drawn, coloured by ${label}: ${range}.`);
  if (f.count > 0 && f.noValue === f.count) out.push(`Particles: no particle has a value of ${label} at this output (NaN: their material does not answer it); all are drawn in the no-value colour.`);
  else if (f.noValue > 0) out.push(`Particles: ${f.noValue.toLocaleString()} of ${f.count.toLocaleString()} have no value of ${label} (NaN: their material does not answer it); they are drawn in the no-value colour.`);
  if (f.count === 0) out.push(`Particles: output ${p.output + 1} of ${st.name} holds no particle.`);
  return out;
}

const baseName = (p: string): string => p.split(/[\\/]/).pop() ?? p;

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
      if (p.particlesOnly === true || PARTICLE_SET.test(resultsName)) {
        return `No model is open for ${resultsName}: it holds particles only, and particles read without a model (ADR 0134).`;
      }
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
