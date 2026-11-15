// The results slice's events: its part of the store's Event union (types/event.ts).
// Not re-exported by src/state/types.ts, which exports only the whole union.

import type { ResultStepStatus, ResultsInfo } from "../results.ts";
import type { ParticleFrameInfo, ParticlesState, ParticleStageInfo } from "../state.ts";

export type ResultsEvent =
  | { type: "setResultStep"; step: number }
  /** V4b: a results file was read (metadata only); the effects send it after `fileOpened` */
  | { type: "resultsLoaded"; info: ResultsInfo; sizeBytes: number; readMs: number; warnings: string[] }
  /** V4b: the effects started (`loading`) or finished fetching the current step; they drop a reply that is not the latest */
  | { type: "resultStep"; status: ResultStepStatus }
  /** ADR 0134 D5: the effects started reading the particles of `path` (the `.ladruno`, or the seeded set) */
  | { type: "particlesOpened"; source: ParticlesState["source"]; path: string }
  /** the particle index as read (metadata only; the effects send it after `particlesOpened` for the same path) */
  | { type: "particlesLoaded"; source: ParticlesState["source"]; path: string; stages: ParticleStageInfo[]; unread: ParticlesState["unread"] }
  | { type: "particlesFailed"; path: string; error: string }
  /** the set holds no particles any more (another set, or the results closed) */
  | { type: "particlesClosed" }
  | { type: "setParticleStage"; stage: number }
  | { type: "setParticleOutput"; output: number }
  /** colour by one component of one field the stage offers; null: one plain colour */
  | { type: "setParticleColour"; colour: { field: string; component: number } | null }
  | { type: "setParticleGrid"; on: boolean }
  | { type: "setParticlesOn"; on: boolean }
  /** the effects pushed (`ready`), are reading (`loading`) or could not read (`failed`) the frame the state names */
  | { type: "particleFrame"; frame: ParticleFrameInfo };
