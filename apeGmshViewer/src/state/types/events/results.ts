// The results slice's events: its part of the store's Event union (types/event.ts).
// Not re-exported by src/state/types.ts, which exports only the whole union.

import type { ResultStepStatus, ResultsInfo } from "../results.ts";

export type ResultsEvent =
  | { type: "setResultStep"; step: number }
  /** V4b: a results file was read (metadata only); the effects send it after `fileOpened` */
  | { type: "resultsLoaded"; info: ResultsInfo; sizeBytes: number; readMs: number; warnings: string[] }
  /** V4b: the effects started (`loading`) or finished fetching the current step; they drop a reply that is not the latest */
  | { type: "resultStep"; status: ResultStepStatus };
