// The material tester's events: its part of the store's Event union (types/event.ts).
// Not re-exported by src/state/types.ts, which exports only the whole union.

import type { MaterialTestForm, MaterialTestProtocol, MaterialTestRun } from "../materialTest.ts";
import type { DeclPath } from "../model.ts";

export type MaterialTestEvent =
  /** Test this uniaxial material; a run still going is superseded. */
  | { type: "materialTestRequested"; decl: DeclPath; protocol: MaterialTestProtocol }
  /** The effects' answer for run `id`. An answer for any other run is ignored. */
  | {
      type: "materialTestFinished";
      id: number;
      build: string | null;
      curves: MaterialTestRun["curves"];
      points: MaterialTestRun["points"];
      error: string | null;
    }
  /** Open, edit or (null) close the tester's form in the materials window. */
  | { type: "materialTestFormSet"; form: MaterialTestForm | null }
  /** Forget the run; one still going is cancelled. */
  | { type: "materialTestCleared" };
