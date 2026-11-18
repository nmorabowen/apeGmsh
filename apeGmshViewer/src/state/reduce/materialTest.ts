// The material tester slice of the reducer (state/reduce.ts delegates here,
// ADR 0135 D5). A request whose job cannot be built fails at once, with why.

import { materialTestJobOf } from "../materialTest.ts";
import type { MaterialTestEvent } from "../types/events/materialTest.ts";
import type { MaterialTestState, State } from "../types.ts";

export const initialMaterialTest: MaterialTestState = { run: null, nextId: 1 };

/** Another model: its run leaves, and the ids go on. */
export const materialTestForAnotherModel = (m: MaterialTestState): MaterialTestState => (m.run ? { ...m, run: null } : m);

export function reduceMaterialTest(s: State, e: MaterialTestEvent): MaterialTestState {
  const m = s.materialTest;
  switch (e.type) {
    case "materialTestRequested": {
      const answer = materialTestJobOf(s, e.decl, e.protocol);
      const error = "error" in answer ? answer.error : null;
      return {
        nextId: m.nextId + 1,
        run: {
          id: m.nextId,
          decl: e.decl,
          type: s.decls[e.decl]?.type ?? "",
          protocol: e.protocol,
          status: error === null ? "running" : "failed",
          error,
          build: null,
          curves: null,
        },
      };
    }
    case "materialTestFinished": {
      const run = m.run;
      if (!run || run.id !== e.id || run.status !== "running") return m;
      const ok = e.curves !== null && e.error === null;
      return {
        ...m,
        run: { ...run, status: ok ? "ready" : "failed", error: ok ? null : (e.error ?? "the run returned no curve"), build: e.build, curves: ok ? e.curves : null },
      };
    }
    case "materialTestCleared":
      return materialTestForAnotherModel(m);
  }
}
