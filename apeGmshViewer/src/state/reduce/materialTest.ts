// The material tester slice of the reducer (state/reduce.ts delegates here,
// ADR 0135 D5 and D7). A request whose job cannot be built fails at once, with why.
//
// Runs are kept for overlay, oldest first, at most MAX_MATERIAL_TEST_RUNS over all
// materials. At most one runs at a time: a new request drops the one going (the
// effect sees it leave and cancels it), so the running run is always the last.

import { materialTestJobOf } from "../materialTest.ts";
import type { MaterialTestEvent } from "../types/events/materialTest.ts";
import { MAX_MATERIAL_TEST_RUNS, type MaterialTestRun, type MaterialTestState, type State } from "../types.ts";

export const initialMaterialTest: MaterialTestState = { runs: [], form: null, nextId: 1, saving: null, notice: null };

/** Another model: its runs, form and notice leave, and the ids go on. */
export const materialTestForAnotherModel = (m: MaterialTestState): MaterialTestState =>
  m.runs.length || m.form || m.saving !== null || m.notice !== null ? { ...m, runs: [], form: null, saving: null, notice: null } : m;

/** The runs without `drop`; a save of a run that leaves is forgotten. */
function without(m: MaterialTestState, drop: (r: MaterialTestRun) => boolean): MaterialTestState {
  const runs = m.runs.filter((r) => !drop(r));
  if (runs.length === m.runs.length) return m;
  const saving = m.saving !== null && runs.some((r) => r.id === m.saving) ? m.saving : null;
  return { ...m, runs, saving };
}

export function reduceMaterialTest(s: State, e: MaterialTestEvent): MaterialTestState {
  const m = s.materialTest;
  switch (e.type) {
    case "materialTestRequested": {
      const overrides = e.overrides ?? [];
      const answer = materialTestJobOf(s, e.decl, e.protocol, overrides);
      const error = "error" in answer ? answer.error : null;
      const run: MaterialTestRun = {
        id: m.nextId,
        decl: e.decl,
        type: s.decls[e.decl]?.type ?? "",
        protocol: e.protocol,
        overrides,
        stopped: null,
        status: error === null ? "running" : "failed",
        error,
        build: null,
        curves: null,
        points: null,
      };
      const kept = without(m, (r) => r.status === "running");
      const runs = [...kept.runs, run].slice(-MAX_MATERIAL_TEST_RUNS);
      const saving = kept.saving !== null && runs.some((r) => r.id === kept.saving) ? kept.saving : null;
      return { ...kept, nextId: m.nextId + 1, runs, saving, notice: null };
    }
    case "materialTestFinished": {
      const i = m.runs.findIndex((r) => r.id === e.id);
      const run = m.runs[i];
      if (!run || run.status !== "running") return m;
      const ok = e.curves !== null && e.error === null;
      const done: MaterialTestRun = {
        ...run,
        status: ok ? "ready" : "failed",
        error: ok ? null : (e.error ?? "the run returned no curve"),
        build: e.build,
        curves: ok ? e.curves : null,
        points: ok ? e.points : null,
        stopped: ok ? (e.stopped ?? null) : null,
      };
      return { ...m, runs: m.runs.map((r, k) => (k === i ? done : r)) };
    }
    case "materialTestCleared":
      return m.runs.length || m.notice !== null ? { ...m, runs: [], saving: null, notice: null } : m;
    case "materialTestRemoved":
      return without(m, (r) => r.id === e.id);
    case "materialTestFormSet":
      return e.form === m.form ? m : { ...m, form: e.form };
    case "materialTestSaveRequested": {
      const run = m.runs.find((r) => r.id === e.id);
      if (!run || run.status !== "ready" || m.saving !== null) return m;
      return { ...m, saving: e.id, notice: null };
    }
    case "materialTestSaved":
      return m.saving === e.id ? { ...m, saving: null, notice: e.notice } : m;
  }
}
