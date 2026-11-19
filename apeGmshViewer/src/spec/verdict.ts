// Whether a shot's view is on screen, still coming, or refused (ADR 0128
// D2, INV-4), from the store's state and what the viewport has applied.
// Pure: the render mode (renderer/agentRender.ts) polls it, and the tests
// run it on states built from the fixtures.
//
// A refusal carries the app's own reason: the sentence the legend or the
// banner gives a human for the same view. There is no fallback: a contour
// whose whole-history range could not be read is refused, never drawn at
// the step's range under the history's name.

import { combinationRequestOf, contourView, deformView, userResultRequestOf } from "../state/selectors.ts";
import type { State } from "../state/types.ts";
import type { ShotPlan } from "./compile.ts";
import type { SpecShot } from "./spec.ts";

export type Verdict =
  | { kind: "shown" }
  /** not on screen yet; `why` is the app's sentence when the view says it cannot be drawn (it may still be loading) */
  | { kind: "waiting"; why: string | null; loading: boolean }
  | { kind: "refused"; reason: string };

/** What the viewport shows now: the contour colours and the deformed positions of the state's keys (renderer/agentRender.ts). */
export interface OnScreen {
  contour: boolean;
  deform: boolean;
}

export function viewVerdict(s: State, shot: SpecShot, plan: ShotPlan, on: OnScreen): Verdict {
  // VU7b-2: a User Combination being evaluated, or its user file being read, is still coming.
  const loading =
    s.resultStep?.status === "loading" ||
    s.artifacts.results?.status === "opened" ||
    s.artifacts.model?.status === "opened" ||
    (s.phase.at?.kind === "combination" &&
      (s.user?.status === "reading" || (s.combination === null ? combinationRequestOf(s) !== null : s.combination.status === "loading"))) ||
    // VU16: a user result being read, or its user file, is still coming (a read for another key is the one asked for next).
    (s.visibility.colourBy === "contour" && s.contour.source === "user" && (s.user?.status === "reading" || userResultLoading(s)));
  if (plan.stage && s.resultStep?.status === "failed") {
    return { kind: "refused", reason: `step ${plan.step} of stage ${JSON.stringify(plan.stage.name)} could not be read: ${s.resultStep.error ?? "unknown error"}` };
  }
  if (s.clip.notice) return { kind: "refused", reason: `the section planes were refused: ${s.clip.notice}` };
  // #2664: over translucent surfaces the ghost lies under the mesh's own blended triangles,
  // which differ from frame to frame (renderer/deform.ts), so a still of it is not reproducible.
  if (shot.deform?.ghost && s.visibility.opacity < 1) {
    return { kind: "refused", reason: `the reference ghost is not drawn over translucent surfaces (surface opacity ${s.visibility.opacity}): a still of it is not reproducible` };
  }
  let why: string | null = null;
  let ready = !loading;
  if (shot.colour_by === "contour") {
    const v = contourView(s);
    if (v.state === "unavailable" || v.state === "refused" || v.state === "not-yet") {
      why = `the contour is not drawn: ${v.reason}`;
      ready = false;
    } else if (v.state === "ready" || v.state === "element") {
      const scan = s.contour.scan;
      if (shot.contour?.range.mode === "history") {
        if (scan?.key === v.keys.scan && scan.status === "failed") {
          return { kind: "refused", reason: `the contour's whole-history range could not be read: ${scan.error ?? "unknown error"}` };
        }
        if (!(scan?.key === v.keys.scan && scan.status === "done")) ready = false;
      }
      if (!v.stepMeasured || !on.contour) ready = false;
    } else if (v.state === "combination" || v.state === "difference" || v.state === "user") {
      // One state: no step to measure and no history to scan; shown once its colours are applied.
      if (!on.contour) ready = false;
    } else ready = false;
  }
  if (shot.deform) {
    const v = deformView(s);
    if (v.state === "unavailable" || v.state === "refused") {
      // The banner's own sentence for a refused shape (selectors `deformNotice`).
      why = why ?? (v.state === "refused" ? `Deformed shape not drawn: ${v.reason}.` : `the deformed shape is not drawn: ${v.reason}`);
      ready = false;
    } else if (v.state === "off" || v.state === "loading" || !on.deform) ready = false;
  }
  return ready ? { kind: "shown" } : { kind: "waiting", why, loading };
}

/** VU16: the user result asked for has no answer under its key yet. */
function userResultLoading(s: State): boolean {
  const want = userResultRequestOf(s);
  if (want === null) return false;
  const r = s.userResult;
  return r === null || r.key !== want.key || r.status === "loading";
}
