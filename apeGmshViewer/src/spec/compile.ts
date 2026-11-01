// The spec compiler (ADR 0128 INV-2): a validated shot and the description
// of the open set (./describe.ts) become the reducer events that set the
// view, plus the camera to ask for once the view is drawn. Pure; the render
// mode (renderer/agentRender.ts) dispatches the plan, and an offer (slice 5)
// will apply the same plan, so a still and an applied view come from one
// compile.
//
// D2 (fail closed): a shot names a stage, a step, a field, a component or a
// group the set does not have, and the shot is refused with a sentence that
// names it. There is no fallback: never the nearest stage, the step before
// or another component. What only the drawn view can tell (a deformed shape
// the banner refuses, an element field that does not join the mesh, a
// contour that never reaches the screen) is refused by the render mode with
// the app's own reason.
//
// Every shot starts from the same base view (BASE_EVENTS), so a shot never
// inherits a choice from the shot before it: two runs of one spec, in any
// order of its shots, set the same state for each.

import { MAGNITUDE } from "../state/contour.ts";
import type { CameraRequest, DeclPath, Event, PhaseKey } from "../state/types.ts";
import type { DescribedField, DescribedStage, SetDescription } from "./describe.ts";
import { SPEC_MAGNITUDE } from "./describe.ts";
import type { SpecClip, SpecDeform, SpecShot } from "./spec.ts";

export interface ShotPlan {
  phase: PhaseKey;
  /** the results stage and step drawn, or null on the mesh */
  stage: { index: number; id: string; name: string } | null;
  step: number | null;
  /** state events, in order: the base view, then the shot's choices */
  events: Event[];
  /** section planes: added by the render mode, which reads back each plane's id */
  clip: SpecClip[];
  /** the drawn contour, as the store names it; null when the shot draws none */
  contour: { field: string; component: string; label: string; source: "nodal" | "element" } | null;
  deform: SpecDeform | null;
  /** applied after the view is drawn: a fit frames what is drawn (the deformed shape, an isolated group) */
  camera: CameraRequest;
}

export type Compiled = { ok: true; plan: ShotPlan } | { ok: false; reason: string };

/** The view every shot starts from: nothing selected, every group shown, no deformed shape, coloured by group, the contour defaults, no gizmo. */
export const BASE_EVENTS: readonly Event[] = [
  { type: "clearSelection" },
  { type: "showAll" },
  { type: "setDeform", on: false },
  { type: "setDeformScale", scale: 1 },
  { type: "setReferenceGhost", on: false },
  { type: "setColourBy", by: "group" },
  { type: "setContourRange", range: "history" },
  { type: "setContourCmap", cmap: "jet" },
  { type: "setContourGauss", mode: "cell" },
  { type: "setLineDiagram", on: false },
  // A plane's gizmo is a handle for the hand; a still shows the cut, not the handle.
  { type: "setShowGizmos", on: false },
];

const list = (xs: readonly string[]): string => (xs.length ? xs.join(", ") : "none");

function stageOf(shot: SpecShot, d: SetDescription): { ok: true; stage: DescribedStage } | { ok: false; reason: string } {
  const want = shot.stage!;
  if (!d.files.results) return { ok: false, reason: `the set has no results file, so it has no stage ${JSON.stringify(want)}` };
  if (d.pairing.state !== "paired") return { ok: false, reason: `the results are not drawn on this model: ${d.pairing.reason ?? d.pairing.state}` };
  const names = d.stages.map((s) => JSON.stringify(s.name));
  if (typeof want === "number") {
    const st = d.stages[want];
    return st ? { ok: true, stage: st } : { ok: false, reason: `the results have ${d.stages.length} stages (indices 0 to ${d.stages.length - 1}); stage ${want} is not one of them` };
  }
  const hits = d.stages.filter((s) => s.name === want);
  if (hits.length === 0) return { ok: false, reason: `the results have no stage named ${JSON.stringify(want)}; their stages are ${list(names)}` };
  if (hits.length > 1) return { ok: false, reason: `${hits.length} stages are named ${JSON.stringify(want)} (indices ${hits.map((s) => s.index).join(", ")}); name the stage by its index` };
  return { ok: true, stage: hits[0]! };
}

function stepOf(shot: SpecShot, st: DescribedStage): { ok: true; step: number } | { ok: false; reason: string } {
  const label = JSON.stringify(st.name || st.id);
  if (st.steps === 0) return { ok: false, reason: `stage ${label} records no step` };
  const k = shot.step === "first" ? 0 : shot.step === "last" ? st.steps - 1 : shot.step;
  if (k >= st.steps) return { ok: false, reason: `stage ${label} has ${st.steps} steps (0 to ${st.steps - 1}); step ${k} is not one of them` };
  return { ok: true, step: k };
}

function fieldOf(shot: SpecShot, st: DescribedStage): { ok: true; field: DescribedField; key: string; label: string } | { ok: false; reason: string } {
  const c = shot.contour!;
  const label = JSON.stringify(st.name || st.id);
  const named = st.fields.filter((f) => f.name === c.field && (c.source === null || f.source === c.source));
  const kinds = c.source ? `${c.source} field` : "field";
  if (named.length === 0) {
    const offered = st.fields.filter((f) => c.source === null || f.source === c.source).map((f) => `${f.name} (${f.source})`);
    return { ok: false, reason: `stage ${label} has no ${kinds} ${JSON.stringify(c.field)}; its fields are ${list(offered)}` };
  }
  if (named.length > 1) return { ok: false, reason: `stage ${label} has a nodal and an element field named ${JSON.stringify(c.field)}; set contour.source` };
  const f = named[0]!;
  // A component by its key (the dataset name, or "magnitude") or by the legend's label; one component only.
  const hits = f.components.filter((x) => x.key === c.component || x.label === c.component);
  if (hits.length === 0) {
    const why = f.source === "nodal" && c.component === SPEC_MAGNITUDE && f.refused ? `: the magnitude is not offered (${f.refused})` : "";
    return {
      ok: false,
      reason: `${f.source} field ${JSON.stringify(f.name)} of stage ${label} has no component ${JSON.stringify(c.component)}${why}; its components are ${list(f.components.map((x) => (x.key === x.label ? x.key : `${x.key} (${x.label})`)))}`,
    };
  }
  if (hits.length > 1) return { ok: false, reason: `component ${JSON.stringify(c.component)} names ${hits.length} components of ${JSON.stringify(f.name)} (${hits.map((x) => x.key).join(", ")}); name it by its key` };
  const hit = hits[0]!;
  return { ok: true, field: f, key: f.source === "nodal" && hit.key === SPEC_MAGNITUDE ? MAGNITUDE : hit.key, label: hit.label };
}

/** Compile one validated shot against the open set (INV-2: the one compiler D4, D5 and D6 share). */
export function compileShot(shot: SpecShot, d: SetDescription): Compiled {
  const events: Event[] = [...BASE_EVENTS];
  let phase: PhaseKey = { kind: "mesh" };
  let stage: ShotPlan["stage"] = null;
  let step: number | null = null;
  let contour: ShotPlan["contour"] = null;
  if (shot.stage !== null) {
    const s = stageOf(shot, d);
    if (!s.ok) return s;
    const k = stepOf(shot, s.stage);
    if (!k.ok) return k;
    stage = { index: s.stage.index, id: s.stage.id, name: s.stage.name };
    step = k.step;
    phase = { kind: "results", stage: s.stage.index, step: 0 };
    events.push({ type: "setPhase", at: phase });
    if (step > 0) events.push({ type: "setResultStep", step });
    phase = { kind: "results", stage: s.stage.index, step };
  } else {
    events.push({ type: "setPhase", at: phase });
  }
  events.push({ type: "setColourBy", by: shot.colour_by });
  if (shot.contour && shot.stage !== null) {
    const f = fieldOf(shot, d.stages[stage!.index]!);
    if (!f.ok) return f;
    const c = shot.contour;
    contour = { field: f.field.name, component: f.key, label: f.label, source: f.field.source };
    events.push({ type: "setContourField", field: f.field.name, component: f.key, source: f.field.source });
    if (f.field.source === "element") events.push({ type: "setContourGauss", mode: c.gauss });
    events.push(c.range.mode === "fixed" ? { type: "setContourRange", range: "fixed", fixed: { min: c.range.min, max: c.range.max } } : { type: "setContourRange", range: c.range.mode });
    events.push({ type: "setContourCmap", cmap: c.cmap });
  } else if (shot.contour) {
    // No stage: the render mode refuses with the app's reason (a contour colours a results step).
    contour = { field: shot.contour.field, component: shot.contour.component, label: shot.contour.component, source: shot.contour.source ?? "nodal" };
  }
  if (shot.deform) {
    events.push({ type: "setDeform", on: true }, { type: "setDeformScale", scale: shot.deform.scale }, { type: "setReferenceGhost", on: shot.deform.ghost });
  }
  if (shot.isolate.length) {
    const known = new Set(d.groups.map((g) => g.path));
    const missing = shot.isolate.filter((p) => !known.has(p));
    if (missing.length) return { ok: false, reason: `the model has no physical group ${missing.join(", ")}; its groups are ${list(d.groups.map((g) => g.path))}` };
    events.push({ type: "isolate", decls: shot.isolate as DeclPath[] });
  }
  const camera: CameraRequest = typeof shot.camera === "string" ? shot.camera : { eye: shot.camera.eye, target: shot.camera.target, up: shot.camera.up };
  return { ok: true, plan: { phase, stage, step, events, clip: shot.clip, contour, deform: shot.deform, camera } };
}
