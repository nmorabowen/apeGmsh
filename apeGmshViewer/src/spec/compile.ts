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
  /** the results stage and step drawn, or null on the mesh (and on a User Combination) */
  stage: { index: number; id: string; name: string } | null;
  /** VU7b-2: the User Combination shown instead of a stage, or null */
  combination: string | null;
  step: number | null;
  /** state events, in order: the base view, then the shot's choices */
  events: Event[];
  /** section planes: added by the render mode, which reads back each plane's id */
  clip: SpecClip[];
  /** the drawn contour, as the store names it; null when the shot draws none. VU16: `user` is a user result (`field` its name) */
  contour: { field: string; component: string; label: string; source: "nodal" | "element" | "user" } | null;
  /** the shot's deformed shape; `scale` null is G3's auto scale, which the render mode dispatches once the step's displacement is read */
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
  // VU10: no difference unless the shot compares (the compare set itself stays open for the next shot that names it).
  { type: "setDifference", on: false },
];

/**
 * VU10: the row of the open study labelled `label` (state/selectors.ts over
 * `variants.study`), whether it is already the compare set, or why a shot
 * cannot compare with it.
 */
export type VariantOf = (label: string) => { ok: true; row: { index: number; label: string }; open: boolean } | { ok: false; reason: string };

const list = (xs: readonly string[]): string => (xs.length ? xs.join(", ") : "none");

function stageOf(shot: SpecShot, d: SetDescription): { ok: true; stage: DescribedStage } | { ok: false; reason: string } {
  const want = shot.stage!;
  if (!d.files.results) return { ok: false, reason: `the set has no results file, so it has no stage ${JSON.stringify(want)}` };
  if (d.pairing.state !== "paired") return { ok: false, reason: `the results are not drawn on this model: ${d.pairing.reason ?? d.pairing.state}` };
  const names = d.stages.map((s) => (s.name === s.programName ? JSON.stringify(s.name) : `${JSON.stringify(s.name)} (program stage ${JSON.stringify(s.programName)})`));
  if (typeof want === "number") {
    const st = d.stages[want];
    return st ? { ok: true, stage: st } : { ok: false, reason: `the results have ${d.stages.length} stages (indices 0 to ${d.stages.length - 1}); stage ${want} is not one of them` };
  }
  // W4d #2843: a stage answers to its program name and to its shown (display) name, so a
  // spec keeps resolving after a `.ladruno` is renamed for display.
  const byProgram = d.stages.filter((s) => s.programName === want);
  const byShown = d.stages.filter((s) => s.name === want);
  const hits = [...new Set([...byProgram, ...byShown])];
  if (hits.length === 0) return { ok: false, reason: `the results have no stage named ${JSON.stringify(want)}; their stages are ${list(names)}` };
  const shownElsewhere = byProgram.length === 1 ? byShown.find((s) => s !== byProgram[0]) : undefined;
  if (shownElsewhere) {
    return {
      ok: false,
      reason: `${JSON.stringify(want)} is the program name of stage ${byProgram[0]!.index} and the shown name of stage ${shownElsewhere.index}; name the stage by its index`,
    };
  }
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

/**
 * VU7b-2: what User Combination `name` colours by (state/selectors.ts
 * `combinationFieldsOf` on the open set), or why it colours by nothing.
 */
export type CombinationFields = (name: string) => { ok: true; fields: readonly DescribedNodal[] } | { ok: false; reason: string };
type DescribedNodal = { name: string; components: readonly { key: string; label: string }[] };

/**
 * A combination shot's field and component, as `fieldOf` finds a stage's: the
 * component by its key (`DISPLACEMENT/Ux`), by the legend's label (`x`), or as
 * the case spells it (`Ux`, the key's part after the field). One only.
 */
function combinationFieldOf(c: NonNullable<SpecShot["contour"]>, name: string, of: CombinationFields): { ok: true; field: string; key: string; label: string } | { ok: false; reason: string } {
  const label = `User Combination ${JSON.stringify(name)}`;
  const r = of(name);
  if (!r.ok) return { ok: false, reason: `${label} colours by nothing: ${r.reason}` };
  const f = r.fields.find((x) => x.name === c.field);
  if (!f) return { ok: false, reason: `${label} has no field ${JSON.stringify(c.field)}; its fields are ${list(r.fields.map((x) => x.name))}` };
  const hits = f.components.filter((x) => x.key === c.component || x.label === c.component || x.key === `${f.name}/${c.component}`);
  if (hits.length === 0) {
    return { ok: false, reason: `field ${JSON.stringify(f.name)} of ${label} has no component ${JSON.stringify(c.component)}; its components are ${list(f.components.map((x) => `${x.key} (${x.label})`))}` };
  }
  if (hits.length > 1) return { ok: false, reason: `component ${JSON.stringify(c.component)} names ${hits.length} components of ${JSON.stringify(f.name)} (${hits.map((x) => x.key).join(", ")}); name it by its key` };
  return { ok: true, field: f.name, key: hits[0]!.key, label: hits[0]!.label };
}

/**
 * VU16: user result `name` of the set's `.user.h5` as a contour can take it
 * (state/selectors.ts `userResultPickOf` on the open set): its components in
 * `DATA` order, or why it cannot be drawn (absent, refused, per state, not on
 * nodes). A stale one is found here; the render mode refuses it with A6's reason.
 */
export type UserResultOf = (name: string) => { ok: true; components: readonly string[] } | { ok: false; reason: string };

/**
 * Compile one validated shot against the open set (INV-2: the one compiler D4,
 * D5 and D6 share). VU7b-2: `combinationFields` finds a combination shot's
 * component; without it, a combination shot is refused. VU16: `userResultOf`
 * finds a user-result shot's component; without it, such a shot is refused.
 */
export function compileShot(shot: SpecShot, d: SetDescription, combinationFields?: CombinationFields, variantOf?: VariantOf, userResultOf?: UserResultOf): Compiled {
  const events: Event[] = [...BASE_EVENTS];
  let phase: PhaseKey = { kind: "mesh" };
  let stage: ShotPlan["stage"] = null;
  let step: number | null = null;
  let contour: ShotPlan["contour"] = null;
  const combination = shot.combination ?? null;
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
  } else if (combination !== null) {
    // VU7b-2: the reducer refuses a combination the user file does not list, with its reason (the render mode reports it).
    phase = { kind: "combination", name: combination };
    events.push({ type: "setPhase", at: phase });
  } else {
    events.push({ type: "setPhase", at: phase });
  }
  events.push({ type: "setColourBy", by: shot.colour_by });
  if (shot.contour?.source === "user") {
    // VU16: the spec validator held a user-result shot to the mesh phase: no stage, step, combination, compare or shape.
    const c = shot.contour;
    const label = `user result ${JSON.stringify(c.field)}`;
    if (!userResultOf) return { ok: false, reason: `this view has no user file to find ${label} in` };
    const r = userResultOf(c.field);
    if (!r.ok) return { ok: false, reason: `${label} cannot be drawn: ${r.reason}` };
    if (!r.components.includes(c.component)) return { ok: false, reason: `${label} has no component ${JSON.stringify(c.component)}; its components are ${list([...r.components])}` };
    contour = { field: c.field, component: c.component, label: c.component, source: "user" };
    // The reducer refuses a name the user file does not list too (setContourField).
    events.push({ type: "setContourField", field: c.field, component: c.component, source: "user" });
    events.push(c.range.mode === "fixed" ? { type: "setContourRange", range: "fixed", fixed: { min: c.range.min, max: c.range.max } } : { type: "setContourRange", range: c.range.mode });
    events.push({ type: "setContourCmap", cmap: c.cmap });
  } else if (shot.contour && combination !== null) {
    const c = shot.contour;
    if (c.source === "element") return { ok: false, reason: `User Combination ${JSON.stringify(combination)} colours by a nodal component; contour.source is "element"` };
    if (!combinationFields) return { ok: false, reason: `this view has no user file to find User Combination ${JSON.stringify(combination)} in` };
    const f = combinationFieldOf(c, combination, combinationFields);
    if (!f.ok) return f;
    contour = { field: f.field, component: f.key, label: f.label, source: "nodal" };
    events.push({ type: "setContourField", field: f.field, component: f.key, source: "nodal" });
    events.push(c.range.mode === "fixed" ? { type: "setContourRange", range: "fixed", fixed: { min: c.range.min, max: c.range.max } } : { type: "setContourRange", range: c.range.mode });
    events.push({ type: "setContourCmap", cmap: c.cmap });
  } else if (shot.contour && shot.stage !== null) {
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
  // VU10: the window variant minus the compare variant (the spec validator held it to a nodal contour of a stage).
  const compare = shot.compare ?? null;
  if (compare !== null) {
    if (!variantOf) return { ok: false, reason: `this view has no variants manifest to find variant ${JSON.stringify(compare)} in` };
    const v = variantOf(compare);
    if (!v.ok) return v;
    if (!v.open) events.push({ type: "variantCompare", row: v.row });
    events.push({ type: "setDifference", on: true });
  }
  if (shot.deform) {
    events.push({ type: "setDeform", on: true });
    // G3 (VA6c, #2765): a scale left out stays at the base view's 1 here; the auto scale needs the step's
    // displacement, so the render mode dispatches it once that is read (`plan.deform.scale` null says so).
    if (shot.deform.scale !== null) events.push({ type: "setDeformScale", scale: shot.deform.scale });
    events.push({ type: "setReferenceGhost", on: shot.deform.ghost });
  }
  if (shot.isolate.length) {
    const known = new Set(d.groups.map((g) => g.path));
    const missing = shot.isolate.filter((p) => !known.has(p));
    if (missing.length) return { ok: false, reason: `the model has no physical group ${missing.join(", ")}; its groups are ${list(d.groups.map((g) => g.path))}` };
    events.push({ type: "isolate", decls: shot.isolate as DeclPath[] });
  }
  const camera: CameraRequest = typeof shot.camera === "string" ? shot.camera : { eye: shot.camera.eye, target: shot.camera.target, up: shot.camera.up };
  return { ok: true, plan: { phase, stage, combination, step, events, clip: shot.clip, contour, deform: shot.deform, camera } };
}
