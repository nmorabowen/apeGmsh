// The capture harness's reset between steps (the 2026-11-24 architecture review, finding 3; #3079 was one case).
//
// Each `capture*` step used to start from whatever the step before it left: a phase at the last step, a contour field
// a file refuses, a deformed shape, a camera turned away, or (after the workspace step) another model's path. A still
// then depended on the order of the sequence, and a step moved or skipped changed stills it does not own. runCapture
// calls `resetView` before every step, so each starts from the same known view: the one main opened.

import { defaultPick } from "../../state/contour.ts";
import { defaultOpacity } from "../../state/reduce.ts";
import { initialCracks, initialHinges } from "../../state/reduce/overlays.ts";
import { initialClip } from "../../state/clip.ts";
import { initialLineDiagramState } from "../../state/lineDiagram.ts";
import { initialContour, initialDeform } from "../../state/reduce/initial.ts";
import { WINDOW_KINDS } from "../../state/types.ts";
import { until, type HarnessContext } from "./context.ts";

/**
 * Put the view back to the capture's baseline (`ctx.baseline`, recorded once main opened the model):
 * - the model main opened, read again when another path is open (the workspace step's set switch) or its read is not ready;
 * - the capture window's own size;
 * - the mesh phase, nothing selected, no stage card selected, no window open, nothing hidden;
 * - coloured by group at the model's opening opacity, undeformed at scale 1 with no ghost;
 * - the contour at its defaults (whole-history range, cell draw mode, jet, the declared units) with the first results
 *   stage's default pick (state/contour.ts `defaultPick`), so a step that colours by contour without picking starts
 *   from a pick the file offers and not from a field a step before it chose;
 * - line diagram, cracks, hinges, nodes, Gauss points, principal axes and vectors off; frames not extruded, joint
 *   offsets at their opening default; no section plane, and the planes panel's switches at their defaults;
 * - the camera at the pose it opened on.
 * Throws when the baseline is missing, the model does not read again, or the window does not take its size.
 */
export async function resetView(ctx: HarnessContext): Promise<void> {
  const { store, effects, bridge } = ctx;
  const b = ctx.baseline;
  if (!b) throw new Error("resetView: the capture recorded no baseline view");
  const model = store.get().artifacts.model;
  if (model?.path !== b.model || model.status !== "ready") {
    if (!(await effects.open(b.model))) throw new Error(`resetView: the model ${b.model} did not read again: ${store.get().artifacts.model?.error}`);
  }
  if (typeof bridge.captureResize === "function" && typeof window !== "undefined" && (window.innerWidth !== b.width || window.innerHeight !== b.height)) {
    await bridge.captureResize(b.width, b.height);
    await until(`resetView: the capture window to take ${b.width}x${b.height}`, () => window.innerWidth === b.width && window.innerHeight === b.height, 5000);
  }
  const s = store.get();
  if (s.phase.axis.some((k) => k.kind === "mesh")) store.dispatch({ type: "setPhase", at: { kind: "mesh" } });
  store.dispatch({ type: "clearSelection" });
  store.dispatch({ type: "selectStage", index: null });
  for (const kind of WINDOW_KINDS) store.dispatch({ type: "closeWindow", window: kind });
  store.dispatch({ type: "showAll" });
  store.dispatch({ type: "setColourBy", by: "group" });
  store.dispatch({ type: "setOpacity", value: s.mesh ? defaultOpacity(s.mesh) : 1 });
  store.dispatch({ type: "setDeform", on: initialDeform.on });
  store.dispatch({ type: "setDeformScale", scale: initialDeform.scale });
  store.dispatch({ type: "setReferenceGhost", on: initialDeform.ghost });
  store.dispatch({ type: "setContourRange", range: initialContour.range });
  store.dispatch({ type: "setContourGauss", mode: initialContour.gauss });
  store.dispatch({ type: "setContourCmap", cmap: initialContour.cmap });
  store.dispatch({ type: "setContourUnits", target: store.get().contour.declaredUnits ?? "as_stored" });
  const pick = (s.results?.stages ?? []).map((st) => defaultPick(st)).find((p) => p !== null);
  if (pick) store.dispatch({ type: "setContourField", ...pick });
  store.dispatch({ type: "setLineDiagram", on: initialLineDiagramState.on });
  store.dispatch({ type: "setLineDiagramRange", range: initialLineDiagramState.range });
  store.dispatch({ type: "setLineDiagramCmap", cmap: initialLineDiagramState.cmap });
  store.dispatch({ type: "setCracks", on: initialCracks.on, minOpening: initialCracks.minOpening, face: initialCracks.face, style: initialCracks.style });
  store.dispatch({ type: "setHinges", on: initialHinges.on, colorBy: initialHinges.colorBy, element: initialHinges.element, end: initialHinges.end });
  store.dispatch({ type: "setNodes", on: false });
  store.dispatch({ type: "setGaussPoints", on: false });
  store.dispatch({ type: "setPrincipal", on: false });
  store.dispatch({ type: "setVectors", on: false });
  store.dispatch({ type: "toggleExtrudeFrames", on: false });
  store.dispatch({ type: "toggleShowOffsets", on: Boolean(s.mesh?.frameExtrusion?.hasJointOffsets) });
  for (const p of store.get().clip.planes) store.dispatch({ type: "removeClipPlane", id: p.id });
  store.dispatch({ type: "setApplyCuts", on: initialClip.applyCuts });
  store.dispatch({ type: "setCutFace", on: initialClip.cutFace });
  store.dispatch({ type: "setShowGizmos", on: initialClip.showGizmos });
  store.dispatch({ type: "setCamera", to: b.camera });
}

/**
 * Pick the shown results stage's default contour (state/contour.ts `defaultPick`: its first nodal field's first
 * component, the magnitude when offered), as a fresh open colours that stage. For a step that colours by contour
 * without choosing a field: the pick `resetView` leaves is the first stage's, which another stage may not offer.
 * False when no results stage is shown or it offers no field.
 */
export function pickStageDefault(store: HarnessContext["store"]): boolean {
  const s = store.get();
  const at = s.phase.at;
  const pick = at?.kind === "results" ? defaultPick(s.results?.stages[at.stage]) : null;
  if (!pick) return false;
  store.dispatch({ type: "setContourField", ...pick });
  return true;
}
