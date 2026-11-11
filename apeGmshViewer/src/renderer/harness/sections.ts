// The capture harness (VH1, #2865): the section-plane stills (D7): cut-face caps, clip planes and plane gizmos.

import * as THREE from "three";
import { deformOf } from "../../state/selectors.ts";
import { tipPoint } from "../gizmo.ts";
import { zoomToward } from "../navigation.ts";
import { painted, until, type HarnessContext } from "./context.ts";
import { addAt, placeAt, removePlanes } from "./planes.ts";
import { clipMissTarget, marker, pointerAt, screenOf } from "./pointer.ts";
import { contourShown, deformShown } from "./shown.ts";

/**
 * The V4e3 cut-face stills (D7), by a fixed rule. On a model with volumetric
 * cells, on the results stage with the most steps whose deformed shape can be
 * drawn: one Z plane keeping the lower 40 % (the cut V4e1's clip-2 showed
 * hollow), its cap a flat fill (coloured by group); the same cap carrying the
 * contour; an X plane at 30 % added, the two caps trimming each other; the
 * deformed (scale 10) contoured model at the last step, cut by a Y plane at 50
 * % and an X plane at 45 % (V4e1's clip-12, now filled); and "Contour on cut
 * face" off, the same cut hollow again. On a model with no volumetric cell
 * (a shell, a frame): one Z plane at 50 % and the planes panel saying the
 * mesh gets no cap. Each still waits until the caps show what the state asks
 * for and checks their count. `<out>.caps-<n>-<what>.png`.
 */
export async function captureCaps(ctx: HarnessContext, settle: (ms?: number) => Promise<unknown>, width: number, height: number) {
  const { store, viewport, bridge } = ctx;
  const s0 = store.get();
  if (!s0.mesh) return;
  if (typeof bridge.captureResize === "function") await bridge.captureResize(width, height);
  removePlanes(ctx);
  store.dispatch({ type: "setApplyCuts", on: true });
  store.dispatch({ type: "setCutFace", on: true });
  store.dispatch({ type: "setShowGizmos", on: true });
  store.dispatch({ type: "setDeform", on: false });
  store.dispatch({ type: "setColourBy", by: "group" });
  store.dispatch({ type: "clearSelection" });
  const still = async (name: string, caps: number) => {
    const ok = () => deformShown(ctx) && (store.get().visibility.colourBy !== "contour" || contourShown(ctx)) && viewport.capsSettled() && viewport.capCount === caps;
    if (!(await until(ok))) throw new Error(`capture ${name}: ${viewport.capCount} caps (want ${caps}), settled ${viewport.capsSettled()}`);
    await settle(150);
    viewport.renderNow();
    await painted();
    await painted();
    await bridge.captureStill(name);
  };
  if (s0.mesh.counts.solidCells === 0) {
    store.dispatch({ type: "frameSelection" });
    addAt(ctx, [0, 0, 1], 0.5);
    const note = document.querySelector("#planes .planes-capnote");
    if (!note?.textContent) throw new Error("capture: the planes panel does not say this mesh gets no cap");
    await still("caps-0-no-volumetric-cells", 0);
    removePlanes(ctx);
    return;
  }
  const stages = s0.results?.stages ?? [];
  let best = -1;
  stages.forEach((st, i) => {
    store.dispatch({ type: "setPhase", at: { kind: "results", stage: i, step: 0 } });
    if (deformOf(store.get()).ok && (best < 0 || st.time.length >= stages[best]!.time.length)) best = i;
  });
  if (best < 0) return;
  const last = stages[best]!.time.length - 1;
  store.dispatch({ type: "setPhase", at: { kind: "results", stage: best, step: 0 } });
  if (last > 0) store.dispatch({ type: "setResultStep", step: last });
  store.dispatch({ type: "frameSelection" });
  const z = addAt(ctx, [0, 0, 1], 0.4);
  store.dispatch({ type: "setClipFlipped", id: z, flipped: true });
  await still("caps-1-flat-fill", 1);
  store.dispatch({ type: "setColourBy", by: "contour" });
  await still("caps-2-contour", 1);
  addAt(ctx, [1, 0, 0], 0.3);
  await still("caps-3-two-planes-trimmed", 2);
  removePlanes(ctx);
  store.dispatch({ type: "setDeform", on: true });
  store.dispatch({ type: "setReferenceGhost", on: true });
  store.dispatch({ type: "setDeformScale", scale: 10 });
  addAt(ctx, [0, 1, 0], 0.5);
  addAt(ctx, [1, 0, 0], 0.45);
  await still(`caps-4-deformed-contour-step${last + 1}`, 2);
  store.dispatch({ type: "setCutFace", on: false });
  await still("caps-5-cut-face-off", 0);
  store.dispatch({ type: "setCutFace", on: true });
  removePlanes(ctx);
  store.dispatch({ type: "setDeform", on: false });
  store.dispatch({ type: "setColourBy", by: "group" });
}

/**
 * The V4e1 section-plane stills (D7), by a fixed rule so no flag is needed:
 * on the results stage with the most steps (the last of a tie) whose
 * deformed shape can be drawn, at its last step, coloured by group,
 * undeformed: the model before any plane; one Z plane, flipped to keep the
 * lower 40 %, so the solid shows hollow from above (no caps until V4e3); an
 * X plane at 30 % added (a corner cut); the Z plane flipped back; its offset moved to
 * 75 %; the panel with the X plane's Custom normal fields open on (1, 1, 0);
 * six planes, each 10 % inside a face of the box; a 7th refused with its
 * notice. Then a click that misses cut-away geometry: a plane along the view
 * cuts away the nearer half (on a planar model shown in plan, which lies in
 * that plane, the first axis plane through the centre that cuts something
 * away), and a scripted click on an element of that half is a miss (the ring
 * marks where it clicked); the same click with "Apply cuts" off selects it.
 * Then the deformed (scale 10, ghost on), contoured
 * model cut by a Y plane (the half nearer the camera gone) and an X plane at
 * 45 %, at the first and the last step of the scrub (the shear carries more
 * of it through the X plane), and that last still at 1280 px. Plane offsets
 * avoid the mesh's own face planes, where a cut coincides with a face.
 * `<out>.clip-<n>-<what>.png`.
 */
export async function captureClip(ctx: HarnessContext, settle: (ms?: number) => Promise<unknown>, width: number, height: number) {
  const { store, viewport, bridge } = ctx;
  const s0 = store.get();
  if (!s0.results || !s0.mesh || !s0.phase.axis.some((k) => k.kind === "results")) return;
  const stages = s0.results.stages;
  const usable = (i: number) => {
    store.dispatch({ type: "setPhase", at: { kind: "results", stage: i, step: 0 } });
    return deformOf(store.get()).ok;
  };
  let best = -1;
  stages.forEach((st, i) => {
    if (usable(i) && (best < 0 || st.time.length >= stages[best]!.time.length)) best = i;
  });
  if (best < 0) return;
  if (typeof bridge.captureResize === "function") await bridge.captureResize(width, height);
  const last = stages[best]!.time.length - 1;
  removePlanes(ctx);
  store.dispatch({ type: "setApplyCuts", on: true });
  // V4e3: these are the V4e1 stills, caps off (the hollow cut they show is what the caps fill: captureCaps).
  store.dispatch({ type: "setCutFace", on: false });
  store.dispatch({ type: "setDeform", on: false });
  store.dispatch({ type: "setColourBy", by: "group" });
  store.dispatch({ type: "clearSelection" });
  store.dispatch({ type: "setPhase", at: { kind: "results", stage: best, step: 0 } });
  if (last > 0) store.dispatch({ type: "setResultStep", step: last });
  store.dispatch({ type: "frameSelection" });
  const still = async (name: string) => {
    if (!(await until(() => deformShown(ctx) && (store.get().visibility.colourBy !== "contour" || contourShown(ctx))))) throw new Error(`capture ${name}: the step did not show`);
    await settle(150);
    viewport.renderNow();
    await painted();
    await painted();
    await bridge.captureStill(name);
  };
  await still("clip-1-before");
  const z = addAt(ctx, [0, 0, 1], 0.4);
  store.dispatch({ type: "setClipFlipped", id: z, flipped: true });
  await still("clip-2-z-plane-hollow");
  const x = addAt(ctx, [1, 0, 0], 0.3);
  await still("clip-3-corner-two-planes");
  store.dispatch({ type: "setClipFlipped", id: z, flipped: false });
  await still("clip-4-z-flipped-back");
  placeAt(ctx, z, 0.75);
  await still("clip-5-z-offset-75");
  store.dispatch({ type: "setClipNormal", id: x, normal: [1, 1, 0] });
  await still("clip-6-panel-custom-normal");
  removePlanes(ctx);
  for (const n of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]] as const) addAt(ctx, n, 0.1);
  await still("clip-7-six-planes");
  store.dispatch({ type: "addClipPlane", normal: [1, 0, 1] });
  if (store.get().clip.notice === null) throw new Error("capture: a 7th plane was not refused");
  await still("clip-8-seventh-refused");

  // The pick that misses cut-away geometry. The view normal through the centre cuts away the near half of a
  // body; a planar model shown in plan lies in that plane, where nothing is cut away (#2707), so the axis
  // normals through the centre are tried next, until one cuts away something a click can reach.
  let target: ReturnType<typeof clipMissTarget> = null;
  const normals: [number, number, number][] = [[...viewport.viewNormal()] as [number, number, number], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
  for (const normal of normals) {
    removePlanes(ctx);
    store.dispatch({ type: "addClipPlane", normal });
    target = clipMissTarget(ctx);
    if (target) break;
  }
  if (!target) throw new Error("capture: no element of the cut-away half answers a click with the cuts off");
  const marker = document.createElement("div");
  marker.style.cssText =
    `position: fixed; left: ${target.x - 9}px; top: ${target.y - 9}px; width: 18px; height: 18px; border-radius: 50%;` +
    "border: 3px solid var(--accent); pointer-events: none; z-index: 10;";
  const label = document.createElement("div");
  label.style.cssText = `position: fixed; left: ${target.x + 16}px; top: ${target.y - 10}px; padding: 2px 6px; border-radius: 5px; font: 600 12px var(--font); color: var(--text); background: var(--panel); border: 1px solid var(--line); pointer-events: none; z-index: 10;`;
  document.body.append(marker, label);
  const click = () => {
    const pick = viewport.pickAt(target.x, target.y);
    store.dispatch(pick ? { type: "select", pick } : { type: "clearSelection" });
    return pick;
  };
  const miss = click();
  if (miss) throw new Error(`capture: the click on cut-away geometry picked ${miss.decl}`);
  label.textContent = "click: miss (the face hit first is cut away; no pick-through)";
  await still("clip-9-pick-miss-cut-away");
  store.dispatch({ type: "setApplyCuts", on: false });
  const hit = click();
  if (hit?.decl !== target.decl) throw new Error(`capture: with the cuts off the click picked ${hit?.decl ?? "nothing"}, not ${target.decl}`);
  label.textContent = "the same click, Apply cuts off: picked";
  await settle(900);
  await still("clip-10-pick-cuts-off");
  marker.remove();
  label.remove();
  store.dispatch({ type: "clearSelection" });
  store.dispatch({ type: "setApplyCuts", on: true });
  removePlanes(ctx);

  // The deformed, contoured (and stage-activated) model cut while scrubbing.
  store.dispatch({ type: "setPhase", at: { kind: "results", stage: best, step: 0 } });
  store.dispatch({ type: "setDeform", on: true });
  store.dispatch({ type: "setReferenceGhost", on: true });
  store.dispatch({ type: "setDeformScale", scale: 10 });
  store.dispatch({ type: "setColourBy", by: "contour" });
  addAt(ctx, [0, 1, 0], 0.5);
  addAt(ctx, [1, 0, 0], 0.45);
  await still("clip-11-deformed-contour-step1");
  if (last > 0) store.dispatch({ type: "setResultStep", step: last });
  await still(`clip-12-deformed-contour-step${last + 1}`);
  if (typeof bridge.captureResize === "function") {
    await bridge.captureResize(1280, height);
    await settle();
    store.dispatch({ type: "frameSelection" });
    await still(`clip-13-deformed-contour-step${last + 1}.1280`);
  }
  removePlanes(ctx);
  store.dispatch({ type: "setDeform", on: false });
  store.dispatch({ type: "setColourBy", by: "group" });
  store.dispatch({ type: "setCutFace", on: true });
}

/**
 * The V4e2 plane-gizmo stills (D7), by a fixed rule, on the undeformed model
 * coloured by group, seen from the iso camera: one Z plane at 55 % with its gizmo; three planes (Z, X at
 * 40 %, Y at 60 %), three gizmos; the Z gizmo mid-drag (a real press on the
 * quad centre, moved a quarter of the arrow up the normal, still held); the Z
 * plane rotated by its tip (a real press, moved to tilt the normal about 30
 * degrees toward the camera, released); and the legend over a gizmo: the camera
 * zoomed in until the Z quad lies under the legend, a real press on the legend
 * there, which the gizmo does not take. Every gesture goes through the canvas's
 * own listeners. `<out>.gizmo-<n>-<what>.png`.
 */
export async function captureGizmo(ctx: HarnessContext, settle: (ms?: number) => Promise<unknown>, width: number, height: number) {
  const { store, viewport, bridge } = ctx;
  if (!store.get().mesh) return;
  if (typeof bridge.captureResize === "function") await bridge.captureResize(width, height);
  removePlanes(ctx);
  store.dispatch({ type: "setApplyCuts", on: true });
  store.dispatch({ type: "setShowGizmos", on: true });
  store.dispatch({ type: "setDeform", on: false });
  store.dispatch({ type: "setColourBy", by: "group" });
  store.dispatch({ type: "clearSelection" });
  // The iso view: a planar model opens in plan, where the Z plane's quad and its arrow tip fall on one screen
  // point, so the press on the quad would land on the tip (#2707).
  store.dispatch({ type: "setCamera", to: "iso" });
  store.dispatch({ type: "frameSelection" });
  await settle(150);
  viewport.renderNow();
  await painted();
  const still = async (name: string) => {
    await settle(150);
    viewport.renderNow();
    await painted();
    await painted();
    await bridge.captureStill(name);
  };
  const plane = (id: string) => store.get().clip.planes.find((p) => p.id === id)!;
  const geom = (id: string) => viewport.gizmos.geometries().get(id)!;
  // Functions, not the getters themselves: each read is after a gesture or an event the type checker cannot see.
  const count = (): number => viewport.gizmoCount;
  const mode = (): string | null => viewport.gizmoInput.mode;
  const z = addAt(ctx, [0, 0, 1], 0.55);
  if (count() !== 1) throw new Error(`capture: ${count()} gizmos for one plane`);
  await still("gizmo-1-one-plane");
  const x = addAt(ctx, [1, 0, 0], 0.4);
  const y = addAt(ctx, [0, 1, 0], 0.6);
  if (count() !== 3) throw new Error(`capture: ${count()} gizmos for three planes`);
  await still("gizmo-2-three-planes");
  store.dispatch({ type: "removeClipPlane", id: x });
  store.dispatch({ type: "removeClipPlane", id: y });

  // Mid-drag: a press on the quad centre, a move up the normal, the button still held.
  const g = geom(z);
  const off0 = plane(z).offset;
  const at0 = screenOf(ctx, g.centre);
  pointerAt(ctx, "pointerdown", at0, viewport.renderer.domElement);
  if (mode() !== "translate") throw new Error(`capture: the press on the Z quad is ${mode()}, not a translate`);
  const up = [g.centre[0] + 0.25 * g.arrowLen * g.normal[0], g.centre[1] + 0.25 * g.arrowLen * g.normal[1], g.centre[2] + 0.25 * g.arrowLen * g.normal[2]];
  for (const t of [0.25, 0.5, 0.75, 1]) {
    const p = g.centre.map((c, k) => c + t * (up[k]! - c));
    pointerAt(ctx, "pointermove", screenOf(ctx, p), viewport.renderer.domElement);
  }
  const moved = plane(z).offset - off0;
  if (!(Math.abs(moved - 0.25 * g.arrowLen) < 1e-3 * g.arrowLen)) throw new Error(`capture: the drag moved the plane ${moved}, not ${0.25 * g.arrowLen}`);
  const done = marker(screenOf(ctx, up), "mid-drag: the Z plane moves with the cursor along its normal");
  await still("gizmo-3-mid-drag");
  done();
  pointerAt(ctx, "pointerup", screenOf(ctx, up), viewport.renderer.domElement);

  // Rotated: a press on the tip, moved to tilt the normal toward the camera, released.
  const r = geom(z);
  pointerAt(ctx, "pointerdown", screenOf(ctx, tipPoint(r)), viewport.renderer.domElement);
  if (mode() !== "rotate") throw new Error(`capture: the press on the Z tip is ${mode()}, not a rotate`);
  const back = new THREE.Vector3(0, 0, 1).applyQuaternion(viewport.camera.quaternion); // toward the camera
  const n0 = new THREE.Vector3(...r.direction);
  const tilt = back.clone().sub(n0.clone().multiplyScalar(back.dot(n0))).normalize();
  const want = n0.clone().multiplyScalar(Math.cos(Math.PI / 6)).addScaledVector(tilt, Math.sin(Math.PI / 6));
  for (const t of [0.33, 0.66, 1]) {
    const d = n0.clone().lerp(want, t).normalize();
    pointerAt(ctx, "pointermove", screenOf(ctx, [r.centre[0] + r.arrowLen * d.x, r.centre[1] + r.arrowLen * d.y, r.centre[2] + r.arrowLen * d.z]), viewport.renderer.domElement);
  }
  pointerAt(ctx, "pointerup", screenOf(ctx, tipPoint(geom(z))), viewport.renderer.domElement);
  const p = plane(z);
  const onPlane = p.normal[0] * r.centre[0] + p.normal[1] * r.centre[1] + p.normal[2] * r.centre[2] - p.offset;
  if (Math.abs(onPlane) > 1e-6 * (store.get().mesh!.radius || 1)) throw new Error(`capture: the rotated plane is ${onPlane} off its quad centre`);
  await still("gizmo-4-rotated");

  // The legend over a gizmo: zoom in until the (re-straightened) Z quad lies under the legend, then press there.
  store.dispatch({ type: "setClipPose", id: z, normal: [0, 0, 1], offset: r.centre[2] });
  const legend = document.getElementById("legend")!;
  const box = legend.getBoundingClientRect();
  const at = { x: box.left + box.width / 2, y: box.top + box.height / 2 };
  const ray = new THREE.Raycaster();
  const gizmoUnder = () => {
    const rect = viewport.renderer.domElement.getBoundingClientRect();
    ray.setFromCamera(new THREE.Vector2(((at.x - rect.left) / rect.width) * 2 - 1, -((at.y - rect.top) / rect.height) * 2 + 1), viewport.camera);
    return viewport.gizmos.hit({ origin: ray.ray.origin.toArray(), direction: ray.ray.direction.toArray() });
  };
  for (let k = 0; k < 12 && !gizmoUnder(); k++) {
    zoomToward(viewport.camera, new THREE.Vector3(...geom(z).centre), 0.75);
    viewport.nav.updated();
  }
  if (!gizmoUnder()) throw new Error("capture: no zoom put the Z gizmo under the legend");
  const target = pointerAt(ctx, "pointerdown", at);
  pointerAt(ctx, "pointerup", at, target);
  if (!legend.contains(target)) throw new Error(`capture: the press under the legend reached ${target.nodeName}#${target.id}, not the legend`);
  if (viewport.gizmoInput.dragging) throw new Error("capture: the gizmo took a press on the legend");
  const off = marker(at, "a press here is the legend's: the gizmo under it does not move");
  await still("gizmo-5-legend-over-gizmo");
  off();
  removePlanes(ctx);
  store.dispatch({ type: "frameSelection" });
}
