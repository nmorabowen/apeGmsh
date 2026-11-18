// The measure harness (VH1, #2865): `npm run measure`'s orbit, scripted click and contour scrubs (MEASUREMENTS.md).

import * as THREE from "three";
import { contourKeysOf, gaussFields } from "../../state/contour.ts";
import { chainOf, contourView, deformView, elementVerdictOf } from "../../state/selectors.ts";
import type { GaussMode } from "../../state/types.ts";
import { nextFrame, painted, until, type HarnessConfig, type HarnessContext } from "./context.ts";
import { addSixPlanes, removePlanes } from "./planes.ts";
import { demoTarget, scriptedClick } from "./pointer.ts";
import { contourShowsNow } from "./shown.ts";

export async function measure(ctx: HarnessContext, t0: number, startup: { appReadyMs: number; configMs: number }) {
  const { store, viewport, bridge } = ctx;
  await nextFrame();
  viewport.renderNow();
  await painted();
  const firstFrameMs = Date.now() - t0;
  const s = store.get();
  const model = s.artifacts.model!;
  const mesh = s.mesh!;

  // Scripted orbit: one turn about the vertical (Z) axis through the fitted
  // centre, 6 s, through the same turntable code a right-drag uses.
  const ORBIT_MS = 6000;
  const target = viewport.nav.target.clone();
  const deltas: number[] = [];
  const renderMs: number[] = [];
  viewport.continuous = true;
  const start = await nextFrame();
  let last = start;
  let turned = 0;
  for (;;) {
    const now = await nextFrame();
    deltas.push(now - last);
    last = now;
    const ang = Math.min(1, (now - start) / ORBIT_MS) * Math.PI * 2;
    viewport.nav.orbit(target, ang - turned, 0);
    turned = ang;
    viewport.renderNow();
    renderMs.push(viewport.lastRenderMs);
    if (now - start >= ORBIT_MS) break;
  }
  viewport.continuous = false;
  viewport.renderNow();
  await painted();

  const median = (a: number[]) => {
    const b = [...a].sort((x, y) => x - y);
    return b.length ? b[Math.floor(b.length / 2)]! : NaN;
  };
  const tgt = demoTarget(ctx);
  const click = tgt ? await scriptedClick(ctx, tgt) : null;
  const chain = click ? chainOf(store.get(), click.decl) : null;
  // A second, shorter orbit with the selection (and its halo pass) active.
  const SELECTED_ORBIT_MS = 2000;
  const selDeltas: number[] = [];
  let orbitSelected: { durationMs: number; frames: number; medianFps: number } | null = null;
  if (click) {
    viewport.continuous = true;
    const s0 = await nextFrame();
    let prev = s0, done = 0;
    for (;;) {
      const now = await nextFrame();
      selDeltas.push(now - prev);
      prev = now;
      const ang = Math.min(1, (now - s0) / SELECTED_ORBIT_MS) * Math.PI * 2;
      viewport.nav.orbit(target, ang - done, 0);
      done = ang;
      viewport.renderNow();
      if (now - s0 >= SELECTED_ORBIT_MS) break;
    }
    viewport.continuous = false;
    orbitSelected = { durationMs: prev - s0, frames: selDeltas.length, medianFps: 1000 / median(selDeltas) };
  }
  const metrics = await bridge.metrics();
  return {
    file: model.path,
    sizeBytes: model.sizeBytes,
    nodes: model.counts.nodes,
    cells: model.counts.cells,
    opsOnlyElements: mesh.counts.opsOnly,
    drawn: { segments: mesh.linePositions.shape[0], triangles: mesh.triPositions.shape[0] },
    readMs: model.readMs,
    firstFrameMs,
    startup: { electronReadyMs: startup.appReadyMs, rendererUpMs: startup.configMs },
    orbit: {
      durationMs: last - start,
      frames: deltas.length,
      medianFps: 1000 / median(deltas),
      medianRenderCpuMs: median(renderMs),
    },
    orbitSelected,
    memory: metrics,
    inspector: click
      ? {
          fillMs: click.ms,
          split: click.split,
          pickedTarget: click.pickedTarget,
          root: chain?.root.name,
          type: chain?.root.type,
          links: chain ? countLinks(chain.root) : 0,
          problems: chain?.problems ?? [],
        }
      : null,
    gpu: viewport.gpuName(),
    pixelRatio: window.devicePixelRatio,
    viewport: [viewport.renderer.domElement.width, viewport.renderer.domElement.height],
    warnings: [...model.warnings],
    three: THREE.REVISION,
  };
}

export function countLinks(n: { children: { children: unknown[] }[] }): number {
  return n.children.reduce((a, c) => a + 1 + countLinks(c as typeof n), 0);
}

/** V4e3: the deformation scale of the capped scrub (see `measureContour`). */
const CAPPED_SCALE = 1e-6;

/**
 * The V4c1 contour scrub (MEASUREMENTS.md, the contour rows): on the results
 * stage with the most steps, coloured by its first field's first component
 * (the magnitude when offered) over the whole-history range, every step in
 * turn, end to end: `setResultStep`, the step read in main, the IPC, the
 * step range, the vertex colours, the upload and a draw with `gl.finish()`.
 * The history scan's time comes first; the machine's CPU load is sampled
 * before (1 s idle) and during the scrub, and the working sets every 100 ms.
 */
export async function measureContour(ctx: HarnessContext, path: string, deform = false, element = false, gauss: GaussMode = "cell", planes = false, capped = false) {
  const { store, effects, viewport, bridge } = ctx;
  store.dispatch({ type: "clearSelection" });
  // V4d: the same scrub with the deformed shape on (true scale): each step also waits for the displaced positions.
  store.dispatch({ type: "setDeform", on: deform });
  if (!(await effects.openResults(path))) throw new Error(`measure: results ${path} did not open: ${store.get().artifacts.results?.error}`);
  const s = store.get();
  if (!s.phase.axis.some((k) => k.kind === "results")) throw new Error(`measure: ${path} does not pair with the model, so there is nothing to scrub`);
  const stages = s.results!.stages;
  // V4c2a: the element scrub colours by the first drawable element field of a verified file (none: no row).
  const elementField = (i: number) => gaussFields(stages[i]!.gauss).find((f) => f.refused === null);
  if (element && !elementVerdictOf(s.results!).verified) return null;
  let best = -1;
  stages.forEach((st, i) => {
    const has = element ? elementField(i) !== undefined : st.nodal.length > 0;
    if (has && (best < 0 || st.time.length > stages[best]!.time.length)) best = i;
  });
  if (best < 0 && element) return null;
  if (best < 0) throw new Error(`measure: ${path} has no stage with nodal results`);
  const cpu = bridge.measureCpu!;
  const memory = bridge.measureMemory!;
  await cpu();
  await new Promise((r) => setTimeout(r, 1000));
  const cpuBefore = await cpu();
  const peak = { main: 0, renderer: 0 };
  const sample = async () => {
    const m = await memory();
    if (m.main + m.renderer > peak.main + peak.renderer) Object.assign(peak, m);
  };
  const sampler = setInterval(() => void sample(), 100);
  // V4e3: the caps toggle and the deformation scale as they were, put back after the scrub.
  const cutFaceBefore = store.get().clip.cutFace;
  const scaleBefore = store.get().deform.scale;
  try {
    const t0 = performance.now();
    store.dispatch({ type: "setPhase", at: { kind: "results", stage: best, step: 0 } });
    store.dispatch({ type: "setColourBy", by: "contour" });
    store.dispatch({ type: "setContourRange", range: "history" });
    if (element) {
      const f = elementField(best)!;
      store.dispatch({ type: "setContourField", field: f.name, component: f.components[0]!.key, source: "element" });
    }
    // V4c2b: the element scrub in a node-located draw mode (`cell` leaves the V4c2a rows as they were).
    store.dispatch({ type: "setContourGauss", mode: element ? gauss : "cell" });
    // V4c2b: a node-located mode waits for its own scan (its key names the mode), never an earlier mode's.
    const scanDone = () =>
      store.get().contour.scan?.status === "done" && (!element || store.get().contour.scan!.key.includes("|elem:")) && (gauss === "cell" || store.get().contour.scan!.key === contourKeysOf(store.get())?.scan);
    if (!(await until(scanDone, 600000))) throw new Error(`measure: the history scan did not finish: ${JSON.stringify(store.get().contour.scan)}`);
    const scanMs = performance.now() - t0;
    await contourShowsNow(ctx);
    const v = contourView(store.get());
    const d = deformView(store.get());
    if (deform && d.state !== "ready" && d.state !== "partial") throw new Error(`measure: the deformed shape is ${d.state}`);
    if (element && v.state !== "element") throw new Error(`measure: the element contour is ${v.state}`);
    const what = v.state === "ready" ? `${v.field} ${v.component.label}` : v.state === "element" ? `element ${v.field} ${v.component.label}, ${v.topology}` : "?";
    // V4e3: the cut-face caps on (the `caps` row) or off (the `clip` row, comparable with the V4e1 and V4e2 rows).
    // The measured files' displacement is synthetic (the node id plus a ramp: thousands of model lengths), so at
    // true scale the deformed solid leaves every plane and no cap is cut; the capped row deforms it at ×1e-6.
    store.dispatch({ type: "setCutFace", on: capped });
    if (capped) store.dispatch({ type: "setDeformScale", scale: CAPPED_SCALE });
    // V4e1: six section planes cutting through the scrub (each 20 % inside the reference box along ±X, ±Y, ±Z).
    if (planes) addSixPlanes(ctx);
    if (planes && viewport.clipCount !== 6) throw new Error(`measure: ${viewport.clipCount} section planes cut, not 6`);
    // V4e2: and their six gizmos drawn (shown by default), so the scrub renders them every step.
    if (planes && viewport.gizmoCount !== 6) throw new Error(`measure: ${viewport.gizmoCount} plane gizmos drawn, not 6`);
    // V4e3: and, capped, a cap on each of the six planes, sliced again at every step (the deformed solid, the step's field).
    if (capped && !(await until(() => viewport.capsSettled() && viewport.capCount === 6))) throw new Error(`measure: ${viewport.capCount} caps drawn, not 6`);
    const slices0 = viewport.caps.slices;
    const field = what + (deform ? `, deformed ×${capped ? CAPPED_SCALE : 1}` : "") + (planes ? ", 6 section planes + gizmos" : "") + (capped ? " + 6 caps" : "");
    const T = stages[best]!.time.length;
    const gl = viewport.renderer.getContext();
    const stepMs: number[] = [];
    await cpu();
    const start = performance.now();
    for (let k = 1; k < T; k++) {
      const t = performance.now();
      store.dispatch({ type: "setResultStep", step: k });
      await contourShowsNow(ctx);
      if (capped && !viewport.capsSettled()) throw new Error(`measure: the caps do not show step ${k} with its contour`);
      viewport.renderNow();
      gl.finish();
      stepMs.push(performance.now() - t);
    }
    const total = performance.now() - start;
    const cpuDuring = await cpu();
    await sample();
    const sorted = [...stepMs].sort((a, b) => a - b);
    // V4e3: one slice per plane per step, never more (F-SETTLED at scale).
    const slices = viewport.caps.slices - slices0;
    if (capped && (slices !== 6 * (T - 1) || viewport.capCount !== 6)) throw new Error(`measure: ${slices} slices over ${T - 1} steps (want ${6 * (T - 1)}), ${viewport.capCount} caps`);
    return {
      file: path,
      field: `${field} (stage ${stages[best]!.name || stages[best]!.id})`,
      steps: T,
      ...(capped ? { capSlices: slices, capTriangles: viewport.caps.drawn.reduce((a, c) => a + c.slice.cell.length, 0) } : {}),
      scanMs,
      scrubStepsPerSec: ((T - 1) * 1000) / total,
      stepMs: { median: sorted[Math.floor((sorted.length - 1) / 2)] ?? NaN, max: sorted[sorted.length - 1] ?? NaN },
      peakMB: peak,
      cpu: { before: cpuBefore, during: cpuDuring },
    };
  } finally {
    clearInterval(sampler);
    if (planes) removePlanes(ctx);
    store.dispatch({ type: "setCutFace", on: cutFaceBefore });
    store.dispatch({ type: "setDeformScale", scale: scaleBefore });
  }
}

/** The measure sequence (`--mode=measure`), after main opened the model: its rows, in this order, to main. */
export async function runMeasure(ctx: HarnessContext, cfg: HarnessConfig): Promise<void> {
  const { store, bridge } = ctx;
  const result = await measure(ctx, cfg.t0, cfg);
  // V4c1: with a results sibling, the contour scrub row too (main prints it).
  const contour = cfg.results ? await measureContour(ctx, cfg.results) : null;
  // V4d: and the contour scrub again with the deformed shape on.
  const deformed = cfg.results ? await measureContour(ctx, cfg.results, true) : null;
  // V4c2a: the scrub with an element contour on (a verified file with an element field), then deformed too.
  const elements = cfg.results ? await measureContour(ctx, cfg.results, false, true) : null;
  const elementsDeformed = elements && cfg.results ? await measureContour(ctx, cfg.results, true, true) : null;
  // V4c2b: the element scrub carried to the nodes, one row per node-located mode.
  const elementsAveraged = elements && cfg.results ? await measureContour(ctx, cfg.results, false, true, "cell_averaged") : null;
  const elementsNode = elements && cfg.results ? await measureContour(ctx, cfg.results, false, true, "node") : null;
  const elementsDiscrete = elements && cfg.results ? await measureContour(ctx, cfg.results, false, true, "node_discrete") : null;
  // V4e1: the nodal scrub, deformed, with six section planes cutting (in the JSON as `clip`; main prints no row for it).
  const clip = cfg.results ? await measureContour(ctx, cfg.results, true, false, "cell", true) : null;
  // V4e3: the same scrub with the six planes capped (in the JSON as `caps`; main prints no row for it). A model with no solid cell has no row.
  const caps = cfg.results && store.get().mesh!.counts.solidCells > 0 ? await measureContour(ctx, cfg.results, true, false, "cell", true, true) : null;
  await bridge.measureDone({ ...result, contour, deform: deformed, elements, elementsDeform: elementsDeformed, elementsAveraged, elementsNode, elementsDiscrete, clip, caps });
}
