// Renderer entry: wires the store, the BlobStore, the effects, the viewport
// and the panels, then runs the mode main asked for (view, measure or
// capture).

import * as THREE from "three";
import { Effects, type Bridge } from "../effects.ts";
import { mountBanner } from "../panels/banner.ts";
import { mountEmpty } from "../panels/empty.ts";
import { mountHeader } from "../panels/header.ts";
import { mountInspector } from "../panels/inspector.ts";
import { mountLegend } from "../panels/legend.ts";
import { mountPhase } from "../panels/phase.ts";
import { mountPlanes } from "../panels/planes.ts";
import { mountSources } from "../panels/sources.ts";
import { mountWindows } from "../panels/windows/frame.ts";
import { BlobStore } from "../state/blobs.ts";
import { offsetRange } from "../state/clip.ts";
import { contourKeysOf, gaussFields } from "../state/contour.ts";
import { chainOf, contourFieldsOf, contourView, deformOf, deformView, elementVerdictOf } from "../state/selectors.ts";
import { Store } from "../state/store.ts";
import { WINDOW_KINDS, type DeclPath, type GaussMode } from "../state/types.ts";
import { applyTheme } from "../theme/tokens.ts";
import { contourKey, contourShownKey, elementContourKey } from "./contour.ts";
import { Viewport } from "./viewport.ts";

const bridge = (window as unknown as { viewer: Bridge }).viewer;
applyTheme(document);

const store = new Store();
const blobs = new BlobStore();
const effects = new Effects(store, blobs, bridge);
const viewport = new Viewport(document.getElementById("viewport")!, store, blobs);
effects.setFrameTarget(viewport);
// V4c1: the contour colours from the step the effects fetched (never kept in the store).
viewport.setStepSource(() => effects.currentStep());
// V4d: the deformed shape moves the drawn vertices by the displacement the effects computed for the step shown.
viewport.setDisplacementSource(() => effects.currentDisplacement());
const panels = [mountHeader(store), mountBanner(store), mountLegend(store), mountInspector(store), mountEmpty(store), mountPhase(store), mountSources(store), mountWindows(store)];
// V4e1: the section planes panel; its slider box and the view direction come from the viewport.
panels.push(mountPlanes(store, { box: () => viewport.sliderBox(), viewNormal: () => viewport.viewNormal() }));
window.addEventListener("beforeunload", () => {
  for (const d of panels) d();
  effects.dispose();
  viewport.dispose();
});

const nextFrame = () => new Promise<number>((r) => requestAnimationFrame(r));
/** Resolves once the frame after the current DOM change has been painted. */
const painted = () => new Promise<void>((r) => requestAnimationFrame(() => setTimeout(r, 0)));

const BEAM_TYPES = new Set(["dispBeamColumn", "forceBeamColumn", "elasticBeamColumn"]);

/**
 * The element a scripted click targets: a beam-column if any; else an element
 * whose chain resolves with at least one link; else the middle drawn element.
 * A bounded sample (from the middle outwards) keeps this cheap on big models.
 * With `pickType`, every drawn element is a candidate (not only the sample),
 * and the first of that OpenSees type whose chain resolves wins; none found
 * raises, naming the type.
 */
function demoTarget(pickType: string | null = null): DeclPath | null {
  const s = store.get();
  if (!s.mesh) return null;
  const drawn = s.mesh.elements;
  if (pickType !== null) {
    for (const p of drawn) {
      if (s.decls[p]?.type !== pickType) continue;
      const c = chainOf(s, p);
      if (c && c.problems.length === 0) return p;
    }
    throw new Error(`--pick=${pickType}: no drawn element of that type with a resolved chain`);
  }
  const step = Math.max(1, Math.floor(drawn.length / 400));
  const candidates: DeclPath[] = [];
  for (let i = Math.floor(drawn.length / 2); i < drawn.length; i += step) candidates.push(drawn[i]!);
  for (let i = 0; i < Math.floor(drawn.length / 2); i += step) candidates.push(drawn[i]!);
  const chains = candidates.map((p) => ({ p, c: chainOf(s, p)! }));
  const beam = chains.find(({ c }) => BEAM_TYPES.has(c.root.type) && c.problems.length === 0);
  const linked = chains.find(({ c }) => c.root.children.length > 0 && c.problems.length === 0);
  return (beam ?? linked)?.p ?? candidates[0] ?? null;
}

/** Click at the target's projected midpoint through the same pick path a mouse uses. */
async function scriptedClick(target: DeclPath) {
  const pt = viewport.screenPointOf(target);
  if (!pt) throw new Error("scripted click: target is not drawn");
  const t = performance.now();
  const pick = viewport.pickAt(pt.x, pt.y);
  if (!pick) throw new Error("scripted click: pick returned nothing");
  const tPick = performance.now();
  store.dispatch({ type: "select", pick });
  const tState = performance.now();
  await painted();
  const end = performance.now();
  return {
    ms: end - t,
    split: { pickMs: tPick - t, stateAndDomMs: tState - tPick, paintMs: end - tState },
    pickedTarget: pick.decl === target,
    decl: pick.decl,
  };
}

async function measure(t0: number, startup: { appReadyMs: number; configMs: number }) {
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
  const tgt = demoTarget();
  const click = tgt ? await scriptedClick(tgt) : null;
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

function countLinks(n: { children: { children: unknown[] }[] }): number {
  return n.children.reduce((a, c) => a + 1 + countLinks(c as typeof n), 0);
}

/** The results stills of a capture (see the call site); each waits for its step fetch to settle. */
async function captureResults(path: string, settle: (ms?: number) => Promise<unknown>) {
  if (!(await effects.openResults(path))) throw new Error(`capture: results ${path} did not open: ${store.get().artifacts.results?.error}`);
  const s = store.get();
  const stages = s.results?.stages ?? [];
  const onAxis = s.phase.axis.some((k) => k.kind === "results");
  const still = async (name: string) => {
    for (let i = 0; i < 100 && store.get().resultStep?.status === "loading"; i++) await settle(20);
    await settle();
    viewport.renderNow();
    // Two painted frames: a hidden window's first capture after a burst of DOM changes can be the frame before them.
    await painted();
    await painted();
    await bridge.captureStill(name);
  };
  if (!onAxis) {
    // Not paired: one still of the banner that says why.
    await still("results-unpaired");
    return;
  }
  const safe = (n: string) => n.replace(/[^A-Za-z0-9_.-]+/g, "_") || "unnamed";
  for (const [i, st] of stages.entries()) {
    store.dispatch({ type: "setPhase", at: { kind: "results", stage: i, step: 0 } });
    if (st.time.length > 1) store.dispatch({ type: "setResultStep", step: st.time.length - 1 });
    await still(`results-${i}-${safe(st.name)}`);
  }
  // The first stage again at a 1280 px window: the scrubber row must still clear the header.
  if (typeof bridge.captureResize === "function" && stages.length) {
    await bridge.captureResize(1280, window.innerHeight);
    store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 0 } });
    if (stages[0]!.time.length > 1) store.dispatch({ type: "setResultStep", step: stages[0]!.time.length - 1 });
    await still(`results-0-${safe(stages[0]!.name)}.1280`);
  }
}

/** Poll the store (every 20 ms, at most `ms`) until `ok` holds; false on timeout. */
async function until(ok: () => boolean, ms = 15000): Promise<boolean> {
  for (let t = 0; t < ms; t += 20) {
    if (ok()) return true;
    await new Promise((r) => setTimeout(r, 20));
  }
  return ok();
}

/** The contour colours on screen are the step the state shows, at its range (renderer/contour.ts keys); V4c2a: an element field's too. */
function contourShown(): boolean {
  const v = contourView(store.get());
  if (v.state === "ready") return v.stepMeasured && viewport.contourApplied === contourKey(v);
  if (v.state === "element") return v.stepMeasured && viewport.contourApplied === elementContourKey(v);
  return false;
}

/** V4c2a: the mesh shows a refused or not-yet element contour as no data (its key names the reason). */
function refusalShown(): boolean {
  const v = contourView(store.get());
  return (v.state === "refused" || v.state === "not-yet") && viewport.contourApplied === contourShownKey(v);
}

/**
 * The V4c2a element-contour stills (D7), by a fixed rule so no flag is
 * needed: on the results stage with the most steps (the last of a tie) that
 * has an element field, its first element field's first component over the
 * whole-history range at the last step: `elements-1-<topology>` (or
 * `elements-1-refused-<code>` when the file is refused, with the banner);
 * then, drawable and deformable, the deformed shape at scale 10 with the
 * reference ghost (`elements-2-deformed-scale10`); V4c2b: then each
 * node-located draw mode (`elements-3-cell_averaged`, `-node`,
 * `-node_discrete`); then the first still again at 1280 px.
 * `<out>.elements-<n>-<what>.png`.
 */
async function captureElements(settle: (ms?: number) => Promise<unknown>, width: number, height: number) {
  const s0 = store.get();
  if (!s0.results || !s0.phase.axis.some((k) => k.kind === "results")) return;
  const stages = s0.results.stages;
  let best = -1;
  stages.forEach((st, i) => {
    if (gaussFields(st.gauss).length && (best < 0 || st.time.length >= stages[best]!.time.length)) best = i;
  });
  if (best < 0) return;
  if (typeof bridge.captureResize === "function") await bridge.captureResize(width, height);
  store.dispatch({ type: "setDeform", on: false });
  store.dispatch({ type: "clearSelection" });
  store.dispatch({ type: "setPhase", at: { kind: "results", stage: best, step: 0 } });
  const last = stages[best]!.time.length - 1;
  if (last > 0) store.dispatch({ type: "setResultStep", step: last });
  const f = gaussFields(stages[best]!.gauss)[0]!;
  store.dispatch({ type: "setColourBy", by: "contour" });
  store.dispatch({ type: "setContourRange", range: "history" });
  store.dispatch({ type: "setContourField", field: f.name, component: f.components[0]!.key, source: "element" });
  const ready = () => {
    const v = contourView(store.get());
    if (v.state === "element") return contourShown() && store.get().contour.scan?.status === "done" && deformShown();
    return refusalShown();
  };
  const still = async (name: string) => {
    if (!(await until(ready, 30000))) throw new Error(`capture ${name}: the element contour did not show (${contourView(store.get()).state})`);
    await settle(100);
    viewport.renderNow();
    await painted();
    await painted();
    await bridge.captureStill(name);
  };
  await until(() => contourView(store.get()).state !== "unavailable", 15000);
  const v = contourView(store.get());
  const first = v.state === "element" ? `elements-1-${v.topology}` : v.state === "refused" ? `elements-1-refused-${v.code}` : `elements-1-${v.state}`;
  await still(first);
  if (v.state === "element" && deformOf(store.get()).ok) {
    store.dispatch({ type: "setDeform", on: true });
    store.dispatch({ type: "setReferenceGhost", on: true });
    store.dispatch({ type: "setDeformScale", scale: 10 });
    await still("elements-2-deformed-scale10");
    store.dispatch({ type: "setDeform", on: false });
  }
  // V4c2b: the node-located draw modes, one still each (`elements-3-<mode>`), then back to `cell`.
  for (const mode of ["cell_averaged", "node", "node_discrete"] as const) {
    store.dispatch({ type: "setContourGauss", mode });
    await still(`elements-3-${mode}`);
  }
  store.dispatch({ type: "setContourGauss", mode: "cell" });
  if (typeof bridge.captureResize === "function") {
    await bridge.captureResize(1280, height);
    await settle();
    // Nothing is selected: frame the whole model in the narrower window.
    store.dispatch({ type: "frameSelection" });
    await settle();
    await still(`${first}.1280`);
  }
}

/**
 * The V4c1 contour stills (D7), by a fixed rule so no flag is needed: on the
 * results stage with the most steps (the last of a tie), coloured by its
 * first nodal field's first component (the magnitude when offered), the
 * legend while the history scan runs (main paces a capture's scan, one step
 * per 2 s), then after it at the first and the last step, then the first
 * axis component at the last step, and that last still again at 1280 px.
 * `<out>.contour-<n>-<what>.png`.
 */
async function captureContours(settle: (ms?: number) => Promise<unknown>, width: number, height: number) {
  const s0 = store.get();
  if (!s0.results || !s0.phase.axis.some((k) => k.kind === "results")) return;
  const stages = s0.results.stages;
  let best = -1;
  stages.forEach((st, i) => {
    if (st.nodal.length && (best < 0 || st.time.length >= stages[best]!.time.length)) best = i;
  });
  if (best < 0) return;
  if (typeof bridge.captureResize === "function") await bridge.captureResize(width, height);
  const still = async (name: string) => {
    if (!(await until(contourShown))) throw new Error(`capture ${name}: the contour did not show the step on screen`);
    await settle(100);
    viewport.renderNow();
    await painted();
    await painted();
    await bridge.captureStill(name);
  };
  store.dispatch({ type: "setPhase", at: { kind: "results", stage: best, step: 0 } });
  store.dispatch({ type: "setColourBy", by: "contour" });
  const scan = () => store.get().contour.scan;
  if (!(await until(() => scan()?.status === "running" && scan()!.done > 0))) throw new Error(`capture: the history scan did not report progress (${JSON.stringify(scan())})`);
  await still("contour-1-scan-running");
  // The still must show a scan in progress: one that finished first would show "whole history" under this name.
  if (scan()?.status !== "running") throw new Error(`capture: the history scan finished before its still was taken (${JSON.stringify(scan())})`);
  if (!(await until(() => scan()?.status === "done", 30000))) throw new Error(`capture: the history scan did not finish (${JSON.stringify(scan())})`);
  await still("contour-2-history-first-step");
  const last = stages[best]!.time.length - 1;
  store.dispatch({ type: "setResultStep", step: last });
  await still("contour-3-history-last-step");
  const v = contourView(store.get());
  const axis = v.state === "ready" ? contourFieldsOf(store.get())?.fields.find((f) => f.name === v.field)?.components.find((c) => !c.magnitude) : undefined;
  if (axis && v.state === "ready") {
    store.dispatch({ type: "setContourField", field: v.field, component: axis.key });
    await until(() => scan()?.status === "done" && scan()!.key.endsWith(`|${axis.key}`), 30000);
    await still(`contour-4-component-${axis.label}`);
  }
  if (typeof bridge.captureResize === "function") {
    await bridge.captureResize(1280, height);
    await settle();
    await still("contour-5-last-step.1280");
  }
}

/** V4d: the drawn positions are the deformed shape of the step the state shows, at its scale (renderer/deform.ts keys); or the shape cannot be drawn. */
function deformShown(): boolean {
  const v = deformView(store.get());
  if (v.state === "off" || v.state === "unavailable" || v.state === "refused") return true;
  if (v.state === "loading") return false;
  return viewport.deformApplied === `${effects.currentDisplacement()?.key}|scale ${v.scale}` && effects.currentDisplacement()?.key === v.keys.disp;
}

/** Resolves once the contour (and, V4d, the deformed shape) shows the step the state names (checked after every store change; the viewport listens first). */
function contourShowsNow(): Promise<void> {
  const shown = () => contourShown() && deformShown();
  return new Promise((resolve) => {
    if (shown()) return resolve();
    const off = store.subscribe(() => {
      if (!shown()) return;
      off();
      resolve();
    });
  });
}

/**
 * The V4d deformed-shape stills (D7), by a fixed rule so no flag is needed:
 * on the results stage with the most steps (the last of a tie) that has a
 * usable displacement, at its last step, coloured by group: deformed at true
 * scale with the reference ghost, then at scale 10, then coloured by
 * contour; then two staged views (the first two results stages at their
 * last step, or the first and the last step of a single stage), and the
 * scale-10 still again at 1280 px. A `partial` stage shows its banner and
 * markers in every still. `<out>.deform-<n>-<what>.png`.
 */
async function captureDeform(settle: (ms?: number) => Promise<unknown>, width: number, height: number) {
  const s0 = store.get();
  if (!s0.results || !s0.phase.axis.some((k) => k.kind === "results")) return;
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
  store.dispatch({ type: "setColourBy", by: "group" });
  store.dispatch({ type: "clearSelection" });
  const show = (stage: number, step: number) => {
    store.dispatch({ type: "setPhase", at: { kind: "results", stage, step: 0 } });
    if (step > 0) store.dispatch({ type: "setResultStep", step });
  };
  const still = async (name: string) => {
    if (!(await until(() => deformShown() && (store.get().visibility.colourBy !== "contour" || contourShown())))) throw new Error(`capture ${name}: the deformed shape did not show the step on screen (${deformView(store.get()).state})`);
    await settle(100);
    viewport.renderNow();
    await painted();
    await painted();
    await bridge.captureStill(name);
  };
  const last = (i: number) => stages[i]!.time.length - 1;
  show(best, last(best));
  store.dispatch({ type: "setDeform", on: true });
  store.dispatch({ type: "setReferenceGhost", on: true });
  store.dispatch({ type: "setDeformScale", scale: 1 });
  await still("deform-1-scale1-ghost");
  store.dispatch({ type: "setDeformScale", scale: 10 });
  await still("deform-2-scale10-ghost");
  store.dispatch({ type: "setColourBy", by: "contour" });
  await still("deform-3-scale10-contour");
  store.dispatch({ type: "setColourBy", by: "group" });
  const safe = (n: string) => n.replace(/[^A-Za-z0-9_.-]+/g, "_") || "unnamed";
  const views: [number, number][] = stages.length >= 2 ? [[0, last(0)], [1, last(1)]] : [[0, 0], [0, last(0)]];
  for (const [i, k] of views) {
    show(i, k);
    await still(`deform-4-stage${i}-${safe(stages[i]!.name)}-step${k + 1}`);
  }
  if (typeof bridge.captureResize === "function") {
    await bridge.captureResize(1280, height);
    show(best, last(best));
    await settle();
    await still("deform-5-scale10-ghost.1280");
  }
  store.dispatch({ type: "setDeform", on: false });
}

/**
 * The V4c1 contour scrub (MEASUREMENTS.md, the contour rows): on the results
 * stage with the most steps, coloured by its first field's first component
 * (the magnitude when offered) over the whole-history range, every step in
 * turn, end to end: `setResultStep`, the step read in main, the IPC, the
 * step range, the vertex colours, the upload and a draw with `gl.finish()`.
 * The history scan's time comes first; the machine's CPU load is sampled
 * before (1 s idle) and during the scrub, and the working sets every 100 ms.
 */
async function measureContour(path: string, deform = false, element = false, gauss: GaussMode = "cell", planes = false) {
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
    await contourShowsNow();
    const v = contourView(store.get());
    const d = deformView(store.get());
    if (deform && d.state !== "ready" && d.state !== "partial") throw new Error(`measure: the deformed shape is ${d.state}`);
    if (element && v.state !== "element") throw new Error(`measure: the element contour is ${v.state}`);
    const what = v.state === "ready" ? `${v.field} ${v.component.label}` : v.state === "element" ? `element ${v.field} ${v.component.label}, ${v.topology}` : "?";
    // V4e1: six section planes cutting through the scrub (each 20 % inside the reference box along ±X, ±Y, ±Z).
    if (planes) addSixPlanes();
    if (planes && viewport.clipCount !== 6) throw new Error(`measure: ${viewport.clipCount} section planes cut, not 6`);
    const field = what + (deform ? ", deformed ×1" : "") + (planes ? ", 6 section planes" : "");
    const T = stages[best]!.time.length;
    const gl = viewport.renderer.getContext();
    const stepMs: number[] = [];
    await cpu();
    const start = performance.now();
    for (let k = 1; k < T; k++) {
      const t = performance.now();
      store.dispatch({ type: "setResultStep", step: k });
      await contourShowsNow();
      viewport.renderNow();
      gl.finish();
      stepMs.push(performance.now() - t);
    }
    const total = performance.now() - start;
    const cpuDuring = await cpu();
    await sample();
    const sorted = [...stepMs].sort((a, b) => a - b);
    return {
      file: path,
      field: `${field} (stage ${stages[best]!.name || stages[best]!.id})`,
      steps: T,
      scanMs,
      scrubStepsPerSec: ((T - 1) * 1000) / total,
      stepMs: { median: sorted[Math.floor((sorted.length - 1) / 2)] ?? NaN, max: sorted[sorted.length - 1] ?? NaN },
      peakMB: peak,
      cpu: { before: cpuBefore, during: cpuDuring },
    };
  } finally {
    clearInterval(sampler);
    if (planes) removePlanes();
  }
}

/** V4e1: six active section planes, one per face of the reference box, each 20 % of its extent inside it. */
function addSixPlanes() {
  const box = viewport.sliderBox();
  if (!box) throw new Error("section planes: no mesh");
  const normals: [number, number, number][] = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
  for (const n of normals) {
    store.dispatch({ type: "addClipPlane", normal: n });
    const p = store.get().clip.planes.at(-1)!;
    const r = offsetRange(p.normal, box);
    store.dispatch({ type: "setClipOffset", id: p.id, offset: r.min + 0.2 * (r.max - r.min) });
  }
}

function removePlanes() {
  for (const p of store.get().clip.planes) store.dispatch({ type: "removeClipPlane", id: p.id });
}

/** The id of the plane added last. */
const lastPlane = (): string => store.get().clip.planes.at(-1)!.id;

/** Put plane `id` at fraction `t` of the reference box along its normal (0: the low end). */
function placeAt(id: string, t: number) {
  const box = viewport.sliderBox()!;
  const p = store.get().clip.planes.find((q) => q.id === id)!;
  const r = offsetRange(p.normal, box);
  store.dispatch({ type: "setClipOffset", id, offset: r.min + t * (r.max - r.min) });
}

/** Add a plane of normal `n` at fraction `t` of the reference box along it; its id. */
function addAt(n: readonly number[], t: number): string {
  store.dispatch({ type: "addClipPlane", normal: n });
  const id = lastPlane();
  placeAt(id, t);
  return id;
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
 * cuts away the nearer half, and a scripted click on an element of that half
 * is a miss (the ring marks where it clicked); the same click with "Apply
 * cuts" off selects it. Then the deformed (scale 10, ghost on), contoured
 * model cut by a Y plane (the half nearer the camera gone) and an X plane at
 * 45 %, at the first and the last step of the scrub (the shear carries more
 * of it through the X plane), and that last still at 1280 px. Plane offsets
 * avoid the mesh's own face planes, where a cut coincides with a face.
 * `<out>.clip-<n>-<what>.png`.
 */
async function captureClip(settle: (ms?: number) => Promise<unknown>, width: number, height: number) {
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
  removePlanes();
  store.dispatch({ type: "setApplyCuts", on: true });
  store.dispatch({ type: "setDeform", on: false });
  store.dispatch({ type: "setColourBy", by: "group" });
  store.dispatch({ type: "clearSelection" });
  store.dispatch({ type: "setPhase", at: { kind: "results", stage: best, step: 0 } });
  if (last > 0) store.dispatch({ type: "setResultStep", step: last });
  store.dispatch({ type: "frameSelection" });
  const still = async (name: string) => {
    if (!(await until(() => deformShown() && (store.get().visibility.colourBy !== "contour" || contourShown())))) throw new Error(`capture ${name}: the step did not show`);
    await settle(150);
    viewport.renderNow();
    await painted();
    await painted();
    await bridge.captureStill(name);
  };
  await still("clip-1-before");
  const z = addAt([0, 0, 1], 0.4);
  store.dispatch({ type: "setClipFlipped", id: z, flipped: true });
  await still("clip-2-z-plane-hollow");
  const x = addAt([1, 0, 0], 0.3);
  await still("clip-3-corner-two-planes");
  store.dispatch({ type: "setClipFlipped", id: z, flipped: false });
  await still("clip-4-z-flipped-back");
  placeAt(z, 0.75);
  await still("clip-5-z-offset-75");
  store.dispatch({ type: "setClipNormal", id: x, normal: [1, 1, 0] });
  await still("clip-6-panel-custom-normal");
  removePlanes();
  for (const n of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]] as const) addAt(n, 0.1);
  await still("clip-7-six-planes");
  store.dispatch({ type: "addClipPlane", normal: [1, 0, 1] });
  if (store.get().clip.notice === null) throw new Error("capture: a 7th plane was not refused");
  await still("clip-8-seventh-refused");

  // The pick that misses cut-away geometry.
  removePlanes();
  store.dispatch({ type: "addClipPlane", normal: [...viewport.viewNormal()] });
  const target = clipMissTarget();
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
  removePlanes();

  // The deformed, contoured (and stage-activated) model cut while scrubbing.
  store.dispatch({ type: "setPhase", at: { kind: "results", stage: best, step: 0 } });
  store.dispatch({ type: "setDeform", on: true });
  store.dispatch({ type: "setReferenceGhost", on: true });
  store.dispatch({ type: "setDeformScale", scale: 10 });
  store.dispatch({ type: "setColourBy", by: "contour" });
  addAt([0, 1, 0], 0.5);
  addAt([1, 0, 0], 0.45);
  await still("clip-11-deformed-contour-step1");
  if (last > 0) store.dispatch({ type: "setResultStep", step: last });
  await still(`clip-12-deformed-contour-step${last + 1}`);
  if (typeof bridge.captureResize === "function") {
    await bridge.captureResize(1280, height);
    await settle();
    store.dispatch({ type: "frameSelection" });
    await still(`clip-13-deformed-contour-step${last + 1}.1280`);
  }
  removePlanes();
  store.dispatch({ type: "setDeform", on: false });
  store.dispatch({ type: "setColourBy", by: "group" });
}

/**
 * The click target of the miss still: of the drawn elements whose own screen
 * point picks them with the cuts off and is cut away with them on, the one
 * nearest the middle of the model on screen.
 */
function clipMissTarget(): { decl: DeclPath; x: number; y: number } | null {
  const s = store.get();
  const c = new THREE.Vector3(...s.mesh!.center).project(viewport.camera);
  const rect = viewport.renderer.domElement.getBoundingClientRect();
  const cx = rect.left + ((c.x + 1) / 2) * rect.width, cy = rect.top + ((1 - c.y) / 2) * rect.height;
  let best: { decl: DeclPath; x: number; y: number; d: number } | null = null;
  for (const decl of s.mesh!.elements) {
    const pt = viewport.screenPointOf(decl);
    if (!pt) continue;
    const d = Math.hypot(pt.x - cx, pt.y - cy);
    if (best && d >= best.d) continue;
    store.dispatch({ type: "setApplyCuts", on: false });
    const off = viewport.pickAt(pt.x, pt.y);
    store.dispatch({ type: "setApplyCuts", on: true });
    if (off?.decl !== decl) continue;
    if (viewport.pickAt(pt.x, pt.y) === null) best = { decl, ...pt, d };
  }
  return best && { decl: best.decl, x: best.x, y: best.y };
}

async function main() {
  const cfg = await bridge.config();
  if (cfg.mode === "view") {
    effects.attach();
    // With V2f's `onOpen`, main replays the open set on subscription; loading
    // `config().file` here as well would read the model twice.
    if (cfg.file && !effects.hasOpenFeed()) await effects.open(cfg.file);
    return;
  }
  if (!cfg.file || !(await effects.open(cfg.file))) {
    await bridge.fail(store.get().artifacts.model?.error ?? "no --file given");
    return;
  }
  if (cfg.mode === "measure") {
    const result = await measure(cfg.t0, cfg);
    // V4c1: with a results sibling, the contour scrub row too (main prints it).
    const contour = cfg.results ? await measureContour(cfg.results) : null;
    // V4d: and the contour scrub again with the deformed shape on.
    const deformed = cfg.results ? await measureContour(cfg.results, true) : null;
    // V4c2a: the scrub with an element contour on (a verified file with an element field), then deformed too.
    const elements = cfg.results ? await measureContour(cfg.results, false, true) : null;
    const elementsDeformed = elements && cfg.results ? await measureContour(cfg.results, true, true) : null;
    // V4c2b: the element scrub carried to the nodes, one row per node-located mode.
    const elementsAveraged = elements && cfg.results ? await measureContour(cfg.results, false, true, "cell_averaged") : null;
    const elementsNode = elements && cfg.results ? await measureContour(cfg.results, false, true, "node") : null;
    const elementsDiscrete = elements && cfg.results ? await measureContour(cfg.results, false, true, "node_discrete") : null;
    // V4e1: the nodal scrub, deformed, with six section planes cutting (in the JSON as `clip`; main prints no row for it).
    const clip = cfg.results ? await measureContour(cfg.results, true, false, "cell", true) : null;
    await bridge.measureDone({ ...result, contour, deform: deformed, elements, elementsDeform: elementsDeformed, elementsAveraged, elementsNode, elementsDiscrete, clip });
    return;
  }
  // capture: select the scripted target and let main grab the page; when the
  // chain is taller than the window, a second still shows its end; a third,
  // `<out>.framed.png`, shows the view after `F` framed the selection.
  const settle = (ms = 300) => new Promise((r) => setTimeout(r, ms));
  viewport.renderNow();
  const tgt = demoTarget(cfg.pick);
  if (tgt) store.dispatch({ type: "select", pick: viewport.pickOf(tgt, null) });
  viewport.renderNow();
  // Past the selection pulse, so the still shows the halo at rest.
  await settle(900);
  viewport.renderNow();
  await bridge.captureStill("");
  const insp = document.getElementById("inspector")!;
  if (!insp.hidden && insp.scrollHeight > insp.clientHeight + 4) {
    insp.scrollTop = insp.scrollHeight;
    await settle();
    viewport.renderNow();
    await bridge.captureStill("chain-end");
    insp.scrollTop = 0;
  }
  if (tgt) {
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "f" }));
    await settle();
    viewport.renderNow();
    await bridge.captureStill("framed");
  }
  // Two palette stills with nothing selected (a selection dims the model):
  // `<out>.groups.png` coloured by physical group, `<out>.role.png` by role.
  // With nothing selected, `frameSelection` frames the whole model again
  // after `F` framed the selection above.
  store.dispatch({ type: "clearSelection" });
  store.dispatch({ type: "frameSelection" });
  await settle();
  viewport.renderNow();
  await bridge.captureStill("groups");
  store.dispatch({ type: "setColourBy", by: "role" });
  await settle();
  viewport.renderNow();
  await bridge.captureStill("role");
  // One still per model-definition window, `<out>.window-<kind>.png`, each
  // opened alone through the store, as the window menu opens it. A window
  // with something to select gets a second still, `<out>.window-<kind>-selected.png`
  // (V4f, D7): its first row that selects a declaration (`.window-row[data-decl]`)
  // selected through the store, or, in a window of stage cards, the last stage
  // selected (`selectStage`). The rule is fixed, so no flag is needed and the
  // stills come from committed code.
  store.dispatch({ type: "setColourBy", by: "group" });
  for (const kind of WINDOW_KINDS) {
    store.dispatch({ type: "openWindow", window: kind });
    await settle();
    viewport.renderNow();
    await bridge.captureStill(`window-${kind}`);
    const row = document.querySelector<HTMLElement>("#windows .window-row[data-decl]");
    const decl = row?.dataset["decl"];
    const cards = document.querySelectorAll("#windows .stage-card").length;
    if (decl) store.dispatch({ type: "select", pick: { decl, at: null, nodes: null } });
    else if (cards) store.dispatch({ type: "selectStage", index: cards - 1 });
    if (decl || cards) {
      await settle();
      viewport.renderNow();
      await bridge.captureStill(`window-${kind}-selected`);
      store.dispatch({ type: "clearSelection" });
      store.dispatch({ type: "selectStage", index: null });
    }
    store.dispatch({ type: "closeWindow", window: kind });
  }
  // V4b (D7): with a results sibling (`<stem>.results.h5`, which main names
  // in `config().results`), one still per results stage,
  // `<out>.results-<i>-<stage name>.png`, at its last step: the scrubber, the
  // stage activation (inactive elements hidden), and the banner when the
  // activation is `ambiguous` or `absent`, or the results do not pair. The
  // rule is fixed, so no flag is needed.
  if (cfg.results) {
    const width = window.innerWidth, height = window.innerHeight;
    await captureResults(cfg.results, settle);
    // V4c1 (D7): the contour stills, at the capture window's own width, then 1280 px.
    await captureContours(settle, width, height);
    // V4d (D7): the deformed-shape stills, at the capture window's own width, then 1280 px.
    await captureDeform(settle, width, height);
    // V4c2a (D7): the element-contour stills, at the capture window's own width, then 1280 px.
    await captureElements(settle, width, height);
    // V4e1 (D7): the section-plane stills, at the capture window's own width, then 1280 px.
    await captureClip(settle, width, height);
  }
  await bridge.captureDone();
}

main().catch((err) => {
  console.error(err);
  void bridge.fail(err instanceof Error ? `${err.message}\n${err.stack}` : String(err));
});
