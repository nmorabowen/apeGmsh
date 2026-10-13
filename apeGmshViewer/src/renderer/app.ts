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
import { mountSources } from "../panels/sources.ts";
import { mountWindows } from "../panels/windows/frame.ts";
import { BlobStore } from "../state/blobs.ts";
import { chainOf, contourFieldsOf, contourView } from "../state/selectors.ts";
import { Store } from "../state/store.ts";
import { WINDOW_KINDS, type DeclPath } from "../state/types.ts";
import { applyTheme } from "../theme/tokens.ts";
import { contourKey } from "./contour.ts";
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

/** The contour colours on screen are the step the state shows, at its range (renderer/contour.ts keys). */
function contourShown(): boolean {
  const v = contourView(store.get());
  return v.state === "ready" && v.stepMeasured && viewport.contourApplied === contourKey(v);
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

/** Resolves once the contour shows the step the state names (checked after every store change; the viewport listens first). */
function contourShowsNow(): Promise<void> {
  return new Promise((resolve) => {
    if (contourShown()) return resolve();
    const off = store.subscribe(() => {
      if (!contourShown()) return;
      off();
      resolve();
    });
  });
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
async function measureContour(path: string) {
  store.dispatch({ type: "clearSelection" });
  if (!(await effects.openResults(path))) throw new Error(`measure: results ${path} did not open: ${store.get().artifacts.results?.error}`);
  const s = store.get();
  if (!s.phase.axis.some((k) => k.kind === "results")) throw new Error(`measure: ${path} does not pair with the model, so there is nothing to scrub`);
  const stages = s.results!.stages;
  let best = -1;
  stages.forEach((st, i) => {
    if (st.nodal.length && (best < 0 || st.time.length > stages[best]!.time.length)) best = i;
  });
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
    if (!(await until(() => store.get().contour.scan?.status === "done", 600000))) throw new Error(`measure: the history scan did not finish: ${JSON.stringify(store.get().contour.scan)}`);
    const scanMs = performance.now() - t0;
    await contourShowsNow();
    const v = contourView(store.get());
    const field = v.state === "ready" ? `${v.field} ${v.component.label}` : "?";
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
  }
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
    await bridge.measureDone({ ...result, contour: cfg.results ? await measureContour(cfg.results) : null });
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
  }
  await bridge.captureDone();
}

main().catch((err) => {
  console.error(err);
  void bridge.fail(err instanceof Error ? `${err.message}\n${err.stack}` : String(err));
});
