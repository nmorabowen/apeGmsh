// The headless agent modes (ADR 0128 D4, D9): `--render` takes every shot of
// a validated view spec in this hidden window and `--describe` writes what
// the open set offers. Main validated the spec before the window opened
// (src/spec/spec.ts) and writes every file; this side opens the set, applies
// each shot through the one compiler (src/spec/compile.ts, INV-2), waits
// until the view asked for is on screen, and either takes the still or
// refuses the shot with the app's own reason (D2, INV-4). A refused shot
// writes no PNG; the other shots still run. Every shot ends as one D3 entry
// of `manifest.json` (src/spec/manifest.ts), built before the still is
// written, so no still stands without its numbers (INV-6).
//
// A shot is taken only after its step has been read, the view it asks for
// is drawn (the contour's colours and range, the deformed positions) and
// two frames have been painted (the rule `captureResults` follows). The
// camera is applied last, after the view is drawn, so a fit frames what is
// drawn; it is reset to `home` first, so `fit` never keeps the heading of
// the shot before it.

import * as THREE from "three";
import type { Effects } from "../effects.ts";
import { compileShot, type ShotPlan } from "../spec/compile.ts";
import { describeSet, type SetDescription } from "../spec/describe.ts";
import { manifestEntry, manifestOf, refusedEntry, type ManifestEntry } from "../spec/manifest.ts";
import type { SpecShot, ViewSpec } from "../spec/spec.ts";
import { viewVerdict } from "../spec/verdict.ts";
import type { BlobStore } from "../state/blobs.ts";
import { contourView, deformView } from "../state/selectors.ts";
import type { Store } from "../state/store.ts";
import type { State } from "../state/types.ts";
import { contourKey, elementContourKey } from "./contour.ts";
import { composeStill, loadStillFonts, type ScaleBar } from "./stillLayers.ts";
import type { Viewport } from "./viewport.ts";

export type RenderJob =
  | { mode: "render"; spec: ViewSpec; model: string; geometry: string | null; results: string | null }
  | { mode: "describe"; model: string | null; geometry: string | null; results: string | null };

/** The render mode's door to main (preload.ts); main writes every file, into `--out` only (INV-1). */
export interface RenderBridge {
  renderJob(): Promise<RenderJob>;
  /** a model-mode still, composed here: main writes `<out>/<name>.png` */
  renderPng(name: string, png: Uint8Array): Promise<{ path: string }>;
  /** an app-mode still: main captures the page into `<out>/<name>.png` */
  renderCapture(name: string): Promise<{ path: string; width: number; height: number }>;
  renderDone(result: unknown): Promise<void>;
}

/** D9: what the layout does with the app's panels for one shot (renderer/app.ts owns the panel names). */
export interface Layout {
  /** omit the panels of the named app layers that are off, or (model mode) every element but the viewport */
  apply(chrome: "app" | "model", layers: Record<string, boolean>): void;
  /** the body's children that hold a layout box now, besides the viewport (the DOM check) */
  inLayout(): string[];
}

export interface RenderDeps {
  store: Store;
  blobs: BlobStore;
  effects: Effects;
  viewport: Viewport;
  bridge: RenderBridge;
  layout: Layout;
}

/** One shot of the run: its D3 entry, and the shot's time, with the part spent until its view was on screen (the rest paints, composes and writes). */
export interface ShotResult {
  entry: ManifestEntry;
  ms: number;
  shownMs: number | null;
}

const painted = () => new Promise<void>((r) => requestAnimationFrame(() => setTimeout(r, 0)));
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Wait at most this long for a shot's view to reach the screen. */
const SHOW_MS = 30000;
/** A view that stays `unavailable` or `refused` this long, with nothing still loading, is the app's answer. */
const SETTLED_MS = 1500;

function contourOnScreen(s: State, viewport: Viewport): boolean {
  const v = contourView(s);
  if (v.state === "ready") return v.stepMeasured && viewport.contourApplied === contourKey(v);
  if (v.state === "element") return v.stepMeasured && viewport.contourApplied === elementContourKey(v);
  return false;
}

function deformOnScreen(s: State, viewport: Viewport, effects: Effects): boolean {
  const v = deformView(s);
  if (v.state !== "ready" && v.state !== "partial") return false;
  const d = effects.currentDisplacement();
  return viewport.deformApplied === `${d?.key}|scale ${v.scale}` && d?.key === v.keys.disp;
}

/**
 * Wait until the shot's view is on screen; null when it is, else the reason
 * it is refused (src/spec/verdict.ts). A contour or deformed shape the app
 * answers with `unavailable` or `refused` is refused with that reason once
 * it has held, with nothing loading, for SETTLED_MS: a stage's node ids and
 * an element join arrive after the step, and say `unavailable` until then.
 */
async function shown(deps: RenderDeps, shot: SpecShot, plan: ShotPlan): Promise<string | null> {
  const { store, viewport, effects } = deps;
  const t0 = performance.now();
  let lastWhy: string | null = null;
  let since = t0;
  for (;;) {
    const s = store.get();
    const now = performance.now();
    const v = viewVerdict(s, shot, plan, { contour: contourOnScreen(s, viewport), deform: deformOnScreen(s, viewport, effects) });
    if (v.kind === "shown") return null;
    if (v.kind === "refused") return v.reason;
    if (v.why !== lastWhy) {
      lastWhy = v.why;
      since = now;
    }
    if (v.why !== null && !v.loading && now - since >= SETTLED_MS) return v.why;
    if (now - t0 >= SHOW_MS) return v.why ?? `the view did not reach the screen within ${SHOW_MS / 1000} s`;
    await sleep(20);
  }
}

/** Each world axis's direction on screen (x right, y up) through the camera's rotation. */
function screenAxes(viewport: Viewport): { x: [number, number]; y: [number, number]; z: [number, number] } {
  const cam = viewport.camera;
  cam.updateMatrixWorld();
  const dir = (x: number, y: number, z: number): [number, number] => {
    const d = new THREE.Vector3(x, y, z).transformDirection(cam.matrixWorldInverse);
    return [d.x, d.y];
  };
  return { x: dir(1, 0, 0), y: dir(0, 1, 0), z: dir(0, 0, 1) };
}

function scaleBarOf(s: State, plan: ShotPlan): ScaleBar | null {
  const v = contourView(s);
  if ((v.state !== "ready" && v.state !== "element") || !plan.contour) return null;
  return { title: `${plan.contour.field} ${plan.contour.label}`, min: v.clim.min, max: v.clim.max, rangeLabel: v.clim.label, cmap: v.cmap };
}

async function toPng(canvas: HTMLCanvasElement): Promise<Uint8Array> {
  const blob = await new Promise<Blob | null>((r) => canvas.toBlob(r, "image/png"));
  if (!blob) throw new Error("render: the still canvas did not encode as PNG");
  return new Uint8Array(await blob.arrayBuffer());
}

/** Add the shot's section planes; each plane's id is the one the reducer gave it. */
function applyClip(store: Store, plan: ShotPlan): void {
  for (const p of store.get().clip.planes) store.dispatch({ type: "removeClipPlane", id: p.id });
  for (const c of plan.clip) {
    const n = store.get().clip.planes.length;
    store.dispatch({ type: "addClipPlane", normal: c.normal });
    const planes = store.get().clip.planes;
    // A plane the reducer refused adds nothing: its notice stays set, and the verdict refuses the shot with it.
    if (planes.length !== n + 1) return;
    const id = planes.at(-1)!.id;
    store.dispatch({ type: "setClipOffset", id, offset: c.offset });
    if (c.flip) store.dispatch({ type: "setClipFlipped", id, flipped: true });
  }
}

async function takeShot(deps: RenderDeps, spec: ViewSpec, shot: SpecShot, d: SetDescription): Promise<ShotResult> {
  const { store, viewport, bridge, layout, blobs, effects } = deps;
  const t0 = performance.now();
  const compiled = compileShot(shot, d);
  // Nothing was applied: this shot has no view to report, only the reason.
  if (!compiled.ok) return { entry: refusedEntry(store.get(), shot, compiled.reason, null), ms: performance.now() - t0, shownMs: null };
  const plan = compiled.plan;
  const at = {
    stage: plan.stage,
    step: plan.step,
    time: plan.stage && plan.step !== null ? (d.stages[plan.stage.index]?.times[plan.step] ?? null) : null,
  };
  const refuse = (reason: string): ShotResult => ({ entry: refusedEntry(store.get(), shot, reason, at), ms: performance.now() - t0, shownMs: null });
  try {
    for (const e of plan.events) store.dispatch(e);
    applyClip(store, plan);
  } catch (err) {
    return refuse(`the app refused the view: ${err instanceof Error ? err.message : String(err)}`);
  }
  // D9: layers are presentation; the view (and so any refusal) is decided before they are applied.
  const why = await shown(deps, shot, plan);
  if (why !== null) return refuse(why);
  const shownMs = performance.now() - t0;
  store.dispatch({ type: "setCamera", to: "home" });
  store.dispatch({ type: "setCamera", to: plan.camera });
  const canvas = viewport.renderer.domElement;
  const box = (): [number, number, number, number] => {
    const b = canvas.getBoundingClientRect();
    return [b.x, b.y, b.width, b.height];
  };
  const before = box();
  layout.apply(shot.chrome, shot.layers);
  viewport.renderNow();
  // D4: two painted frames after the last change (a hidden window's first capture after a burst of DOM changes can be the frame before them).
  await painted();
  await painted();
  viewport.renderNow();
  const s = store.get();
  // The DOM checks (D9): the layers moved nothing, a model still lays out the viewport alone, at the spec's size.
  const r = box();
  if (r.some((x, i) => x !== before[i])) throw new Error(`render ${shot.name}: the layers moved the viewport from [${before.join(", ")}] to [${r.join(", ")}]`);
  const panels = layout.inLayout();
  if (shot.chrome === "model" && panels.length) throw new Error(`render ${shot.name}: model mode left ${panels.join(", ")} in the layout`);
  if (canvas.width !== spec.size[0] || canvas.height !== spec.size[1]) {
    throw new Error(`render ${shot.name}: the viewport is ${canvas.width} x ${canvas.height} pixels, not the spec's ${spec.size[0]} x ${spec.size[1]}`);
  }
  // INV-6: the entry first. A view the manifest cannot state is refused, and no still is written for it.
  let entry: ManifestEntry;
  try {
    const cam = viewport.camera;
    entry = manifestEntry(s, {
      shot,
      png: `${shot.name}.png`,
      image: spec.size,
      camera: { ...viewport.currentPose(), fov: cam instanceof THREE.PerspectiveCamera ? cam.fov : null },
      layout: { viewport: r, panels },
      blobs,
      step: effects.currentStep(),
    });
  } catch (err) {
    return refuse(`the manifest cannot state this view: ${err instanceof Error ? err.message : String(err)}`);
  }
  if (shot.chrome === "model") {
    const out = composeStill({
      source: canvas,
      width: spec.size[0],
      height: spec.size[1],
      theme: s.theme,
      background: shot.background,
      layers: { scale_bar: shot.layers["scale_bar"]!, axes: shot.layers["axes"]!, banner: shot.layers["banner"]!, caption: shot.layers["caption"]! },
      scale: scaleBarOf(s, plan),
      axes: screenAxes(viewport),
      banner: entry.banner,
      caption: shot.caption,
    });
    if (out.width !== spec.size[0] || out.height !== spec.size[1]) throw new Error(`render ${shot.name}: the still is ${out.width} x ${out.height} pixels, not the spec's ${spec.size[0]} x ${spec.size[1]}`);
    await bridge.renderPng(shot.name, await toPng(out));
  } else {
    // Main checks the captured page is exactly the spec's size.
    await bridge.renderCapture(shot.name);
  }
  return { entry, ms: performance.now() - t0, shownMs };
}

/** The drawn vertex positions (undeformed), for the description's box. */
function drawnPositions(s: State, blobs: BlobStore): ArrayLike<number>[] {
  if (!s.mesh) return [];
  return [blobs.get(s.mesh.linePositions), blobs.get(s.mesh.triPositions)];
}

/** Open the set the job names and wait for the first frame. */
async function openSet(deps: RenderDeps, job: { model: string | null; geometry: string | null; results: string | null }): Promise<string | null> {
  const { effects, viewport } = deps;
  if (job.model && !(await effects.open(job.model))) return `the model ${job.model} did not open: ${deps.store.get().artifacts.model?.error ?? "unknown error"}`;
  if (job.geometry) await effects.openGeometry(job.geometry);
  if (job.results) await effects.openResults(job.results);
  viewport.renderNow();
  await painted();
  return null;
}

export async function runAgentMode(deps: RenderDeps): Promise<void> {
  const { store, bridge } = deps;
  const job = await bridge.renderJob();
  // INV-7: one framing for every chrome, so app and model stills of one camera show the same model at the same pose.
  deps.viewport.setRailCollapsed(true);
  if (job.mode === "describe") {
    const failed = await openSet(deps, job);
    if (failed) throw new Error(failed);
    await bridge.renderDone({ describe: describeSet(store.get(), drawnPositions(store.get(), deps.blobs)) });
    return;
  }
  // D8: the spec's theme (light by default for batch stills), before anything is drawn.
  store.dispatch({ type: "setTheme", theme: job.spec.theme });
  await loadStillFonts();
  const failed = await openSet(deps, job);
  const d = describeSet(store.get(), drawnPositions(store.get(), deps.blobs));
  const shots: ShotResult[] = [];
  for (const shot of job.spec.shots) {
    if (failed) {
      shots.push({ entry: refusedEntry(store.get(), shot, failed, null), ms: 0, shownMs: null });
      continue;
    }
    const r = await takeShot(deps, job.spec, shot, d);
    console.info(`apeGmshViewer: render ${shot.name}: ${r.entry.status}${r.entry.reason ? ` (${r.entry.reason})` : ""} in ${Math.round(r.ms)} ms`);
    shots.push(r);
  }
  await bridge.renderDone({ manifest: manifestOf(job.spec, shots) });
}
