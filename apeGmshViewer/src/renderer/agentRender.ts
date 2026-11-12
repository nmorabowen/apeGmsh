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
//
// VU18: a shot may name a variant row of the spec's `files.variants` by index
// and label (`variant`). Before such a shot, main resolves the row afresh
// (`renderVariant`: the manifest read again, D9.3) and names the row's set,
// which this side opens in place of the one on screen; a later shot without a
// row asks for the spec's set back. A row main refuses refuses the shot with
// main's reason, and nothing is drawn for it from the set on screen.

import * as THREE from "three";
import type { Effects } from "../effects.ts";
import { compileShot, type ShotPlan } from "../spec/compile.ts";
import { describeSet, type SetDescription } from "../spec/describe.ts";
import { manifestEntry, manifestOf, refusedEntry, type ManifestEntry } from "../spec/manifest.ts";
import type { SpecShot, SpecVariant, ViewSpec } from "../spec/spec.ts";
import { applyClip, shownView } from "../spec/snapshot.ts";
import type { OnScreen } from "../spec/verdict.ts";
import type { BlobStore } from "../state/blobs.ts";
import { combinationFieldsOf, contourView, deformView, userResultPickOf, variantRowOf } from "../state/selectors.ts";
import type { Store } from "../state/store.ts";
import type { State, VariantsIn } from "../state/types.ts";
import { combinationContourKey, contourKey, differenceContourKey, differenceHeld, elementContourKey, userResultContourKey, type CombinationColumnOf } from "./contour.ts";
import { composeStill, loadStillFonts, type ScaleBar } from "./stillLayers.ts";
import type { Viewport } from "./viewport.ts";

export type RenderJob =
  | { mode: "render"; spec: ViewSpec; model: string; geometry: string | null; results: string | null }
  | { mode: "describe"; model: string | null; geometry: string | null; results: string | null };

/** VU18: main's answer to `renderVariant`: the set the shot draws (the files to open), or why the row is refused. */
export type VariantAnswer =
  | { ok: true; row: SpecVariant | null; set: { model: string; geometry: string | null; results: string | null } }
  | { ok: false; reason: string };

/** The render mode's door to main (preload.ts); main writes every file, into `--out` only (INV-1). */
export interface RenderBridge {
  renderJob(): Promise<RenderJob>;
  /** VU18: the set the next shot draws: a variant row by index and label, or null for the spec's own set (main/modes/render.ts) */
  renderVariant(row: SpecVariant | null): Promise<VariantAnswer>;
  /** VU10: the study main holds now (preload.ts `variants`); read again after main changed the drawn set or the compare slot */
  variants(): Promise<VariantsIn | null>;
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
  /** VU7b-2: the User Combination column on screen (renderer/contour.ts `CombinationFeed.held`), for the manifest */
  combination: () => CombinationColumnOf | null;
}

/** One shot of the run: its D3 entry, and the shot's time, with the part spent until its view was on screen (the rest paints, composes and writes). */
export interface ShotResult {
  entry: ManifestEntry;
  ms: number;
  shownMs: number | null;
}

const painted = () => new Promise<void>((r) => requestAnimationFrame(() => setTimeout(r, 0)));
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function contourOnScreen(s: State, viewport: Viewport, effects: Effects): boolean {
  const v = contourView(s);
  // VU7b-2: a User Combination is one state, with no step to read.
  if (v.state === "combination") return viewport.contourApplied === combinationContourKey(v);
  // VU10: a difference is drawn from its own read, which waited for the window's step.
  if (v.state === "difference") return viewport.contourApplied === differenceContourKey(v);
  // VU16: a user result is one state, read on its own, at whatever phase is shown.
  if (v.state === "user") return viewport.contourApplied === userResultContourKey(v);
  // A shot may repeat the last shot's view: the applied key already matches while its step is read again, and the manifest needs that step's values.
  const at = s.phase.at;
  const step = effects.currentStep();
  if (at?.kind !== "results" || step?.stage !== at.stage || step.step !== at.step) return false;
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

/** What the viewport shows now, for the verdict (ADR 0128 D2): the render mode and an applied offer ask the same question. */
export function onScreenOf(s: State, viewport: Viewport, effects: Effects): OnScreen {
  return { contour: contourOnScreen(s, viewport, effects), deform: deformOnScreen(s, viewport, effects) };
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
  if ((v.state !== "ready" && v.state !== "element" && v.state !== "combination" && v.state !== "difference" && v.state !== "user") || !plan.contour) return null;
  // VU7b-2: a combination's scale names the combination, as the app's does; VU10: a difference names its two variants.
  // VU16: a user result's field is its name, so the title already names it.
  const what = v.state === "combination" ? `${v.name} · ` : v.state === "difference" ? `${v.window} − ${v.compare} · ` : "";
  return { title: `${what}${plan.contour.field} ${plan.contour.label}`, min: v.clim.min, max: v.clim.max, rangeLabel: v.clim.label, cmap: v.cmap };
}

async function toPng(canvas: HTMLCanvasElement): Promise<Uint8Array> {
  const blob = await new Promise<Blob | null>((r) => canvas.toBlob(r, "image/png"));
  if (!blob) throw new Error("render: the still canvas did not encode as PNG");
  return new Uint8Array(await blob.arrayBuffer());
}

async function takeShot(deps: RenderDeps, spec: ViewSpec, shot: SpecShot, d: SetDescription): Promise<ShotResult> {
  const { store, viewport, bridge, layout, blobs, effects } = deps;
  const t0 = performance.now();
  const compiled = compileShot(
    shot,
    d,
    (name) => combinationFieldsOf(store.get(), name),
    (label) => variantRowOf(store.get(), label),
    (name) => {
      const p = userResultPickOf(store.get(), name);
      return p.ok ? { ok: true, components: p.entry.components } : p;
    },
  );
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
  const why = await shownView({ store, onScreen: (st) => onScreenOf(st, viewport, effects), sleep, now: () => performance.now() }, shot, plan);
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
      combination: deps.combination(),
      difference: differenceHeld.current,
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

/**
 * VU18: draw `want` (a variant row, or null for the spec's set) for the next
 * shot. Main resolves it now and names the set; each file that differs from
 * the one on screen is opened, and a slot the set lacks is closed. Answers
 * the description of the set now on screen, or why the shot cannot be drawn.
 */
async function drawSet(deps: RenderDeps, want: SpecVariant | null): Promise<{ ok: true; d: SetDescription } | { ok: false; reason: string }> {
  const { store, effects, bridge, viewport } = deps;
  const a = await bridge.renderVariant(want);
  if (!a.ok) return a;
  const on = store.get().artifacts;
  const modelChanged = on.model?.path !== a.set.model || on.model.status !== "ready";
  if (modelChanged && !(await effects.open(a.set.model))) {
    return { ok: false, reason: `the model ${a.set.model} did not open: ${store.get().artifacts.model?.error ?? "unknown error"}` };
  }
  if (a.set.geometry === null) {
    if (on.geometry !== null) store.dispatch({ type: "fileClosed", artifact: "geometry" });
  } else if (modelChanged || on.geometry?.path !== a.set.geometry) await effects.openGeometry(a.set.geometry);
  if (a.set.results === null) {
    if (on.results !== null) store.dispatch({ type: "fileClosed", artifact: "results" });
  } else if (modelChanged || on.results?.path !== a.set.results) await effects.openResults(a.set.results);
  // VU9a: another model's set has another workspace (main reads the drawn set's).
  if (modelChanged) await effects.readWorkspace();
  // The study as main holds it now: the row on screen, and a compare slot main emptied because it is that row.
  store.dispatch({ type: "variantsRead", study: await bridge.variants() });
  viewport.renderNow();
  await painted();
  return { ok: true, d: describeSet(store.get(), drawnPositions(store.get(), deps.blobs)) };
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
  // VU18: a spec that draws variant rows shows no user file in any shot (spec.ts refuses its user shots):
  // the spec set's `.user.h5`, read before the run, is not the file of a row's set.
  if (job.spec.shots.some((s) => s.variant !== undefined)) store.dispatch({ type: "userClosed" });
  const failed = await openSet(deps, job);
  const d = describeSet(store.get(), drawnPositions(store.get(), deps.blobs));
  await bridge.renderDone({ manifest: manifestOf(job.spec, await takeShots(deps, job.spec, d, failed)) });
}

/**
 * Every shot of the spec, each ending as one entry. INV-6: a shot that throws
 * (a DOM check, a still that could not be written) is refused with what it
 * threw, and the run goes on to its manifest, so the stills already written
 * never stand without one. Main writes a still as its handler's last step, so
 * a shot that threw has no PNG.
 */
export async function takeShots(deps: RenderDeps, spec: ViewSpec, d: SetDescription, failed: string | null): Promise<ShotResult[]> {
  const shots: ShotResult[] = [];
  // VU18: the row whose set is on screen; null is the spec's own set, which runAgentMode opened. "unknown" after a failed switch.
  let on: SpecVariant | null | "unknown" = null;
  for (const shot of spec.shots) {
    if (failed) {
      shots.push({ entry: refusedEntry(deps.store.get(), shot, failed, null), ms: 0, shownMs: null });
      continue;
    }
    // VU18: a shot naming a row always asks main (the manifest is read again for it, D9.3); a shot of the spec's set asks only after another set was drawn.
    const want = shot.variant ?? null;
    if (want !== null || on !== null) {
      const t0 = performance.now();
      let sw: Awaited<ReturnType<typeof drawSet>>;
      try {
        sw = await drawSet(deps, want);
      } catch (err) {
        sw = { ok: false, reason: `the set did not open: ${err instanceof Error ? err.message : String(err)}` };
      }
      if (!sw.ok) {
        on = "unknown";
        const what = want === null ? "the spec's set" : `variant row ${want.index} (${JSON.stringify(want.label)})`;
        const refused: ShotResult = { entry: refusedEntry(deps.store.get(), shot, `${what} is not drawn: ${sw.reason}`, null), ms: performance.now() - t0, shownMs: null };
        console.info(`apeGmshViewer: render ${shot.name}: refused (${refused.entry.reason})`);
        shots.push(refused);
        continue;
      }
      on = want;
      d = sw.d;
    }
    let r: ShotResult;
    try {
      r = await takeShot(deps, spec, shot, d);
    } catch (err) {
      r = { entry: refusedEntry(deps.store.get(), shot, `the shot failed: ${err instanceof Error ? err.message : String(err)}`, null), ms: 0, shownMs: null };
    }
    console.info(`apeGmshViewer: render ${shot.name}: ${r.entry.status}${r.entry.reason ? ` (${r.entry.reason})` : ""} in ${Math.round(r.ms)} ms`);
    shots.push(r);
  }
  return shots;
}
