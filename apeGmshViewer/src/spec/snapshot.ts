// The live side of the view spec (ADR 0128 D5, D6): an offer applied in the
// human's window, and the view read back as a spec. No DOM and no Electron;
// the window (renderer/app.ts) hands in the camera and the paint, and the
// tests run it in Node on the fixtures.
//
// INV-2: an offer applies the plan of the one compiler (./compile.ts), so a
// still and an applied view come from the same events. Before them it puts
// back what a still never changes because a new window never has it (the
// opacity, the edges, the loads a human turned on): `freshViewEvents`, without
// which the human would not see the view the agent's still showed.
//
// INV-3: `snapshotOf` reads the store's state back into a spec, and that spec
// compiles to the same plan. Right after an offer was applied it returns that
// offer's own words (its name, its "last", its "iso"); once the human has
// moved anything, it returns the view as it is, named "view".
//
// INV-5: nothing here runs on the arrival of an offer but the assessment,
// which only lists. The view moves on a click: `applyOffer` or `backOffer`,
// which only panels/offers.ts dispatches.
//
// What the spec cannot say (a line diagram on, a set opacity, a hidden group
// outside an isolate) is never silently dropped from a snapshot: `outside`
// names it, and goes into the manifest entry beside the closed key set (D1).

import type { ResolvedFiles } from "./files.ts";
import { defaultOpacity } from "../state/reduce.ts";
import { MAGNITUDE } from "../state/contour.ts";
import type { Store } from "../state/store.ts";
import type { CameraPose, DeclPath, Event, Offer, OfferIn, State, Vec3, ViewMemo } from "../state/types.ts";
import type { ThemeMode } from "../theme/tokens.ts";
import { compileShot, type ShotPlan } from "./compile.ts";
import { describeSet, SPEC_MAGNITUDE, type SetDescription } from "./describe.ts";
import { APP_LAYERS, SPEC_VERSION, validateSpec, type SpecShot, type ViewSpec } from "./spec.ts";
import { viewVerdict, type OnScreen } from "./verdict.ts";

/** The longest spec `--present` takes: more shots than the list holds would drop its own first shots unseen (D5). */
export const MAX_PRESENT_SHOTS = 8;

/** The spec's side of an offer: opaque to the store and the panels (Offer.payload). */
export interface OfferPayload {
  /** the set, as the present process resolved it by ADR 0112's pairing rules */
  files: { model: string; geometry: string | null; results: string | null };
  /** D8: the theme the spec names; null leaves the human's alone */
  theme: ThemeMode | null;
  shot: SpecShot;
  /** whether paths compare ignoring case (main's file system, main/pairing.ts FOLD_CASE) */
  fold: boolean;
}

export type Presented = { ok: true; offers: OfferIn[] } | { ok: false; key: string; reason: string };

const IGNORED = "A still's chrome, layers and background are not applied to this window.";

/** D9: whether a shot carries a still-only key this window will not apply. */
function stillOnly(shot: SpecShot): boolean {
  return shot.chrome !== "app" || shot.background !== "theme" || APP_LAYERS.some((k) => shot.layers[k] === false);
}

/**
 * The offers of one parsed spec document, or the whole spec refused (D2): the
 * present process runs this before it reaches the window, so the window only
 * ever hears valid specs. One offer per shot, since a name and a caption
 * belong to a shot. `resolve` is ./files.ts `resolveSpecFiles` bound to the
 * spec's folder; it comes in so this module stays free of `node:path`.
 */
export function offersOfDocument(raw: unknown, resolve: (spec: ViewSpec) => ResolvedFiles, fold: boolean): Presented {
  const v = validateSpec(raw);
  if (!v.ok) return v;
  if (v.spec.shots.length > MAX_PRESENT_SHOTS) {
    return { ok: false, key: "shots", reason: `holds ${v.spec.shots.length} shots; the window lists at most ${MAX_PRESENT_SHOTS} offers, so the first would be dropped unseen` };
  }
  const f = resolve(v.spec);
  if (!f.ok) return f;
  // D8: the validator fills in the batch default; an offer needs to know whether the spec named a theme at all.
  const named = (raw as { theme?: unknown }).theme !== undefined;
  const files = { model: f.set.model, geometry: f.set.geometry, results: f.set.results };
  return {
    ok: true,
    offers: v.spec.shots.map((shot) => {
      const payload: OfferPayload = { files, theme: named ? v.spec.theme : null, shot, fold };
      return { name: shot.name, caption: shot.caption, ignored: stillOnly(shot) ? IGNORED : null, payload };
    }),
  };
}

/** The offer's spec side; anything else in the list is a bug, and says so. */
function payloadOf(o: Offer): OfferPayload {
  const p = o.payload as OfferPayload | null;
  if (!p || typeof p !== "object" || !p.files || !p.shot) throw new Error(`offer ${o.id} (${o.name}) carries no spec`);
  return p;
}

const same = (a: string | null, b: string | null, fold: boolean): boolean => {
  if (a === null || b === null) return a === b;
  return fold ? a.toLowerCase() === b.toLowerCase() : a === b;
};

/** D5: the files of an offer outside the open set; a click opens them first. */
function opensOf(s: State, p: OfferPayload): string[] {
  const out: string[] = [];
  if (!same(s.artifacts.model?.path ?? null, p.files.model, p.fold)) out.push(p.files.model);
  if (p.files.results !== null && !same(s.artifacts.results?.path ?? null, p.files.results, p.fold)) out.push(p.files.results);
  return out;
}

const describe = (s: State): SetDescription => describeSet(s, []);

// ---- the view a human's panels can change and a new window does not have ------

/** What a window that just opened this set shows, for everything the plan's base events do not set. */
function freshViewEvents(s: State): Event[] {
  const declared = s.contour.declaredUnits;
  return [
    { type: "setEdges", on: true },
    ...(s.mesh ? [{ type: "setOpacity", value: defaultOpacity(s.mesh) } as const] : []),
    { type: "toggleExtrudeFrames", on: false },
    { type: "toggleShowOffsets", on: Boolean(s.mesh?.frameExtrusion?.hasJointOffsets) },
    { type: "setLoads", on: false },
    { type: "setApplyCuts", on: true },
    { type: "setCutFace", on: true },
    { type: "setContourUnits", source: declared, target: declared ?? "as_stored" },
  ];
}

/** Everything an offer changes, for Back. */
export function viewMemoOf(s: State, pose: CameraPose | null, model: string): ViewMemo {
  const c = s.contour;
  return {
    files: { model, results: s.artifacts.results?.path ?? null },
    at: s.phase.at,
    picks: s.selection.picks,
    visibility: s.visibility,
    contour: { source: c.source, field: c.field, component: c.component, gauss: c.gauss, range: c.range, fixed: c.fixed, cmap: c.cmap, sourceUnits: c.sourceUnits, targetUnits: c.targetUnits },
    deform: { on: s.deform.on, scale: s.deform.scale, ghost: s.deform.ghost },
    clip: { planes: s.clip.planes, applyCuts: s.clip.applyCuts, showGizmos: s.clip.showGizmos, cutFace: s.clip.cutFace },
    lineDiagram: s.lineDiagram.on,
    loads: s.loads.on,
    camera: pose,
    theme: s.theme,
  };
}

/** The shot's section planes through the store; each plane's id is the one the reducer gave it (the same steps as the render mode). */
function applyClip(store: Store, plan: ShotPlan): void {
  for (const p of store.get().clip.planes) store.dispatch({ type: "removeClipPlane", id: p.id });
  for (const c of plan.clip) {
    const n = store.get().clip.planes.length;
    store.dispatch({ type: "addClipPlane", normal: c.normal });
    const planes = store.get().clip.planes;
    // A plane the reducer refused adds nothing: its notice stays set, and the verdict refuses the offer with it.
    if (planes.length !== n + 1) return;
    const id = planes.at(-1)!.id;
    store.dispatch({ type: "setClipOffset", id, offset: c.offset });
    if (c.flip) store.dispatch({ type: "setClipFlipped", id, flipped: true });
  }
}

/** Put the view of `memo` back, through the same events a human's panels send. The set must be the memo's. */
export function restoreView(store: Store, memo: ViewMemo): void {
  const d = (e: Event) => store.dispatch(e);
  const { contour: c, visibility: v, clip } = memo;
  d({ type: "clearSelection" });
  if (memo.at) {
    if (memo.at.kind === "results") {
      d({ type: "setPhase", at: { ...memo.at, step: 0 } });
      if (memo.at.step > 0) d({ type: "setResultStep", step: memo.at.step });
    } else d({ type: "setPhase", at: memo.at });
  }
  d({ type: "showAll" });
  if (v.hidden.length) d({ type: "setHidden", decls: v.hidden, hidden: true });
  d({ type: "setEdges", on: v.edges });
  d({ type: "setOpacity", value: v.opacity });
  d({ type: "toggleExtrudeFrames", on: v.extrudeFrames });
  d({ type: "toggleShowOffsets", on: v.showOffsets });
  // The pick only where the phase shown has it: a pick is checked against the stage shown.
  if (c.field !== null && c.component !== null) d({ type: "setContourField", field: c.field, component: c.component, source: c.source });
  d({ type: "setContourGauss", mode: c.gauss });
  d(c.range === "fixed" && c.fixed ? { type: "setContourRange", range: "fixed", fixed: c.fixed } : { type: "setContourRange", range: c.range === "fixed" ? "history" : c.range });
  d({ type: "setContourCmap", cmap: c.cmap });
  d({ type: "setContourUnits", source: c.sourceUnits, target: c.targetUnits });
  d({ type: "setColourBy", by: v.colourBy });
  d({ type: "setDeform", on: memo.deform.on });
  d({ type: "setDeformScale", scale: memo.deform.scale });
  d({ type: "setReferenceGhost", on: memo.deform.ghost });
  for (const p of store.get().clip.planes) d({ type: "removeClipPlane", id: p.id });
  for (const p of clip.planes) {
    d({ type: "addClipPlane", normal: p.normal });
    const id = store.get().clip.planes.at(-1)!.id;
    d({ type: "setClipOffset", id, offset: p.offset });
    if (p.flipped) d({ type: "setClipFlipped", id, flipped: true });
    if (!p.gizmo) d({ type: "setClipGizmo", id, visible: false });
    if (!p.active) d({ type: "setClipActive", id, active: false });
  }
  d({ type: "setApplyCuts", on: clip.applyCuts });
  d({ type: "setShowGizmos", on: clip.showGizmos });
  d({ type: "setCutFace", on: clip.cutFace });
  d({ type: "setLineDiagram", on: memo.lineDiagram });
  d({ type: "setLoads", on: memo.loads });
  d({ type: "setTheme", theme: memo.theme });
  memo.picks.forEach((pick, i) => d({ type: i === 0 ? "select" : "selectAdd", pick }));
  if (memo.camera) d({ type: "setCamera", to: memo.camera });
}

// ---- the view read back as a spec (D1, D6) -------------------------------------

/** The closed-key document of a validated spec: `validateSpec(toDocument(s))` gives `s` back. */
export function toDocument(spec: ViewSpec): Record<string, unknown> {
  return {
    spec: spec.spec,
    files: spec.files.results === null ? { model: spec.files.model } : { model: spec.files.model, results: spec.files.results },
    size: spec.size,
    theme: spec.theme,
    shots: spec.shots.map(shotDocument),
  };
}

function shotDocument(s: SpecShot): Record<string, unknown> {
  const c = s.contour;
  return {
    name: s.name,
    ...(s.caption === null ? {} : { caption: s.caption }),
    ...(s.stage === null ? {} : { stage: s.stage, step: s.step }),
    camera: s.camera,
    colour_by: s.colour_by,
    ...(c === null
      ? {}
      : {
          contour: {
            field: c.field,
            component: c.component,
            ...(c.source === null ? {} : { source: c.source }),
            range: c.range.mode === "fixed" ? { min: c.range.min, max: c.range.max } : c.range.mode,
            cmap: c.cmap,
            gauss: c.gauss,
          },
        }),
    ...(s.deform === null ? {} : { deform: s.deform }),
    ...(s.clip.length ? { clip: s.clip } : {}),
    ...(s.isolate.length ? { isolate: s.isolate } : {}),
    chrome: s.chrome,
    layers: s.layers,
    background: s.background,
  };
}

/** D9: a snapshot writes the window as it is. */
const WINDOW_LAYERS: Record<string, boolean> = Object.fromEntries(APP_LAYERS.map((k) => [k, true]));

/** The stage as a spec names it: its name when that names one stage, else its index. */
function stageName(s: State, index: number): string | number {
  const stages = s.results?.stages ?? [];
  const name = stages[index]?.name;
  return name && stages.filter((x) => x.name === name).length === 1 ? name : index;
}

function groupPaths(s: State): DeclPath[] {
  return Object.values(s.decls).filter((d) => d.kind === "physical_group").map((d) => d.path);
}

/** The current view as a shot; `outside` names what a v1 spec cannot say about it. */
export function viewShotOf(s: State, pose: CameraPose): { shot: SpecShot; outside: string[] } {
  const outside: string[] = [];
  const at = s.phase.at;
  const results = at?.kind === "results" ? at : null;
  if (at && at.kind !== "results" && at.kind !== "mesh") outside.push(`the ${at.kind} phase is shown; the spec names the mesh or a results stage`);

  const colourBy = s.visibility.colourBy;
  const c = s.contour;
  const drawsContour = colourBy === "contour" && results !== null && c.field !== null && c.component !== null;
  if (colourBy === "contour" && !drawsContour) outside.push("colouring by contour with no contour drawn");
  const contour: SpecShot["contour"] = drawsContour
    ? {
        field: c.field!,
        component: c.component === MAGNITUDE ? SPEC_MAGNITUDE : c.component!,
        source: c.source,
        range: c.range === "fixed" && c.fixed ? { mode: "fixed", min: c.fixed.min, max: c.fixed.max } : { mode: c.range === "fixed" ? "history" : c.range },
        cmap: c.cmap,
        gauss: c.gauss,
      }
    : null;

  // Maintainer ruling on #2651: v1 refuses the reference ghost, so a snapshot cannot carry it.
  if (s.deform.on && s.deform.ghost) outside.push("the reference ghost is on; apegmsh-view/1 does not draw it");
  if (s.deform.on && !results) outside.push("the deformed shape is on, with no results step to draw");
  const deform: SpecShot["deform"] = s.deform.on && results ? { scale: s.deform.scale, ghost: false } : null;

  const hidden = new Set(s.visibility.hidden);
  const groups = groupPaths(s);
  const kept = groups.filter((p) => !hidden.has(p));
  const isolates = hidden.size > 0 && kept.length > 0 && groups.length - kept.length === hidden.size;
  if (hidden.size > 0 && !isolates) outside.push(`${hidden.size} hidden declarations that are not an isolate of physical groups`);

  const cuts = s.clip.applyCuts ? s.clip.planes.filter((p) => p.active) : [];
  if (!s.clip.applyCuts && s.clip.planes.length) outside.push("section planes with Apply cuts off");
  if (s.clip.applyCuts && cuts.length !== s.clip.planes.length) outside.push(`${s.clip.planes.length - cuts.length} inactive section planes`);
  if (!s.clip.cutFace) outside.push("the contour on the cut face is off");

  if (!s.visibility.edges) outside.push("the mesh edges are off");
  if (s.mesh && s.visibility.opacity !== defaultOpacity(s.mesh)) outside.push(`the opacity is ${s.visibility.opacity}`);
  if (s.visibility.extrudeFrames) outside.push("the frames are drawn extruded");
  if (s.visibility.showOffsets !== Boolean(s.mesh?.frameExtrusion?.hasJointOffsets)) outside.push(`the joint offsets are ${s.visibility.showOffsets ? "shown" : "hidden"}`);
  if (s.lineDiagram.on) outside.push("a line diagram is on");
  if (s.loads.on) outside.push("the loads are on");
  if (c.targetUnits !== (c.declaredUnits ?? "as_stored")) outside.push(`the contour is displayed in ${c.targetUnits}`);

  const shot: SpecShot = {
    name: "view",
    caption: null,
    stage: results ? stageName(s, results.stage) : null,
    step: results ? results.step : "last",
    camera: { eye: pose.eye, target: pose.target, up: pose.up },
    colour_by: drawsContour ? "contour" : colourBy === "contour" ? "group" : colourBy,
    contour,
    deform,
    clip: cuts.map((p) => ({ normal: p.normal, offset: p.offset, flip: p.flipped })),
    isolate: isolates ? kept : [],
    chrome: "app",
    layers: WINDOW_LAYERS,
    background: "theme",
  };
  return { shot, outside };
}

/** A plan in the form two plans of one view share: planes as unit normals, an isolate as a set. */
function planKey(p: ShotPlan): string {
  const unit = (v: Vec3): number[] => {
    const m = Math.hypot(...v);
    return v.map((x) => x / m);
  };
  return JSON.stringify({
    phase: p.phase,
    events: p.events.map((e) => (e.type === "isolate" ? { ...e, decls: [...e.decls].sort() } : e)),
    clip: p.clip.map((c) => ({ normal: unit(c.normal), offset: c.offset, flip: c.flip })),
    contour: p.contour,
    deform: p.deform,
  });
}

/** Two shots of one set draw one view (INV-2), the camera aside. */
export function sameView(a: SpecShot, b: SpecShot, d: SetDescription): boolean {
  const pa = compileShot(a, d);
  const pb = compileShot(b, d);
  return pa.ok && pb.ok && planKey(pa.plan) === planKey(pb.plan);
}

/** INV-3's float tolerance on a camera: relative to the eye-to-target distance. */
export function samePoseWithin(a: CameraPose, b: CameraPose, tol = 1e-6): boolean {
  const dist = Math.hypot(a.eye[0] - a.target[0], a.eye[1] - a.target[1], a.eye[2] - a.target[2]);
  const close = (u: Vec3, v: Vec3, scale: number) => u.every((x, i) => Math.abs(x - v[i]!) <= tol * scale);
  return close(a.eye, b.eye, dist) && close(a.target, b.target, dist) && close(a.up, b.up, 1);
}

/** An offer the human applied: its words, and the camera it left, for the snapshot taken before anything moves. */
export interface AppliedView {
  files: OfferPayload["files"];
  fold: boolean;
  shot: SpecShot;
  pose: CameraPose;
}

export interface Snapshot {
  /** the closed spec document, `view.json` */
  view: Record<string, unknown>;
  /** D6: the decl paths the inspector shows */
  selection: DeclPath[];
  /** what the spec cannot say about this view; [] when it says all of it */
  outside: string[];
  camera: CameraPose;
}

export type SnapshotResult = { ok: true; snapshot: Snapshot } | { ok: false; reason: string };

/** D6: the view as a spec. Pure: it changes nothing, and the window's theme and size go in as they are. */
export function snapshotOf(s: State, pose: CameraPose | null, size: [number, number], applied: AppliedView | null): SnapshotResult {
  const model = s.artifacts.model?.path ?? null;
  if (model === null || s.artifacts.model?.status !== "ready") return { ok: false, reason: "the window has no model open, so it has no view to read back" };
  if (pose === null) return { ok: false, reason: "the window has not framed a view yet" };
  const results = s.artifacts.results?.path ?? null;
  const { shot: derived, outside } = viewShotOf(s, pose);
  let shot = derived;
  if (
    applied &&
    outside.length === 0 &&
    same(applied.files.model, model, applied.fold) &&
    same(applied.files.results, results, applied.fold) &&
    samePoseWithin(applied.pose, pose) &&
    sameView(applied.shot, derived, describe(s))
  ) {
    shot = { ...applied.shot, chrome: "app", layers: WINDOW_LAYERS, background: "theme" };
  }
  const spec: ViewSpec = { spec: SPEC_VERSION, files: { model, results }, size, theme: s.theme, shots: [shot] };
  const doc = toDocument(spec);
  // A snapshot is a spec an agent can render: one the validator refuses is a bug here, and says so.
  const v = validateSpec(doc);
  if (!v.ok) return { ok: false, reason: `the view does not read back as a spec: ${v.key}: ${v.reason}` };
  return { ok: true, snapshot: { view: doc, selection: [...s.selection.decls], outside, camera: pose } };
}

// ---- the offer runner: carries out the human's clicks --------------------------

/** Wait at most this long for a view, or a set, to reach the screen. */
const SHOW_MS = 30000;
/** A view that stays refused this long, with nothing still loading, is the app's answer (renderer/agentRender.ts `shown`). */
const SETTLED_MS = 1500;

export interface RunnerDeps {
  store: Store;
  /** open the set of `path`, as a drop does (main's requestOpen) */
  open(path: string): Promise<{ ok: boolean; reason?: string } | unknown>;
  /** what the viewport shows now, for the verdict */
  onScreen(): OnScreen;
  /** the camera the viewport shows, once the last request has been applied */
  pose(): CameraPose | null;
  /** two painted frames */
  settle(): Promise<void>;
  sleep?(ms: number): Promise<unknown>;
  now?(): number;
}

export class OfferRunner {
  private readonly deps: RunnerDeps;
  private readonly sleep: (ms: number) => Promise<unknown>;
  private readonly now: () => number;
  private seen = 0;
  private applied: (AppliedView | null)[] = [];
  private idle: Promise<void> = Promise.resolve();
  private assessKey: unknown[] = [];

  constructor(deps: RunnerDeps) {
    this.deps = deps;
    this.sleep = deps.sleep ?? ((ms) => new Promise((r) => setTimeout(r, ms)));
    this.now = deps.now ?? (() => performance.now());
  }

  /** Follow the store; the returned function stops. */
  start(): () => void {
    const { store } = this.deps;
    const off = store.subscribe(() => queueMicrotask(() => this.tick()));
    this.tick();
    return off;
  }

  /** `--present` reached the window: it lists, and nothing else happens (INV-5). */
  receive(offers: readonly OfferIn[], at: number = Date.now()): void {
    this.deps.store.dispatch({ type: "offerReceived", offers, at });
  }

  /** Resolves when the click being carried out has finished. */
  whenIdle(): Promise<void> {
    return this.idle;
  }

  /** D6: the current view; never changes the window (INV-5). */
  snapshot(size: [number, number]): SnapshotResult {
    return snapshotOf(this.deps.store.get(), this.deps.pose(), size, this.applied.at(-1) ?? null);
  }

  private tick(): void {
    this.assess();
    const r = this.deps.store.get().offers.request;
    if (r && r.seq > this.seen) {
      this.seen = r.seq;
      this.idle = r.kind === "apply" ? this.apply(r.id) : this.back();
    }
  }

  /** Judge the listed offers against the open set (D5): only ever lists. */
  private assess(): void {
    const { store } = this.deps;
    const s = store.get();
    const key = [s.offers.list, s.artifacts.model, s.artifacts.results, s.results, s.decls];
    if (key.every((k, i) => k === this.assessKey[i])) return;
    this.assessKey = key;
    // Being read: the answer comes when the read lands.
    if (s.artifacts.model?.status === "opened" || s.artifacts.results?.status === "opened") return;
    let d: SetDescription | null = null;
    for (const o of s.offers.list) {
      const p = payloadOf(o);
      const opens = opensOf(s, p);
      let refused: string | null = null;
      // Files outside the open set cannot be judged until they are open: the click finds out.
      if (opens.length === 0) {
        d ??= describe(s);
        const c = compileShot(p.shot, d);
        refused = c.ok ? null : c.reason;
      }
      store.dispatch({ type: "offerAssessed", id: o.id, refused, opens });
    }
  }

  private async apply(id: number): Promise<void> {
    const { store } = this.deps;
    const before = store.get();
    const offer = before.offers.list.find((o) => o.id === id);
    if (!offer) {
      // The cap dropped it between the click and now: the click ends, and the view stays.
      store.dispatch({ type: "offerRefused", id, reason: `offer ${id} has left the list` });
      return;
    }
    const p = payloadOf(offer);
    const model = before.artifacts.model?.status === "ready" ? before.artifacts.model.path : null;
    const memo = model === null ? null : viewMemoOf(before, this.deps.pose(), model);
    const refuse = async (reason: string) => {
      const failed = memo ? await this.putBack(memo) : null;
      store.dispatch({ type: "offerRefused", id, reason: failed ? `${reason} (${failed})` : reason });
    };
    try {
      if (opensOf(before, p).length) {
        const failed = await this.openSet(p.files.results ?? p.files.model, p.files.model, p.files.results, p.fold);
        if (failed) return await refuse(failed);
      }
      const c = compileShot(p.shot, describe(store.get()));
      if (!c.ok) return await refuse(c.reason);
      if (p.theme) store.dispatch({ type: "setTheme", theme: p.theme });
      for (const e of freshViewEvents(store.get())) store.dispatch(e);
      for (const e of c.plan.events) store.dispatch(e);
      applyClip(store, c.plan);
      const why = await this.shown(p.shot, c.plan);
      if (why !== null) return await refuse(why);
      store.dispatch({ type: "setCamera", to: "home" });
      store.dispatch({ type: "setCamera", to: c.plan.camera });
      await this.deps.settle();
      const pose = this.deps.pose();
      this.applied.push(pose ? { files: p.files, fold: p.fold, shot: p.shot, pose } : null);
      store.dispatch({ type: "offerApplied", id, memo });
    } catch (err) {
      await refuse(`the app refused the view: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  private async back(): Promise<void> {
    const { store } = this.deps;
    const top = store.get().offers.back[0];
    if (!top) throw new Error("backOffer: no applied offer to go back from");
    this.applied.pop();
    const notice = await this.putBack(top.memo);
    store.dispatch({ type: "offerBackDone", notice: notice ? `Back could not restore the view: ${notice}` : null });
  }

  /** The view of `memo`, its set opened first when another is open now; null when it is back, else why not. */
  private async putBack(memo: ViewMemo): Promise<string | null> {
    const { store } = this.deps;
    try {
      const s = store.get();
      if (s.artifacts.model?.path !== memo.files.model || (s.artifacts.results?.path ?? null) !== memo.files.results) {
        const failed = await this.openSet(memo.files.results ?? memo.files.model, memo.files.model, memo.files.results, false);
        if (failed) return failed;
      }
      restoreView(store, memo);
      await this.deps.settle();
      return null;
    } catch (err) {
      return err instanceof Error ? err.message : String(err);
    }
  }

  /** Ask main to open a set and wait for it; null when it is read, else why not. */
  private async openSet(path: string, model: string, results: string | null, fold: boolean): Promise<string | null> {
    const { store } = this.deps;
    const answer = (await this.deps.open(path)) as { ok?: boolean; reason?: string } | undefined;
    if (answer && answer.ok === false) return answer.reason ?? `${path} did not open`;
    const t0 = this.now();
    for (;;) {
      const a = store.get().artifacts;
      if (same(a.model?.path ?? null, model, fold) && a.model?.status === "failed") return `the model ${model} did not open: ${a.model.error ?? "unknown error"}`;
      if (results !== null && same(a.results?.path ?? null, results, fold) && a.results?.status === "failed") return `the results ${results} did not open: ${a.results.error ?? "unknown error"}`;
      const modelIn = same(a.model?.path ?? null, model, fold) && a.model?.status === "ready";
      const resultsIn = results === null || (same(a.results?.path ?? null, results, fold) && a.results?.status === "ready");
      if (modelIn && resultsIn) return null;
      if (this.now() - t0 >= SHOW_MS) return `${path} did not open within ${SHOW_MS / 1000} s`;
      await this.sleep(20);
    }
  }

  /** The render mode's rule for "the view is on screen" (ADR 0128 D2): null when it is, else the reason it is refused. */
  private async shown(shot: SpecShot, plan: ShotPlan): Promise<string | null> {
    const t0 = this.now();
    let lastWhy: string | null = null;
    let since = t0;
    for (;;) {
      const now = this.now();
      const v = viewVerdict(this.deps.store.get(), shot, plan, this.deps.onScreen());
      if (v.kind === "shown") return null;
      if (v.kind === "refused") return v.reason;
      if (v.why !== lastWhy) {
        lastWhy = v.why;
        since = now;
      }
      if (v.why !== null && !v.loading && now - since >= SETTLED_MS) return v.why;
      if (now - t0 >= SHOW_MS) return v.why ?? `the view did not reach the screen within ${SHOW_MS / 1000} s`;
      await this.sleep(20);
    }
  }
}
