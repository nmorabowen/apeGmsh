// The viewport's side of the nodal contour (V4c1, #1756): the vertex
// colours of the drawn primitives for the step on screen. The values and the
// colours come from state/contour.ts (pure, tested in Node); this module
// picks the step data the effects fetched, checks it is the step the state
// shows, and converts the sRGB ramp to the linear colours three.js expects
// for a vertex colour attribute (it writes sRGB on output). VF-a (#2939): the
// surfaces take each vertex's ramp coordinate instead (`triU`, `lineU`) and
// sample the colormap texture per fragment (renderer/colormapTexture.ts), so
// a coarse element shows only ramp colours; the beams' lines still
// interpolate their vertex colours (`line`).

import type { CombinationTable } from "../combination/evaluate.ts";
import type { BlobStore } from "../state/blobs.ts";
import { cellVertexValues, columnOf, differenceColumn, elementCellValues, gaussCellValues, isNodeTopology, joinElements, meshFemIds, NO_DATA_RGB, primitiveColours, vertexValues, type CombinedColumn } from "../state/contour.ts";
import type { ContourCmap } from "../state/types/contour.ts";
import { noDataRamp, primitiveRamp, toLinear } from "./colormapTexture.ts";
import { applyPlan, cornerVertexValues, planFor, vertexCorners, type NodeTopology } from "../state/extrapolation.ts";
import type { UserColumnRead } from "../reader/userFile.ts";
import { combinationRequestOf, combinationTableOf, compareUserRequestOf, differencePlanOf, userResultRequestOf, type ContourView } from "../state/selectors.ts";
import type { Store } from "../state/store.ts";
import type { SpringStepData } from "../state/springs.ts";
import type { CombinationRead, CompareIn, CompareModelAnswer, CompareSlotIn, CompareUserAnswer, DifferenceRead, MeshInfo, State, UserResultRead } from "../state/types.ts";

/** One step's nodal values, as the effects hold them (effects.ts `StepData`). */
export interface StepValues {
  stage: number;
  step: number;
  nodal: Record<string, Float64Array>;
  /** V4c2a: the element component fetched with the step (null: none), and its values by gauss group, (E * G,) */
  element?: string | null;
  gauss?: Record<string, Float64Array>;
  /** the Principal card's tensor components, by component then group */
  tensor?: Record<string, Record<string, Float64Array>>;
  /** The zero-length springs' force, fetched while the Springs card is on; null when its read failed */
  springs?: SpringStepData | null;
}

export { toLinear };

function linearise(rgb: Float32Array): Float32Array {
  for (let i = 0; i < rgb.length; i++) rgb[i] = toLinear(rgb[i]!);
  return rgb;
}

/** Every drawn vertex in the no-data colour (nothing can be coloured: the legend says why). */
export function noDataColours(map: Int32Array, perPrim: number): Float32Array {
  const out = new Float32Array(map.length * perPrim * 3);
  const c = NO_DATA_RGB.map(toLinear);
  for (let i = 0; i < out.length; i += 3) out.set(c, i);
  return out;
}

/**
 * What a contour draws on the primitives on screen (VF-a, #2939). The faces
 * and the frame-extrusion fill sample the colormap per fragment from the ramp
 * coordinates (`triU`, `lineU`: renderer/colormapTexture.ts); the beams'
 * lines interpolate their vertex colours (`line`).
 */
export interface ContourShades {
  /** linear RGB per drawn line vertex (the beams) */
  line: Float32Array;
  /** the ramp coordinate per drawn line vertex (the frame-extrusion fill) */
  lineU: Float32Array;
  /** the ramp coordinate per drawn triangle vertex (the faces) */
  triU: Float32Array;
  /** linear RGB per drawn triangle vertex: the ramp at each vertex, made on first read (the faces draw `triU`) */
  readonly tri: Float32Array;
  /** the colormap the coordinates index */
  cmap: ContourCmap;
  key: string;
}

/** The shades of per-vertex values (original primitive order) through a range and a colormap: one no-data rule for both. */
function shades(
  lineValues: Float64Array,
  triValues: Float64Array,
  lineMap: Int32Array,
  triMap: Int32Array,
  clim: { min: number | null; max: number | null },
  cmap: ContourCmap,
  key: string,
): ContourShades {
  let tri: Float32Array | null = null;
  return {
    line: linearise(primitiveColours(lineValues, lineMap, 2, clim, cmap).rgb),
    lineU: primitiveRamp(lineValues, lineMap, 2, clim).u,
    triU: primitiveRamp(triValues, triMap, 3, clim).u,
    get tri() {
      return (tri ??= linearise(primitiveColours(triValues, triMap, 3, clim, cmap).rgb));
    },
    cmap,
    key,
  };
}

/** Every drawn primitive no data (the coordinates index the no-data texel, which every colormap's texture has). */
export function noDataShades(lineMap: Int32Array, triMap: Int32Array, key: string): ContourShades {
  let tri: Float32Array | null = null;
  return {
    line: noDataColours(lineMap, 2),
    lineU: noDataRamp(lineMap.length * 2),
    triU: noDataRamp(triMap.length * 3),
    get tri() {
      return (tri ??= noDataColours(triMap, 3));
    },
    cmap: "jet",
    key,
  };
}

let columnMemo: { ids: Float64Array; column: Map<number, number> } | null = null;

/**
 * What a set of contour colours shows: the results read (a re-read at the
 * same path can hold other values), the step and component, the range and
 * the colormap. The viewport skips a recolour with the same key.
 */
export const contourKey = (view: Extract<ContourView, { state: "ready" }>): string => `read ${view.reads}|${view.keys.step}|${view.clim.min}|${view.clim.max}|${view.cmap}`;

/**
 * The linear vertex colours of the drawn lines and triangles for the step
 * on screen, and the key of that step; null when the step's data or its
 * measured range is not here yet (the viewport keeps what it shows).
 */
export function contourColours(
  s: State,
  view: Extract<ContourView, { state: "ready" }>,
  mesh: MeshInfo,
  blobs: BlobStore,
  step: StepValues | null,
  lineMap: Int32Array,
  triMap: Int32Array,
): ContourShades | null {
  const at = s.phase.at;
  if (at?.kind !== "results" || !step || step.stage !== at.stage || step.step !== at.step || !view.stepMeasured) return null;
  const rows = view.component.datasets.map((d) => step.nodal[d]);
  if (!rows.every((r): r is Float64Array => r instanceof Float64Array)) {
    throw new Error(`contour: the step data has no ${view.component.datasets.filter((d) => !(step.nodal[d] instanceof Float64Array)).join(", ")}`);
  }
  const ids = blobs.get(view.nodeIds);
  if (!(ids instanceof Float64Array)) throw new TypeError(`contour: ${view.nodeIds.id} is not f64`);
  if (!columnMemo || columnMemo.ids !== ids) columnMemo = { ids, column: columnOf(ids) };
  const vn = mesh.vertexNodes;
  const lineNodes = blobs.get(vn.line) as Float64Array;
  const triNodes = blobs.get(vn.tri) as Float64Array;
  const clim = { min: view.clim.min, max: view.clim.max };
  const lineValues = vertexValues(lineNodes, columnMemo.column, rows, view.component.magnitude);
  const triValues = vertexValues(triNodes, columnMemo.column, rows, view.component.magnitude);
  return shades(lineValues, triValues, lineMap, triMap, clim, view.cmap, contourKey(view));
}

// -- V4c2a: element (Gauss) contours, flat per cell -------------------------------

/** What a set of element colours shows: the nodal key's parts, and the join the rows went through (it names the read, the mesh and the rows). */
export const elementContourKey = (view: Extract<ContourView, { state: "element" }>): string =>
  `read ${view.reads}|${view.keys.step}|${view.clim.min}|${view.clim.max}|${view.cmap}|${view.topology}|join ${view.joinKey}`;

/** The key of the colours a contour view shows once they are applied (nodal, element, or no data with its reason); null when off. */
export function contourShownKey(v: ContourView): string | null {
  switch (v.state) {
    case "ready":
      return contourKey(v);
    case "element":
      return elementContourKey(v);
    case "combination":
      return combinationContourKey(v);
    case "difference":
      return differenceContourKey(v);
    case "user":
      return userResultContourKey(v);
    case "unavailable":
    case "refused":
    case "not-yet":
      return `no-data:${v.reason}`;
    case "off":
      return null;
    default: {
      const unknown: never = v;
      throw new Error(`unknown contour view ${JSON.stringify(unknown)}`);
    }
  }
}

// -- VU7b-2: a User Combination's nodal value (ADR 0130 D4) -----------------------

/** What a set of combination colours shows: the read (the file, the combination, the column, A6), the range and the colormap. */
export const combinationContourKey = (view: Extract<ContourView, { state: "combination" }>): string =>
  `combination ${view.key}|${view.clim.min}|${view.clim.max}|${view.cmap}`;

/** One combination's value for one nodal quantity, under the key of the read (selectors `combinationRequestOf`). */
export interface CombinationColumnOf {
  key: string;
  column: CombinedColumn;
}

/**
 * The linear vertex colours of the drawn lines and triangles for the User
 * Combination on screen: each vertex takes its node's value (by node id,
 * state/contour.ts `vertexValues`), and a node the combination gives no value
 * is no data. Null while the column on hand is not the read the view names
 * (the viewport keeps what it shows).
 */
export function combinationColours(
  view: Extract<ContourView, { state: "combination" }>,
  mesh: MeshInfo,
  blobs: BlobStore,
  held: CombinationColumnOf | null,
  lineMap: Int32Array,
  triMap: Int32Array,
): ContourShades | null {
  if (!held || held.key !== view.key) return null;
  const column = columnOf(held.column.ids);
  const vn = mesh.vertexNodes;
  const clim = { min: view.clim.min, max: view.clim.max };
  const rows = [held.column.values];
  const lineValues = vertexValues(blobs.get(vn.line) as Float64Array, column, rows, false);
  const triValues = vertexValues(blobs.get(vn.tri) as Float64Array, column, rows, false);
  return shades(lineValues, triValues, lineMap, triMap, clim, view.cmap, combinationContourKey(view));
}

/** What the feed asks for one read: `userCombinationColumn`'s request (effects.ts), without the bridge. */
export interface CombinationFetch {
  (req: { path: string; table: CombinationTable; name: string; quantity: { label: string; component: string }; staleness: { stale: boolean; reason: string } }): Promise<
    { ok: true; column: CombinedColumn } | { ok: false; refusal: { reason: string; detail: string } }
  >;
}

/**
 * VU7b-2: the feed of the User Combination shown. It reads the combination's
 * column for the quantity picked whenever the state asks for one it does not
 * hold (selectors `combinationRequestOf`), holds the values (`held`, read by
 * the viewport and the manifest, as a results step's values stay in the
 * effects), and reports each read to the store (`combinationRead`): loading,
 * then the value's range and sign, the evaluator's refusal, or what it raised.
 * An answer for a request no longer wanted is dropped. Each check runs in a
 * microtask, so a dispatch never happens inside another's listeners.
 */
export class CombinationFeed {
  /** the column of the latest read that answered, under its key */
  held: CombinationColumnOf | null = null;
  private seq = 0;
  private asked: string | null = null;
  private queued = false;
  private readonly store: Store;
  private readonly fetch: CombinationFetch;

  constructor(store: Store, fetch: CombinationFetch) {
    this.store = store;
    this.fetch = fetch;
  }

  /** Follow the store; returns the unsubscribe. */
  start(): () => void {
    const off = this.store.subscribe(() => this.schedule());
    this.schedule();
    return off;
  }

  private schedule(): void {
    if (this.queued) return;
    this.queued = true;
    queueMicrotask(() => {
      this.queued = false;
      this.check();
    });
  }

  /** Start the read the state asks for, if it is not running or answered; exposed for the tests. */
  check(): Promise<void> | null {
    const s = this.store.get();
    const want = combinationRequestOf(s);
    if (want === null) {
      // Nothing is asked for: a read still on its way answers nothing.
      if (this.asked !== null) this.seq++;
      this.asked = null;
      return null;
    }
    if (this.asked === want.key && s.combination?.key === want.key) return null;
    const table = combinationTableOf(s);
    if (table === null) throw new Error(`combination feed: ${want.name} is asked for, and the state holds no combination table`);
    const mine = ++this.seq;
    this.asked = want.key;
    this.store.dispatch({ type: "combinationRead", read: { key: want.key, status: "loading" } });
    const answered = (read: CombinationRead, column: CombinedColumn | null) => {
      if (mine !== this.seq) return;
      if (column) this.held = { key: want.key, column };
      this.store.dispatch({ type: "combinationRead", read });
    };
    return this.fetch({ path: want.path, table, name: want.name, quantity: want.quantity, staleness: want.staleness }).then(
      (r) => {
        if (!r.ok) return answered({ key: want.key, status: "refused", reason: r.refusal.reason, detail: r.refusal.detail }, null);
        const c = r.column;
        answered({ key: want.key, status: "ready", label: c.label, range: c.range, nodes: c.ids.length, signed: c.signed }, c);
      },
      // What the evaluator raised (a case that repeats a node id raises in Python too) is stated, never a blank view.
      (err: unknown) => answered({ key: want.key, status: "failed", error: err instanceof Error ? err.message : String(err) }, null),
    );
  }
}

// -- VU16: a state-free user result's nodal value (ADR 0130 D7) --------------------

/** What a set of user-result colours shows: the read (the file, the entry, the component, the feed's read), the range and the colormap. */
export const userResultContourKey = (view: Extract<ContourView, { state: "user" }>): string => `user ${view.key}|read ${view.read}|${view.clim.min}|${view.clim.max}|${view.cmap}`;

/** The user-result column on hand, under the key of its request and the feed's read (selectors `userResultRequestOf`). */
export interface UserResultColumnOf {
  key: string;
  read: number;
  /** FEM node ids, in the file's row order */
  ids: Float64Array;
  /** one value per id; NaN is no value (drawn as no data) */
  values: Float64Array;
}

/**
 * The column the user-result feed read last (`UserResultFeed`), read by the
 * viewport, the caps and the manifest, as a step's values stay in the effects:
 * one window, one feed, so one holder (as `differenceHeld`).
 */
export const userResultHeld: { current: UserResultColumnOf | null } = { current: null };

/** Whether the column on hand is the one the view names. */
export const userResultOnHand = (view: Extract<ContourView, { state: "user" }>): UserResultColumnOf | null => {
  const held = userResultHeld.current;
  return held && held.key === view.key && held.read === view.read ? held : null;
};

/**
 * The linear vertex colours of the drawn lines and triangles for the user
 * result on screen: each vertex takes its node's value by node id
 * (state/contour.ts `vertexValues`), and a node the file gives no finite value
 * (absent from its `ID`, or NaN) is no data. Null while the column on hand is
 * not the read the view names (the viewport keeps what it shows).
 */
export function userResultColours(
  view: Extract<ContourView, { state: "user" }>,
  mesh: MeshInfo,
  blobs: BlobStore,
  lineMap: Int32Array,
  triMap: Int32Array,
): ContourShades | null {
  const held = userResultOnHand(view);
  if (!held) return null;
  const column = columnOf(held.ids);
  const vn = mesh.vertexNodes;
  const clim = { min: view.clim.min, max: view.clim.max };
  const rows = [held.values];
  const lineValues = vertexValues(blobs.get(vn.line) as Float64Array, column, rows, false);
  const triValues = vertexValues(blobs.get(vn.tri) as Float64Array, column, rows, false);
  return shades(lineValues, triValues, lineMap, triMap, clim, view.cmap, userResultContourKey(view));
}

/** What the feed asks main for one read (`user:readResult`, effects.ts `userResultColumn`). */
export interface UserResultFetch {
  (req: { path: string; name: string; component: string }): Promise<UserColumnRead>;
}

/**
 * VU16: the feed of the user result picked. It reads the column whenever the
 * state asks for one it does not hold (selectors `userResultRequestOf`), keeps
 * the values in `userResultHeld`, and reports each read to the store
 * (`userResultRead`): loading, then the column's range, main's refusal (a
 * stale entry is refused there with A6's reason), or what the read raised. An
 * answer for a request no longer wanted is dropped. Each check runs in a
 * microtask, so a dispatch never happens inside another's listeners.
 */
export class UserResultFeed {
  private seq = 0;
  private asked: string | null = null;
  private queued = false;
  private readonly store: Store;
  private readonly fetch: UserResultFetch;

  constructor(store: Store, fetch: UserResultFetch) {
    this.store = store;
    this.fetch = fetch;
  }

  /** Follow the store; returns the unsubscribe. */
  start(): () => void {
    const off = this.store.subscribe(() => this.schedule());
    this.schedule();
    return off;
  }

  private schedule(): void {
    if (this.queued) return;
    this.queued = true;
    queueMicrotask(() => {
      this.queued = false;
      this.check();
    });
  }

  /** Start the read the state asks for, if it is not running or answered; exposed for the tests. */
  check(): Promise<void> | null {
    const s = this.store.get();
    const want = userResultRequestOf(s);
    if (want === null) {
      // Nothing is asked for: a read still on its way answers nothing.
      if (this.asked !== null) this.seq++;
      this.asked = null;
      return null;
    }
    if (this.asked === want.key && s.userResult?.key === want.key) return null;
    const mine = ++this.seq;
    this.asked = want.key;
    this.store.dispatch({ type: "userResultRead", read: { key: want.key, status: "loading" } });
    const answered = (read: UserResultRead, column: { ids: Float64Array; values: Float64Array } | null) => {
      if (mine !== this.seq) return;
      if (column) userResultHeld.current = { key: want.key, read: mine, ...column };
      this.store.dispatch({ type: "userResultRead", read });
    };
    // Through a promise, so a fetch that throws at once (a bridge with no door) is stated as a failed read too.
    return Promise.resolve()
      .then(() => this.fetch({ path: want.path, name: want.name, component: want.component }))
      .then((r) => {
        const a = userColumnAnswer("user:readResult", want.key, mine, r);
        answered(a.read, a.column);
      })
      // What the read raised (a broken door, a malformed answer) is stated, never a blank view.
      .catch((err: unknown) => answered({ key: want.key, status: "failed", error: err instanceof Error ? err.message : String(err) }, null));
  }
}

/**
 * A user-result column door's answer as a read under `key` and the feed's
 * read `mine`: its range and counts with the column, or the door's refusal.
 * Ids and values that are not two f64 columns of one length raise, so the
 * caller states a failed read, never a blank view.
 */
function userColumnAnswer(door: string, key: string, mine: number, r: UserColumnRead): { read: UserResultRead; column: { ids: Float64Array; values: Float64Array } | null } {
  if (!r.ok) return { read: { key, status: "refused", reason: r.reason, detail: r.detail }, column: null };
  const { ids, values } = r.value;
  if (!(ids instanceof Float64Array) || !(values instanceof Float64Array) || ids.length !== values.length) {
    throw new Error(`${door} answered ids and values that are not two f64 columns of one length`);
  }
  let min = Infinity, max = -Infinity, finite = 0;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    finite++;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return { read: { key, status: "ready", read: mine, range: finite ? { min, max } : null, nodes: ids.length, finite }, column: { ids, values } };
}

// -- VU10: the difference of two variants (ADR 0132 D9.5) ---------------------------

/** What a set of difference colours shows: the read (both results, the compare row, the stage, the step, the dataset), the range and the colormap. */
export const differenceContourKey = (view: Extract<ContourView, { state: "difference" }>): string => `difference ${view.key}|${view.clim.min}|${view.clim.max}|${view.cmap}`;

/** The difference column on hand, under the key of its read (selectors `differencePlanOf`). */
export interface DifferenceColumnOf {
  key: string;
  ids: Float64Array;
  values: Float64Array;
}

/**
 * The difference column the runner read last (`VariantRunner`), read by the
 * viewport, the caps and the manifest, as a step's values stay in the effects:
 * one window, one runner, so one holder.
 */
export const differenceHeld: { current: DifferenceColumnOf | null } = { current: null };

/**
 * The linear vertex colours of the drawn lines and triangles for the
 * difference on screen: each vertex takes its node's value (by node id), and a
 * node either variant lacks is no data. Null while the column on hand is not
 * the read the view names (the viewport keeps what it shows).
 */
export function differenceColours(
  view: Extract<ContourView, { state: "difference" }>,
  mesh: MeshInfo,
  blobs: BlobStore,
  lineMap: Int32Array,
  triMap: Int32Array,
): ContourShades | null {
  const held = differenceHeld.current;
  if (!held || held.key !== view.key) return null;
  const column = columnOf(held.ids);
  const vn = mesh.vertexNodes;
  const clim = { min: view.clim.min, max: view.clim.max };
  const lineValues = vertexValues(blobs.get(vn.line) as Float64Array, column, [held.values], false);
  const triValues = vertexValues(blobs.get(vn.tri) as Float64Array, column, [held.values], false);
  return shades(lineValues, triValues, lineMap, triMap, clim, view.cmap, differenceContourKey(view));
}

/** Main's variant doors, as the preload exposes them (main.ts `variants:*`, `compare:*`): a row by index and label, never a path. */
export interface VariantDoor {
  variantsSwitch(index: number, label: string): Promise<{ ok: true } | { ok: false; reason: string }>;
  variantsCompare(index: number, label: string): Promise<{ ok: true; compare: CompareIn; nodeIds: Record<string, Float64Array> } | { ok: false; reason: string }>;
  compareClose(): Promise<unknown>;
  readCompareStep(req: { stage: string; step: number; seq: number }): Promise<{ ok: true; seq: number; nodal: Record<string, Float64Array> } | { ok: false; seq: number; error: string }>;
  /** VU19: the compare slot's model (`compare:model`); it names no file and no row */
  compareModel(): Promise<CompareModelAnswer>;
  /** VU19: the compare slot's `.user.h5` judged against its own model (`compare:userRead`) */
  compareUserRead(): Promise<CompareUserAnswer>;
  /** VU19: one component of a state-free ON_NODES user result of that file (`compare:userReadResult`) */
  compareUserReadResult(req: { name: string; component: string }): Promise<{ slot: CompareSlotIn | null; read: UserColumnRead }>;
}

/** The VU19 doors a variant door must have: this build's preload has them, so a missing one is a broken build, never "no user file". */
const COMPARE_DOORS = ["compareModel", "compareUserRead", "compareUserReadResult"] as const;

/**
 * VU10: the variants panel's side effects. A switch the state asks for goes
 * to main (`variants:switch`), which opens that row's set as the window set
 * and tells the page as any open does. A compare goes to main's compare door
 * (`variants:compare`), which reads the row's set into its own slot and never
 * adopts it; closing it empties the slot (`compare:close`). The difference
 * shown is read from main's compare host one step at a time, subtracted from
 * the window's step by node id (state/contour.ts `differenceColumn`), held in
 * `differenceHeld` and reported (`differenceRead`). An answer for a request no
 * longer wanted is dropped. Each check runs in a microtask, so a dispatch
 * never happens inside another's listeners.
 */
export class VariantRunner {
  private readonly store: Store;
  private readonly blobs: BlobStore;
  private readonly door: VariantDoor;
  private readonly windowStep: () => StepValues | null;
  private queued = false;
  private switchAsked: string | null = null;
  private compareAsked: string | null = null;
  /** the compare row main holds for this page, and its stages' node ids */
  private slot: { index: number; label: string; nodeIds: Record<string, Float64Array> } | null = null;
  private seq = 0;
  private asked: string | null = null;
  /** VU19: the open compare slot whose model and user file this runner asked main for */
  private slotRead: string | null = null;
  /** VU19: the compare user result read, as `seq` and `asked` for the difference */
  private shownSeq = 0;
  private shownAsked: string | null = null;

  constructor(store: Store, blobs: BlobStore, door: VariantDoor, windowStep: () => StepValues | null) {
    // Fail closed: a door without the compare doors is a broken build, never a compare set with no user file.
    const missing = COMPARE_DOORS.filter((k) => typeof (door as unknown as Record<string, unknown>)[k] !== "function");
    if (missing.length) throw new Error(`VariantRunner: the variant door has no ${missing.join(", ")} (VU19's compare doors)`);
    this.store = store;
    this.blobs = blobs;
    this.door = door;
    this.windowStep = windowStep;
  }

  /** Follow the store; returns the unsubscribe. */
  start(): () => void {
    const off = this.store.subscribe(() => this.schedule());
    this.schedule();
    return off;
  }

  private schedule(): void {
    if (this.queued) return;
    this.queued = true;
    queueMicrotask(() => {
      this.queued = false;
      this.check();
    });
  }

  /** Start what the state asks for; exposed for the tests, which await the reads it started. */
  check(): Promise<void>[] {
    const s = this.store.get();
    const running: Promise<void>[] = [];
    const sw = s.variants.switching;
    const swKey = sw ? `${sw.index}|${sw.label}` : null;
    if (sw && this.switchAsked !== swKey) {
      this.switchAsked = swKey;
      running.push(
        this.door.variantsSwitch(sw.index, sw.label).then(
          (r) => this.store.dispatch({ type: "variantSwitched", reason: r.ok ? null : r.reason }),
          (err: unknown) => this.store.dispatch({ type: "variantSwitched", reason: err instanceof Error ? err.message : String(err) }),
        ),
      );
    } else if (!sw) this.switchAsked = null;
    const c = s.variants.compare;
    if (c?.status === "opening" && this.compareAsked !== `${c.index}|${c.label}`) {
      const want = { index: c.index, label: c.label };
      this.compareAsked = `${c.index}|${c.label}`;
      this.slot = null;
      running.push(
        this.door.variantsCompare(want.index, want.label).then(
          (r) => {
            if (r.ok) this.slot = { ...want, nodeIds: r.nodeIds };
            this.store.dispatch({ type: "compareOpened", result: r.ok ? r.compare : { ...want, reason: r.reason } });
          },
          (err: unknown) => this.store.dispatch({ type: "compareOpened", result: { ...want, reason: err instanceof Error ? err.message : String(err) } }),
        ),
      );
    } else if (c === null && (this.slot !== null || this.compareAsked !== null)) {
      this.slot = null;
      this.compareAsked = null;
      differenceHeld.current = null;
      running.push(this.door.compareClose().then(() => undefined));
    }
    running.push(...this.compareReads(s));
    const d = this.difference(s);
    if (d) running.push(d);
    const u = this.compareUserResult(s);
    if (u) running.push(u);
    return running;
  }

  /**
   * VU19: once a compare set is open, its model and its `.user.h5` through the
   * compare doors, once per slot. An answer for a slot this runner no longer
   * reads is dropped here; main naming no slot (it closed it) is stated as a
   * refusal of the slot asked, and the reducer drops an answer for any slot but
   * the open one.
   */
  private compareReads(s: State): Promise<void>[] {
    const c = s.variants.compare;
    const key = c?.status === "open" ? `${c.index}|${c.label}` : null;
    if (key === null || c?.status !== "open") {
      this.slotRead = null;
      return [];
    }
    if (this.slotRead === key) return [];
    this.slotRead = key;
    const asked: CompareSlotIn = { index: c.index, label: c.label };
    const live = () => this.slotRead === key;
    const why = (err: unknown) => (err instanceof Error ? err.message : String(err));
    const model = Promise.resolve()
      .then(() => this.door.compareModel())
      .then(
        (a) => {
          if (live()) this.store.dispatch({ type: "compareModelRead", answer: a.slot === null ? { ok: false, slot: asked, reason: a.ok ? "compare:model named no slot" : a.reason } : a });
        },
        (err: unknown) => {
          if (live()) this.store.dispatch({ type: "compareModelRead", answer: { ok: false, slot: asked, reason: `compare:model failed: ${why(err)}` } });
        },
      );
    const user = Promise.resolve()
      .then(() => this.door.compareUserRead())
      .then(
        (a) => {
          if (live()) this.store.dispatch({ type: "compareUserRead", answer: a.slot === null ? { ...a, slot: asked, path: null } : a });
        },
        (err: unknown) => {
          if (live()) this.store.dispatch({ type: "compareUserRead", answer: { slot: asked, path: null, read: { ok: false, error: `compare:userRead failed: ${why(err)}` } } });
        },
      );
    return [model, user];
  }

  /**
   * VU19: the compare user result shown, read through `compare:userReadResult`
   * whenever the state asks for one it does not hold (selectors
   * `compareUserRequestOf`), kept in `userResultHeld` under its own key (the
   * viewport draws it as any user result, by node id on the window's mesh,
   * which the selector has checked is the compare set's mesh too), and reported
   * (`compareUserResultRead`). An answer for a request no longer wanted, or one
   * main read from another slot, answers nothing of it.
   */
  private compareUserResult(s: State): Promise<void> | null {
    const want = compareUserRequestOf(s);
    if (want === null) {
      if (this.shownAsked !== null) this.shownSeq++;
      this.shownAsked = null;
      return null;
    }
    if (this.shownAsked === want.key && s.variants.shownRead?.key === want.key) return null;
    const c = s.variants.compare as Extract<NonNullable<State["variants"]["compare"]>, { status: "open" }>;
    const mine = ++this.shownSeq;
    this.shownAsked = want.key;
    this.store.dispatch({ type: "compareUserResultRead", read: { key: want.key, status: "loading" } });
    const answered = (read: UserResultRead, column: { ids: Float64Array; values: Float64Array } | null) => {
      if (mine !== this.shownSeq) return;
      if (column) userResultHeld.current = { key: want.key, read: mine, ...column };
      this.store.dispatch({ type: "compareUserResultRead", read });
    };
    return Promise.resolve()
      .then(() => this.door.compareUserReadResult({ name: want.name, component: want.component }))
      .then((a) => {
        if (a.slot === null || a.slot.index !== c.index || a.slot.label !== c.label) {
          const detail = a.slot === null ? (a.read.ok ? "main holds no compare set" : a.read.detail) : `main read the compare variant ${JSON.stringify(a.slot.label)}, not ${JSON.stringify(c.label)}`;
          return answered({ key: want.key, status: "refused", reason: "refused", detail }, null);
        }
        const r = userColumnAnswer("compare:userReadResult", want.key, mine, a.read);
        answered(r.read, r.column);
      })
      .catch((err: unknown) => answered({ key: want.key, status: "failed", error: err instanceof Error ? err.message : String(err) }, null));
  }

  private difference(s: ReturnType<Store["get"]>): Promise<void> | null {
    const plan = differencePlanOf(s);
    if (plan === null || !plan.ok) {
      if (this.asked !== null) this.seq++;
      this.asked = null;
      return null;
    }
    if (this.asked === plan.key && s.variants.read?.key === plan.key) return null;
    const at = s.phase.at;
    const step = this.windowStep();
    // The window's own step first: it is the one the effects read for the phase shown.
    if (at?.kind !== "results" || !step || step.stage !== at.stage || step.step !== at.step || !(step.nodal[plan.dataset] instanceof Float64Array)) return null;
    const slot = this.slot;
    if (!slot || slot.index !== plan.row.index || slot.label !== plan.row.label) return null;
    const cmpIds = slot.nodeIds[plan.stageId];
    const mine = ++this.seq;
    this.asked = plan.key;
    const key = plan.key;
    const answered = (read: DifferenceRead, column: DifferenceColumnOf | null) => {
      if (mine !== this.seq) return;
      if (column) differenceHeld.current = column;
      this.store.dispatch({ type: "differenceRead", read });
    };
    if (!(cmpIds instanceof Float64Array)) {
      answered({ key, status: "refused", reason: `main sent no node ids of stage ${plan.stageId} of the compare set` }, null);
      return null;
    }
    this.store.dispatch({ type: "differenceRead", read: { key, status: "loading" } });
    const winIds = this.blobs.get(plan.nodeIds);
    if (!(winIds instanceof Float64Array)) throw new TypeError(`difference: ${plan.nodeIds.id} is not f64`);
    const winValues = step.nodal[plan.dataset]!;
    return this.door.readCompareStep({ stage: plan.stageId, step: plan.step, seq: mine }).then(
      (r) => {
        if (!r.ok) return answered({ key, status: "refused", reason: `the compare variant's step could not be read: ${r.error}` }, null);
        const values = r.nodal[plan.dataset];
        if (!(values instanceof Float64Array)) return answered({ key, status: "refused", reason: `the compare variant's step holds no ${plan.dataset}` }, null);
        try {
          const col = differenceColumn(winIds, winValues, cmpIds, values);
          answered({ key, status: "ready", range: col.range, nodes: col.ids.length, missing: col.missing }, { key, ids: col.ids, values: col.values });
        } catch (err) {
          answered({ key, status: "refused", reason: err instanceof Error ? err.message : String(err) }, null);
        }
      },
      (err: unknown) => answered({ key, status: "refused", reason: err instanceof Error ? err.message : String(err) }, null),
    );
  }
}

let joinMemo:{ key: string; at: Map<number, { group: number; row: number }> } | null = null;

/**
 * The join of the element field on screen to the model's cells, by FEM id
 * (built once per join key). The state says it is ok (the effects checked it
 * for this key); a refused one here is a bug, never a guessed join. V4e3: the
 * cut face colours through the same join.
 */
export function elementJoinAt(s: State, view: Extract<ContourView, { state: "element" }>, blobs: BlobStore): ReadonlyMap<number, { group: number; row: number }> {
  if (!joinMemo || joinMemo.key !== view.joinKey) {
    const j = joinElements(
      s.decls,
      view.groups.map((g) => ({ id: g.id, eids: blobs.get(g.eids) as Float64Array })),
    );
    if (!j.ok) throw new Error(`element contour: the join ${view.joinKey} is refused (${j.reason}) while the state shows it`);
    joinMemo = { key: view.joinKey, at: j.at };
  }
  return joinMemo.at;
}
let femMemo: { elements: readonly string[]; decls: unknown; ids: Float64Array } | null = null;

/**
 * The linear vertex colours of the drawn lines and triangles for an element
 * field at the step on screen: each primitive in its element's value (the
 * cell's one Gauss value, or the mean of its Gauss values), joined to the
 * element by FEM id only. A drawn element no group has is no data. Null while
 * the step's data or its measured range is not here.
 */
export function elementColours(
  s: State,
  view: Extract<ContourView, { state: "element" }>,
  mesh: MeshInfo,
  blobs: BlobStore,
  step: StepValues | null,
  lineMap: Int32Array,
  triMap: Int32Array,
): ContourShades | null {
  const at = s.phase.at;
  if (at?.kind !== "results" || !step || step.stage !== at.stage || step.step !== at.step || !view.stepMeasured) return null;
  if (step.element !== view.component.key || !step.gauss) return null;
  const join = elementJoinAt(s, view, blobs);
  if (!femMemo || femMemo.elements !== mesh.elements || femMemo.decls !== s.decls) femMemo = { elements: mesh.elements, decls: s.decls, ids: meshFemIds(mesh.elements, s.decls) };
  if (isNodeTopology(view.topology)) return nodeColours(s, { ...view, topology: view.topology }, mesh, blobs, step.gauss, lineMap, triMap);
  const perGroup = view.groups.map((g) => {
    const v = step.gauss![g.id];
    if (!(v instanceof Float64Array)) throw new Error(`element contour: the step data has no ${view.component.key} for group ${g.id}`);
    return gaussCellValues(v, g.nElements, g.nGauss);
  });
  const values = elementCellValues(femMemo.ids, join, perGroup);
  const clim = { min: view.clim.min, max: view.clim.max };
  return shades(cellVertexValues(values, blobs.i32(mesh.lineElement), 2), cellVertexValues(values, blobs.i32(mesh.triElement), 3), lineMap, triMap, clim, view.cmap, elementContourKey(view));
}

// -- V4c2b: element fields carried to the nodes -------------------------------------

let cornerMemo: { key: string; femIds: Float64Array; line: Int32Array; tri: Int32Array } | null = null;

/**
 * The linear vertex colours for a node-located topology (V4c2b): the step's
 * Gauss values carried through the field's plan (state/extrapolation.ts),
 * each vertex coloured by its own element's corner at its node (discrete) or
 * by that node's average (averaged); a vertex of an element with no data is
 * no data, whatever its neighbours hold.
 */
function nodeColours(
  s: State,
  view: Extract<ContourView, { state: "element" }> & { topology: NodeTopology },
  mesh: MeshInfo,
  blobs: BlobStore,
  gauss: Record<string, Float64Array>,
  lineMap: Int32Array,
  triMap: Int32Array,
): ContourShades {
  const plan = planFor(view, s, blobs);
  const key = `${view.joinKey}|${view.topology}|${mesh.triElement.id}`;
  if (!cornerMemo || cornerMemo.key !== key || cornerMemo.femIds !== femMemo!.ids) {
    const fem = femMemo!.ids;
    const vn = mesh.vertexNodes;
    cornerMemo = {
      key,
      femIds: fem,
      line: vertexCorners(plan, joinMemo!.at, fem, blobs.i32(mesh.lineElement), blobs.get(vn.line) as Float64Array, 2),
      tri: vertexCorners(plan, joinMemo!.at, fem, blobs.i32(mesh.triElement), blobs.get(vn.tri) as Float64Array, 3),
    };
  }
  const r = applyPlan(
    plan,
    view.groups.map((g) => {
      const v = gauss[g.id];
      if (!(v instanceof Float64Array)) throw new Error(`element contour: the step data has no ${view.component.key} for group ${g.id}`);
      return v;
    }),
  );
  const clim = { min: view.clim.min, max: view.clim.max };
  return shades(cornerVertexValues(plan, r, cornerMemo.line), cornerVertexValues(plan, r, cornerMemo.tri), lineMap, triMap, clim, view.cmap, elementContourKey(view));
}
