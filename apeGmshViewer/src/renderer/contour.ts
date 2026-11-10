// The viewport's side of the nodal contour (V4c1, #1756): the vertex
// colours of the drawn primitives for the step on screen. The values and the
// colours come from state/contour.ts (pure, tested in Node); this module
// picks the step data the effects fetched, checks it is the step the state
// shows, and converts the sRGB ramp to the linear colours three.js expects
// for a vertex colour attribute (it writes sRGB on output). The GPU then
// interpolates the colours inside each primitive.

import type { CombinationTable } from "../combination/evaluate.ts";
import type { BlobStore } from "../state/blobs.ts";
import { cellVertexValues, columnOf, differenceColumn, elementCellValues, gaussCellValues, isNodeTopology, joinElements, meshFemIds, NO_DATA_RGB, primitiveColours, vertexValues, type CombinedColumn } from "../state/contour.ts";
import { applyPlan, cornerVertexValues, planFor, vertexCorners, type NodeTopology } from "../state/extrapolation.ts";
import type { UserColumnRead } from "../reader/userFile.ts";
import { combinationRequestOf, combinationTableOf, differencePlanOf, userResultRequestOf, type ContourView } from "../state/selectors.ts";
import type { Store } from "../state/store.ts";
import type { SpringStepData } from "../state/springs.ts";
import type { CombinationRead, CompareIn, DifferenceRead, MeshInfo, State, UserResultRead } from "../state/types.ts";

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

/** sRGB (0..1) to linear, per channel (the sRGB transfer function). */
export const toLinear = (c: number): number => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);

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
): { line: Float32Array; tri: Float32Array; key: string } | null {
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
  const line = primitiveColours(vertexValues(lineNodes, columnMemo.column, rows, view.component.magnitude), lineMap, 2, clim, view.cmap).rgb;
  const tri = primitiveColours(vertexValues(triNodes, columnMemo.column, rows, view.component.magnitude), triMap, 3, clim, view.cmap).rgb;
  return { line: linearise(line), tri: linearise(tri), key: contourKey(view) };
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
): { line: Float32Array; tri: Float32Array; key: string } | null {
  if (!held || held.key !== view.key) return null;
  const column = columnOf(held.column.ids);
  const vn = mesh.vertexNodes;
  const clim = { min: view.clim.min, max: view.clim.max };
  const rows = [held.column.values];
  const line = primitiveColours(vertexValues(blobs.get(vn.line) as Float64Array, column, rows, false), lineMap, 2, clim, view.cmap).rgb;
  const tri = primitiveColours(vertexValues(blobs.get(vn.tri) as Float64Array, column, rows, false), triMap, 3, clim, view.cmap).rgb;
  return { line: linearise(line), tri: linearise(tri), key: combinationContourKey(view) };
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
): { line: Float32Array; tri: Float32Array; key: string } | null {
  const held = userResultOnHand(view);
  if (!held) return null;
  const column = columnOf(held.ids);
  const vn = mesh.vertexNodes;
  const clim = { min: view.clim.min, max: view.clim.max };
  const rows = [held.values];
  const line = primitiveColours(vertexValues(blobs.get(vn.line) as Float64Array, column, rows, false), lineMap, 2, clim, view.cmap).rgb;
  const tri = primitiveColours(vertexValues(blobs.get(vn.tri) as Float64Array, column, rows, false), triMap, 3, clim, view.cmap).rgb;
  return { line: linearise(line), tri: linearise(tri), key: userResultContourKey(view) };
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
      .then(
      (r) => {
        if (!r.ok) return answered({ key: want.key, status: "refused", reason: r.reason, detail: r.detail }, null);
        const { ids, values } = r.value;
        if (!(ids instanceof Float64Array) || !(values instanceof Float64Array) || ids.length !== values.length) {
          throw new Error(`user:readResult answered ids and values that are not two f64 columns of one length`);
        }
        let min = Infinity, max = -Infinity, finite = 0;
        for (const v of values) {
          if (!Number.isFinite(v)) continue;
          finite++;
          if (v < min) min = v;
          if (v > max) max = v;
        }
        answered({ key: want.key, status: "ready", read: mine, range: finite ? { min, max } : null, nodes: ids.length, finite }, { ids, values });
      },
      // What the read raised (a broken door, a malformed answer) is stated, never a blank view.
    ).catch((err: unknown) => answered({ key: want.key, status: "failed", error: err instanceof Error ? err.message : String(err) }, null));
  }
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
): { line: Float32Array; tri: Float32Array; key: string } | null {
  const held = differenceHeld.current;
  if (!held || held.key !== view.key) return null;
  const column = columnOf(held.ids);
  const vn = mesh.vertexNodes;
  const clim = { min: view.clim.min, max: view.clim.max };
  const line = primitiveColours(vertexValues(blobs.get(vn.line) as Float64Array, column, [held.values], false), lineMap, 2, clim, view.cmap).rgb;
  const tri = primitiveColours(vertexValues(blobs.get(vn.tri) as Float64Array, column, [held.values], false), triMap, 3, clim, view.cmap).rgb;
  return { line: linearise(line), tri: linearise(tri), key: differenceContourKey(view) };
}

/** Main's variant doors, as the preload exposes them (main.ts `variants:*`, `compare:*`): a row by index and label, never a path. */
export interface VariantDoor {
  variantsSwitch(index: number, label: string): Promise<{ ok: true } | { ok: false; reason: string }>;
  variantsCompare(index: number, label: string): Promise<{ ok: true; compare: CompareIn; nodeIds: Record<string, Float64Array> } | { ok: false; reason: string }>;
  compareClose(): Promise<unknown>;
  readCompareStep(req: { stage: string; step: number; seq: number }): Promise<{ ok: true; seq: number; nodal: Record<string, Float64Array> } | { ok: false; seq: number; error: string }>;
}

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

  constructor(store: Store, blobs: BlobStore, door: VariantDoor, windowStep: () => StepValues | null) {
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
    const d = this.difference(s);
    if (d) running.push(d);
    return running;
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
): { line: Float32Array; tri: Float32Array; key: string } | null {
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
  const line = primitiveColours(cellVertexValues(values, blobs.i32(mesh.lineElement), 2), lineMap, 2, clim, view.cmap).rgb;
  const tri = primitiveColours(cellVertexValues(values, blobs.i32(mesh.triElement), 3), triMap, 3, clim, view.cmap).rgb;
  return { line: linearise(line), tri: linearise(tri), key: elementContourKey(view) };
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
): { line: Float32Array; tri: Float32Array; key: string } {
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
  const line = primitiveColours(cornerVertexValues(plan, r, cornerMemo.line), lineMap, 2, clim, view.cmap).rgb;
  const tri = primitiveColours(cornerVertexValues(plan, r, cornerMemo.tri), triMap, 3, clim, view.cmap).rgb;
  return { line: linearise(line), tri: linearise(tri), key: elementContourKey(view) };
}
