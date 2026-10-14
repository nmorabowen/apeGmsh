// The section-plane gizmos (V4e2, #1903; ADR 0083 S2, the Qt
// viewers/core/_clip_gizmo.py and _clip_gizmo_interactor.py, read-only
// context, nothing of them imported): per plane with a gizmo, a translucent
// quad sized to the reference box's cross-section and an arrow along the
// normal whose cone tip is the rotate handle.
//
// - **Clip-exempt.** The gizmo root carries `userData.clipExempt`; the
//   viewport's clip stamping (renderer/clip.ts `beforeRender`) skips that
//   subtree, so neither its own plane nor any other cuts a gizmo.
// - **Unpickable.** Every gizmo object's `raycast` is a no-op, so no scene
//   raycast (the pick, the orbit pivot, the zoom point) can hit one; the
//   gizmo's own presses are hit-tested here against the exact geometry drawn
//   (`rayHit`), never through three.js's raycaster.
// - **Constant topology.** One quad, one outline, one shaft and one cone per
//   plane, shared unit geometries placed by their transforms: a drag moves
//   objects, it never builds one.
//
// The interactor (`GizmoInteractor`) listens on the canvas ahead of the
// navigator (viewport.ts `wireInput`): a left press on a gizmo is claimed
// (`stopImmediatePropagation`, so the navigator never sees it); a miss is not
// touched, so camera navigation and picking are what they were. A press
// released within the click slop is a click, handed to the model pick (the
// pick goes through the quad). Past the slop, a translate drag dispatches one
// `setClipOffset` per move (the mouse ray projected on the normal axis), and a
// rotate one `setClipPose` per move (a trackball about the quad centre, the
// offset re-anchored so the plane pivots there, not about the world origin).
// Pointer-leave and cancel end a drag; an overlapping legend or panel wins the
// press (`chromeAt`).

import * as THREE from "three";
import type { Box, Vec3 } from "../state/clip.ts";
import type { ClipPlane, Event } from "../state/types.ts";
import { DARK, DARK_MAIN, toHexNumber } from "../theme/tokens.ts";
import { CLICK_SLOP_PX } from "./bindings.ts";
import type { ListenTarget } from "./navigation.ts";

/** The quad reaches a hair past the box's cross-section, so it reads as a tool, not a face of the model. */
export const QUAD_MARGIN = 1.06;
/** The quad's least half-extent, a fraction of the box diagonal (a flat model's thin axis still gets a grabbable quad). */
export const MIN_HALF_FRAC = 0.05;
/** The arrow's length, a fraction of the box diagonal. */
export const ARROW_LEN_FRAC = 0.22;
/** The cone tip's length and radius, fractions of the arrow length; the radius is also the shaft's grab radius. */
export const TIP_LEN_FRAC = 0.3;
export const TIP_RADIUS_FRAC = 0.09;
/** Grab slack on the tip: a handle that must be hit exactly is one nobody hits. */
export const TIP_GRAB = 1.8;
/**
 * The drag projection's floor on `1 - (n·d)^2` (sin^2 14 degrees; the Qt S2
 * review's C1): looking nearly along the normal, the gain saturates instead
 * of throwing the plane off the model; exactly along it, the drag is dead.
 */
export const DRAG_MIN_SEPARATION = 0.06;
export const QUAD_OPACITY = 0.25;

export type GizmoMode = "translate" | "rotate";

/** One gizmo's world geometry: what is drawn and what a press is tested against are the same numbers. */
export interface GizmoGeometry {
  id: string;
  /** the box centre projected onto the plane */
  centre: Vec3;
  /** the stored unit normal: the axis a drag slides along */
  normal: Vec3;
  /** the arrow's direction: the kept side (the flip folded in) */
  direction: Vec3;
  /** the quad's in-plane orthonormal pair */
  u: Vec3;
  v: Vec3;
  halfU: number;
  halfV: number;
  arrowLen: number;
  tipLen: number;
  tipRadius: number;
}

const dot = (a: readonly number[], b: readonly number[]): number => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
const sub = (a: readonly number[], b: readonly number[]): Vec3 => [a[0]! - b[0]!, a[1]! - b[1]!, a[2]! - b[2]!];
const along = (a: readonly number[], d: readonly number[], t: number): Vec3 => [a[0]! + t * d[0]!, a[1]! + t * d[1]!, a[2]! + t * d[2]!];
const cross = (a: readonly number[], b: readonly number[]): Vec3 => [a[1]! * b[2]! - a[2]! * b[1]!, a[2]! * b[0]! - a[0]! * b[2]!, a[0]! * b[1]! - a[1]! * b[0]!];
const unit = (a: readonly number[]): Vec3 => {
  const m = Math.hypot(a[0]!, a[1]!, a[2]!);
  return m > 0 ? [a[0]! / m, a[1]! / m, a[2]! / m] : [1, 0, 0];
};

/** The tip of the arrow (the rotate handle's centre). */
export const tipPoint = (g: GizmoGeometry): Vec3 => along(g.centre, g.direction, g.arrowLen);
/** Where the shaft ends and the cone begins. */
export const shaftEnd = (g: GizmoGeometry): Vec3 => along(g.centre, g.direction, g.arrowLen - g.tipLen);

/** The gizmo of `plane` against the reference box (the one the offset slider maps). */
export function gizmoGeometry(plane: ClipPlane, box: Box): GizmoGeometry {
  const n = unit(plane.normal);
  const d: Vec3 = plane.flipped ? [-n[0], -n[1], -n[2]] : n;
  const lo = box.lo, hi = box.hi;
  const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 1;
  const b: Vec3 = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
  const centre = along(b, n, -(dot(b, n) - plane.offset));
  const ref: Vec3 = Math.abs(n[2]) < 0.9 ? [0, 0, 1] : [0, 1, 0];
  const u = unit(cross(n, ref));
  const v = cross(n, u);
  let hu = 0, hv = 0;
  for (const x of [lo[0], hi[0]]) for (const y of [lo[1], hi[1]]) for (const z of [lo[2], hi[2]]) {
    const r = sub([x, y, z], centre);
    hu = Math.max(hu, Math.abs(dot(r, u)));
    hv = Math.max(hv, Math.abs(dot(r, v)));
  }
  const floor = MIN_HALF_FRAC * diag;
  const arrowLen = ARROW_LEN_FRAC * diag;
  return {
    id: plane.id, centre, normal: n, direction: d, u, v,
    halfU: Math.max(hu * QUAD_MARGIN, floor), halfV: Math.max(hv * QUAD_MARGIN, floor),
    arrowLen, tipLen: TIP_LEN_FRAC * arrowLen, tipRadius: TIP_RADIUS_FRAC * arrowLen,
  };
}

/** A world ray (`direction` need not be unit). */
export interface Ray {
  origin: readonly number[];
  direction: readonly number[];
}

/**
 * The signed distance along the axis of its point nearest the ray (the
 * translate gesture in one number); null when the ray runs along the axis.
 * With `minSeparation`, the denominator `1 - (n·d)^2` is floored to it and
 * the result is never null (the drag passes DRAG_MIN_SEPARATION).
 */
export function axisParam(point: readonly number[], axis: readonly number[], ray: Ray, minSeparation?: number): number | null {
  const n = unit(axis), d = unit(ray.direction);
  const w = sub(point, ray.origin);
  const b = dot(n, d);
  let denom = 1 - b * b;
  if (minSeparation !== undefined) denom = Math.max(denom, minSeparation);
  else if (denom < 1e-12) return null;
  return (b * dot(w, d) - dot(w, n)) / denom;
}

/**
 * Where the ray meets the trackball about `centre`: the near intersection
 * when it pierces the sphere, else the ray's closest approach (the rotate
 * keeps working as the cursor slides off the silhouette); null when the
 * sphere is behind the ray's origin.
 */
export function spherePoint(centre: readonly number[], radius: number, ray: Ray): Vec3 | null {
  const d = unit(ray.direction);
  const w = sub(centre, ray.origin);
  const tca = dot(w, d);
  if (tca <= 0) return null;
  const dist2 = dot(w, w) - tca * tca;
  const r2 = radius * radius;
  const t = dist2 < r2 ? tca - Math.sqrt(r2 - dist2) : tca;
  return along(ray.origin, d, t);
}

/**
 * Where the ray grabs this gizmo: the mode and the ray parameter, or null.
 * The tip (rotate) wins over the quad and the shaft whatever their depth: it
 * is the small target, and the quad behind it is huge.
 */
export function rayHit(g: GizmoGeometry, ray: Ray): { mode: GizmoMode; t: number } | null {
  const d = unit(ray.direction), o = ray.origin;
  const w = sub(tipPoint(g), o);
  const tca = dot(w, d);
  if (tca > 0) {
    const r = g.tipRadius * TIP_GRAB;
    if (dot(w, w) - tca * tca <= r * r) return { mode: "rotate", t: tca };
  }
  let best = Infinity;
  const denom = dot(d, g.normal);
  if (Math.abs(denom) > 1e-12) {
    const t = dot(sub(g.centre, o), g.normal) / denom;
    if (t > 0) {
      const p = sub(along(o, d, t), g.centre);
      if (Math.abs(dot(p, g.u)) <= g.halfU && Math.abs(dot(p, g.v)) <= g.halfV) best = Math.min(best, t);
    }
  }
  const segLen = g.arrowLen - g.tipLen;
  if (segLen > 0) {
    const s = Math.min(Math.max(axisParam(g.centre, g.direction, { origin: o, direction: d }) ?? 0, 0), segLen);
    const ps = along(g.centre, g.direction, s);
    const t = Math.max(dot(sub(ps, o), d), 0);
    const q = sub(ps, along(o, d, t));
    if (Math.hypot(q[0], q[1], q[2]) <= g.tipRadius) best = Math.min(best, t);
  }
  return best < Infinity ? { mode: "translate", t: best } : null;
}

/** A gizmo under a ray: the plane, the mode, the ray parameter. */
export interface GizmoHit {
  id: string;
  mode: GizmoMode;
  t: number;
}

/** The objects of one plane's gizmo. */
interface Drawn {
  quad: THREE.Mesh;
  outline: THREE.LineSegments;
  arrow: THREE.Group;
  cone: THREE.Mesh;
  shaft: THREE.Mesh;
}

/** No raycast reaches a gizmo object (three.js's idiom for an unpickable object). */
const NO_RAYCAST = (): void => {};

const Y = new THREE.Vector3(0, 1, 0);

/** The drawn gizmos of one viewport: a projection of the store's planes; it holds no plane state. */
export class PlaneGizmos {
  /** The gizmo root: clip-exempt (renderer/clip.ts skips it), its objects unpickable. */
  readonly root = new THREE.Group();
  private readonly drawn = new Map<string, Drawn>();
  private geoms = new Map<string, GizmoGeometry>();
  private key = "";
  private readonly quadGeo = new THREE.PlaneGeometry(1, 1);
  private readonly outlineGeo = new THREE.EdgesGeometry(this.quadGeo);
  private readonly shaftGeo = new THREE.CylinderGeometry(1, 1, 1, 12).translate(0, 0.5, 0);
  private readonly coneGeo = new THREE.ConeGeometry(1, 1, 16).translate(0, 0.5, 0);
  private readonly quadMat = new THREE.MeshBasicMaterial({ color: toHexNumber(DARK_MAIN[3]!), transparent: true, opacity: QUAD_OPACITY, side: THREE.DoubleSide, depthWrite: false });
  private readonly outlineMat = new THREE.LineBasicMaterial({ color: toHexNumber(DARK_MAIN[3]!), transparent: true, opacity: 0.8 });
  private readonly arrowMat = new THREE.MeshBasicMaterial({ color: toHexNumber(DARK.accent) });

  constructor() {
    this.root.name = "plane-gizmos";
    this.root.userData["clipExempt"] = true;
    this.root.raycast = NO_RAYCAST;
  }

  /** The geometry of each drawn gizmo, by plane id (the hit-test surface). */
  geometries(): ReadonlyMap<string, GizmoGeometry> {
    return this.geoms;
  }

  /** How many gizmos are drawn. */
  get count(): number {
    return this.drawn.size;
  }

  /** Draw a gizmo for each of `planes` against `box` (none without a box); true when anything moved. */
  sync(planes: readonly ClipPlane[], box: Box | null): boolean {
    const want = box ? planes.map((p) => gizmoGeometry(p, box)) : [];
    const key = JSON.stringify(want);
    if (key === this.key) return false;
    this.key = key;
    const ids = new Set(want.map((g) => g.id));
    for (const [id, d] of this.drawn) {
      if (ids.has(id)) continue;
      this.root.remove(d.quad, d.outline, d.arrow);
      this.drawn.delete(id);
    }
    this.geoms = new Map(want.map((g) => [g.id, g]));
    for (const g of want) this.place(this.drawn.get(g.id) ?? this.make(g.id), g);
    return true;
  }

  /** The nearest gizmo the ray grabs, or null. */
  hit(ray: Ray): GizmoHit | null {
    let best: GizmoHit | null = null;
    for (const g of this.geoms.values()) {
      const h = rayHit(g, ray);
      if (h && (!best || h.t < best.t)) best = { id: g.id, mode: h.mode, t: h.t };
    }
    return best;
  }

  private make(id: string): Drawn {
    const quad = new THREE.Mesh(this.quadGeo, this.quadMat);
    const outline = new THREE.LineSegments(this.outlineGeo, this.outlineMat);
    const shaft = new THREE.Mesh(this.shaftGeo, this.arrowMat);
    const cone = new THREE.Mesh(this.coneGeo, this.arrowMat);
    const arrow = new THREE.Group();
    arrow.add(shaft, cone);
    for (const o of [quad, outline, arrow, shaft, cone]) {
      o.raycast = NO_RAYCAST;
      o.name = `gizmo:${id}`;
    }
    // Drawn after the model, so the translucent quad blends over it.
    quad.renderOrder = outline.renderOrder = 1;
    this.root.add(quad, outline, arrow);
    const d = { quad, outline, arrow, cone, shaft };
    this.drawn.set(id, d);
    return d;
  }

  private place(d: Drawn, g: GizmoGeometry): void {
    const basis = new THREE.Matrix4().makeBasis(new THREE.Vector3(...g.u), new THREE.Vector3(...g.v), new THREE.Vector3(...g.normal));
    for (const o of [d.quad, d.outline]) {
      o.position.set(...g.centre);
      o.quaternion.setFromRotationMatrix(basis);
      o.scale.set(2 * g.halfU, 2 * g.halfV, 1);
    }
    d.arrow.position.set(...g.centre);
    d.arrow.quaternion.setFromUnitVectors(Y, new THREE.Vector3(...g.direction));
    const shaftR = 0.35 * g.tipRadius;
    d.shaft.scale.set(shaftR, g.arrowLen - g.tipLen, shaftR);
    d.cone.position.set(0, g.arrowLen - g.tipLen, 0);
    d.cone.scale.set(g.tipRadius, g.tipLen, g.tipRadius);
  }

  /** Release the shared geometries and materials. */
  dispose(): void {
    this.root.clear();
    this.drawn.clear();
    this.geoms = new Map();
    for (const x of [this.quadGeo, this.outlineGeo, this.shaftGeo, this.coneGeo, this.quadMat, this.outlineMat, this.arrowMat]) x.dispose();
  }
}

/** What the interactor needs of the viewport. */
export interface GizmoHost {
  /** the canvas: pointer events (the cursor shows the hover) */
  readonly element: ListenTarget & { readonly style?: { cursor: string } };
  /** the world ray through a client point */
  rayAt(clientX: number, clientY: number): Ray;
  /** the gizmo the ray grabs (PlaneGizmos.hit) */
  hit(ray: Ray): GizmoHit | null;
  /** the drawn geometry of a plane's gizmo */
  geometry(id: string): GizmoGeometry | undefined;
  /** the plane as the store holds it now */
  plane(id: string): ClipPlane | undefined;
  dispatch(e: Event): void;
  /** HTML chrome (the legend, a panel) is over this client point: it wins the press */
  chromeAt(clientX: number, clientY: number): boolean;
  /** a click on a gizmo goes through to the model pick */
  select(clientX: number, clientY: number): void;
}

/** The gesture in flight, frozen at the press. */
type Drag =
  | { id: string; pointer: number; x0: number; y0: number; moved: boolean; mode: "translate"; axisPoint: Vec3; axisDir: Vec3; s0: number; offset0: number }
  | { id: string; pointer: number; x0: number; y0: number; moved: boolean; mode: "rotate"; centre: Vec3; radius: number; flipped: boolean };

/** Pointer gestures on the gizmos of one viewport (listen before the navigator: `wireInput`). */
export class GizmoInteractor {
  private readonly host: GizmoHost;
  private drag: Drag | null = null;
  private cursor = "";
  private readonly off: (() => void)[] = [];

  constructor(host: GizmoHost) {
    this.host = host;
    const on = (type: string, fn: (e: PointerEvent) => void) => {
      const listener = fn as unknown as (e: globalThis.Event) => void;
      host.element.addEventListener(type, listener);
      this.off.push(() => host.element.removeEventListener(type, listener));
    };
    on("pointerdown", (e) => this.onDown(e));
    on("pointermove", (e) => this.onMove(e));
    on("pointerup", (e) => this.onUp(e));
    // The release may never come (a pointer that left the canvas, a lost press): the drag must not outlive it.
    on("pointerleave", () => this.endDrag());
    on("pointercancel", () => this.endDrag());
  }

  /** A gizmo gesture is in flight. */
  get dragging(): boolean {
    return this.drag !== null;
  }

  /** The gesture in flight, if any (for tests and the capture). */
  get mode(): GizmoMode | null {
    return this.drag?.mode ?? null;
  }

  dispose(): void {
    for (const f of this.off.splice(0)) f();
    this.endDrag();
  }

  private claim(e: PointerEvent): void {
    e.stopImmediatePropagation();
    e.preventDefault();
  }

  private onDown(e: PointerEvent): void {
    if (e.button !== 0 || e.shiftKey || this.drag) return;
    if (this.host.chromeAt(e.clientX, e.clientY)) return;
    const ray = this.host.rayAt(e.clientX, e.clientY);
    const hit = this.host.hit(ray);
    if (!hit) return; // a miss: untouched, the navigator and the pick see it as before
    const plane = this.host.plane(hit.id);
    const g = this.host.geometry(hit.id);
    if (!plane || !g) return;
    const base = { id: hit.id, pointer: e.pointerId, x0: e.clientX, y0: e.clientY, moved: false };
    if (hit.mode === "rotate") {
      // The pivot is frozen at the press: re-deriving it from the moving plane would walk it sideways.
      this.drag = { ...base, mode: "rotate", centre: g.centre, radius: g.arrowLen, flipped: plane.flipped };
    } else {
      // Floored: a press that hit always becomes a drag, near face-on too (Qt S2 review C1).
      const s0 = axisParam(g.centre, g.normal, ray, DRAG_MIN_SEPARATION)!;
      this.drag = { ...base, mode: "translate", axisPoint: g.centre, axisDir: g.normal, s0, offset0: plane.offset };
    }
    this.claim(e);
  }

  private onMove(e: PointerEvent): void {
    const d = this.drag;
    if (!d) return this.hover(e.clientX, e.clientY);
    if (e.pointerId !== d.pointer) return;
    this.claim(e);
    if (!d.moved && Math.hypot(e.clientX - d.x0, e.clientY - d.y0) <= CLICK_SLOP_PX) return;
    d.moved = true;
    const plane = this.host.plane(d.id);
    // The plane went away mid-drag (deleted from the panel, the file closed): the gesture ends with it.
    if (!plane) return this.endDrag();
    const ray = this.host.rayAt(e.clientX, e.clientY);
    const event = this.eventFor(d, ray);
    if (!event) return;
    try {
      this.host.dispatch(event);
    } catch (err) {
      // A raise ends the gesture: no half-applied drag keeps following the cursor.
      this.endDrag();
      throw err;
    }
  }

  /** The one event a move makes: the offset (translate) or the whole pose (rotate). */
  private eventFor(d: Drag, ray: Ray): Event | null {
    if (d.mode === "translate") {
      const s = axisParam(d.axisPoint, d.axisDir, ray, DRAG_MIN_SEPARATION)!;
      return { type: "setClipOffset", id: d.id, offset: d.offset0 + (s - d.s0) };
    }
    const p = spherePoint(d.centre, d.radius, ray);
    if (!p) return null;
    const vec = sub(p, d.centre);
    const m = Math.hypot(vec[0], vec[1], vec[2]);
    // The arrow points at the kept side; the stored normal folds the flip back out.
    const k = (d.flipped ? -1 : 1) / m;
    const normal: Vec3 = [vec[0] * k, vec[1] * k, vec[2] * k];
    // Through the frozen quad centre: the plane pivots there, not about the world origin. A zero or
    // non-finite normal (the cursor on the centre) reaches the reducer, which refuses it and keeps the pose.
    return { type: "setClipPose", id: d.id, normal, offset: dot(d.centre, normal) };
  }

  private onUp(e: PointerEvent): void {
    const d = this.drag;
    if (!d || e.pointerId !== d.pointer) return;
    this.claim(e);
    this.endDrag();
    // A press and release within the slop is a click: the model under the quad is picked through it.
    if (!d.moved) this.host.select(e.clientX, e.clientY);
  }

  private endDrag(): void {
    this.drag = null;
  }

  private hover(x: number, y: number): void {
    const style = this.host.element.style;
    if (!style) return;
    let hit = this.host.hit(this.host.rayAt(x, y));
    if (hit && this.host.chromeAt(x, y)) hit = null;
    const cursor = hit ? (hit.mode === "rotate" ? "grab" : "move") : "";
    if (cursor === this.cursor) return;
    this.cursor = cursor;
    style.cursor = cursor;
  }
}
