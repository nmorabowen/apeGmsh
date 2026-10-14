// Section planes (V4e1, #1834): the store's side of ADR 0083 Part 1. The
// plane records, their reducer (refusals are notices, never a changed
// plane), and the pure geometry the renderer and the pick filter share: the
// half-space each active plane keeps, whether a point is cut away, and the
// range the offset slider maps.
//
// A plane keeps the half-space its normal points into: `p` survives when
// `n·p - offset >= 0`; a flipped plane keeps the other half. Every active
// plane cuts (the kept region is the intersection of the kept halves), as
// ADR 0083's mapper clip planes do and three.js does by default.
//
// The Qt controller this mirrors is src/apeGmsh/viewers/core/_clip_planes.py
// (read-only context; nothing of it is imported). One deliberate difference:
// a zero normal is refused here, where Qt falls back to +X.
//
// V4e2 (#1903, ADR 0083 S2): the gizmo's own events. Its translate drag is
// `setClipOffset`, one per move; its rotate is `setClipPose`, one event that
// sets the normal and the offset together (Qt `set_pose`), so a rotate never
// shows a frame with the new normal and the old offset. The footer's "Show
// plane gizmos" and each plane's eye decide which planes get one
// (`gizmoPlanes`).

import type { ClipPlane, ClipState, Event } from "./types.ts";

export type Vec3 = readonly [number, number, number];

/** The most planes that may cut at once: the WebGL/OpenGL guaranteed minimum of user clip distances (ADR 0083 C-LIMIT). */
export const MAX_ACTIVE_PLANES = 6;

export const initialClip: ClipState = { planes: [], applyCuts: true, showGizmos: true, cutFace: true, seq: 0, notice: null };

/** The clip events, the reducer's own slice of the event union. */
export type ClipEvent = Extract<Event, { type: "addClipPlane" | "removeClipPlane" | "setClipActive" | "setClipFlipped" | "setClipOffset" | "setClipNormal" | "setApplyCuts" | "setClipPose" | "setShowGizmos" | "setClipGizmo" | "setCutFace" }>;

export const LIMIT_NOTICE = `${MAX_ACTIVE_PLANES} section planes are cutting already, the most the renderer guarantees: deactivate one first.`;

/** `v` as a unit vector; null when it has no direction (zero, too short to normalise, or not three finite numbers). */
export function unitNormal(v: readonly number[]): Vec3 | null {
  if (v.length !== 3 || !v.every((x) => typeof x === "number" && Number.isFinite(x))) return null;
  const [x, y, z] = v as [number, number, number];
  const m = Math.hypot(x, y, z);
  if (!(m > 1e-12) || !Number.isFinite(m)) return null;
  return [x / m, y / m, z / m];
}

const dot = (a: readonly number[], b: readonly number[]): number => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;

export const activeCount = (c: ClipState): number => c.planes.filter((p) => p.active).length;

/** The planes whose gizmo is drawn: each with its eye open while "Show plane gizmos" is on, active or not. */
export function gizmoPlanes(c: ClipState): ClipPlane[] {
  return c.showGizmos ? c.planes.filter((p) => p.gizmo) : [];
}

/** The orientation chip a normal lights: an axis when it is exactly +X, +Y or +Z, else custom. */
export function orientationOf(normal: Vec3): "x" | "y" | "z" | "custom" {
  const [x, y, z] = normal;
  if (x === 1 && y === 0 && z === 0) return "x";
  if (x === 0 && y === 1 && z === 0) return "y";
  if (x === 0 && y === 0 && z === 1) return "z";
  return "custom";
}

/** One half-space a plane keeps: a point `p` survives when `n·p + constant >= 0` (three.js `Plane` form). */
export interface Cut {
  normal: Vec3;
  constant: number;
}

/** The half-space a plane keeps, its flip applied. */
export function cutOf(p: ClipPlane): Cut {
  return p.flipped
    ? { normal: [-p.normal[0], -p.normal[1], -p.normal[2]], constant: p.offset }
    : { normal: p.normal, constant: -p.offset };
}

/** The half-spaces that cut now: every active plane while "Apply cuts" is on, none otherwise. */
export function cutsOf(c: ClipState): Cut[] {
  return c.applyCuts ? c.planes.filter((p) => p.active).map(cutOf) : [];
}

/**
 * Whether `p` is on the discarded side of any cut (ADR 0083 C-PICK): a
 * signed distance below `-tol`. `tol` keeps a hit that lands on the cut face
 * from being rejected by floating-point noise.
 */
export function isClipped(cuts: readonly Cut[], p: readonly number[], tol = 0): boolean {
  for (const c of cuts) if (dot(c.normal, p) + c.constant < -tol) return true;
  return false;
}

/** An axis-aligned box. */
export interface Box {
  lo: Vec3;
  hi: Vec3;
}

/** The box of every xyz triple in `arrays`; null when there is none. */
export function boxOf(...arrays: readonly Float32Array[]): Box | null {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const a of arrays) {
    for (let i = 0; i + 2 < a.length; i += 3) {
      for (let k = 0; k < 3; k++) {
        const v = a[i + k]!;
        if (v < lo[k]!) lo[k] = v;
        if (v > hi[k]!) hi[k] = v;
      }
    }
  }
  if (!(lo[0]! <= hi[0]!)) return null;
  return { lo: lo as unknown as Vec3, hi: hi as unknown as Vec3 };
}

/**
 * The offsets a plane of this normal sweeps across `box` (the offset
 * slider's range, ADR 0083 Part 4): the box's centre along the normal, plus
 * and minus its half-extent projected on it. A box flat along the normal
 * gets a tiny span, so the slider still moves.
 */
export function offsetRange(normal: Vec3, box: Box): { min: number; max: number } {
  const c = [0, 1, 2].map((k) => (box.lo[k]! + box.hi[k]!) / 2);
  const h = [0, 1, 2].map((k) => (box.hi[k]! - box.lo[k]!) / 2);
  const mid = dot(normal, c);
  const half = Math.abs(normal[0]) * h[0]! + Math.abs(normal[1]) * h[1]! + Math.abs(normal[2]) * h[2]!;
  const span = Math.max(half, 1e-9 * Math.max(1, Math.abs(mid)));
  return { min: mid - span, max: mid + span };
}

function planeOf(c: ClipState, id: string, who: string): ClipPlane {
  const p = c.planes.find((q) => q.id === id);
  if (!p) throw new Error(`${who}: no section plane ${JSON.stringify(id)}`);
  return p;
}

const refuse = (c: ClipState, notice: string): ClipState => (c.notice === notice ? c : { ...c, notice });
const accept = (c: ClipState, planes: readonly ClipPlane[], extra: Partial<ClipState> = {}): ClipState => ({ ...c, ...extra, planes, notice: null });
const replace = (c: ClipState, id: string, patch: Partial<ClipPlane>): ClipState => accept(c, c.planes.map((p) => (p.id === id ? { ...p, ...patch } : p)));

/**
 * The clip slice's reducer. `center` is the reference mesh centre (the
 * loader's bounds of every node, undeformed): a new plane passes through it,
 * and a new normal keeps the plane through the point of the old plane
 * nearest it. Refusals (a 7th active plane, a zero or non-finite normal, a
 * non-finite offset) leave every plane as it was and set the notice; an
 * unknown plane id is a programming error and raises.
 */
export function reduceClip(c: ClipState, e: ClipEvent, center: Vec3): ClipState {
  switch (e.type) {
    case "addClipPlane": {
      const n = unitNormal(e.normal);
      if (!n) return refuse(c, `A section plane needs a direction: ${JSON.stringify(e.normal)} has none.`);
      const active = activeCount(c) < MAX_ACTIVE_PLANES;
      const seq = c.seq + 1;
      const plane: ClipPlane = { id: `plane-${seq}`, normal: n, offset: dot(n, center), flipped: false, active, gizmo: true };
      // Created inactive past the limit: the plane exists, ready to swap in for one of the six (the Qt behaviour).
      const next = accept(c, [...c.planes, plane], { seq });
      return active ? next : { ...next, notice: LIMIT_NOTICE };
    }
    case "removeClipPlane":
      planeOf(c, e.id, "removeClipPlane");
      return accept(c, c.planes.filter((p) => p.id !== e.id));
    case "setClipActive": {
      if (typeof e.active !== "boolean") throw new RangeError(`setClipActive: ${JSON.stringify(e.active)}`);
      const p = planeOf(c, e.id, "setClipActive");
      if (p.active === e.active) return c;
      if (e.active && activeCount(c) >= MAX_ACTIVE_PLANES) return refuse(c, LIMIT_NOTICE);
      return replace(c, e.id, { active: e.active });
    }
    case "setClipFlipped": {
      if (typeof e.flipped !== "boolean") throw new RangeError(`setClipFlipped: ${JSON.stringify(e.flipped)}`);
      const p = planeOf(c, e.id, "setClipFlipped");
      return p.flipped === e.flipped ? c : replace(c, e.id, { flipped: e.flipped });
    }
    case "setClipOffset": {
      const p = planeOf(c, e.id, "setClipOffset");
      if (typeof e.offset !== "number" || !Number.isFinite(e.offset)) return refuse(c, `A section plane offset must be a finite number; ${String(e.offset)} is not.`);
      return p.offset === e.offset && c.notice === null ? c : replace(c, e.id, { offset: e.offset });
    }
    case "setClipNormal": {
      const p = planeOf(c, e.id, "setClipNormal");
      const n = unitNormal(e.normal);
      if (!n) return refuse(c, `A section plane needs a direction: ${JSON.stringify(e.normal)} has none.`);
      // The point of the old plane nearest the centre stays on the new one (Qt `set_pose`).
      const d = dot(p.normal, center) - p.offset;
      const keep = [0, 1, 2].map((k) => center[k]! - d * p.normal[k]!);
      return replace(c, e.id, { normal: n, offset: dot(n, keep) });
    }
    case "setApplyCuts":
      if (typeof e.on !== "boolean") throw new RangeError(`setApplyCuts: ${JSON.stringify(e.on)}`);
      return c.applyCuts === e.on ? c : { ...c, applyCuts: e.on, notice: null };
    case "setClipPose": {
      const p = planeOf(c, e.id, "setClipPose");
      const raw = e.normal;
      const n = unitNormal(raw);
      if (!n) return refuse(c, `A section plane needs a direction: ${JSON.stringify(raw)} has none.`);
      if (typeof e.offset !== "number" || !Number.isFinite(e.offset)) return refuse(c, `A section plane offset must be a finite number; ${String(e.offset)} is not.`);
      // `normal · p = offset` scaled to the unit normal: the same plane, whatever length the normal came in.
      const offset = e.offset / Math.hypot(raw[0]!, raw[1]!, raw[2]!);
      if (!Number.isFinite(offset)) return refuse(c, `A section plane offset must be a finite number; ${String(offset)} is not.`);
      const same = p.normal.every((x, k) => x === n[k]) && p.offset === offset;
      return same && c.notice === null ? c : replace(c, e.id, { normal: n, offset });
    }
    case "setShowGizmos":
      if (typeof e.on !== "boolean") throw new RangeError(`setShowGizmos: ${JSON.stringify(e.on)}`);
      return c.showGizmos === e.on ? c : { ...c, showGizmos: e.on, notice: null };
    case "setCutFace":
      if (typeof e.on !== "boolean") throw new RangeError(`setCutFace: ${JSON.stringify(e.on)}`);
      return c.cutFace === e.on ? c : { ...c, cutFace: e.on, notice: null };
    case "setClipGizmo": {
      if (typeof e.visible !== "boolean") throw new RangeError(`setClipGizmo: ${JSON.stringify(e.visible)}`);
      const p = planeOf(c, e.id, "setClipGizmo");
      return p.gizmo === e.visible ? c : replace(c, e.id, { gizmo: e.visible });
    }
    default: {
      const unknown: never = e;
      throw new Error(`reduceClip: unknown event ${JSON.stringify(unknown)}`);
    }
  }
}
