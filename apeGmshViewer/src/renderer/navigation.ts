// Camera navigation: a Z-up turntable that orbits about the point under the
// cursor, pans by grabbing that point, and zooms toward it. What each mouse
// button does comes from bindings.ts.
//
// three's OrbitControls orbits about `target` and re-aims the camera at it
// (lookAt), so a pivot off the screen centre would snap the view. It also
// takes its orbit axis from `camera.up` once, in its constructor; the old
// viewport set `up` to Z after that, so the model orbited about Y and
// tumbled. Here the camera's orientation is held as a heading (yaw about
// world Z, tilt from looking straight down), so it has no roll by
// construction, and an orbit rotates position and orientation together about
// the pivot, which keeps the pivot on the same pixel.

import * as THREE from "three";
import { bindingFor, buttonOf, CLICK_SLOP_PX, keyAction, type DragAction } from "./bindings.ts";

export const Z_UP: Readonly<THREE.Vector3> = new THREE.Vector3(0, 0, 1);

/** Camera orientation with no roll: yaw about world Z, tilt in [0, pi] from looking straight down. */
export interface Heading {
  yaw: number;
  tilt: number;
}

const X_AXIS = new THREE.Vector3(1, 0, 0);

/** The quaternion of a heading: Rz(yaw) * Rx(tilt). Heading {0, 0} looks down -Z with +Y up the screen. */
export function orientationOf(h: Heading): THREE.Quaternion {
  const qz = new THREE.Quaternion().setFromAxisAngle(Z_UP, h.yaw);
  return qz.multiply(new THREE.Quaternion().setFromAxisAngle(X_AXIS, h.tilt));
}

/** The heading whose view direction is `forward` (any length). Straight down or up gives yaw 0. */
export function headingOf(forward: THREE.Vector3): Heading {
  const f = forward.clone().normalize();
  const tilt = Math.acos(THREE.MathUtils.clamp(-f.z, -1, 1));
  const horizontal = Math.hypot(f.x, f.y);
  return { yaw: horizontal < 1e-12 ? 0 : Math.atan2(-f.x, f.y), tilt };
}

/** Point the camera along a heading. */
export function applyHeading(camera: THREE.Camera, h: Heading): void {
  camera.quaternion.copy(orientationOf(h));
  camera.updateMatrixWorld();
}

/**
 * Turntable orbit about `pivot`: yaw about the world Z axis through it, tilt
 * about the camera's (horizontal) right axis through it, tilt clamped to
 * [0, pi]. Position and orientation turn together, so the pivot keeps its
 * pixel. The camera must already be at `heading`. Returns the new heading.
 */
export function orbitAbout(camera: THREE.Camera, heading: Heading, pivot: THREE.Vector3, dYaw: number, dTilt: number): Heading {
  const next = { yaw: heading.yaw + dYaw, tilt: THREE.MathUtils.clamp(heading.tilt + dTilt, 0, Math.PI) };
  const before = orientationOf(heading);
  const after = orientationOf(next);
  const turn = after.clone().multiply(before.invert());
  camera.position.sub(pivot).applyQuaternion(turn).add(pivot);
  camera.quaternion.copy(after);
  camera.updateMatrixWorld();
  return next;
}

/** Distance from the camera to `point` along the view direction. */
export function depthOf(camera: THREE.Camera, point: THREE.Vector3): number {
  const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  return point.clone().sub(camera.position).dot(forward);
}

/**
 * Grab-pan: move the camera in its view plane so a point at `depth` follows
 * the cursor by (dxPx, dyPx) screen pixels (y down). `viewHeightPx` is the
 * canvas height the vertical field of view spans.
 */
export function panBy(camera: THREE.PerspectiveCamera, depth: number, dxPx: number, dyPx: number, viewHeightPx: number): void {
  const perPx = (2 * depth * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) / (camera.zoom * viewHeightPx);
  const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
  const up = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion);
  camera.position.addScaledVector(right, -dxPx * perPx).addScaledVector(up, dyPx * perPx);
  camera.updateMatrixWorld();
}

/**
 * Move the camera along the line to `point`, keeping `keep` of the distance
 * (below 1 zooms in, above 1 out). `point` keeps its pixel because it stays
 * on the same view ray and the orientation does not change.
 */
export function zoomToward(camera: THREE.Camera, point: THREE.Vector3, keep: number): void {
  camera.position.sub(point).multiplyScalar(keep).add(point);
  camera.updateMatrixWorld();
}

/** Zoom limits, as fractions of the model's bounding radius: the camera never comes closer to the zoom point than ZOOM_MIN, nor farther than ZOOM_MAX. */
export const ZOOM_MIN = 1e-3;
export const ZOOM_MAX = 40;

/**
 * The `keep` factor a wheel step may apply from `distance` (camera to the zoom
 * point), clamped so the new distance stays in [ZOOM_MIN, ZOOM_MAX] * radius.
 * A camera already outside the band can only move back into it.
 */
export function clampZoom(distance: number, keep: number, radius: number): number {
  const lo = ZOOM_MIN * radius, hi = ZOOM_MAX * radius;
  if (distance <= 0) return 1;
  const next = THREE.MathUtils.clamp(distance * keep, lo, hi);
  // Outside the band, allow only the direction that re-enters it.
  if (distance < lo && keep < 1) return 1;
  if (distance > hi && keep > 1) return 1;
  return next / distance;
}

/** The nearest point where the ray hits `targets`, or null on a miss. */
export function nearestHit(raycaster: THREE.Raycaster, targets: readonly THREE.Object3D[]): THREE.Vector3 | null {
  const hit = raycaster.intersectObjects(targets as THREE.Object3D[], false)[0];
  return hit ? hit.point.clone() : null;
}

/** The orbit pivot: the mesh point under the cursor, else the model's bounding-box centre. */
export function pivotFor(hit: THREE.Vector3 | null, bboxCenter: THREE.Vector3): THREE.Vector3 {
  return hit ? hit.clone() : bboxCenter.clone();
}

/** Near and far planes that enclose the bounding sphere from where the camera stands. */
export function clipPlanes(camera: THREE.PerspectiveCamera, center: THREE.Vector3, radius: number): void {
  const d = camera.position.distanceTo(center);
  camera.near = Math.max(d - 1.1 * radius, 1e-4 * radius);
  camera.far = d + 1.1 * radius;
  camera.updateProjectionMatrix();
}

/** The subset of a DOM event target the navigator listens on (a canvas, or `window` for keys). */
export interface ListenTarget {
  addEventListener(type: string, listener: (e: Event) => void, options?: AddEventListenerOptions | boolean): void;
  removeEventListener(type: string, listener: (e: Event) => void, options?: EventListenerOptions | boolean): void;
}

/** What the navigator needs from the viewport. */
export interface NavHost {
  readonly camera: THREE.PerspectiveCamera;
  /** the canvas: pointer and wheel events, pointer capture, its height for drag rates */
  readonly element: ListenTarget & {
    readonly clientHeight: number;
    setPointerCapture(id: number): void;
    hasPointerCapture(id: number): boolean;
    releasePointerCapture(id: number): void;
  };
  /** where key events arrive (the window) */
  readonly keys: ListenTarget;
  /** The model's bounding box centre and radius, or null when no model is loaded. */
  bounds(): { center: THREE.Vector3; radius: number } | null;
  /** The mesh point under a client-space point, or null on a miss. */
  hitAt(clientX: number, clientY: number): THREE.Vector3 | null;
  /** The world ray through a client-space point. */
  rayAt(clientX: number, clientY: number): THREE.Ray;
  select(clientX: number, clientY: number): void;
  /** Frame the whole model, keeping the heading. */
  fit(): void;
  changed(): void;
}

/** Wheel: one 100 px notch keeps 1/1.15 of the distance (zoom in) or 1.15 of it (out). */
const WHEEL_STEP = 1.15;

interface Drag {
  id: number;
  action: DragAction | null;
  click: boolean;
  x0: number;
  y0: number;
  x: number;
  y: number;
  pivot: THREE.Vector3;
  depth: number;
}

export class Navigator {
  heading: Heading = { yaw: 0, tilt: 0 };
  /** The centre the last fit framed; scripted orbits turn about it. */
  readonly target = new THREE.Vector3();
  private readonly host: NavHost;
  private drag: Drag | null = null;
  private readonly off: (() => void)[] = [];

  constructor(host: NavHost) {
    this.host = host;
    const on = <E>(target: ListenTarget, type: string, fn: (e: E) => void, options?: AddEventListenerOptions) => {
      const listener = fn as unknown as (e: Event) => void;
      target.addEventListener(type, listener, options);
      this.off.push(() => target.removeEventListener(type, listener, options));
    };
    const el = host.element;
    on<PointerEvent>(el, "pointerdown", (e) => this.onDown(e));
    on<PointerEvent>(el, "pointermove", (e) => this.onMove(e));
    on<PointerEvent>(el, "pointerup", (e) => this.onUp(e));
    on<PointerEvent>(el, "pointercancel", () => this.endDrag());
    // The press ends without a pointerup when the capture is lost (a window
    // switch, a release the OS swallowed): the drag state must not outlive it.
    on<PointerEvent>(el, "lostpointercapture", () => this.endDrag());
    on<Event>(el, "contextmenu", (e) => e.preventDefault());
    on<WheelEvent>(el, "wheel", (e) => this.onWheel(e), { passive: false });
    on<KeyboardEvent>(host.keys, "keydown", (e) => this.onKey(e));
  }

  /** True while a button is held (for tests). */
  get dragging(): boolean {
    return this.drag !== null;
  }

  /** Remove every listener this navigator added. */
  dispose(): void {
    for (const f of this.off.splice(0)) f();
    this.drag = null;
  }

  /** Set the heading directly (fit on load). */
  setHeading(h: Heading): void {
    this.heading = { ...h };
    applyHeading(this.host.camera, this.heading);
  }

  /** Orbit about a pivot (also the scripted orbit of `npm run measure`). */
  orbit(pivot: THREE.Vector3, dYaw: number, dTilt: number): void {
    this.heading = orbitAbout(this.host.camera, this.heading, pivot, dYaw, dTilt);
    this.updated();
  }

  /** Recompute the clip planes and redraw. */
  updated(): void {
    const b = this.host.bounds();
    if (b) clipPlanes(this.host.camera, b.center, b.radius);
    this.host.changed();
  }

  private endDrag(): void {
    this.drag = null;
  }

  private onDown(e: PointerEvent): void {
    const button = buttonOf(e.button);
    const b = this.host.bounds();
    if (!button || !b || this.drag) return;
    const binding = bindingFor(button, e.shiftKey);
    const anchor = pivotFor(binding.drag ? this.host.hitAt(e.clientX, e.clientY) : null, b.center);
    this.drag = {
      id: e.pointerId,
      action: binding.drag,
      click: binding.click === "select",
      x0: e.clientX, y0: e.clientY, x: e.clientX, y: e.clientY,
      pivot: anchor,
      depth: Math.max(depthOf(this.host.camera, anchor), 1e-3 * b.radius),
    };
    // Capture every press, a click-only one too: the release then reaches
    // this element wherever the pointer went, so no press is left pending.
    this.host.element.setPointerCapture(e.pointerId);
  }

  private onMove(e: PointerEvent): void {
    const d = this.drag;
    if (!d || e.pointerId !== d.id || !d.action) return;
    const dx = e.clientX - d.x, dy = e.clientY - d.y;
    if (dx === 0 && dy === 0) return;
    d.x = e.clientX;
    d.y = e.clientY;
    const h = this.host.element.clientHeight || 1;
    if (d.action === "orbit") {
      // A drag across the canvas height turns a full circle (OrbitControls' rate).
      this.orbit(d.pivot, (-2 * Math.PI * dx) / h, (-2 * Math.PI * dy) / h);
    } else {
      panBy(this.host.camera, d.depth, dx, dy, h);
      this.updated();
    }
  }

  private onUp(e: PointerEvent): void {
    const d = this.drag;
    if (!d || e.pointerId !== d.id) return;
    this.drag = null;
    if (this.host.element.hasPointerCapture(e.pointerId)) this.host.element.releasePointerCapture(e.pointerId);
    if (d.click && Math.hypot(e.clientX - d.x0, e.clientY - d.y0) <= CLICK_SLOP_PX) this.host.select(e.clientX, e.clientY);
  }

  private onWheel(e: WheelEvent): void {
    e.preventDefault();
    const b = this.host.bounds();
    if (!b || e.deltaY === 0) return;
    const px = e.deltaMode === 1 ? e.deltaY * 33 : e.deltaMode === 2 ? e.deltaY * 800 : e.deltaY;
    const keep = Math.pow(WHEEL_STEP, THREE.MathUtils.clamp(px, -300, 300) / 100);
    // On a miss, zoom toward the cursor ray at the depth of the model's centre.
    let point = this.host.hitAt(e.clientX, e.clientY);
    if (!point) {
      const ray = this.host.rayAt(e.clientX, e.clientY);
      const depth = depthOf(this.host.camera, b.center);
      const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(this.host.camera.quaternion);
      const along = depth > 1e-3 * b.radius ? depth / ray.direction.dot(forward) : b.radius;
      point = ray.at(along, new THREE.Vector3());
    }
    const limited = clampZoom(this.host.camera.position.distanceTo(point), keep, b.radius);
    if (limited === 1) return;
    zoomToward(this.host.camera, point, limited);
    this.updated();
  }

  private onKey(e: KeyboardEvent): void {
    if (e.ctrlKey || e.altKey || e.metaKey || e.repeat) return;
    const t = e.target as HTMLElement | null;
    if (t && (t.isContentEditable || t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT")) return;
    if (keyAction(e.key) === "fit" && this.host.bounds()) this.host.fit();
  }
}
