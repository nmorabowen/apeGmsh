// The camera slice of the reducer (state/reduce.ts delegates here): VA1's
// pose, the requests the viewport resolves, and the checks on both.

import type { CameraEvent } from "../types/events/camera.ts";
import { NAMED_VIEWS, type CameraPose, type CameraRequest, type CameraState, type Vec3 } from "../types.ts";

/** VA1: no camera until the viewport first frames a scene. */
export const initialCamera: CameraState = { pose: null, request: null, seq: 0, read: null };

/** The largest camera roll (radians) an explicit `up` may imply; the camera is a Z-up turntable with none. */
export const ROLL_TOLERANCE = 1e-6;

function vec3Of(v: unknown, who: string, key: string): Vec3 {
  if (!Array.isArray(v) || v.length !== 3 || !v.every((x) => typeof x === "number" && Number.isFinite(x))) {
    throw new RangeError(`${who}: ${key} must be three finite numbers; got ${JSON.stringify(v)}`);
  }
  return [v[0] as number, v[1] as number, v[2] as number];
}

/**
 * VA1: a camera pose, checked and made canonical: `up` becomes the unit
 * screen-up, its part across the view direction. Refused, with the reason: a
 * key other than eye, target and up; a value that is not three finite
 * numbers; eye = target; an up along the view direction; and an up that
 * would roll the Z-up camera by more than ROLL_TOLERANCE. Looking straight
 * down or up, any up across the view is allowed: it sets the heading.
 */
export function canonicalPose(p: CameraPose, who: string): CameraPose {
  const keys = Object.keys(p).sort().join(", ");
  if (keys !== "eye, target, up") throw new RangeError(`${who}: a pose has exactly eye, target and up; got ${keys || "no key"}`);
  const eye = vec3Of(p.eye, who, "eye");
  const target = vec3Of(p.target, who, "target");
  const up = vec3Of(p.up, who, "up");
  const f = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]];
  const fl = Math.hypot(f[0]!, f[1]!, f[2]!);
  if (!(fl > 0)) throw new RangeError(`${who}: eye and target are the same point ${JSON.stringify(eye)}`);
  const [fx, fy, fz] = [f[0]! / fl, f[1]! / fl, f[2]! / fl];
  const along = up[0] * fx + up[1] * fy + up[2] * fz;
  const across = [up[0] - along * fx, up[1] - along * fy, up[2] - along * fz];
  const al = Math.hypot(across[0]!, across[1]!, across[2]!);
  if (!(al > 1e-9 * Math.hypot(...up))) throw new RangeError(`${who}: up ${JSON.stringify(up)} is along the view direction`);
  const u: Vec3 = [across[0]! / al, across[1]! / al, across[2]! / al];
  // A Z-up turntable's screen-up lies in the plane of the view direction and
  // world Z, on the +Z side: (Z - fz f) / |..|, whose length is hypot(fx, fy).
  const h = Math.hypot(fx, fy);
  if (h > 1e-12) {
    const [ix, iy, iz] = [(-fz * fx) / h, (-fz * fy) / h, h];
    const cross = Math.hypot(iy * u[2] - iz * u[1], iz * u[0] - ix * u[2], ix * u[1] - iy * u[0]);
    const roll = Math.atan2(cross, ix * u[0] + iy * u[1] + iz * u[2]);
    // target - eye is rounded: each component of the view direction is off by
    // up to about eps * |coordinate| / |target - eye|, which turns the no-roll
    // up by that much over h. Near the vertical, far from the origin (a
    // georeferenced model), that exceeds ROLL_TOLERANCE, so it is allowed for.
    const scale = Math.max(...eye.map(Math.abs), ...target.map(Math.abs));
    const tolerance = ROLL_TOLERANCE + (16 * Number.EPSILON * scale) / (fl * h);
    if (!(roll <= tolerance)) {
      const deg = (roll * 180) / Math.PI;
      throw new RangeError(`${who}: up ${JSON.stringify(up)} rolls the camera by ${deg.toPrecision(4)} degrees; the camera has no roll (world Z stays up the screen)`);
    }
  }
  return { eye, target, up: u };
}

const samePose = (a: CameraPose, b: CameraPose): boolean =>
  a.eye.every((x, i) => x === b.eye[i]) && a.target.every((x, i) => x === b.target[i]) && a.up.every((x, i) => x === b.up[i]);

/** What a camera request reads of the rest of the state. */
export interface CameraInputs {
  /** A mesh or a geometry is loaded. */
  drawn: boolean;
  /** Nothing selected resolves to an element; asked only for `fit-selection`. */
  nothingSelected: () => boolean;
}

/** VA1: a `setCamera` request checked; null when it is a defined no-op (nothing drawn, or `fit-selection` with nothing selected). */
function cameraRequestOf(to: CameraRequest, inputs: CameraInputs): CameraRequest | null {
  const { drawn } = inputs;
  if (typeof to === "string") {
    if (!(NAMED_VIEWS as readonly string[]).includes(to)) throw new RangeError(`setCamera: ${JSON.stringify(to)} is not a named view (${NAMED_VIEWS.join(", ")})`);
    if (!drawn) return null;
    if (to === "fit-selection" && inputs.nothingSelected()) return null;
    return to;
  }
  if (to !== null && typeof to === "object" && "eye" in to) return canonicalPose(to, "setCamera");
  if (to !== null && typeof to === "object" && "yaw" in to) {
    const keys = Object.keys(to).sort().join(", ");
    if (keys !== "tilt, yaw") throw new RangeError(`setCamera: a turn has exactly yaw and tilt; got ${keys}`);
    if (!Number.isFinite(to.yaw) || !Number.isFinite(to.tilt)) throw new RangeError(`setCamera: a turn needs a finite yaw and tilt; got ${JSON.stringify(to)}`);
    if (!drawn) return null;
    return { yaw: to.yaw, tilt: Math.min(Math.PI, Math.max(0, to.tilt)) };
  }
  throw new RangeError(`setCamera: ${JSON.stringify(to)} is not a named view, a pose (eye, target, up) or a turn (yaw, tilt)`);
}

/** The camera after `e`: the same object when nothing changed. */
export function reduceCamera(c: CameraState, e: CameraEvent, inputs: CameraInputs): CameraState {
  switch (e.type) {
    case "setCamera": {
      const to = cameraRequestOf(e.to, inputs);
      if (to === null) return c;
      const seq = c.seq + 1;
      // An explicit pose is the camera at once; a named view or a turn is resolved by the viewport, which writes the pose back.
      const pose = typeof to === "object" && "eye" in to ? to : c.pose;
      return { ...c, pose, request: { to, seq }, seq };
    }
    case "cameraMoved": {
      const pose = canonicalPose(e.pose, "cameraMoved");
      if (c.pose && samePose(c.pose, pose)) return c;
      return { ...c, pose };
    }
    default: {
      const unknown: never = e;
      throw new Error(`reduceCamera: unknown event ${JSON.stringify(unknown)}`);
    }
  }
}
