// The camera slice (ADR 0128 D1): the named views, a pose, and the request to move.
// One slice of the store's types; src/state/types.ts re-exports it, and the
// panels import it from there. A slice file imports sibling files, never the barrel.

/**
 * VA1 (ADR 0128 D1): the named camera views. Each of `iso` to `bottom` turns
 * to a fixed heading and fits the whole drawn scene; `+x` is the view from
 * the +X side (the ViewCube's RIGHT face), `top` looks down -Z with +Y up the
 * screen. `fit` keeps the heading and fits the whole scene, `fit-selection`
 * keeps it and fits the selection's drawn bounds, and `home` is the view a
 * model opens on (plan for a planar model, else `iso`), fitted.
 */
export const NAMED_VIEWS = ["iso", "+x", "-x", "+y", "-y", "top", "bottom", "fit", "fit-selection", "home"] as const;
export type NamedView = (typeof NAMED_VIEWS)[number];

export type Vec3 = readonly [number, number, number];

/**
 * A camera as a look-at: the eye, a target on the view axis, and the unit
 * screen-up vector. The camera has no roll (a Z-up turntable), so `up` is
 * fixed by the view direction except when looking straight down or up, where
 * it sets the heading. In the orthographic projection the frustum height is
 * set by |eye - target|, as the perspective view at that distance shows it.
 */
export interface CameraPose {
  eye: Vec3;
  target: Vec3;
  up: Vec3;
}

/** Turn to a heading (yaw about world Z, tilt in [0, pi] from looking straight down) about the orbit pivot, keeping the distance: the ViewCube's clicks and drag. */
export interface CameraTurn {
  yaw: number;
  tilt: number;
}

/** What `setCamera` asks for: a named view, an explicit pose, or a turn. */
export type CameraRequest = NamedView | CameraPose | CameraTurn;

export interface CameraState {
  /**
   * The camera shown now. The viewport writes it back (`cameraMoved`) after
   * every orbit, pan, zoom or fit and after it applies a request, once the
   * view has held still for a frame (a drag writes once, when it stops), so
   * the store holds the current camera whenever the view is at rest. An
   * explicit pose is recorded at once by `setCamera`. null until the viewport first
   * frames a scene, and again when another model is opened; a re-read of the
   * same model keeps it, and the view with it.
   */
  pose: CameraPose | null;
  /** The latest `setCamera` request; the viewport applies each new `seq` once. Cleared when another model is opened. */
  request: { to: CameraRequest; seq: number } | null;
  /** Counts `setCamera` requests for the whole session; never reset, so a request can never reuse a seq the viewport saw. */
  seq: number;
  /**
   * The model path read at least once since it was opened, or null. The pose
   * belongs to it: a load or a failed read of this path keeps the camera (a
   * watch reload, a retry after a failed re-read); a path opened but never
   * read (its first read pending, or failed) opens framed.
   */
  read: string | null;
}
