// The camera slice's events: its part of the store's Event union (types/event.ts).
// Not re-exported by src/state/types.ts, which exports only the whole union.

import type { CameraPose, CameraRequest } from "../camera.ts";

export type CameraEvent =
  /**
   * VA1: move the camera to a named view, an explicit pose or a heading; the
   * viewport applies it. A pose that is not finite, has eye = target, an up
   * along the view direction, or an up that would roll the camera is refused.
   * A named view or a turn with nothing drawn, and `fit-selection` with
   * nothing selected, are no-ops.
   */
  | { type: "setCamera"; to: CameraRequest }
  /** VA1 (viewport): the camera after an orbit, pan, zoom or fit, or after it applied a `setCamera` */
  | { type: "cameraMoved"; pose: CameraPose };
