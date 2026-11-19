// The capture and measure harness (VH1, #2865): whether the frame on screen shows what the state asks for.

import { contourView, deformView } from "../../state/selectors.ts";
import { contourKey, contourShownKey, elementContourKey } from "../contour.ts";
import type { HarnessContext } from "./context.ts";

/** The contour colours on screen are the step the state shows, at its range (renderer/contour.ts keys); V4c2a: an element field's too. */
export function contourShown(ctx: HarnessContext): boolean {
  const { store, viewport } = ctx;
  const v = contourView(store.get());
  if (v.state === "ready") return v.stepMeasured && viewport.contourApplied === contourKey(v);
  if (v.state === "element") return v.stepMeasured && viewport.contourApplied === elementContourKey(v);
  return false;
}

/** V4c2a: the mesh shows a refused or not-yet element contour as no data (its key names the reason). */
export function refusalShown(ctx: HarnessContext): boolean {
  const { store, viewport } = ctx;
  const v = contourView(store.get());
  return (v.state === "refused" || v.state === "not-yet") && viewport.contourApplied === contourShownKey(v);
}

/** V4d: the drawn positions are the deformed shape of the step the state shows, at its scale (renderer/deform.ts keys); or the shape cannot be drawn. */
export function deformShown(ctx: HarnessContext): boolean {
  const { store, effects, viewport } = ctx;
  const v = deformView(store.get());
  if (v.state === "off" || v.state === "unavailable" || v.state === "refused") return true;
  if (v.state === "loading") return false;
  return viewport.deformApplied === `${effects.currentDisplacement()?.key}|scale ${v.scale}` && effects.currentDisplacement()?.key === v.keys.disp;
}

/** Resolves once the contour (and, V4d, the deformed shape) shows the step the state names (checked after every store change; the viewport listens first). */
export function contourShowsNow(ctx: HarnessContext): Promise<void> {
  const { store } = ctx;
  const shown = () => contourShown(ctx) && deformShown(ctx);
  return new Promise((resolve) => {
    if (shown()) return resolve();
    const off = store.subscribe(() => {
      if (!shown()) return;
      off();
      resolve();
    });
  });
}
