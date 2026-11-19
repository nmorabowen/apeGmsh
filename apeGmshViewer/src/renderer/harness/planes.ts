// The capture and measure harness (VH1, #2865): section planes placed through the store.

import { offsetRange } from "../../state/clip.ts";
import type { HarnessContext } from "./context.ts";

/** V4e1: six active section planes, one per face of the reference box, each 20 % of its extent inside it. */
export function addSixPlanes(ctx: HarnessContext) {
  const { store, viewport } = ctx;
  const box = viewport.sliderBox();
  if (!box) throw new Error("section planes: no mesh");
  const normals: [number, number, number][] = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
  for (const n of normals) {
    store.dispatch({ type: "addClipPlane", normal: n });
    const p = store.get().clip.planes.at(-1)!;
    const r = offsetRange(p.normal, box);
    store.dispatch({ type: "setClipOffset", id: p.id, offset: r.min + 0.2 * (r.max - r.min) });
  }
}

export function removePlanes(ctx: HarnessContext) {
  const { store } = ctx;
  for (const p of store.get().clip.planes) store.dispatch({ type: "removeClipPlane", id: p.id });
}

/** The id of the plane added last. */
const lastPlane = (ctx: HarnessContext): string => ctx.store.get().clip.planes.at(-1)!.id;

/** Put plane `id` at fraction `t` of the reference box along its normal (0: the low end). */
export function placeAt(ctx: HarnessContext, id: string, t: number) {
  const { store, viewport } = ctx;
  const box = viewport.sliderBox()!;
  const p = store.get().clip.planes.find((q) => q.id === id)!;
  const r = offsetRange(p.normal, box);
  store.dispatch({ type: "setClipOffset", id, offset: r.min + t * (r.max - r.min) });
}

/** Add a plane of normal `n` at fraction `t` of the reference box along it; its id. */
export function addAt(ctx: HarnessContext, n: readonly number[], t: number): string {
  const { store } = ctx;
  store.dispatch({ type: "addClipPlane", normal: n });
  const id = lastPlane(ctx);
  placeAt(ctx, id, t);
  return id;
}
