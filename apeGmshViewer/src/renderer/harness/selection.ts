// The selection stills of the capture harness (ADR 0138 SEL-1b, SEL-2), for
// every model with a mesh:
//
// - `<out>.selection-click.png` (models with line elements): a click at the
//   middle of the line element drawn nearest the centre, which it must take.
// - `<out>.selection-hover.png`: the pointer over an element, through the
//   canvas's own pointermove listener, so the element is marked and named.
// - `<out>.selection-box.png`: a box drawn left to right over the middle of the
//   canvas, mid-drag, with the count of elements fully inside it.
// - `<out>.selection-boxed.png`: the same box released: those elements selected.
// - `<out>.selection-nodes-box.png`: Nodes active, a box drawn right to left,
//   mid-drag, with the node markers on.
//
// The box goes through the viewport's own `boxAt` and `boxEnd`, the calls a
// drag makes (renderer/navigation.ts); a scripted pointer cannot take the
// pointer capture a real press takes.

import type { HarnessContext } from "./context.ts";

export async function captureSelection(ctx: HarnessContext, settle: (ms?: number) => Promise<unknown>): Promise<void> {
  const { store, viewport, bridge } = ctx;
  const s = store.get();
  if (!s.mesh || s.mesh.elements.length === 0) return;
  const shot = async (name: string) => {
    await settle();
    viewport.renderNow();
    await bridge.captureStill(name);
  };
  const canvas = document.querySelector<HTMLCanvasElement>("#viewport canvas");
  if (!canvas) throw new Error("capture: no viewport canvas");
  store.dispatch({ type: "clearSelection", kind: "element" });
  await captureClick(ctx, canvas, shot);
  const middle = s.mesh.elements[Math.floor(s.mesh.elements.length / 2)]!;
  const pt = viewport.screenPointOf(middle);
  if (pt) {
    canvas.dispatchEvent(new PointerEvent("pointermove", { clientX: pt.x, clientY: pt.y, buttons: 0, bubbles: true }));
    await settle(100);
    await shot("selection-hover");
    canvas.dispatchEvent(new PointerEvent("pointerleave", { bubbles: true }));
  }
  const r = canvas.getBoundingClientRect();
  const a = { x: r.left + r.width * 0.3, y: r.top + r.height * 0.3 }, b = { x: r.left + r.width * 0.6, y: r.top + r.height * 0.7 };
  viewport.boxAt(a.x, a.y, b.x, b.y, "replace");
  await shot("selection-box");
  viewport.boxEnd(a.x, a.y, b.x, b.y, "replace");
  await shot("selection-boxed");
  store.dispatch({ type: "setWorkspaceKind", kind: "nodes" });
  store.dispatch({ type: "setNodes", on: true });
  await settle();
  viewport.boxAt(b.x, b.y, a.x, a.y, "replace");
  await shot("selection-nodes-box");
  viewport.boxEnd(b.x, b.y, a.x, a.y, "replace");
  await shot("selection-nodes-boxed");
  store.dispatch({ type: "clearSelection", kind: "node" });
  store.dispatch({ type: "clearSelection", kind: "element" });
  store.dispatch({ type: "setNodes", on: false });
  store.dispatch({ type: "setWorkspaceKind", kind: "elements" });
  await settle();
}

/**
 * `<out>.selection-click.png` (ADR 0138 D5), for a model with line elements:
 * a click at the middle of the line element drawn nearest the canvas centre,
 * selected and hovered. The click must take that element: the ray cast clicks
 * used before took the neighbour of a segment shorter than about 10 px, so a
 * capture of the slab model fails if that comes back.
 */
async function captureClick(ctx: HarnessContext, canvas: HTMLCanvasElement, shot: (name: string) => Promise<void>): Promise<void> {
  const { store, viewport, blobs } = ctx;
  const info = store.get().mesh!;
  const r = canvas.getBoundingClientRect(), cx = r.left + r.width / 2, cy = r.top + r.height / 2;
  let best: { decl: string; x: number; y: number } | null = null, bestD = Infinity;
  for (const e of new Set(blobs.i32(info.lineElement))) {
    const decl = info.elements[e]!;
    const pt = viewport.screenPointOf(decl);
    if (!pt || pt.x < r.left || pt.x > r.right || pt.y < r.top || pt.y > r.bottom) continue;
    const d = Math.hypot(pt.x - cx, pt.y - cy);
    if (d < bestD) [best, bestD] = [{ decl, ...pt }, d];
  }
  if (!best) return;
  const pick = viewport.pickAt(best.x, best.y);
  if (pick?.decl !== best.decl) throw new Error(`capture selection-click: a click at the middle of ${best.decl} took ${pick?.decl ?? "nothing"}`);
  store.dispatch({ type: "select", pick });
  canvas.dispatchEvent(new PointerEvent("pointermove", { clientX: best.x, clientY: best.y, buttons: 0, bubbles: true }));
  await shot("selection-click");
  canvas.dispatchEvent(new PointerEvent("pointerleave", { bubbles: true }));
  store.dispatch({ type: "clearSelection", kind: "element" });
}
