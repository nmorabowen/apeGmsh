// The selection stills of the capture harness (ADR 0138 SEL-1b), for every
// model with a mesh:
//
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
