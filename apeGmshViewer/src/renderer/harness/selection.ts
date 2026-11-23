// The selection stills of the capture harness (ADR 0138 SEL-1b, SEL-2), for
// every model with a mesh:
//
// - `<out>.selection-click.png` (models with line elements): a click at the
//   middle of the line element drawn nearest the centre, which it must take.
// - `<out>.selection-hover.png`: the pointer over an element, through the
//   canvas's own pointermove listener, so the element is marked and named.
// - `<out>.selection-box.png`: a box drawn left to right over the middle of the
//   canvas (a wider one where that takes nothing: `captureBox`), mid-drag, with
//   the count of elements fully inside it.
// - `<out>.selection-boxed.png`: the same box released: those elements selected.
// - `<out>.selection-undone.png`: that selection cleared, then brought back
//   by undo (ADR 0138 D6).
// - `<out>.selection-through-box.png`, `<out>.selection-through-boxed.png`: the
//   same box through the model (ADR 0138 D7), which takes what is hidden too.
// - `<out>.selection-lasso.png`, `<out>.selection-lassoed.png`: a lasso drawn
//   clockwise (fully inside) around the box in Lasso mode, mid-drag and
//   released, which must take every element the box took (ADR 0138 D7).
// - `<out>.selection-sets.png`, `<out>.selection-set-selected.png`: the bar's
//   list of saved sets, and a set selected from it (ADR 0138 D12).
// - `<out>.selection-nodes-box.png`: Nodes active, a box drawn right to left,
//   mid-drag, with the node markers on.
//
// The box goes through the viewport's own `boxAt` and `boxEnd`, the calls a
// drag makes (renderer/navigation.ts); a scripted pointer cannot take the
// pointer capture a real press takes.

import { selectedElementsOf } from "../../state/selectors.ts";
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
  // The box must take something, or the undo check below has nothing to bring back: a clear of an empty selection is no
  // undo step (ADR 0138 D6), so the undo would put back the click's selection instead (the beam-only hinge fixtures, #3108).
  const { a, b } = captureBox(canvas.getBoundingClientRect(), (a, b) => {
    const n = viewport.boxAt(a.x, a.y, b.x, b.y, "replace")?.length ?? 0;
    viewport.boxCancel();
    return n;
  });
  viewport.boxAt(a.x, a.y, b.x, b.y, "replace");
  await shot("selection-box");
  viewport.boxEnd(a.x, a.y, b.x, b.y, "replace");
  await shot("selection-boxed");
  // ADR 0138 D6: a click on empty space clears the box's selection, and Ctrl+Z brings it back.
  const boxed = store.get().selection.decls;
  store.dispatch({ type: "clearSelection" });
  store.dispatch({ type: "undoSelection" });
  if (store.get().selection.decls !== boxed) throw new Error("capture selection-undone: undo did not bring back the boxed selection");
  await shot("selection-undone");
  // ADR 0138 D7: the same box through the model takes what is hidden too (on the slab, beams under it); undo puts the visible box back.
  store.dispatch({ type: "setSelectThrough", on: true });
  viewport.boxAt(a.x, a.y, b.x, b.y, "replace");
  await shot("selection-through-box");
  viewport.boxEnd(a.x, a.y, b.x, b.y, "replace");
  await shot("selection-through-boxed");
  if (selectedElementsOf(store.get()).length < boxed.length) throw new Error("capture selection-through-boxed: the box through the model took fewer elements than the visible box");
  store.dispatch({ type: "setSelectThrough", on: false });
  store.dispatch({ type: "undoSelection" });
  // ADR 0138 D7: a lasso drawn clockwise around the box, mid-drag and released; undo puts the box back. It holds the box, so it
  // takes every element the box took (a fixed loop held none of a coarse wall's elements whole, and took nothing).
  store.dispatch({ type: "setSelectLasso", on: true });
  const boxedElements = selectedElementsOf(store.get());
  const loop = lassoAround(a, b);
  for (const p of loop.slice(1, 40)) viewport.boxAt(loop[0]!.x, loop[0]!.y, p.x, p.y, "replace");
  await shot("selection-lasso");
  for (const p of loop.slice(40)) viewport.boxAt(loop[0]!.x, loop[0]!.y, p.x, p.y, "replace");
  viewport.boxEnd(loop[0]!.x, loop[0]!.y, loop.at(-1)!.x, loop.at(-1)!.y, "replace");
  const lassoed = new Set(selectedElementsOf(store.get()));
  const missed = boxedElements.filter((d) => !lassoed.has(d));
  if (missed.length > 0) throw new Error(`capture selection-lassoed: a lasso around the box missed ${missed.length} of the ${boxedElements.length} elements the box took (${missed[0]})`);
  await shot("selection-lassoed");
  store.dispatch({ type: "setSelectLasso", on: false });
  store.dispatch({ type: "undoSelection" });
  store.dispatch({ type: "setWorkspaceKind", kind: "nodes" });
  store.dispatch({ type: "setNodes", on: true });
  await settle();
  viewport.boxAt(b.x, b.y, a.x, a.y, "replace");
  await shot("selection-nodes-box");
  viewport.boxEnd(b.x, b.y, a.x, a.y, "replace");
  await shot("selection-nodes-boxed");
  await captureSets(ctx, shot);
  store.dispatch({ type: "clearSelection", kind: "node" });
  store.dispatch({ type: "clearSelection", kind: "element" });
  store.dispatch({ type: "setNodes", on: false });
  store.dispatch({ type: "setWorkspaceKind", kind: "elements" });
  await settle();
}

/**
 * The capture's box, from `a` to `b` (screen points in `r`): the middle of the
 * canvas, from 30 % to 60 % across and 30 % to 70 % down, which the approved
 * stills of the solid models show; where that takes no element whole (a few
 * long beams through the middle), the canvas inset by 3 %, which holds the
 * framed model (the 2-D hinge fixture's beams reach 7 % of the width).
 * `takes` counts the elements a box takes. It throws where neither takes one.
 */
export function captureBox(
  r: { left: number; top: number; width: number; height: number },
  takes: (a: { x: number; y: number }, b: { x: number; y: number }) => number,
): { a: { x: number; y: number }; b: { x: number; y: number } } {
  const at = (fx: number, fy: number) => ({ x: r.left + r.width * fx, y: r.top + r.height * fy });
  for (const [a, b] of [[at(0.3, 0.3), at(0.6, 0.7)], [at(0.03, 0.03), at(0.97, 0.97)]] as const) if (takes(a, b) > 0) return { a, b };
  throw new Error("capture selection-box: no capture box takes an element whole (30-60 % x 30-70 %, nor 3-97 %)");
}

/**
 * The capture's lasso (ADR 0138 D7): a wavy loop (five lobes, so it is no
 * box), drawn clockwise on screen around the box from `a` to `b`, closed back
 * to its first point. Its semi-axes are √2 times the box's half-sides, which
 * reach the corners. The 1.08 factor covers the wobble's low point (0.94) and
 * the sag of its 48 chords (cos(π/48)). So it holds the whole box, and takes
 * every element a box drawn left to right takes.
 */
export function lassoAround(a: { x: number; y: number }, b: { x: number; y: number }): { x: number; y: number }[] {
  const cx = (a.x + b.x) / 2, cy = (a.y + b.y) / 2;
  const rx = (Math.SQRT2 * 1.08 * Math.abs(b.x - a.x)) / 2, ry = (Math.SQRT2 * 1.08 * Math.abs(b.y - a.y)) / 2;
  const loop: { x: number; y: number }[] = [];
  for (let i = 0; i <= 48; i++) {
    // From the left (t = π), so the count, which sits at the press, stays clear of the right rail.
    const t = Math.PI + (2 * Math.PI * i) / 48, rad = 1 + 0.06 * Math.sin(5 * t);
    loop.push({ x: cx + rx * rad * Math.cos(t), y: cy + ry * rad * Math.sin(t) });
  }
  return loop;
}

/**
 * `<out>.selection-sets.png` and `<out>.selection-set-selected.png` (ADR 0138
 * D12), after the node box: the boxed elements and nodes as two saved sets,
 * the bar's list of element sets open, then the element set selected from it.
 * The sets are a workspace read made in memory (the model's own workspace
 * file is never written); the read on screen before is put back after.
 */
async function captureSets(ctx: HarnessContext, shot: (name: string) => Promise<void>): Promise<void> {
  const { store } = ctx;
  const s = store.get();
  const tags = [...new Set(selectedElementsOf(s).flatMap((d) => (s.decls[d]?.element?.metas ?? []).map((m) => m.tag)))];
  if (tags.length === 0) return;
  const before = s.workspace.read;
  const all = (n: number) => ({ verdict: "current" as const, matched: n, total: n, members: Array<boolean>(n).fill(true), checked: false, maxDistance: null });
  const sets = [
    { row: 0, name: "Boxed elements", kind: "elements" as const, ids: tags, verdict: all(tags.length) },
    { row: 1, name: "Boxed nodes", kind: "nodes" as const, ids: [...s.selection.nodes], verdict: all(s.selection.nodes.length) },
  ];
  store.dispatch({ type: "workspaceRead", read: { status: "ready", file: "capture.workspace.h5", token: "capture", version: "1", revision: 1, writer: null, saveId: null, readOnly: null, sets, plots: [], views: [], refused: [], broken: [] } });
  store.dispatch({ type: "clearSelection", kind: "element" });
  store.dispatch({ type: "setWorkspaceKind", kind: "elements" });
  // The node box's markers and selected nodes would bury the element highlight; the node set holds them now.
  store.dispatch({ type: "setNodes", on: false });
  store.dispatch({ type: "clearSelection", kind: "node" });
  const toggle = () => {
    const b = document.querySelector<HTMLButtonElement>("#selection-bar button[aria-expanded]");
    if (!b) throw new Error("capture selection-sets: the selection bar has no Sets button");
    b.click();
  };
  toggle();
  await shot("selection-sets");
  const select = [...document.querySelectorAll<HTMLButtonElement>(".selection-set button")].find((b) => b.textContent === "Select");
  if (!select) throw new Error("capture selection-sets: the list has no Select button");
  select.click();
  if (selectedElementsOf(store.get()).length === 0) throw new Error("capture selection-set-selected: selecting the set selected nothing");
  await shot("selection-set-selected");
  toggle();
  store.dispatch({ type: "workspaceRead", read: before });
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
