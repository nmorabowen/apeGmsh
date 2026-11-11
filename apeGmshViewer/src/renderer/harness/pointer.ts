// The capture and measure harness (VH1, #2865): the scripted target, clicks, pointer events and on-page markers.

import * as THREE from "three";
import { chainOf } from "../../state/selectors.ts";
import type { DeclPath } from "../../state/types.ts";
import { painted, type HarnessContext } from "./context.ts";

const BEAM_TYPES = new Set(["dispBeamColumn", "forceBeamColumn", "elasticBeamColumn"]);

/**
 * The element a scripted click targets: a beam-column if any; else an element
 * whose chain resolves with at least one link; else the middle drawn element.
 * A bounded sample (from the middle outwards) keeps this cheap on big models.
 * With `pickType`, every drawn element is a candidate (not only the sample),
 * and the first of that OpenSees type whose chain resolves wins; none found
 * raises, naming the type.
 */
export function demoTarget(ctx: HarnessContext, pickType: string | null = null): DeclPath | null {
  const { store } = ctx;
  const s = store.get();
  if (!s.mesh) return null;
  const drawn = s.mesh.elements;
  if (pickType !== null) {
    for (const p of drawn) {
      if (s.decls[p]?.type !== pickType) continue;
      const c = chainOf(s, p);
      if (c && c.problems.length === 0) return p;
    }
    throw new Error(`--pick=${pickType}: no drawn element of that type with a resolved chain`);
  }
  const step = Math.max(1, Math.floor(drawn.length / 400));
  const candidates: DeclPath[] = [];
  for (let i = Math.floor(drawn.length / 2); i < drawn.length; i += step) candidates.push(drawn[i]!);
  for (let i = 0; i < Math.floor(drawn.length / 2); i += step) candidates.push(drawn[i]!);
  const chains = candidates.map((p) => ({ p, c: chainOf(s, p)! }));
  const beam = chains.find(({ c }) => BEAM_TYPES.has(c.root.type) && c.problems.length === 0);
  const linked = chains.find(({ c }) => c.root.children.length > 0 && c.problems.length === 0);
  return (beam ?? linked)?.p ?? candidates[0] ?? null;
}

/** Click at the target's projected midpoint through the same pick path a mouse uses. */
export async function scriptedClick(ctx: HarnessContext, target: DeclPath) {
  const { store, viewport } = ctx;
  const pt = viewport.screenPointOf(target);
  if (!pt) throw new Error("scripted click: target is not drawn");
  const t = performance.now();
  const pick = viewport.pickAt(pt.x, pt.y);
  if (!pick) throw new Error("scripted click: pick returned nothing");
  const tPick = performance.now();
  store.dispatch({ type: "select", pick });
  const tState = performance.now();
  await painted();
  const end = performance.now();
  return {
    ms: end - t,
    split: { pickMs: tPick - t, stateAndDomMs: tState - tPick, paintMs: end - tState },
    pickedTarget: pick.decl === target,
    decl: pick.decl,
  };
}

/**
 * The click target of the miss still: of the drawn elements whose own screen
 * point picks them with the cuts off and is cut away with them on, the one
 * nearest the middle of the model on screen.
 */
export function clipMissTarget(ctx: HarnessContext): { decl: DeclPath; x: number; y: number } | null {
  const { store, viewport } = ctx;
  const s = store.get();
  const c = new THREE.Vector3(...s.mesh!.center).project(viewport.camera);
  const rect = viewport.renderer.domElement.getBoundingClientRect();
  const cx = rect.left + ((c.x + 1) / 2) * rect.width, cy = rect.top + ((1 - c.y) / 2) * rect.height;
  let best: { decl: DeclPath; x: number; y: number; d: number } | null = null;
  for (const decl of s.mesh!.elements) {
    const pt = viewport.screenPointOf(decl);
    if (!pt) continue;
    const d = Math.hypot(pt.x - cx, pt.y - cy);
    if (best && d >= best.d) continue;
    store.dispatch({ type: "setApplyCuts", on: false });
    const off = viewport.pickAt(pt.x, pt.y);
    store.dispatch({ type: "setApplyCuts", on: true });
    if (off?.decl !== decl) continue;
    if (viewport.pickAt(pt.x, pt.y) === null) best = { decl, ...pt, d };
  }
  return best && { decl: best.decl, x: best.x, y: best.y };
}

/** The client point of a world point. */
export function screenOf(ctx: HarnessContext, p: readonly number[]): { x: number; y: number } {
  const { viewport } = ctx;
  const v = new THREE.Vector3(p[0], p[1], p[2]).project(viewport.camera);
  const rect = viewport.renderer.domElement.getBoundingClientRect();
  return { x: rect.left + ((v.x + 1) / 2) * rect.width, y: rect.top + ((1 - v.y) / 2) * rect.height };
}

/** A real pointer event at a client point, sent where the browser would send it (the topmost element there). */
export function pointerAt(ctx: HarnessContext, type: string, at: { x: number; y: number }, target: Element = document.elementFromPoint(at.x, at.y) ?? ctx.viewport.renderer.domElement): Element {
  target.dispatchEvent(new PointerEvent(type, { clientX: at.x, clientY: at.y, button: 0, buttons: type === "pointerup" ? 0 : 1, pointerId: 1, bubbles: true, cancelable: true }));
  return target;
}

/** A ring and a label on the page at a client point (capture only). */
export function marker(at: { x: number; y: number }, text: string): () => void {
  const ring = document.createElement("div");
  ring.style.cssText =
    `position: fixed; left: ${at.x - 9}px; top: ${at.y - 9}px; width: 18px; height: 18px; border-radius: 50%;` +
    "border: 3px solid var(--accent); pointer-events: none; z-index: 10;";
  const label = document.createElement("div");
  label.style.cssText = `position: fixed; left: ${at.x + 16}px; top: ${at.y - 10}px; padding: 2px 6px; border-radius: 5px; font: 600 12px var(--font); color: var(--text); background: var(--panel); border: 1px solid var(--line); pointer-events: none; z-index: 10;`;
  label.textContent = text;
  document.body.append(ring, label);
  return () => {
    ring.remove();
    label.remove();
  };
}
