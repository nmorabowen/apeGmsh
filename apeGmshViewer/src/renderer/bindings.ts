// The app's mouse and key bindings: the one table. Navigation reads it and
// nothing else decides what a button does. The convention is the CAD one
// (Plasticity, Rhino): left selects and never orbits, right orbits.
//
//   input                      action
//   left click                 select the element under the cursor
//   shift/ctrl + left click    toggle the element under the cursor in the selection
//   left drag                  nothing (never orbits)
//   right drag                 orbit (turntable, Z stays vertical, about the point under the cursor)
//   shift + right drag         pan
//   middle drag                pan
//   wheel                      zoom toward the point under the cursor
//   F                          frame the selection, else fit the whole model
//   E                          toggle mesh outline edges
//   Escape                     clear the selection

export type Button = "left" | "middle" | "right";
export type ClickAction = "select";
export type DragAction = "orbit" | "pan";
export type KeyAction = "fit" | "clearSelection" | "toggleEdges";

export interface PointerBinding {
  readonly button: Button;
  readonly shift: boolean;
  /** What a press and release within CLICK_SLOP_PX does; null: nothing. */
  readonly click: ClickAction | null;
  /** What a drag does; null: nothing. */
  readonly drag: DragAction | null;
}

/** Every (button, shift) pair, exactly once. */
export const POINTER_BINDINGS: readonly PointerBinding[] = [
  { button: "left", shift: false, click: "select", drag: null },
  { button: "left", shift: true, click: "select", drag: null },
  { button: "middle", shift: false, click: null, drag: "pan" },
  { button: "middle", shift: true, click: null, drag: "pan" },
  { button: "right", shift: false, click: null, drag: "orbit" },
  { button: "right", shift: true, click: null, drag: "pan" },
];

export const WHEEL_ACTION = "zoom-to-cursor" as const;

export const KEY_BINDINGS: readonly { readonly key: string; readonly action: KeyAction }[] = [
  { key: "f", action: "fit" },
  { key: "escape", action: "clearSelection" },
  { key: "e", action: "toggleEdges" },
];

/** A press that moves farther than this (px) is a drag, not a click. */
export const CLICK_SLOP_PX = 4;

export { isSelectAdd } from "../ui/dom.ts";

/**
 * The bound button for a DOM `MouseEvent.button`, or null for a button the
 * app does not bind (4 and 5, back and forward, belong to the system).
 */
export function buttonOf(domButton: number): Button | null {
  switch (domButton) {
    case 0:
      return "left";
    case 1:
      return "middle";
    case 2:
      return "right";
    default:
      return null;
  }
}

export function bindingFor(button: Button, shift: boolean): PointerBinding {
  const b = POINTER_BINDINGS.find((r) => r.button === button && r.shift === shift);
  if (!b) throw new Error(`bindings: no row for ${shift ? "shift + " : ""}${button}`);
  return b;
}

/** The key action for a KeyboardEvent.key with no Ctrl/Alt/Meta held, or null. */
export function keyAction(key: string): KeyAction | null {
  return KEY_BINDINGS.find((k) => k.key === key.toLowerCase())?.action ?? null;
}
