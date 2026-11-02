// The contour scale's place on screen: dragging it, keeping it in the window,
// and remembering its orientation and position between launches. This is the
// viewer's own preference, not model data or a view a spec states, so it lives
// on the element and in this profile's localStorage rather than in the store.
// An agent run has a profile of its own (main.ts), so its stills always draw
// the default place.

import { listen } from "./dom.ts";

const KEY = "apegmsh.viewer.contourScale";

export type ScaleOrient = "horizontal" | "vertical";

export interface SavedScale {
  orient: ScaleOrient;
  /** null: never moved, so the stylesheet's default place holds */
  at: { left: number; top: number } | null;
}

/** What was stored, or null when it is absent or not what this version writes. */
export function parseSavedScale(raw: string | null): SavedScale | null {
  if (raw === null) return null;
  let v: unknown;
  try {
    v = JSON.parse(raw);
  } catch {
    return null;
  }
  if (typeof v !== "object" || v === null) return null;
  const { orient, at } = v as { orient?: unknown; at?: unknown };
  if (orient !== "horizontal" && orient !== "vertical") return null;
  if (at === null) return { orient, at: null };
  const { left, top } = (at ?? {}) as { left?: unknown; top?: unknown };
  if (typeof left !== "number" || typeof top !== "number" || !Number.isFinite(left) || !Number.isFinite(top)) return null;
  return { orient, at: { left, top } };
}

export interface ScaleChrome {
  /** Pull a moved scale back inside the window (a resize or a change of orientation can push it out). */
  clamp(): void;
  /** Store the current orientation and place. */
  save(): void;
  dispose(): void;
}

export function mountScaleChrome(scale: HTMLElement): ScaleChrome {
  const place = (left: number, top: number) => {
    const r = scale.getBoundingClientRect();
    scale.style.left = `${Math.max(0, Math.min(left, window.innerWidth - r.width))}px`;
    scale.style.top = `${Math.max(0, Math.min(top, window.innerHeight - r.height))}px`;
  };
  const clamp = () => {
    if (!scale.hidden && scale.dataset["moved"]) place(scale.offsetLeft, scale.offsetTop);
  };
  const save = () => {
    const saved: SavedScale = {
      orient: scale.dataset["orient"] === "vertical" ? "vertical" : "horizontal",
      at: scale.dataset["moved"] ? { left: scale.offsetLeft, top: scale.offsetTop } : null,
    };
    try {
      localStorage.setItem(KEY, JSON.stringify(saved));
    } catch {
      // Blocked storage only costs the memory of the place; the scale still works.
    }
  };

  let stored: string | null = null;
  try {
    stored = localStorage.getItem(KEY);
  } catch {
    // As above: the default place holds.
  }
  const saved = parseSavedScale(stored);
  if (saved) {
    scale.dataset["orient"] = saved.orient;
    if (saved.at) {
      scale.dataset["moved"] = "true";
      scale.style.left = `${saved.at.left}px`;
      scale.style.top = `${saved.at.top}px`;
    }
  }

  let drag: { dx: number; dy: number } | null = null;
  const unlisten = [
    listen(scale, "pointerdown", (e: PointerEvent) => {
      if (e.target instanceof Element && e.target.closest("button, input, select")) return;
      const r = scale.getBoundingClientRect();
      drag = { dx: e.clientX - r.left, dy: e.clientY - r.top };
      scale.setPointerCapture(e.pointerId);
      scale.classList.add("dragging");
    }),
    listen(scale, "pointermove", (e: PointerEvent) => {
      if (!drag) return;
      scale.dataset["moved"] = "true";
      place(e.clientX - drag.dx, e.clientY - drag.dy);
    }),
    listen(scale, "pointerup", () => {
      if (!drag) return;
      drag = null;
      scale.classList.remove("dragging");
      save();
    }),
    listen(window as unknown as HTMLElement, "resize", clamp),
  ];
  return { clamp, save, dispose: () => { for (const u of unlisten) u(); } };
}
