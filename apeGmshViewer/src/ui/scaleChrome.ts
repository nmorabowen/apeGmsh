// The colour scales in the viewport (contour, line diagram): what each draws,
// dragging it, keeping it in the window, and remembering its orientation and
// position between launches, each under its own key. This is the
// viewer's own preference, not model data or a view a spec states, so it lives
// on the element and in this profile's localStorage rather than in the store.
// An agent run has a profile of its own (main.ts), so its stills always draw
// the default place.

import { el, listen } from "./dom.ts";

export type ScaleOrient = "horizontal" | "vertical";

export interface ScaleContent {
  title: string;
  /** where the unit came from, as the title's tooltip */
  tooltip: string;
  /** the colormap's CSS gradient, running `to right` */
  ramp: string;
  ticks: readonly string[];
  range: string;
}

/** Draw a scale's title, orientation button, bar, values and range line. The caller appends its range controls. */
export function fillScale(scale: HTMLElement, c: ScaleContent, unlisten: (() => void)[], onOrient: () => void): void {
  const vertical = scale.dataset["orient"] === "vertical";
  const bar = el("div", "vs-ramp");
  bar.style.backgroundImage = vertical ? c.ramp.replace("to right", "to top") : c.ramp;
  const ticks = el("div", "vs-ticks");
  for (const t of c.ticks) ticks.append(el("span", undefined, t));
  const body = el("div", "vs-body");
  body.append(bar, ticks);
  const orient = el("button", "vs-orient", vertical ? "horizontal" : "vertical");
  orient.type = "button";
  orient.title = vertical ? "lay the scale out horizontally" : "lay the scale out vertically";
  unlisten.push(
    listen(orient, "click", () => {
      scale.dataset["orient"] = vertical ? "horizontal" : "vertical";
      onOrient();
    }),
  );
  const title = el("span", "vs-title", c.title);
  title.title = c.tooltip;
  const head = el("div", "vs-head");
  head.append(title, orient);
  scale.replaceChildren(head, body, el("div", "vs-range", c.range));
}

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

export function mountScaleChrome(scale: HTMLElement, key: string): ScaleChrome {
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
      localStorage.setItem(key, JSON.stringify(saved));
    } catch {
      // Blocked storage only costs the memory of the place; the scale still works.
    }
  };

  let stored: string | null = null;
  try {
    stored = localStorage.getItem(key);
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
