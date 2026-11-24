// The shared chart (ADR 0137 D11, H2): one small x-y plot on a canvas, for the panels and, from
// H2b, the tools. It replaces ui/xyPlot.ts (the Hinges card's moment-rotation plot and the
// material tester's curves) with the same call: the curve, the zero lines, dashed reference lines,
// a marked point, and muted overlaid curves each with its own dash.
//
// Each curve is reduced to the canvas's pixel columns before it is drawn (`reducePath`), so a
// curve of a million points costs what one of a few hundred does. The reduction follows the path:
// consecutive points in one pixel column collapse to the first, the lowest, the highest and the
// last of them. For a curve whose x only grows (a time series) that is the per-column minimum and
// maximum; for one that turns back (a hysteresis loop) every turn is kept, where a per-column
// minimum and maximum would fill the loop in.
//
// Pure DOM: no store, no app imports (the ui-allow-list rule). The colours are the theme's CSS
// custom properties, read from the canvas when it draws.

const SVG_NS = "http://www.w3.org/2000/svg";
/** The plot's logical size: its aspect ratio, and the margins the labels sit in. */
const W = 260, H = 150, L = 50, R = 10, T = 10, B = 24;

/** A plot label: three significant figures, exponent form outside 1e-2 to 1e4. */
export const fmt = (v: number): string => (v === 0 ? "0" : Math.abs(v) >= 1e4 || Math.abs(v) < 1e-2 ? v.toExponential(2) : Number(v.toPrecision(3)).toString());

export interface XyPlot {
  xs: ArrayLike<number>;
  ys: ArrayLike<number>;
  xName: string;
  yName: string;
  /** the accessible name */
  title: string;
  /** horizontal reference lines (a capacity), dashed */
  caps?: readonly number[];
  /** the index of a point to mark (the step on screen); none when absent */
  mark?: number;
  /** curves drawn under the main one, muted, each with its own dash (`xySwatch` draws the same dash for a legend) */
  others?: readonly { xs: ArrayLike<number>; ys: ArrayLike<number>; dash: number }[];
  /**
   * fit the chart inside its parent's box, keeping its aspect (the parent must have a height of its
   * own, e.g. a flex item in a tool tile); otherwise it is as wide as its container and as tall as
   * its aspect makes it
   */
  fit?: boolean;
}

/** How many dashes the overlaid curves cycle through. */
export const XY_DASHES = 5;
/** Each overlaid curve's dash, in logical pixels (index `dash % XY_DASHES`); [] is solid. */
export const DASHES: readonly (readonly number[])[] = [[], [5, 3], [1.5, 2.5], [8, 3, 2, 3], [3, 5]];
const dashOf = (dash: number): readonly number[] => DASHES[((dash % XY_DASHES) + XY_DASHES) % XY_DASHES]!;

/** The finite range of `lists`, always spanning zero; a flat range is widened by one each way. */
export function spanZero(...lists: ArrayLike<number>[]): [number, number] {
  let lo = 0, hi = 0;
  for (const list of lists) {
    for (let i = 0; i < list.length; i++) {
      const v = list[i]!;
      if (!Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  return lo === hi ? [lo - 1, hi + 1] : [lo, hi];
}

/**
 * A path in canvas pixels, reduced along its length to at most four points per run of
 * consecutive points in one pixel column (`Math.floor(x)`): the run's first, lowest, highest and
 * last, in the order they came. Non-finite points break the path; each piece is returned apart.
 */
export function reducePath(px: ArrayLike<number>, py: ArrayLike<number>): number[][] {
  const pieces: number[][] = [];
  let out: number[] = [];
  let col = NaN;
  // The current run: its first, lowest, highest and last points, by index.
  let first = -1, lo = -1, hi = -1, last = -1;
  const flush = () => {
    if (first < 0) return;
    const keep = [first, lo, hi, last].filter((v, i, a) => a.indexOf(v) === i).sort((a, b) => a - b);
    for (const k of keep) out.push(px[k]!, py[k]!);
    first = lo = hi = last = -1;
  };
  for (let i = 0; i < px.length; i++) {
    const x = px[i]!, y = py[i]!;
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      flush();
      if (out.length) pieces.push(out);
      out = [];
      col = NaN;
      continue;
    }
    const c = Math.floor(x);
    if (c !== col) {
      flush();
      col = c;
      first = lo = hi = last = i;
      continue;
    }
    last = i;
    if (y < py[lo]!) lo = i;
    if (y > py[hi]!) hi = i;
  }
  flush();
  if (out.length) pieces.push(out);
  return pieces;
}

/** What `drawChart` needs of a canvas 2-D context; a test hands it a recorder. */
export interface ChartContext {
  setTransform(a: number, b: number, c: number, d: number, e: number, f: number): void;
  clearRect(x: number, y: number, w: number, h: number): void;
  beginPath(): void;
  moveTo(x: number, y: number): void;
  lineTo(x: number, y: number): void;
  stroke(): void;
  fill(): void;
  arc(x: number, y: number, r: number, a0: number, a1: number): void;
  fillText(text: string, x: number, y: number): void;
  setLineDash(segments: number[]): void;
  strokeStyle: unknown;
  fillStyle: unknown;
  lineWidth: number;
  lineJoin: string;
  globalAlpha: number;
  font: string;
  textAlign: string;
}

/** The theme colours the chart draws with (the stylesheet's custom properties). */
export interface ChartColours {
  accent: string;
  muted: string;
  line: string;
  faint: string;
  panel: string;
  font: string;
}

/**
 * Draw `p` on `ctx`, `width` CSS pixels wide (the height follows the aspect ratio) at device
 * pixel ratio `dpr`. Each curve is mapped to device pixels and reduced (`reducePath`) first.
 * Returns how many points were drawn, curve by curve (the main curve last).
 */
export function drawChart(ctx: ChartContext, p: XyPlot, width: number, dpr: number, c: ChartColours): number[] {
  const s = width / W;
  const caps = p.caps ?? [];
  const others = p.others ?? [];
  const [x0, x1] = spanZero(p.xs, ...others.map((o) => o.xs));
  const [y0, y1] = spanZero(p.ys, ...others.map((o) => o.ys), caps);
  const px = (v: number) => L + ((v - x0) / (x1 - x0)) * (W - L - R);
  const py = (v: number) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, W * s * dpr, H * s * dpr);
  // Everything below is drawn in logical units; the curves are reduced in device pixels.
  const k = s * dpr;
  ctx.setTransform(k, 0, 0, k, 0, 0);
  const line = (xa: number, ya: number, xb: number, yb: number) => {
    ctx.beginPath();
    ctx.moveTo(xa, ya);
    ctx.lineTo(xb, yb);
    ctx.stroke();
  };
  ctx.lineWidth = 1;
  ctx.globalAlpha = 1;
  ctx.setLineDash([]);
  ctx.strokeStyle = c.line;
  line(L, py(0), W - R, py(0));
  line(px(0), T, px(0), H - B);
  ctx.strokeStyle = c.faint;
  ctx.setLineDash([3, 3]);
  for (const cap of caps) line(L, py(cap), W - R, py(cap));
  const drawn: number[] = [];
  const curve = (xs: ArrayLike<number>, ys: ArrayLike<number>) => {
    const n = Math.min(xs.length, ys.length);
    const dx = new Float64Array(n), dy = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      dx[i] = px(xs[i]!) * k;
      dy[i] = py(ys[i]!) * k;
    }
    let count = 0;
    for (const piece of reducePath(dx, dy)) {
      ctx.beginPath();
      ctx.moveTo(piece[0]! / k, piece[1]! / k);
      for (let i = 2; i < piece.length; i += 2) ctx.lineTo(piece[i]! / k, piece[i + 1]! / k);
      ctx.stroke();
      count += piece.length / 2;
    }
    drawn.push(count);
  };
  ctx.lineJoin = "round";
  ctx.strokeStyle = c.muted;
  ctx.lineWidth = 1.25;
  for (const o of others) {
    ctx.setLineDash([...dashOf(o.dash)]);
    ctx.globalAlpha = o.dash % XY_DASHES === 0 ? 0.8 : 1;
    curve(o.xs, o.ys);
  }
  ctx.globalAlpha = 1;
  ctx.setLineDash([]);
  ctx.strokeStyle = c.accent;
  ctx.lineWidth = 1.5;
  curve(p.xs, p.ys);
  const m = p.mark;
  if (m !== undefined && Number.isFinite(p.xs[m]) && Number.isFinite(p.ys[m])) {
    ctx.beginPath();
    ctx.arc(px(p.xs[m]!), py(p.ys[m]!), 3.5, 0, 2 * Math.PI);
    ctx.fillStyle = c.accent;
    ctx.fill();
    ctx.strokeStyle = c.panel;
    ctx.lineWidth = 1;
    ctx.stroke();
  }
  ctx.fillStyle = c.muted;
  ctx.font = `10px ${c.font}`;
  const text = (t: string, x: number, y: number, align: string) => {
    ctx.textAlign = align;
    ctx.fillText(t, x, y);
  };
  text(fmt(y1), L - 4, T + 8, "end");
  text(fmt(y0), L - 4, H - B, "end");
  text(fmt(x0), L, H - 8, "start");
  text(fmt(x1), W - R, H - 8, "end");
  text(p.xName, (L + W - R) / 2, H - 8, "center");
  text(p.yName, L + 4, T + 8, "start");
  return drawn;
}

/** The theme colours as the canvas sees them now. */
function coloursOf(el: Element): ChartColours {
  const cs = getComputedStyle(el);
  const v = (name: string, fallback: string) => cs.getPropertyValue(name).trim() || fallback;
  return { accent: v("--accent", "currentColor"), muted: v("--muted", "currentColor"), line: v("--line", "currentColor"), faint: v("--faint", "currentColor"), panel: v("--panel", "transparent"), font: v("--font", "sans-serif") };
}

/**
 * The plot, as a canvas the width of its container. It draws once it is laid out and again on
 * every resize; it reads its colours from the theme each time.
 */
export function xyPlot(p: XyPlot): HTMLElement {
  const canvas = document.createElement("canvas");
  canvas.className = "xy-plot";
  canvas.width = W;
  canvas.height = H;
  canvas.setAttribute?.("role", "img");
  canvas.setAttribute?.("aria-label", p.title);
  // What the canvas draws, for the tests and the harness (a canvas has no child nodes to count).
  canvas.dataset["curves"] = String(1 + (p.others?.length ?? 0));
  canvas.dataset["x"] = p.xName;
  canvas.dataset["y"] = p.yName;
  const draw = () => {
    // A panel rebuilds its plots on a state change: a canvas taken out of the page stops observing.
    if (!canvas.isConnected && observed) {
      ro.disconnect();
      return;
    }
    observed ||= canvas.isConnected;
    const box = canvas.parentElement;
    if (p.fit && box) {
      // The parent sets the room; the canvas is sized to it here, never from its own size (no feedback).
      if (!boxObserved) {
        ro.observe(box);
        boxObserved = true;
      }
      const w = Math.max(0, Math.floor(Math.min(box.clientWidth, (box.clientHeight * W) / H)));
      canvas.style.width = `${w}px`;
      canvas.style.height = `${(w * H) / W}px`;
    }
    const width = canvas.clientWidth;
    if (width <= 0) return;
    const dpr = window.devicePixelRatio || 1;
    const bw = Math.round(width * dpr), bh = Math.round(((width * H) / W) * dpr);
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw;
      canvas.height = bh;
    }
    const ctx = canvas.getContext("2d");
    if (ctx) drawChart(ctx as unknown as ChartContext, p, width, dpr, coloursOf(canvas));
  };
  let observed = false;
  let boxObserved = false;
  // No ResizeObserver (a test's fake document): the canvas is built and never drawn.
  if (typeof ResizeObserver === "undefined") return canvas;
  const ro = new ResizeObserver(draw);
  ro.observe(canvas);
  return canvas;
}

/** A short line in the style of the main curve (`dash` null) or of an overlaid one, for a legend. */
export function xySwatch(dash: number | null): HTMLElement {
  const svg = (tag: string, attrs: Record<string, string | number>) => {
    const doc = document as Document & { createElementNS?: (ns: string, q: string) => Element };
    const node = (typeof doc.createElementNS === "function" ? doc.createElementNS(SVG_NS, tag) : doc.createElement(tag)) as HTMLElement;
    for (const [k, v] of Object.entries(attrs)) node.setAttribute?.(k, String(v));
    return node;
  };
  const root = svg("svg", { class: "xy-swatch", viewBox: "0 0 24 8", width: 24, height: 8, "aria-hidden": "true" });
  const cls = dash === null ? "xy-curve" : `xy-other xy-dash-${((dash % XY_DASHES) + XY_DASHES) % XY_DASHES}`;
  root.append(svg("line", { class: cls, x1: 1, y1: 4, x2: 23, y2: 4 }));
  return root;
}
