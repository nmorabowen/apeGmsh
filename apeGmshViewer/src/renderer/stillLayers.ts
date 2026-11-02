// The still-only layers of a `chrome: "model"` shot (ADR 0128 D9): the scale
// bar, the axes triad, the banner line and the caption, drawn by the render
// mode into the image, never by the interactive panels. The image is the
// viewport's WebGL canvas at the spec's `size`, on the theme's viewport
// background (the page's radial gradient, as style.css draws it) or on
// transparent pixels, with these layers painted over it on a 2-D canvas.
//
// Colours and the font come only from theme/tokens.ts (D8); the ramp from
// state/contour.ts, the same table the mesh is coloured from. The triad
// tells its axes apart by their letters, not by hue (protanopia-safe).

import { fmtValue, sampleCmap } from "../state/contour.ts";
import type { ContourCmap } from "../state/types.ts";
import { FONT_FAMILY, themeTokens, type ThemeMode } from "../theme/tokens.ts";

export interface ScaleBar {
  /** `<field> <component>` as the legend titles it */
  title: string;
  min: number | null;
  max: number | null;
  /** the legend's sentence for the range in use (`range: whole history (12 steps)`) */
  rangeLabel: string;
  cmap: ContourCmap;
}

export interface StillInput {
  source: HTMLCanvasElement;
  width: number;
  height: number;
  theme: ThemeMode;
  background: "theme" | "transparent";
  layers: { scale_bar: boolean; axes: boolean; banner: boolean; caption: boolean };
  scale: ScaleBar | null;
  /** each world axis's direction on screen (x right, y up), from the camera; null when no axis is drawn */
  axes: { x: [number, number]; y: [number, number]; z: [number, number] } | null;
  banner: readonly string[];
  caption: string | null;
}

const FONT = (px: number, weight = 400) => `${weight} ${px}px "${FONT_FAMILY}"`;
const PAD = 16;

/** Load the faces the layers draw with, so the first still never falls back to another font. */
export async function loadStillFonts(): Promise<void> {
  await Promise.all([document.fonts.load(FONT(14)), document.fonts.load(FONT(14, 600))]);
}

/** The radial gradient of style.css `html, body` (`ellipse at 50% 40%`, farthest-corner, bg1 to bg0). */
function paintBackground(ctx: CanvasRenderingContext2D, w: number, h: number, t: ReturnType<typeof themeTokens>): void {
  const cx = 0.5 * w, cy = 0.4 * h;
  // An ellipse at farthest-corner keeps the farthest-side aspect, scaled by sqrt 2 to reach the corner.
  const rx = Math.SQRT2 * Math.max(cx, w - cx), ry = Math.SQRT2 * Math.max(cy, h - cy);
  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(1, ry / rx);
  const g = ctx.createRadialGradient(0, 0, 0, 0, 0, rx);
  g.addColorStop(0, t.bg1);
  g.addColorStop(1, t.bg0);
  ctx.fillStyle = g;
  ctx.fillRect(-cx, -cy * (rx / ry), w, h * (rx / ry));
  ctx.restore();
}

function panel(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, t: ReturnType<typeof themeTokens>): void {
  ctx.fillStyle = t.panel;
  ctx.strokeStyle = t.line;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, 6);
  ctx.fill();
  ctx.stroke();
}

function drawScale(ctx: CanvasRenderingContext2D, s: ScaleBar, t: ReturnType<typeof themeTokens>): void {
  const rampW = 260, rampH = 14, w = rampW + 24, h = 86;
  const x = PAD, y = PAD;
  panel(ctx, x, y, w, h, t);
  ctx.fillStyle = t.text;
  ctx.textBaseline = "top";
  ctx.font = FONT(15, 600);
  ctx.fillText(s.title, x + 12, y + 8, rampW);
  const ry = y + 32;
  for (let i = 0; i < rampW; i++) {
    const [r, g, b] = sampleCmap(s.cmap, (i + 0.5) / rampW);
    // A colour of the ramp table itself (state/contour.ts), as the mesh draws it.
    ctx.fillStyle = `rgb(${Math.round(255 * r)} ${Math.round(255 * g)} ${Math.round(255 * b)})`;
    ctx.fillRect(x + 12 + i, ry, 1, rampH);
  }
  ctx.strokeStyle = t.line;
  ctx.strokeRect(x + 12 + 0.5, ry + 0.5, rampW - 1, rampH - 1);
  ctx.fillStyle = t.text;
  ctx.font = FONT(13);
  const lo = s.min === null ? "no data" : fmtValue(s.min);
  const hi = s.max === null ? "no data" : fmtValue(s.max);
  ctx.textAlign = "left";
  ctx.fillText(lo, x + 12, ry + rampH + 4);
  ctx.textAlign = "right";
  ctx.fillText(hi, x + 12 + rampW, ry + rampH + 4);
  ctx.textAlign = "left";
  ctx.fillStyle = t.muted;
  ctx.fillText(s.rangeLabel, x + 12, ry + rampH + 22, rampW);
}

function drawAxes(ctx: CanvasRenderingContext2D, a: NonNullable<StillInput["axes"]>, h: number, bottomInset: number, t: ReturnType<typeof themeTokens>): void {
  const L = 34;
  const cx = PAD + L + 10, cy = h - bottomInset - PAD - L - 10;
  ctx.strokeStyle = t.text;
  ctx.fillStyle = t.text;
  ctx.lineWidth = 2;
  ctx.font = FONT(13, 600);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (const [name, d] of [["X", a.x], ["Y", a.y], ["Z", a.z]] as const) {
    const ex = cx + L * d[0], ey = cy - L * d[1];
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(ex, ey);
    ctx.stroke();
    // An axis seen end-on has no direction on screen: its letter sits on the origin.
    const len = Math.hypot(d[0], d[1]);
    const k = len > 1e-3 ? (L + 10) / len : 0;
    ctx.fillText(name, cx + k * d[0], cy - k * d[1]);
  }
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
}

/** The lines under the model: the caption, then the banner as one line. */
function drawFooter(ctx: CanvasRenderingContext2D, lines: { text: string; muted: boolean }[], w: number, h: number, t: ReturnType<typeof themeTokens>): number {
  if (!lines.length) return 0;
  const lineH = 20;
  const boxH = lines.length * lineH + 12;
  panel(ctx, PAD, h - PAD - boxH, w - 2 * PAD, boxH, t);
  ctx.textBaseline = "top";
  ctx.font = FONT(14);
  lines.forEach((l, i) => {
    ctx.fillStyle = l.muted ? t.muted : t.text;
    ctx.fillText(l.text, PAD + 10, h - PAD - boxH + 6 + i * lineH, w - 2 * PAD - 20);
  });
  return boxH + PAD;
}

/** Paint a model-mode still: the background, the viewport's frame, then the layers that are on. */
export function composeStill(input: StillInput): HTMLCanvasElement {
  const { width: w, height: h } = input;
  const t = themeTokens(input.theme);
  const out = document.createElement("canvas");
  out.width = w;
  out.height = h;
  const ctx = out.getContext("2d");
  if (!ctx) throw new Error("render: no 2-D canvas context for the still");
  if (input.background === "theme") paintBackground(ctx, w, h, t);
  ctx.drawImage(input.source, 0, 0, w, h);
  const footer: { text: string; muted: boolean }[] = [];
  if (input.layers.caption && input.caption) footer.push({ text: input.caption, muted: false });
  if (input.layers.banner && input.banner.length) footer.push({ text: input.banner.join("  "), muted: true });
  const inset = drawFooter(ctx, footer, w, h, t);
  if (input.layers.scale_bar && input.scale) drawScale(ctx, input.scale, t);
  if (input.layers.axes && input.axes) drawAxes(ctx, input.axes, h, inset, t);
  return out;
}
