// Frame-level checks of `npm run profile`: is a recorded frame a flicker?
// Pure, so the rule is tested without a window.

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Raster {
  width: number;
  height: number;
  rgba: Uint8Array;
}

/** A region's picture, coarse: the mean colour of each cell of a grid, so one moved edge does not count as a new picture. */
export function fingerprint(p: Raster, r: Rect, cols = 32, rows = 20): Float32Array {
  const out = new Float32Array(cols * rows * 3);
  const count = new Uint32Array(cols * rows);
  const x0 = Math.max(0, Math.floor(r.x)), y0 = Math.max(0, Math.floor(r.y));
  const x1 = Math.min(p.width, Math.ceil(r.x + r.width)), y1 = Math.min(p.height, Math.ceil(r.y + r.height));
  for (let y = y0; y < y1; y++) {
    const row = Math.min(rows - 1, Math.floor(((y - y0) * rows) / Math.max(1, y1 - y0)));
    for (let x = x0; x < x1; x++) {
      const col = Math.min(cols - 1, Math.floor(((x - x0) * cols) / Math.max(1, x1 - x0)));
      const c = row * cols + col, i = (y * p.width + x) * 4;
      out[3 * c] = out[3 * c]! + p.rgba[i]!;
      out[3 * c + 1] = out[3 * c + 1]! + p.rgba[i + 1]!;
      out[3 * c + 2] = out[3 * c + 2]! + p.rgba[i + 2]!;
      count[c] = count[c]! + 1;
    }
  }
  for (let c = 0; c < count.length; c++) {
    const n = count[c]! || 1;
    for (let k = 0; k < 3; k++) out[3 * c + k] = out[3 * c + k]! / n / 255;
  }
  return out;
}

/** Mean absolute difference of two fingerprints, 0 to 1. */
export function distance(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) return 1;
  let d = 0;
  for (let i = 0; i < a.length; i++) d += Math.abs(a[i]! - b[i]!);
  return a.length ? d / a.length : 0;
}

/** A step of change below this is compression noise or a small move, not a visible jump. */
export const FLICKER_MIN_JUMP = 0.004;

/**
 * The frames that jump away and come back: frame i differs from both
 * neighbours, while the neighbours are much closer to each other than to it.
 * Steady motion (each frame between its neighbours) never qualifies, however
 * fast, and neither does a single change that stays.
 */
export function flickerFrames(prints: Float32Array[], minJump = FLICKER_MIN_JUMP): number[] {
  const out: number[] = [];
  for (let i = 1; i + 1 < prints.length; i++) {
    const before = distance(prints[i - 1]!, prints[i]!), after = distance(prints[i]!, prints[i + 1]!);
    if (before < minJump || after < minJump) continue;
    if (distance(prints[i - 1]!, prints[i + 1]!) < 0.5 * Math.min(before, after)) out.push(i);
  }
  return out;
}
