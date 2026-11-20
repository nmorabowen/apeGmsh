// Continuous crack lines from a crack direction field in a plane (CRACK_PLOTS_SPEC.md §3.3): evenly spaced
// streamlines (Jobard & Lefer 1997, "Creating evenly-spaced streamlines of arbitrary density") traced through the
// Gauss points' crack traces, as the maintainer's reference `trace_cracks` does. Pure 2-D: the caller projects a
// panel's points into its plane and lifts the lines back (state/crackLines.ts).
//
// Rules this file keeps (the reference's):
// - a crack direction is an axis, not a vector: each neighbour's T_j is flipped to agree with the travel direction
//   before it is summed;
// - the field at q is the opening-weighted mean of the cracked neighbours' axes within rmax, each weighted
//   W_j / (|P_j - q| + soft), so wide cracks steer and a point right on the path does not dominate;
// - a line stops where the nearest Gauss point is farther than rmax or uncracked (W < wMin): holes and uncracked
//   zones need no special case; where it turns sharper than maxTurn in one step; and within dsep / 2 of another line;
// - seeds are every cracked point, widest first, skipped within dsep of a line already drawn; each is traced both
//   ways and kept when at least minLen long.
// The spacing is a drawing choice, not a predicted crack spacing; the direction and the cracked extent are the model's.

export interface TraceParams {
  /** the minimum spacing between lines */
  dsep: number;
  /** the integration step */
  step: number;
  /** the neighbourhood of the field and of the "inside" test */
  rmax: number;
  /** the largest turn in one step, radians */
  maxTurn: number;
  /** lines shorter than this are discarded */
  minLen: number;
  /** a point opened less than this is uncracked */
  wMin: number;
  /** the distance softening of the weights */
  soft: number;
  /** the most steps one way from a seed (a guard: a closed loop never ends otherwise) */
  maxSteps?: number;
}

/** The reference's parameters (0.40 / 0.05 / 0.60 / 0.25 m on ~1 m elements) in units of the Gauss pitch `h`. */
export function traceParamsFor(h: number, wMin: number): TraceParams {
  return { dsep: 0.8 * h, step: 0.1 * h, rmax: 1.2 * h, maxTurn: (40 * Math.PI) / 180, minLen: 0.5 * h, wMin, soft: 0.1 * h };
}

/** One traced line: its points (x, y, x, y, ...) and the opening at each (the nearest Gauss point's). */
export interface Polyline {
  xy: number[];
  w: number[];
}

/**
 * A fixed set of points in a uniform grid, cells stored as flat arrays (CSR): the Gauss points the field reads at
 * every step, so a query allocates nothing. A disc of radius <= the cell size around a point lies in its 3 x 3 cells.
 */
class PointGrid {
  private readonly x0: number;
  private readonly y0: number;
  private readonly nx: number;
  private readonly ny: number;
  private readonly start: Int32Array;
  private readonly ids: Int32Array;
  readonly size: number;
  constructor(P: ArrayLike<number>, n: number, size: number) {
    this.size = size;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let i = 0; i < n; i++) {
      x0 = Math.min(x0, P[2 * i]!);
      y0 = Math.min(y0, P[2 * i + 1]!);
      x1 = Math.max(x1, P[2 * i]!);
      y1 = Math.max(y1, P[2 * i + 1]!);
    }
    if (n === 0) [x0, y0, x1, y1] = [0, 0, 0, 0];
    this.x0 = x0;
    this.y0 = y0;
    this.nx = Math.floor((x1 - x0) / size) + 1;
    this.ny = Math.floor((y1 - y0) / size) + 1;
    const count = new Int32Array(this.nx * this.ny + 1);
    const cell = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      cell[i] = Math.floor((P[2 * i]! - x0) / size) * this.ny + Math.floor((P[2 * i + 1]! - y0) / size);
      count[cell[i]! + 1]!++;
    }
    for (let c = 0; c < this.nx * this.ny; c++) count[c + 1]! += count[c]!;
    this.start = count;
    this.ids = new Int32Array(n);
    const fill = count.slice(0, this.nx * this.ny);
    for (let i = 0; i < n; i++) this.ids[fill[cell[i]!]!++] = i;
  }
  /** Call `f` with every point in the 3 x 3 cells around (x, y). */
  each(x: number, y: number, f: (i: number) => void): void {
    const ix = Math.floor((x - this.x0) / this.size), iy = Math.floor((y - this.y0) / this.size);
    for (let a = Math.max(0, ix - 1); a <= Math.min(this.nx - 1, ix + 1); a++) {
      for (let b = Math.max(0, iy - 1); b <= Math.min(this.ny - 1, iy + 1); b++) {
        const c = a * this.ny + b;
        for (let k = this.start[c]!; k < this.start[c + 1]!; k++) f(this.ids[k]!);
      }
    }
  }
}

/** A uniform grid over points, for neighbourhood queries (a disc of radius <= the cell size). */
export class Grid {
  private readonly cells = new Map<number, number[]>();
  readonly size: number;
  constructor(size: number) {
    this.size = size;
  }
  private key(ix: number, iy: number): number {
    return ix * 73856093 + iy * 19349663;
  }
  add(x: number, y: number, id: number): void {
    const k = this.key(Math.floor(x / this.size), Math.floor(y / this.size));
    const c = this.cells.get(k);
    if (c) c.push(id);
    else this.cells.set(k, [id]);
  }
  /** Every id in the cells that a disc of radius `r` (<= size) around (x, y) touches. */
  near(x: number, y: number, out: number[]): number[] {
    out.length = 0;
    const ix = Math.floor(x / this.size), iy = Math.floor(y / this.size);
    for (let a = ix - 1; a <= ix + 1; a++) {
      for (let b = iy - 1; b <= iy + 1; b++) {
        const c = this.cells.get(this.key(a, b));
        if (c) for (const id of c) out.push(id);
      }
    }
    return out;
  }
}

/**
 * Trace the crack lines of one panel. `P` (2 per point) the Gauss points in the panel's plane, `T` (2 per point) each
 * point's crack trace there (unit, sign-free), `W` each point's opening.
 */
export function traceCracks(P: ArrayLike<number>, T: ArrayLike<number>, W: ArrayLike<number>, p: TraceParams): Polyline[] {
  const n = W.length;
  const pts = new PointGrid(P, n, p.rmax);
  const occ = new Grid(Math.max(p.dsep, 1e-12));
  const occXY: number[] = [];
  const buf2: number[] = [];
  const cosTurn = Math.cos(p.maxTurn);
  const r2 = p.rmax * p.rmax;
  const maxSteps = p.maxSteps ?? Math.ceil((1e4 * p.dsep) / p.step);

  // One pass over the neighbours: the nearest point, and the opening-weighted sum of the cracked ones' axes.
  let i0 = -1, best = Infinity, sx = 0, sy = 0, qx = 0, qy = 0, ppx = 0, ppy = 0;
  const visit = (j: number) => {
    const ex = P[2 * j]! - qx, ey = P[2 * j + 1]! - qy;
    const d2 = ex * ex + ey * ey;
    if (d2 < best) [best, i0] = [d2, j];
    if (d2 > r2 || !(W[j]! >= p.wMin)) return;
    const tx = T[2 * j]!, ty = T[2 * j + 1]!;
    const k = ((tx * ppx + ty * ppy < 0 ? -1 : 1) * W[j]!) / (Math.sqrt(d2) + p.soft);
    sx += k * tx;
    sy += k * ty;
  };
  /** The field at (x, y) given the travel direction (px, py): its unit direction and the nearest point's opening; null outside. */
  const field = (x: number, y: number, px: number, py: number): [number, number, number] | null => {
    [i0, best, sx, sy, qx, qy, ppx, ppy] = [-1, Infinity, 0, 0, x, y, px, py];
    pts.each(x, y, visit);
    if (i0 < 0 || best > r2 || !(W[i0]! >= p.wMin)) return null;
    const l = Math.hypot(sx, sy);
    return l > 0 ? [sx / l, sy / l, W[i0]!] : null;
  };

  /** Within `r` of a line already drawn. */
  const occupied = (x: number, y: number, r: number): boolean => {
    occ.near(x, y, buf2);
    for (const k of buf2) {
      const dx = occXY[2 * k]! - x, dy = occXY[2 * k + 1]! - y;
      if (dx * dx + dy * dy < r * r) return true;
    }
    return false;
  };

  /** One way from (x, y) along (dx, dy): the points after the start, and their openings. */
  const oneWay = (x: number, y: number, dx: number, dy: number): { xy: number[]; w: number[] } => {
    const xy: number[] = [], w: number[] = [];
    for (let s = 0; s < maxSteps; s++) {
      const f1 = field(x, y, dx, dy);
      if (!f1) break;
      const f2 = field(x + 0.5 * p.step * f1[0], y + 0.5 * p.step * f1[1], f1[0], f1[1]);
      if (!f2 || f2[0] * dx + f2[1] * dy < cosTurn) break;
      const nx = x + p.step * f2[0], ny = y + p.step * f2[1];
      // A point that left the field ends the line at the next step's own field evaluation.
      if (occupied(nx, ny, 0.5 * p.dsep)) break;
      [x, y, dx, dy] = [nx, ny, f2[0], f2[1]];
      xy.push(x, y);
      w.push(f2[2]);
    }
    return { xy, w };
  };

  const seeds = Array.from({ length: n }, (_, i) => i).filter((i) => W[i]! >= p.wMin).sort((a, b) => W[b]! - W[a]!);
  const lines: Polyline[] = [];
  for (const i of seeds) {
    const sx = P[2 * i]!, sy = P[2 * i + 1]!;
    if (occupied(sx, sy, p.dsep)) continue;
    // An axis has no sign: give the seed's a canonical one (x > 0, else y > 0), so a line never depends on it.
    const flip = T[2 * i]! < 0 || (T[2 * i]! === 0 && T[2 * i + 1]! < 0) ? -1 : 1;
    const tx = flip * T[2 * i]!, ty = flip * T[2 * i + 1]!;
    const fwd = oneWay(sx, sy, tx, ty);
    const back = oneWay(sx, sy, -tx, -ty);
    const xy: number[] = [], w: number[] = [];
    for (let k = back.w.length - 1; k >= 0; k--) {
      xy.push(back.xy[2 * k]!, back.xy[2 * k + 1]!);
      w.push(back.w[k]!);
    }
    xy.push(sx, sy);
    w.push(W[i]!);
    xy.push(...fwd.xy);
    w.push(...fwd.w);
    let len = 0;
    for (let k = 1; k < w.length; k++) len += Math.hypot(xy[2 * k]! - xy[2 * k - 2]!, xy[2 * k + 1]! - xy[2 * k - 1]!);
    if (len < p.minLen) continue;
    lines.push({ xy, w });
    for (let k = 0; k < w.length; k++) {
      occ.add(xy[2 * k]!, xy[2 * k + 1]!, occXY.length / 2);
      occXY.push(xy[2 * k]!, xy[2 * k + 1]!);
    }
  }
  return lines;
}
