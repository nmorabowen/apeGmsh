// The Cracks card's traced crack lines on layered shells (CRACK_PLOTS_SPEC.md §3.3): the shells are grouped into
// planar panels (a wall, a slab level); in each panel and on each face, the outer concrete layer's widest crack (C1)
// gives a trace direction and an opening at every Gauss point (state/shellCracks.ts's frame and face rules), the
// tracer draws the lines (state/crackTrace.ts), and each line vertex is located in its element as natural
// coordinates (xi, eta), so the renderer places it through the element's own shape functions, deformed or not, and
// never re-traces for a deformation scale. Pure: tested without a window.

import { Grid, traceCracks, traceParamsFor } from "./crackTrace.ts";
import { FACE_RGB, frameRows, type ShellFace, type ShellPointIndex, type ShellTickGroup } from "./shellCracks.ts";

/** The face of a line: the face toward the positive global axis (`plus`, red) or the other. */
export type LineFace = "plus" | "minus";

export interface CrackLineInput {
  /** each Gauss point's reference (undeformed) position, 3 per point */
  base: Float32Array;
  points: ShellPointIndex;
  groups: readonly ShellTickGroup[];
  /** each group's element ids, in row order */
  elementIds: readonly Float64Array[];
  /** an element's four corners, reference positions, corners in gmsh quad order; null for any other cell */
  corners: (elementId: number) => Float64Array | null;
}

export interface CrackLineOptions {
  face: ShellFace;
  /** an opening below this is uncracked for the tracer */
  minOpening: number;
  /** the Gauss pitch, the unit of every tracing length */
  pitch: number;
  /** the hair a line sits off its face */
  eps: number;
}

/** The traced lines of one step, as vertices located in their elements. */
export interface CrackLines {
  /** per vertex: the element id, its natural coordinates (2), its lift off the mid-surface (3), its opening */
  elementId: Float64Array;
  xi: Float64Array;
  lift: Float32Array;
  w: Float32Array;
  /** per line: its first vertex (and one past the last at the end), and its face */
  lineStart: Int32Array;
  face: LineFace[];
  /** the panels traced, and the Gauss points left out for want of a frame or a quad */
  panels: number;
  skipped: number;
  maxOpening: number;
}

/** Of the traced points, every this many is drawn (the trace step is 0.1 x the pitch). */
const DRAW_EVERY = 4;

const QUAD_N = (xi: number, eta: number) => [0.25 * (1 - xi) * (1 - eta), 0.25 * (1 + xi) * (1 - eta), 0.25 * (1 + xi) * (1 + eta), 0.25 * (1 - xi) * (1 + eta)];

/** The natural coordinates of (x, y) in a quad with corners `c` (4 x 2), by Newton on the bilinear map; null when it does not converge. */
export function invertQuad(c: ArrayLike<number>, x: number, y: number): [number, number] | null {
  let xi = 0, eta = 0;
  for (let it = 0; it < 12; it++) {
    const N = QUAD_N(xi, eta);
    let fx = -x, fy = -y;
    for (let k = 0; k < 4; k++) {
      fx += N[k]! * c[2 * k]!;
      fy += N[k]! * c[2 * k + 1]!;
    }
    // dN/dxi, dN/deta
    const dxi = [-(1 - eta), 1 - eta, 1 + eta, -(1 + eta)].map((v) => 0.25 * v);
    const deta = [-(1 - xi), -(1 + xi), 1 + xi, 1 - xi].map((v) => 0.25 * v);
    let a = 0, b = 0, cc = 0, d = 0;
    for (let k = 0; k < 4; k++) {
      a += dxi[k]! * c[2 * k]!;
      b += deta[k]! * c[2 * k]!;
      cc += dxi[k]! * c[2 * k + 1]!;
      d += deta[k]! * c[2 * k + 1]!;
    }
    const det = a * d - b * cc;
    if (!(Math.abs(det) > 0)) return null;
    const sx = (d * fx - b * fy) / det, sy = (-cc * fx + a * fy) / det;
    xi -= sx;
    eta -= sy;
    if (Math.abs(sx) + Math.abs(sy) < 1e-12) break;
  }
  return Number.isFinite(xi) && Number.isFinite(eta) ? [xi, eta] : null;
}

/** Two unit vectors spanning the plane normal to the unit n. */
function basisOf(n: readonly number[]): { u: number[]; v: number[] } {
  const ax = Math.abs(n[0]!), ay = Math.abs(n[1]!), az = Math.abs(n[2]!);
  const e = ax <= ay && ax <= az ? [1, 0, 0] : ay <= az ? [0, 1, 0] : [0, 0, 1];
  let u = [n[1]! * e[2]! - n[2]! * e[1]!, n[2]! * e[0]! - n[0]! * e[2]!, n[0]! * e[1]! - n[1]! * e[0]!];
  const l = Math.hypot(u[0]!, u[1]!, u[2]!);
  u = u.map((x) => x / l);
  return { u, v: [n[1]! * u[2]! - n[2]! * u[1]!, n[2]! * u[0]! - n[0]! * u[2]!, n[0]! * u[1]! - n[1]! * u[0]!] };
}

const dot3 = (a: ArrayLike<number>, b: ArrayLike<number>, bo = 0) => a[0]! * b[bo]! + a[1]! * b[bo + 1]! + a[2]! * b[bo + 2]!;

/** Trace the crack lines of one step (see the header). */
export function buildCrackLines(input: CrackLineInput, opts: CrackLineOptions): CrackLines {
  const { base, points, groups, elementIds } = input;
  const h = opts.pitch;
  // Per (group, element): its frame rows, face layers and lift, and its panel, computed once.
  type Elem = { e1: number[]; e2: number[]; e3: number[]; plusAt: number; minusAt: number; plusSide: number; half: number; id: number; corners: Float64Array };
  const elems = new Map<string, Elem | null>();
  const elemOf = (gi: number, e: number): Elem | null => {
    const key = `${gi}:${e}`;
    if (elems.has(key)) return elems.get(key)!;
    const g = groups[gi]!;
    const id = elementIds[gi]![e]!;
    const corners = g.recorded[e] ? input.corners(id) : null;
    let out: Elem | null = null;
    if (corners && corners.length === 12) {
      const q = g.quaternions;
      const [e1, e2, e3] = frameRows(q[4 * e]!, q[4 * e + 1]!, q[4 * e + 2]!, q[4 * e + 3]!);
      const k = [0, 1, 2].reduce((a, b) => (Math.abs(e3[b]!) > Math.abs(e3[a]!) ? b : a), 0);
      const lastIsPlus = e3[k]! > 0;
      const last = g.layers.length - 1;
      out = { e1, e2, e3, plusAt: lastIsPlus ? last : 0, minusAt: lastIsPlus ? 0 : last, plusSide: lastIsPlus ? 1 : -1, half: g.thickness ? g.thickness[e]! / 2 : 0, id, corners };
    }
    elems.set(key, out);
    return out;
  };

  // Panels: shells in one plane (the normal's direction, sign-free, and its offset to the half pitch).
  type Panel = { n: number[]; u: number[]; v: number[]; pts: number[] };
  const panels = new Map<string, Panel>();
  let skipped = 0;
  const n = points.group.length;
  for (let i = 0; i < n; i++) {
    const el = elemOf(points.group[i]!, points.element[i]!);
    if (!el || !Number.isFinite(base[3 * i]!)) {
      skipped++;
      continue;
    }
    const k = [0, 1, 2].reduce((a, b) => (Math.abs(el.e3[b]!) > Math.abs(el.e3[a]!) ? b : a), 0);
    const s = el.e3[k]! < 0 ? -1 : 1;
    const nn = el.e3.map((x) => s * x);
    const off = dot3(nn, base, 3 * i);
    const key = `${nn.map((x) => Math.round(x * 50)).join(",")}|${Math.round(off / (0.5 * h))}`;
    let p = panels.get(key);
    if (!p) {
      const { u, v } = basisOf(nn);
      p = { n: nn, u, v, pts: [] };
      panels.set(key, p);
    }
    p.pts.push(i);
  }

  const faces: LineFace[] = opts.face === "both" ? ["plus", "minus"] : [opts.face];
  const elementId: number[] = [], xi: number[] = [], lift: number[] = [], wv: number[] = [], lineStart: number[] = [];
  const lineFace: LineFace[] = [];
  let maxOpening = 0;
  const params = traceParamsFor(h, Math.max(opts.minOpening, 1e-12));
  for (const p of panels.values()) {
    const m = p.pts.length;
    const P2 = new Float64Array(2 * m);
    p.pts.forEach((i, a) => {
      P2[2 * a] = dot3(p.u, base, 3 * i);
      P2[2 * a + 1] = dot3(p.v, base, 3 * i);
    });
    const near = new Grid(2 * h);
    for (let a = 0; a < m; a++) near.add(P2[2 * a]!, P2[2 * a + 1]!, a);
    const hits: number[] = [];
    // Each point's element, as 2-D corners, for locating the line vertices.
    const corners2 = p.pts.map((i) => {
      const el = elemOf(points.group[i]!, points.element[i]!)!;
      const c = new Float64Array(8);
      for (let k = 0; k < 4; k++) {
        c[2 * k] = dot3(p.u, el.corners, 3 * k);
        c[2 * k + 1] = dot3(p.v, el.corners, 3 * k);
      }
      return c;
    });
    for (const f of faces) {
      const T2 = new Float64Array(2 * m), W = new Float64Array(m);
      p.pts.forEach((i, a) => {
        const gi = points.group[i]!, e = points.element[i]!, gp = points.gauss[i]!;
        const g = groups[gi]!, el = elemOf(gi, e)!;
        const at = ((e * g.nGauss + gp) * g.layers.length + (f === "plus" ? el.plusAt : el.minusAt)) * 9;
        const cx = g.values[at]!, cy = g.values[at + 1]!, cz = g.values[at + 2]!;
        const w = Math.hypot(cx, cy, cz);
        if (!(w > 0)) return;
        const nx = (cx * el.e1[0]! + cy * el.e2[0]! + cz * el.e3[0]!) / w;
        const ny = (cx * el.e1[1]! + cy * el.e2[1]! + cz * el.e3[1]!) / w;
        const nz = (cx * el.e1[2]! + cy * el.e2[2]! + cz * el.e3[2]!) / w;
        const t = [el.e3[1]! * nz - el.e3[2]! * ny, el.e3[2]! * nx - el.e3[0]! * nz, el.e3[0]! * ny - el.e3[1]! * nx];
        const tu = dot3(p.u, t), tv = dot3(p.v, t);
        const tl = Math.hypot(tu, tv);
        if (!(tl > 1e-9)) return;
        T2[2 * a] = tu / tl;
        T2[2 * a + 1] = tv / tl;
        W[a] = w;
      });
      for (const line of traceCracks(P2, T2, W, params)) {
        // A vertex outside every element (a line runs up to rmax past the last Gauss point) ends the line there;
        // a line that leaves the mesh and comes back is split. Concrete ends at the mesh's edge, and so does a crack.
        let start = elementId.length;
        const close = () => {
          if (elementId.length - start >= 2) {
            lineStart.push(start);
            lineFace.push(f);
          } else {
            elementId.length = start;
            xi.length = 2 * start;
            lift.length = 3 * start;
            wv.length = start;
          }
          start = elementId.length;
        };
        // Every 4th traced point (0.4 x the pitch apart; the trace step is 0.1), and the last: smooth enough to draw.
        const last = line.w.length - 1;
        for (let k = 0; k <= last; k = k === last ? last + 1 : Math.min(k + DRAW_EVERY, last)) {
          const x = line.xy[2 * k]!, y = line.xy[2 * k + 1]!;
          // The element holding the vertex: among the nearest points' elements, the one it is inside.
          let best: { a: number; xi: number; eta: number } | null = null;
          const tried = new Set<string>();
          const order = near
            .near(x, y, hits)
            .map((a) => ({ a, d: (P2[2 * a]! - x) ** 2 + (P2[2 * a + 1]! - y) ** 2 }))
            .sort((o1, o2) => o1.d - o2.d);
          for (const { a } of order) {
            const i = p.pts[a]!;
            const key = `${points.group[i]}:${points.element[i]}`;
            if (tried.has(key)) continue;
            tried.add(key);
            const r = invertQuad(corners2[a]!, x, y);
            if (r && Math.abs(r[0]) <= 1 + 1e-6 && Math.abs(r[1]) <= 1 + 1e-6) {
              best = { a, xi: r[0], eta: r[1] };
              break;
            }
          }
          if (!best) {
            close();
            continue;
          }
          const i = p.pts[best.a]!;
          const el = elemOf(points.group[i]!, points.element[i]!)!;
          const side = f === "plus" ? el.plusSide : -el.plusSide;
          const off = side * (el.half + opts.eps);
          elementId.push(el.id);
          xi.push(Math.max(-1, Math.min(1, best.xi)), Math.max(-1, Math.min(1, best.eta)));
          lift.push(off * el.e3[0]!, off * el.e3[1]!, off * el.e3[2]!);
          wv.push(line.w[k]!);
          maxOpening = Math.max(maxOpening, line.w[k]!);
        }
        close();
      }
    }
  }
  lineStart.push(elementId.length);
  return {
    elementId: Float64Array.from(elementId),
    xi: Float64Array.from(xi),
    lift: Float32Array.from(lift),
    w: Float32Array.from(wv),
    lineStart: Int32Array.from(lineStart),
    face: lineFace,
    panels: panels.size,
    skipped,
    maxOpening,
  };
}

export interface CrackRibbons {
  /** non-indexed triangles, 3 floats per vertex */
  positions: Float32Array;
  /** sRGB 0..1, 3 per vertex */
  colours: Float32Array;
  segments: number;
}

/**
 * Ribbons along the lines: `at` the lines' vertex positions on the mid-surface (3 per vertex, deformed or not, from
 * the element's shape functions), lifted by `lines.lift`. Each segment is a strip lying in the face, its half-width
 * from the opening on a log scale, `halfMin` at the smallest drawn to `halfMax` at the largest.
 */
export function buildRibbons(lines: CrackLines, at: Float32Array, halfMin: number, halfMax: number, wMin: number): CrackRibbons {
  const pos: number[] = [], col: number[] = [];
  const lo = Math.max(wMin, lines.maxOpening / 400, 1e-30);
  const span = Math.log(Math.max(lines.maxOpening, lo * 1.0001) / lo);
  const halfOf = (w: number) => halfMin + (halfMax - halfMin) * Math.min(1, Math.max(0, Math.log(Math.max(w, lo) / lo) / span));
  for (let l = 0; l + 1 < lines.lineStart.length; l++) {
    const rgb = FACE_RGB[lines.face[l]!];
    for (let v = lines.lineStart[l]!; v + 1 < lines.lineStart[l + 1]!; v++) {
      const a = [0, 1, 2].map((k) => at[3 * v + k]! + lines.lift[3 * v + k]!);
      const b = [0, 1, 2].map((k) => at[3 * v + 3 + k]! + lines.lift[3 * v + 3 + k]!);
      if (!a.every(Number.isFinite) || !b.every(Number.isFinite)) continue;
      const d = [b[0]! - a[0]!, b[1]! - a[1]!, b[2]! - a[2]!];
      const nl = Math.hypot(lines.lift[3 * v]!, lines.lift[3 * v + 1]!, lines.lift[3 * v + 2]!) || 1;
      const nrm = [lines.lift[3 * v]! / nl, lines.lift[3 * v + 1]! / nl, lines.lift[3 * v + 2]! / nl];
      const s = [nrm[1]! * d[2]! - nrm[2]! * d[1]!, nrm[2]! * d[0]! - nrm[0]! * d[2]!, nrm[0]! * d[1]! - nrm[1]! * d[0]!];
      const sl = Math.hypot(s[0]!, s[1]!, s[2]!);
      if (!(sl > 0)) continue;
      const ha = halfOf(lines.w[v]!) / sl, hb = halfOf(lines.w[v + 1]!) / sl;
      const a0 = a.map((x, k) => x - ha * s[k]!), a1 = a.map((x, k) => x + ha * s[k]!);
      const b0 = b.map((x, k) => x - hb * s[k]!), b1 = b.map((x, k) => x + hb * s[k]!);
      pos.push(...a0, ...b0, ...b1, ...a0, ...b1, ...a1);
      for (let k = 0; k < 6; k++) col.push(...rgb);
    }
  }
  return { positions: Float32Array.from(pos), colours: Float32Array.from(col), segments: pos.length / 18 };
}
