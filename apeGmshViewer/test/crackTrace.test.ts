// The crack-line tracer (state/crackTrace.ts) on the four fields CRACK_PLOTS_SPEC.md §5.3 names, where the right
// answer is known without the app: (a) a uniform field gives parallel straight lines at least dsep apart; (b) a radial
// field around a point gives spokes that stop at the uncracked edge; (c) sign flips of the axes change nothing;
// (d) a hole stops the lines.

import assert from "node:assert/strict";
import { test } from "node:test";
import { traceCracks, traceParamsFor, type Polyline } from "../src/state/crackTrace.ts";

const H = 1;
const P = traceParamsFor(H, 0.05);

/** A square grid of points at pitch H over [0, n) x [0, n), with a trace and an opening from (x, y). */
function grid(n: number, at: (x: number, y: number) => { t: [number, number]; w: number } | null) {
  const p: number[] = [], t: number[] = [], w: number[] = [];
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      const x = i * H, y = j * H;
      const f = at(x, y);
      if (!f) continue;
      p.push(x, y);
      t.push(...f.t);
      w.push(f.w);
    }
  return { p, t, w };
}

const pts = (l: Polyline) => Array.from({ length: l.w.length }, (_, k) => [l.xy[2 * k]!, l.xy[2 * k + 1]!] as const);

test("(a) a uniform field: straight lines along it, parallel, at least dsep apart", () => {
  const g = grid(20, () => ({ t: [1, 0], w: 1 }));
  const lines = traceCracks(g.p, g.t, g.w, P);
  assert.ok(lines.length >= 5, `${lines.length} lines`);
  const ys: number[] = [];
  for (const l of lines) {
    const ps = pts(l);
    const y0 = ps[0]![1];
    for (const [, y] of ps) assert.ok(Math.abs(y - y0) < 1e-9, "straight along x");
    ys.push(y0);
    const span = ps[ps.length - 1]![0] - ps[0]![0];
    assert.ok(span > 15 * H, `a line runs across the panel (${span})`);
  }
  ys.sort((a, b) => a - b);
  for (let k = 1; k < ys.length; k++) assert.ok(ys[k]! - ys[k - 1]! >= P.dsep - 1e-9, `spacing ${ys[k]! - ys[k - 1]!} >= dsep ${P.dsep}`);
});

test("(b) a radial field: spokes through the centre's direction, stopping at the uncracked edge", () => {
  const c = 10, R = 6;
  const g = grid(21, (x, y) => {
    const dx = x - c, dy = y - c, r = Math.hypot(dx, dy);
    return { t: r > 0 ? [dx / r, dy / r] : [1, 0], w: r <= R ? 1 : 0 };
  });
  const lines = traceCracks(g.p, g.t, g.w, P);
  assert.ok(lines.length >= 4, `${lines.length} spokes`);
  for (const l of lines) {
    for (const [x, y] of pts(l)) assert.ok(Math.hypot(x - c, y - c) <= R + P.rmax, `a spoke stops near the cracked edge (r = ${Math.hypot(x - c, y - c)})`);
    // Radial: the line's end-to-end direction points through the centre's neighbourhood.
    const ps = pts(l);
    const [ax, ay] = ps[0]!, [bx, by] = ps[ps.length - 1]!;
    const lx = bx - ax, ly = by - ay;
    const cross = Math.abs((c - ax) * ly - (c - ay) * lx) / Math.hypot(lx, ly);
    assert.ok(cross < 2 * H, `the spoke's line passes within 2 pitches of the centre (${cross})`);
  }
});

test("(c) sign flips of the axes change nothing", () => {
  const at = (x: number, y: number) => ({ t: [Math.cos(0.3 + 0.02 * y), Math.sin(0.3 + 0.02 * y)] as [number, number], w: 1 + 0.1 * x });
  const a = grid(15, at);
  const b = grid(15, (x, y) => {
    const f = at(x, y);
    return (x + y) % 2 ? { t: [-f.t[0], -f.t[1]], w: f.w } : f;
  });
  const la = traceCracks(a.p, a.t, a.w, P);
  const lb = traceCracks(b.p, b.t, b.w, P);
  assert.equal(la.length, lb.length);
  la.forEach((l, i) => {
    assert.equal(l.w.length, lb[i]!.w.length);
    l.xy.forEach((v, k) => assert.ok(Math.abs(v - lb[i]!.xy[k]!) < 1e-9));
  });
});

test("(d) a hole stops the lines: none enters it", () => {
  const hole = (x: number, y: number) => x >= 8 && x <= 11 && y >= 8 && y <= 11;
  const g = grid(20, (x, y) => (hole(x, y) ? null : { t: [1, 0], w: 1 }));
  const lines = traceCracks(g.p, g.t, g.w, P);
  assert.ok(lines.length > 0);
  for (const l of lines) for (const [x, y] of pts(l)) assert.ok(!(x > 8 + P.rmax && x < 11 - P.rmax && y > 8 + P.rmax && y < 11 - P.rmax), `(${x}, ${y}) is inside the hole`);
  // The rows through the hole are split: lines end on both sides of it.
  const through = lines.filter((l) => pts(l).every(([, y]) => y > 8.5 && y < 10.5));
  assert.ok(through.every((l) => pts(l).every(([x]) => x < 9.5) || pts(l).every(([x]) => x > 9.5)), "no line crosses the hole");
});

test("uncracked points seed nothing; a line shorter than minLen is dropped", () => {
  assert.deepEqual(traceCracks([0, 0, 1, 0], [1, 0, 1, 0], [0.01, 0.01], P), []);
  assert.deepEqual(traceCracks([0, 0], [1, 0], [1], { ...P, minLen: 10 }), []);
});
