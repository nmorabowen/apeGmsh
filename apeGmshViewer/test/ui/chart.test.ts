// ADR 0137 D11 (H2): the shared chart's reduction to pixel columns, and what it draws.
//
// - A time series (x only grows) reduces to the per-column minimum and maximum: every column's
//   extremes survive, and at most four points per column are drawn.
// - A curve that turns back (a hysteresis loop) keeps its path order and every point where it
//   reaches a column's extreme, so the loop is drawn, never filled in.
// - Non-finite points break the path.
// - `drawChart` draws the overlaid curves first, each dashed, the main curve last and solid, and
//   the marked point; a curve of a million points draws as many points as its pixel columns allow.

import assert from "node:assert/strict";
import { test } from "node:test";
import { DASHES, drawChart, reducePath, spanZero, type ChartContext } from "../../src/ui/chart.ts";

/** The reduced points of one piece, as [x, y] pairs. */
const pairs = (piece: number[]) => Array.from({ length: piece.length / 2 }, (_, i) => [piece[2 * i]!, piece[2 * i + 1]!] as const);

test("a time series keeps every pixel column's minimum and maximum, with at most four points per column", () => {
  const n = 100_000, width = 520;
  const xs = new Float64Array(n), ys = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    xs[i] = (i / (n - 1)) * (width - 1);
    ys[i] = Math.sin(i / 37) * 50 + Math.sin(i / 3) * 5 + (i % 997 === 0 ? 40 : 0);
  }
  const [piece, ...rest] = reducePath(xs, ys);
  assert.equal(rest.length, 0);
  const kept = pairs(piece!);
  assert.ok(kept.length <= 4 * width, `${kept.length} points for ${width} columns`);
  // The oracle: each column's extremes from the raw points.
  const lo = new Map<number, number>(), hi = new Map<number, number>();
  for (let i = 0; i < n; i++) {
    const c = Math.floor(xs[i]!);
    lo.set(c, Math.min(lo.get(c) ?? Infinity, ys[i]!));
    hi.set(c, Math.max(hi.get(c) ?? -Infinity, ys[i]!));
  }
  const keptLo = new Map<number, number>(), keptHi = new Map<number, number>();
  for (const [x, y] of kept) {
    const c = Math.floor(x);
    keptLo.set(c, Math.min(keptLo.get(c) ?? Infinity, y));
    keptHi.set(c, Math.max(keptHi.get(c) ?? -Infinity, y));
  }
  assert.deepEqual([...keptLo], [...lo], "every column's minimum");
  assert.deepEqual([...keptHi], [...hi], "every column's maximum");
});

test("a hysteresis loop keeps its turns and its order: the reduced path is a subsequence of the raw one", () => {
  const n = 200_000;
  const xs = new Float64Array(n), ys = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const t = (i / n) * 12 * Math.PI;
    const amp = 100 + 0.002 * i;
    xs[i] = 260 + amp * Math.sin(t);
    ys[i] = 150 + amp * 0.6 * Math.sin(t + 0.9);
  }
  const [piece] = reducePath(xs, ys);
  const kept = pairs(piece!);
  // A loop crosses the same columns again each cycle: the bound is four points per run of one
  // column along the path, which grows with the path's length in pixels, not with its points.
  let runs = 1;
  for (let i = 1; i < n; i++) if (Math.floor(xs[i]!) !== Math.floor(xs[i - 1]!)) runs++;
  assert.ok(kept.length <= 4 * runs, `${kept.length} points for ${runs} column runs`);
  assert.ok(kept.length < n / 10, `${kept.length} of ${n} points kept`);
  // In path order: each kept point is a raw point after the one before it.
  let j = 0;
  for (const [x, y] of kept) {
    while (j < n && !(xs[j] === x && ys[j] === y)) j++;
    assert.ok(j < n, `kept point (${x}, ${y}) is not a raw point after the previous one`);
    j++;
  }
  // Every horizontal turn (a local extreme of x) is kept: the loop's ends are not cut.
  for (let i = 1; i < n - 1; i++) {
    const turn = (xs[i]! > xs[i - 1]! && xs[i]! >= xs[i + 1]!) || (xs[i]! < xs[i - 1]! && xs[i]! <= xs[i + 1]!);
    if (!turn) continue;
    const c = Math.floor(xs[i]!);
    assert.ok(kept.some(([x]) => Math.floor(x) === c), `the turn at x=${xs[i]!.toFixed(2)} has a point in its column`);
  }
  const extent = (v: Iterable<number>) => {
    let lo = Infinity, hi = -Infinity;
    for (const x of v) [lo, hi] = [Math.min(lo, x), Math.max(hi, x)];
    return [lo, hi];
  };
  // The loop spans the same pixel columns: within a column the reduction keeps the lowest and highest y,
  // so a turn's tip may sit under a pixel inward, never a column.
  assert.deepEqual(extent(kept.map((p) => p[0])).map(Math.floor), extent(xs).map(Math.floor), "the loop's full width, to the pixel column");
});

test("non-finite points break the path into pieces; a lone point stays", () => {
  const pieces = reducePath([0, 1, NaN, 3, 4, Infinity, 9], [0, 1, 5, 3, 4, 2, 2]);
  assert.deepEqual(pieces, [[0, 0, 1, 1], [3, 3, 4, 4], [9, 2]]);
  assert.deepEqual(reducePath([], []), []);
});

test("the range always spans zero, and a flat range is widened", () => {
  assert.deepEqual(spanZero([2, 5, NaN]), [0, 5]);
  assert.deepEqual(spanZero([-3, -1], [4]), [-3, 4]);
  assert.deepEqual(spanZero([0, 0]), [-1, 1]);
});

/** A 2-D context that records the calls the chart makes. */
function recorder() {
  const calls: { op: string; args: unknown[]; dash?: number[]; stroke?: unknown }[] = [];
  let dash: number[] = [];
  const ctx = {
    strokeStyle: "" as unknown,
    fillStyle: "" as unknown,
    lineWidth: 1,
    lineJoin: "miter",
    globalAlpha: 1,
    font: "",
    textAlign: "start",
    setTransform: () => undefined,
    clearRect: () => undefined,
    beginPath: () => calls.push({ op: "begin", args: [] }),
    moveTo: (...a: number[]) => calls.push({ op: "move", args: a }),
    lineTo: (...a: number[]) => calls.push({ op: "line", args: a }),
    stroke: () => calls.push({ op: "stroke", args: [], dash: [...dash], stroke: ctx.strokeStyle }),
    fill: () => calls.push({ op: "fill", args: [] }),
    arc: (...a: number[]) => calls.push({ op: "arc", args: a }),
    fillText: (...a: unknown[]) => calls.push({ op: "text", args: a }),
    setLineDash: (d: number[]) => void (dash = d),
  };
  return { ctx: ctx as unknown as ChartContext & typeof ctx, calls };
}
const COLOURS = { accent: "ACCENT", muted: "MUTED", line: "LINE", faint: "FAINT", panel: "PANEL", font: "Archivo" };

test("the chart draws the overlays dashed and muted, the main curve last in the accent, the mark, and the labels", () => {
  const { ctx, calls } = recorder();
  const drawn = drawChart(
    ctx,
    { xs: [0, 1, 2], ys: [0, 2, 1], xName: "θ", yName: "M", title: "M against θ", mark: 1, caps: [3], others: [{ xs: [0, 2], ys: [0, 1], dash: 2 }] },
    260,
    1,
    COLOURS,
  );
  assert.deepEqual(drawn, [2, 3], "the overlay, then the main curve");
  const strokes = calls.filter((c) => c.op === "stroke");
  const curveStrokes = strokes.filter((c) => c.stroke === "MUTED" || c.stroke === "ACCENT");
  assert.deepEqual(curveStrokes.map((c) => [c.stroke, c.dash]), [["MUTED", [...DASHES[2]!]], ["ACCENT", []]]);
  assert.ok(strokes.some((c) => c.stroke === "FAINT" && c.dash!.length > 0), "the capacity line is dashed");
  assert.equal(calls.filter((c) => c.op === "arc").length, 1, "the marked point");
  assert.deepEqual(calls.filter((c) => c.op === "text").map((c) => c.args[0]), ["3", "0", "0", "2", "θ", "M"]);
});

test("a curve of a million points draws no more points than its pixel columns allow", () => {
  const n = 1_000_000;
  const xs = new Float64Array(n), ys = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    xs[i] = i;
    ys[i] = Math.sin(i / 1000);
  }
  const { ctx } = recorder();
  const width = 300, dpr = 2;
  const [count] = drawChart(ctx, { xs, ys, xName: "step", yName: "v", title: "v" }, width, dpr, COLOURS);
  assert.ok(count! <= 4 * width * dpr, `${count} points drawn for ${width * dpr} device columns`);
});
