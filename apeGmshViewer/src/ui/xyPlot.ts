// One small x-y plot as SVG: the curve, the zero lines, optional horizontal
// reference lines, and an optional marked point. Moved here from the Hinges card
// (panels/hinges.ts) so the material tester (panels/windows/materialTest.ts) draws
// with the same code; ADR 0137 H2's shared chart replaces both.
//
// Pure DOM: no store, no app imports (the ui-allow-list rule).

const SVG_NS = "http://www.w3.org/2000/svg";
const W = 260, H = 150, L = 50, R = 10, T = 10, B = 24;

function svgEl(tag: string, attrs: Record<string, string | number>, text?: string): HTMLElement {
  const doc = document as Document & { createElementNS?: (ns: string, q: string) => Element };
  const node = (typeof doc.createElementNS === "function" ? doc.createElementNS(SVG_NS, tag) : doc.createElement(tag)) as HTMLElement;
  for (const [k, v] of Object.entries(attrs)) node.setAttribute?.(k, String(v));
  if (text !== undefined) node.textContent = text;
  return node;
}

/** A plot label: three significant figures, exponent form outside 1e-2 to 1e4. */
export const fmt = (v: number): string => (v === 0 ? "0" : Math.abs(v) >= 1e4 || Math.abs(v) < 1e-2 ? v.toExponential(2) : Number(v.toPrecision(3)).toString());

function finiteRange(values: readonly number[], extra: readonly number[]): [number, number] {
  let lo = 0, hi = 0;
  for (const list of [values, extra]) {
    for (const v of list) {
      if (!Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  return lo === hi ? [lo - 1, hi + 1] : [lo, hi];
}

export interface XyPlot {
  xs: readonly number[];
  ys: readonly number[];
  xName: string;
  yName: string;
  /** the accessible name */
  title: string;
  /** horizontal reference lines (a capacity), dashed */
  caps?: readonly number[];
  /** the index of a point to mark (the step on screen); none when absent */
  mark?: number;
}

/** The plot. Both axes span zero; non-finite points are skipped. */
export function xyPlot(p: XyPlot): HTMLElement {
  const caps = p.caps ?? [];
  const [x0, x1] = finiteRange(p.xs, []);
  const [y0, y1] = finiteRange(p.ys, caps);
  const px = (v: number) => L + ((v - x0) / (x1 - x0)) * (W - L - R);
  const py = (v: number) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
  const root = svgEl("svg", { class: "xy-plot", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": p.title });
  const axis = { class: "xy-axis" };
  root.append(svgEl("line", { ...axis, x1: L, y1: py(0), x2: W - R, y2: py(0) }), svgEl("line", { ...axis, x1: px(0), y1: T, x2: px(0), y2: H - B }));
  for (const c of caps) root.append(svgEl("line", { class: "xy-capacity", x1: L, y1: py(c), x2: W - R, y2: py(c) }));
  const points: string[] = [];
  p.xs.forEach((x, i) => {
    const y = p.ys[i]!;
    if (Number.isFinite(x) && Number.isFinite(y)) points.push(`${px(x).toFixed(1)},${py(y).toFixed(1)}`);
  });
  root.append(svgEl("polyline", { class: "xy-curve", points: points.join(" ") }));
  const k = p.mark;
  if (k !== undefined && Number.isFinite(p.xs[k]) && Number.isFinite(p.ys[k])) root.append(svgEl("circle", { class: "xy-now", cx: px(p.xs[k]!), cy: py(p.ys[k]!), r: 3.5 }));
  const label = { class: "xy-label" };
  root.append(
    svgEl("text", { ...label, x: L - 4, y: T + 8, "text-anchor": "end" }, fmt(y1)),
    svgEl("text", { ...label, x: L - 4, y: H - B, "text-anchor": "end" }, fmt(y0)),
    svgEl("text", { ...label, x: L, y: H - 8 }, fmt(x0)),
    svgEl("text", { ...label, x: W - R, y: H - 8, "text-anchor": "end" }, fmt(x1)),
    svgEl("text", { ...label, x: (L + W - R) / 2, y: H - 8, "text-anchor": "middle" }, p.xName),
    svgEl("text", { ...label, x: L + 4, y: T + 8 }, p.yName),
  );
  return root;
}
