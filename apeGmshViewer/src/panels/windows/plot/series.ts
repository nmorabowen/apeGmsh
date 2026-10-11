// The time-series plot of the patterns window (V3c2): a series' samples from
// its parameters by field name (`/opensees/decl_params`), drawn as a small
// 2-D SVG line. Pure: `archivedSamples` takes a series group's `time` /
// `values` datasets when the archive has them, `sampleSeries` reads the
// fields otherwise, `seriesSvg` builds the element; none reads the store.
//
// The closed forms are OpenSees's (the bridge's `time_series.py` fields):
//   Constant  v(t) = factor
//   Linear    v(t) = factor * t
//   Path      v(t_i) = factor * values[i], t_i = time[i] or start_time + i*dt
//   Trig      v(t) = factor * sin(2*pi*(t - t_start)/period + shift) + zero_shift, t_start <= t <= t_end; 0 outside
// A Constant or Linear series has no time span of its own; it is drawn over
// pseudo-time 0..1 and its caption gives the factor. Any other type, or a
// field the archive does not carry, is a sentence, never a guessed curve.

import type { DeclValue, Table } from "../../../state/types.ts";

export interface Samples {
  t: number[];
  v: number[];
  /** what the plot shows, in words (`linear · factor 1`) */
  caption: string;
}

export type SeriesPlot = { ok: true; samples: Samples } | { ok: false; reason: string };

/** The series types this plot draws. */
export const PLOTTED: readonly string[] = ["Constant", "Linear", "Path", "Trig"];

const fmt = (x: number): string => (Number.isInteger(x) ? String(x) : Number(x.toPrecision(4)).toString());

function num(p: Record<string, DeclValue>, field: string, fallback?: number): number {
  const v = p[field];
  if (v === undefined || v === null) {
    if (fallback !== undefined) return fallback;
    throw new Error(`no ${field}`);
  }
  if (typeof v !== "number" || !Number.isFinite(v)) throw new Error(`${field} is ${JSON.stringify(v)}, not a number`);
  return v;
}

function nums(p: Record<string, DeclValue>, field: string): number[] | null {
  const v = p[field];
  if (v === undefined || v === null) return null;
  if (!Array.isArray(v) || !v.every((x) => typeof x === "number" && Number.isFinite(x))) throw new Error(`${field} is not a list of numbers`);
  return v as number[];
}

/** A series' samples from its type and its parameters by field name, or why there are none. */
export function sampleSeries(type: string, p: Record<string, DeclValue>): SeriesPlot {
  try {
    switch (type) {
      case "Constant": {
        const f = num(p, "factor", 1);
        return { ok: true, samples: { t: [0, 1], v: [f, f], caption: `constant · factor ${fmt(f)}` } };
      }
      case "Linear": {
        const f = num(p, "factor", 1);
        return { ok: true, samples: { t: [0, 1], v: [0, f], caption: `linear · factor ${fmt(f)}: value = ${fmt(f)}·t, drawn over pseudo-time 0..1` } };
      }
      case "Path": {
        const values = nums(p, "values");
        if (!values) {
          const file = p["file"];
          return { ok: false, reason: typeof file === "string" ? `a Path series read from ${file} at run time: the archive carries no samples` : "a Path series with no values in the archive" };
        }
        const f = num(p, "factor", 1);
        const time = nums(p, "time");
        let t: number[];
        let v = values.map((x) => f * x);
        if (time) {
          if (time.length !== values.length) throw new Error(`time has ${time.length} entries and values ${values.length}`);
          t = [...time];
        } else {
          const dt = num(p, "dt");
          const t0 = num(p, "start_time", 0);
          t = values.map((_, i) => t0 + i * dt);
        }
        if (p["prepend_zero"] === true) {
          // OpenSees -prependZero: a zero value ahead of the first sample (and time 0 ahead of a time list).
          t = time ? [0, ...t] : [t[0]! - num(p, "dt"), ...t];
          v = [0, ...v];
        }
        if (t.length === 0) return { ok: false, reason: "a Path series with no samples" };
        return { ok: true, samples: { t, v, caption: `path · ${values.length} samples · factor ${fmt(f)}` } };
      }
      case "Trig": {
        const t0 = num(p, "t_start"), t1 = num(p, "t_end"), T = num(p, "period");
        const f = num(p, "factor", 1), shift = num(p, "shift", 0), zs = num(p, "zero_shift", 0);
        if (!(t1 > t0) || !(T > 0)) throw new Error(`t_start ${t0}, t_end ${t1}, period ${T} span no time`);
        const n = 200;
        const t: number[] = [], v: number[] = [];
        if (t0 > 0) t.push(0), v.push(0);
        for (let i = 0; i <= n; i++) {
          const ti = t0 + ((t1 - t0) * i) / n;
          t.push(ti);
          v.push(f * Math.sin((2 * Math.PI * (ti - t0)) / T + shift) + zs);
        }
        return { ok: true, samples: { t, v, caption: `trig · period ${fmt(T)} · factor ${fmt(f)} · t ${fmt(t0)}..${fmt(t1)}` } };
      }
      default:
        return { ok: false, reason: `no plot for a ${type} series (the window draws ${PLOTTED.join(", ")})` };
    }
  } catch (err) {
    return { ok: false, reason: `the ${type} series cannot be drawn: ${err instanceof Error ? err.message : String(err)}` };
  }
}

/**
 * The `time` / `values` datasets h5-schema.md documents under a series group
 * ("sampled ... so the viewer can plot them"), drawn as stored; null when the
 * group carries neither (the 2.26 writer records only the raw args), and a
 * reason when they are malformed.
 */
export function archivedSamples(time: Table | undefined, values: Table | undefined): SeriesPlot | null {
  if (!time && !values) return null;
  const col = (x: Table | undefined): number[] | null =>
    x && x.columns.length === 1 && x.rows.every((r) => typeof r[0] === "number" && Number.isFinite(r[0])) ? x.rows.map((r) => r[0] as number) : null;
  const t = col(time), v = col(values);
  if (!t || !v || t.length !== v.length || t.length === 0) {
    return { ok: false, reason: `the series' time / values datasets do not pair (${time?.path ?? "no time"}, ${values?.path ?? "no values"})` };
  }
  return { ok: true, samples: { t, v, caption: `${t.length} samples, as stored in the archive` } };
}

const SVG = "http://www.w3.org/2000/svg";
const W = 240, H = 60, L = 34, R = 6, T = 6, B = 16;

function svg<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>, text?: string): SVGElementTagNameMap[K] {
  const e = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  if (text !== undefined) e.textContent = text;
  return e;
}

/** The samples as a line plot: the t and value ranges on the axes, the zero line, the curve in the accent colour. */
export function seriesSvg(s: Samples): SVGSVGElement {
  const tMin = Math.min(...s.t), tMax = Math.max(...s.t);
  const vMin = Math.min(0, ...s.v), vMax = Math.max(0, ...s.v);
  const tSpan = tMax - tMin || 1, vSpan = vMax - vMin || 1;
  const x = (t: number) => L + ((t - tMin) / tSpan) * (W - L - R);
  const y = (v: number) => T + (1 - (v - vMin) / vSpan) * (H - T - B);
  const root = svg("svg", { class: "series-plot", viewBox: `0 0 ${W} ${H}`, width: W, height: H, style: "flex: none", role: "img", "aria-label": s.caption });
  const axis = { stroke: "var(--line)", "stroke-width": 1, fill: "none" };
  root.append(
    svg("line", { ...axis, x1: L, y1: T, x2: L, y2: H - B }),
    svg("line", { ...axis, x1: L, y1: H - B, x2: W - R, y2: H - B }),
    svg("line", { stroke: "var(--faint)", "stroke-width": 1, "stroke-dasharray": "3 3", x1: L, y1: y(0), x2: W - R, y2: y(0) }),
    svg("polyline", {
      points: s.t.map((t, i) => `${x(t).toFixed(2)},${y(s.v[i]!).toFixed(2)}`).join(" "),
      stroke: "var(--accent)",
      "stroke-width": 2,
      fill: "none",
      "stroke-linejoin": "round",
    }),
  );
  const label = { fill: "var(--muted)", "font-size": 11, "font-family": "var(--font)" };
  root.append(
    svg("text", { ...label, x: L - 4, y: T + 8, "text-anchor": "end" }, fmt(vMax)),
    svg("text", { ...label, x: L - 4, y: H - B, "text-anchor": "end" }, fmt(vMin)),
    svg("text", { ...label, x: L, y: H - 4, "text-anchor": "start" }, `t ${fmt(tMin)}`),
    svg("text", { ...label, x: W - R, y: H - 4, "text-anchor": "end" }, fmt(tMax)),
  );
  return root;
}
