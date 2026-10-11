// V3c2 (#1643): the patterns and time series window.
//
// Oracles: fixtures/definitions.h5 and fixtures/shoebuckle.h5, written by
// make_zone_fixtures.py; every name, key, count and target asserted here is a
// literal of that script (`definitions`: three Linear series #1 / ramp / #2,
// factor 1; pattern #1 global, `gravity` in stage 0 and `push` in stage 1,
// each stage pattern one `pg Arch` load). The series plots are checked
// against the OpenSees closed forms written out in the test, not values read
// back from the app. The view runs on a minimal DOM (below): it proves what
// the window shows and that a click dispatches `requestSource` through `act`
// and nothing else.

import assert from "node:assert/strict";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { patternsOf, view } from "../../src/panels/windows/patterns.ts";
import { archivedSamples, sampleSeries, seriesSvg } from "../../src/panels/windows/plot/series.ts";
import { WINDOWS } from "../../src/panels/windows/registry.ts";
import { openModel } from "../../src/reader/node.ts";
import { BlobStore } from "../../src/state/blobs.ts";
import { loadModel } from "../../src/state/load.ts";
import { initialState, reduce } from "../../src/state/reduce.ts";
import type { Event, State } from "../../src/state/types.ts";

// ---- a minimal DOM: elements with attributes, children, text and click listeners ----

class FakeText {
  textContent: string;
  constructor(text: string) {
    this.textContent = text;
  }
}
class FakeEl {
  children: (FakeEl | FakeText)[] = [];
  attrs: Record<string, string> = {};
  dataset: Record<string, string> = {};
  className = "";
  title = "";
  tag: string;
  ns: string | null;
  private listeners: ((e: { stopPropagation(): void }) => void)[] = [];
  constructor(tag: string, ns: string | null = null) {
    this.tag = tag;
    this.ns = ns;
  }
  append(...xs: (FakeEl | FakeText | string)[]): void {
    for (const x of xs) this.children.push(typeof x === "string" ? new FakeText(x) : x);
  }
  get textContent(): string {
    return this.children.map((c) => c.textContent).join("");
  }
  set textContent(v: string) {
    this.children = [new FakeText(v)];
  }
  setAttribute(k: string, v: string): void {
    this.attrs[k] = v;
  }
  addEventListener(type: string, fn: (e: { stopPropagation(): void }) => void): void {
    assert.equal(type, "click", "the window listens for clicks only");
    this.listeners.push(fn);
  }
  get clickable(): boolean {
    return this.listeners.length > 0;
  }
  click(): void {
    for (const f of this.listeners) f({ stopPropagation() {} });
  }
  /** every descendant (and this) that `pred` accepts */
  all(pred: (e: FakeEl) => boolean): FakeEl[] {
    const out: FakeEl[] = pred(this) ? [this] : [];
    for (const c of this.children) if (c instanceof FakeEl) out.push(...c.all(pred));
    return out;
  }
}
(globalThis as { document?: unknown }).document = {
  createElement: (tag: string) => new FakeEl(tag),
  createElementNS: (ns: string, tag: string) => new FakeEl(tag, ns),
};

// ---- the fixtures ----

await h5wasm.ready;
const fixtures = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "fixtures");
const load = async (name: string): Promise<State> =>
  reduce(initialState, { type: "fileLoaded", artifact: "model", load: loadModel(await openModel(join(fixtures, name)), new BlobStore()) });
const defs = await load("definitions.h5");
const plain = await load("shoebuckle.h5");

const render = (s: State): { root: FakeEl; events: Event[] } => {
  const events: Event[] = [];
  const root = view(s, (e) => events.push(e)) as unknown as FakeEl;
  return { root, events };
};
const rowsOf = (root: FakeEl) => root.all((e) => e.className === "window-row");

test("the registry's patterns entry is this window's view", () => {
  assert.equal(WINDOWS.patterns.view, view);
});

test("every pattern by name under the stage that adds it, its series by {$decl} ref, its tables and named targets", () => {
  const m = patternsOf(defs);
  assert.equal(m.note, null);
  assert.deepEqual(m.sections.map((x) => x.name), ["global", "stage 0 · gravity", "stage 1 · push"]);
  const ps = m.sections.flatMap((x) => x.patterns);
  // Names are handles; the unnamed global pattern shows its #k key.
  assert.deepEqual(ps.map((p) => [p.title, p.self?.key, p.type, p.tag]), [
    ["#1", "opensees/pattern/#1", "Plain", 1],
    ["gravity", "opensees/pattern/gravity", "Plain", 2],
    ["push", "opensees/pattern/push", "Plain", 3],
  ]);
  // The series is the decl_params `series` field's ref, not the HDF5 series_ref.
  assert.deepEqual(ps.map((p) => [p.series?.key, p.series?.label, p.seriesH5]), [
    ["opensees/timeSeries/#1", "#1", null],
    ["opensees/timeSeries/ramp", "ramp", null],
    ["opensees/timeSeries/#2", "#2", null],
  ]);
  // `series` is shown as the series ref, so no parameter line repeats it.
  assert.deepEqual(ps.map((p) => p.params.length), [0, 0, 0]);
  assert.deepEqual(ps.map((p) => p.tables), [["124 nodal loads (loads)"], ["1 nodal load (loads)"], ["1 nodal load (loads)"]]);
  // The stage loads name their group; the global one lists 124 node rows, which the archive does not group.
  assert.deepEqual(ps.map((p) => p.targets), [["124 node rows (no group name in the archive)"], ["pg Arch"], ["pg Arch"]]);
});

test("every time series with its parameters by field name, the patterns it drives, and its plot", () => {
  const ts = patternsOf(defs).series;
  assert.deepEqual(ts.map((t) => [t.title, t.self?.key, t.type, t.tag]), [
    ["#1", "opensees/timeSeries/#1", "Linear", 1],
    ["ramp", "opensees/timeSeries/ramp", "Linear", 2],
    ["#2", "opensees/timeSeries/#2", "Linear", 3],
  ]);
  for (const t of ts) assert.deepEqual(t.params, [{ field: "factor", text: "1", ref: null }]);
  assert.deepEqual(ts.map((t) => t.usedBy), [["#1 (global)"], ["gravity (stage 0 · gravity)"], ["push (stage 1 · push)"]]);
  for (const t of ts) {
    assert.ok(t.plot.ok);
    // Linear, factor 1: v = 1 * t over pseudo-time 0..1.
    assert.deepEqual([t.plot.samples.t, t.plot.samples.v], [[0, 1], [0, 1]]);
    assert.match(t.plot.samples.caption, /factor 1/);
  }
});

test("a file without /opensees/decls (opensees 2.22) says so, and never re-derives parameters from positional argv", () => {
  const m = patternsOf(plain);
  const [p] = m.sections.flatMap((x) => x.patterns);
  assert.equal(p!.title, "Plain_1", "with no decls row, the HDF5 group name is the only handle");
  assert.equal(p!.self, null);
  assert.equal(p!.series, null);
  assert.equal(p!.seriesH5, "/opensees/time_series/Linear_1", "the archive's series_ref attribute is shown as written");
  assert.deepEqual(p!.params, []);
  assert.match(p!.paramsNote!, /no \/opensees\/decls, opensees < 2\.25/);
  const [t] = m.series;
  assert.deepEqual(t!.params, []);
  assert.match(t!.paramsNote!, /opensees < 2\.25/);
  assert.equal(t!.plot.ok, false, "no plot without the parameters by field name");
  assert.deepEqual(t!.usedBy, ["Plain_1 (global)"], "joined by the series_ref path");
});

test("empty states are explicit", () => {
  assert.equal(patternsOf(initialState).note, "no model is open");
  assert.equal(patternsOf({ ...defs, definitions: null }).note, "the model has no /opensees zone");
  const none = { ...defs, definitions: { ...defs.definitions!, patterns: [], timeSeries: [], stages: [] } };
  assert.equal(patternsOf(none).note, "the model declares no load pattern or time series");
  assert.equal(patternsOf({ ...defs, definitions: { ...defs.definitions!, timeSeries: [] } }).note, "the model declares no time series");
  const { root } = render(none);
  assert.equal(root.textContent, "the model declares no load pattern or time series");
});

test("the view shows the sections, names, refs and one SVG plot per series", () => {
  const { root, events } = render(defs);
  const sections = root.all((e) => e.className === "window-section").map((e) => e.textContent);
  assert.deepEqual(sections, ["patterns · global", "patterns · stage 0 · gravity", "patterns · stage 1 · push", "time series"]);
  const titles = root.all((e) => e.className === "window-row-title").map((e) => e.textContent);
  assert.deepEqual(titles, ["#1", "gravity", "push", "#1", "ramp", "#2"]);
  const rows = rowsOf(root);
  assert.match(rows[1]!.textContent, /Plain · tag 2 · series ramp/);
  assert.match(rows[1]!.textContent, /targets pg Arch/);
  assert.match(rows[4]!.textContent, /Linear · tag 2 · drives gravity \(stage 0 · gravity\)/);
  assert.match(rows[4]!.textContent, /factor=1/);
  const svgs = root.all((e) => e.tag === "svg");
  assert.equal(svgs.length, 3);
  assert.ok(svgs.every((s) => s.ns === "http://www.w3.org/2000/svg"));
  // The file records no source for these: no ref is clickable, and each says why.
  const refs = root.all((e) => e.className === "window-ref");
  assert.ok(refs.length >= 3);
  assert.ok(refs.every((r) => !r.clickable));
  assert.match(refs.find((r) => r.textContent === "ramp")!.title, /no \/provenance record for opensees\/timeSeries\/ramp/);
  assert.deepEqual(events, []);
});

test("a click on a referenced series with a recorded source dispatches requestSource, which the store accepts", () => {
  const key = "opensees/timeSeries/ramp";
  const s: State = {
    ...defs,
    provenance: [{ key, origin: "user", seq: 0, source: { file: "/work/model.py", line: 12, function: "<module>", sha256: "ab", recorded: true, script: null } }],
  };
  const { root, events } = render(s);
  const refs = root.all((e) => e.className === "window-ref" && e.clickable);
  // The ramp series is reachable from gravity's `series` ref and from its own card's source link.
  assert.deepEqual(refs.map((r) => r.textContent), ["ramp", "↗ model.py:12"]);
  refs[0]!.click();
  refs[1]!.click();
  assert.deepEqual(events, [{ type: "requestSource", decl: key }, { type: "requestSource", decl: key }]);
  const after = reduce(s, events[0]!);
  assert.deepEqual(after.source.request, { decl: key, seq: 1 });
});

// ---- the plot: closed forms ------------------------------------------------------

test("Constant and Linear draw their factor; Constant v = f, Linear v = f*t", () => {
  assert.deepEqual(sampleSeries("Constant", { factor: 2.5 }), { ok: true, samples: { t: [0, 1], v: [2.5, 2.5], caption: "constant · factor 2.5" } });
  const lin = sampleSeries("Linear", { factor: -3 });
  assert.ok(lin.ok);
  assert.deepEqual([lin.samples.t, lin.samples.v], [[0, 1], [0, -3]]);
  assert.match(lin.samples.caption, /factor -3/);
  // OpenSees's default factor is 1 when the field is absent.
  const d = sampleSeries("Constant", {});
  assert.ok(d.ok && d.samples.v[0] === 1);
});

test("Path: t from dt and start_time or from time, v = factor * values; a file-only Path says it has no samples", () => {
  const a = sampleSeries("Path", { values: [0, 1, 4], dt: 0.5, start_time: 1, factor: 2 });
  assert.ok(a.ok);
  assert.deepEqual([a.samples.t, a.samples.v], [[1, 1.5, 2], [0, 2, 8]]);
  const b = sampleSeries("Path", { values: [1, 2], time: [0, 3], factor: 1 });
  assert.ok(b.ok);
  assert.deepEqual([b.samples.t, b.samples.v], [[0, 3], [1, 2]]);
  const f = sampleSeries("Path", { file: "elcentro.txt", values: null, dt: 0.01 });
  assert.equal(f.ok, false);
  assert.match(!f.ok ? f.reason : "", /read from elcentro\.txt at run time/);
  const bad = sampleSeries("Path", { values: [1, 2], time: [0] });
  assert.equal(bad.ok, false, "time and values of different lengths are refused, not truncated");
});

test("Trig: v = factor*sin(2*pi*(t - t_start)/period + shift) + zero_shift inside [t_start, t_end]", () => {
  const r = sampleSeries("Trig", { t_start: 0, t_end: 2, period: 2, factor: 3 });
  assert.ok(r.ok);
  const at = (t: number) => r.samples.v[r.samples.t.findIndex((x) => Math.abs(x - t) < 1e-12)]!;
  assert.ok(Math.abs(at(0.5) - 3) < 1e-12, "a quarter period: the peak");
  assert.ok(Math.abs(at(1.5) + 3) < 1e-12, "three quarters: the trough");
  assert.ok(Math.abs(at(1)) < 1e-12);
  const late = sampleSeries("Trig", { t_start: 1, t_end: 2, period: 1 });
  assert.ok(late.ok);
  assert.deepEqual([late.samples.t[0], late.samples.v[0]], [0, 0], "zero before t_start");
});

test("an unplotted type, or a field that is not a number, is a sentence and never a guessed curve", () => {
  const u = sampleSeries("Ricker", { f_n: 2, t_total: 1, dt: 0.01 });
  assert.equal(u.ok, false);
  assert.match(!u.ok ? u.reason : "", /no plot for a Ricker series/);
  const t = sampleSeries("Trig", { t_start: 0, t_end: 1 });
  assert.match(!t.ok ? t.reason : "", /no period/);
  const c = sampleSeries("Constant", { factor: "x" });
  assert.match(!c.ok ? c.reason : "", /factor is "x", not a number/);
});

test("the archive's time / values datasets, when present, are drawn as stored; a mismatched pair is refused", () => {
  const col = (path: string, xs: number[]) => ({ path, columns: ["value"], rows: xs.map((x) => [x]) });
  assert.equal(archivedSamples(undefined, undefined), null);
  const ok = archivedSamples(col("/t", [0, 0.1, 0.2]), col("/v", [0, 9.81, 0]));
  assert.deepEqual(ok, { ok: true, samples: { t: [0, 0.1, 0.2], v: [0, 9.81, 0], caption: "3 samples, as stored in the archive" } });
  assert.equal(archivedSamples(col("/t", [0, 1]), col("/v", [1]))?.ok, false);
  assert.equal(archivedSamples(col("/t", [0]), undefined)?.ok, false);
});

test("the SVG maps the samples into its box and labels both ranges", () => {
  const svg = seriesSvg({ t: [0, 1], v: [0, 2], caption: "c" }) as unknown as FakeEl;
  const line = svg.all((e) => e.tag === "polyline")[0]!;
  const pts = line.attrs["points"]!.split(" ").map((p) => p.split(",").map(Number));
  // t 0 -> left edge of the plot area, t 1 -> right; v 0 -> bottom, v 2 -> top.
  assert.ok(pts[0]![0]! < pts[1]![0]! && pts[0]![1]! > pts[1]![1]!);
  assert.equal(line.attrs["stroke"], "var(--accent)", "colours come from the theme tokens");
  assert.deepEqual(svg.all((e) => e.tag === "text").map((e) => e.textContent), ["2", "0", "t 0", "1"]);
});
