// The pure half of `npm run profile`: profile attribution, source maps,
// frame and load statistics, the verdict that refuses an empty green, and the
// ranked summary.

import assert from "node:assert/strict";
import { test } from "node:test";
import {
  cleanSource,
  cpuBusyPct,
  dispatchBreakLine,
  exitCodeOf,
  frameStats,
  pixelDelta,
  quantile,
  rankFunctions,
  rankSubscribers,
  renderSummary,
  sourceMapper,
  summarizeProfile,
  verdict,
  type CpuProfile,
  type RunResult,
  type ScenarioResult,
} from "../../scripts/profile/analyze.ts";

const frame = (functionName: string, lineNumber = 0) => ({ functionName, url: "file:///dist/renderer.js", lineNumber, columnNumber: 0 });

test("self and inclusive time; idle, program and GC split out; GC share is of busy time", () => {
  const p: CpuProfile = {
    nodes: [
      { id: 1, callFrame: frame("(root)"), children: [2, 3, 4, 5] },
      { id: 2, callFrame: frame("(idle)") },
      { id: 3, callFrame: frame("(garbage collector)") },
      { id: 4, callFrame: frame("(program)") },
      { id: 5, callFrame: frame("outer", 10), children: [6] },
      { id: 6, callFrame: frame("inner", 20) },
    ],
    samples: [2, 3, 4, 5, 6],
    timeDeltas: [10_000, 5_000, 5_000, 10_000, 20_000],
  };
  const s = summarizeProfile(p);
  assert.equal(s.sampledMs, 50);
  assert.equal(s.idleMs, 10);
  assert.equal(s.gcMs, 5);
  assert.equal(s.programMs, 5);
  assert.equal(s.busyMs, 40);
  assert.equal(s.gcPct, 12.5);
  const outer = s.top.find((f) => f.name === "outer")!;
  const inner = s.top.find((f) => f.name === "inner")!;
  assert.deepEqual([outer.selfMs, outer.inclusiveMs], [10, 30]);
  assert.deepEqual([inner.selfMs, inner.inclusiveMs], [20, 20]);
  assert.ok(!s.top.some((f) => f.name.startsWith("(") && f.name !== "(anonymous)"), "meta frames are not functions");
});

test("a recursive function's inclusive time is counted once", () => {
  const p: CpuProfile = {
    nodes: [
      { id: 1, callFrame: frame("(root)"), children: [2] },
      { id: 2, callFrame: frame("rec", 5), children: [3] },
      { id: 3, callFrame: frame("rec", 5) },
    ],
    samples: [2, 3],
    timeDeltas: [1000, 3000],
  };
  const rec = summarizeProfile(p).top.find((f) => f.name === "rec")!;
  assert.equal(rec.selfMs, 4);
  assert.equal(rec.inclusiveMs, 4);
});

test("the mapper names the source file; without one the script is named", () => {
  const p: CpuProfile = { nodes: [{ id: 1, callFrame: frame("f", 7) }], samples: [1], timeDeltas: [1000] };
  assert.equal(summarizeProfile(p).top[0]!.file, "renderer.js");
  const mapped = summarizeProfile(p, () => ({ file: "src/panels/loads.ts", line: 33 })).top[0]!;
  assert.deepEqual([mapped.file, mapped.line], ["src/panels/loads.ts", 33]);
});

test("the tool's own page code comes out, and the viewer code it calls stays", () => {
  const tool = (functionName: string) => ({ functionName, url: "agv-probe", lineNumber: 0, columnNumber: 0 });
  const p: CpuProfile = {
    nodes: [
      { id: 1, callFrame: frame("(root)"), children: [2] },
      { id: 2, callFrame: tool("tick"), children: [3, 4] },
      { id: 3, callFrame: { functionName: "requestAnimationFrame", url: "", lineNumber: 0, columnNumber: 0 } },
      { id: 4, callFrame: frame("render", 30) },
    ],
    samples: [2, 3, 4],
    timeDeltas: [1000, 1000, 8000],
  };
  const s = summarizeProfile(p);
  assert.equal(s.toolMs, 2);
  assert.equal(s.busyMs, 8);
  assert.deepEqual(s.top.map((f) => f.name), ["render"]);
});

test("source maps: VLQ across lines, with sources and lines delta-coded", () => {
  // Line 0: col 0 -> a.ts line 1, col 4 -> a.ts line 2. Line 1: col 0 -> b.ts line 2 (delta from line 2 is 0).
  const m = sourceMapper({ sources: ["../src/a.ts", "../node_modules/three/build/three.module.js"], mappings: "AAAA,IACA;ACAA" });
  assert.deepEqual(m(0, 0), { file: "src/a.ts", line: 1 });
  assert.deepEqual(m(0, 3), { file: "src/a.ts", line: 1 });
  assert.deepEqual(m(0, 9), { file: "src/a.ts", line: 2 });
  assert.deepEqual(m(1, 0), { file: "node_modules/three/build/three.module.js", line: 2 });
  assert.equal(m(9, 0), null);
});

test("cleanSource keeps the package path and drops the relative prefix", () => {
  assert.equal(cleanSource("../src/state/loads.ts"), "src/state/loads.ts");
  assert.equal(cleanSource("../../node_modules/h5wasm/x.js"), "node_modules/h5wasm/x.js");
});

test("quantile and frame stats; no interval is no statistic", () => {
  assert.equal(quantile([3, 1, 2], 0.5), 2);
  assert.ok(Number.isNaN(quantile([], 0.5)));
  assert.equal(frameStats([]), null);
  const f = frameStats([10, 10, 10, 50])!;
  assert.equal(f.frames, 4);
  assert.equal(f.medianFps, 100);
  assert.equal(f.meanFps, 50);
  assert.equal(f.worstFrameMs, 50);
});

test("machine load from two CPU samples", () => {
  assert.equal(cpuBusyPct({ idle: 100, total: 200 }, { idle: 125, total: 300 }), 75);
  assert.ok(Number.isNaN(cpuBusyPct({ idle: 1, total: 2 }, { idle: 1, total: 2 })));
});

test("pixelDelta honours the tolerance and treats a resize as all changed", () => {
  const a = { width: 2, height: 1, rgba: new Uint8Array([0, 0, 0, 255, 10, 10, 10, 255]) };
  const b = { width: 2, height: 1, rgba: new Uint8Array([5, 0, 0, 255, 200, 10, 10, 255]) };
  assert.equal(pixelDelta(a, b), 0.5);
  assert.equal(pixelDelta(a, a), 0);
  assert.equal(pixelDelta(a, { width: 1, height: 2, rgba: a.rgba }), 1);
});

const base = { finished: true, elapsedMs: 100, softMs: 1000, stateProblem: null, pixels: 0.5 };

test("verdict: a late finish is slow with its true time, never a timeout", () => {
  const v = verdict({ ...base, elapsedMs: 653_000, softMs: 600_000 });
  assert.equal(v.status, "slow");
  assert.match(v.reason!, /653000 ms/);
});

test("verdict: an unfinished job says what it last saw", () => {
  const v = verdict({ ...base, finished: false, lastSeen: "scan 3/51", elapsedMs: 900_000 });
  assert.equal(v.status, "failed");
  assert.match(v.reason!, /scan 3\/51/);
});

test("verdict: a scenario that changed nothing is never green", () => {
  assert.equal(verdict({ ...base, stateProblem: "the step stayed 0" }).status, "no-effect");
  assert.equal(verdict({ ...base, pixels: 0 }).status, "no-effect");
  assert.equal(verdict({ ...base, pixels: null }).status, "ok", "a scenario with no visual claim rests on its state check");
  assert.equal(verdict(base).status, "ok");
});

test("subscribers rank by median of one event, so a spike does not outrank a steady cost", () => {
  const ranked = rankSubscribers(
    [
      { id: 0, calls: 3, totalMs: 100, maxMs: 98, samples: [1, 1, 98] },
      { id: 1, calls: 3, totalMs: 477, maxMs: 160, samples: [159, 158, 160] },
      { id: 2, calls: 0, totalMs: 0, maxMs: 0, samples: [] },
    ],
    (id) => (id === 1 ? "src/panels/loads.ts:render:30" : `s${id}`),
  );
  assert.deepEqual(ranked.map((r) => r.where), ["src/panels/loads.ts:render:30", "s0"]);
  assert.equal(ranked[0]!.medianMs, 159);
});

const scenario = (over: Partial<ScenarioResult>): ScenarioResult => ({
  name: "x",
  what: "x",
  status: "ok",
  reason: null,
  durationMs: 10,
  repMs: [10],
  proof: { state: "x", pixelsChangedPct: 1 },
  frames: null,
  longTasks: { count: 0, totalMs: 0, maxMs: 0 },
  renderer: null,
  main: null,
  subscribers: [],
  events: {},
  machineCpuPct: 10,
  noisy: false,
  metrics: {},
  ...over,
});

const run = (scenarios: ScenarioResult[]): RunResult => ({
  tool: "apegmsh-viewer-profile",
  version: 1,
  file: "C:\\m\\model.h5",
  sizeBytes: 1048576,
  date: "2026-11-02",
  commit: "abc",
  machine: { cpus: "cpu", cores: 8, gpu: "gpu", viewport: [100, 100], pixelRatio: 1 },
  model: { cells: 1, nodes: 2, results: null, steps: null },
  samplingUs: 200,
  warnings: [],
  scenarios,
});

const summary = (selfMs: number, name: string) => ({ sampledMs: 0, idleMs: 0, programMs: 0, gcMs: 0, busyMs: 0, gcPct: 0, toolMs: 0, top: [{ name, file: "src/a.ts", line: 1, selfMs, inclusiveMs: selfMs }] });

test("functions rank by self time across scenarios and processes, skipping what did not run", () => {
  const rows = rankFunctions(
    run([
      scenario({ name: "a", renderer: summary(5, "small") }),
      scenario({ name: "b", main: summary(50, "big") }),
      scenario({ name: "c", status: "skipped", renderer: summary(99, "ghost") }),
    ]),
  );
  assert.deepEqual(rows.map((r) => [r.name, r.process]), [["big", "main"], ["small", "renderer"]]);
});

test("the summary leads with what proved nothing and marks noisy scenarios", () => {
  const text = renderSummary(
    run([
      scenario({ name: "store-event", repMs: [1, 2], subscribers: [{ id: 1, where: "src/panels/loads.ts:render:30", calls: 2, medianMs: 159, maxMs: 160, totalMs: 318 }] }),
      scenario({ name: "orbit", status: "no-effect", reason: "nothing moved", noisy: true, machineCpuPct: 99 }),
    ]),
  );
  assert.match(text, /1 scenario\(s\) did not prove anything: orbit/);
  assert.match(text, /Noisy .*orbit 99%/);
  assert.match(text, /src\/panels\/loads\.ts:render:30 \| 159\.000/);
  assert.match(text, /\*\*orbit\*\* \(no-effect\): nothing moved/);
});

test("the exit code fails on no-effect and failed; slow and skipped pass", () => {
  assert.equal(exitCodeOf(run([scenario({ status: "ok" }), scenario({ status: "slow" }), scenario({ status: "skipped" })])), 0);
  assert.equal(exitCodeOf(run([scenario({ status: "no-effect" })])), 1);
  assert.equal(exitCodeOf(run([scenario({ status: "failed" })])), 1);
});

test("the breakpoint goes on the first statement of Store.dispatch, and a bundle without one is refused", () => {
  const bundle = ["var x = 1;", "var Store = class {", "  get() {", "  }", "  dispatch(e) {", "    const prev = this.state;", "  }", "};"].join("\n");
  assert.equal(dispatchBreakLine(bundle), 5);
  assert.throws(() => dispatchBreakLine("var Other = class { dispatch(e) {\n } };"), /Store.dispatch was not found/);
});
