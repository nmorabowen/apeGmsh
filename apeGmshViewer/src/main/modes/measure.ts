// The measurement harness of the main process (VH2, #2866: moved out of
// ../main.ts unchanged). `--mode=measure` reports through the `measure:*`
// handlers, registered once by `registerMeasure`; `--mode=measure-results`
// runs `measureResultsMode` once the app is ready. The modes and their
// arguments are described at the top of ../main.ts.
//
// What it needs of main.ts comes through `registerMeasure` alone, so this
// module never imports main.ts: the run's end on failure (`headlessEnd`),
// main's error text (`errText`), the window set's results host
// (`resultsHost`, closed before the .ladruno read row), and the preload
// script's path (`preload`: main.ts names it from its own folder; this file's
// folder is another one when it runs unbundled, VH3 #2868).

import { app, BrowserWindow, ipcMain } from "electron";
import { existsSync, mkdirSync, statSync, writeFileSync } from "node:fs";
import { cpus } from "node:os";
import { dirname, resolve } from "node:path";
import * as h5wasm from "h5wasm/node";
import { openH5, type H5Openable } from "../../reader/h5open.ts";
import { LadrunoFile, openLadruno } from "../../reader/ladruno.ts";
import { openResults, type ResultsFile, type ResultsH5 } from "../../reader/results.ts";
import { arg, file, mode, out, t0 } from "../args.ts";
import { pairSet, type OpenSet } from "../pairing.ts";
import type { ResultsHost } from "../results.ts";

/** What the measurement harness takes from main.ts, handed over once (`registerMeasure`). */
export interface MeasureHost {
  /** End a headless run: the reason on stderr, exit 1 (main.ts). */
  headlessEnd: (reason: string) => void;
  /** main.ts's text of a thrown value. */
  errText: (err: unknown) => string;
  /** The window set's results host, closed before the .ladruno read row. */
  resultsHost: ResultsHost;
  /** The preload script every window of the app loads (main.ts's, beside dist/main.mjs). */
  preload: string;
}

/** The preload path `registerMeasure` was handed; `measureResultsMode` refuses to run without it. */
let preloadPath: string | null = null;

/** Register the `measure:*` handlers, each once. */
export function registerMeasure(host: MeasureHost): void {
  const { headlessEnd, errText, resultsHost, preload } = host;
  preloadPath = preload;

  ipcMain.handle("measure:metrics", async () => {
    const metrics = app.getAppMetrics();
    const kb = (type: string) =>
      metrics.filter((m) => m.type === type).reduce((s, m) => s + (m.memory?.workingSetSize ?? 0), 0);
    const gpu = (await app.getGPUInfo("basic")) as { gpuDevice?: { vendorId: number; deviceId: number; active?: boolean }[] };
    return {
      mainMB: kb("Browser") / 1024,
      rendererMB: kb("Tab") / 1024,
      gpuProcessMB: kb("GPU") / 1024,
      gpuDevices: gpu.gpuDevice ?? [],
    };
  });

  ipcMain.handle("measure:done", (_e, result: unknown) => {
    writeFileSync(out!, JSON.stringify(result, null, 2));
    // V4c1: the contour scrub (a model with a results sibling) prints its own ledger row.
    const contour = (result as { contour?: ContourMeasure | null } | null)?.contour;
    if (contour) printContourRow(contour);
    // V4d: the same scrub with the deformed shape on.
    const deform = (result as { deform?: ContourMeasure | null } | null)?.deform;
    if (deform) printContourRow(deform);
    // V4c2a: the scrub with an element contour on, then deformed too.
    // V4c2b: and once per node-located draw mode.
    for (const key of ["elements", "elementsDeform", "elementsAveraged", "elementsNode", "elementsDiscrete"] as const) {
      const row = (result as Partial<Record<typeof key, ContourMeasure | null>> | null)?.[key];
      if (row) printContourRow(row);
    }
    // ADR 0122 slice 4 (#2129): a `<stem>.ladruno` beside the model gets its read row.
    if (mode === "measure" && file) {
      try {
        measureLadrunoRead(resolve(file), resultsHost);
      } catch (err) {
        // headlessEnd, not exitCode + app.quit(): app.quit() exits 0 whatever process.exitCode says.
        headlessEnd(`the .ladruno read row: ${errText(err)}`);
        return;
      }
    }
    app.quit();
  });

  let cpuMark = cpuTicks();
  /** V4c1 measure: the machine's CPU load (%) since the last call (a reviewer re-runs under other loads). */
  ipcMain.handle("measure:cpu", () => {
    const now = cpuTicks();
    const d = now.total - cpuMark.total;
    const busy = d > 0 ? (100 * (d - (now.idle - cpuMark.idle))) / d : NaN;
    cpuMark = now;
    return busy;
  });
  /** V4c1 measure: the main and renderer working sets now (MB). */
  ipcMain.handle("measure:memory", () => {
    const m = app.getAppMetrics();
    const mb = (type: string) => m.filter((x) => x.type === type).reduce((s, x) => s + (x.memory?.workingSetSize ?? 0), 0) / 1024;
    return { main: mb("Browser"), renderer: mb("Tab") };
  });
}

// ---------------------------------------------------------------------------
// The .ladruno read row (ADR 0122 slice 4, #2129). With a `<stem>.ladruno`
// beside the measured model, main times the .ladruno's index and one nodal
// step, and the same for `<stem>.results.h5` of the same run when one exists,
// then prints one MEASUREMENTS row per file. It runs after the orbit, so it
// shares no clock with the first frame; the renderer's handle on the file is
// closed first, so the index is a fresh open. The `layout` column names the
// displacement dataset's chunk shape and filters: the fork writes chunked,
// deflated datasets and NativeWriter contiguous ones, and that is what the
// two read columns compare.
// ---------------------------------------------------------------------------

/** The translational displacement components, under the .ladruno and the native reader's names. */
const DISPLACEMENT = /^(DISPLACEMENT\/U[xyz]|displacement_[xyz])$/;
const WARM_REPEATS = 21;

interface ReadRow {
  file: string;
  sizeBytes: number;
  stage: string;
  steps: number;
  nodes: number;
  components: string[];
  layout: string;
  indexMs: number;
  stepColdMs: number;
  stepWarmMs: number;
}

function measureLadrunoRead(model: string, resultsHost: ResultsHost): void {
  const results = pairSet(model, existsSync).results;
  if (results === null || !/\.ladruno$/i.test(results)) return;
  resultsHost.close();
  const rows = [timeRead(results, openLadruno)];
  const native = results.replace(/\.ladruno$/i, ".results.h5");
  if (existsSync(native)) rows.push(timeRead(native, openResults));
  printReadRows(rows);
}

/** The index (open to a full index) and the middle step of the displacement components, cold then warm. */
function timeRead(path: string, open: (h5: ResultsH5, path: string) => ResultsFile | LadrunoFile): ReadRow {
  const sizeBytes = statSync(path).size;
  const [rf, indexMs] = timed(() => open(h5wasm as unknown as ResultsH5, path));
  let row: Omit<ReadRow, "layout">;
  let dataset: string;
  try {
    const s = rf.index.stages.find((st) => st.kind !== "mode" && st.nodes?.components.some((c) => DISPLACEMENT.test(c)));
    if (!s?.nodes) {
      const seen = rf.index.stages.map((st) => `${st.id}: ${st.nodes ? st.nodes.components.join(", ") : "no nodal results"}`);
      throw new Error(`${path}: the read row needs a stage with nodal displacements; the file has ${seen.join("; ") || "no stages"}`);
    }
    const nodes = s.nodes;
    const comps = nodes.components.filter((c) => DISPLACEMENT.test(c));
    const mid = Math.floor(s.nSteps / 2);
    const [, stepColdMs] = timed(() => rf.readNodalStep(s.id, comps, mid));
    const warm = Array.from({ length: WARM_REPEATS }, () => timed(() => rf.readNodalStep(s.id, comps, mid))[1]);
    // The .ladruno keeps one (T, N, C) DATA per response; the native file one (T, N) dataset per component.
    dataset = rf instanceof LadrunoFile ? `${nodes.path}/DISPLACEMENT/DATA` : `${nodes.path}/${comps[0]!}`;
    row = { file: path, sizeBytes, stage: s.name, steps: s.nSteps, nodes: nodes.ids.length, components: comps, indexMs, stepColdMs, stepWarmMs: median(warm) };
  } finally {
    rf.close();
  }
  return { ...row, layout: layoutOf(path, dataset) };
}

/** `chunks [a, b, c], deflate` or `contiguous`, of one dataset (read after the timed handle is closed). */
function layoutOf(path: string, dataset: string): string {
  const f = openH5(h5wasm as unknown as H5Openable<h5wasm.File>, path);
  try {
    if (Number(f.file_id) < 0) throw new Error(`HDF5 could not open ${path}`);
    const ds = f.get(dataset);
    if (!(ds instanceof h5wasm.Dataset)) throw new Error(`${path}: ${dataset} is not a dataset`);
    const chunks = ds.metadata.chunks;
    const filters = ds.filters.map((x) => x.name || `filter ${x.id}`);
    return [chunks ? `chunks [${chunks.join(", ")}]` : "contiguous", ...filters].join(", ");
  } finally {
    f.close();
  }
}

function printReadRows(rows: ReadRow[]): void {
  process.stdout.write(
    "| file | size (MB) | stage | steps | nodes | components | layout | index (ms) | one step cold / warm (ms) |\n" +
      "|---|---|---|---|---|---|---|---|---|\n" +
      rows
        .map(
          (r) =>
            `| ${r.file.split(/[\\/]/).pop()} | ${(r.sizeBytes / 1048576).toFixed(2)} | ${r.stage} | ${r.steps} | ${r.nodes} | ${r.components.join(", ")} | ${r.layout} | ` +
            `${r.indexMs.toFixed(1)} | ${r.stepColdMs.toFixed(2)} / ${r.stepWarmMs.toFixed(2)} |\n`,
        )
        .join(""),
  );
}

/** The renderer's V4c1 contour scrub measurement (renderer/app.ts `measureContour`). */
interface ContourMeasure {
  file: string;
  field: string;
  steps: number;
  scanMs: number;
  scrubStepsPerSec: number;
  stepMs: { median: number; max: number };
  peakMB: { main: number; renderer: number };
  cpu: { before: number; during: number };
}

function printContourRow(c: ContourMeasure): void {
  const name = c.file.split(/[\\/]/).pop();
  const peak = `${(c.peakMB.main + c.peakMB.renderer).toFixed(0)} (${c.peakMB.main.toFixed(0)} + ${c.peakMB.renderer.toFixed(0)})`;
  process.stdout.write(
    "| file | field | steps | CPU before / during | history scan (ms) | scrub (steps/s) | step median / max (ms) | peak MB main+renderer |\n" +
      "|---|---|---|---|---|---|---|---|\n" +
      `| ${name} | ${c.field} | ${c.steps} | ${c.cpu.before.toFixed(0)} / ${c.cpu.during.toFixed(0)} % | ${c.scanMs.toFixed(0)} | ` +
      `${c.scrubStepsPerSec.toFixed(1)} | ${c.stepMs.median.toFixed(2)} / ${c.stepMs.max.toFixed(1)} | ${peak} |\n`,
  );
}

/** Summed os.cpus() ticks: idle and total, for the CPU load of the machine. */
function cpuTicks(): { idle: number; total: number } {
  let idle = 0;
  let total = 0;
  for (const c of cpus()) {
    idle += c.times.idle;
    total += c.times.user + c.times.nice + c.times.sys + c.times.idle + c.times.irq;
  }
  return { idle, total };
}

// ---------------------------------------------------------------------------
// V4a measurement mode (#1707): the results.h5 read path, timed end to end.
// Main reads with reader/results.ts; the renderer gets each step over this
// app's IPC (ipcRenderer.invoke, preload `resultsStep`) and applies it to a
// colour buffer. The renderer page is a blank hidden page: the app's own page
// would start its view/measure flow on the file. Its apply is the WebGL work
// three.js r186 does for a `needsUpdate` colour attribute (one bufferSubData
// of the whole Float32Array, then a draw), written as raw WebGL2 calls
// because three.js is bundled inside the app page and cannot be reached here.
// Nothing here runs outside --mode=measure-results.
// ---------------------------------------------------------------------------

const median = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? s[Math.floor((s.length - 1) / 2)]! : NaN;
};
/** Scrub columns are means over the steps: the renderer's clock is coarsened to 0.1 ms. */
const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN);
const timed = <T>(f: () => T): [T, number] => {
  const t = performance.now();
  const v = f();
  return [v, performance.now() - t];
};

/** Peak working sets of the main and renderer processes, sampled every 20 ms. */
function memorySampler() {
  const peak = { mainMB: 0, rendererMB: 0, sumMB: 0 };
  const sample = () => {
    const m = app.getAppMetrics();
    const mb = (type: string) => m.filter((x) => x.type === type).reduce((s, x) => s + (x.memory?.workingSetSize ?? 0), 0) / 1024;
    const a = mb("Browser");
    const b = mb("Tab");
    peak.mainMB = Math.max(peak.mainMB, a);
    peak.rendererMB = Math.max(peak.rendererMB, b);
    peak.sumMB = Math.max(peak.sumMB, a + b);
  };
  const id = setInterval(sample, 20);
  return { peak, sample, stop: () => clearInterval(id) };
}

async function measureMpco(path: string, dataset: string) {
  await h5wasm.ready;
  const size = statSync(path).size;
  let f: h5wasm.File | null = null;
  const t = performance.now();
  try {
    f = openH5(h5wasm as unknown as H5Openable<h5wasm.File>, path);
    // h5wasm returns a negative id instead of throwing when HDF5 refuses the file.
    if (Number(f.file_id) < 0) throw new Error("HDF5 could not open the file (truncated, or not HDF5)");
  } catch (err) {
    return { kind: "mpco", file: path, sizeBytes: size, openMs: performance.now() - t, openError: err instanceof Error ? err.message : String(err) };
  }
  const openMs = performance.now() - t;
  try {
    const ds = f.get(dataset);
    if (!(ds instanceof h5wasm.Dataset)) throw new Error(`${dataset} is not a dataset in ${path}`);
    const [v, readMs] = timed(() => ds.value as ArrayBufferView);
    return { kind: "mpco", file: path, sizeBytes: size, openMs, dataset, shape: ds.shape, readMs, bytes: v.byteLength };
  } finally {
    f.close();
  }
}

/** `--mode=measure-results`, once the app is ready: time the read path of `path`, write the JSON to `outPath`, print the row, quit. */
export async function measureResultsMode(path: string, outPath: string): Promise<void> {
  if (preloadPath === null) throw new Error("measure-results: registerMeasure has not run, so the preload path is unknown");
  const preload = preloadPath;
  const mem = memorySampler();
  mem.sample();
  if (/\.mpco$/i.test(path)) {
    const dataset = arg("dataset");
    if (!dataset) throw new Error("an .mpco file needs --dataset=<path>");
    const r = await measureMpco(path, dataset);
    mem.sample();
    mem.stop();
    const result = { ...r, peakMemory: mem.peak };
    writeFileSync(outPath, JSON.stringify(result, null, 2));
    process.stdout.write(`${JSON.stringify(result)}\n`);
    app.quit();
    return;
  }
  await h5wasm.ready;
  const sizeBytes = statSync(path).size;
  const [rf, openMs] = timed(() => openResults(h5wasm as unknown as ResultsH5, path));
  try {
    const result = await measureOpen(rf, path, sizeBytes, openMs, mem, preload);
    writeFileSync(outPath, JSON.stringify(result, null, 2));
    printResultsRow(result);
  } finally {
    rf.close();
    mem.stop();
    app.quit();
  }
}

async function measureOpen(rf: ResultsFile, path: string, sizeBytes: number, openMs: number, mem: ReturnType<typeof memorySampler>, preload: string) {
  const s = rf.index.stages[0];
  if (!s?.nodes || s.gauss.length === 0) throw new Error(`${path}: the measurement needs a stage with nodal results and a gauss group`);
  const nodes = s.nodes;
  const vec = ["displacement_x", "displacement_y", "displacement_z"].filter((c) => nodes.components.includes(c));
  const g = s.gauss[0]!;
  const comp = g.components[0]!;
  const T = s.nSteps;
  const mid = Math.floor(T / 2);
  const rep = 21;

  // Cold: the first read of this step in this process. Warm: the same read again (median of 21).
  const [, nodalColdMs] = timed(() => rf.readNodalStep(s.id, vec, mid));
  const nodalWarm = Array.from({ length: rep }, () => timed(() => rf.readNodalStep(s.id, vec, mid))[1]);
  const [, elemColdMs] = timed(() => rf.readGaussStep(s.id, g.id, comp, mid));
  const elemWarm = Array.from({ length: rep }, () => timed(() => rf.readGaussStep(s.id, g.id, comp, mid))[1]);
  const node = nodes.ids[Math.floor(nodes.ids.length / 2)]!;
  const [hist, historyColdMs] = timed(() => rf.readNodeHistory(s.id, vec[0]!, node));
  const historyWarm = Array.from({ length: rep }, () => timed(() => rf.readNodeHistory(s.id, vec[0]!, node))[1]);
  mem.sample();

  const nodalEager = nodes.components.length * T * nodes.ids.length * 8;
  const gaussEager = s.gauss.reduce((a, x) => a + x.components.length * T * x.nElements * x.nGauss * 8, 0);

  // Main → renderer → apply, 100 consecutive steps from step 0 (or all, if fewer).
  const SCRUB = Math.min(100, T);
  ipcMain.handle("results:step", (_e, req: unknown) => {
    const r = req as { kind: "nodal" | "gauss"; step: number };
    const [values, readMs] =
      r.kind === "gauss" ? timed(() => [rf.readGaussStep(s.id, g.id, comp, r.step)]) : timed(() => rf.readNodalStep(s.id, vec, r.step));
    return { values, readMs };
  });
  const w = new BrowserWindow({
    width: 1600,
    height: 1000,
    show: false,
    webPreferences: { preload, contextIsolation: true, sandbox: true, nodeIntegration: false, backgroundThrottling: false },
  });
  await w.loadURL("data:text/html,<!doctype html><title>measure-results</title><body></body>");
  mem.sample();
  const rr = (await w.webContents.executeJavaScript(`(${rendererHarness.toString()})(${nodes.ids.length}, ${SCRUB}, ${mid})`)) as Awaited<
    ReturnType<typeof rendererHarness>
  >;
  mem.sample();
  w.destroy();
  ipcMain.removeHandler("results:step");

  const steps = rr.steps;
  return {
    kind: "results",
    file: path,
    sizeBytes,
    nodes: nodes.ids.length,
    elements: s.gauss.reduce((a, x) => a + x.nElements, 0),
    steps: T,
    fields: { nodal: nodes.components, gauss: s.gauss.map((x) => ({ group: x.id, nGauss: x.nGauss, components: x.components })) },
    sourceType: rf.index.sourceType,
    openMs,
    nodalStep: { components: vec, coldMs: nodalColdMs, warmMs: median(nodalWarm) },
    elementStep: { component: `${g.id}/${comp}`, values: g.nElements * g.nGauss, coldMs: elemColdMs, warmMs: median(elemWarm) },
    nodeHistory: { node, values: hist.length, coldMs: historyColdMs, warmMs: median(historyWarm) },
    transfer: {
      nodalBytes: steps[0]?.bytes ?? 0,
      nodalRoundTripMs: mean(steps.map((x) => x.roundTripMs)),
      nodalReadInMainMs: mean(steps.map((x) => x.readMs)),
      nodalTransferMs: mean(steps.map((x) => x.roundTripMs - x.readMs)),
      gaussBytes: rr.gauss.bytes,
      gaussTransferMs: rr.gauss.transferMs,
    },
    apply: { colourMs: mean(steps.map((x) => x.colourMs)), uploadDrawMs: mean(steps.map((x) => x.uploadDrawMs)) },
    scrub: { steps: SCRUB, totalMs: rr.totalMs, stepsPerSecond: (1000 * SCRUB) / rr.totalMs },
    peakMemory: mem.peak,
    eagerBytes: { nodal: nodalEager, gauss: gaussEager, total: nodalEager + gaussEager },
    gpu: rr.gpu,
  };
}

/**
 * Runs in the renderer (serialised with toString, so it references nothing
 * outside itself). A points draw of `n` nodes; per step: invoke main, map the
 * displacement magnitude through jet into a Float32Array colour buffer,
 * bufferSubData it, draw, gl.finish().
 */
async function rendererHarness(n: number, scrub: number, gaussStep: number) {
  type Bridge = { resultsStep: (req: { kind: "nodal" | "gauss"; step: number }) => Promise<{ values: Float64Array[]; readMs: number }> };
  const viewer = (window as unknown as { viewer: Bridge }).viewer;
  const canvas = document.createElement("canvas");
  canvas.width = 1584;
  canvas.height = 961;
  document.body.appendChild(canvas);
  const gl = canvas.getContext("webgl2");
  if (!gl) throw new Error("no WebGL2 context");
  const sh = (type: number, src: string) => {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "shader");
    return s;
  };
  const prog = gl.createProgram()!;
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, "#version 300 es\nin vec3 p; in vec3 c; out vec3 vc; void main(){ vc=c; gl_Position=vec4(p*1.8-0.9,1.0); gl_PointSize=2.0; }"));
  gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, "#version 300 es\nprecision mediump float; in vec3 vc; out vec4 o; void main(){ o=vec4(vc,1.0); }"));
  gl.linkProgram(prog);
  gl.useProgram(prog);
  const pos = new Float32Array(3 * n);
  for (let i = 0; i < pos.length; i++) pos[i] = ((i * 7919) % 1000) / 1000;
  const pb = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, pb);
  gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
  const pl = gl.getAttribLocation(prog, "p");
  gl.enableVertexAttribArray(pl);
  gl.vertexAttribPointer(pl, 3, gl.FLOAT, false, 0, 0);
  const col = new Float32Array(3 * n);
  const cb = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, cb);
  gl.bufferData(gl.ARRAY_BUFFER, col, gl.DYNAMIC_DRAW);
  const cl = gl.getAttribLocation(prog, "c");
  gl.enableVertexAttribArray(cl);
  gl.vertexAttribPointer(cl, 3, gl.FLOAT, false, 0, 0);
  const mag = new Float32Array(n);
  const steps: { roundTripMs: number; readMs: number; bytes: number; colourMs: number; uploadDrawMs: number }[] = [];
  const t0 = performance.now();
  for (let k = 0; k < scrub; k++) {
    const a = performance.now();
    const r = await viewer.resultsStep({ kind: "nodal", step: k });
    const b = performance.now();
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < n; i++) {
      let s2 = 0;
      for (const v of r.values) s2 += v[i]! * v[i]!;
      const m = Math.sqrt(s2);
      mag[i] = m;
      if (m < lo) lo = m;
      if (m > hi) hi = m;
    }
    const span = hi > lo ? hi - lo : 1;
    for (let i = 0; i < n; i++) {
      const t = (mag[i]! - lo) / span;
      col[3 * i] = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 3)));
      col[3 * i + 1] = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 2)));
      col[3 * i + 2] = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 1)));
    }
    const c = performance.now();
    gl.bindBuffer(gl.ARRAY_BUFFER, cb);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, col);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.POINTS, 0, n);
    gl.finish();
    const d = performance.now();
    steps.push({ roundTripMs: b - a, readMs: r.readMs, bytes: r.values.reduce((s, v) => s + v.byteLength, 0), colourMs: c - b, uploadDrawMs: d - c });
  }
  const totalMs = performance.now() - t0;
  const gt: number[] = [];
  let gaussBytes = 0;
  for (let i = 0; i < 5; i++) {
    const a = performance.now();
    const r = await viewer.resultsStep({ kind: "gauss", step: gaussStep });
    gt.push(performance.now() - a - r.readMs);
    gaussBytes = r.values[0]!.byteLength;
  }
  gt.sort((x, y) => x - y);
  const dbg = gl.getExtension("WEBGL_debug_renderer_info");
  return {
    steps,
    totalMs,
    gauss: { bytes: gaussBytes, transferMs: gt[2]! },
    gpu: dbg ? String(gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL)) : "unknown",
  };
}

function printResultsRow(r: Awaited<ReturnType<typeof measureOpen>>): void {
  const f0 = (x: number) => x.toFixed(0);
  const f1 = (x: number) => x.toFixed(1);
  const f2 = (x: number) => x.toFixed(2);
  const MB = (b: number) => (b / 1048576).toFixed(1);
  const fields = `${r.fields.nodal.length} nodal + ${r.fields.gauss.map((x) => `${x.components.length} x ${x.nGauss}-pt gauss`).join(", ")}`;
  process.stdout.write(
    "| date | commit | file | size (MB) | nodes | elements | steps | fields | open+index (ms) | nodal step cold / warm (ms) | element step cold / warm (ms) | node history (ms) | transfer main→renderer | apply colour + upload/draw (ms) | scrub (steps/s) | peak memory MB main+renderer | eager whole-history (MB) |\n" +
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n" +
      `| ${new Date().toISOString().slice(0, 10)} | ${arg("commit") ?? "?"} | ${r.file.split(/[\\/]/).pop()} | ${MB(r.sizeBytes)} | ${r.nodes} | ${r.elements} | ${r.steps} | ${fields} | ` +
      `${f1(r.openMs)} | ${f2(r.nodalStep.coldMs)} / ${f2(r.nodalStep.warmMs)} | ${f2(r.elementStep.coldMs)} / ${f2(r.elementStep.warmMs)} | ${f1(r.nodeHistory.coldMs)} | ` +
      `${(r.transfer.nodalBytes / 1024).toFixed(0)} KB in ${f2(r.transfer.nodalTransferMs)} ms | ${f2(r.apply.colourMs)} + ${f2(r.apply.uploadDrawMs)} | ${f1(r.scrub.stepsPerSecond)} | ` +
      `${f0(r.peakMemory.sumMB)} (${f0(r.peakMemory.mainMB)} + ${f0(r.peakMemory.rendererMB)}) | ${MB(r.eagerBytes.total)} |\n\n` +
      `detail: element step = ${r.elementStep.values} values; history = ${r.nodeHistory.values} values (warm ${f2(r.nodeHistory.warmMs)} ms); ` +
      `scrub round trip ${f2(r.transfer.nodalRoundTripMs)} ms (read in main ${f2(r.transfer.nodalReadInMainMs)} ms); ` +
      `gauss step transfer ${(r.transfer.gaussBytes / 1024).toFixed(0)} KB in ${f2(r.transfer.gaussTransferMs)} ms; ` +
      `eager nodal ${MB(r.eagerBytes.nodal)} MB + gauss ${MB(r.eagerBytes.gauss)} MB; GPU "${r.gpu}"\n`,
  );
}

// ---------------------------------------------------------------------------
// ADR 0134 D5 (M2-S4 #2983): the particle layer's measure and capture runs.
// `--mode=measure` or `--mode=capture` on a `.ladruno` or a `<stem>.particles.h5`
// runs the page as the human's window would (main tells it "view" and no file,
// so it never reads a model), and the page asks `particles:harness` for its job:
// it opens the set, waits for the first particle frame on screen, then scrubs
// every output (measure) or shows `--colour=FIELD/COMPONENT` at `--output=K`
// (capture). Main writes the still or the JSON to `--out`, prints the
// MEASUREMENTS row, and exits; a failure ends the run through `headlessEnd`.
// ---------------------------------------------------------------------------

/** Whether this run is a particle run: measure or capture on a file whose set holds particles and no model. */
export const particleRun = (mode === "measure" || mode === "capture") && !!file && /\.(ladruno|particles\.h5)$/i.test(file);

/** The page size of a particle run's window (a still is the window's content). */
export const PARTICLE_WINDOW: readonly [number, number] = [1280, 800];

/** `--colour=FIELD/COMPONENT` (the component's index), or null; a malformed one ends the run. */
function colourArg(): { field: string; component: number } | null {
  const c = arg("colour");
  if (c === undefined) return null;
  const m = /^([A-Za-z][A-Za-z0-9_]*)\/([0-9]+)$/.exec(c);
  if (!m) throw new Error(`--colour=${c}: expected FIELD/COMPONENT, the component by index (POSITION/1)`);
  return { field: m[1]!, component: Number(m[2]) };
}

/** Register the particle run's doors. `set` is the window's set (main.ts `windowSet`). */
export function registerParticleHarness(host: { headlessEnd: (reason: string) => void; errText: (err: unknown) => string; set: () => OpenSet | { reason: string } }): void {
  const { headlessEnd, errText } = host;
  ipcMain.handle("particles:harness", () => {
    if (!particleRun) return null;
    try {
      const set = host.set();
      if ("reason" in set) throw new Error(set.reason);
      const output = arg("output");
      if (output !== undefined && !/^[0-9]+$/.test(output)) throw new Error(`--output=${output}: expected an output index`);
      const scrub = arg("scrub-ms");
      if (scrub !== undefined && !/^[1-9][0-9]*$/.test(scrub)) throw new Error(`--scrub-ms=${scrub}: expected milliseconds above 0`);
      return { op: mode, set, colour: colourArg(), output: output === undefined ? null : Number(output), t0, scrubMs: scrub === undefined ? 6000 : Number(scrub) };
    } catch (err) {
      headlessEnd(`the particle run: ${errText(err)}`);
      return null;
    }
  });
  ipcMain.handle("particles:harnessStill", async (e) => {
    if (!particleRun || mode !== "capture" || !out) throw new Error("particles:harnessStill: only a particle capture run writes a still");
    const w = BrowserWindow.fromWebContents(e.sender);
    if (!w) throw new Error("particles:harnessStill: the page has no window");
    const png = (await w.webContents.capturePage()).toPNG();
    mkdirSync(dirname(resolve(out)), { recursive: true });
    writeFileSync(out, png);
    return out;
  });
  ipcMain.handle("particles:harnessDone", (_e, result: unknown) => {
    if (!particleRun || !out) throw new Error("particles:harnessDone: this is no particle run");
    if (mode === "measure") {
      writeFileSync(out, JSON.stringify(result, null, 2));
      printParticleRow(result as ParticleMeasure);
    }
    app.quit();
  });
}

interface ParticleMeasure {
  file: string;
  stage: string;
  particles: number;
  outputs: number;
  colour: { field: string; component: number } | null;
  firstFrameMs: number;
  scrub: { frames: number; ms: number; fps: number; medianFrameMs: number | null };
  memory: { first: { mainMB: number; rendererMB: number; gpuProcessMB: number } | null; scrub: { mainMB: number; rendererMB: number; gpuProcessMB: number } };
}

function printParticleRow(r: ParticleMeasure): void {
  const f0 = (x: number) => (Number.isFinite(x) ? x.toFixed(0) : "n/a");
  const f1 = (x: number) => (Number.isFinite(x) ? x.toFixed(1) : "n/a");
  const mem = (m: { mainMB: number; rendererMB: number; gpuProcessMB: number } | null) =>
    m ? `${f0(m.mainMB + m.rendererMB)} (${f0(m.mainMB)} + ${f0(m.rendererMB)}), GPU ${f0(m.gpuProcessMB)}` : "n/a";
  const size = statSync(r.file).size / 1048576;
  process.stdout.write(
    "| date | file | size (MB) | particles | outputs | colour | first frame (ms) | scrub (outputs/s) | median output (ms) | memory MB main+renderer at first frame | after scrub |\n" +
      "|---|---|---|---|---|---|---|---|---|---|---|\n" +
      `| ${new Date().toISOString().slice(0, 10)} | ${r.file.split(/[\\/]/).pop()} | ${size.toFixed(1)} | ${r.particles.toLocaleString("en-US")} | ${r.outputs} | ` +
      `${r.colour ? `${r.colour.field}/${r.colour.component}` : "plain"} | ${f0(r.firstFrameMs)} | ${f1(r.scrub.fps)} (${r.scrub.frames} in ${f0(r.scrub.ms)} ms) | ` +
      `${r.scrub.medianFrameMs === null ? "n/a" : f1(r.scrub.medianFrameMs)} | ${mem(r.memory.first)} | ${mem(r.memory.scrub)} |\n`,
  );
}
