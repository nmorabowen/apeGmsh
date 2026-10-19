// Electron main process. It reads files and nothing else (ADR 0112 D4):
// no network, no writes except the measurement / capture outputs a script
// asked for, and no child process except the user's editor for go-to-source
// (D3, ./source.ts).
//
// Modes (argv, set by scripts/launch.mjs):
//   --mode=view     open a window on --file, or on the first positional
//                   argument (a double-clicked file), or an empty drop target
//   --mode=measure  open, orbit, pick, write a JSON measurement to --out
//   --mode=capture  open hidden, select a beam, write a PNG still to --out
//                   (and, with a results sibling, one still per results stage)
//   --mode=measure-results  time the results.h5 read path (V4a, #1707; the
//                   block at the end of this file), write a JSON to --out and
//                   print a MEASUREMENTS row. Started directly, after
//                   `npm run build`: `electron . --mode=measure-results
//                   --file=<x.results.h5> --out=<json>`, or with an `.mpco`
//                   file and `--dataset=<path>` for the raw MPCO point.
//
// In view mode the app is single-instance: a second launch forwards its file
// here. Opening a file opens its set (./pairing.ts), and the set is watched
// (./watch.ts). The renderer hears of a new set or a rewritten file through
// `onOpen` / `onFileChanged` once it subscribes; a renderer that has not
// subscribed (the P0 page) is reloaded instead, and reads the new model from
// `app:config`.

import { app, BrowserWindow, dialog, ipcMain } from "electron";
import { existsSync, statSync, writeFileSync } from "node:fs";
import { cpus } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { openModel } from "../reader/node.ts";
import { LadrunoFile, openLadruno } from "../reader/ladruno.ts";
import { openResults, type ResultsFile, type ResultsH5 } from "../reader/results.ts";
import { fileFromArgv, pairSet } from "./pairing.ts";
import { ResultsHost, type ScanRequest, type StepRequest } from "./results.ts";
import { checkPlan } from "../state/extrapolation.ts";
import { OpenSession } from "./session.ts";
import { goToSource } from "./source.ts";
import { SetWatcher } from "./watch.ts";
import { openGeometry, openLineage, openProvenance, openStageElements } from "./zones.ts";

const here = dirname(fileURLToPath(import.meta.url));

function arg(name: string): string | undefined {
  const pre = `--${name}=`;
  return process.argv.find((a) => a.startsWith(pre))?.slice(pre.length);
}

const mode = arg("mode") ?? "view";
if (mode !== "view" && mode !== "measure" && mode !== "capture" && mode !== "measure-results") {
  throw new Error(`unknown --mode=${mode}; expected view, measure, capture or measure-results`);
}
// Unpackaged, argv[1] is the app path; packaged, the file follows the exe.
const argvSkip = process.defaultApp ? 2 : 1;
const file = mode === "view" ? fileFromArgv(process.argv, argvSkip, process.cwd()) : arg("file");
const out = arg("out");
const t0 = Number(arg("t0") ?? Date.now());

// Measurement of GPU throughput: lift the vsync and frame-rate caps.
if (arg("uncapped") === "1") {
  app.commandLine.appendSwitch("disable-frame-rate-limit");
  app.commandLine.appendSwitch("disable-gpu-vsync");
}

if ((mode === "measure" || mode === "capture" || mode === "measure-results") && (!file || !out)) {
  throw new Error(`--mode=${mode} needs --file and --out`);
}

// View mode is single-instance: a second launch hands its argv to the first
// (the "second-instance" handler below) and exits.
if (mode === "view" && !app.requestSingleInstanceLock()) app.exit(0);

let win: BrowserWindow | null = null;
/** Renderer subscriptions ("open", "fileChanged"); a page load clears them. */
const subscribed = new Set<string>();

function fail(message: string): void {
  process.stderr.write(`apeGmshViewer: ${message}\n`);
  if (mode === "view") void app.whenReady().then(() => dialog.showErrorBox("apeGmshViewer", message));
}

/** The renderer hears `channel`; not subscribed, it reloads and re-reads config. */
function deliver(channel: "open" | "fileChanged", payload: unknown): void {
  if (!win) return;
  if (subscribed.has(channel)) win.webContents.send(`viewer:${channel}`, payload);
  else win.webContents.reload();
}

/** The open set and its watcher (view mode only). */
const session = new OpenSession({
  exists: existsSync,
  watch: (opened, sink) => new SetWatcher(opened, sink).start(),
  deliver,
  fail,
  note: (message) => process.stderr.write(`apeGmshViewer: ${message}\n`),
});
const openPath = (path: string, notify: boolean) => session.open(path, notify);

if (mode === "view") {
  if (file) openPath(file, false);
  app.on("second-instance", (_e, argv, cwd) => {
    const next = fileFromArgv(argv, argvSkip, cwd);
    if (next) openPath(next, true);
    if (win) {
      if (win.isMinimized()) win.restore();
      win.focus();
    }
  });
  // macOS delivers a double-clicked file as an event, not in argv.
  app.on("open-file", (e, path) => {
    e.preventDefault();
    openPath(path, app.isReady());
  });
}

let appReadyAt = 0;
ipcMain.handle("app:config", () => ({
  mode,
  file: mode === "view" ? (session.set?.model ?? null) : file ? resolve(file) : null,
  t0,
  appReadyMs: appReadyAt - t0,
  configMs: Date.now() - t0,
  pick: arg("pick") ?? null,
  // capture: the results sibling of the captured model, when one exists (the
  // V4b stills come from the set the file belongs to, as a double-click opens it);
  // measure (V4c1): the same sibling, for the contour scrub row
  results: (mode === "capture" || mode === "measure") && file ? pairSet(resolve(file), existsSync).results : null,
}));

ipcMain.handle("model:open", async (_e, path: string) => {
  // A model the renderer opened on its own (a drop on the P0 page): the open
  // set follows what is on screen (./session.ts).
  if (mode === "view" && typeof path === "string") session.rendererOpened(path);
  let model;
  try {
    model = await openModel(path);
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) };
  }
  // /provenance rides with the model; a refused or broken zone does not stop
  // the model from loading (the renderer shows why go-to-source is off).
  return {
    ok: true,
    model,
    provenance: await openProvenance(path),
    // V4b: the results pairing key, and the element tags of the /sequence element rows.
    lineage: await openLineage(path),
    stageElements: await openStageElements(path, model.sequence),
  };
});

// V4b (#1728): the results file, read lazily here; the renderer holds its
// metadata and asks for one step at a time (./results.ts).
const resultsHost = new ResultsHost(h5wasm as unknown as ResultsH5, h5wasm.ready);
ipcMain.handle("results:open", (_e, path: unknown) => {
  if (typeof path !== "string") return { ok: false, error: `results:open needs a path; got ${JSON.stringify(path)}` };
  return resultsHost.open(path);
});
ipcMain.handle("results:readStep", (_e, req: unknown) => {
  const r = req as StepRequest | null;
  if (
    !r ||
    typeof r.path !== "string" ||
    typeof r.stage !== "string" ||
    typeof r.step !== "number" ||
    typeof r.seq !== "number" ||
    (r.element !== undefined && typeof r.element !== "string") ||
    (r.lineDiagram !== undefined && typeof r.lineDiagram !== "string")
  ) {
    return { ok: false, seq: -1, error: `results:readStep: malformed request ${JSON.stringify(req)}` };
  }
  return resultsHost.readStep(r);
});
ipcMain.handle("results:close", () => resultsHost.close());
// V4c1 (#1756): the history-range scan of a contour's field; progress goes to
// the asking page. In capture mode it reads one step per chunk, 2 s apart,
// so a still can show the legend while the scan is still running (a hidden
// window paints slowly; renderer/app.ts checks the scan was still running).
ipcMain.handle("results:scan", (e, req: unknown) => {
  const r = req as ScanRequest | null;
  if (
    !r ||
    typeof r.path !== "string" ||
    typeof r.stage !== "string" ||
    !Array.isArray(r.components) ||
    typeof r.magnitude !== "boolean" ||
    typeof r.scanId !== "number" ||
    (r.source !== undefined && r.source !== "element" && r.source !== "line") ||
    // V4c2b: a plan only with an element scan, and only one that checks (state/extrapolation.ts `checkPlan`).
    (r.plan !== undefined && (r.source !== "element" || !checkPlan(r.plan)))
  ) {
    return { ok: false, scanId: -1, error: `results:scan: malformed request ${JSON.stringify(req)}` };
  }
  const sender = e.sender;
  const progress = (p: { scanId: number; done: number; total: number }) => {
    if (!sender.isDestroyed()) sender.send("results:scanProgress", p);
  };
  return resultsHost.scan(r, progress, mode === "capture" ? { chunkSteps: 1, paceMs: 2000 } : {});
});
ipcMain.handle("results:scanCancel", (_e, scanId: unknown) => {
  if (typeof scanId === "number") resultsHost.cancelScan(scanId);
});

ipcMain.handle("geometry:open", (_e, path: unknown) => {
  if (typeof path !== "string") return { ok: false, error: `geometry:open needs a path; got ${JSON.stringify(path)}` };
  return openGeometry(path);
});

for (const verb of ["subscribe", "unsubscribe"] as const) {
  ipcMain.on(`viewer:${verb}`, (_e, channel: unknown) => {
    if (channel !== "open" && channel !== "fileChanged") {
      fail(`viewer:${verb}: unknown channel ${JSON.stringify(channel)}`);
      return;
    }
    if (verb === "subscribe") subscribed.add(channel);
    else subscribed.delete(channel);
  });
}

// The set already open, for a new onOpen listener (the preload replays it).
ipcMain.handle("viewer:currentSet", () => (mode === "view" ? session.set : null));

ipcMain.handle("viewer:requestOpen", (_e, path: unknown) => {
  if (typeof path !== "string") return { ok: false, reason: `requestOpen needs a path; got ${JSON.stringify(path)}` };
  return openPath(path, true) ? { ok: true } : { ok: false, reason: `cannot open ${path}` };
});

ipcMain.handle("source:goto", (_e, path: unknown, line: unknown) => goToSource(path, line));

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
      measureLadrunoRead(resolve(file));
    } catch (err) {
      process.stderr.write(`apeGmshViewer: measure: the .ladruno read row: ${err instanceof Error ? (err.stack ?? err.message) : String(err)}\n`);
      process.exitCode = 1;
    }
  }
  app.quit();
});

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

function measureLadrunoRead(model: string): void {
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
  const f = new h5wasm.File(path, "r");
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

/** Write one still: `--out` itself, or `<stem>.<suffix>.png` beside it. */
ipcMain.handle("capture:still", async (e, suffix: string) => {
  const win = BrowserWindow.fromWebContents(e.sender);
  if (!win) throw new Error("capture: no window");
  const img = await win.webContents.capturePage();
  const size = img.getSize();
  const safeSuffix = suffix ? suffix.replace(/[\\/]/g, "_") : "";
  const path = safeSuffix ? out!.replace(/\.png$/i, "") + `.${safeSuffix}.png` : out!;
  if (img.isEmpty() || size.width === 0) {
    writeFileSync(`${out!}.error.txt`, `capturePage returned an empty image for ${path}`);
    return null;
  }
  writeFileSync(path, img.toPNG());
  return path;
});

ipcMain.handle("capture:done", () => app.quit());

/** Capture only (V4b): resize the page to `width` x `height` CSS pixels, for a still at another window width. */
ipcMain.handle("capture:resize", (e, width: unknown, height: unknown) => {
  if (mode !== "capture") throw new Error("capture:resize: only in --mode=capture");
  if (typeof width !== "number" || typeof height !== "number") throw new Error("capture:resize needs a width and a height");
  BrowserWindow.fromWebContents(e.sender)?.setContentSize(Math.round(width), Math.round(height));
});

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

ipcMain.handle("app:fail", (_e, message: string) => {
  process.stderr.write(`apeGmshViewer: ${message}\n`);
  if (mode !== "view") {
    process.exitCode = 1;
    app.quit();
  }
});

app.whenReady().then(() => {
  appReadyAt = Date.now();
  if (mode === "measure-results") {
    measureResultsMode(file!, out!).catch((err) => {
      process.stderr.write(`apeGmshViewer: measure-results: ${err instanceof Error ? (err.stack ?? err.message) : String(err)}\n`);
      process.exitCode = 1;
      app.quit();
    });
    return;
  }
  const w = new BrowserWindow({
    width: 1600,
    // A tall hidden window lets one still show the whole definition chain.
    height: mode === "capture" ? 1900 : 1000,
    show: mode !== "capture",
    backgroundColor: "#16181d",
    title: "apeGmshViewer",
    webPreferences: {
      preload: join(here, "preload.cjs"),
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
      backgroundThrottling: false,
    },
  });
  win = w;
  w.setMenuBarVisibility(false);
  // A page load drops the old page's subscriptions.
  w.webContents.on("did-start-loading", () => subscribed.clear());
  w.on("closed", () => {
    win = null;
  });
  void w.loadFile(join(here, "index.html"));
});

app.on("window-all-closed", () => {
  session.close();
  resultsHost.close();
  app.quit();
});

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
    f = new h5wasm.File(path, "r");
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

async function measureResultsMode(path: string, outPath: string): Promise<void> {
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
    const result = await measureOpen(rf, path, sizeBytes, openMs, mem);
    writeFileSync(outPath, JSON.stringify(result, null, 2));
    printResultsRow(result);
  } finally {
    rf.close();
    mem.stop();
    app.quit();
  }
}

async function measureOpen(rf: ResultsFile, path: string, sizeBytes: number, openMs: number, mem: ReturnType<typeof memorySampler>) {
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
    webPreferences: { preload: join(here, "preload.cjs"), contextIsolation: true, sandbox: true, nodeIntegration: false, backgroundThrottling: false },
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
