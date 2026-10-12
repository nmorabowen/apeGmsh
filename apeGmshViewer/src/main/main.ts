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
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { openModel } from "../reader/node.ts";
import { openResults, type ResultsFile, type ResultsH5 } from "../reader/results.ts";
import { fileFromArgv } from "./pairing.ts";
import { OpenSession } from "./session.ts";
import { goToSource } from "./source.ts";
import { SetWatcher } from "./watch.ts";
import { openGeometry, openProvenance } from "./zones.ts";

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
  return { ok: true, model, provenance: await openProvenance(path) };
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
  app.quit();
});

/** Write one still: `--out` itself, or `<stem>.<suffix>.png` beside it. */
ipcMain.handle("capture:still", async (e, suffix: string) => {
  const win = BrowserWindow.fromWebContents(e.sender);
  if (!win) throw new Error("capture: no window");
  const img = await win.webContents.capturePage();
  const size = img.getSize();
  const path = suffix ? out!.replace(/\.png$/i, "") + `.${suffix}.png` : out!;
  if (img.isEmpty() || size.width === 0) {
    writeFileSync(`${out!}.error.txt`, `capturePage returned an empty image for ${path}`);
    return null;
  }
  writeFileSync(path, img.toPNG());
  return path;
});

ipcMain.handle("capture:done", () => app.quit());

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
