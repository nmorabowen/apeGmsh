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
//
// In view mode the app is single-instance: a second launch forwards its file
// here. Opening a file opens its set (./pairing.ts), and the set is watched
// (./watch.ts). The renderer hears of a new set or a rewritten file through
// `onOpen` / `onFileChanged` once it subscribes; a renderer that has not
// subscribed (the P0 page) is reloaded instead, and reads the new model from
// `app:config`.

import { app, BrowserWindow, dialog, ipcMain } from "electron";
import { existsSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { openModel } from "../reader/node.ts";
import { fileFromArgv } from "./pairing.ts";
import { OpenSession } from "./session.ts";
import { goToSource } from "./source.ts";
import { SetWatcher } from "./watch.ts";

const here = dirname(fileURLToPath(import.meta.url));

function arg(name: string): string | undefined {
  const pre = `--${name}=`;
  return process.argv.find((a) => a.startsWith(pre))?.slice(pre.length);
}

const mode = arg("mode") ?? "view";
if (mode !== "view" && mode !== "measure" && mode !== "capture") {
  throw new Error(`unknown --mode=${mode}; expected view, measure or capture`);
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

if ((mode === "measure" || mode === "capture") && (!file || !out)) {
  throw new Error(`--mode=${mode} needs --file and --out`);
}

// View mode is single-instance: a second launch hands its argv to the first
// (the "second-instance" handler below) and exits.
if (mode === "view" && !app.requestSingleInstanceLock()) app.exit(0);

let win: BrowserWindow | null = null;
/** Renderer subscriptions ("open", "fileChanged"); a page load clears them. */
const subscribed = new Set<string>();

function fail(message: string): void {
  process.stderr.write(`apeGmshViewer: ${message}
`);
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
  note: (message) => process.stderr.write(`apeGmshViewer: ${message}
`),
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
  try {
    return { ok: true, model: await openModel(path) };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) };
  }
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
