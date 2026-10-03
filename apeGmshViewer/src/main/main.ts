// Electron main process. It reads files and nothing else (ADR 0112 D4):
// no child processes, no network, no writes except the measurement /
// capture outputs a script asked for.
//
// Modes (argv, set by scripts/launch.mjs):
//   --mode=view     open a window on --file (or an empty drop target)
//   --mode=measure  open, orbit, pick, write a JSON measurement to --out
//   --mode=capture  open hidden, select a beam, write a PNG still to --out

import { app, BrowserWindow, ipcMain } from "electron";
import { writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { openModel } from "../reader/node.ts";

const here = dirname(fileURLToPath(import.meta.url));

function arg(name: string): string | undefined {
  const pre = `--${name}=`;
  return process.argv.find((a) => a.startsWith(pre))?.slice(pre.length);
}

const mode = arg("mode") ?? "view";
if (mode !== "view" && mode !== "measure" && mode !== "capture") {
  throw new Error(`unknown --mode=${mode}; expected view, measure or capture`);
}
const file = arg("file");
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

let appReadyAt = 0;
ipcMain.handle("app:config", () => ({
  mode,
  file: file ? resolve(file) : null,
  t0,
  appReadyMs: appReadyAt - t0,
  configMs: Date.now() - t0,
  pick: arg("pick") ?? null,
}));

ipcMain.handle("model:open", async (_e, path: string) => {
  try {
    return { ok: true, model: await openModel(path) };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) };
  }
});

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
  const win = new BrowserWindow({
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
  win.setMenuBarVisibility(false);
  void win.loadFile(join(here, "index.html"));
});

app.on("window-all-closed", () => app.quit());
