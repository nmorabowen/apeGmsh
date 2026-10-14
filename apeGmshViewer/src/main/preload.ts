// The renderer's only door to the main process.
//
// Contract for the renderer (V2e's effects.ts feature-detects these):
//   onOpen(cb(set))        cb hears the set already open (if any) once, then
//                          every later open: {model, geometry, results}, each
//                          an absolute path or null (./pairing.ts)
//   onFileChanged(cb(path)) a file of the open set was rewritten and closed
//   goToSource(file, line)  open the user's editor; Promise<{ok, reason?}>
//   requestOpen(path)       open the set of `path` (a dropped file), as a
//                          double-click would; Promise<{ok, reason?}>
//   openGeometry(path)      read a /geometry sibling (./zones.ts); openModel's
//                          answer also carries the model's /provenance, its
//                          /meta/lineage and its /sequence element tags (V4b)
//   openResults(path)       read a results.h5's index (./results.ts, V4b)
//   readResultsStep(req)    one step of the open results file; the reply
//                          echoes `req.seq` so the renderer keeps the latest
//   scanResults(req)        the history range of a contour field (V4c1);
//                          cancelResultsScan(id) stops it, and
//                          onScanProgress(cb) hears its progress
// onOpen and onFileChanged return a function that removes the listener.
// Until the page subscribes to a channel, main reloads the page instead of
// sending on it, so the P0 page (which reads the model from config) follows.

import { contextBridge, ipcRenderer, webUtils, type IpcRendererEvent } from "electron";

interface OpenSet {
  model: string | null;
  geometry: string | null;
  results: string | null;
}

const listeners = { open: 0, fileChanged: 0 };

function listen<T>(channel: "open" | "fileChanged", cb: (value: T) => void, onLive: () => void): () => void {
  const handler = (_e: IpcRendererEvent, value: T) => {
    onLive();
    cb(value);
  };
  ipcRenderer.on(`viewer:${channel}`, handler);
  if (listeners[channel]++ === 0) ipcRenderer.send("viewer:subscribe", channel);
  let live = true;
  return () => {
    if (!live) return;
    live = false;
    ipcRenderer.removeListener(`viewer:${channel}`, handler);
    if (--listeners[channel] === 0) ipcRenderer.send("viewer:unsubscribe", channel);
  };
}

contextBridge.exposeInMainWorld("viewer", {
  config: () => ipcRenderer.invoke("app:config"),
  openModel: (path: string) => ipcRenderer.invoke("model:open", path),
  pathForFile: (f: File) => webUtils.getPathForFile(f),
  metrics: () => ipcRenderer.invoke("measure:metrics"),
  measureDone: (result: unknown) => ipcRenderer.invoke("measure:done", result),
  captureStill: (suffix: string) => ipcRenderer.invoke("capture:still", suffix),
  captureDone: () => ipcRenderer.invoke("capture:done"),
  captureResize: (width: number, height: number) => ipcRenderer.invoke("capture:resize", width, height),
  fail: (message: string) => ipcRenderer.invoke("app:fail", message),
  // V4a (#1707): one results step for --mode=measure-results; main answers only in that mode.
  resultsStep: (req: { kind: "nodal" | "gauss"; step: number }) => ipcRenderer.invoke("results:step", req),

  onOpen: (cb: (set: OpenSet) => void) => {
    // The set already open is replayed to this callback only, unless a live
    // open has reached it first.
    let heardLive = false;
    let removed = false;
    const off = listen<OpenSet>("open", cb, () => {
      heardLive = true;
    });
    void ipcRenderer.invoke("viewer:currentSet").then((set: OpenSet | null) => {
      if (set && !heardLive && !removed) cb(set);
    });
    return () => {
      removed = true;
      off();
    };
  },
  onFileChanged: (cb: (path: string) => void) => listen<string>("fileChanged", cb, () => {}),
  goToSource: (file: string, line: number) => ipcRenderer.invoke("source:goto", file, line),
  requestOpen: (path: string) => ipcRenderer.invoke("viewer:requestOpen", path),
  openGeometry: (path: string) => ipcRenderer.invoke("geometry:open", path),
  // V4b (#1728): a results file's metadata, then one step at a time (./results.ts).
  openResults: (path: string) => ipcRenderer.invoke("results:open", path),
  readResultsStep: (req: { path: string; stage: string; step: number; seq: number }) => ipcRenderer.invoke("results:readStep", req),
  // V4c1 (#1756): the history-range scan of a contour field, its cancel, and its progress.
  // V4c2b: an element scan may carry a node-located topology's extrapolation plan (main checks it).
  scanResults: (req: { path: string; stage: string; components: string[]; magnitude: boolean; scanId: number; source?: "element"; plan?: unknown }) => ipcRenderer.invoke("results:scan", req),
  cancelResultsScan: (scanId: number) => ipcRenderer.invoke("results:scanCancel", scanId),
  onScanProgress: (cb: (p: { scanId: number; done: number; total: number }) => void) => {
    const handler = (_e: IpcRendererEvent, p: { scanId: number; done: number; total: number }) => cb(p);
    ipcRenderer.on("results:scanProgress", handler);
    return () => {
      ipcRenderer.removeListener("results:scanProgress", handler);
    };
  },
  // V4c1 measure mode: the CPU load since the last call, and the working sets now.
  measureCpu: () => ipcRenderer.invoke("measure:cpu"),
  measureMemory: () => ipcRenderer.invoke("measure:memory"),
});
