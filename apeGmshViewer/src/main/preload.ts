// The renderer's only door to the main process.
//
// Contract for the renderer (V2e's effects.ts feature-detects these):
//   onOpen(cb(set))        cb hears the set already open (if any) once, then
//                          every later open: {model, geometry, results, user}, each
//                          an absolute path or null (./pairing.ts)
//   onFileChanged(cb(path)) a file of the open set was rewritten and closed
//   goToSource(file, line)  open the user's editor; Promise<{ok, reason?}>
//   requestOpen(file)       open the set of a dropped File, as a double-click
//                          would; Promise<{ok, reason?}>. It takes the File, not
//                          a path: the page never names a path to open (VU12)
//   pathForFile(file)       a dropped File's path, told to main as a drop
//   openGeometry(path)      read a /geometry sibling (./zones.ts); openModel's
//                          answer also carries the model's /provenance, its
//                          /meta/lineage and its /sequence element tags (V4b)
//   openResults(path)       read a results.h5's index (./results.ts, V4b)
//   readResultsStep(req)    one step of the open results file; the reply
//                          echoes `req.seq` so the renderer keeps the latest
//   scanResults(req)        the history range of a contour field (V4c1);
//                          cancelResultsScan(id) stops it, and
//                          onScanProgress(cb) hears its progress
//   readUser(req)           ADR 0130: the set's `<stem>.user.h5` index and its
//                          staleness against the model and results files the
//                          request names (`user:read`, reader/userFile.ts
//                          `UserRead`); never raises, a refusal is `ok: false`
//   readUserCase(req)       VU7b-2: one result of one `/spectra` case of that
//                          file, with its values (`user:readCase`,
//                          reader/userSpectra.ts `CaseRead`); never raises
//   readUserResult(req)     VU16: one component of a state-free ON_NODES user
//                          result of that file, as a contour column
//                          (`user:readResult`, reader/userFile.ts
//                          `UserColumnRead`); a stale entry is refused (A6)
//   renderJob() ... renderDone(result)  the ADR 0128 agent modes (below)
//   onOffers(cb) onSnapshotRequest(cb) snapshotReply(id, reply)  ADR 0128 D5, D6 (below)
//   variants() onVariants(cb) variantsSwitch(i, label) variantsCompare(i, label)
//   readCompareStep(req) compareClose()  VU10: the study, a switch and the compare set (below)
//   workspaceRead()         VU9a (ADR 0132): the open set's `<stem>.workspace.h5`,
//                          its tables and each set's verdict (main/workspace.ts);
//                          it takes no argument, main names the file
//   workspaceSave(edit)     VU9a: one edit (`{op, ...}`) into that file; never a
//                          path (option A, #2753). A refusal replaced nothing
//   combinationsSave(edit)  VU9b (ADR 0130 A4): add a User Combination to the open
//                          set's `<stem>.user.h5`, or revert the last save, by
//                          copy-then-replace (main/combinations.ts); never a path
// onOpen and onFileChanged return a function that removes the listener.
// Until the page subscribes to a channel, main reloads the page instead of
// sending on it, so the P0 page (which reads the model from config) follows.

import { contextBridge, ipcRenderer, webUtils, type IpcRendererEvent } from "electron";

interface OpenSet {
  model: string | null;
  geometry: string | null;
  results: string | null;
  /** ADR 0130: the `<stem>.user.h5` beside the model, or null */
  user: string | null;
  /** ADR 0132 D1: the `<stem>.workspace.h5` beside the model, or null */
  workspace: string | null;
}

/**
 * VU12 (#2753), owner ruling "option A": the path of a File the user dropped,
 * as Electron names it, told to main (synchronously, so it is recorded before
 * any open that names it) and then returned. A File the page made has no path
 * on disk: "" and nothing is told. Main opens only such paths and the paths it
 * sent the page itself (main.ts `ungestured`).
 */
function gestured(f: File): string {
  const path = webUtils.getPathForFile(f);
  if (path) ipcRenderer.sendSync("viewer:dropped", path);
  return path;
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
  pathForFile: (f: File) => gestured(f),
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
  requestOpen: (f: File) => {
    const path = gestured(f);
    return path ? ipcRenderer.invoke("viewer:requestOpen", path) : Promise.resolve({ ok: false, reason: "requestOpen opens a file dropped from disk; this one has no path" });
  },
  openGeometry: (path: string) => ipcRenderer.invoke("geometry:open", path),
  // V4b (#1728): a results file's metadata, then one step at a time (./results.ts).
  openResults: (path: string) => ipcRenderer.invoke("results:open", path),
  readResultsStep: (req: { path: string; stage: string; step: number; seq: number; element?: string; lineDiagram?: string; tensor?: string[]; springs?: boolean }) => ipcRenderer.invoke("results:readStep", req),
  // V4c1 (#1756): the history-range scan of a contour field, its cancel, and its progress.
  // V4c2b: an element scan may carry a node-located topology's extrapolation plan (main checks it).
  scanResults: (req: { path: string; stage: string; components: string[]; magnitude: boolean; scanId: number; source?: "element" | "line"; plan?: unknown }) => ipcRenderer.invoke("results:scan", req),
  cancelResultsScan: (scanId: number) => ipcRenderer.invoke("results:scanCancel", scanId),
  // ADR 0130 (VU3b, #2688): the `.user.h5` index, read on request after the first frame (renderer/app.ts).
  readUser: (req: { path: string; model: string | null; results: string | null }) => ipcRenderer.invoke("user:read", req),
  // VU7b-2 (#2747): one result of one `/spectra` case with its values (`user:readCase`, VU7b-1). The path is the
  // one `readUser` named, which main gave the page; main reads only the open set's user file (VU11), so this door
  // opens nothing else.
  readUserCase: (req: { path: string; case: string; target: string; label: string }) => ipcRenderer.invoke("user:readCase", req),
  // VU16 (#2800): one component of a state-free ON_NODES user result, as a contour column (`user:readResult`). As
  // readUserCase, the path is the one `readUser` named; main reads only the open set's user file, and refuses a stale entry.
  readUserResult: (req: { path: string; name: string; component: string }) => ipcRenderer.invoke("user:readResult", req),
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
  // ADR 0128 D4 (--mode=render / describe, renderer/agentRender.ts): the job, each still, and the run's result.
  // Main answers only in those modes, and writes a PNG only under a shot name of the spec.
  renderJob: () => ipcRenderer.invoke("render:job"),
  renderPng: (name: string, png: Uint8Array) => ipcRenderer.invoke("render:png", name, png),
  renderCapture: (name: string) => ipcRenderer.invoke("render:capture", name),
  renderDone: (result: unknown) => ipcRenderer.invoke("render:done", result),
  // ADR 0128 D5 (--present): the offers main heard over the second-instance path. Main holds them until
  // this subscribes; it never reloads the page for one (INV-5), which is why they do not use `listen`.
  onOffers: (cb: (offers: unknown[]) => void) => {
    const handler = (_e: IpcRendererEvent, offers: unknown[]) => cb(offers);
    ipcRenderer.on("viewer:offers", handler);
    ipcRenderer.send("viewer:offersReady");
    return () => {
      ipcRenderer.removeListener("viewer:offers", handler);
    };
  },
  // ADR 0128 D6 (--snapshot): main asks for the view, and writes what the page replies with.
  onSnapshotRequest: (cb: (req: { id: number; image: [number, number] }) => void) => {
    const handler = (_e: IpcRendererEvent, req: { id: number; image: [number, number] }) => cb(req);
    ipcRenderer.on("viewer:snapshot", handler);
    return () => {
      ipcRenderer.removeListener("viewer:snapshot", handler);
    };
  },
  // ADR 0128 D5: a click on an offer opens its set; refused with the reason, never with a dialog.
  offerOpen: (path: string, needs: readonly string[]) => ipcRenderer.invoke("viewer:offerOpen", path, needs),
  // VU10 (ADR 0132 D9): the study main holds, and its changes. A variant is named by its row index and label,
  // never by a path: main resolves the row. The compare set's step names a stage, a step and a request number only.
  variants: () => ipcRenderer.invoke("variants:current"),
  onVariants: (cb: (study: unknown) => void) => {
    const handler = (_e: IpcRendererEvent, study: unknown) => cb(study);
    ipcRenderer.on("viewer:variants", handler);
    return () => {
      ipcRenderer.removeListener("viewer:variants", handler);
    };
  },
  variantsSwitch: (index: number, label: string) => ipcRenderer.invoke("variants:switch", index, label),
  variantsCompare: (index: number, label: string) => ipcRenderer.invoke("variants:compare", index, label),
  readCompareStep: (req: { stage: string; step: number; seq: number }) => ipcRenderer.invoke("compare:readStep", { stage: req.stage, step: req.step, seq: req.seq }),
  compareClose: () => ipcRenderer.invoke("compare:close"),
  snapshotReply: (id: number, reply: unknown) => ipcRenderer.invoke("viewer:snapshotReply", id, reply),
  // VU9a (ADR 0132 D6, D7; HG9 on #2803): the window set's workspace. Neither door names a file: main reads and writes
  // the set's own `<stem>.workspace.h5`, and refuses an edit with any key its op does not name (a path included).
  workspaceRead: () => ipcRenderer.invoke("workspace:read"),
  workspaceSave: (edit: unknown) => ipcRenderer.invoke("workspace:save", edit),
  // VU9b (ADR 0130 A4; HG9 on #2693): one combination edit (`{op: add, row, expect}` or `{op: revert, expect}`) into
  // the window set's own `<stem>.user.h5`; it never names a file, and main refuses any key the op does not name.
  combinationsSave: (edit: unknown) => ipcRenderer.invoke("combinations:save", edit),
});
