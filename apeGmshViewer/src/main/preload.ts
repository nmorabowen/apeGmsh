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
//   renderJob() ... renderDone(result)  the ADR 0128 agent modes (below);
//                          renderVariant(row) VU18: the set a shot draws, a
//                          variant row by index and label, or null (the spec's)
//   onOffers(cb) onSnapshotRequest(cb) snapshotReply(id, reply)  ADR 0128 D5, D6 (below)
//   variants() onVariants(cb) variantsSwitch(i, label) variantsCompare(i, label)
//   readCompareStep(req) compareClose()  VU10: the study, a switch and the compare set (below)
//   compareModel() compareUserRead() compareUserReadResult({name, component})
//                          VU19: the compare set's model and `.user.h5`, judged
//                          against its own model (A6); never a path or a row
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

import { contextBridge, ipcRenderer, webUtils } from "electron";
import { BULK_KEY, type BulkMessage } from "../bulk.ts";
import type { ToolPortMessage, ToolPortMeta } from "../tools/contract.ts";
import { hear, invoke, tell, type Invoke, type OpenSet, type PageFeed, type PreloadApi } from "./ipc.ts";

/** A one-argument request's argument, as main's map names it. */
type Req<C extends keyof Invoke> = Invoke[C]["req"][0];

/**
 * VU12 (#2753), owner ruling "option A": the path of a File the user dropped,
 * as Electron names it, told to main (synchronously, so it is recorded before
 * any open that names it) and then returned. A File the page made has no path
 * on disk: "" and nothing is told. Main opens only such paths and the paths it
 * sent the page itself (main.ts `ungestured`).
 */
function gestured(f: File): string {
  const path = webUtils.getPathForFile(f);
  // Synchronously: main records the drop before any open names it (ipc.ts `Send`).
  if (path) ipcRenderer.sendSync("viewer:dropped" satisfies keyof import("./ipc.ts").Send, path);
  return path;
}

const listeners = { open: 0, fileChanged: 0 };

/** Hear one of the two page feeds; the first listener subscribes, the last unsubscribes (main reloads an unsubscribed page instead). */
function listen<F extends PageFeed>(feed: F, cb: (value: F extends "open" ? OpenSet : string) => void, onLive: () => void): () => void {
  const heard = (value: OpenSet | string) => {
    onLive();
    cb(value as F extends "open" ? OpenSet : string);
  };
  const off = feed === "open" ? hear(ipcRenderer, "viewer:open", heard) : hear(ipcRenderer, "viewer:fileChanged", heard);
  if (listeners[feed]++ === 0) tell(ipcRenderer, "viewer:subscribe", feed);
  let live = true;
  return () => {
    if (!live) return;
    live = false;
    off();
    if (--listeners[feed] === 0) tell(ipcRenderer, "viewer:unsubscribe", feed);
  };
}

// ADR 0137 H1: the end of a tool window's MessagePort main hands this page, passed on to the page
// (renderer/tools.ts) with the tool's id. The tool window's own preload is ./toolPreload.ts.
ipcRenderer.on("tool:port", (e, meta: ToolPortMeta) => {
  window.postMessage({ toolPort: { window: meta.window, tool: meta.tool } } satisfies ToolPortMessage, "*", e.ports as unknown as Transferable[]);
});

let bulkSeq = 0;

/**
 * #3130: an answer too large for `contextBridge`'s copy goes to the page as one
 * structured clone (../bulk.ts); the bridge returns only its id, which the page
 * (renderer/app.ts) takes the answer by.
 */
function bulk(answer: Promise<unknown>): Promise<number> {
  return answer.then((value) => {
    const id = ++bulkSeq;
    const message: BulkMessage = { [BULK_KEY]: id, value };
    window.postMessage(message, "*");
    return id;
  });
}

// Every member is checked against ipc.ts `PreloadApi`, whose requests and answers are main's map.
const api = {
  config: () => invoke(ipcRenderer, "app:config"),
  // #3130: the model crosses as a bulk answer; the bridge returns its id (renderer/app.ts takes it).
  openModel: (path: string) => bulk(invoke(ipcRenderer, "model:open", path)),
  pathForFile: (f: File) => gestured(f),
  metrics: () => invoke(ipcRenderer, "measure:metrics"),
  measureDone: (result: unknown) => invoke(ipcRenderer, "measure:done", result),
  captureStill: (suffix: string) => invoke(ipcRenderer, "capture:still", suffix),
  captureDone: () => invoke(ipcRenderer, "capture:done"),
  captureResize: (width: number, height: number) => invoke(ipcRenderer, "capture:resize", width, height),
  fail: (message: string) => invoke(ipcRenderer, "app:fail", message),
  // V4a (#1707): one results step for --mode=measure-results; main answers only in that mode.
  resultsStep: (req: { kind: "nodal" | "gauss"; step: number }) => invoke(ipcRenderer, "results:step", req),

  onOpen: (cb: (set: OpenSet) => void) => {
    // The set already open is replayed to this callback only, unless a live
    // open has reached it first.
    let heardLive = false;
    let removed = false;
    const off = listen("open", cb, () => {
      heardLive = true;
    });
    void invoke(ipcRenderer, "viewer:currentSet").then((set: OpenSet | null) => {
      if (set && !heardLive && !removed) cb(set);
    });
    return () => {
      removed = true;
      off();
    };
  },
  onFileChanged: (cb: (path: string) => void) => listen("fileChanged", cb, () => {}),
  goToSource: (file: string, line: number) => invoke(ipcRenderer, "source:goto", file, line),
  requestOpen: (f: File) => {
    const path = gestured(f);
    return path ? invoke(ipcRenderer, "viewer:requestOpen", path) : Promise.resolve({ ok: false, reason: "requestOpen opens a file dropped from disk; this one has no path" });
  },
  openGeometry: (path: string) => invoke(ipcRenderer, "geometry:open", path),
  // V4b (#1728): a results file's metadata, then one step at a time (./results.ts).
  openResults: (path: string) => invoke(ipcRenderer, "results:open", path),
  readResultsStep: (req: Req<"results:readStep">) => invoke(ipcRenderer, "results:readStep", req),
  readHingeHistory: (req: { path: string; stage: string; element: number; seq: number }) => invoke(ipcRenderer, "results:hingeHistory", req),
  // V4c1 (#1756): the history-range scan of a contour field, its cancel, and its progress.
  // V4c2b: an element scan may carry a node-located topology's extrapolation plan (main checks it).
  scanResults: (req: Req<"results:scan">) => invoke(ipcRenderer, "results:scan", req),
  cancelResultsScan: (scanId: number) => invoke(ipcRenderer, "results:scanCancel", scanId),
  // ADR 0130 (VU3b, #2688): the `.user.h5` index, read on request after the first frame (renderer/app.ts).
  readUser: (req: { path: string; model: string | null; results: string | null }) => invoke(ipcRenderer, "user:read", req),
  // VU7b-2 (#2747): one result of one `/spectra` case with its values (`user:readCase`, VU7b-1). The path is the
  // one `readUser` named, which main gave the page; main reads only the open set's user file (VU11), so this door
  // opens nothing else.
  readUserCase: (req: Req<"user:readCase">) => invoke(ipcRenderer, "user:readCase", req),
  // VU16 (#2800): one component of a state-free ON_NODES user result, as a contour column (`user:readResult`). As
  // readUserCase, the path is the one `readUser` named; main reads only the open set's user file, and refuses a stale entry.
  readUserResult: (req: { path: string; name: string; component: string }) => invoke(ipcRenderer, "user:readResult", req),
  // ADR 0137 D4 (#3057): main's memory reading for a model load, before it and every second during it (view mode).
  // ADR 0137 H1: open a tool window (main checks the id and memory), and hear each window's status and the memory pause.
  openTool: (tool: string) => invoke(ipcRenderer, "tool:open", tool),
  onToolStatus: (cb: Parameters<PreloadApi["onToolStatus"]>[0]) => hear(ipcRenderer, "tool:status", cb),
  onToolsPaused: (cb: Parameters<PreloadApi["onToolsPaused"]>[0]) => hear(ipcRenderer, "tool:paused", cb),
  onMemoryStatus: (cb: Parameters<PreloadApi["onMemoryStatus"]>[0]) => hear(ipcRenderer, "memory:status", cb),
  onScanProgress: (cb: Parameters<PreloadApi["onScanProgress"]>[0]) => hear(ipcRenderer, "results:scanProgress", cb),
  // V4c1 measure mode: the CPU load since the last call, and the working sets now.
  measureCpu: () => invoke(ipcRenderer, "measure:cpu"),
  measureMemory: () => invoke(ipcRenderer, "measure:memory"),
  // ADR 0128 D4 (--mode=render / describe, renderer/agentRender.ts): the job, each still, and the run's result.
  // Main answers only in those modes, and writes a PNG only under a shot name of the spec.
  renderJob: () => invoke(ipcRenderer, "render:job"),
  renderPng: (name: string, png: Uint8Array) => invoke(ipcRenderer, "render:png", name, png),
  renderCapture: (name: string) => invoke(ipcRenderer, "render:capture", name),
  renderDone: (result: unknown) => invoke(ipcRenderer, "render:done", result),
  // VU18 (ADR 0132 D9.2): the set a render shot draws, a variant row by index and label (null: the spec's set), never a path.
  renderVariant: (row: { index: number; label: string } | null) => invoke(ipcRenderer, "render:variant", row === null ? null : { index: row.index, label: row.label }),
  // ADR 0128 D5 (--present): the offers main heard over the second-instance path. Main holds them until
  // this subscribes; it never reloads the page for one (INV-5), which is why they do not use `listen`.
  onOffers: (cb: Parameters<PreloadApi["onOffers"]>[0]) => {
    const off = hear(ipcRenderer, "viewer:offers", cb);
    tell(ipcRenderer, "viewer:offersReady");
    return off;
  },
  // ADR 0128 D6 (--snapshot): main asks for the view, and writes what the page replies with.
  onSnapshotRequest: (cb: Parameters<PreloadApi["onSnapshotRequest"]>[0]) => hear(ipcRenderer, "viewer:snapshot", cb),
  // ADR 0128 D5: a click on an offer opens its set; refused with the reason, never with a dialog.
  offerOpen: (path: string, needs: readonly string[]) => invoke(ipcRenderer, "viewer:offerOpen", path, needs),
  // VU10 (ADR 0132 D9): the study main holds, and its changes. A variant is named by its row index and label,
  // never by a path: main resolves the row. The compare set's step names a stage, a step and a request number only.
  variants: () => invoke(ipcRenderer, "variants:current"),
  onVariants: (cb: Parameters<PreloadApi["onVariants"]>[0]) => hear(ipcRenderer, "viewer:variants", cb),
  variantsSwitch: (index: number, label: string) => invoke(ipcRenderer, "variants:switch", index, label),
  variantsCompare: (index: number, label: string) => invoke(ipcRenderer, "variants:compare", index, label),
  readCompareStep: (req: { stage: string; step: number; seq: number }) => invoke(ipcRenderer, "compare:readStep", { stage: req.stage, step: req.step, seq: req.seq }),
  compareClose: () => invoke(ipcRenderer, "compare:close"),
  // VU19 (ADR 0132 D9.5): the compare set's model and `.user.h5`, through doors of the compare slot. None names a
  // file or a row: main reads the set it holds in its compare slot, and each answer names the slot it read.
  compareModel: () => invoke(ipcRenderer, "compare:model", {}),
  compareUserRead: () => invoke(ipcRenderer, "compare:userRead", {}),
  compareUserReadResult: (req: { name: string; component: string }) => invoke(ipcRenderer, "compare:userReadResult", { name: req.name, component: req.component }),
  snapshotReply: (id: number, reply: unknown) => invoke(ipcRenderer, "viewer:snapshotReply", id, reply),
  // VU9a (ADR 0132 D6, D7; HG9 on #2803): the window set's workspace. Neither door names a file: main reads and writes
  // the set's own `<stem>.workspace.h5`, and refuses an edit with any key its op does not name (a path included).
  workspaceRead: () => invoke(ipcRenderer, "workspace:read"),
  workspaceSave: (edit: Req<"workspace:save">) => invoke(ipcRenderer, "workspace:save", edit),
  // VU9b (ADR 0130 A4; HG9 on #2693): one combination edit (`{op: add, row, expect}` or `{op: revert, expect}`) into
  // the window set's own `<stem>.user.h5`; it never names a file, and main refuses any key the op does not name.
  combinationsSave: (edit: Req<"combinations:save">) => invoke(ipcRenderer, "combinations:save", edit),
  // ADR 0134 D5 (M2-S4 #2983): the particles. The index and one output of the window set's results file (main checks the
  // path is that file, as for `results:*`), and the set's own `<stem>.particles.h5`, which the door names itself (no path).
  particleIndex: (path: string) => invoke(ipcRenderer, "particles:index", path),
  readParticleOutput: (req: { path: string; stage: string; output: number; field: string | null; seq: number }) =>
    invoke(ipcRenderer, "particles:readOutput", { path: req.path, stage: req.stage, output: req.output, field: req.field, seq: req.seq }),
  particleSet: () => invoke(ipcRenderer, "particles:set"),
  // The particle measure and capture run (main/modes/measure.ts): its job (null in every other run), its still, its end.
  particleHarness: () => invoke(ipcRenderer, "particles:harness"),
  particleHarnessStill: () => invoke(ipcRenderer, "particles:harnessStill"),
  particleHarnessDone: (result: unknown) => invoke(ipcRenderer, "particles:harnessDone", result),
  // ADR 0135: one material tester run. The job names no executable and no file; main validates all of it.
  materialTestRun: (job: Invoke["materialTest:run"]["req"][0], seq: number) => invoke(ipcRenderer, "materialTest:run", job, seq),
  materialTestCancel: () => invoke(ipcRenderer, "materialTest:cancel"),
  materialTestSaveCsv: (text: string, name: string) => invoke(ipcRenderer, "materialTest:saveCsv", text, name),
} satisfies PreloadApi;

contextBridge.exposeInMainWorld("viewer", api);
