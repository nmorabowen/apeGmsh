// The page↔main IPC contract (finding 5 of the 2026-11-24 viewer architecture review): one map of
// every channel, its request and its answer, which main (`handle`, `push`), the preload (`invoke`,
// `on`, `PreloadApi`) and the page (effects.ts `Bridge`, the doors in renderer/) all read. A renamed
// field or a changed answer then fails the typecheck on both sides instead of at run time.
//
// - `Invoke`: the page asks (`ipcRenderer.invoke`) and main answers (`ipcMain.handle`). Main takes
//   the arguments as `unknown` and checks them by hand, as before: a request is the page's, and is
//   never trusted. What main answers is checked against `res`.
// - `Send`: the page tells main and waits for nothing (`ipcRenderer.send` / `sendSync`).
// - `Push`: main tells the page (`webContents.send`).
//
// Types only, from neutral modules (state/, reader/, model/, spec/, and main's own answer types), so
// main, the preload and the page import it without importing each other. The tool windows' port
// envelopes are in src/tools/contract.ts.

import type { IpcMainEvent, IpcMainInvokeEvent, IpcRendererEvent } from "electron";
import type { CaseRead, SpectrumTarget } from "../reader/userSpectra.ts";
import type { UserColumnRead } from "../reader/userFile.ts";
import type { SpecVariant, ViewSpec } from "../spec/spec.ts";
import type { ExtrapPlan } from "../state/extrapolation.ts";
import type {
  CombinationEdit,
  CombinationSaveIn,
  CompareIn,
  CompareModelAnswer,
  CompareSlotIn,
  CompareUserAnswer,
  LineStationComponent,
  MaterialTestJob,
  MemoryStatus,
  OfferIn,
  ToolStatus,
  UserReadIn,
  VariantsIn,
  WorkspaceEdit,
  WorkspaceIn,
  WorkspaceSaveIn,
} from "../state/types.ts";
import type { ToolAnswer } from "../tools/contract.ts";
import type { MaterialTestAnswer } from "./materialTest.ts";
import type { OpenSet } from "./pairing.ts";
import type { ModelOpened } from "./loader/protocol.ts";
import type {
  HingeHistoryAnswer,
  ParticleIndexAnswer,
  ParticleOutputAnswer,
  ParticleSetAnswer,
  ResultsOpenAnswer,
  ScanAnswer,
  ScanProgress,
  StepAnswer,
} from "./results.ts";
import type { SourceResult } from "./source.ts";
import type { Mode } from "./args.ts";
import type { GeometryAnswer } from "./zones.ts";

export type { OpenSet } from "./pairing.ts";

/** `app:config`: how the page starts. */
export interface AppConfig {
  mode: Mode | "refused";
  file: string | null;
  t0: number;
  appReadyMs: number;
  configMs: number;
  pick: string | null;
  /** V4b, capture mode: the results sibling of the captured model, or null */
  results?: string | null;
}

/** `model:open`: the model and its zones (loader/process.ts), or the reader's refusal. */
export type ModelOpenAnswer = ({ ok: true } & ModelOpened) | { ok: false; error: string };

/** ADR 0128 D4: the agent run's job (main/modes/render.ts `render:job`). */
export type RenderJob =
  | { mode: "render"; spec: ViewSpec; model: string; geometry: string | null; results: string | null }
  | { mode: "describe"; model: string | null; geometry: string | null; results: string | null };

/** VU18: main's answer to `render:variant`: the set the shot draws (the files to open), or why the row is refused. */
export type VariantAnswer = { ok: true; row: SpecVariant | null; set: { model: string; geometry: string | null; results: string | null } } | { ok: false; reason: string };

/** ADR 0134 D5: a measure or capture run of the particle layer, as main hands it (main/modes/measure.ts `particles:harness`). */
export interface ParticleHarnessJob {
  op: "measure" | "capture";
  set: OpenSet;
  /** the colour to show, or null for the default (POSITION's vertical component) */
  colour: { field: string; component: number } | null;
  /** the output to show, or null for the default (the last) */
  output: number | null;
  /** draw the grid's wire box (capture) */
  grid: boolean;
  /** when the launcher started the app (Date.now()), for the time to first frame */
  t0: number;
  /** how long the scrub of outputs runs (measure) */
  scrubMs: number;
}

/** A request that is refused or done, with why when refused. */
export type Done = { ok: true } | { ok: false; reason: string };

/** The step request the page sends `results:readStep` (main/results.ts `StepRequest`, as the page names it). */
export interface StepRequestIn {
  path: string;
  stage: string;
  step: number;
  seq: number;
  element?: string;
  lineDiagram?: LineStationComponent;
  tensor?: string[];
  layers?: { components: string[]; pick: "faces" | "all" };
  springs?: boolean;
  hinges?: boolean;
}

/** Every request the page makes of main, with its arguments and main's answer. */
export interface Invoke {
  "app:config": { req: []; res: AppConfig };
  "app:fail": { req: [message: string]; res: void };
  "model:open": { req: [path: string]; res: ModelOpenAnswer };
  "geometry:open": { req: [path: string]; res: GeometryAnswer };
  "results:open": { req: [path: string]; res: ResultsOpenAnswer };
  "results:readStep": { req: [req: StepRequestIn]; res: StepAnswer };
  "results:hingeHistory": { req: [req: { path: string; stage: string; element: number; seq: number }]; res: HingeHistoryAnswer };
  "results:scan": { req: [req: { path: string; stage: string; components: string[]; magnitude: boolean; scanId: number; source?: "element" | "line"; plan?: ExtrapPlan }]; res: ScanAnswer };
  "results:scanCancel": { req: [scanId: number]; res: void };
  "results:close": { req: []; res: void };
  "results:step": { req: [req: { kind: "nodal" | "gauss"; step: number }]; res: { values: Float64Array[]; readMs: number } };
  "particles:index": { req: [path: string]; res: ParticleIndexAnswer };
  "particles:readOutput": { req: [req: { path: string; stage: string; output: number; field: string | null; seq: number }]; res: ParticleOutputAnswer };
  "particles:set": { req: []; res: ParticleSetAnswer };
  "particles:harness": { req: []; res: ParticleHarnessJob | null };
  "particles:harnessStill": { req: []; res: string };
  "particles:harnessDone": { req: [result: unknown]; res: void };
  "source:goto": { req: [file: string, line: number]; res: SourceResult };
  "user:read": { req: [req: { path: string; model: string | null; results: string | null }]; res: UserReadIn };
  "user:readCase": { req: [req: { path: string; case: string; target: SpectrumTarget; label: string }]; res: CaseRead };
  "user:readResult": { req: [req: { path: string; name: string; component: string }]; res: UserColumnRead };
  "workspace:read": { req: []; res: WorkspaceIn };
  "workspace:save": { req: [edit: WorkspaceEdit]; res: WorkspaceSaveIn };
  "combinations:save": { req: [edit: CombinationEdit]; res: CombinationSaveIn };
  "variants:current": { req: []; res: VariantsIn | null };
  "variants:switch": { req: [index: number, label: string]; res: Done };
  "variants:compare": { req: [index: number, label: string]; res: { ok: true; compare: CompareIn; nodeIds: Record<string, Float64Array> } | { ok: false; reason: string } };
  "compare:readStep": { req: [req: { stage: string; step: number; seq: number }]; res: StepAnswer };
  "compare:close": { req: []; res: void };
  "compare:model": { req: [req: Record<string, never>]; res: CompareModelAnswer };
  "compare:userRead": { req: [req: Record<string, never>]; res: CompareUserAnswer };
  "compare:userReadResult": { req: [req: { name: string; component: string }]; res: { slot: CompareSlotIn | null; read: UserColumnRead } };
  "viewer:currentSet": { req: []; res: OpenSet | null };
  "viewer:requestOpen": { req: [path: string]; res: Done };
  "viewer:offerOpen": { req: [path: string, needs: readonly string[]]; res: Done };
  "viewer:snapshotReply": { req: [id: number, reply: unknown]; res: void };
  "render:job": { req: []; res: RenderJob };
  "render:variant": { req: [row: SpecVariant | null]; res: VariantAnswer };
  "render:png": { req: [name: string, png: Uint8Array]; res: { path: string } };
  "render:capture": { req: [name: string]; res: { path: string; width: number; height: number } };
  "render:done": { req: [result: unknown]; res: void };
  "capture:still": { req: [suffix: string]; res: string | null };
  "capture:done": { req: []; res: void };
  "capture:resize": { req: [width: number, height: number]; res: void };
  "measure:metrics": { req: []; res: { mainMB: number; rendererMB: number; gpuProcessMB: number; gpuDevices: unknown[] } };
  "measure:done": { req: [result: unknown]; res: void };
  "measure:cpu": { req: []; res: number };
  "measure:memory": { req: []; res: { main: number; renderer: number } };
  "materialTest:run": { req: [job: MaterialTestJob, seq: number]; res: MaterialTestAnswer };
  "materialTest:cancel": { req: []; res: void };
  "materialTest:saveCsv": { req: [text: string, name: string]; res: { ok: true; path: string } | { ok: false; cancelled?: true; error: string } };
  "tool:open": { req: [tool: string]; res: { ok: true } | { ok: false; reason: string } };
}

/** What the page tells main and waits for nothing on (`ipcRenderer.send` or `sendSync`). */
export interface Send {
  /** VU12: the path of a File the user dropped, sent synchronously so it is recorded before any open names it */
  "viewer:dropped": [path: string];
  "viewer:subscribe": [channel: PageFeed];
  "viewer:unsubscribe": [channel: PageFeed];
  "viewer:offersReady": [];
}

/** The two page feeds main reloads the page for until it subscribes (main.ts `deliver`). */
export type PageFeed = "open" | "fileChanged";

/** What main tells the page (`webContents.send`). */
export interface Push {
  "viewer:open": [set: OpenSet];
  "viewer:fileChanged": [path: string];
  "viewer:offers": [offers: OfferIn[]];
  "viewer:snapshot": [req: { id: number; image: [number, number] }];
  "viewer:variants": [study: VariantsIn | null];
  "results:scanProgress": [p: ScanProgress];
  "memory:status": [m: MemoryStatus];
  "tool:status": [s: ToolStatus];
  "tool:paused": [paused: boolean];
}

/** The tool windows' channels (src/main/toolPreload.ts); a tool's data answer is a `ToolAnswer`. */
export interface ToolInvoke {
  "tool:request": { req: [q: unknown, seq: number]; res: ToolAnswer };
}

// ---------------------------------------------------------------------------
// Main's side.

/** What `handle` needs of Electron's `ipcMain`. */
export interface MainIpc {
  handle(channel: string, fn: (e: IpcMainInvokeEvent, ...args: unknown[]) => unknown): void;
  on(channel: string, fn: (e: IpcMainEvent, ...args: unknown[]) => void): unknown;
}

/**
 * Answer `channel`: `handle(ipcMain, "results:open")((e, path) => …)`. The channel is settled
 * first, so the handler is checked against that channel's answer with its literals kept (an
 * early `return { ok: false, … }` stays `false`). The arguments arrive as `unknown`: main checks them.
 */
export function handle<C extends keyof Invoke | keyof ToolInvoke>(ipc: MainIpc, channel: C) {
  return (fn: (e: IpcMainInvokeEvent, ...args: unknown[]) => Res<C> | Promise<Res<C>>): void => {
    ipc.handle(channel, fn);
  };
}

/** Hear `channel`. The arguments arrive as `unknown` (main checks them). */
export function listen<C extends keyof Send | "tool:ready">(ipc: MainIpc, channel: C, fn: (e: IpcMainEvent, ...args: unknown[]) => void): void {
  ipc.on(channel, fn);
}

type Res<C> = C extends keyof Invoke ? Invoke[C]["res"] : C extends keyof ToolInvoke ? ToolInvoke[C]["res"] : never;

/** What `push` needs of a page's `webContents`. */
export interface PushTarget {
  send(channel: string, ...args: unknown[]): void;
}

/** Tell a page on `channel`. */
export function push<C extends keyof Push>(to: PushTarget, channel: C, ...args: Push[C]): void {
  to.send(channel, ...args);
}

// ---------------------------------------------------------------------------
// The preload's side.

/** What the preload's helpers need of Electron's `ipcRenderer`. */
export interface RendererIpc {
  invoke(channel: string, ...args: unknown[]): Promise<unknown>;
  send(channel: string, ...args: unknown[]): void;
  sendSync(channel: string, ...args: unknown[]): unknown;
  on(channel: string, fn: (e: IpcRendererEvent, ...args: unknown[]) => void): unknown;
  removeListener(channel: string, fn: (e: IpcRendererEvent, ...args: unknown[]) => void): unknown;
}

/** Ask main on `channel`; the answer is the map's. */
export function invoke<C extends keyof Invoke>(ipc: RendererIpc, channel: C, ...args: Invoke[C]["req"]): Promise<Invoke[C]["res"]> {
  return ipc.invoke(channel, ...args) as Promise<Invoke[C]["res"]>;
}

/** Tell main on `channel`. */
export function tell<C extends keyof Send>(ipc: RendererIpc, channel: C, ...args: Send[C]): void {
  ipc.send(channel, ...args);
}

/** Hear main on `channel`; returns the unsubscribe. */
export function hear<C extends keyof Push>(ipc: RendererIpc, channel: C, cb: (...args: Push[C]) => void): () => void {
  // Main sends what the map says (main.ts `push`), so the arguments are this channel's.
  const handler = (_e: IpcRendererEvent, ...args: unknown[]) => cb(...(args as Push[C]));
  ipc.on(channel, handler);
  return () => {
    ipc.removeListener(channel, handler);
  };
}

/** A request of `channel`, as the preload exposes it. */
type Call<C extends keyof Invoke> = (...args: Invoke[C]["req"]) => Promise<Invoke[C]["res"]>;

/**
 * `window.viewer`, exactly as preload.ts exposes it (`satisfies PreloadApi`). The page's door types
 * (effects.ts `Bridge`, renderer/contour.ts `VariantDoor`, renderer/agentRender.ts `RenderBridge`,
 * …) take it without a cast. `openModel` answers the id of a bulk answer (#3130, ../bulk.ts); the
 * page takes the answer by it.
 */
export interface PreloadApi {
  config: Call<"app:config">;
  openModel(path: string): Promise<number>;
  pathForFile(f: File): string;
  metrics: Call<"measure:metrics">;
  measureDone: Call<"measure:done">;
  captureStill: Call<"capture:still">;
  captureDone: Call<"capture:done">;
  captureResize: Call<"capture:resize">;
  fail: Call<"app:fail">;
  resultsStep: Call<"results:step">;
  onOpen(cb: (set: OpenSet) => void): () => void;
  onFileChanged(cb: (path: string) => void): () => void;
  goToSource: Call<"source:goto">;
  requestOpen(f: File): Promise<Done>;
  openGeometry: Call<"geometry:open">;
  openResults: Call<"results:open">;
  readResultsStep: Call<"results:readStep">;
  readHingeHistory: Call<"results:hingeHistory">;
  scanResults: Call<"results:scan">;
  cancelResultsScan: Call<"results:scanCancel">;
  readUser: Call<"user:read">;
  readUserCase: Call<"user:readCase">;
  readUserResult: Call<"user:readResult">;
  openTool: Call<"tool:open">;
  onToolStatus(cb: (s: ToolStatus) => void): () => void;
  onToolsPaused(cb: (paused: boolean) => void): () => void;
  onMemoryStatus(cb: (m: MemoryStatus) => void): () => void;
  onScanProgress(cb: (p: ScanProgress) => void): () => void;
  measureCpu: Call<"measure:cpu">;
  measureMemory: Call<"measure:memory">;
  renderJob: Call<"render:job">;
  renderPng: Call<"render:png">;
  renderCapture: Call<"render:capture">;
  renderDone: Call<"render:done">;
  renderVariant: Call<"render:variant">;
  onOffers(cb: (offers: OfferIn[]) => void): () => void;
  onSnapshotRequest(cb: (req: { id: number; image: [number, number] }) => void): () => void;
  offerOpen: Call<"viewer:offerOpen">;
  variants: Call<"variants:current">;
  onVariants(cb: (study: VariantsIn | null) => void): () => void;
  variantsSwitch: Call<"variants:switch">;
  variantsCompare: Call<"variants:compare">;
  readCompareStep: Call<"compare:readStep">;
  compareClose: Call<"compare:close">;
  compareModel(): Promise<CompareModelAnswer>;
  compareUserRead(): Promise<CompareUserAnswer>;
  compareUserReadResult: Call<"compare:userReadResult">;
  snapshotReply: Call<"viewer:snapshotReply">;
  workspaceRead: Call<"workspace:read">;
  workspaceSave: Call<"workspace:save">;
  combinationsSave: Call<"combinations:save">;
  particleIndex: Call<"particles:index">;
  readParticleOutput: Call<"particles:readOutput">;
  particleSet: Call<"particles:set">;
  particleHarness: Call<"particles:harness">;
  particleHarnessStill: Call<"particles:harnessStill">;
  particleHarnessDone: Call<"particles:harnessDone">;
  materialTestRun: Call<"materialTest:run">;
  materialTestCancel: Call<"materialTest:cancel">;
  materialTestSaveCsv: Call<"materialTest:saveCsv">;
}
