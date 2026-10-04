// Every side effect of the renderer (D6, decision 15): opening a file
// through the main-process bridge, the drop target, the open and
// file-changed notifications V2f adds to `window.viewer`, go-to-source, and
// releasing blobs the state no longer holds. Nothing here renders, and
// nothing outside this module calls the bridge for a file.
//
// V2f (src/main, preload.ts) adds `onOpen`, `onFileChanged` and
// `goToSource` to the bridge; each is feature-detected here, so this module
// works before and after that PR.

import type { ModelFile } from "./model/types.ts";
import { NEUTRAL_TARGET, OPENSEES_TARGET } from "./reader/read.ts";
import type { BlobStore } from "./state/blobs.ts";
import { loadModel } from "./state/load.ts";
import type { Store } from "./state/store.ts";
import type { ArtifactKind, BlobRef, State } from "./state/types.ts";

export interface OpenSet {
  model: string | null;
  geometry: string | null;
  results: string | null;
}

/** `window.viewer`, as preload.ts exposes it; the optional members arrive with V2f. */
export interface Bridge {
  config(): Promise<{
    mode: "view" | "measure" | "capture";
    file: string | null;
    t0: number;
    appReadyMs: number;
    configMs: number;
    pick: string | null;
  }>;
  openModel(path: string): Promise<{ ok: true; model: ModelFile } | { ok: false; error: string }>;
  pathForFile(f: File): string;
  metrics(): Promise<{ mainMB: number; rendererMB: number; gpuProcessMB: number; gpuDevices: unknown[] }>;
  measureDone(result: unknown): Promise<void>;
  captureStill(suffix: string): Promise<string | null>;
  captureDone(): Promise<void>;
  fail(message: string): Promise<void>;
  /** V2f: replays the set already open at subscription, then every later open; the page must not also load `config().file` */
  onOpen?(cb: (set: OpenSet) => void): void;
  onFileChanged?(cb: (path: string) => void): void;
  /** V2f: ask main to open a dropped file; main answers through `onOpen` */
  requestOpen?(path: string): Promise<unknown>;
  goToSource?(file: string, line: number): Promise<{ ok: boolean; reason?: string }>;
}

/**
 * A reader refusal, from the message the reader throws (`<zone>_schema_version
 * <version>: <reason>`): the zone, its version and the window this app reads.
 * Any other message is a plain load failure. (#1303 owns the reader's policy;
 * a structured refusal across the bridge would retire this parse.)
 */
export function parseRefusal(error: string): { zone: string; version: string; window: string; reason: string } | null {
  const m = /^(neutral|opensees)_schema_version (\d+\.\d+\.\d+): (.+)$/.exec(error);
  if (!m) return null;
  const t = m[1] === "neutral" ? NEUTRAL_TARGET : OPENSEES_TARGET;
  return { zone: m[1]!, version: m[2]!, window: `${t.major}.${t.minor - 1}-${t.major}.${t.minor}`, reason: m[3]! };
}

/** Every BlobRef the state holds (what the BlobStore keeps after a load). */
export function blobRefsOf(s: State): BlobRef[] {
  const out: BlobRef[] = [];
  if (s.mesh) {
    const m = s.mesh;
    out.push(m.linePositions, m.triPositions, m.edgePositions, m.lineElement, m.triElement, m.lineGroup, m.triGroup);
  }
  for (const d of Object.values(s.decls)) if (d.group) out.push(d.group.ids);
  return out;
}

export class Effects {
  private readonly store: Store;
  private readonly blobs: BlobStore;
  private readonly bridge: Bridge;
  private loading: string | null = null;

  constructor(store: Store, blobs: BlobStore, bridge: Bridge) {
    this.store = store;
    this.blobs = blobs;
    this.bridge = bridge;
  }

  /** Open a model artifact; resolves true when it loaded. */
  async open(path: string): Promise<boolean> {
    if (this.loading === path) return false;
    this.loading = path;
    try {
      this.store.dispatch({ type: "fileOpened", artifact: "model", path });
      const res = await this.bridge.openModel(path);
      if (!res.ok) {
        const refusal = parseRefusal(res.error);
        if (refusal) this.store.dispatch({ type: "zoneRefused", artifact: "model", ...refusal });
        this.store.dispatch({ type: "fileFailed", artifact: "model", path, error: res.error });
        return false;
      }
      const load = loadModel(res.model, this.blobs);
      for (const w of load.info.warnings) console.warn(w);
      this.store.dispatch({ type: "fileLoaded", artifact: "model", load });
      this.blobs.retain(blobRefsOf(this.store.get()));
      return true;
    } catch (err) {
      const error = err instanceof Error ? err.message : String(err);
      this.store.dispatch({ type: "fileFailed", artifact: "model", path, error });
      return false;
    } finally {
      this.loading = null;
    }
  }

  /** The files a V2f `onOpen` names: the model is read; the siblings are recorded until their readers exist (V2f phase 2). */
  openSet(set: OpenSet): void {
    for (const k of ["geometry", "results"] as const) {
      if (set[k]) this.store.dispatch({ type: "fileOpened", artifact: k, path: set[k]! });
    }
    if (set.model) void this.open(set.model);
  }

  /** Jump to the user code that declared something; without V2f it says so. */
  goToSource(file: string, line: number): Promise<{ ok: boolean; reason?: string }> {
    if (typeof this.bridge.goToSource !== "function") {
      return Promise.resolve({ ok: false, reason: "go-to-source is not available in this build (it arrives with V2f)" });
    }
    return this.bridge.goToSource(file, line);
  }

  /** Whether main feeds the open set through `onOpen` (V2f); then the page never loads `config().file` itself. */
  hasOpenFeed(): boolean {
    return typeof this.bridge.onOpen === "function";
  }

  /** A file the user dropped: main decides what it is when it can (V2f `requestOpen`), else it is read as the model. */
  openDropped(path: string): void {
    if (typeof this.bridge.requestOpen === "function") void this.bridge.requestOpen(path);
    else void this.open(path);
  }

  /** Attach the window and bridge listeners; the returned function detaches them. */
  attach(): () => void {
    const onDragOver = (e: DragEvent) => e.preventDefault();
    const onDrop = (e: DragEvent) => {
      e.preventDefault();
      const f = e.dataTransfer?.files[0];
      if (f) this.openDropped(this.bridge.pathForFile(f));
    };
    window.addEventListener("dragover", onDragOver);
    window.addEventListener("drop", onDrop);
    if (typeof this.bridge.onOpen === "function") this.bridge.onOpen((set) => this.openSet(set));
    if (typeof this.bridge.onFileChanged === "function") {
      this.bridge.onFileChanged((path) => this.store.dispatch({ type: "fileChanged", path }));
    }
    // A file rewritten on disk (D1 rewrites on every run) is read again.
    const unsubscribe = this.store.subscribe((s) => {
      const m = s.artifacts.model;
      if (m && m.stale && this.loading === null) void this.open(m.path);
    });
    return () => {
      window.removeEventListener("dragover", onDragOver);
      window.removeEventListener("drop", onDrop);
      unsubscribe();
    };
  }

  /** The artifact kinds whose path is known (for the header). */
  static siblings(s: State): { kind: ArtifactKind; name: string; status: string }[] {
    const out: { kind: ArtifactKind; name: string; status: string }[] = [];
    for (const k of ["geometry", "results"] as const) {
      const a = s.artifacts[k];
      if (a) out.push({ kind: k, name: a.path.split(/[\\/]/).pop() ?? a.path, status: a.status });
    }
    return out;
  }
}
