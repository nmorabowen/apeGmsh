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
import { NEUTRAL_TARGET, OPENSEES_TARGET, ZONE_FLOOR } from "./reader/read.ts";
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
  /** V2f: replays the set already open at subscription, then every later open; the page must not also load `config().file`. Returns the unsubscribe. */
  onOpen?(cb: (set: OpenSet) => void): (() => void) | void;
  onFileChanged?(cb: (path: string) => void): (() => void) | void;
  /** V2f: ask main to open a dropped file; main answers through `onOpen` */
  requestOpen?(path: string): Promise<unknown>;
  goToSource?(file: string, line: number): Promise<{ ok: boolean; reason?: string }>;
}

/**
 * A reader refusal, from the message the reader throws (`<zone>_schema_version
 * <version>: <reason>`): the zone, its version, the range this app accepts
 * (the zone's floor through the current major, from the reader's one floor
 * table; a later minor is read with a banner, so it is not a refusal) and
 * whether the file is ahead of this app (a newer major) or below the floor.
 * Any other message is a plain load failure. (#1303 owns the reader's policy;
 * a structured refusal across the bridge would retire this parse.)
 */
export function parseRefusal(error: string): { zone: string; version: string; accepted: string; newer: boolean; reason: string } | null {
  const m = /^(neutral|opensees)_schema_version (\d+)\.(\d+)\.(\d+): (.+)$/.exec(error);
  if (!m) return null;
  const zone = m[1] as keyof typeof ZONE_FLOOR;
  const major = Number(m[2]);
  const t = zone === "neutral" ? NEUTRAL_TARGET : OPENSEES_TARGET;
  const accepted = `${t.major}.${ZONE_FLOOR[zone]} and any later ${t.major}.x`;
  return { zone, version: `${major}.${m[3]}.${m[4]}`, accepted, newer: major > t.major, reason: m[5]! };
}

/** Every BlobRef the state holds (what the BlobStore keeps after a load). */
export function blobRefsOf(s: State): BlobRef[] {
  const out: BlobRef[] = [];
  if (s.mesh) {
    const m = s.mesh;
    out.push(m.linePositions, m.triPositions, m.edgePositions, m.lineElement, m.triElement, m.lineGroup, m.triGroup);
  }
  for (const d of Object.values(s.decls)) if (d.group) out.push(d.group.ids);
  for (const b of s.blocks) out.push(b.connectivity);
  return out;
}

export class Effects {
  private readonly store: Store;
  private readonly blobs: BlobStore;
  private readonly bridge: Bridge;
  /** The latest open wins: a read that finishes after a newer open started is dropped. */
  private openToken = 0;
  private reading = false;
  /** Failed re-reads of the current path after a mid-read change; reset by a successful read or a new path. */
  private retries = 0;
  private retryPath: string | null = null;
  private readonly off: (() => void)[] = [];

  constructor(store: Store, blobs: BlobStore, bridge: Bridge) {
    this.store = store;
    this.blobs = blobs;
    this.bridge = bridge;
    // A file rewritten on disk (D1 rewrites on every run) is read again; one
    // that changes mid-read is re-read when that read ends (see `open`).
    this.off.push(
      this.store.subscribe((s) => {
        const m = s.artifacts.model;
        if (!m || !m.stale || this.reading) return;
        // A change after the retries ran out starts a fresh count.
        if (m.status === "failed" && m.path === this.retryPath && this.retries >= Effects.MAX_RETRIES) this.retries = 0;
        void this.open(m.path);
      }),
    );
  }

  /** Open a model artifact; resolves true when it loaded and is still the latest open. */
  async open(path: string): Promise<boolean> {
    const token = ++this.openToken;
    this.reading = true;
    if (this.retryPath !== path) {
      this.retryPath = path;
      this.retries = 0;
    }
    this.store.dispatch({ type: "fileOpened", artifact: "model", path });
    let loaded = false;
    try {
      const res = await this.bridge.openModel(path);
      if (token !== this.openToken) return false;
      if (!res.ok) {
        const refusal = parseRefusal(res.error);
        if (refusal) this.store.dispatch({ type: "zoneRefused", artifact: "model", ...refusal });
        this.store.dispatch({ type: "fileFailed", artifact: "model", path, error: res.error });
        return false;
      }
      const load = loadModel(res.model, this.blobs);
      for (const w of load.info.warnings) console.warn(w);
      // One line per read, so a double load shows in the Electron log.
      console.info(`apeGmshViewer: loaded ${path} (read ${Math.round(load.info.readMs)} ms)`);
      this.store.dispatch({ type: "fileLoaded", artifact: "model", load });
      this.blobs.retain(blobRefsOf(this.store.get()));
      loaded = true;
      return true;
    } catch (err) {
      if (token !== this.openToken) return false;
      const error = err instanceof Error ? err.message : String(err);
      this.store.dispatch({ type: "fileFailed", artifact: "model", path, error });
      return false;
    } finally {
      if (token === this.openToken) {
        this.reading = false;
        if (loaded) this.retries = 0;
        // A fileChanged that arrived during this read: read once more. After
        // a failed read too (it may have hit a half-written file), but at
        // most MAX_RETRIES times in a row, so a broken file does not loop.
        const m = this.store.get().artifacts.model;
        if (m && m.path === path && m.stale) {
          if (loaded) void this.open(path);
          else if (this.retries < Effects.MAX_RETRIES) {
            this.retries++;
            void this.open(path);
          } else console.warn(`apeGmshViewer: ${path} changed during ${Effects.MAX_RETRIES} failed reads; not read again until it changes`);
        }
      }
    }
  }

  /** Failed re-reads allowed in a row after a mid-read change. */
  static readonly MAX_RETRIES = 3;

  /**
   * The files a V2f `onOpen` names: the model is read; the siblings are
   * recorded until their readers exist (V2f phase 2); a kind the set does not
   * name is closed.
   */
  openSet(set: OpenSet): void {
    for (const k of ["geometry", "results"] as const) {
      if (set[k]) this.store.dispatch({ type: "fileOpened", artifact: k, path: set[k]! });
      else this.store.dispatch({ type: "fileClosed", artifact: k });
    }
    if (set.model) void this.open(set.model);
    else this.store.dispatch({ type: "fileClosed", artifact: "model" });
  }

  /** Detach everything `attach` and the bridge subscriptions registered. */
  dispose(): void {
    for (const f of this.off.splice(0)) f();
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
    this.off.push(() => {
      window.removeEventListener("dragover", onDragOver);
      window.removeEventListener("drop", onDrop);
    });
    const keep = (u: (() => void) | void) => {
      if (typeof u === "function") this.off.push(u);
    };
    if (typeof this.bridge.onOpen === "function") keep(this.bridge.onOpen((set) => this.openSet(set)));
    if (typeof this.bridge.onFileChanged === "function") {
      keep(this.bridge.onFileChanged((path) => this.store.dispatch({ type: "fileChanged", path })));
    }
    return () => this.dispose();
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
