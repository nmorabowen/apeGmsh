// The open set of the view window, and what follows from opening a file.
//
// main.ts owns the window and the IPC; this class owns the decisions, so
// they are tested in Node with a fake watcher and a fake disk:
//   open(path)            a double-click, a second instance, requestOpen: pair
//                         the file, watch the set, tell the renderer
//   rendererOpened(path)  the renderer read a model on its own (a drop on the
//                         P0 page): the set follows what is on screen, and a
//                         file that cannot be paired ends the old set rather
//                         than leaving it watched (Fable, #1310 finding 2)

import { resolve } from "node:path";
import type { VariantsIn } from "../state/types.ts";
import { rowOf, type VariantRow, type VariantsRead } from "../reader/variants.ts";
import { pairSet, samePath, type OpenSet } from "./pairing.ts";
import type { WatchSink } from "./watch.ts";

export interface SessionWatcher {
  readonly set: OpenSet;
  close(): void;
}

export interface SessionDeps {
  exists(path: string): boolean;
  /** A started watcher of the set `opened` belongs to. */
  watch(opened: string, sink: WatchSink): SessionWatcher;
  deliver(channel: "open" | "fileChanged", payload: unknown): void;
  /** A loud error the user sees. */
  fail(message: string): void;
  /** A note for the log only (the renderer already shows the cause). */
  note(message: string): void;
}

export class OpenSession {
  private readonly deps: SessionDeps;
  private watcher: SessionWatcher | null = null;
  set: OpenSet | null = null;

  constructor(deps: SessionDeps) {
    this.deps = deps;
  }

  /** Open the set of `path`; false, after a loud error, when it cannot be paired. */
  open(path: string, notify: boolean): boolean {
    const abs = resolve(path);
    let set: OpenSet;
    try {
      set = pairSet(abs, this.deps.exists);
    } catch (err) {
      this.deps.fail(err instanceof Error ? err.message : String(err));
      return false;
    }
    this.adopt(abs, set);
    if (notify) this.deps.deliver("open", set);
    return true;
  }

  /** The renderer is reading `path` itself; make the open set match it. */
  rendererOpened(path: string): void {
    const abs = resolve(path);
    if (this.set && samePath(abs, this.set.model)) return;
    let set: OpenSet;
    try {
      set = pairSet(abs, this.deps.exists);
    } catch (err) {
      // The window now shows (or fails to show) a file outside any set: stop
      // watching the old one, so its rewrite cannot reload the page onto it.
      this.close();
      this.deps.note(`not watching ${abs}: ${err instanceof Error ? err.message : String(err)}`);
      return;
    }
    this.adopt(abs, set);
  }

  close(): void {
    this.watcher?.close();
    this.watcher = null;
    this.set = null;
  }

  private adopt(opened: string, set: OpenSet): void {
    this.watcher?.close();
    const watcher = this.deps.watch(opened, {
      changed: (p) => {
        if (this.watcher === watcher) this.deps.deliver("fileChanged", p);
      },
      reopened: (next) => {
        if (this.watcher !== watcher) return; // a closed watcher's late report
        this.set = next;
        this.deps.deliver("open", next);
      },
      error: (m) => this.deps.fail(m),
    });
    this.watcher = watcher;
    this.set = set;
  }
}

// ---------------------------------------------------------------------------
// VU10 (ADR 0132 D9, under owner ruling option A, #2753): the study main holds.
// Main reads a manifest only from a path the user gave (a drop, argv, a
// double-click, a render spec), keeps the rows that pass with their resolved
// paths, and re-reads the same path when it changes. The page names a row by
// index and label, never by path (`row`). The compare slot holds at most one
// row's set, read-only, apart from the window set (`OpenSession`): nothing
// here adopts it, watches it or writes into it.
// ---------------------------------------------------------------------------

export interface StudyDeps {
  /** reader/variants.ts `readVariants` over the disk */
  read(manifest: string): VariantsRead;
  /** watch the manifest; `changed` is called when it is rewritten */
  watch(manifest: string, changed: () => void): { close(): void };
  /** a note for the log */
  note(message: string): void;
  /** reader/variants.ts `resolveRow` over the disk now: the file to open, or why the row no longer opens */
  recheck(manifest: string, row: VariantRow): { ok: true; real: string } | { ok: false; code: string; reason: string };
  /** reader/variants.ts `confinedToStudy` over the disk now: null when `path` lies inside the manifest's folder */
  confined(manifest: string, path: string): string | null;
}

export class Study {
  private readonly deps: StudyDeps;
  private watcher: { close(): void } | null = null;
  /** the manifest the user opened, and its last read */
  manifest: string | null = null;
  read: VariantsRead | null = null;
  /**
   * The compare slot: a row, the set paired from it, and the `fem_hash` and
   * `model_hash` its results and model were paired by when main opened it
   * (VU19: the compare doors check the model against them again at each read).
   */
  compare: { row: VariantRow; set: OpenSet; lineage: { femHash: string; modelHash: string } } | null = null;
  /** called after a re-read of the watched manifest */
  onChange: (() => void) | null = null;

  constructor(deps: StudyDeps) {
    this.deps = deps;
  }

  /** Read `manifest` (absolute, a path the user gave). A refused file leaves the study as it was and answers why. */
  open(manifest: string): VariantsRead {
    const r = this.deps.read(manifest);
    if (!r.ok) return r;
    this.close();
    this.manifest = manifest;
    this.read = r;
    this.watcher = this.deps.watch(manifest, () => this.reread());
    return r;
  }

  /**
   * D9.3: re-read the same path; the rows come from that read. A compare row
   * that no longer passes with its label and its file leaves the slot.
   */
  reread(): void {
    if (this.manifest === null) return;
    const r = this.deps.read(this.manifest);
    this.read = r;
    if (!r.ok) this.deps.note(`the variants manifest ${this.manifest} is refused on re-read: ${r.reason}`);
    const c = this.compare;
    if (c) {
      const now = rowOf(r, c.row.index, c.row.label);
      if (!now.ok || !samePath(now.row.resolved, c.row.resolved)) this.compare = null;
    }
    this.onChange?.();
  }

  /**
   * The row the page named, when its label matches (D9.2), and the file to
   * open: the row resolved and confined again now, so a folder swapped for a
   * junction since the read is refused, never followed (INV-5; #2817 review 3).
   */
  row(index: unknown, label: unknown): { ok: true; row: VariantRow; open: string } | { ok: false; reason: string } {
    const r = rowOf(this.read, index, label);
    if (!r.ok) return r;
    const now = this.deps.recheck(this.manifest!, r.row);
    if (!now.ok) return { ok: false, reason: `variant row ${r.row.index} (${JSON.stringify(r.row.label)}) no longer opens (${now.code}): ${now.reason}` };
    return { ok: true, row: r.row, open: now.real };
  }

  /**
   * VU18: `row` as the manifest says it now. An agent run watches nothing, so
   * the render mode asks this for every shot that names a row: the same path
   * is read again first (D9.3), never the rows of the first shot's read, and a
   * file refused on that read refuses the row with its reason.
   */
  rowNow(index: unknown, label: unknown): ReturnType<Study["row"]> {
    if (this.manifest === null) return { ok: false, reason: "no variants manifest is open" };
    this.reread();
    const r = this.read;
    if (r !== null && !r.ok) return { ok: false, reason: `the variants manifest is refused on re-read (${r.code}): ${r.reason}` };
    return this.row(index, label);
  }

  /** Why a file of an opened row's set lies outside the manifest's folder, or null (D9.4: the paired siblings are confined too). */
  confine(path: string | null): string | null {
    if (path === null || this.manifest === null) return null;
    return this.deps.confined(this.manifest, path);
  }

  /** The row whose model file is `model` (as written or resolved), or null. */
  rowOfModel(model: string | null): VariantRow | null {
    if (model === null || !this.read?.ok) return null;
    return this.read.rows.find((r) => samePath(r.resolved, model) || samePath(r.real, model)) ?? null;
  }

  /** What the page sees: never a resolved path. Null with no study open. */
  view(windowModel: string | null): VariantsIn | null {
    if (this.manifest === null || this.read === null) return null;
    const r = this.read;
    const compare = this.compare ? { index: this.compare.row.index, label: this.compare.row.label } : null;
    if (!r.ok) return { manifest: this.manifest, version: "", newer: null, rows: [], refused: [{ index: -1, label: null, code: r.code, reason: r.reason }], active: null, compare };
    return {
      manifest: r.manifest,
      version: r.version,
      newer: r.newer,
      rows: r.rows.map((x) => ({ index: x.index, label: x.label, path: x.path, params: { ...x.params } })),
      refused: r.refused.map((x) => ({ ...x })),
      active: this.rowOfModel(windowModel)?.index ?? null,
      compare,
    };
  }

  close(): void {
    this.watcher?.close();
    this.watcher = null;
    this.manifest = null;
    this.read = null;
    this.compare = null;
  }
}
