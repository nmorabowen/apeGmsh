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
