// Watch an open set (ADR 0112 D1: every run rewrites the files).
//
// The directory is watched, not the files, so a writer that truncates, or
// one that deletes and recreates, is seen either way. An event only starts a
// poll: the change is reported once the file's size and mtime have held still
// for `stableMs` (the writer has closed it), never mid-write. A poll that
// settles on the signature last reported is dropped, so a burst of events
// for one write reports once.
//
// What settles is reported in one of two ways:
//   reopened(set)  the set itself changed: a sibling appeared or vanished
//   changed(path)  a file of the unchanged set was rewritten
//
// No Electron import: the tests run in Node.

import { existsSync, statSync, watch, type FSWatcher } from "node:fs";
import { basename, dirname } from "node:path";
import { candidates, FOLD_CASE, inSet, sameSet, stemSet, type Candidates, type OpenSet } from "./pairing.ts";

export interface WatchSink {
  changed(path: string): void;
  reopened(set: OpenSet): void;
  error(message: string): void;
}

export interface WatchOptions {
  /** How long size and mtime must hold still before a change is reported. */
  stableMs: number;
  pollMs: number;
}

export const DEFAULT_WATCH: WatchOptions = { stableMs: 300, pollMs: 50 };

const key = (name: string) => (FOLD_CASE ? name.toLowerCase() : name);

/** `size:mtime`, `missing`, or `error:<code>` (never settles). */
function signature(path: string): string {
  try {
    const s = statSync(path);
    return `${s.size}:${s.mtimeMs}`;
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    return code === "ENOENT" ? "missing" : `error:${code ?? "unknown"}`;
  }
}

export class SetWatcher {
  private readonly dir: string;
  private readonly cands: Candidates;
  private readonly paths: Map<string, string>; // basename key -> candidate path
  private readonly reported = new Map<string, string>(); // path -> last signature
  private readonly polls = new Map<string, ReturnType<typeof setInterval>>();
  private readonly sink: WatchSink;
  private readonly opts: WatchOptions;
  private fsw: FSWatcher | null = null;
  set: OpenSet;

  /** `opened` is the file the user opened; its set is read now. */
  constructor(opened: string, sink: WatchSink, opts: WatchOptions = DEFAULT_WATCH) {
    this.sink = sink;
    this.opts = opts;
    this.cands = candidates(opened);
    this.dir = dirname(opened);
    this.paths = new Map(Object.values(this.cands).map((p) => [key(basename(p)), p]));
    for (const p of this.paths.values()) this.reported.set(p, signature(p));
    this.set = this.current();
  }

  private current(): OpenSet {
    return stemSet(this.cands, existsSync);
  }

  start(): this {
    this.fsw = watch(this.dir, (_event, filename) => {
      if (filename === null) {
        // Some platforms omit the name: check every candidate.
        for (const p of this.paths.values()) this.poll(p);
        return;
      }
      const p = this.paths.get(key(basename(filename.toString())));
      if (p) this.poll(p);
    });
    this.fsw.on("error", (err) => this.sink.error(`watching ${this.dir}: ${err.message}`));
    return this;
  }

  close(): void {
    this.fsw?.close();
    this.fsw = null;
    for (const t of this.polls.values()) clearInterval(t);
    this.polls.clear();
  }

  private poll(path: string): void {
    if (this.polls.has(path)) return; // the running poll sees this change too
    let last = signature(path);
    let since = Date.now();
    const timer = setInterval(() => {
      const now = signature(path);
      if (now !== last || now.startsWith("error:")) {
        last = now;
        since = Date.now();
        return;
      }
      if (Date.now() - since < this.opts.stableMs) return;
      clearInterval(timer);
      this.polls.delete(path);
      this.settled(path, now);
    }, this.opts.pollMs);
    this.polls.set(path, timer);
  }

  private settled(path: string, sig: string): void {
    if (this.reported.get(path) === sig) return;
    this.reported.set(path, sig);
    const next = this.current();
    if (!sameSet(next, this.set)) {
      this.set = next;
      this.sink.reopened(next);
    } else if (inSet(next, path)) {
      this.sink.changed(path);
    }
  }
}
