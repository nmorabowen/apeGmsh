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
// The `<stem>.user.h5` is a member like the others (ADR 0130, slice VU3a). Its
// writers replace it through `<stem>.user.h5.<host>-<pid>-<random>.partial`,
// which is no candidate, so only the replace is seen. The user file appearing,
// being replaced or being deleted is a `changed(<its path>)`, never a
// `reopened`: whether the path exists says which. Its staleness depends on
// the model and the results file too, so a `changed` of either is a reason to
// read it again (`user:read`, main.ts).
//
// The `<stem>.workspace.h5` (ADR 0132 D1) is a member by the same rule: its
// writers (the app's ./workspace.ts, and Python) replace it through a
// `.partial` that is no candidate, and its appearing, being replaced or being
// deleted is a `changed(<its path>)`, never a `reopened`.
//
// No Electron import: the tests run in Node.

import { existsSync, statSync, watch, type FSWatcher } from "node:fs";
import { basename, dirname } from "node:path";
import { candidates, FOLD_CASE, inSet, samePath, sameSet, stemSet, type Candidates, type OpenSet } from "./pairing.ts";

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
    const watched = Object.values(this.cands);
    if (this.cands.results.toLowerCase().endsWith(".results.h5")) {
      watched.push(`${this.cands.results.slice(0, -".results.h5".length)}.ladruno`);
    }
    this.paths = new Map(watched.map((p) => [key(basename(p)), p]));
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
    const prev = this.set;
    this.set = next;
    // Model, geometry or results moved: the window re-reads the set.
    if (!sameSet({ ...next, user: null, workspace: null }, { ...prev, user: null, workspace: null })) {
      this.sink.reopened(next);
      return;
    }
    // Only the user and workspace slots can differ here, and either file
    // appearing, being replaced or being deleted is never a reopen (which
    // re-reads model, geometry and results and clears the step). Each path
    // reports its own change: a model rewritten while the user file appears is
    // two changes, never one (review of #2729: the re-run order emit, run,
    // write_user_file).
    const members = [[this.cands.user, next.user, prev.user], [this.cands.workspace, next.workspace, prev.workspace]] as const;
    if (inSet(next, path) || members.some(([own]) => samePath(path, own))) this.sink.changed(path);
    for (const [own, now, before] of members) {
      if (samePath(path, own) || samePath(now, before) || this.polls.has(own)) continue;
      // The slot moved with no poll of its own running (its event was missed
      // or merged): report it here, once, under the signature it now has.
      const sig = signature(own);
      if (this.reported.get(own) !== sig) {
        this.reported.set(own, sig);
        this.sink.changed(own);
      }
    }
  }
}
