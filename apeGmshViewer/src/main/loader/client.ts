// Main's side of the loader process (ADR 0137 D2, #3058; the process is ./process.ts).
//
// The process starts on the first read and is kept for the next. A process that exits fails
// every read it had not answered, with the exit code, and the next read starts a new one, so
// a crash in a reader costs one refused open, never the app. No Electron import: main hands
// in how to start the process (`utilityProcess.fork`), and the tests hand in a Node fork.

import type { LoaderOp, LoaderOps, LoaderReply, LoaderRequest } from "./protocol.ts";

/** What the loader needs of a started process: Electron's `UtilityProcess`, or a test's stand-in. */
export interface LoaderChild {
  postMessage(m: LoaderRequest): void;
  onMessage(fn: (m: LoaderReply) => void): void;
  onExit(fn: (code: number) => void): void;
}

interface Pending {
  child: LoaderChild;
  resolve(v: unknown): void;
  reject(err: Error): void;
}

export class Loader {
  private child: LoaderChild | null = null;
  private next = 1;
  private readonly pending = new Map<number, Pending>();

  private readonly spawn: () => LoaderChild;

  constructor(spawn: () => LoaderChild) {
    this.spawn = spawn;
  }

  /** The model and its zones. Rejects with the reader's message when the model cannot be read. */
  model(path: string): Promise<LoaderOps["model"]> {
    return this.call("model", path);
  }

  /** The model's node and element counts. Rejects with the reader's message. */
  modelCounts(path: string): Promise<LoaderOps["modelCounts"]> {
    return this.call("modelCounts", path);
  }

  /** A /geometry sibling. A stopped loader is an `ok: false` answer, as a reader error is. */
  geometry(path: string): Promise<LoaderOps["geometry"]> {
    return this.call("geometry", path).catch((err: Error) => ({ ok: false as const, error: err.message }));
  }

  /** A model's /provenance. A stopped loader is an `ok: false` answer, as a reader error is. */
  provenance(path: string): Promise<LoaderOps["provenance"]> {
    return this.call("provenance", path).catch((err: Error) => ({ ok: false as const, error: err.message }));
  }

  /** Start the process now, if none is running, so the first read does not wait for it to start. */
  warm(): void {
    if (this.child === null) this.start();
  }

  /** How many reads are waiting for an answer. */
  waiting(): number {
    return this.pending.size;
  }

  private call<K extends LoaderOp>(op: K, path: string): Promise<LoaderOps[K]> {
    const child = this.child ?? this.start();
    const id = this.next++;
    return new Promise<LoaderOps[K]>((resolve, reject) => {
      this.pending.set(id, { child, resolve: resolve as (v: unknown) => void, reject });
      child.postMessage({ id, op, path });
    });
  }

  private start(): LoaderChild {
    const child = this.spawn();
    this.child = child;
    child.onMessage((m) => {
      const p = this.pending.get(m.id);
      if (p === undefined || p.child !== child) return;
      this.pending.delete(m.id);
      if (m.ok) p.resolve(m.value);
      else p.reject(new Error(m.error));
    });
    child.onExit((code) => {
      if (this.child === child) this.child = null;
      for (const [id, p] of this.pending) {
        if (p.child !== child) continue;
        this.pending.delete(id);
        p.reject(new Error(`the loader process stopped (exit code ${code}) before it finished reading`));
      }
    });
    return child;
  }
}
