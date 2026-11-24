// Main's side of the loader process (ADR 0137 D2, #3058; the process is ./process.ts).
//
// The process starts with the window (`warm`) or on the first read, and is kept for the next.
// A process that exits fails every read it had not answered, with the exit code, and the next
// read starts a new one, so a crash in a reader costs the reads in flight, never the app. The
// results files the loader had open are opened again in the new process before any other read
// of them, so a step read after a crash finds its file indexed as before.
//
// No Electron import: main hands in how to start the process (./fork.ts), and the tests hand
// in a Node fork or a stand-in.

import type {
  HingeHistoryAnswer,
  HingeHistoryRequest,
  NodeHistoryAnswer,
  NodeHistoryRequest,
  ParticleIndexAnswer,
  ParticleOutputAnswer,
  ParticleOutputRequest,
  ResultsOpenAnswer,
  ScanAnswer,
  ScanProgress,
  ScanRequest,
  StepAnswer,
  StepRequest,
} from "../results.ts";
import type { LoaderMessage, LoaderOp, LoaderOps, LoaderRequest, ResultsCall, ResultsCalls, ResultsHostName, ScanOptions } from "./protocol.ts";

/** What the loader needs of a started process: Electron's `UtilityProcess`, or a test's stand-in. */
export interface LoaderChild {
  postMessage(m: LoaderRequest): void;
  onMessage(fn: (m: LoaderMessage) => void): void;
  onExit(fn: (code: number) => void): void;
}

/**
 * A results host in the loader, as main's doors call it: `ResultsHost`'s calls, each answered
 * asynchronously. Every answer is an answer: a stopped loader is `ok: false` with why, as a
 * reader error is, so a door never rejects.
 */
export interface RemoteResults {
  open(path: string): Promise<ResultsOpenAnswer>;
  readStep(req: StepRequest): Promise<StepAnswer>;
  readHingeHistory(req: HingeHistoryRequest): Promise<HingeHistoryAnswer>;
  readNodeHistory(req: NodeHistoryRequest): Promise<NodeHistoryAnswer>;
  scan(req: ScanRequest, progress: (p: ScanProgress) => void, opts?: ScanOptions): Promise<ScanAnswer>;
  cancelScan(scanId: number): void;
  particleIndex(path: string): Promise<ParticleIndexAnswer>;
  readParticleOutput(req: ParticleOutputRequest): Promise<ParticleOutputAnswer>;
  close(): void;
}

interface Pending {
  child: LoaderChild;
  resolve(v: unknown): void;
  reject(err: Error): void;
}

const stopped = (code: number) => `the loader process stopped (exit code ${code}) before it finished reading`;
const why = (err: unknown) => (err instanceof Error ? err.message : String(err));

export class Loader {
  private child: LoaderChild | null = null;
  private next = 1;
  private readonly pending = new Map<number, Pending>();
  private readonly spawn: () => LoaderChild;
  /** The results file each host has open, so a new process opens it again (none after `close`). */
  private readonly opened = new Map<ResultsHostName, string>();
  /** The re-opens a new process runs before any other results read reaches it. */
  private replay: { child: LoaderChild; done: Promise<unknown> } | null = null;
  /** The progress listener of each running scan, by host and scan id. */
  private readonly progress = new Map<string, (p: ScanProgress) => void>();

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

  /** The model's cell count, read from dataset shapes only. Rejects with the reader's message. */
  cells(path: string): Promise<number> {
    return this.call("cells", path);
  }

  /** A /geometry sibling. A stopped loader is an `ok: false` answer, as a reader error is. */
  geometry(path: string): Promise<LoaderOps["geometry"]> {
    return this.call("geometry", path).catch((err: Error) => ({ ok: false as const, error: err.message }));
  }

  /** A model's /provenance. A stopped loader is an `ok: false` answer, as a reader error is. */
  provenance(path: string): Promise<LoaderOps["provenance"]> {
    return this.call("provenance", path).catch((err: Error) => ({ ok: false as const, error: err.message }));
  }

  /** A `<stem>.particles.h5`. A stopped loader is an `ok: false` answer, as a reader error is. */
  particleSet(path: string): Promise<LoaderOps["particleSet"]> {
    return this.call("particleSet", path).catch((err: Error) => ({ ok: false as const, error: err.message }));
  }

  /** One of the loader's two results hosts. */
  results(host: ResultsHostName): RemoteResults {
    const r = <K extends ResultsCall>(call: K, ...args: ResultsCalls[K]["args"]) => this.resultsCall(host, call, args);
    return {
      open: (path) => {
        // Recorded after the call has started a process (if none ran), so that process's replay does not open it twice.
        const answer = r("open", path);
        this.opened.set(host, path);
        return answer.catch((err) => ({ ok: false as const, error: why(err) }));
      },
      readStep: (req) => r("readStep", req).catch((err) => ({ ok: false as const, seq: req.seq, error: why(err) })),
      readHingeHistory: (req) => r("readHingeHistory", req).catch((err) => ({ ok: false as const, seq: req.seq, error: why(err) })),
      readNodeHistory: (req) => r("readNodeHistory", req).catch((err) => ({ ok: false as const, error: why(err) })),
      scan: (req, progress, opts = {}) => {
        const key = `${host}:${req.scanId}`;
        this.progress.set(key, progress);
        return r("scan", req, opts)
          .catch((err) => ({ ok: false as const, scanId: req.scanId, error: why(err) }))
          .finally(() => {
            if (this.progress.get(key) === progress) this.progress.delete(key);
          });
      },
      // A cancel of a scan in a stopped process has nothing left to stop.
      cancelScan: (scanId) => void r("cancelScan", scanId).catch(() => undefined),
      particleIndex: (path) => r("particleIndex", path).catch((err) => ({ ok: false as const, error: why(err) })),
      readParticleOutput: (req) => r("readParticleOutput", req).catch((err) => ({ ok: false as const, seq: req.seq, error: why(err) })),
      close: () => {
        this.opened.delete(host);
        // No process, nothing open: a new one opens only what it is asked to.
        if (this.child !== null) void r("close").catch(() => undefined);
      },
    };
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
    return this.send(this.child ?? this.start(), { op, path }) as Promise<LoaderOps[K]>;
  }

  private async resultsCall<K extends ResultsCall>(host: ResultsHostName, call: K, args: unknown[]): Promise<ResultsCalls[K]["answer"]> {
    const child = this.child ?? this.start();
    // An `open` replaces what a replay opens; any other call waits for the replay to finish.
    if (call !== "open" && this.replay?.child === child) await this.replay.done;
    return this.send(child, { op: "results", host, call, args }) as Promise<ResultsCalls[K]["answer"]>;
  }

  private send(child: LoaderChild, req: { op: LoaderOp; path: string } | { op: "results"; host: ResultsHostName; call: ResultsCall; args: unknown[] }): Promise<unknown> {
    const id = this.next++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { child, resolve, reject });
      child.postMessage({ ...req, id });
    });
  }

  private start(): LoaderChild {
    const child = this.spawn();
    this.child = child;
    child.onMessage((m) => {
      if ("event" in m) {
        this.progress.get(`${m.host}:${m.progress.scanId}`)?.(m.progress);
        return;
      }
      const p = this.pending.get(m.id);
      if (p === undefined || p.child !== child) return;
      this.pending.delete(m.id);
      if (m.ok) p.resolve(m.value);
      else p.reject(new Error(m.error));
    });
    child.onExit((code) => {
      if (this.child === child) this.child = null;
      if (this.replay?.child === child) this.replay = null;
      for (const [id, p] of this.pending) {
        if (p.child !== child) continue;
        this.pending.delete(id);
        p.reject(new Error(stopped(code)));
      }
    });
    // A new process after a stopped one: open again the results files the old one had open.
    const again = [...this.opened].map(([host, path]) => this.send(child, { op: "results", host, call: "open", args: [path] }).catch(() => undefined));
    this.replay = again.length ? { child, done: Promise.all(again) } : null;
    return child;
  }
}
