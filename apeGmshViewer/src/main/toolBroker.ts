// The data broker (ADR 0137 D10, H2b, H3): what a tool asks main for, read once however many ask.
//
// - Main checks the sender before this (a window main made for a tool, ./toolWindows.ts) and
//   names the file: the model window's results file. A tool never names a path.
// - `checkToolRequest` refuses anything but a well-formed request.
// - Requests for the same data while one is being read share that read.
// - Newest wins, per tile and per kind of request: a request answered after the same tile asked
//   again for the same kind is answered `superseded`, so a tool never draws an answer it no longer
//   wants. Another kind, or another tile of the same window (H4), is another stream.
// - The reads run in the loader process (./loader/client.ts), like every results read; a step
//   range reads through the shared step cache (./results.ts `readStep`, H3).
//
// No Electron import: the read and the senders are handed in, so a test drives this.

import type { ToolAnswer, ToolRequest } from "../tools/contract.ts";

const isInt = (v: unknown): v is number => typeof v === "number" && Number.isInteger(v);

/** The request if it is well formed, else why not. */
export function checkToolRequest(q: unknown): { ok: true; q: ToolRequest } | { ok: false; error: string } {
  const r = q as Record<string, unknown> | null;
  if (r === null || typeof r !== "object") return { ok: false, error: `tool:request: unknown request ${JSON.stringify(r)}` };
  const keys = Object.keys(r).sort().join(",");
  if (typeof r["stage"] !== "string" || typeof r["component"] !== "string") return { ok: false, error: "tool:request: stage and component are strings" };
  if (r["kind"] === "nodeHistory") {
    if (keys !== "component,kind,node,stage") return { ok: false, error: `tool:request: a nodeHistory request takes kind, stage, node and component; got ${keys}` };
    if (!isInt(r["node"])) return { ok: false, error: "tool:request: node is an integer" };
    return { ok: true, q: { kind: "nodeHistory", stage: r["stage"], node: r["node"], component: r["component"] } };
  }
  if (r["kind"] === "stepRange") {
    if (keys !== "component,kind,stage,step") return { ok: false, error: `tool:request: a stepRange request takes kind, stage, step and component; got ${keys}` };
    if (!isInt(r["step"]) || r["step"] < 0) return { ok: false, error: "tool:request: step is a non-negative integer" };
    return { ok: true, q: { kind: "stepRange", stage: r["stage"], step: r["step"], component: r["component"] } };
  }
  return { ok: false, error: `tool:request: unknown request ${JSON.stringify(r["kind"])}` };
}

export class ToolBroker {
  private readonly read: (path: string, q: ToolRequest) => Promise<ToolAnswer>;
  /** the reads in flight, by what they read */
  private readonly inFlight = new Map<string, Promise<ToolAnswer>>();
  /** each tool window's newest request number, by tile and kind */
  private readonly newest = new WeakMap<object, Map<string, number>>();
  /** how many reads were made (tests and the harness read it) */
  reads = 0;

  constructor(read: (path: string, q: ToolRequest) => Promise<ToolAnswer>) {
    this.read = read;
  }

  /** Answer request number `seq` of tile `tile` in tool window `sender` from results file `path`. */
  async request(sender: object, path: string, q: ToolRequest, seq: number, tile = 0): Promise<ToolAnswer> {
    let byKind = this.newest.get(sender);
    if (byKind === undefined) this.newest.set(sender, (byKind = new Map()));
    const stream = `${tile}:${q.kind}`;
    byKind.set(stream, Math.max(seq, byKind.get(stream) ?? -Infinity));
    const key = JSON.stringify([path, q]);
    let pending = this.inFlight.get(key);
    if (pending === undefined) {
      this.reads++;
      pending = this.read(path, q);
      this.inFlight.set(key, pending);
      void pending.finally(() => {
        if (this.inFlight.get(key) === pending) this.inFlight.delete(key);
      });
    }
    const a = await pending;
    if ((byKind.get(stream) ?? seq) > seq) return { ok: false, superseded: true, error: "superseded by a newer request from this tool" };
    return a;
  }
}
