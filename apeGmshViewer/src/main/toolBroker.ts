// The data broker (ADR 0137 D10, H2b): what a tool asks main for, read once however many ask.
//
// - Main checks the sender before this (a window main made for a tool, ./toolWindows.ts) and
//   names the file: the model window's results file. A tool never names a path.
// - `checkToolRequest` refuses anything but a well-formed request.
// - Requests for the same data while one is being read share that read.
// - Newest wins, per tool: a request answered after the same tool asked again is answered
//   `superseded`, so a tool never draws an answer it no longer wants.
// - The read itself runs in the loader process (./loader/client.ts), like every results read.
//
// No Electron import: the read and the senders are handed in, so a test drives this.

import type { ToolAnswer, ToolRequest } from "../tools/contract.ts";
import type { NodeHistoryAnswer, NodeHistoryRequest } from "./results.ts";

/** The request if it is well formed, else why not. */
export function checkToolRequest(q: unknown): { ok: true; q: ToolRequest } | { ok: false; error: string } {
  const r = q as Partial<ToolRequest> | null;
  if (r === null || typeof r !== "object" || r.kind !== "nodeHistory") return { ok: false, error: `tool:request: unknown request ${JSON.stringify(r?.kind ?? r)}` };
  const keys = Object.keys(r).sort().join(",");
  if (keys !== "component,kind,node,stage") return { ok: false, error: `tool:request: a nodeHistory request takes kind, stage, node and component; got ${keys}` };
  if (typeof r.stage !== "string" || typeof r.component !== "string" || typeof r.node !== "number" || !Number.isInteger(r.node)) {
    return { ok: false, error: "tool:request: stage and component are strings and node an integer" };
  }
  return { ok: true, q: { kind: "nodeHistory", stage: r.stage, node: r.node, component: r.component } };
}

export class ToolBroker {
  private readonly read: (req: NodeHistoryRequest) => Promise<NodeHistoryAnswer>;
  /** the reads in flight, by what they read */
  private readonly inFlight = new Map<string, Promise<NodeHistoryAnswer>>();
  /** each tool's newest request number */
  private readonly newest = new WeakMap<object, number>();
  /** how many reads were made (tests and the harness read it) */
  reads = 0;

  constructor(read: (req: NodeHistoryRequest) => Promise<NodeHistoryAnswer>) {
    this.read = read;
  }

  /** Answer tool `sender`'s request number `seq` from results file `path`. */
  async request(sender: object, path: string, q: ToolRequest, seq: number): Promise<ToolAnswer> {
    this.newest.set(sender, Math.max(seq, this.newest.get(sender) ?? -Infinity));
    const key = JSON.stringify([path, q.stage, q.node, q.component]);
    let pending = this.inFlight.get(key);
    if (pending === undefined) {
      this.reads++;
      pending = this.read({ path, stage: q.stage, node: q.node, component: q.component });
      this.inFlight.set(key, pending);
      void pending.finally(() => {
        if (this.inFlight.get(key) === pending) this.inFlight.delete(key);
      });
    }
    const a = await pending;
    if ((this.newest.get(sender) ?? seq) > seq) return { ok: false, superseded: true, error: "superseded by a newer request from this tool" };
    return a.ok ? { ok: true, kind: "nodeHistory", stage: a.stage, node: a.node, component: a.component, values: a.values } : { ok: false, error: a.error };
  }
}
