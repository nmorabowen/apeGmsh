// A minimal DevTools protocol client over Node's built-in WebSocket.

import { TOOL_URL } from "./analyze.ts";

export { TOOL_URL };

type Handler = (params: Record<string, unknown>) => void;

export class Cdp {
  private ws: WebSocket;
  private nextId = 1;
  private pending = new Map<number, { resolve: (v: any) => void; reject: (e: Error) => void; method: string }>();
  private handlers = new Map<string, Handler[]>();
  closed = false;

  private constructor(ws: WebSocket) {
    this.ws = ws;
    ws.addEventListener("message", (ev) => {
      const m = JSON.parse(String(ev.data));
      if (m.id !== undefined) {
        const p = this.pending.get(m.id);
        this.pending.delete(m.id);
        if (!p) return;
        if (m.error) p.reject(new Error(`${p.method}: ${m.error.message}`));
        else p.resolve(m.result);
      } else {
        for (const h of this.handlers.get(m.method) ?? []) h(m.params);
      }
    });
    ws.addEventListener("close", () => {
      this.closed = true;
      for (const p of this.pending.values()) p.reject(new Error(`${p.method}: the connection closed`));
      this.pending.clear();
    });
  }

  static connect(url: string): Promise<Cdp> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(url);
      ws.addEventListener("open", () => resolve(new Cdp(ws)), { once: true });
      ws.addEventListener("error", () => reject(new Error(`cannot connect to ${url}`)), { once: true });
    });
  }

  send(method: string, params: Record<string, unknown> = {}): Promise<any> {
    // A closed socket drops a send silently, and the promise would never settle.
    if (this.closed) return Promise.reject(new Error(`${method}: the connection is closed`));
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, method });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  on(method: string, h: Handler): void {
    this.handlers.set(method, [...(this.handlers.get(method) ?? []), h]);
  }

  /** The value of an expression in the page; a thrown page error raises here. */
  async evaluate<T>(expression: string): Promise<T> {
    const r = await this.send("Runtime.evaluate", { expression: `${expression}\n//# sourceURL=${TOOL_URL}`, returnByValue: true, awaitPromise: true });
    if (r.exceptionDetails) throw new Error(`page: ${r.exceptionDetails.exception?.description ?? r.exceptionDetails.text}`);
    return r.result.value as T;
  }

  close(): void {
    this.ws.close();
  }
}

/** The first target a DevTools HTTP endpoint lists that `want` accepts, polling until `ms`. */
export async function findTarget(port: number, want: (t: { type: string; url: string }) => boolean, ms: number): Promise<string> {
  const end = Date.now() + ms;
  let last = "no answer";
  while (Date.now() < end) {
    try {
      const list = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()) as { type: string; url: string; webSocketDebuggerUrl: string }[];
      const t = list.find(want);
      if (t) return t.webSocketDebuggerUrl;
      last = JSON.stringify(list.map((x) => `${x.type} ${x.url}`));
    } catch (err) {
      last = err instanceof Error ? err.message : String(err);
    }
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error(`no DevTools target on port ${port} after ${ms} ms; last answer: ${last}`);
}
