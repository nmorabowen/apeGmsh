// A large answer from main crosses from the preload to the page as one
// structured clone (#3130).
//
// `contextBridge` copies a value object by object, in native code the page
// cannot even be paused in. A model of 19 476 zeroLength springs (170 MB as a
// clone, most of it /opensees decl params and stages) reached the preload in
// 19 s and was still being copied into the page 90 s later; V8's structured
// clone of the same answer takes about 1.3 s. So the preload posts such an
// answer to its own window (`window.postMessage`, a structured clone) and the
// bridge returns only the answer's id; the page takes the answer by that id.
//
// A leaf: preload.ts (the sender) and renderer/app.ts (the receiver) import it.

/** The key of a bulk message's id: `{ [BULK_KEY]: id, value }`. */
export const BULK_KEY = "apegmshViewerBulk";

/** What the preload posts. */
export interface BulkMessage {
  [BULK_KEY]: number;
  value: unknown;
}

/** The page's window, as far as the inbox needs it. */
export interface BulkWindow {
  addEventListener(type: "message", listener: (e: { source: unknown; data: unknown }) => void): void;
}

/**
 * The page side: holds each posted answer until its id is taken. Either may
 * come first: the bridge's promise of the id and the message event are two
 * tasks, in no order the page can rely on. Only messages from the page's own
 * window count (the preload posts to it; the app opens no other frame).
 */
export class BulkInbox {
  private readonly arrived = new Map<number, unknown>();
  private readonly waiting = new Map<number, (value: unknown) => void>();

  constructor(win: BulkWindow, self: unknown = win) {
    win.addEventListener("message", (e) => {
      if (e.source !== self || typeof e.data !== "object" || e.data === null) return;
      const id = (e.data as Partial<BulkMessage>)[BULK_KEY];
      if (typeof id !== "number") return;
      const value = (e.data as BulkMessage).value;
      const take = this.waiting.get(id);
      if (take) {
        this.waiting.delete(id);
        take(value);
      } else this.arrived.set(id, value);
    });
  }

  /** The answer posted under the id `id` resolves to. A rejected `id` rejects. */
  async take(id: Promise<number>): Promise<unknown> {
    const n = await id;
    if (this.arrived.has(n)) {
      const value = this.arrived.get(n);
      this.arrived.delete(n);
      return value;
    }
    return new Promise((resolve) => this.waiting.set(n, resolve));
  }
}
