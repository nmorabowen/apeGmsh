// Tool windows (ADR 0137 D9, D11, D13 items 3-5; H1): main creates each one, as a separate
// sandboxed renderer, and links it to the model window's page with one MessagePort pair. The
// context and the tool's events then go page to tool directly; main only creates, links, watches
// and closes.
//
// - D13 item 3: `sandbox`, `contextIsolation`, no `nodeIntegration`, the tool's own preload, and
//   the content security policy in src/tools/tool.html.
// - D13 item 5: `tool:ready` is honoured only from a window main made for a tool.
// - D11: a renderer that dies or stops responding shows as stopped or not responding in the model
//   window, which keeps running; the next tool window is a new one.
// - H4: a tool window holds tiles of several tools (src/tools/window.ts). Each window has an id
//   ("w1", "w2", …) and opens with one tile; the model window adds more over its port.
// - D4: while a tool window is open, main reads memory every 2 s; under 1 GB available it pauses
//   tool updates (the model window stops sending context) and refuses new tool windows.
//
// No Electron import: main.ts hands in how to make a window and a port pair, so a test drives this.

import type { ToolStatus } from "../state/types.ts";
import type { ToolPortMeta } from "../tools/contract.ts";
import { push } from "./ipc.ts";

/** D4: under this much available RAM, tool updates pause and no tool window opens. */
export const TOOL_FLOOR_MB = 1024;
/** How often main reads memory while a tool window is open. */
export const TOOL_MEMORY_MS = 2000;

/** What this needs of a tool window's `webContents`. */
export interface ToolContents {
  postMessage(channel: string, message: unknown, transfer?: unknown[]): void;
  on(event: "render-process-gone", fn: (e: unknown, details: { reason: string; exitCode: number }) => void): unknown;
  on(event: "unresponsive" | "responsive", fn: () => void): unknown;
}

/** What this needs of a tool window. */
export interface ToolWin {
  readonly webContents: ToolContents;
  focus(): void;
  close(): void;
  isDestroyed(): boolean;
  on(event: "closed", fn: () => void): unknown;
}

/** What this needs of the model window's page. */
export interface ToolPage {
  send(channel: string, ...args: unknown[]): void;
  postMessage(channel: string, message: unknown, transfer?: unknown[]): void;
  isDestroyed(): boolean;
}

export interface ToolWindowDeps {
  /** make the window for `tool`, titled `title`, and start loading its page */
  create(tool: string, title: string): ToolWin;
  /** a linked MessagePort pair (Electron's `MessageChannelMain`) */
  channel(): { port1: unknown; port2: unknown };
  /** available RAM in MB, or null where it cannot be read */
  availableMB(): number | null;
  /** run `fn` every `ms`; the returned function stops it */
  every(ms: number, fn: () => void): () => void;
}

/** D13 item 3: a tool window's `webPreferences`. */
export function toolWebPreferences(preload: string) {
  return { preload, sandbox: true, contextIsolation: true, nodeIntegration: false, backgroundThrottling: false } as const;
}

interface Open {
  win: ToolWin;
  page: ToolPage;
  title: string;
  /** the tool its first tile mounts (H4: the window adds more tiles on the model window's word) */
  tool: string;
  /** its renderer died: the window shows nothing, and the next tool window is a new one */
  gone: boolean;
}

export class ToolWindows {
  private readonly deps: ToolWindowDeps;
  private readonly tools: ReadonlySet<string>;
  /** the open tool windows, by window id */
  private readonly open = new Map<string, Open>();
  private next = 1;
  private stopPoll: (() => void) | null = null;
  private paused = false;

  constructor(deps: ToolWindowDeps, tools: Iterable<string>) {
    this.deps = deps;
    this.tools = new Set(tools);
  }

  /**
   * H4: open a new tool window for the model window's `page`, its first tile a `tool`. Its id names
   * it to the page (`ToolStatus.window`, the port's meta); the page adds more tiles to it over the
   * port. A window whose renderer died is closed first. Refused: why.
   */
  openTool(page: ToolPage, tool: unknown): { ok: true; window: string } | { ok: false; reason: string } {
    if (typeof tool !== "string" || !this.tools.has(tool)) return { ok: false, reason: `no tool named ${JSON.stringify(tool)}` };
    for (const o of [...this.open.values()]) if (o.gone && !o.win.isDestroyed()) o.win.close();
    const id = `w${this.next++}`;
    const title = `Tool window ${id.slice(1)}`;
    const mb = this.deps.availableMB();
    if (mb !== null && mb < TOOL_FLOOR_MB) {
      const reason = `${(mb / 1024).toFixed(1)} GB of RAM is available, under ${(TOOL_FLOOR_MB / 1024).toFixed(1)} GB`;
      this.status(page, id, title, "refused", reason);
      return { ok: false, reason };
    }
    const win = this.deps.create(tool, title);
    const entry: Open = { win, page, title, tool, gone: false };
    this.open.set(id, entry);
    this.status(page, id, title, "opening", null);
    win.webContents.on("render-process-gone", (_e, d) => {
      entry.gone = true;
      this.status(page, id, title, "stopped", `its renderer ${d.reason === "crashed" ? "crashed" : `stopped (${d.reason})`}`);
    });
    win.webContents.on("unresponsive", () => this.status(page, id, title, "unresponsive", null));
    win.webContents.on("responsive", () => this.status(page, id, title, "open", null));
    win.on("closed", () => {
      this.open.delete(id);
      this.status(page, id, title, "closed", null);
      if (this.open.size === 0) this.stopMemory();
    });
    this.startMemory();
    return { ok: true, window: id };
  }

  /**
   * D13 item 5: a tool page said it is ready. Honoured only from a window main made for a tool:
   * link it to its model window's page. False for any other sender.
   */
  ready(sender: unknown): boolean {
    for (const [id, o] of this.open) {
      if (o.win.isDestroyed() || o.win.webContents !== sender) continue;
      if (o.page.isDestroyed()) return false;
      const { port1, port2 } = this.deps.channel();
      const meta: ToolPortMeta = { window: id, tool: o.tool };
      o.win.webContents.postMessage("tool:port", meta, [port2]);
      o.page.postMessage("tool:port", meta, [port1]);
      this.status(o.page, id, o.title, "open", null);
      if (this.paused) push(o.page, "tool:paused", true);
      return true;
    }
    return false;
  }

  /** D13 item 5: whether `sender` is the page of a tool window main made and still holds. */
  isTool(sender: unknown): boolean {
    for (const o of this.open.values()) if (!o.win.isDestroyed() && !o.gone && o.win.webContents === sender) return true;
    return false;
  }

  /** Close every tool window (the model window closed, or its page reloaded and dropped its ports). */
  closeAll(): void {
    for (const o of [...this.open.values()]) if (!o.win.isDestroyed()) o.win.close();
  }

  /** The tool windows open now, by id, for the tests and the harness. */
  openTools(): string[] {
    return [...this.open.keys()];
  }

  private status(page: ToolPage, window: string, title: string, status: ToolStatus["status"], reason: string | null): void {
    if (!page.isDestroyed()) push(page, "tool:status", { window, title, status, reason });
  }

  private startMemory(): void {
    if (this.stopPoll !== null) return;
    this.checkMemory();
    this.stopPoll = this.deps.every(TOOL_MEMORY_MS, () => this.checkMemory());
  }

  private stopMemory(): void {
    this.stopPoll?.();
    this.stopPoll = null;
    if (this.paused) this.setPaused(false);
  }

  private checkMemory(): void {
    const mb = this.deps.availableMB();
    this.setPaused(mb !== null && mb < TOOL_FLOOR_MB);
  }

  private setPaused(on: boolean): void {
    if (on === this.paused) return;
    this.paused = on;
    const pages = new Set([...this.open.values()].map((o) => o.page));
    for (const p of pages) if (!p.isDestroyed()) push(p, "tool:paused", on);
  }
}
