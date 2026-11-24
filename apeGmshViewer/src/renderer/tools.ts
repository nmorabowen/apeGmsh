// The model window's links to its tool windows (ADR 0137 D10, H1).
//
// - An `openTool` request in the state (H4) adds a tile to the newest linked tool window, over its
//   port (`{ addTile }`). With none linked, or for a new window, it goes to main (`tool:open`), which
//   makes the window and, once its page is ready, hands this page one end of a MessagePort pair
//   (`tool:port`) with the window's id.
// - Each state change sends every linked tool the new context, if it changed and tools are not
//   paused (ADR 0137 D4). A tool draws at most one per frame (src/tools/host.ts).
// - What a tool sends back is checked against the state (state/tools.ts `checkToolEvent`) and
//   dispatched; anything else is dropped.
// - Main's reports of each window (`tool:status`, `tool:paused`) are dispatched as they come.

import type { Store } from "../state/store.ts";
import { checkToolEvent, toolContextOf } from "../state/tools.ts";
import type { ToolStatus } from "../state/types.ts";
import type { AddTileMessage, ToolContext, ToolContextMessage, ToolEventMessage, ToolPortMessage } from "../tools/contract.ts";

/** What this needs of the preload's `viewer` bridge. */
export interface ToolBridge {
  openTool?(tool: string): Promise<unknown>;
  onToolStatus?(cb: (s: ToolStatus) => void): (() => void) | void;
  onToolsPaused?(cb: (paused: boolean) => void): (() => void) | void;
}

interface Link {
  port: MessagePort;
  /** the last context sent, as JSON, so an unchanged one is not sent again */
  sent: string | null;
}

export function mountToolLinks(store: Store, bridge: ToolBridge, win: Window = window): () => void {
  const links = new Map<string, Link>();
  const off: (() => void)[] = [];

  const send = (link: Link, force = false) => {
    const s = store.get();
    if (s.tools.paused) return;
    const json = JSON.stringify(toolContextOf(s));
    if (!force && json === link.sent) return;
    link.sent = json;
    link.port.postMessage({ context: JSON.parse(json) as ToolContext } satisfies ToolContextMessage);
  };

  const onMessage = (ev: MessageEvent) => {
    const meta = (ev.data as Partial<ToolPortMessage> | null)?.toolPort;
    const port = ev.ports[0];
    if (!meta || !port || ev.source !== win) return;
    // H4: by tool window, newest last (a Map keeps insertion order): "Add tile" goes to the newest.
    links.get(meta.window)?.port.close();
    links.delete(meta.window);
    const link: Link = { port, sent: null };
    links.set(meta.window, link);
    port.onmessage = (m: MessageEvent) => {
      // The tool is another process: its event is checked against the state before it is dispatched.
      const e = checkToolEvent(store.get(), (m.data as Partial<ToolEventMessage> | null)?.event);
      if (e !== null) store.dispatch(e);
    };
    send(link, true);
  };
  win.addEventListener("message", onMessage);
  off.push(() => win.removeEventListener("message", onMessage));

  off.push(
    store.subscribe((s, prev) => {
      // H4: a tile goes to the newest linked tool window, on its port; a new window (asked for, or
      // because none is linked) is main's to open, with that tile as its first.
      const r = s.tools.request;
      if (r !== prev.tools.request && r !== null) {
        const newest = [...links.values()].at(-1);
        if (!r.newWindow && newest) newest.port.postMessage({ addTile: r.tool } satisfies AddTileMessage);
        else if (typeof bridge.openTool === "function") void bridge.openTool(r.tool);
      }
      // A window that is gone drops its link; a resume sends each tool the newest context.
      for (const [id, w] of Object.entries(s.tools.windows)) {
        if ((w.status === "closed" || w.status === "stopped") && links.has(id)) {
          links.get(id)!.port.close();
          links.delete(id);
        }
      }
      const resumed = prev.tools.paused && !s.tools.paused;
      for (const link of links.values()) send(link, resumed);
    }),
  );
  const offStatus = bridge.onToolStatus?.((status) => store.dispatch({ type: "toolStatus", status }));
  if (typeof offStatus === "function") off.push(offStatus);
  const offPaused = bridge.onToolsPaused?.((paused) => store.dispatch({ type: "toolsPaused", paused }));
  if (typeof offPaused === "function") off.push(offPaused);

  return () => {
    for (const f of off) f();
    for (const l of links.values()) l.port.close();
    links.clear();
  };
}
