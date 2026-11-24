// The model window's links to its tool windows (ADR 0137 D10, H1).
//
// - An `openTool` request in the state goes to main (`tool:open`); main makes the window and,
//   once its page is ready, hands this page one end of a MessagePort pair (`tool:port`).
// - Each state change sends every linked tool the new context, if it changed and tools are not
//   paused (ADR 0137 D4). A tool draws at most one per frame (src/tools/host.ts).
// - What a tool sends back is checked against the state (state/tools.ts `checkToolEvent`) and
//   dispatched; anything else is dropped.
// - Main's reports of each window (`tool:status`, `tool:paused`) are dispatched as they come.

import type { Store } from "../state/store.ts";
import { checkToolEvent, toolContextOf } from "../state/tools.ts";
import type { ToolStatus } from "../state/types.ts";

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
    link.port.postMessage({ context: JSON.parse(json) as unknown });
  };

  const onMessage = (ev: MessageEvent) => {
    const meta = (ev.data as { toolPort?: { tool: string } } | null)?.toolPort;
    const port = ev.ports[0];
    if (!meta || !port || ev.source !== win) return;
    links.get(meta.tool)?.port.close();
    const link: Link = { port, sent: null };
    links.set(meta.tool, link);
    port.onmessage = (m: MessageEvent) => {
      const e = checkToolEvent(store.get(), (m.data as { event?: unknown } | null)?.event);
      if (e !== null) store.dispatch(e);
    };
    send(link, true);
  };
  win.addEventListener("message", onMessage);
  off.push(() => win.removeEventListener("message", onMessage));

  off.push(
    store.subscribe((s, prev) => {
      if (s.tools.request !== prev.tools.request && s.tools.request !== null && typeof bridge.openTool === "function") {
        void bridge.openTool(s.tools.request.tool);
      }
      // A window that is gone drops its link; a resume sends each tool the newest context.
      for (const [tool, w] of Object.entries(s.tools.windows)) {
        if ((w.status === "closed" || w.status === "stopped") && links.has(tool)) {
          links.get(tool)!.port.close();
          links.delete(tool);
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
