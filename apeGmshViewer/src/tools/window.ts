// The tool window's page (ADR 0137 D9, H1): a separate sandboxed renderer that mounts one tool.
//
// Its preload (src/main/toolPreload.ts) hands it one MessagePort, linked to the model window's
// page, and the tool's id. Contexts arrive on the port and are drawn at most once per frame
// (./host.ts); the tool's events go back on it. The page holds no store and reads no file.

import type { ToolAnswer, ToolContextMessage, ToolEvent, ToolEventMessage, ToolPortMessage, ToolRequest } from "./contract.ts";
import { ToolScheduler } from "./host.ts";
import { TOOLS } from "./registry.ts";
import { cssVariables } from "../theme/tokens.ts";

const bridge = (window as unknown as { toolHost: { ready(): void; request(q: ToolRequest, seq: number): Promise<ToolAnswer> } }).toolHost;
const root = document.getElementById("tool")!;

window.addEventListener("message", (ev: MessageEvent) => {
  const meta = (ev.data as Partial<ToolPortMessage> | null)?.toolPort;
  const port = ev.ports[0];
  if (!meta || !port) return;
  const def = TOOLS[meta.tool];
  if (!def) {
    root.textContent = `No tool named ${JSON.stringify(meta.tool)}`;
    return;
  }
  document.title = `${def.title} - apeGmshViewer`;
  let seq = 0;
  const handle = def.mount({ root, send: (event: ToolEvent) => port.postMessage({ event } satisfies ToolEventMessage), request: (q) => bridge.request(q, ++seq) });
  const scheduler = new ToolScheduler(handle, { request: (fn) => requestAnimationFrame(() => fn()) }, () => document.visibilityState === "visible");
  document.addEventListener("visibilitychange", () => scheduler.shown());
  // The app's colour tokens, as the model window's page has them (not applyTheme: its font faces are
  // files, and the tool page's content security policy loads none).
  let theme: string | null = null;
  port.onmessage = (m: MessageEvent) => {
    const ctx = (m.data as Partial<ToolContextMessage> | null)?.context;
    if (!ctx) return;
    if (ctx.theme !== theme) {
      theme = ctx.theme;
      for (const [k, v] of Object.entries(cssVariables(ctx.theme))) document.documentElement.style.setProperty(k, v);
    }
    scheduler.push(ctx);
  };
});

bridge.ready();
