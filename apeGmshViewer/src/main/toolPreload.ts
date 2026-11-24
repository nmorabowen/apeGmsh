// The tool window's preload (ADR 0137 D13 item 4, H1). Its whole privilege: say the page is
// ready, receive the one MessagePort main links to the model window's page, and (H2b) ask main
// for data, which main checks. It exposes nothing of the model window's `viewer` bridge: a tool
// names no file and opens no window.

import { contextBridge, ipcRenderer } from "electron";

ipcRenderer.on("tool:port", (e, meta: { tool: string }) => {
  window.postMessage({ toolPort: { tool: meta.tool } }, "*", e.ports as unknown as Transferable[]);
});

contextBridge.exposeInMainWorld("toolHost", {
  ready: () => ipcRenderer.send("tool:ready"),
  // ADR 0137 D10 (H2b): ask main for data; main checks this window and the request, and names the file.
  request: (q: unknown, seq: number) => ipcRenderer.invoke("tool:request", q, seq),
});
