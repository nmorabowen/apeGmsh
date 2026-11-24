// The tool window's preload (ADR 0137 D13 item 4, H1). Its whole privilege: say the page is
// ready, and receive the one MessagePort main links to the model window's page. It exposes
// nothing of the model window's `viewer` bridge: a tool reads no file and opens no window.

import { contextBridge, ipcRenderer } from "electron";

ipcRenderer.on("tool:port", (e, meta: { tool: string }) => {
  window.postMessage({ toolPort: { tool: meta.tool } }, "*", e.ports as unknown as Transferable[]);
});

contextBridge.exposeInMainWorld("toolHost", {
  ready: () => ipcRenderer.send("tool:ready"),
});
