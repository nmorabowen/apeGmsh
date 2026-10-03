// The renderer's only door to the main process.

import { contextBridge, ipcRenderer, webUtils } from "electron";

contextBridge.exposeInMainWorld("viewer", {
  config: () => ipcRenderer.invoke("app:config"),
  openModel: (path: string) => ipcRenderer.invoke("model:open", path),
  pathForFile: (f: File) => webUtils.getPathForFile(f),
  metrics: () => ipcRenderer.invoke("measure:metrics"),
  measureDone: (result: unknown) => ipcRenderer.invoke("measure:done", result),
  captureStill: (suffix: string) => ipcRenderer.invoke("capture:still", suffix),
  captureDone: () => ipcRenderer.invoke("capture:done"),
  fail: (message: string) => ipcRenderer.invoke("app:fail", message),
});
