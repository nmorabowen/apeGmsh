// The renderer's only door to the main process.
//
// Contract for the renderer (V2e's effects.ts feature-detects these):
//   onOpen(cb(set))        cb hears the set already open (if any) once, then
//                          every later open: {model, geometry, results}, each
//                          an absolute path or null (./pairing.ts)
//   onFileChanged(cb(path)) a file of the open set was rewritten and closed
//   goToSource(file, line)  open the user's editor; Promise<{ok, reason?}>
//   requestOpen(path)       open the set of `path` (a dropped file), as a
//                          double-click would; Promise<{ok, reason?}>
//   openGeometry(path)      read a /geometry sibling (./zones.ts); openModel's
//                          answer also carries the model's /provenance
// onOpen and onFileChanged return a function that removes the listener.
// Until the page subscribes to a channel, main reloads the page instead of
// sending on it, so the P0 page (which reads the model from config) follows.

import { contextBridge, ipcRenderer, webUtils, type IpcRendererEvent } from "electron";

interface OpenSet {
  model: string | null;
  geometry: string | null;
  results: string | null;
}

const listeners = { open: 0, fileChanged: 0 };

function listen<T>(channel: "open" | "fileChanged", cb: (value: T) => void, onLive: () => void): () => void {
  const handler = (_e: IpcRendererEvent, value: T) => {
    onLive();
    cb(value);
  };
  ipcRenderer.on(`viewer:${channel}`, handler);
  if (listeners[channel]++ === 0) ipcRenderer.send("viewer:subscribe", channel);
  let live = true;
  return () => {
    if (!live) return;
    live = false;
    ipcRenderer.removeListener(`viewer:${channel}`, handler);
    if (--listeners[channel] === 0) ipcRenderer.send("viewer:unsubscribe", channel);
  };
}

contextBridge.exposeInMainWorld("viewer", {
  config: () => ipcRenderer.invoke("app:config"),
  openModel: (path: string) => ipcRenderer.invoke("model:open", path),
  pathForFile: (f: File) => webUtils.getPathForFile(f),
  metrics: () => ipcRenderer.invoke("measure:metrics"),
  measureDone: (result: unknown) => ipcRenderer.invoke("measure:done", result),
  captureStill: (suffix: string) => ipcRenderer.invoke("capture:still", suffix),
  captureDone: () => ipcRenderer.invoke("capture:done"),
  fail: (message: string) => ipcRenderer.invoke("app:fail", message),

  onOpen: (cb: (set: OpenSet) => void) => {
    // The set already open is replayed to this callback only, unless a live
    // open has reached it first.
    let heardLive = false;
    let removed = false;
    const off = listen<OpenSet>("open", cb, () => {
      heardLive = true;
    });
    void ipcRenderer.invoke("viewer:currentSet").then((set: OpenSet | null) => {
      if (set && !heardLive && !removed) cb(set);
    });
    return () => {
      removed = true;
      off();
    };
  },
  onFileChanged: (cb: (path: string) => void) => listen<string>("fileChanged", cb, () => {}),
  goToSource: (file: string, line: number) => ipcRenderer.invoke("source:goto", file, line),
  requestOpen: (path: string) => ipcRenderer.invoke("viewer:requestOpen", path),
  openGeometry: (path: string) => ipcRenderer.invoke("geometry:open", path),
});
