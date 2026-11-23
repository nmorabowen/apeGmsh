// The one place the app starts an Electron utility process (ADR 0137 D2, #3058): the loader,
// ./process.ts. scripts/check_app_wall.py W6 allows `utilityProcess` in this file only, so a
// new kind of process still needs a decision. ./client.ts takes what this returns and imports no
// Electron, so it runs in plain Node in the tests.

import { utilityProcess } from "electron";
import type { LoaderChild } from "./client.ts";

/** Fork the loader process at `entry` (dist/loader.mjs, or ./process.ts when main runs from src/). */
export function forkLoader(entry: string): LoaderChild {
  const child = utilityProcess.fork(entry, [], { serviceName: "apeGmshViewer loader", stdio: "inherit" });
  return {
    postMessage: (m) => child.postMessage(m),
    onMessage: (fn) => void child.on("message", fn),
    onExit: (fn) => void child.on("exit", fn),
  };
}
