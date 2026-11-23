// ADR 0137 D4 stage 1 (#3057): the machine's memory, as main reads it for a model load.
//
// Electron's `process.getSystemMemoryInfo()` (KB) on Windows reports available RAM as `free`, and
// the commit headroom as `swapFree` (the commit limit minus the commit charge); the #3056 rows in
// apeGmshViewer/MEASUREMENTS.md checked both against the OS counters. On Linux and macOS the swap
// fields mean swap, so the commit rule is not applied there. No Electron import: the call is on
// `process`, and a test puts its own reading there.

import { estimatePeakMB, memoryReasons } from "../state/memory.ts";
import type { MemoryStatus } from "../state/types.ts";

interface SystemMemoryInfo {
  free: number;
  swapFree?: number;
}

/** Available RAM and commit headroom, MB, or null where the call does not exist (plain Node). */
export function readMemory(): { availableMB: number; commitHeadroomMB: number | null } | null {
  const info = (process as { getSystemMemoryInfo?: () => SystemMemoryInfo }).getSystemMemoryInfo?.();
  if (!info) return null;
  const commit = process.platform === "win32" && typeof info.swapFree === "number" ? info.swapFree / 1024 : null;
  return { availableMB: info.free / 1024, commitHeadroomMB: commit };
}

/** The status of one reading for a load of `path` with `cells` cells (null: not counted), or null with no reading. */
export function memoryStatus(path: string, cells: number | null, phase: MemoryStatus["phase"]): MemoryStatus | null {
  const m = readMemory();
  if (m === null) return null;
  const estimateMB = cells === null ? null : estimatePeakMB(cells);
  return { path, phase, cells, estimateMB, ...m, reasons: memoryReasons(m.availableMB, m.commitHeadroomMB, estimateMB) };
}

/** How often main reads memory again while a model loads. */
export const DURING_MS = 1000;

/**
 * Send `send` a status before a load of `path`, then every `DURING_MS` while `load` runs, and
 * once when it ends. A count that fails sends the RAM rule alone. The load runs whatever memory
 * reads: this warns and never refuses.
 */
export async function watchLoad<T>(path: string, cells: Promise<number | null>, send: (m: MemoryStatus) => void, load: () => Promise<T>): Promise<T> {
  const n = await cells.catch(() => null);
  const say = (phase: MemoryStatus["phase"]) => {
    const m = memoryStatus(path, n, phase);
    if (m !== null) send(m);
  };
  say("before");
  const timer = setInterval(() => say("during"), DURING_MS);
  try {
    return await load();
  } finally {
    clearInterval(timer);
    say("after");
  }
}
