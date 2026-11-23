// ADR 0137 D4 stage 1 (#3057): warn, never refuse, when memory is short for a model load.
//
// Main reads the machine's memory (src/main/memory.ts) and the model's cell count, and sends the
// page a `MemoryStatus` before the read and every second while it runs. This module is the pure
// part both sides share: the load-peak estimate, the two thresholds, and the banner's words.
//
// The estimate is the whole app's private memory at the peak of a load, from the #3056 rows
// (apeGmshViewer/MEASUREMENTS.md, "What the page memory depends on"): the page peaks at about
// 250 MB + 10 KB per cell, and the rest of the app (main, GPU, utility processes) adds about
// 650 MB. It bounds every measured model with margin and rests on five models, so it warns and
// never refuses (ADR 0137 D4 stage 2 is a later decision).

import type { MemoryStatus } from "./types/model.ts";

/** Under this much available RAM, a load warns whatever the model (ADR 0137 D4). */
export const AVAILABLE_FLOOR_MB = 2048;
/** A load warns when the commit headroom is under this many times its estimated peak (ADR 0137 D4). */
export const COMMIT_FACTOR = 2;

const PAGE_BASE_MB = 250;
const PAGE_KB_PER_CELL = 10;
const REST_OF_APP_MB = 650;

/** The whole app's estimated private-memory peak while it loads a model of `cells` cells, in MB. */
export function estimatePeakMB(cells: number): number {
  return PAGE_BASE_MB + (cells * PAGE_KB_PER_CELL) / 1024 + REST_OF_APP_MB;
}

/**
 * Why memory is short for this load, one reason per threshold crossed; empty when it is not.
 * `commitHeadroomMB` is null where the platform does not report it (only Windows does, as
 * `getSystemMemoryInfo().swapFree`), and `estimateMB` null when the cell count could not be read:
 * each rule that needs a missing figure is not applied.
 */
export function memoryReasons(availableMB: number, commitHeadroomMB: number | null, estimateMB: number | null): string[] {
  const out: string[] = [];
  if (availableMB < AVAILABLE_FLOOR_MB) out.push(`${gb(availableMB)} of RAM is available, under ${gb(AVAILABLE_FLOOR_MB)}`);
  if (commitHeadroomMB !== null && estimateMB !== null && commitHeadroomMB < COMMIT_FACTOR * estimateMB) {
    out.push(`${gb(commitHeadroomMB)} of memory can still be committed, under twice the estimate`);
  }
  return out;
}

/** The banner's line for a status with reasons, or null. It states the estimate and the memory available. */
export function memoryNotice(m: MemoryStatus): string | null {
  if (m.reasons.length === 0) return null;
  const est = m.estimateMB === null ? "" : `, for a load estimated to peak at about ${gb(m.estimateMB)} (${m.cells!.toLocaleString("en-US")} cells)`;
  const when = m.phase === "before" ? "Memory is short" : m.phase === "during" ? "Memory is short while the model loads" : "Memory was short while the model loaded";
  return `${when}: ${m.reasons.join("; ")}${est}. The app may be slow or stop responding; closing other programs frees memory.`;
}

/** MB as GB with one decimal: "1.9 GB". */
function gb(mb: number): string {
  return `${(mb / 1024).toFixed(1)} GB`;
}
