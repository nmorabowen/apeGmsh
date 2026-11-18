// The capture and measure harness (#2669): what a wait was waiting for, and why
// it failed, in words that name the failed condition. Pure, so the tests run it
// in Node.

import type { ContourScan } from "../../state/types.ts";

/** What a cut-face still waits for. */
export interface CapsStillFacts {
  caps: number;
  want: number;
  settled: boolean;
  deformShown: boolean;
  /** the still colours by contour, so the contour must show its step */
  contourWanted: boolean;
  contourShown: boolean;
  /** the contour's state, with its reason when it has one */
  contour: string;
}

/** The message of a cut-face still that never became ready: every condition that failed. */
export function capsStillProblem(name: string, f: CapsStillFacts): string {
  const failed: string[] = [];
  if (f.caps !== f.want) failed.push(`${f.caps} caps drawn, want ${f.want}`);
  if (!f.settled) failed.push("the caps have not settled");
  if (!f.deformShown) failed.push("the deformed shape does not show the step");
  if (f.contourWanted && !f.contourShown) failed.push(`the contour does not show the step (${f.contour})`);
  return `capture ${name}: ${failed.join("; ") || "the still was not ready, though every condition now holds"}`;
}

/** Which history scan a contour scrub needs. */
export interface ScanWant {
  /** an element field's scan, not a nodal one left from the row before */
  element: boolean;
  /** a node-located mode's own scan key; null when any scan of the right kind will do */
  key: string | null;
}

export function scanSatisfied(scan: ContourScan | null, want: ScanWant): boolean {
  return scan?.status === "done" && (!want.element || scan.key.includes("|elem:")) && (want.key === null || scan.key === want.key);
}

/** Why the wait for a scan ended without it: a scan that finished is not reported as unfinished. */
export function scanProblem(scan: ContourScan | null, want: ScanWant): string {
  if (scan?.status === "done") {
    const wanted = want.key ?? (want.element ? "an element-field scan" : "the contour's scan");
    return `measure: the history scan that finished is not the one this row needs: it is ${scan.key}, the row needs ${wanted}`;
  }
  return `measure: the history scan did not finish: ${JSON.stringify(scan)}`;
}

/** Why a stage gives no scrub row; null when it has steps to scrub. */
export function scrubSkipNote(file: string, stage: string, steps: number): string | null {
  return steps < 2 ? `${file}: stage ${stage} has ${steps} step${steps === 1 ? "" : "s"}, so there is nothing to scrub; no contour row` : null;
}
