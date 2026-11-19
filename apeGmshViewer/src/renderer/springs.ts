// The Springs card's arrows (`s.springs`): one per zero-length spring of the
// results step on screen (state/springs.ts `buildSprings3D`), drawn by the
// shared arrow engine (renderer/loads.ts `ArrowLayer`) on this card's own
// sizing, colour and thinning.

import type { BlobStore } from "../state/blobs.ts";
import { buildSprings3D } from "../state/springs.ts";
import type { State } from "../state/types.ts";
import type { StepValues } from "./contour.ts";
import type { FullPositions } from "./deform.ts";
import { ArrowLayer } from "./loads.ts";

const NO_RECORDS = new Float32Array(0);

export class SpringsLayer extends ArrowLayer {
  private appliedKey: string | null = null;
  /** The step data drawn: a re-read of the same step rebuilds the arrows. */
  private drawnStep: StepValues | null = null;
  /** The results stage the arrows on screen were built for. */
  private drawnStage: number | null = null;

  sync(s: State, blobs: BlobStore, full: FullPositions | null = null, step: StepValues | null = null): boolean {
    const at = s.phase.at;
    if (!s.springs.on || !s.mesh || at?.kind !== "results") {
      this.appliedKey = null;
      this.drawnStep = null;
      this.drawnStage = null;
      return this.hide();
    }
    const sized = this.size(s.springs);
    // Until the step's springs are here (a scrub), the last step's arrows stay.
    const here = step && step.stage === at.stage && step.step === at.step && step.springs ? step : null;
    const nextKey = [s.mesh.linePositions.id, full ? full.key : "base", s.springs.scale, s.springs.colorBy, s.springs.component, s.contour.sourceUnits ?? "", s.contour.targetUnits ?? "", this.theme].join("|");
    if (!here) {
      // Another stage's arrows, or a step whose read failed, are not this step's springs.
      const failed = (s.resultStep?.stage === at.stage && s.resultStep.step === at.step && s.resultStep.status === "failed") || (step?.stage === at.stage && step.step === at.step && step.springs === null);
      if (this.drawnStage !== at.stage || failed) {
        this.appliedKey = null;
        this.drawnStep = null;
        this.drawnStage = null;
        return this.hide() || sized;
      }
      return sized;
    }
    if (this.appliedKey === nextKey && here === this.drawnStep) return sized;
    const g = buildSprings3D(s, blobs, here.springs!, { theme: this.theme, ...(full ? { drawnLinePositions: full.line, drawnTriPositions: full.tri } : {}) });
    this.drawnStep = here;
    this.drawnStage = at.stage;
    this.appliedKey = nextKey;
    this.upload(g.arrows, NO_RECORDS, s.mesh.radius);
    return true;
  }
}
