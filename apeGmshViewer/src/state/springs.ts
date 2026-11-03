// The Springs card's glyphs: one force arrow per zero-length spring of the
// results step on screen (the viewer twin of the Qt `SpringForceDiagram`,
// src/apeGmsh/viewers/diagrams/_spring_force.py).
//
// The force is node j's global resisting force from the `.ladruno` `force`
// token (reader/ladruno.ts `readSpringStep`): the force node j applies to the
// spring, tension-positive like the Qt diagram's `basicForce x direction` (a
// spring stretched along +x draws +x). It is the sum of the spring's basic
// forces along its directions, so a skewed (`-orient`) or an interface spring
// needs no frame. The spring acts on node j with the opposite force, so at a
// spring's ground node the Reactions card draws the opposite arrow. The Qt
// diagram draws one spring direction at a time along a global axis or the
// interface frame; per-direction arrows from `basicForce` are a follow-up.
//
// Pure: no three.js and no DOM, so it runs in Node tests.

import type { BlobStore } from "./blobs.ts";
import { sampleCmap } from "./contour.ts";
import { ARROW, ARROW_STRIDE, BAR, buildNodeCoordsMap, loadsColourPalette, loadsUnitsOf, thinningLevels, type BuildLoads3DOptions } from "./loads.ts";
import type { State } from "./types.ts";

/** One step of the springs, as main sends it (reader/ladruno.ts `LadrunoSpringStep`). */
export interface SpringStepData {
  /** (n * 2,) end nodes `[iNode, jNode]` per spring */
  readonly nodes: Float64Array;
  /** (n * 3,) node j's force along global x, y, z */
  readonly force: Float64Array;
}

/** The zero-length springs the results stage on screen records a force for; 0 off a results phase. */
export function springCountOf(s: State): number {
  const at = s.phase.at;
  if (at?.kind !== "results" || !s.results) return 0;
  return s.results.stages[at.stage]?.springs ?? 0;
}

/** A spring arrow below this fraction of the step's largest is solver noise, not drawn (as the reactions' floor). */
export const SPRING_FLOOR = 1e-6;

export interface Springs3D {
  /** One record per arrow, `ARROW_STRIDE` floats, no mark. */
  readonly arrows: Float32Array;
  /** Arrows drawn: one per spring for the resultant or one axis, up to three for `xyz`. */
  readonly count: number;
  /** Springs whose two nodes are both missing from the model on screen, so they have no place to draw. */
  readonly unplaced: number;
  /** The step's largest arrow (resultant or component, in the display units), which sets the longest arrow. */
  readonly maxMag: number;
}

/**
 * The spring arrows of one results step: at each spring with a finite,
 * non-zero force, an arrow along node j's force (the resultant, its global-axis
 * components, or one of them, as `s.springs.component` asks), its tip on the
 * spring as the reaction arrows' is on the node. The spring sits at node j, or
 * at node i when the model on screen lacks node j (a ground node the bridge
 * minted); a spring with neither is counted in `unplaced`. Lengths follow the
 * step's largest arrow; thinning is per spring, so a spring's components show
 * or hide together.
 */
export function buildSprings3D(s: State, blobs: BlobStore, step: SpringStepData, opts?: BuildLoads3DOptions): Springs3D {
  const none: Springs3D = { arrows: new Float32Array(0), count: 0, unplaced: 0, maxMag: 0 };
  if (!s.mesh) return none;
  const n = Math.min(step.nodes.length / 2, step.force.length / 3);
  if (n === 0) return none;
  const factor = loadsUnitsOf(s).force.factor;
  const coords = buildNodeCoordsMap(s, blobs, opts);
  const want = s.springs.component;
  const axes = want === "xyz" ? [0, 1, 2] : want === "x" ? [0] : want === "y" ? [1] : want === "z" ? [2] : null;
  type Item = { tip: readonly [number, number, number]; vec: [number, number, number]; mag: number; spring: number };
  const items: Item[] = [];
  let unplaced = 0;
  let max = 0;
  for (let k = 0; k < n; k++) {
    const tip = coords.get(step.nodes[k * 2 + 1]!) ?? coords.get(step.nodes[k * 2]!);
    if (!tip) {
      unplaced++;
      continue;
    }
    const f = [step.force[k * 3]! * factor, step.force[k * 3 + 1]! * factor, step.force[k * 3 + 2]! * factor] as const;
    if (!f.every(Number.isFinite)) continue;
    const parts: [number, number, number][] = axes ? axes.map((a) => [a === 0 ? f[0] : 0, a === 1 ? f[1] : 0, a === 2 ? f[2] : 0]) : [[f[0], f[1], f[2]]];
    for (const vec of parts) {
      const mag = Math.hypot(...vec);
      if (mag <= 1e-30) continue;
      items.push({ tip, vec, mag, spring: k });
      if (mag > max) max = mag;
    }
  }
  // Solver noise is not a force: a component near 1e-12 where the true value is 0 would otherwise get the shortest arrow.
  const kept = items.filter((it) => it.mag >= SPRING_FLOOR * max);
  if (kept.length === 0) return { ...none, unplaced };
  const radius = s.mesh.radius > 0 && Number.isFinite(s.mesh.radius) ? s.mesh.radius : 1;
  const baseLen = Math.max(radius * 0.22, 1e-4) * s.springs.scale;
  const palette = loadsColourPalette(opts?.theme ?? "dark");
  const arrows = new Float32Array(kept.length * ARROW_STRIDE);
  kept.forEach((it, i) => {
    const ratio = it.mag / max;
    const rgb = s.springs.colorBy === "magnitude" ? sampleCmap("jet", ratio) : palette.mono;
    const o = i * ARROW_STRIDE;
    arrows.set(it.tip, o + ARROW.tip);
    arrows.set([it.vec[0] / it.mag, it.vec[1] / it.mag, it.vec[2] / it.mag], o + ARROW.dir);
    arrows[o + ARROW.len] = baseLen * (0.35 + 0.65 * ratio);
    arrows.set(rgb, o + ARROW.rgb);
    arrows[o + ARROW.heads] = 1;
    arrows[o + ARROW.bar] = BAR.none;
    arrows[o + ARROW.priority] = ratio;
  });
  // Thinning per spring: its strongest arrow decides, and its components take that level.
  const first = new Map<number, number>();
  kept.forEach((it, i) => {
    const j = first.get(it.spring);
    if (j === undefined || kept[j]!.mag < it.mag) first.set(it.spring, i);
  });
  const reps = [...first.values()];
  const per = new Float32Array(reps.length * ARROW_STRIDE);
  reps.forEach((i, j) => per.set(arrows.subarray(i * ARROW_STRIDE, (i + 1) * ARROW_STRIDE), j * ARROW_STRIDE));
  thinningLevels(per, radius);
  const levelOf = new Map<number, number>();
  reps.forEach((i, j) => levelOf.set(kept[i]!.spring, per[j * ARROW_STRIDE + ARROW.level]!));
  kept.forEach((it, i) => (arrows[i * ARROW_STRIDE + ARROW.level] = levelOf.get(it.spring)!));
  return { arrows, count: kept.length, unplaced, maxMag: max };
}
