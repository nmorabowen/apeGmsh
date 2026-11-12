// The overlays slice's events: its part of the store's Event union (types/event.ts).
// Not re-exported by src/state/types.ts, which exports only the whole union.

import type { ContourCmap } from "../contour.ts";
import type { LoadsColorBy, PrincipalHead, ReactionComponent, SpringsState } from "../overlays.ts";

export type OverlaysEvent =
  /** The Display card's BC toggle */
  | { type: "setSupports"; on?: boolean; scale?: number }
  /** The Principal card's controls */
  | { type: "setPrincipal"; on?: boolean; field?: string | null; show?: readonly [boolean, boolean, boolean]; head?: PrincipalHead; cmap?: ContourCmap; scale?: number; headSize?: number; thin?: boolean }
  /** The Reactions card's controls */
  | { type: "setReactions"; on?: boolean; scale?: number; headSize?: number; colorBy?: LoadsColorBy; thin?: boolean; component?: ReactionComponent; allNodes?: boolean; forces?: boolean; moments?: boolean }
  /** The Springs card's controls; `notice` is set by the effects when the step's springs' read fails */
  | { type: "setSprings"; on?: boolean; scale?: number; headSize?: number; colorBy?: SpringsState["colorBy"]; thin?: boolean; component?: ReactionComponent; notice?: string | null }
  /** The Vectors card's controls (`field` null: the stage's first vector field) */
  | { type: "setVectors"; on?: boolean; field?: string | null; scale?: number; headSize?: number; colorBy?: LoadsColorBy; thin?: boolean; component?: ReactionComponent }
  /** 3-D structural loading and prescribed-SP visualization controls */
  | {
      type: "setLoads";
      on?: boolean;
      scale?: number;
      showNodalForces?: boolean;
      showNodalMoments?: boolean;
      showElementLoads?: boolean;
      showSp?: boolean;
      includeSupportReactions?: boolean;
      scaleBySeries?: boolean;
      headSize?: number;
      thin?: boolean;
      colorBy?: LoadsColorBy;
      selectedPatterns?: readonly string[] | null;
      togglePattern?: string;
    };
