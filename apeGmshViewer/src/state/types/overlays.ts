// The overlay slice: reactions, principal axes, vectors, springs, supports and loads.
// One slice of the store's types; src/state/types.ts re-exports it, and the
// panels import it from there. A slice file imports sibling files, never the barrel.

import type { ContourCmap } from "./contour.ts";

/**
 * The Reactions card: the results step's reaction forces as arrows pointing at
 * the support, a 45° tick across the tail; sized, coloured and thinned like the
 * load arrows but on their own settings.
 */
export interface ReactionsState {
  readonly on: boolean;
  /** Shaft multiplier (default 1, > 0). */
  readonly scale: number;
  /** Arrowhead length on screen, in CSS pixels (default 12, > 0). */
  readonly headSize: number;
  readonly colorBy: LoadsColorBy;
  readonly thin: boolean;
  /** What each node draws: the resultant, its three global-axis components as separate arrows, or one of them (default resultant). */
  readonly component: ReactionComponent;
  /** Draw every node the results record a reaction at, not only the supports (default false): free nodes then show their residual, constrained nodes their constraint force. */
  readonly allNodes: boolean;
  /** Draw the reaction forces (default true). */
  readonly forces: boolean;
  /** Draw the reaction moments as half-circle arcs about their axes (default true). */
  readonly moments: boolean;
}

/**
 * The Principal card: at each Gauss point an arrow along each principal
 * direction of a stress or strain tensor (state/principal.ts), the length from
 * |p|, the colour from the signed p on a diverging map.
 */
export interface PrincipalState {
  readonly on: boolean;
  /** the tensor field drawn (a `.ladruno` RESULT, `stress` or `strain`); null: the stage's first */
  readonly field: string | null;
  /** which of p1 >= p2 >= p3 are drawn (default all three) */
  readonly show: readonly [boolean, boolean, boolean];
  readonly head: PrincipalHead;
  /** the map of the signed value, over -max..max (default RdBu_r: compression blue, tension red) */
  readonly cmap: ContourCmap;
  /** Shaft multiplier (default 1, > 0). */
  readonly scale: number;
  /** Arrowhead length on screen, in CSS pixels (default 10, > 0). */
  readonly headSize: number;
  readonly thin: boolean;
}

/** `axis`: two heads, one each way from the point (a direction is an axis); `arrow`: one along the sign-canonical direction (Qt). */
export type PrincipalHead = "axis" | "arrow";

/**
 * The Vectors card: a nodal vector field of the results step on screen as
 * arrows starting at the nodes (state/vectors.ts `buildVectors3D`), or half
 * circles about their axes for a rotational field; sized, coloured and
 * thinned like the reaction arrows, on its own settings.
 */
export interface VectorsState {
  readonly on: boolean;
  /** The field drawn, by its nodal field name (`DISPLACEMENT`, `velocity`); null: the stage's first vector field. */
  readonly field: string | null;
  /** Shaft multiplier (default 1, > 0). */
  readonly scale: number;
  /** Arrowhead length on screen, in CSS pixels (default 12, > 0). */
  readonly headSize: number;
  readonly colorBy: LoadsColorBy;
  readonly thin: boolean;
  /** What each node draws: the resultant, its three global-axis components as separate arrows, or one of them (default resultant). */
  readonly component: ReactionComponent;
}

/**
 * The Springs card: the results step's zero-length spring forces, one arrow per
 * spring at its node along node j's global force (state/springs.ts
 * `buildSprings3D`); sized, coloured and thinned like the reaction arrows.
 */
export interface SpringsState {
  readonly on: boolean;
  /** Shaft multiplier (default 1, > 0). */
  readonly scale: number;
  /** Arrowhead length on screen, in CSS pixels (default 12, > 0). */
  readonly headSize: number;
  /** One colour, or by magnitude (the type colour is the reactions' and loads' own). */
  readonly colorBy: "mono" | "magnitude";
  readonly thin: boolean;
  /** What each spring draws: the resultant, its global-axis components, or one of them (default resultant). */
  readonly component: ReactionComponent;
  /** Why the step on screen draws no spring arrows (its springs' read failed), or null */
  readonly notice: string | null;
}

/** The Display card's BC toggle: restraint glyphs at the supports (state/loads.ts `buildSupports3D`). */
export interface SupportsState {
  readonly on: boolean;
  /** Glyph length multiplier (default 1, > 0). */
  readonly scale: number;
}

/** A reaction arrow per node along the resultant, per global axis (`xyz`), or along one global axis. */
export type ReactionComponent = "resultant" | "xyz" | "x" | "y" | "z";

/** How the load arrows are coloured. */
export type LoadsColorBy = "mono" | "type" | "magnitude";

/** 3-D structural loading and prescribed-SP visualization state. */
export interface LoadsState {
  readonly on: boolean;
  /** User glyph scale multiplier (default 1, > 0). */
  readonly scale: number;
  readonly showNodalForces: boolean;
  readonly showNodalMoments: boolean;
  readonly showElementLoads: boolean;
  readonly showSp: boolean;
  /** Include support-reaction HOLD patterns (`@role = "support_reaction"` and `sp_holds`). */
  readonly includeSupportReactions: boolean;
  /** Scale load magnitudes by the pattern's timeSeries factor λ(t) on a results step (default false: nominal). */
  readonly scaleBySeries: boolean;
  /** Arrowhead length on screen, in CSS pixels (default 12, > 0); every head is 2:1, length to width. */
  readonly headSize: number;
  /** Thin out arrows closer on screen than about a head length (default true). */
  readonly thin: boolean;
  /** Arrow colour: one colour for every load (default), the load type's, or jet over the magnitude. */
  readonly colorBy: LoadsColorBy;
  /** null = all active patterns/cases for current phase/stage; string[] = explicit pattern h5 paths or `neutral:<case>` keys */
  readonly selectedPatterns: readonly string[] | null;
}
