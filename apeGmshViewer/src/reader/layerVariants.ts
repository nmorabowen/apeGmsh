// Layered-shell results as element contour fields (crack plots slice 4): each layer group (reader/ladrunoLayers.ts)
// is offered to the Contour card as three Gauss groups, one value per element and Gauss point:
// - `<result>@+face/<label>`: the outer responding layer on the face toward the positive global axis the shell's
//   normal is most along (a slab's top, a wall's +x or +y side), the red face of the crack plots
//   (state/shellCracks.ts); `@-face` the other outer layer. Both need the element's frame (`MODEL/LOCAL_AXES`): an
//   element without one reads NaN (no data), never a guessed side;
// - `<result>@max/<label>`: the largest value over every responding layer (CRACK_PLOTS_SPEC.md §2.3's "max" variant,
//   meaningful for a quantity like damage, which grows from 0).
// A variant group is a view of its layer group: it lists the same elements, so the one-row-per-element checks
// (reader/ladrunoChecks.ts) skip it, and its rows are its layer group's.
//
// Browser-safe: no Node import.

import type { GaussGroupInfo, LayerGroupInfo } from "./results.ts";

export const LAYER_VARIANTS = ["+face", "-face", "max"] as const;
export type LayerVariant = (typeof LAYER_VARIANTS)[number];

/** The variant group id of a layer group: `<layer group id>@<variant>`. */
export const variantGroupId = (layerGroupId: string, v: LayerVariant) => `${layerGroupId}@${v}`;

/** A variant group id's layer group and variant; null for any other group id. */
export function parseVariantGroup(groupId: string): { layerGroupId: string; variant: LayerVariant } | null {
  const at = groupId.lastIndexOf("@");
  if (at < 0) return null;
  const v = groupId.slice(at + 1) as LayerVariant;
  return LAYER_VARIANTS.includes(v) ? { layerGroupId: groupId.slice(0, at), variant: v } : null;
}

/** A layer group component (`<result>/<label>`) under a variant: `<result>@<variant>/<label>`. */
export function variantComponent(component: string, v: LayerVariant): string {
  const slash = component.lastIndexOf("/");
  return `${component.slice(0, slash)}@${v}${component.slice(slash)}`;
}

/** The layer group component a variant component names. */
export function layerComponentOf(component: string): string {
  return component.replace(/@(\+face|-face|max)\//, "/");
}

/**
 * The variant groups of a stage's layer groups. `framed(g)` says whether the file records a frame for any element
 * of g; without one, the face variants are refused by name (their components listed in `refused`).
 */
export function variantGroupsOf(layers: readonly LayerGroupInfo[], framed: (g: LayerGroupInfo) => boolean): GaussGroupInfo[] {
  const out: GaussGroupInfo[] = [];
  for (const g of layers) {
    if (g.refused.length) continue;
    const hasFrames = framed(g);
    for (const v of LAYER_VARIANTS) {
      const comps = g.components.map((c) => variantComponent(c, v));
      const refuse = v !== "max" && !hasFrames;
      out.push({
        id: variantGroupId(g.id, v),
        path: g.path,
        classTag: g.classTag,
        intRule: g.intRule,
        nElements: g.nElements,
        nGauss: g.nGauss,
        components: refuse ? [] : comps,
        refused: refuse
          ? comps.map((component) => ({ component, reason: "the file records no frame for these shells (MODEL/LOCAL_AXES, written by the fork since nmorabowen/OpenSees#1133), so their faces cannot be told apart" }))
          : [],
        localAxes: false,
        naturalCoords: g.naturalCoords,
        layerVariant: { layerGroupId: g.id, variant: v },
      });
    }
  }
  return out;
}

/**
 * Reduce one component of a layer group read (`values`: (T, E, G, L) for that component, every layer) to the variant:
 * (T, E, G). `plusLayer[e]` is the responding layer on element e's + face (-1: no frame, the value is NaN);
 * the - face is the other outer layer.
 */
export function reduceLayers(values: ArrayLike<number>, T: number, E: number, G: number, L: number, v: LayerVariant, plusLayer: Int32Array | null): Float64Array {
  const out = new Float64Array(T * E * G);
  for (let t = 0; t < T; t++)
    for (let e = 0; e < E; e++) {
      const pl = plusLayer ? plusLayer[e]! : -1;
      const k = v === "+face" ? pl : v === "-face" ? (pl < 0 ? -1 : pl === 0 ? L - 1 : 0) : -2;
      for (let g = 0; g < G; g++) {
        const row = ((t * E + e) * G + g) * L;
        let x: number;
        if (k === -2) {
          x = -Infinity;
          for (let j = 0; j < L; j++) x = Math.max(x, values[row + j]!);
        } else x = k < 0 ? NaN : values[row + k]!;
        out[(t * E + e) * G + g] = x;
      }
    }
  return out;
}

/** Each element's + face layer from its frame (rows e1, e2, e3 of the scalar-first quaternion): last when e3 points to the positive axis it is most along. */
export function plusLayersOf(quaternions: ArrayLike<number>, recorded: ArrayLike<number>, L: number): Int32Array {
  const E = recorded.length;
  const out = new Int32Array(E).fill(-1);
  for (let e = 0; e < E; e++) {
    if (!recorded[e]) continue;
    const w = quaternions[4 * e]!, x = quaternions[4 * e + 1]!, y = quaternions[4 * e + 2]!, z = quaternions[4 * e + 3]!;
    const e3 = [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)];
    const k = [0, 1, 2].reduce((a, b) => (Math.abs(e3[b]!) > Math.abs(e3[a]!) ? b : a), 0);
    out[e] = e3[k]! > 0 ? L - 1 : 0;
  }
  return out;
}
