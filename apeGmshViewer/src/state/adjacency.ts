// Which colour groups are neighbours in the view (#1330, maintainer ruling
// on R2): two groups are adjacent when an element of one and an element of
// the other share a node. Computed once at load from the file's mesh, kept
// in the state as plain pairs (MeshInfo.adjacency), never held by a view.

import type { ModelFile } from "../model/types.ts";

/**
 * Adjacent pairs of colour groups, as indices into `order` (the colour
 * group of each cell is `byElement`, from mesh/build.ts `colourGroups`).
 * Each pair once, `a < b`, sorted.
 */
export function groupAdjacency(model: ModelFile, byElement: ReadonlyMap<number, number>, groupCount: number): [number, number][] {
  // The groups touching each node.
  const touching = new Map<number, Set<number>>();
  for (const b of model.blocks) {
    for (let row = 0; row < b.ids.length; row++) {
      const gi = byElement.get(b.ids[row]!);
      if (gi === undefined) continue;
      for (let k = 0; k < b.npe; k++) {
        const node = b.connectivity[row * b.npe + k]!;
        const set = touching.get(node);
        if (set) set.add(gi);
        else touching.set(node, new Set([gi]));
      }
    }
  }
  const pairs = new Set<number>();
  for (const set of touching.values()) {
    if (set.size < 2) continue;
    const list = [...set].sort((x, y) => x - y);
    for (let i = 0; i < list.length; i++) for (let j = i + 1; j < list.length; j++) pairs.add(list[i]! * groupCount + list[j]!);
  }
  return [...pairs].sort((x, y) => x - y).map((p) => [Math.floor(p / groupCount), p % groupCount]);
}
