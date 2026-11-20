// The Cracks card's layered-shell Gauss points on the mesh (state/shellCracks.ts), placed as the Principal card
// places its points (state/principal.ts `principalPointsOf`). Apart from state/shellCracks.ts so that state/selectors.ts,
// which imports the field, never reaches state/gaussPoints.ts (it imports the selectors).

import type { BlobStore } from "./blobs.ts";
import { cellOfDecl, GaussLayoutBuilder, shownElementOf, type GaussLayout } from "./gaussPoints.ts";
import type { ShellCrackField, ShellPointIndex } from "./shellCracks.ts";
import type { State } from "./types.ts";

/** The field's Gauss points: the layout, and each point's group (index into `field.groups`), element row and Gauss point. */
export interface ShellCrackPoints extends ShellPointIndex {
  layout: GaussLayout;
}

/** Place the field's Gauss points on the mesh (as the Principal card's points: state/principal.ts `principalPointsOf`). */
export function shellCrackPointsOf(s: State, blobs: BlobStore, field: ShellCrackField): ShellCrackPoints | null {
  const at = s.phase.at;
  const stage = at?.kind === "results" ? s.results?.stages[at.stage] : undefined;
  if (!stage || !s.mesh) return null;
  const b = new GaussLayoutBuilder(cellOfDecl(s.decls, s.blocks, blobs), shownElementOf(s));
  const group: number[] = [], element: number[] = [], gauss: number[] = [];
  field.groups.forEach((fg, gi) => {
    const g = stage.layers?.find((x) => x.id === fg.id);
    if (!g || !g.elementIds) return;
    if (!g.naturalCoords || g.naturalCoords.length !== g.nGauss) return b.refuse("no Gauss point coordinates in the file", g.nElements);
    const eids = blobs.get(g.elementIds);
    if (!(eids instanceof Float64Array)) throw new TypeError(`shell cracks: ${g.elementIds.id} is not f64`);
    eids.forEach((id, e) => {
      const n0 = b.count;
      b.add(`mesh/element/${id}`, g.naturalCoords!);
      for (let k = 0; k < b.count - n0; k++) {
        group.push(gi);
        element.push(e);
        gauss.push(k);
      }
    });
  });
  return { layout: b.build(), group: Int32Array.from(group), element: Int32Array.from(element), gauss: Int32Array.from(gauss) };
}

/** gmsh's 4-node quadrangle: the only cell the crack lines locate their vertices in (state/crackLines.ts `invertQuad`). */
const GMSH_QUAD4 = 3;

/** An element's four corners in reference (undeformed) positions, gmsh corner order, by element id; null for any other cell. */
export function elementCornersOf(s: State, blobs: BlobStore): (elementId: number) => Float64Array | null {
  const cellOf = cellOfDecl(s.decls, s.blocks, blobs);
  const nodes = s.mesh?.nodes;
  if (!nodes) return () => null;
  const ids = blobs.get(nodes.ids) as Float64Array;
  const xyz = blobs.get(nodes.xyz) as Float64Array;
  const column = new Map<number, number>();
  ids.forEach((id, j) => column.set(id, j));
  return (id) => {
    const c = cellOf(`mesh/element/${id}`);
    if (!c || c.code !== GMSH_QUAD4 || c.nodes.length < 4) return null;
    const out = new Float64Array(12);
    for (let k = 0; k < 4; k++) {
      const j = column.get(c.nodes[k]!);
      if (j === undefined) return null;
      out.set(xyz.subarray(3 * j, 3 * j + 3), 3 * k);
    }
    return out;
  };
}

/**
 * The traced lines' vertices as Gauss-style points (one per vertex, at its element's natural coordinates), so the
 * renderer places them, deformed or not, as it places Gauss points. `placed[v]` is vertex v's point, or -1 when its
 * element could not be placed (it is then not drawn).
 */
export function lineVertexLayoutOf(s: State, blobs: BlobStore, elementId: Float64Array, xi: Float64Array): { layout: GaussLayout; placed: Int32Array } {
  const b = new GaussLayoutBuilder(cellOfDecl(s.decls, s.blocks, blobs), () => true);
  const placed = new Int32Array(elementId.length).fill(-1);
  for (let v = 0; v < elementId.length; v++) {
    const n0 = b.count;
    b.add(`mesh/element/${elementId[v]}`, [[xi[2 * v]!, xi[2 * v + 1]!]]);
    if (b.count > n0) placed[v] = n0;
  }
  return { layout: b.build(), placed };
}
