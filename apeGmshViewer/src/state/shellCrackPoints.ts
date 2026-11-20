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
