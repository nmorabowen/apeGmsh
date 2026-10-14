// The viewport's side of the section planes (V4e1, #1834, ADR 0083 Parts 3
// and 5): three.js clipping planes built from the store's clip slice, put on
// every drawn material, and the pick filter.
//
// One array of `THREE.Plane` is shared by every material in the scene. It is
// stamped on before every render (`beforeRender`), over the whole scene, not
// when the planes change: an object created after a plane exists (a stage
// change, a re-read, the ghost toggled, a new selection outline, a contour
// rebuild) is cut from its first frame, with no creation site to remember
// (ADR 0083 C-REBUILD and C-SUBSTRATE, which needed a stamp at every creation
// site in the Qt viewer). The planes move in place, so a plane change costs no
// re-stamp; three.js recompiles a material's program only when the number of
// planes changes. HTML chrome (the legend, the panels) is not in the scene and
// is never cut; the halo's ring pass is its own scene, so it rings only what
// the cut kept (its mask is the outline, cut).
//
// three.js does not cut raycasts, so a pick is filtered here: a hit on the
// discarded side of any active plane is a miss (C-PICK). Picking through the
// hole onto geometry behind it is not attempted (ADR 0083 open question 3).

import * as THREE from "three";
import type { BlobStore } from "../state/blobs.ts";
import { boxOf, cutsOf, isClipped, type Box, type Cut } from "../state/clip.ts";
import type { ClipState, MeshInfo } from "../state/types.ts";
import { basePositions, type FullPositions } from "./deform.ts";

/** The clip planes of the viewport, shared by every drawn material. */
export class SectionClip {
  /** The one array every drawn material's `clippingPlanes` is (moved in place on a plane change). */
  readonly planes: THREE.Plane[] = [];
  private cuts: Cut[] = [];
  private key = "";

  /** Follow the store's clip slice; true when the cut changed (the viewport renders again). */
  sync(clip: ClipState): boolean {
    const cuts = cutsOf(clip);
    const key = JSON.stringify(cuts);
    if (key === this.key) return false;
    this.key = key;
    this.cuts = cuts;
    this.planes.length = cuts.length;
    cuts.forEach((c, i) => {
      const p = this.planes[i] ?? new THREE.Plane();
      p.normal.set(c.normal[0], c.normal[1], c.normal[2]);
      p.constant = c.constant;
      this.planes[i] = p;
    });
    return true;
  }

  /** How many planes cut now. */
  get count(): number {
    return this.cuts.length;
  }

  /**
   * Put the planes on every material under `root`, every frame: what was
   * created since the last frame is cut from its first one. Returns the
   * number of materials it newly stamped.
   */
  beforeRender(root: THREE.Object3D): number {
    let stamped = 0;
    root.traverse((o) => {
      const m = (o as THREE.Mesh).material as THREE.Material | THREE.Material[] | undefined;
      if (!m) return;
      for (const x of Array.isArray(m) ? m : [m]) {
        if (x.clippingPlanes === this.planes) continue;
        x.clippingPlanes = this.planes;
        stamped++;
      }
    });
    return stamped;
  }

  /** Whether a world point is on the discarded side of a plane that cuts now (`tol`: the on-plane band). */
  clipped(p: THREE.Vector3, tol: number): boolean {
    return this.cuts.length > 0 && isClipped(this.cuts, [p.x, p.y, p.z], tol);
  }
}

/**
 * The box the offset slider maps (ADR 0083 Part 4): the reference mesh, the
 * undeformed positions, whatever `full` (the viewport's drawn positions,
 * deformed while the deformed shape shows) draws now. The slider must not
 * jump as the shape deforms or the scale changes; the numeric field beside
 * it is unclamped for the geometry the deformation pushes outside this box.
 */
export function sliderBox(full: FullPositions | null, info: MeshInfo, blobs: BlobStore): Box | null {
  void full; // taken, and ignored, so the choice is explicit where the drawn positions are at hand (test/clip)
  if (BOXES.has(info)) return BOXES.get(info)!;
  const base = basePositions(info, blobs);
  const box = boxOf(base.line, base.tri);
  BOXES.set(info, box);
  return box;
}

/** The reference box per mesh read (a MeshInfo is never mutated; a re-read is a new one). */
const BOXES = new WeakMap<MeshInfo, Box | null>();
