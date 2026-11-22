// The Cracks card's layered shells (renderer/cracks.ts `CracksLayer`): `section.fiber.crackPattern` on a LayeredShell
// section records, at each Gauss point and responding (concrete) layer, up to three crack-opening vectors in the
// shell's material frame (e1, e2 in plane, e3 the normal; ASDConcrete3D `getCrackPattern`, via PlateFromPlaneStress).
// Each crack is drawn as a short tick on a face of the shell, along its trace t = e3 x n (n the crack normal in world
// axes), centred at the Gauss point: the first prototype of CRACK_PLOTS_SPEC.md §3.2.
//
// Faces: the outermost concrete layer on each side (layer 0 is on the -e3 side, the last on the +e3 side). The face
// shown red is the one facing the positive global axis the shell's normal is most along (a slab's top, a wall's
// +x or +y side), blue the other (CRACK_PLOTS_SPEC.md §2.3 `want_last`, §3.4 `FACE_COLORS`), so the colour means the
// same side of every panel, whatever its local frame.
//
// The frame is the file's own (`MODEL/LOCAL_AXES`, nmorabowen/OpenSees#1133). An element without one is never drawn
// with a guessed frame: it is counted and named. Pure: tested without a window.

import { CRACK_FACE, hexToRgb } from "../theme/tokens.ts";
import { crackSlotsOf } from "./cracks.ts";
import type { LayerGroup } from "./types.ts";

/** Which faces the ticks are drawn on. */
export type ShellFace = "both" | "plus" | "minus";
export const SHELL_FACES: readonly ShellFace[] = ["both", "plus", "minus"];

/** sRGB 0..1: the face toward the positive axis and the other (theme/tokens.ts `CRACK_FACE`, CRACK_PLOTS_SPEC.md §3.4). */
export const FACE_RGB = { plus: hexToRgb(CRACK_FACE.plus), minus: hexToRgb(CRACK_FACE.minus) } as const;

/** One layer group of the field: its nine crack components, slot-major. */
export interface ShellCrackGroup {
  id: string;
  slots: string[];
}

/** The stage's layered-shell crack field: the groups drawn, the components every one shares, and why a group is not drawn. */
export interface ShellCrackField {
  groups: ShellCrackGroup[];
  components: string[];
  refused: string[];
}

/** The shell crack field of a stage's layer groups; null when no group records a crack pattern. */
export function shellCrackFieldOf(layers: readonly LayerGroup[]): ShellCrackField | null {
  const out: ShellCrackField = { groups: [], components: [], refused: [] };
  for (const g of layers) {
    if (g.refused.length) {
      if (/crack/i.test(g.refused[0]!.component)) out.refused.push(`${g.id}: ${g.refused[0]!.reason}`);
      continue;
    }
    const r = crackSlotsOf(g.components);
    if (r === null) continue;
    if ("partial" in r) {
      out.refused.push(`${g.id}: ${r.partial}`);
      continue;
    }
    if (out.groups.length && r.slots.join(",") !== out.components.join(",")) {
      out.refused.push(`${g.id}: names its crack components ${r.slots[0]}... where ${out.groups[0]!.id} names them ${out.components[0]}...`);
      continue;
    }
    if (!g.elementIds || !g.quaternions || !g.recorded) {
      out.refused.push(`${g.id}: its element rows are not verified as the model's (the file's index space is not stamped)`);
      continue;
    }
    // The crack vectors are in the shell's frame: with none recorded they cannot be placed (never a guessed frame).
    const why = "the file records no frame for them (MODEL/LOCAL_AXES; the fork writes shell frames from nmorabowen/OpenSees#1133)";
    if ((g.unframed ?? 0) >= g.nElements) {
      out.refused.push(`${g.id}: ${why}`);
      continue;
    }
    if ((g.unframed ?? 0) > 0) out.refused.push(`${g.unframed} of the ${g.nElements} elements of ${g.id}: ${why}`);
    out.groups.push({ id: g.id, slots: r.slots });
    if (!out.components.length) out.components = [...r.slots];
  }
  return out.groups.length || out.refused.length ? out : null;
}

/** The rows of the rotation a scalar-first unit quaternion (global to local) stands for: e1, e2, e3 in world axes. */
export function frameRows(w: number, x: number, y: number, z: number): [number[], number[], number[]] {
  // The rotation matrix of the quaternion, row by row (row 2 is reader/ladrunoLocalAxes.ts `quaternionToZRec`).
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
  ];
}

/** Which point is which: each placed Gauss point's group (index into the field's groups), element row and Gauss point. */
export interface ShellPointIndex {
  group: Int32Array;
  element: Int32Array;
  gauss: Int32Array;
}

/** One group's step and frames, as the builder reads them. */
export interface ShellTickGroup {
  /** (E, G, layers read, 9) */
  values: Float64Array;
  /** the responding layers read, in `values` order (main reads the first and the last) */
  layers: readonly number[];
  nLayers: number;
  nGauss: number;
  /** 4 per element */
  quaternions: Float64Array;
  recorded: Uint8Array;
  /** per element; null when not placed */
  thickness: Float64Array | null;
}

export interface ShellTickOptions {
  face: ShellFace;
  /** an opening below this is not drawn (>= 0) */
  minOpening: number;
  /** half a tick's length, in world units */
  halfLength: number;
  /** the hair a tick sits off its face, so it is drawn over it */
  eps: number;
}

export interface ShellTicks {
  /** 2 vertices per tick, 3 floats each */
  positions: Float32Array;
  /** sRGB 0..1, 3 per vertex */
  colours: Float32Array;
  ticks: number;
  /** the Gauss points with a tick */
  points: number;
  /** the Gauss points of elements with no recorded frame (never drawn) */
  unframed: number;
  maxOpening: number;
}

/**
 * The ticks of one step: at each placed Gauss point (`base`, deformed or not), on each face asked, one tick per crack
 * with an opening > 0 and >= `minOpening`, along its trace, `halfLength` each way, offset along e3 to its face: t/2 + eps
 * off the mid-surface, where the outer layer is (the face of the slab renderer/shellSlab.ts draws when the shells are
 * extruded); eps alone when the section's thickness is not known. Drawn at the face whether or not the slab is, a
 * tick on the far face stays behind the mid-surface, never through it.
 */
export function buildShellTicks(base: Float32Array, points: ShellPointIndex, groups: readonly ShellTickGroup[], opts: ShellTickOptions): ShellTicks {
  const pos: number[] = [];
  const col: number[] = [];
  const hit = new Set<number>();
  let unframed = 0;
  let maxOpening = 0;
  const n = points.group.length;
  for (let i = 0; i < n; i++) {
    const px = base[3 * i]!, py = base[3 * i + 1]!, pz = base[3 * i + 2]!;
    if (!Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(pz)) continue;
    const g = groups[points.group[i]!]!;
    const e = points.element[i]!;
    if (!g.recorded[e]) {
      unframed++;
      continue;
    }
    const q = g.quaternions;
    const [e1, e2, e3] = frameRows(q[4 * e]!, q[4 * e + 1]!, q[4 * e + 2]!, q[4 * e + 3]!);
    // The global axis the normal is most along; the +e3 layer faces its positive side when e3 points that way.
    const k = [0, 1, 2].reduce((a, b) => (Math.abs(e3[b]!) > Math.abs(e3[a]!) ? b : a), 0);
    const lastIsPlus = e3[k]! > 0;
    const nRead = g.layers.length;
    const lastAt = nRead - 1;
    // Each face: which read layer, its side along e3 (+1 for the +e3 layer), and its colour.
    const faces: { at: number; side: number; rgb: readonly number[] }[] = [];
    const plus = { at: lastIsPlus ? lastAt : 0, side: lastIsPlus ? 1 : -1, rgb: FACE_RGB.plus };
    const minus = { at: lastIsPlus ? 0 : lastAt, side: lastIsPlus ? -1 : 1, rgb: FACE_RGB.minus };
    if (opts.face !== "minus") faces.push(plus);
    if (opts.face !== "plus" && (nRead > 1 || opts.face === "minus")) faces.push(minus);
    const half = g.thickness ? g.thickness[e]! / 2 : 0;
    const gp = points.gauss[i]!;
    for (const f of faces) {
      const lift = f.side * (half + opts.eps);
      const cx = px + lift * e3[0]!, cy = py + lift * e3[1]!, cz = pz + lift * e3[2]!;
      const at = ((e * g.nGauss + gp) * nRead + f.at) * 9;
      for (let j = 0; j < 3; j++) {
        const a = g.values[at + 3 * j]!, b = g.values[at + 3 * j + 1]!, c = g.values[at + 3 * j + 2]!;
        const open = Math.hypot(a, b, c);
        if (!(open > 0) || open < opts.minOpening) continue;
        // The crack normal in world axes, then its trace on the shell surface.
        const nx = (a * e1[0]! + b * e2[0]! + c * e3[0]!) / open;
        const ny = (a * e1[1]! + b * e2[1]! + c * e3[1]!) / open;
        const nz = (a * e1[2]! + b * e2[2]! + c * e3[2]!) / open;
        let tx = e3[1]! * nz - e3[2]! * ny, ty = e3[2]! * nx - e3[0]! * nz, tz = e3[0]! * ny - e3[1]! * nx;
        const tl = Math.hypot(tx, ty, tz);
        if (!(tl > 1e-9)) continue;
        tx /= tl;
        ty /= tl;
        tz /= tl;
        const h = opts.halfLength;
        pos.push(cx - h * tx, cy - h * ty, cz - h * tz, cx + h * tx, cy + h * ty, cz + h * tz);
        col.push(...f.rgb, ...f.rgb);
        hit.add(i);
        maxOpening = Math.max(maxOpening, open);
      }
    }
  }
  return { positions: Float32Array.from(pos), colours: Float32Array.from(col), ticks: pos.length / 6, points: hit.size, unframed, maxOpening };
}
