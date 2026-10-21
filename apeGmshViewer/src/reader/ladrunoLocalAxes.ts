// `.ladruno` reference local frames: `MODEL/LOCAL_AXES/<class>/{ID, FRAME}` (or the
// flat layout, one `ID` and `FRAME` on `MODEL/LOCAL_AXES` itself), and the decode of
// one LOCAL_AXES row, which `RESULTS/ON_ELEMENTS/LOCAL_AXES` shares (#2321).
//
// A row is a scalar-first unit quaternion (4 columns), a 3x3 row-major rotation
// matrix (9), or the local z axis alone (3); a row that is not finite decodes to
// nothing, and the element keeps no frame from it.
//
// An element's reference frame is listed once: once in one class's ID, and in no
// other class's (#2321). An id twice in one ID, or in two classes' groups (or in the
// flat ID and a class group), would hand the element the last row's frame and leave
// the element whose row it was without one, so it is refused by name (the file, each
// ID dataset, the ids and their rows) through reader/ladrunoChecks.ts, before any
// frame is kept. The check fires where the frames are collected, on readLocalAxes
// (and so on any line-station read, which attaches them), not at open:
// MODEL/LOCAL_AXES is read nowhere else, and open does not read it. Only the groups
// whose frames are read are listed (one with an ID and a FRAME or LOCAL_AXES whose
// rows match it), as reader/ladrunoElements.ts lists only buckets a read takes.
//
// Browser-safe: no Node import, and no HDF5 access (W5): the caller passes its own
// group, dataset and dtype reads, as reader/ladrunoNodal.ts does, so this module
// imports nothing from ladruno.ts.

import { uniqueElementIds } from "./ladrunoChecks.ts";

/** Where an element's frame was read: `RESULTS/ON_ELEMENTS/LOCAL_AXES` at a step, or `MODEL/LOCAL_AXES`. */
export type LocalAxesSource = "step_local_axes" | "model_local_axes";

/** The group the reads walk: its path and its children's names. */
export interface LocalAxesGroup {
  path: string;
  keys(): string[];
}

/** A dataset the reads take: its path, its shape and its values. */
export interface LocalAxesDataset {
  path: string;
  shape: number[] | null;
  value: unknown;
}

/** The caller's reads: a child group (null when absent or not a group), a dataset (refused when absent), and the dtype checks. */
export interface LocalAxesReads {
  optionalGroup(parent: LocalAxesGroup, name: string): LocalAxesGroup | null;
  dataset(parent: LocalAxesGroup, name: string): LocalAxesDataset;
  ints(ds: LocalAxesDataset): Float64Array;
  floats(v: unknown, where: string): Float64Array;
  /** The file's name, for a refusal. */
  file: string;
  /** The caller's error for a refusal (a LadrunoError of kind "layout"). */
  fail(message: string): Error;
}

/** Fill the three maps from `MODEL/LOCAL_AXES` (`modelLa`): each element's quaternion (when the row has one), its local z, and its source. */
export function collectModelLocalAxes(
  modelLa: LocalAxesGroup,
  reads: LocalAxesReads,
  idToQuat: Map<number, [number, number, number, number]>,
  idToZRec: Map<number, [number, number, number]>,
  idToSource: Map<number, LocalAxesSource>,
): void {
  // One entry per group whose frames are read (`at` is its ID dataset): sized by the class count, not the id count.
  const read: { at: string; ids: Float64Array; raw: Float64Array; cols: number }[] = [];
  const readSub = (sub: LocalAxesGroup) => {
    const frameName = sub.keys().includes("FRAME") ? "FRAME" : sub.keys().includes("LOCAL_AXES") ? "LOCAL_AXES" : null;
    if (!sub.keys().includes("ID") || !frameName) return;
    const idDs = reads.dataset(sub, "ID");
    const frameDs = reads.dataset(sub, frameName);
    const ids = reads.ints(idDs);
    const raw = reads.floats(frameDs.value, frameDs.path);
    const nE = ids.length;
    const frameShape = frameDs.shape ?? [];
    if (frameShape.length >= 2 && frameShape[0] !== nE) return;
    if (nE <= 0 || raw.length === 0 || raw.length % nE !== 0) return;
    read.push({ at: idDs.path, ids, raw, cols: raw.length / nE });
  };
  if (modelLa.keys().includes("ID")) {
    readSub(modelLa);
  }
  for (const k of modelLa.keys()) {
    const sub = reads.optionalGroup(modelLa, k);
    if (sub) readSub(sub);
  }
  uniqueElementIds(
    read,
    `${reads.file}: ${modelLa.path}`,
    "the recorder writes each element's reference frame once, under one class, so a repeat would give one element another row's frame and leave the other without one; the local axes are not read",
    reads.fail,
  );
  for (const { ids, raw, cols } of read) {
    for (let i = 0; i < ids.length; i++) {
      const eid = ids[i]!;
      const decoded = decodeLocalAxesRow(raw, i * cols, cols);
      if (decoded) {
        if (decoded.quat) idToQuat.set(eid, decoded.quat);
        else idToQuat.delete(eid);
        idToZRec.set(eid, decoded.zRec);
        idToSource.set(eid, "model_local_axes");
      }
    }
  }
}

/**
 * Convert a scalar-first unit quaternion `(w, x, y, z)` (global -> local,
 * schema §5, where rows `(0, 1, 2)` of the 3x3 matrix are `(x_local, y_local, z_local)`)
 * into row 2 (`z_local` in global coords), matching Python `LocalAxes.matrices`
 * (`src/apeGmsh/results/_slabs.py`) and `axes_from_quaternion` (`_beam_geometry.py`).
 */
export function quaternionToZRec(w: number, x: number, y: number, z: number): [number, number, number] {
  const norm = Math.hypot(w, x, y, z);
  if (!Number.isFinite(norm) || norm < 1e-15) return [Number.NaN, Number.NaN, Number.NaN];
  const wn = w / norm;
  const xn = x / norm;
  const yn = y / norm;
  const zn = z / norm;
  return [
    2 * (xn * zn - yn * wn),
    2 * (yn * zn + xn * wn),
    1 - 2 * (xn * xn + yn * yn),
  ];
}

export function decodeLocalAxesRow(
  raw: Float64Array,
  offset: number,
  cols: number,
): { quat: [number, number, number, number] | null; zRec: [number, number, number] } | null {
  if (cols === 4) {
    const w = raw[offset]!;
    const x = raw[offset + 1]!;
    const y = raw[offset + 2]!;
    const z = raw[offset + 3]!;
    if (!Number.isFinite(w) || !Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return null;
    const zRec = quaternionToZRec(w, x, y, z);
    if (!Number.isFinite(zRec[0])) return null;
    return { quat: [w, x, y, z], zRec };
  }
  if (cols === 9) {
    // 3x3 row-major rotation matrix [x_local(3), y_local(3), z_local(3)]: row 2 is cols 6..8.
    const zx = raw[offset + 6]!;
    const zy = raw[offset + 7]!;
    const zz = raw[offset + 8]!;
    if (!Number.isFinite(zx) || !Number.isFinite(zy) || !Number.isFinite(zz)) return null;
    return { quat: null, zRec: [zx, zy, zz] };
  }
  if (cols === 3) {
    const zx = raw[offset]!;
    const zy = raw[offset + 1]!;
    const zz = raw[offset + 2]!;
    if (!Number.isFinite(zx) || !Number.isFinite(zy) || !Number.isFinite(zz)) return null;
    return { quat: null, zRec: [zx, zy, zz] };
  }
  return null;
}
