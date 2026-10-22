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
// A group whose frames cannot be paired with its ids is not read, and says so (#2336).
// A FRAME whose row count is not the ID's (more rows or fewer), or whose rows are not
// 3, 4 or 9 wide, warns by name (the file, the dataset, both counts) through the
// read's warningsOut, and at open, from the shapes alone, through the line-station
// listing of a stage that has line stations (warnModelLocalAxes); its elements keep no recorded frame, and a diagram computes
// theirs from the model (the geomTransf vecxz, or the default), as for any element
// the file gives no frame. Python and MPCO do the same (#2090). An empty ID beside
// FRAME rows is refused by name: those frames belong to no element.
//
// A per-step bucket (`RESULTS/ON_ELEMENTS/LOCAL_AXES`, which the fork at FORK_PIN
// does not write) has one layout, ruled on #2451: one group per element class,
// `<classTag>-<ClassName>[<g>:<h>]/{ID[nE], DATA[steps, nE, 3|4|9], STEP[steps]?,
// TIME[steps]?}`, the chunked form fork schema §7.0 gives every streaming group, in
// §7.2's per-class buckets. Schema §5 still draws the superseded per-step form
// (`ID` and `DATA/STEP_<k>` on LOCAL_AXES itself); that form, and any ID, DATA, STEP or
// TIME on LOCAL_AXES itself, is refused by name. It has no fallback either (#2349): at
// a step the stored frame is the answer, and the reference frame in its place is a
// wrong frame for a co-rotational element, warned or not. So a bucket that cannot be
// paired with its ids, or with the stage's steps, is refused by name (the file, the
// bucket, the step): ID or DATA missing or not a dataset (a DATA group of STEP_<k>
// datasets is named as §5's form), a DATA that is not (steps, elements, 3|4|9), a
// step or row count that is not the stage's or the ID's, a STEP or TIME that is not a
// dataset of one value per step (either may be absent), an empty ID beside rows, or a
// row that does not decode (not finite, or a zero quaternion). (An id twice in the
// step's buckets is refused at open, as in any ON_ELEMENTS token, by
// reader/ladrunoElements.ts.) A step-less read does not read them.
//
// The frames are resolved with sorted typed arrays, never a Set or Map sized by the
// element count, which V8 caps at 2^24 entries (as reader/ladrunoChecks.ts does).
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
  /** A child dataset, or null when absent or not a dataset. */
  optionalDataset(parent: LocalAxesGroup, name: string): LocalAxesDataset | null;
  /** One step of a `(steps, elements, components)` dataset, as float64. */
  stepSlice(ds: LocalAxesDataset, step: number): Float64Array;
  ints(ds: LocalAxesDataset): Float64Array;
  floats(v: unknown, where: string): Float64Array;
  /** The file's name, for a refusal or a warning. */
  file: string;
  /** The caller's error for a refusal (a LadrunoError of kind "layout"). */
  fail(message: string): Error;
  /** The read's warningsOut: a group that is not read says why here. */
  warn(message: string): void;
}

/** One group's frame rows, as read: its ID dataset (`at`), its ids, its rows (`cols` wide) and where they were read. */
export interface FrameRows {
  at: string;
  ids: Float64Array;
  raw: Float64Array;
  cols: number;
  source: LocalAxesSource;
}

/** The row widths decodeLocalAxesRow decodes: the local z axis, a quaternion, a rotation matrix. */
const FRAME_WIDTHS = [3, 4, 9];

/** Call `fn` on each group of `MODEL/LOCAL_AXES` (the flat layout first) that holds an ID and a FRAME (or LOCAL_AXES). */
function eachFrameGroup(modelLa: LocalAxesGroup, reads: LocalAxesReads, fn: (idDs: LocalAxesDataset, frameDs: LocalAxesDataset) => void): void {
  const visit = (sub: LocalAxesGroup) => {
    const frameName = sub.keys().includes("FRAME") ? "FRAME" : sub.keys().includes("LOCAL_AXES") ? "LOCAL_AXES" : null;
    if (!sub.keys().includes("ID") || !frameName) return;
    fn(reads.dataset(sub, "ID"), reads.dataset(sub, frameName));
  };
  if (modelLa.keys().includes("ID")) {
    visit(modelLa);
  }
  for (const k of modelLa.keys()) {
    const sub = reads.optionalGroup(modelLa, k);
    if (sub) visit(sub);
  }
}

/** What FRAME holds: its rows when it is 2-D, else its value count. */
const heldBy = (rows: number | null, values: number): string => (rows !== null ? `${rows} row(s)` : `${values} value(s)`);

/** Why a FRAME of `rows` rows (null when not 2-D) and `values` values cannot be paired with the `nE` (> 0) ids of `idDs`, or null when it can. */
function frameFault(nE: number, rows: number | null, values: number, idDs: LocalAxesDataset): string | null {
  const held = heldBy(rows, values);
  if (rows !== null && rows !== nE) return `holds ${held} for the ${nE} id(s) of ${idDs.path}`;
  if (values === 0 || values % nE !== 0) return `holds ${held}, not one whole row for each of the ${nE} id(s) of ${idDs.path}`;
  if (!FRAME_WIDTHS.includes(values / nE)) return `holds rows ${values / nE} wide for the ${nE} id(s) of ${idDs.path}, not 3 (the local z axis), 4 (a quaternion) or 9 (a rotation matrix)`;
  return null;
}

const notRead = (reads: LocalAxesReads, frameDs: LocalAxesDataset, fault: string): string =>
  `${reads.file}: ${frameDs.path} ${fault}; its frames are not read, so its elements have no recorded frame and a diagram computes theirs from the model (the geomTransf vecxz, or the default)`;

const frameRowsOf = (shape: number[] | null): number | null => (shape && shape.length >= 2 ? shape[0]! : null);

/** The frame rows of `MODEL/LOCAL_AXES` (`modelLa`), one entry per group read: sized by the class count, not the id count. */
export function collectModelLocalAxes(modelLa: LocalAxesGroup, reads: LocalAxesReads): FrameRows[] {
  const read: FrameRows[] = [];
  eachFrameGroup(modelLa, reads, (idDs, frameDs) => {
    const ids = reads.ints(idDs);
    const raw = reads.floats(frameDs.value, frameDs.path);
    const nE = ids.length;
    const rows = frameRowsOf(frameDs.shape);
    if (nE === 0) {
      if (raw.length === 0) return;
      throw reads.fail(`${reads.file}: ${idDs.path} is empty, but ${frameDs.path} holds ${heldBy(rows, raw.length)}; a frame row belongs to the element its ID row names, so these belong to no element; the local axes are not read`);
    }
    const fault = frameFault(nE, rows, raw.length, idDs);
    if (fault !== null) {
      reads.warn(notRead(reads, frameDs, fault));
      return;
    }
    read.push({ at: idDs.path, ids, raw, cols: raw.length / nE, source: "model_local_axes" });
  });
  uniqueElementIds(
    read,
    `${reads.file}: ${modelLa.path}`,
    "the recorder writes each element's reference frame once, under one class, so a repeat would give one element another row's frame and leave the other without one; the local axes are not read",
    reads.fail,
  );
  return read;
}

/**
 * The open-time half of collectModelLocalAxes: from the datasets' shapes alone (no
 * values read), warn through `reads.warn` of each group whose FRAME cannot be paired
 * with its ID, in the words a read uses. An empty ID is left to the read, which refuses it.
 */
export function warnModelLocalAxes(modelLa: LocalAxesGroup, reads: LocalAxesReads): void {
  const count = (shape: number[] | null): number => (shape ? shape.reduce((a, b) => a * b, 1) : 0);
  eachFrameGroup(modelLa, reads, (idDs, frameDs) => {
    const nE = count(idDs.shape);
    if (nE === 0) return;
    const fault = frameFault(nE, frameRowsOf(frameDs.shape), count(frameDs.shape), idDs);
    if (fault !== null) reads.warn(notRead(reads, frameDs, fault));
  });
}

/** The one step-bucket layout (#2451), as a refusal states it. */
const STEP_LAYOUT =
  "a step bucket is one <classTag>-<ClassName> group on LOCAL_AXES holding one ID[nE] dataset, one chunked DATA[steps, nE, 3|4|9] dataset, and optionally STEP[steps] and TIME[steps] (fork schema §7.0, in §7.2's per-class buckets; #2451)";

/** The superseded per-step form schema §5 draws, as a refusal names it. */
const SCHEMA5_FORM = "the per-step DATA/STEP_<k> form schema §5 draws, which §7.0's chunked DATA superseded";

/**
 * The frame rows of `RESULTS/ON_ELEMENTS/LOCAL_AXES` (`stepLa`) at `step`, one entry
 * per bucket read; `nSteps` is the stage's step count. A bucket is one class group
 * `{ID[nE], DATA[nSteps, nE, 3|4|9], STEP[nSteps]?, TIME[nSteps]?}` (#2451). An ID,
 * DATA, STEP or TIME on `stepLa` itself (the flat form, and schema §5's per-step form)
 * and any other bucket are refused by name (the file, the bucket, the step), as is a
 * row that does not decode to a frame (naming its element and row). An id listed
 * twice in the step's buckets is refused at open, by reader/ladrunoElements.ts.
 */
export function collectStepLocalAxes(stepLa: LocalAxesGroup, step: number, nSteps: number, reads: LocalAxesReads): FrameRows[] {
  const refuser = (at: string) => (what: string) =>
    reads.fail(`${reads.file}: ${at} ${what}; a step's frame is read only as stored, and the reference frame in its place would be the wrong frame, so the local axes at step ${step} are not read`);
  const flat = ["ID", "DATA", "STEP", "TIME"].filter((k) => stepLa.keys().includes(k));
  if (flat.length > 0) {
    const form = reads.optionalGroup(stepLa, "DATA") ? SCHEMA5_FORM : "a flat layout this reader does not take";
    throw refuser(stepLa.path)(`holds ${flat.join(", ")} on itself (${form}; ${STEP_LAYOUT})`);
  }
  const read: FrameRows[] = [];
  const readBucket = (b: LocalAxesGroup) => {
    const refuse = refuser(b.path);
    const hasId = b.keys().includes("ID");
    const hasData = b.keys().includes("DATA");
    if (!hasId || !hasData) throw refuse(`${hasId ? "holds an ID but no DATA" : hasData ? "holds a DATA but no ID" : "holds neither an ID nor a DATA"} (${STEP_LAYOUT})`);
    const idDs = reads.optionalDataset(b, "ID");
    const dataDs = reads.optionalDataset(b, "DATA");
    if (!dataDs && reads.optionalGroup(b, "DATA")) throw refuse(`holds a DATA group, not a DATA dataset (${SCHEMA5_FORM}; ${STEP_LAYOUT})`);
    if (!idDs || !dataDs) throw refuse(`holds ${idDs ? "a DATA" : "an ID"} that is not a dataset (${STEP_LAYOUT})`);
    const shape = dataDs.shape ?? [];
    if (shape.length !== 3) throw refuse(`has DATA of shape (${shape.join(", ")}), not (steps, elements, components)`);
    if (shape[0] !== nSteps) throw refuse(`has DATA for ${shape[0]} step(s), but the stage has ${nSteps}`);
    // STEP and TIME may be absent; present, each holds one value per DATA step.
    for (const axis of ["STEP", "TIME"]) {
      if (!b.keys().includes(axis)) continue;
      const ax = reads.optionalDataset(b, axis);
      if (!ax) throw refuse(`holds a ${axis} that is not a dataset (${STEP_LAYOUT})`);
      const n = (ax.shape ?? []).reduce((x, y) => x * y, 1);
      if (n !== shape[0]) throw refuse(`has ${axis} holding ${n} value(s), but its DATA holds ${shape[0]} step(s)`);
    }
    const ids = reads.ints(idDs);
    const nE = ids.length;
    if (nE === 0) {
      if (shape[1] === 0) return;
      throw refuse(`has an empty ID, but its DATA holds ${shape[1]} row(s) per step; a frame row belongs to the element its ID row names, so these belong to no element`);
    }
    if (shape[1] !== nE) throw refuse(`has DATA holding ${shape[1]} row(s) per step for the ${nE} id(s) of its ID`);
    const cols = shape[2]!;
    if (!FRAME_WIDTHS.includes(cols)) throw refuse(`has DATA rows ${cols} wide, not 3 (the local z axis), 4 (a quaternion) or 9 (a rotation matrix)`);
    const slice = reads.stepSlice(dataDs, step);
    if (slice.length !== nE * cols) throw refuse(`read ${slice.length} value(s) from DATA at step ${step}, not ${nE} row(s) ${cols} wide`);
    // A row that does not decode would leave its element the reference frame (resolveLocalAxes keeps the last row that does).
    for (let i = 0; i < nE; i++) {
      if (decodeLocalAxesRow(slice, i * cols, cols)) continue;
      const row = Array.from(slice.subarray(i * cols, (i + 1) * cols)).join(", ");
      throw refuse(`holds at step ${step} a DATA row that is not a frame (element ${ids[i]} at row ${i}: [${row}]); a frame row is finite, and a quaternion is not zero`);
    }
    read.push({ at: idDs.path, ids, raw: slice, cols, source: "step_local_axes" });
  };
  for (const k of stepLa.keys()) {
    const sub = reads.optionalGroup(stepLa, k);
    if (sub) readBucket(sub);
  }
  return read;
}

/** Per-element frames, as reader/ladruno.ts `LadrunoLocalAxes` documents them. */
export interface ResolvedLocalAxes {
  elementIds: Float64Array;
  quaternions: Float64Array;
  zRec: Float64Array;
  hasRecorded: Uint8Array;
  sources: (LocalAxesSource | null)[];
}

/** The index of `n` in the ascending `sorted`, or -1 (a binary search). */
function rowOf(sorted: Float64Array, n: number): number {
  let [lo, hi] = [0, sorted.length - 1];
  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    const v = sorted[mid]!;
    if (v === n) return mid;
    if (v < n) lo = mid + 1;
    else hi = mid - 1;
  }
  return -1;
}

/**
 * Each element's frame from `groups`, taken in order: a later group's row wins over
 * an earlier one's (the step's over the reference's), and a row that decodes to
 * nothing leaves the earlier frame. `elementIds` picks the elements, in its order;
 * without it, every element with a frame, ascending. An element with no frame gets
 * the identity quaternion, a NaN z and hasRecorded 0. Sorted typed arrays only.
 */
export function resolveLocalAxes(groups: readonly FrameRows[], elementIds?: ArrayLike<number>): ResolvedLocalAxes {
  let n = 0;
  for (const g of groups) n += g.ids.length;
  // Which rows decode, in group order (one byte per row), and the ids they frame, sorted and deduplicated.
  const decodes = new Uint8Array(n);
  const framed = new Float64Array(n);
  let m = 0;
  for (let gi = 0, r = 0; gi < groups.length; gi++) {
    const { ids, raw, cols } = groups[gi]!;
    for (let i = 0; i < ids.length; i++, r++) {
      if (decodeLocalAxesRow(raw, i * cols, cols)) {
        decodes[r] = 1;
        framed[m++] = ids[i]!;
      }
    }
  }
  const sorted = framed.subarray(0, m).sort();
  let u = 0;
  for (let i = 0; i < m; i++) if (u === 0 || sorted[i] !== sorted[u - 1]) sorted[u++] = sorted[i]!;
  const ids = sorted.slice(0, u);
  // Each id's frame: the last row that decodes.
  const winGroup = new Uint32Array(u);
  const winRow = new Uint32Array(u);
  for (let gi = 0, r = 0; gi < groups.length; gi++) {
    const g = groups[gi]!;
    for (let i = 0; i < g.ids.length; i++, r++) {
      if (!decodes[r]) continue;
      const k = rowOf(ids, g.ids[i]!);
      winGroup[k] = gi;
      winRow[k] = i;
    }
  }
  const outIds = elementIds ? Float64Array.from(elementIds, Number) : ids;
  const count = outIds.length;
  const quaternions = new Float64Array(count * 4);
  const zRec = new Float64Array(count * 3);
  const hasRecorded = new Uint8Array(count);
  const sources: (LocalAxesSource | null)[] = new Array(count).fill(null);
  for (let k = 0; k < count; k++) {
    const j = elementIds ? rowOf(ids, outIds[k]!) : k;
    const g = j < 0 ? null : groups[winGroup[j]!]!;
    const decoded = g ? decodeLocalAxesRow(g.raw, winRow[j]! * g.cols, g.cols) : null;
    quaternions.set(decoded?.quat ?? [1, 0, 0, 0], k * 4);
    zRec.set(decoded ? decoded.zRec : [Number.NaN, Number.NaN, Number.NaN], k * 3);
    hasRecorded[k] = decoded ? 1 : 0;
    sources[k] = decoded ? g!.source : null;
  }
  return { elementIds: outIds, quaternions, zRec, hasRecorded, sources };
}

/** The frame of each line station (`elementIndex` holds its element's id), from `recorded` (ascending ids); NaN where none was recorded. */
export function stationLocalAxes(
  recorded: ResolvedLocalAxes,
  elementIndex: Float64Array,
): { localAxesQuaternion: Float64Array | null; localAxesZRec: Float64Array | null; localAxesSources: (LocalAxesSource | null)[] | null } {
  if (recorded.elementIds.length === 0) {
    return { localAxesQuaternion: null, localAxesZRec: null, localAxesSources: null };
  }
  const n = elementIndex.length;
  const quats = new Float64Array(n * 4).fill(Number.NaN);
  const zRec = new Float64Array(n * 3).fill(Number.NaN);
  const sources: (LocalAxesSource | null)[] = new Array(n).fill(null);
  for (let i = 0; i < n; i++) {
    const idx = rowOf(recorded.elementIds, elementIndex[i]!);
    if (idx < 0 || !recorded.hasRecorded[idx]) continue;
    quats.set(recorded.quaternions.subarray(idx * 4, idx * 4 + 4), i * 4);
    zRec.set(recorded.zRec.subarray(idx * 3, idx * 3 + 3), i * 3);
    sources[i] = recorded.sources[idx] ?? "model_local_axes";
  }
  return { localAxesQuaternion: quats, localAxesZRec: zRec, localAxesSources: sources };
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
