// Simplex Gauss rows written by apeGmsh before #2238 (#2274, the viewer side
// of B #2239 / PR #2273). No three.js and no DOM.
//
// #2238 moved the response catalog's triangle and tet Gauss tables into gmsh
// reference coordinates, which is what state/extrapolation.ts evaluates. The
// native writers copy the catalog's rows into `_natural_coords`, so a
// `results.h5` written before #2238 holds the old node-0-first rows (column k
// is node k's weight, the last node's implicit) with no stamp of its own: the
// results version is 1.2.0 on both sides of the change. Read as gmsh
// coordinates, those rows put every Gauss point at the wrong position, and a
// Tet10's extrapolated corner values come back rotated by one node, silently.
//
// Python's native reader (`_native.py`, `_SIMPLEX_GP_ROWS` and
// `_gauss_natural_coords`) recognises the old rows by value. This is its rule,
// restated (ADR 0112 D4: read as context, never imported):
// - a class_tag absent from the table reads its rows unchanged;
// - rows exactly equal to the class's current table read unchanged;
// - rows exactly equal to the class's older table read as the current table:
//   the same Gauss points, in the same order, in the gmsh frame;
// - any other rows cannot be placed. Python reads them unchanged and warns;
//   the app is stricter (fail closed): the group offers no natural
//   coordinates, so it is never extrapolated, and a warning names the class.
//
// The table is pinned to Python's `_SIMPLEX_GP_ROWS`, value for value, by
// tests/test_app_legacy_simplex_drift.py, which parses this file as text: keep
// one entry per line, `<tag>: { cls: "<name>", legacy: [[...]], current: [[...]] },`
// with each number written as Python's `repr` writes it (the shortest
// round-trip form, which JavaScript parses to the same double).

/** One class's rows: the older writers' node-0-first table and the current gmsh one. */
export interface SimplexRows {
  cls: string;
  legacy: readonly (readonly number[])[];
  current: readonly (readonly number[])[];
}

/** class_tag -> its rows; Python's `_SIMPLEX_GP_ROWS`, value for value. */
export const LEGACY_SIMPLEX_GP_ROWS: Readonly<Record<number, SimplexRows>> = {
  33: { cls: "Tri31", legacy: [[0.3333333333333333, 0.3333333333333333]], current: [[0.3333333333333333, 0.33333333333333337]] },
  167: { cls: "ShellDKGT", legacy: [[0.3333333333333333, 0.3333333333333333], [0.2, 0.6], [0.6, 0.2], [0.2, 0.2]], current: [[0.3333333333333333, 0.33333333333333337], [0.6, 0.19999999999999996], [0.2, 0.19999999999999996], [0.2, 0.6]] },
  168: { cls: "ShellNLDKGT", legacy: [[0.3333333333333333, 0.3333333333333333], [0.2, 0.6], [0.6, 0.2], [0.2, 0.2]], current: [[0.3333333333333333, 0.33333333333333337], [0.6, 0.19999999999999996], [0.2, 0.19999999999999996], [0.2, 0.6]] },
  179: { cls: "FourNodeTetrahedron", legacy: [[0.25, 0.25, 0.25]], current: [[0.25, 0.25, 0.25]] },
  209: { cls: "SixNodeTri", legacy: [[0.6666666666666666, 0.16666666666666666], [0.16666666666666666, 0.6666666666666666], [0.16666666666666666, 0.16666666666666666]], current: [[0.16666666666666666, 0.16666666666666674], [0.6666666666666666, 0.16666666666666674], [0.16666666666666666, 0.6666666666666667]] },
  256: { cls: "TenNodeTetrahedron", legacy: [[0.5854101966249685, 0.1381966011250105, 0.1381966011250105], [0.1381966011250105, 0.5854101966249685, 0.1381966011250105], [0.1381966011250105, 0.1381966011250105, 0.5854101966249685], [0.1381966011250105, 0.1381966011250105, 0.1381966011250105]], current: [[0.1381966011250105, 0.1381966011250105, 0.13819660112501042], [0.5854101966249685, 0.1381966011250105, 0.13819660112501042], [0.1381966011250105, 0.5854101966249685, 0.13819660112501042], [0.1381966011250105, 0.1381966011250105, 0.5854101966249685]] },
  33001: { cls: "BezierTet10", legacy: [[0.5854101966249685, 0.1381966011250105, 0.1381966011250105], [0.1381966011250105, 0.5854101966249685, 0.1381966011250105], [0.1381966011250105, 0.1381966011250105, 0.5854101966249685], [0.1381966011250105, 0.1381966011250105, 0.1381966011250105]], current: [[0.1381966011250105, 0.1381966011250105, 0.13819660112501042], [0.5854101966249685, 0.1381966011250105, 0.13819660112501042], [0.1381966011250105, 0.5854101966249685, 0.13819660112501042], [0.1381966011250105, 0.1381966011250105, 0.5854101966249685]] },
  33008: { cls: "LadrunoCST", legacy: [[0.3333333333333333, 0.3333333333333333]], current: [[0.3333333333333333, 0.33333333333333337]] },
  33016: { cls: "LadrunoLST", legacy: [[0.6666666666666666, 0.16666666666666666], [0.16666666666666666, 0.6666666666666666], [0.16666666666666666, 0.16666666666666666]], current: [[0.16666666666666666, 0.16666666666666674], [0.6666666666666666, 0.16666666666666674], [0.16666666666666666, 0.6666666666666667]] },
};

/** Python's `np.array_equal` on two row tables: the same shape and every value `===`. */
function sameRows(a: readonly (readonly number[])[], b: readonly (readonly number[])[]): boolean {
  return a.length === b.length && a.every((r, i) => r.length === b[i]!.length && r.every((x, j) => x === b[i]![j]));
}

/** What a group's stored rows read as: gmsh rows, or null with the reason it cannot be placed. */
export type GmshRows = { rows: number[][]; refused: null } | { rows: null; refused: string };

/**
 * A native Gauss group's `_natural_coords` in gmsh reference coordinates
 * (the rule above). `where` names the group in the refusal (its file and
 * path), which reads as a warning when the file opens.
 */
export function gmshSimplexRows(classTag: number, rows: number[][], where: string): GmshRows {
  const t = LEGACY_SIMPLEX_GP_ROWS[classTag];
  if (!t || sameRows(rows, t.current)) return { rows, refused: null };
  if (sameRows(rows, t.legacy)) return { rows: t.current.map((r) => [...r]), refused: null };
  return {
    rows: null,
    refused:
      `${where} (${t.cls}) stores natural coordinates that match neither the Gauss table older apeGmsh versions wrote for ${t.cls} ` +
      `nor the current one, so its Gauss points cannot be placed: the group is not extrapolated to nodes. ` +
      `Re-capture the results, or re-transcode the recorder output, with a current apeGmsh`,
  };
}
