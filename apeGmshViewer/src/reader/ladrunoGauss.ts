// Where a `.ladruno` beam's section stations sit along it: the natural coordinate
// xi in [-1, 1] of each `section.*` station, from the element group's
// `QUADRATURE/GP_PARAM` keyed by the bucket's `GAUSS_ID` (#2472).
//
// GP_PARAM is in the group's `@PARAM_DOMAIN` (schema v1 §3: "[-1,1]", "[0,1]" or
// "bary"), so its rows are read only when the domain says how. A line station
// reads "[-1,1]" as it is; a missing domain, or any other one, is refused by name,
// as Python's `_gp_param_for` / `gp_param_to_gmsh` refuse it
// (src/apeGmsh/results/readers/_ladruno_element_io.py): Python knows "[-1,1]" and
// "bary", and "bary" is a simplex rule, never a line. A row that is not finite, or
// lies outside [-1, 1], is refused too: it is not a point on the beam.
//
// When the stations cannot be placed from GP_PARAM (the group has none, or the
// GAUSS_IDs do not index its rows), the stations fall back to evenly spaced xi,
// and that fallback always says so in a named warning: evenly spaced stations are
// a guess about a force-based beam, whose Legendre or Lobatto points are not.
//
// Browser-safe: no Node import, no HDF5 open (rule W5: those stay in h5open.ts).

/** The one PARAM_DOMAIN a line station reads (schema v1 §3; Python `gp_param_to_gmsh`). */
export const LINE_PARAM_DOMAIN = "[-1,1]";

/** Column 0 of an element group's GP_PARAM, with the group's PARAM_DOMAIN, as read from the file. */
export interface GpParamRows {
  /** `<group>/QUADRATURE/GP_PARAM`, for messages. */
  path: string;
  /** `<group>`, for messages. */
  group: string;
  /** The group's `@PARAM_DOMAIN`, trimmed, or null when the attribute is absent. */
  paramDomain: string | null;
  /** Column 0 of each GP_PARAM row. */
  xi: Float64Array;
}

/**
 * GP_PARAM column 0 as line-station xi, or a thrown refusal (`fail`): the domain
 * must be "[-1,1]" and every row a finite number in [-1, 1].
 */
export function lineGpParamXi(rows: GpParamRows, fail: (message: string) => Error): Float64Array {
  if (rows.paramDomain === null) {
    throw fail(
      `${rows.group} carries QUADRATURE/GP_PARAM but no PARAM_DOMAIN attribute, so the frame of its ` +
        `Gauss-point coordinates is unknown. Refusing to guess where the section stations are.`,
    );
  }
  if (rows.paramDomain !== LINE_PARAM_DOMAIN) {
    throw fail(
      `${rows.group}: PARAM_DOMAIN ${JSON.stringify(rows.paramDomain)} is not one a line station reads ` +
        `(known: ${JSON.stringify(LINE_PARAM_DOMAIN)}). Refusing to guess where the section stations are.`,
    );
  }
  for (let i = 0; i < rows.xi.length; i++) {
    const x = rows.xi[i]!;
    if (!Number.isFinite(x) || x < -1 || x > 1) {
      throw fail(`${rows.path}[${i}] = ${x} is not a natural coordinate on the beam (a finite xi in [-1, 1]); refusing to draw a station there.`);
    }
  }
  return rows.xi;
}

/** Evenly spaced stations on [-1, 1]: one station sits at 0. */
export function evenlySpacedXi(nStations: number): Float64Array {
  if (nStations <= 0) return new Float64Array(0);
  if (nStations === 1) return new Float64Array([0]);
  const out = new Float64Array(nStations);
  for (let i = 0; i < nStations; i++) {
    out[i] = -1 + (2 * i) / (nStations - 1);
  }
  return out;
}

/**
 * The xi of each section station, in the order of `gaussIds` (already sorted), and
 * the warning when they are evenly spaced instead of read from GP_PARAM.
 *
 * GAUSS_IDs index GP_PARAM 0-based, or else 1-based, when every one is non-negative,
 * distinct and in range (`distinct` says the first two hold); otherwise, or when
 * there is no GP_PARAM (`xi` null), the stations are evenly spaced, with a warning
 * naming the bucket, the GAUSS_IDs and why.
 */
export function sectionStationXi(
  bucketKey: string,
  gaussIds: readonly number[],
  distinct: boolean,
  gp: { path: string; xi: Float64Array } | null,
  missingWhere: string,
  token = "section",
): { xi: Float64Array; warning: string | null } {
  const n = gaussIds.length;
  if (gp !== null && distinct) {
    if (gaussIds.every((g) => g >= 0 && g < gp.xi.length)) {
      return { xi: Float64Array.from(gaussIds, (g) => gp.xi[g]!), warning: null };
    }
    if (gaussIds.every((g) => g >= 1 && g <= gp.xi.length)) {
      return { xi: Float64Array.from(gaussIds, (g) => gp.xi[g - 1]!), warning: null };
    }
  }
  const why =
    gp === null
      ? `there is no ${missingWhere}`
      : !distinct
        ? `its GAUSS_IDs [${gaussIds.join(", ")}] are not distinct non-negative integers`
        : `its GAUSS_IDs [${gaussIds.join(", ")}] do not index the ${gp.xi.length} row(s) of ${gp.path}`;
  return {
    xi: evenlySpacedXi(n),
    warning:
      `${token} bucket ${JSON.stringify(bucketKey)}: ${why}, so its ${n} section station(s) are drawn evenly spaced ` +
      `along the beam (xi = linspace(-1, 1)), which need not be where its integration points are.`,
  };
}
