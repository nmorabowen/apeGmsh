// The viewer's supported scale, and the one check every open route makes against it
// (maintainer ruling on #2362, 2026-10-21, "Ceiling + refusal"; slice #2373).
//
// V8 caps a Map or a Set at 2^24 entries and a plain JS array at 2^27. About 30
// sites in state/ (listed on #2362) hold a Map or a Set sized by a stage's node or
// element count, and mesh/build.ts fills plain number[] buffers. Past those sizes
// they would throw an unnamed RangeError. The ruling keeps those sites as they are
// and sets the supported scale instead: 2^24 nodes and 2^24 elements per stage,
// and as many triangles (and outline edges) as the mesh buffers hold. Each reader
// calls `assertWithinCeiling` as soon as it knows a count, before state/ or mesh/
// code can reach a cap, so a larger model is refused by name at open: the count,
// the limit and the cause. Exactly at a limit is accepted; one past it is refused.
//
// Browser-safe: no Node import.

/** Nodes per stage, and elements per stage: V8's Map/Set entry limit. */
export const NODE_LIMIT = 2 ** 24;
export const ELEMENT_LIMIT = 2 ** 24;

/**
 * The most entries a plain JS array holds. Measured on Node 24 (V8's FixedArray
 * limit): pushing entry 2^27 + 1 throws "RangeError: Invalid array length", for
 * a packed double array and a packed small-integer array alike.
 */
export const JS_ARRAY_LIMIT = 2 ** 27;

/** mesh/build.ts pushes 9 floats per triangle (three xyz corners) into one number[]. */
export const TRIANGLE_LIMIT = Math.floor(JS_ARRAY_LIMIT / 9);
/** ... and 6 floats per outline edge (two xyz endpoints) into another. */
export const EDGE_LIMIT = Math.floor(JS_ARRAY_LIMIT / 6);

export type CeilingKind = "nodes" | "elements" | "triangles" | "outline edges";

const RULE: Readonly<Record<CeilingKind, { limit: number; shown: (limit: number) => string; cause: string }>> = {
  nodes: { limit: NODE_LIMIT, shown: (l) => `${fmt(l)} (2^24) nodes per stage`, cause: "JavaScript Map/Set limit" },
  elements: { limit: ELEMENT_LIMIT, shown: (l) => `${fmt(l)} (2^24) elements per stage`, cause: "JavaScript Map/Set limit" },
  triangles: { limit: TRIANGLE_LIMIT, shown: (l) => `${fmt(l)} triangles`, cause: "JavaScript array limit of 2^27 entries, 9 per triangle" },
  "outline edges": { limit: EDGE_LIMIT, shown: (l) => `${fmt(l)} outline edges`, cause: "JavaScript array limit of 2^27 entries, 6 per edge" },
};

const fmt = (n: number): string => n.toLocaleString("en-US");

/** A model or a stage past the supported scale, refused at open. Not a RangeError: it names its count, limit and cause. */
export class CeilingError extends Error {
  readonly kind: CeilingKind;
  readonly count: number;
  readonly limit: number;
  constructor(kind: CeilingKind, count: number, limit: number, message: string) {
    super(message);
    this.name = "CeilingError";
    this.kind = kind;
    this.count = count;
    this.limit = limit;
  }
}

/**
 * Refuse `count` of `kind` past its limit: `<where>: <scope> has <count> <kind>; the
 * viewer supports at most <limit> (<cause>, ruling on #2362)`. `where` is the file,
 * `scope` what holds the count (`stage "<name>"`, `the model`, `the mesh`). `limit`
 * is the rule's own unless a test passes a smaller one. A count that is not a
 * non-negative integer is not a count, and raises.
 */
export function assertWithinCeiling(where: string, scope: string, kind: CeilingKind, count: number, limit: number = RULE[kind].limit): void {
  if (!Number.isSafeInteger(count) || count < 0) throw new TypeError(`${where}: ${scope}: ${String(count)} is not a count of ${kind}`);
  if (count <= limit) return;
  const r = RULE[kind];
  throw new CeilingError(kind, count, limit, `${where}: ${scope} has ${fmt(count)} ${kind}; the viewer supports at most ${r.shown(limit)} (${r.cause}, ruling on #2362)`);
}

/** The rows of an HDF5 dataset from its shape alone (no read): a null dataspace 0, a scalar 1, else the first axis. */
export function datasetRows(shape: readonly number[] | null): number {
  if (shape === null) return 0;
  return shape.length === 0 ? 1 : shape[0]!;
}

/**
 * A `.ladruno` stage's element count: each result token (`ON_ELEMENTS/<token>`)
 * lists the elements it records once across its buckets, and two tokens (stress and
 * strain) list the same elements, so the stage holds the largest token's sum, never
 * the sum over tokens. A bucket's token is its path's parent.
 */
export function ladrunoStageElements(groups: readonly { path: string; nElements: number }[]): number {
  const byToken = new Map<string, number>(); // keyed by token: sized by the token count
  for (const g of groups) {
    const token = g.path.slice(0, g.path.lastIndexOf("/"));
    byToken.set(token, (byToken.get(token) ?? 0) + g.nElements);
  }
  return Math.max(0, ...byToken.values());
}

/**
 * A native results.h5 stage's element count: its gauss groups split the elements by
 * class, and so do its line-station groups. The two categories feed different state
 * (contours, line diagrams) and may list the same beam, so the stage holds the larger
 * category's sum.
 */
export function nativeStageElements(gauss: readonly { nElements: number }[], lineStations: readonly { nElements: number }[]): number {
  const sum = (gs: readonly { nElements: number }[]): number => gs.reduce((n, g) => n + g.nElements, 0);
  return Math.max(sum(gauss), sum(lineStations));
}
