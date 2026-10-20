// `.ladruno` nodal reads, one hyperslab per dataset: one step (#2202) and a
// range of steps, the history scan (#2270).
//
// A nodal result is one `DATA` of shape (T, N, C) (a mode stage's is (N, C)),
// chunked over whole steps (fork schema §7.0), and each component is one column
// of it. Slicing one column at a time inflates the same chunk once per
// component. `readColumnsBySlab` groups the requested columns by dataset, reads
// the steps' columns c0..c1 of each dataset in one slice, and splits them in
// memory. A column's values are the same bits as its own one-column slice: the
// same float64 values, or the same float32 values each widened once.
//
// A node's column is found by binary search over the stage's ID list
// (`nodeColumnLookup`, #2354), not a Map from id to column: V8 caps a Map at 2^24
// entries, so a stage of more nodes threw an unnamed RangeError on the first
// node-history read. An ascending list (what the fork writes) is searched in place;
// any other order is searched through a sorted copy and its permutation.
//
// Browser-safe: no Node import. The caller passes its own read and error
// factory, as reader/ladrunoInfo.ts does, so this module imports nothing from
// ladruno.ts.

import { rowOf } from "./ladrunoLocalAxes.ts";

/** One component's place: column `col` of the dataset at `path`. */
export interface NodalColumnRef {
  path: string;
  col: number;
}

/**
 * The values of each column in `cols`, in order, each (rows,). `readSlab(first,
 * c0, c1)` returns columns c0..c1 (exclusive) of every row of the dataset at
 * `first.path` (`first` is the group's first column), row-major:
 * (rows * (c1 - c0),). Each dataset is read once; a component asked for twice
 * gets its own array.
 */
export function readColumnsBySlab<C extends NodalColumnRef>(
  cols: readonly C[],
  readSlab: (first: C, c0: number, c1: number) => Float64Array,
  error: (message: string) => Error,
): Float64Array[] {
  const byPath = new Map<string, number[]>();
  cols.forEach((c, i) => {
    const at = byPath.get(c.path);
    if (at) at.push(i);
    else byPath.set(c.path, [i]);
  });
  const out: Float64Array[] = new Array<Float64Array>(cols.length);
  for (const [path, members] of byPath) {
    const wanted = members.map((i) => cols[i]!.col);
    const c0 = Math.min(...wanted);
    const c1 = Math.max(...wanted) + 1;
    const width = c1 - c0;
    const slab = readSlab(cols[members[0]!]!, c0, c1);
    if (slab.length % width !== 0) throw error(`${path}: a read of columns ${c0}..${c1} returned ${slab.length} values, not a whole number of rows`);
    if (members.length === 1 && width === 1) {
      out[members[0]!] = slab;
      continue;
    }
    const rows = slab.length / width;
    for (const i of members) {
      const k = cols[i]!.col - c0;
      const v = new Float64Array(rows);
      for (let r = 0; r < rows; r++) v[r] = slab[r * width + k]!;
      out[i] = v;
    }
  }
  return out;
}

/**
 * The column of each node id in `ids` (a stage's node ID list): the returned
 * function gives `ids.indexOf(n)`, or -1 for an id the list does not hold. Sized
 * by typed arrays only, so it holds past V8's 2^24 Map cap. A list that holds an
 * id twice (refused at open, reader/ladrunoChecks.ts) is refused here too, through
 * `error`, rather than answering one of its columns.
 */
export function nodeColumnLookup(ids: Float64Array, error: (message: string) => Error): (n: number) => number {
  let ascending = true;
  for (let i = 1; i < ids.length && ascending; i++) ascending = ids[i]! > ids[i - 1]!;
  if (ascending) return (n) => rowOf(ids, n);
  const perm = new Uint32Array(ids.length);
  for (let i = 0; i < perm.length; i++) perm[i] = i;
  perm.sort((a, b) => ids[a]! - ids[b]! || a - b);
  const sorted = Float64Array.from(perm, (j) => ids[j]!);
  for (let k = 1; k < sorted.length; k++) {
    if (sorted[k] === sorted[k - 1]) throw error(`the node ID list holds id ${sorted[k]} at rows ${perm[k - 1]} and ${perm[k]}; a node is listed once per file`);
  }
  return (n) => {
    const k = rowOf(sorted, n);
    return k < 0 ? -1 : perm[k]!;
  };
}
