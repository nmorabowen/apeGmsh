// Checks reader/ladruno.ts runs on what it has read from a `.ladruno` (#2299).
//
// A node ID list that holds one id twice is refused by name: the repeated id would
// read as two columns of one node, and a part set that sums partition copies
// (PARTITION_REDUCTION=SUM, reader/ladrunoParts.ts) would count it twice. A node in
// two part files is legal; one file lists it once. Python's
// `_ladruno.py::LadrunoRepeatedNodeId` (#2300) refuses the same files.
//
// The check sorts a typed copy and compares neighbours: no Set or Map sized by the
// node count, which V8 caps at 2^24 entries (a file with more nodes would throw an
// unnamed RangeError), and an O(n log n) cost of a few hundred ms at 10^7 ids.
//
// Browser-safe: no Node import, and no HDF5 access (W5): the caller reads the ids
// and passes its own error factory, as reader/ladrunoInfo.ts does, so this module
// imports nothing from ladruno.ts.

/** How many repeated ids a refusal names (Python's `_REPEATED_SHOWN`). */
const REPEATED_SHOWN = 10;

/** `ids`, read from the dataset `dataset` of `file`; one holding an id twice is refused through `err`. */
export function uniqueNodeIds(ids: Float64Array, file: string, dataset: string, err: (message: string) => Error): Float64Array {
  const sorted = Float64Array.from(ids).sort();
  let repeated = 0;
  for (let i = 1; i < sorted.length; i++) if (sorted[i] === sorted[i - 1] && (i === 1 || sorted[i - 1] !== sorted[i - 2])) repeated++;
  if (repeated === 0) return ids;
  throw err(`${file}: ${dataset} lists node id(s) more than once (${repeatedRows(ids, sorted, repeated)}); a node is listed once per file, and a part set that sums partition copies (PARTITION_REDUCTION=SUM) would count it twice`);
}

/** The first REPEATED_SHOWN repeated ids, in the order the file first lists them, each with its rows. */
function repeatedRows(ids: Float64Array, sorted: Float64Array, repeated: number): string {
  const reps = new Float64Array(repeated);
  for (let i = 1, k = 0; i < sorted.length; i++) if (sorted[i] === sorted[i - 1] && (i === 1 || sorted[i - 1] !== sorted[i - 2])) reps[k++] = sorted[i]!;
  const isRepeated = (n: number): boolean => {
    let [lo, hi] = [0, reps.length - 1];
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (reps[mid] === n) return true;
      if (reps[mid]! < n) lo = mid + 1;
      else hi = mid - 1;
    }
    return false;
  };
  const rows = new Map<number, number[]>();
  ids.forEach((n, i) => {
    const r = rows.get(n);
    if (r) r.push(i);
    else if (rows.size < REPEATED_SHOWN && isRepeated(n)) rows.set(n, [i]);
  });
  const shown = [...rows].map(([n, r]) => `id ${n} at rows ${r.join(", ")}`).join("; ");
  return shown + (repeated > REPEATED_SHOWN ? `; and ${repeated - REPEATED_SHOWN} more` : "");
}
