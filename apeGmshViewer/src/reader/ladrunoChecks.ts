// Checks reader/ladruno.ts runs on what it has read from a `.ladruno` (#2299).
//
// A node ID list that holds one id twice is refused by name: the repeated id would
// read as two columns of one node, and a part set that sums partition copies
// (PARTITION_REDUCTION=SUM, reader/ladrunoParts.ts) would count it twice. A node in
// two part files is legal; one file lists it once. Python's
// `_ladruno.py::LadrunoRepeatedNodeId` (#2300) refuses the same files.
//
// Browser-safe: no Node import, and no HDF5 access (W5): the caller reads the ids
// and passes its own error factory, as reader/ladrunoInfo.ts does, so this module
// imports nothing from ladruno.ts.

/** How many repeated ids a refusal names (Python's `_REPEATED_SHOWN`). */
const REPEATED_SHOWN = 10;

/** `ids`, read from the dataset `dataset` of `file`; one holding an id twice is refused through `err`. */
export function uniqueNodeIds(ids: Float64Array, file: string, dataset: string, err: (message: string) => Error): Float64Array {
  if (new Set(ids).size === ids.length) return ids;
  const rows = new Map<number, number[]>();
  ids.forEach((n, i) => (rows.get(n) ?? rows.set(n, []).get(n)!).push(i));
  const twice = [...rows].filter(([, r]) => r.length > 1).map(([n, r]) => `id ${n} at rows ${r.join(", ")}`);
  const shown = twice.slice(0, REPEATED_SHOWN).join("; ") + (twice.length > REPEATED_SHOWN ? `; and ${twice.length - REPEATED_SHOWN} more` : "");
  throw err(`${file}: ${dataset} lists node id(s) more than once (${shown}); a node is listed once per file, and a part set that sums partition copies (PARTITION_REDUCTION=SUM) would count it twice`);
}
