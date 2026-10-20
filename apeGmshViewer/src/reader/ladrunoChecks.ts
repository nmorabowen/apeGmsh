// Checks reader/ladruno.ts and its sibling modules run on what they have read from a
// `.ladruno` (#2299, #2316).
//
// A node ID list that holds one id twice is refused by name: the repeated id would
// read as two columns of one node, and a part set that sums partition copies
// (PARTITION_REDUCTION=SUM, reader/ladrunoParts.ts) would count it twice. A node in
// two part files is legal; one file lists it once. Python's
// `_ladruno.py::LadrunoRepeatedNodeId` (#2300) refuses the same files.
//
// An element id is listed once per result: once in one bucket of a token
// (reader/ladrunoElements.ts), and in one part file of a part set, because each
// element lives on one rank (reader/ladrunoParts.ts). A repeat would read as two rows
// of one element. Python's `MPCOElementInTwoBuckets` (#2310) refuses the same repeats.
//
// Both checks sort a typed copy and compare neighbours: no Set or Map sized by the
// id count, which V8 caps at 2^24 entries (a file with more ids would throw an
// unnamed RangeError), and an O(n log n) cost of about a second at 10^7 ids. The
// element check first tries an O(n) pass with no copy, which proves the ids the
// fork writes (ascending, in buckets whose ranges do not overlap) unique.
//
// Browser-safe: no Node import, and no HDF5 access (W5): the caller reads the ids
// and passes its own error factory, as reader/ladrunoInfo.ts does, so this module
// imports nothing from ladruno.ts.

/** How many repeated ids a refusal names (Python's `_REPEATED_SHOWN`). */
const REPEATED_SHOWN = 10;

/** `ids`, read from the dataset `dataset` of `file`; one holding an id twice is refused through `err`. */
export function uniqueNodeIds(ids: Float64Array, file: string, dataset: string, err: (message: string) => Error): Float64Array {
  const reps = repeatsOf(Float64Array.from(ids).sort());
  if (reps.length === 0) return ids;
  throw err(`${file}: ${dataset} lists node id(s) more than once (${repeatedRows(ids, reps)}); a node is listed once per file, and a part set that sums partition copies (PARTITION_REDUCTION=SUM) would count it twice`);
}

/** One element-id list: where it was read (`at`: the dataset, with its file when a set has several), and its ids. */
export interface IdList {
  at: string;
  ids: ArrayLike<number>;
}

/**
 * Refuse, through `err`, an element id that `lists` hold more than once: twice in one
 * list, or in two of them. `what` names the lists (a token of one file, or a result
 * of a part set) and `why` says why an element is listed once there.
 */
export function uniqueElementIds(lists: readonly IdList[], what: string, why: string, err: (message: string) => Error): void {
  if (plainlyUnique(lists)) return;
  let n = 0;
  for (const l of lists) n += l.ids.length;
  if (n < 2) return;
  const all = new Float64Array(n);
  let o = 0;
  for (const l of lists) {
    all.set(l.ids, o);
    o += l.ids.length;
  }
  const reps = repeatsOf(all.sort());
  if (reps.length === 0) return;
  throw err(`${what} lists element id(s) more than once (${elementPlaces(lists, reps)}); ${why}`);
}

/**
 * The fast path: every list strictly increasing and no two lists' [first, last] ranges
 * overlapping, so no id repeats. The fork writes its buckets so (ascending ids, one
 * class per bucket), and the check is O(n) with no copy. Anything else, including an
 * overlap that holds no repeat, takes the sort.
 */
function plainlyUnique(lists: readonly IdList[]): boolean {
  const ranges: [number, number][] = []; // one per list: sized by the bucket count, not the id count
  for (const { ids } of lists) {
    if (ids.length === 0) continue;
    for (let i = 1; i < ids.length; i++) if (!(ids[i]! > ids[i - 1]!)) return false;
    ranges.push([ids[0]!, ids[ids.length - 1]!]);
  }
  ranges.sort((a, b) => a[0] - b[0]);
  for (let i = 1; i < ranges.length; i++) if (!(ranges[i]![0] > ranges[i - 1]![1])) return false;
  return true;
}

/** The distinct ids that `sorted` (ascending) holds more than once, ascending; empty when none. */
function repeatsOf(sorted: Float64Array): Float64Array {
  const starts = (i: number): boolean => sorted[i] === sorted[i - 1] && (i === 1 || sorted[i - 1] !== sorted[i - 2]);
  let repeated = 0;
  for (let i = 1; i < sorted.length; i++) if (starts(i)) repeated++;
  const reps = new Float64Array(repeated);
  if (repeated === 0) return reps;
  for (let i = 1, k = 0; i < sorted.length; i++) if (starts(i)) reps[k++] = sorted[i]!;
  return reps;
}

/** Whether the ascending `reps` holds `n` (a binary search). */
function holds(reps: Float64Array, n: number): boolean {
  let [lo, hi] = [0, reps.length - 1];
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (reps[mid] === n) return true;
    if (reps[mid]! < n) lo = mid + 1;
    else hi = mid - 1;
  }
  return false;
}

const more = (reps: Float64Array): string => (reps.length > REPEATED_SHOWN ? `; and ${reps.length - REPEATED_SHOWN} more` : "");

/** The first REPEATED_SHOWN repeated ids, in the order the file first lists them, each with its rows. */
function repeatedRows(ids: Float64Array, reps: Float64Array): string {
  const rows = new Map<number, number[]>();
  ids.forEach((n, i) => {
    const r = rows.get(n);
    if (r) r.push(i);
    else if (rows.size < REPEATED_SHOWN && holds(reps, n)) rows.set(n, [i]);
  });
  return [...rows].map(([n, r]) => `id ${n} at rows ${r.join(", ")}`).join("; ") + more(reps);
}

/** The first REPEATED_SHOWN repeated ids, in the order the lists first hold them, each with every list and row it sits at. */
function elementPlaces(lists: readonly IdList[], reps: Float64Array): string {
  // At most REPEATED_SHOWN ids, each with at most one entry per list: sized by the refusal, never by the id count.
  const places = new Map<number, Map<string, number[]>>();
  for (const l of lists) {
    for (let i = 0; i < l.ids.length; i++) {
      const n = l.ids[i]!;
      let at = places.get(n);
      if (!at) {
        if (places.size >= REPEATED_SHOWN || !holds(reps, n)) continue;
        places.set(n, (at = new Map()));
      }
      const rows = at.get(l.at);
      if (rows) rows.push(i);
      else at.set(l.at, [i]);
    }
  }
  const one = ([at, rows]: [string, number[]]): string => `${at} row${rows.length > 1 ? "s" : ""} ${rows.join(", ")}`;
  return [...places].map(([n, at]) => `id ${n} at ${[...at].map(one).join(" and ")}`).join("; ") + more(reps);
}
