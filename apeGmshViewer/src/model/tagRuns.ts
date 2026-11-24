// `/opensees/decls/tags`, indexed (#3133): which run owns a (kind, tag), and how
// many elements each declaration owns, without a scan per question.
//
// A model of 19 476 inline zeroLength springs, each with its own uniaxial
// material, has 39 217 tag runs. A scan of every run per object made the
// reader's tag check, the archive keys and the Materials window quadratic: 9 s
// to list the materials, and on every rebuild of the window. The index is built
// once per tag table and kept beside it (a WeakMap on the array), so every
// caller shares it.

import type { DeclTagRun } from "./types.ts";

interface KindRuns {
  /** the kind's runs, sorted by `first` */
  runs: DeclTagRun[];
  /** each run's position in the file's table (the scan's tie-break: the first listed wins) */
  pos: number[];
  /** the largest `first + count` of runs[0..i]: no run at or before i reaches a tag at or past it */
  reach: number[];
}

interface TagIndex {
  byKind: Map<string, KindRuns>;
  /** decl row -> the elements its `element` runs own */
  elementsByDecl: Map<number, number>;
}

const cache = new WeakMap<readonly DeclTagRun[], TagIndex>();

function indexOf(tags: readonly DeclTagRun[]): TagIndex {
  let ix = cache.get(tags);
  if (ix) return ix;
  const listed = new Map<string, number[]>();
  const elementsByDecl = new Map<number, number>();
  tags.forEach((t, i) => {
    (listed.get(t.kind) ?? listed.set(t.kind, []).get(t.kind)!).push(i);
    if (t.kind === "element") elementsByDecl.set(t.decl, (elementsByDecl.get(t.decl) ?? 0) + t.count);
  });
  const byKind = new Map<string, KindRuns>();
  for (const [kind, positions] of listed) {
    positions.sort((a, b) => tags[a]!.first - tags[b]!.first || a - b);
    const runs = positions.map((i) => tags[i]!);
    const reach: number[] = [];
    runs.forEach((r, i) => reach.push(Math.max(i ? reach[i - 1]! : -Infinity, r.first + r.count)));
    byKind.set(kind, { runs, pos: positions, reach });
  }
  ix = { byKind, elementsByDecl };
  cache.set(tags, ix);
  return ix;
}

/**
 * The run of `tags` that holds tag `tag` of allocator kind `kind`, or
 * undefined: the same answer as `tags.find(...)` (of overlapping runs, which
 * only a malformed table has, the one listed first), in O(log n).
 */
export function runOfTag(tags: readonly DeclTagRun[], kind: string, tag: number): DeclTagRun | undefined {
  const k = indexOf(tags).byKind.get(kind);
  if (!k) return undefined;
  // The last run that starts at or before the tag.
  let lo = 0, hi = k.runs.length - 1, i = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (k.runs[mid]!.first <= tag) {
      i = mid;
      lo = mid + 1;
    } else hi = mid - 1;
  }
  let hit = -1;
  // Walk back only while some run at or before i still reaches the tag: one step in a well-formed table.
  for (; i >= 0 && k.reach[i]! > tag; i--) {
    const r = k.runs[i]!;
    if (tag < r.first + r.count && (hit < 0 || k.pos[i]! < k.pos[hit]!)) hit = i;
  }
  return hit < 0 ? undefined : k.runs[hit];
}

/** How many elements declaration row `decl` owns: the sum of its `element` runs. */
export function elementsOfDecl(tags: readonly DeclTagRun[], decl: number): number {
  return indexOf(tags).elementsByDecl.get(decl) ?? 0;
}
