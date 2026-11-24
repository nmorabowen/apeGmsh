// #3133: `/opensees/decls/tags`, indexed (src/model/tagRuns.ts). The oracle is
// the scan it replaces, `tags.find((t) => t.kind === kind && tag >= t.first &&
// tag < t.first + t.count)`, asked the same questions of the same tables,
// including malformed ones whose runs overlap.

import assert from "node:assert/strict";
import { test } from "node:test";
import { elementsOfDecl, runOfTag } from "../src/model/tagRuns.ts";
import type { DeclTagRun } from "../src/model/types.ts";

const scan = (tags: readonly DeclTagRun[], kind: string, tag: number) => tags.find((t) => t.kind === kind && tag >= t.first && tag < t.first + t.count);

/** A deterministic generator, so a failure names a reproducible table. */
function rng(seed: number): () => number {
  let x = seed >>> 0 || 1;
  return () => {
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    return (x >>> 0) / 2 ** 32;
  };
}

function table(seed: number, overlapping: boolean): DeclTagRun[] {
  const r = rng(seed);
  const kinds = ["element", "uniaxialMaterial", "node", "pattern"];
  const next = new Map(kinds.map((k) => [k, 1]));
  const tags: DeclTagRun[] = [];
  for (let i = 0; i < 200; i++) {
    const kind = kinds[Math.floor(r() * kinds.length)]!;
    const count = Math.floor(r() * 6);
    // A well-formed allocator hands out disjoint runs in any order; an overlapping table reuses tags.
    const first = overlapping ? 1 + Math.floor(r() * 60) : next.get(kind)! + Math.floor(r() * 3);
    if (!overlapping) next.set(kind, first + count);
    tags.push({ kind, first, count, decl: Math.floor(r() * 40) });
  }
  // Listed in no particular order, as a file may list them.
  return overlapping ? tags : tags.sort(() => r() - 0.5);
}

test("#3133: runOfTag answers as the scan does, on well-formed and overlapping tables", () => {
  for (let seed = 1; seed <= 40; seed++) {
    for (const overlapping of [false, true]) {
      const tags = table(seed, overlapping);
      for (const kind of ["element", "uniaxialMaterial", "node", "pattern", "section"]) {
        for (let tag = -2; tag < 400; tag++) {
          assert.equal(runOfTag(tags, kind, tag), scan(tags, kind, tag), `seed ${seed}${overlapping ? " (overlapping)" : ""}: ${kind} ${tag}`);
        }
      }
    }
  }
});

test("#3133: elementsOfDecl sums a declaration's element runs, and nothing of other kinds", () => {
  const tags: DeclTagRun[] = [
    { kind: "element", first: 1, count: 10, decl: 3 },
    { kind: "uniaxialMaterial", first: 1, count: 99, decl: 3 },
    { kind: "element", first: 40, count: 5, decl: 3 },
    { kind: "element", first: 11, count: 7, decl: 4 },
  ];
  assert.equal(elementsOfDecl(tags, 3), 15);
  assert.equal(elementsOfDecl(tags, 4), 7);
  assert.equal(elementsOfDecl(tags, 5), 0);
});

test("#3133: 40 000 single-tag runs answer 40 000 questions without a scan per question", () => {
  const n = 20_000;
  const tags: DeclTagRun[] = [];
  for (let i = 0; i < n; i++) tags.push({ kind: "element", first: 1 + i, count: 1, decl: i }, { kind: "uniaxialMaterial", first: 1 + i, count: 1, decl: n + i });
  const t0 = performance.now();
  for (let i = 0; i < n; i++) {
    assert.equal(runOfTag(tags, "uniaxialMaterial", 1 + i)!.decl, n + i);
    assert.equal(elementsOfDecl(tags, i), 1);
  }
  const ms = performance.now() - t0;
  // An index answers in tens of ms; the scan took seconds here (20k x 40k). The bound leaves room for a slow runner.
  assert.ok(ms < 2_000, `${ms.toFixed(0)} ms for ${n} lookups`);
});
