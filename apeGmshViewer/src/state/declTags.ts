// The `/opensees/decls` row that owns a tag: a leaf, so modules that selectors.ts
// re-exports (state/materialTest.ts) can use it without importing selectors.ts.
// selectors.ts re-exports it unchanged.

import type { DeclRecord, Definitions } from "../model/types.ts";
import { elementsOfDecl, runOfTag } from "../model/tagRuns.ts";

/** The `/opensees/decls` row that owns tag `tag` of allocator kind `kind`, or null. */
export function declOfTag(defs: Definitions, kind: string, tag: number | null): DeclRecord | null {
  if (!defs.decls || tag === null) return null;
  const run = runOfTag(defs.decls.tags, kind, tag);
  return run ? (defs.decls.records[run.decl] ?? null) : null;
}

/** How many elements declaration `rec` owns (its `element` tag runs). */
export function elementsOwned(defs: Definitions, rec: DeclRecord): number {
  return defs.decls ? elementsOfDecl(defs.decls.tags, rec.row) : 0;
}
