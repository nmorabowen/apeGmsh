// The `/opensees/decls` row that owns a tag: a leaf, so modules that selectors.ts
// re-exports (state/materialTest.ts) can use it without importing selectors.ts.
// selectors.ts re-exports it unchanged.

import type { DeclRecord, Definitions } from "../model/types.ts";

/** The `/opensees/decls` row that owns tag `tag` of allocator kind `kind`, or null. */
export function declOfTag(defs: Definitions, kind: string, tag: number | null): DeclRecord | null {
  if (!defs.decls || tag === null) return null;
  const run = defs.decls.tags.find((t) => t.kind === kind && tag >= t.first && tag < t.first + t.count);
  return run ? (defs.decls.records[run.decl] ?? null) : null;
}
