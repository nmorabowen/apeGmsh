// The manifest's oracle, test/spec/manifest.oracle.json (written and kept
// current by tests/viewers/test_agent_manifest_oracle.py, from Python's
// `Results`), and the comparison the Node test and the Electron run share.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import type { ManifestEntry, ManifestExtrema } from "../../src/spec/manifest.ts";

interface OracleExtrema {
  min: { value: number; node: number | null; element: number | null; position: number[] };
  max: { value: number; node: number | null; element: number | null; position: number[] };
}
export interface OracleCase {
  name: string;
  set: string;
  stage: string;
  step: number;
  field: string;
  component: string;
  gauss?: string;
  isolate?: string[];
  time: number;
  recorded: OracleExtrema | null;
  drawn: OracleExtrema | null;
  no_data: { kind: string; count: number; of: number };
}
interface OracleRow {
  file: string;
  size: number;
  model_hash: string | null;
  fem_hash: string | null;
}
export const ORACLE = JSON.parse(readFileSync(new URL("./manifest.oracle.json", import.meta.url), "utf8")) as {
  sets: Record<string, { model: OracleRow; results: OracleRow }>;
  cases: OracleCase[];
};

/** The spec shot of a case, as a spec's JSON states it. */
export function caseShot(c: OracleCase): Record<string, unknown> {
  return {
    stage: c.stage,
    step: c.step,
    contour: { field: c.field, component: c.component, ...(c.gauss ? { gauss: c.gauss } : {}) },
    ...(c.isolate ? { isolate: c.isolate.map((g) => `mesh/physical_group/${g}`) } : {}),
  };
}

const close = (a: number, b: number) => Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(b));

function sameExtrema(got: ManifestExtrema | null, want: OracleExtrema | null, what: string): void {
  if (!want || !got) {
    assert.equal(got, want, `${what}: both or neither have a finite value`);
    return;
  }
  for (const end of ["min", "max"] as const) {
    const g: ManifestExtrema["min"] = got[end];
    const w: OracleExtrema["min"] = want[end];
    assert.ok(close(g.value, w.value), `${what}.${end}: value ${g.value}, Python ${w.value}`);
    assert.equal(g.node, w.node, `${what}.${end}: node`);
    assert.equal(g.element, w.element, `${what}.${end}: element`);
    g.position.forEach((x, i) => assert.ok(close(x, w.position[i]!), `${what}.${end}: position ${g.position}, Python ${w.position}`));
  }
}

export function sameAsOracle(entry: ManifestEntry, c: OracleCase): void {
  const k = entry.contour;
  assert.ok(k?.drawn, `${c.name}: the entry states a drawn contour`);
  assert.equal(entry.stage?.name, c.stage);
  assert.equal(entry.step, c.step);
  assert.ok(close(entry.time!, c.time), `${c.name}: time ${entry.time}, Python ${c.time}`);
  sameExtrema(k.extrema.recorded, c.recorded, `${c.name} recorded`);
  sameExtrema(k.extrema.drawn, c.drawn, `${c.name} drawn`);
  assert.deepEqual(k.no_data, c.no_data, `${c.name}: no data`);
  const set = ORACLE.sets[c.set]!;
  const m = entry.files.model!, r = entry.files.results!;
  assert.equal(m.size, set.model.size);
  assert.equal(r.size, set.results.size);
  assert.deepEqual([m.model_hash, m.fem_hash, r.model_hash, r.fem_hash], [set.model.model_hash, set.model.fem_hash, set.results.model_hash, set.results.fem_hash]);
}
