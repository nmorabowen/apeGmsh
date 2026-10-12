// Main's side of V4b (#1728): ResultsHost answers a results file's index as
// plain metadata and one step at a time; a file still being written opens
// read-only with a typed `partial` reason; a truncated file is an answer,
// never a crash. zones.ts reads a model's /meta/lineage and the tags of its
// /sequence element rows, and refuses a row whose source is not a stage's
// owned_element_ids or remove_element.

import assert from "node:assert/strict";
import { copyFileSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import * as h5wasm from "h5wasm/node";
import { ResultsHost } from "../../src/main/results.ts";
import { openLineage, openStageElements } from "../../src/main/zones.ts";
import { openModel } from "../../src/reader/node.ts";
import { partialOf, type ResultsH5 } from "../../src/reader/results.ts";
import { MODEL, RESULTS } from "../results/staged.ts";

await h5wasm.ready;
const host = () => new ResultsHost(h5wasm as unknown as ResultsH5, h5wasm.ready);

function edited(src: string, edit: (f: h5wasm.File) => void): string {
  const p = join(mkdtempSync(join(tmpdir(), "agv-v4b-")), src.split(/[\\/]/).pop()!);
  copyFileSync(src, p);
  const f = new h5wasm.File(p, "a");
  try {
    edit(f);
  } finally {
    f.close();
  }
  return p;
}

test("open: the index as plain metadata (stages, time, fields, lineage); no component array", async () => {
  const h = host();
  const a = await h.open(RESULTS);
  assert.ok(a.ok);
  if (!a.ok) return;
  const m = a.meta;
  assert.deepEqual(m.stages.map((s) => [s.id, s.name, s.kind, s.time.length]), [
    ["stage_0", "base", "static", 3],
    ["stage_1", "erect", "static", 2],
    ["stage_2", "unprop", "static", 4],
    ["stage_3", "mode_1", "mode", 1],
  ]);
  // The generator's closed form: time[k] = (k + 1) / T.
  assert.deepEqual(m.stages[2]!.time, [0.25, 0.5, 0.75, 1]);
  assert.deepEqual(m.stages[0]!.nodal, ["displacement_x"]);
  assert.match(m.lineage!.modelHash!, /^[0-9a-f]{32}$/);
  assert.equal(m.partial, null);
  assert.ok(Array.isArray(m.stages[0]!.time), "the time vector crosses as a plain array");
  h.close();
});

test("readStep: one step's nodal components, the generator's closed form id + 0.001 k; another path is refused", async () => {
  const h = host();
  await h.open(RESULTS);
  const model = await openModel(MODEL);
  const s = h.readStep({ path: RESULTS, stage: "stage_2", step: 3, seq: 7 });
  assert.ok(s.ok);
  if (!s.ok) return;
  assert.equal(s.seq, 7);
  const v = s.nodal["displacement_x"]!;
  assert.equal(v.length, model.nodeIds.length);
  // Node ids of the results are the model's (one partition, written from the same FEMData).
  for (let j = 0; j < v.length; j++) assert.ok(Math.abs(v[j]! - (model.nodeIds[j]! + 0.003)) < 1e-9);
  const bad = h.readStep({ path: "other.results.h5", stage: "stage_2", step: 0, seq: 8 });
  assert.equal(bad.ok, false);
  assert.match((bad as { error: string }).error, /not the open results file/);
  const out = h.readStep({ path: RESULTS, stage: "stage_2", step: 4, seq: 9 });
  assert.match((out as { error: string }).error, /step 4 is outside stage stage_2/);
  h.close();
});

test("partial writes: no results_hash, or no /meta/lineage at 1.1.0, opens read-only with the reason", async () => {
  const noHash = edited(RESULTS, (f) => (f.get("meta/lineage") as h5wasm.Group).delete_attribute("results_hash"));
  const a = await host().open(noHash);
  assert.ok(a.ok);
  assert.match((a as { meta: { partial: string } }).meta.partial, /no results_hash.*may still be being written/);
  // h5wasm cannot delete a group, so the no-lineage case is the rule itself.
  assert.match(partialOf("1.1.0", null)!, /results 1\.1\.0 has no \/meta\/lineage/);
  assert.equal(partialOf("1.0.0", null), null, "results 1.0.0 predates the lineage: not partial, just unpairable");
  assert.equal(partialOf("1.1.0", { femHash: "a", modelHash: "b", resultsHash: "c" }), null);
});

test("a truncated results file is a refusal answer, never a crash", async () => {
  const p = join(mkdtempSync(join(tmpdir(), "agv-v4b-")), "cut.results.h5");
  writeFileSync(p, readFileSync(RESULTS).subarray(0, 4096));
  const a = await host().open(p);
  assert.equal(a.ok, false);
});

test("zones: the model's lineage, and the tags of its /sequence element rows", async () => {
  const l = await openLineage(MODEL);
  assert.ok(l.ok);
  const r = await host().open(RESULTS);
  assert.ok(r.ok && l.ok);
  if (!r.ok || !l.ok) return;
  assert.equal(l.lineage!.modelHash, r.meta.lineage!.modelHash, "the pair carries one model_hash");
  const model = await openModel(MODEL);
  const e = await openStageElements(MODEL, model.sequence);
  assert.ok(e.ok);
  if (!e.ok) return;
  assert.deepEqual(Object.keys(e.sets).sort(), ["/opensees/stages/stage_001/owned_element_ids", "/opensees/stages/stage_002/remove_element"]);
  assert.equal(e.sets["/opensees/stages/stage_001/owned_element_ids"]!.length, 175);
  assert.equal(e.sets["/opensees/stages/stage_002/remove_element"]!.length, 220);
});

test("zones: an element row whose source is not a stage's owned_element_ids or remove_element is refused, not read", async () => {
  const model = await openModel(MODEL);
  const seq = model.sequence;
  assert.equal(seq.status, "ready");
  if (seq.status !== "ready") return;
  const sc = seq.changes.columns.indexOf("source");
  const rows = seq.changes.rows.map((row) => (row.includes("element") ? row.map((c, i) => (i === sc ? "/opensees/element_meta/FourNodeTetrahedron/ids" : c)) : row));
  const e = await openStageElements(MODEL, { ...seq, changes: { ...seq.changes, rows } });
  assert.equal(e.ok, false);
  assert.match((e as { error: string }).error, /not a stage's owned_element_ids or remove_element/);
});
