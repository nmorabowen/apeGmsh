// Main's side of V4b (#1728): ResultsHost answers a results file's index as
// plain metadata and one step at a time; a file still being written opens
// read-only with a typed `partial` reason; a truncated file is an answer,
// never a crash. zones.ts reads a model's /meta/lineage and the tags of its
// /sequence element rows, and refuses a row whose source is not a stage's
// owned_element_ids or remove_element.

import assert from "node:assert/strict";
import { copyFileSync, mkdtempSync, readFileSync, renameSync, statSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import * as h5wasm from "h5wasm/node";
import { IDLE_CLOSE_MS, ResultsHost } from "../../src/main/results.ts";
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

// -- the Windows lock (review round 1): the handle closes when idle, and a changed file is never read under the old index

const SMALL = RESULTS.replace(/staged_activation\.results\.h5$/, "results_small.h5");
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
function scratchCopy(src: string): string {
  const p = join(mkdtempSync(join(tmpdir(), "agv-v4b-lock-")), "run.results.h5");
  copyFileSync(src, p);
  return p;
}

test("idle close: the handle closes after idleMs without a read, and the next read re-opens it", async () => {
  const p = scratchCopy(RESULTS);
  const h = new ResultsHost(h5wasm as unknown as ResultsH5, h5wasm.ready, { idleMs: 30 });
  assert.ok((await h.open(p)).ok);
  assert.equal(h.isOpen(), true);
  await sleep(80);
  assert.equal(h.isOpen(), false, "closed while idle");
  const s = h.readStep({ path: p, stage: "stage_0", step: 1, seq: 1 });
  assert.ok(s.ok, JSON.stringify(s));
  assert.equal(h.isOpen(), true, "re-opened by the read");
  assert.equal(IDLE_CLOSE_MS <= 1000, true);
  h.close();
});

test("a writer's temp-file-and-replace onto the path succeeds while idle; the next read answers `changed` and a re-open re-indexes", async () => {
  const p = scratchCopy(RESULTS);
  const h = new ResultsHost(h5wasm as unknown as ResultsH5, h5wasm.ready, { idleMs: 30 });
  assert.ok((await h.open(p)).ok);
  await sleep(80);
  // os.replace: write a temp file beside it, rename it over the path (fails on Windows while a handle is open).
  const tmp = `${p}.tmp`;
  copyFileSync(SMALL, tmp);
  renameSync(tmp, p);
  const s = h.readStep({ path: p, stage: "stage_0", step: 1, seq: 2 });
  assert.equal(s.ok, false, "the new file is never read under the old index");
  assert.equal((s as { changed?: true }).changed, true);
  const again = await h.open(p);
  assert.ok(again.ok);
  if (!again.ok) return;
  assert.deepEqual(again.meta.stages.map((x) => x.name), ["history"], "the re-index is the new file's");
  assert.ok(h.readStep({ path: p, stage: "stage_0", step: 1, seq: 3 }).ok);
  h.close();
});

test("an in-place rewrite under an open handle is caught before the read (size/mtime), never served under the old index", async () => {
  const p = scratchCopy(RESULTS);
  const h = new ResultsHost(h5wasm as unknown as ResultsH5, h5wasm.ready, { idleMs: 10_000 });
  assert.ok((await h.open(p)).ok);
  assert.ok(h.readStep({ path: p, stage: "stage_0", step: 0, seq: 1 }).ok);
  const later = new Date(Date.now() + 5000);
  utimesSync(p, later, later);
  const s = h.readStep({ path: p, stage: "stage_0", step: 0, seq: 2 });
  assert.equal((s as { changed?: true }).changed, true);
  assert.equal(h.isOpen(), false, "the stale handle is closed");
  h.close();
});

// -- V4L-b1 (#2127): a .ladruno under the same rules; its identity is its INFO/PROVENANCE

const LADRUNO = RESULTS.replace(/staged_activation\.results\.h5$/, "ladruno_frame3d.ladruno");
const LADRUNO_UNSTAMPED = RESULTS.replace(/staged_activation\.results\.h5$/, "ladruno_truss2d.ladruno");
function scratchLadruno(src: string, name = "run.ladruno"): string {
  const p = join(mkdtempSync(join(tmpdir(), "agv-v4l-")), name);
  copyFileSync(src, p);
  return p;
}

test("a .ladruno opens through main: the PROVENANCE lineage, stage names and nodal components as the reader names them", async () => {
  const h = host();
  const a = await h.open(LADRUNO);
  assert.ok(a.ok, JSON.stringify(a));
  if (!a.ok) return;
  assert.equal(a.meta.sourceType, "ladruno");
  assert.deepEqual(a.meta.lineage, { modelHash: "0123456789abcdef0123456789abcdef", femHash: "fedcba9876543210fedcba9876543210", resultsHash: null });
  assert.equal(a.meta.lineageReason, null);
  assert.deepEqual(a.meta.stages.map((s) => s.name), ["gravity", "lateral"]);
  const comps = a.meta.stages[0]!.nodal;
  assert.ok(comps.includes("DISPLACEMENT/Ux"), comps.join(", "));
  const s = h.readStep({ path: LADRUNO, stage: a.meta.stages[0]!.id, step: 0, seq: 1 });
  assert.ok(s.ok, JSON.stringify(s));
  if (!s.ok) return;
  assert.deepEqual(Object.keys(s.nodal), comps, "a step carries every component under the reader's name");
  h.close();
});

test("a .ladruno the recorder is still writing (no stamp yet) opens unpaired: no lineage, and the reader's reason", async () => {
  const a = await host().open(LADRUNO_UNSTAMPED);
  assert.ok(a.ok);
  if (!a.ok) return;
  assert.equal(a.meta.lineage, null);
  assert.match(a.meta.lineageReason!, /INFO\/PROVENANCE has no @model_hash; INFO\/PROVENANCE has no @fem_hash: the file is not stamped for pairing/);
});

test("reload seam: a .ladruno re-stamped while idle (same size and mtime) is re-indexed, never read under the old stamp", async () => {
  // Two stamps written the same way, same length, other names: the hashes stay,
  // so only the whole PROVENANCE tells them apart (size and mtime cannot).
  const stamp = (names: string) =>
    edited(LADRUNO, (f) => {
      const g = f.get("INFO/PROVENANCE") as h5wasm.Group;
      g.delete_attribute("stage_names");
      g.create_attribute("stage_names", names);
    });
  const first = stamp('["gravity","lateral"]');
  const second = stamp('["GRAVITY","LATERAL"]');
  const p = scratchLadruno(first);
  const T = 1_700_000_000; // whole seconds: the mtime round-trips exactly
  utimesSync(p, T, T);
  const h = new ResultsHost(h5wasm as unknown as ResultsH5, h5wasm.ready, { idleMs: 30 });
  const a = await h.open(p);
  assert.ok(a.ok);
  if (!a.ok) return;
  await sleep(80);
  assert.equal(h.isOpen(), false);
  const before = statSync(p);
  copyFileSync(second, p);
  utimesSync(p, T, T);
  const after = statSync(p);
  assert.equal(after.size, before.size, "precondition: the size and mtime cannot tell the two stamps apart");
  assert.equal(after.mtimeMs, before.mtimeMs);
  const s = h.readStep({ path: p, stage: a.meta.stages[0]!.id, step: 0, seq: 1 });
  assert.equal(s.ok, false);
  assert.equal((s as { changed?: true }).changed, true, JSON.stringify(s));
  const again = await h.open(p);
  assert.ok(again.ok);
  if (!again.ok) return;
  assert.deepEqual(again.meta.stages.map((x) => x.name), ["GRAVITY", "LATERAL"], "the re-index reads the new stamp");
  h.close();
});

test("shared namespace: a .ladruno opened with a .results.h5 beside it reads only itself, and warns that the other is not opened", async () => {
  const p = scratchLadruno(LADRUNO);
  copyFileSync(RESULTS, p.replace(/\.ladruno$/, ".results.h5"));
  const a = await host().open(p);
  assert.ok(a.ok);
  if (!a.ok) return;
  assert.equal(a.meta.sourceType, "ladruno");
  assert.ok(a.warnings.some((w) => /^run\.results\.h5 is not opened: a set reads one results file, and run\.ladruno is this one \(the file opened wins; otherwise the \.ladruno, ADR 0122 D1\)/.test(w)), a.warnings.join("\n"));
  // And the reverse: the native file opened beside a .ladruno names the .ladruno.
  const n = await host().open(p.replace(/\.ladruno$/, ".results.h5"));
  assert.ok(n.ok);
  if (!n.ok) return;
  assert.ok(n.warnings.some((w) => w.startsWith("run.ladruno is not opened")), n.warnings.join("\n"));
  // Alone, no such warning.
  const alone = await host().open(scratchLadruno(LADRUNO));
  assert.ok(alone.ok && !alone.warnings.some((w) => /is not opened/.test(w)));
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
