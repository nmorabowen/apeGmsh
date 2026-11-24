// ADR 0137 D5 (H3, #3059): one read serves every consumer of a step.
//
// `ResultsHost.readStep` keeps the last STEP_CACHE answers, keyed by the file's on-disk signature and
// identity, the stage, the step and each optional part of the request. Pinned here: a hit, a miss,
// eviction, a changed file, element, tensor, spring and hinge requests keyed apart, a nodal-only
// request served by a fuller answer (nodal values only), and the step range a tool reads through it.

import assert from "node:assert/strict";
import { copyFileSync, mkdtempSync, rmSync, statSync, utimesSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { ResultsHost, STEP_CACHE } from "../../src/main/results.ts";
import type { ResultsH5 } from "../../src/reader/results.ts";

const FIX = fileURLToPath(new URL("../../fixtures/", import.meta.url));
const dir = mkdtempSync(join(tmpdir(), "agv-stepcache-"));
after(() => rmSync(dir, { recursive: true, force: true }));

async function opened(name: string) {
  const path = join(dir, name);
  copyFileSync(join(FIX, name), path);
  const host = new ResultsHost(h5wasm as unknown as ResultsH5, h5wasm.ready);
  const o = await host.open(path);
  assert.ok(o.ok, !o.ok ? o.error : "");
  return { host, path, meta: o.meta };
}

test("the same step asked twice is read once; another step is read; the seq is the asker's", async () => {
  const { host, path, meta } = await opened("ladruno_hinges2d.ladruno");
  const stage = meta.stages[0]!.id;
  const a = host.readStep({ path, stage, step: 3, seq: 1 });
  const b = host.readStep({ path, stage, step: 3, seq: 2 });
  assert.ok(a.ok && b.ok);
  assert.equal(b.seq, 2);
  assert.deepEqual(host.stepCounts(), { reads: 1, hits: 1 });
  host.readStep({ path, stage, step: 4, seq: 3 });
  assert.deepEqual(host.stepCounts(), { reads: 2, hits: 1 });
  host.close();
});

test(`the last ${STEP_CACHE} steps are kept: the oldest is read again, a recent one is not`, async () => {
  const { host, path, meta } = await opened("ladruno_hinges2d.ladruno");
  const stage = meta.stages[0]!.id;
  for (let k = 0; k <= STEP_CACHE; k++) host.readStep({ path, stage, step: k, seq: k });
  assert.equal(host.stepCounts().reads, STEP_CACHE + 1);
  host.readStep({ path, stage, step: STEP_CACHE, seq: 99 });
  assert.equal(host.stepCounts().hits, 1, "the newest is kept");
  host.readStep({ path, stage, step: 0, seq: 100 });
  assert.equal(host.stepCounts().reads, STEP_CACHE + 2, "step 0 was evicted and is read again");
  host.close();
});

test("a file rewritten in place misses: the read finds it changed, never the kept answer", async () => {
  const { host, path, meta } = await opened("ladruno_hinges2d.ladruno");
  const stage = meta.stages[0]!.id;
  assert.ok(host.readStep({ path, stage, step: 2, seq: 1 }).ok);
  const t = statSync(path);
  utimesSync(path, t.atime, new Date(t.mtimeMs + 5000));
  const after = host.readStep({ path, stage, step: 2, seq: 2 });
  assert.equal(after.ok, false);
  assert.equal(!after.ok && after.changed, true);
  assert.equal(host.stepCounts().hits, 0);
});

test("a reopen drops the kept answers", async () => {
  const { host, path, meta } = await opened("ladruno_hinges2d.ladruno");
  const stage = meta.stages[0]!.id;
  host.readStep({ path, stage, step: 1, seq: 1 });
  assert.ok((await host.open(path)).ok);
  host.readStep({ path, stage, step: 1, seq: 2 });
  assert.deepEqual(host.stepCounts(), { reads: 2, hits: 0 });
  host.close();
});

test("element and tensor requests are keyed apart from each other and from a nodal one", async () => {
  const { host, path, meta } = await opened("elem_mixed.results.h5");
  const stage = meta.stages[0]!.id;
  const el = host.readStep({ path, stage, step: 1, seq: 1, element: "stress_xx" });
  const el2 = host.readStep({ path, stage, step: 1, seq: 2, element: "stress_yy" });
  const ten = host.readStep({ path, stage, step: 1, seq: 3, tensor: ["stress_xx", "stress_yy", "stress_xy"] });
  assert.ok(el.ok && el2.ok && ten.ok);
  assert.notDeepEqual(el.gauss, el2.gauss, "another element component is another read");
  assert.equal(host.stepCounts().reads, 3);
  assert.deepEqual(Object.keys(ten.tensor!).sort(), ["stress_xx", "stress_xy", "stress_yy"]);
  // The same element request again: a hit with the same values.
  const again = host.readStep({ path, stage, step: 1, seq: 4, element: "stress_xx" });
  assert.ok(again.ok);
  assert.deepEqual(again.gauss, el.gauss);
  assert.equal(host.stepCounts().hits, 1);
  host.close();
});

test("a nodal-only request is served by a fuller answer for the step, with the nodal values only", async () => {
  const { host, path, meta } = await opened("elem_mixed.results.h5");
  const stage = meta.stages[0]!.id;
  const full = host.readStep({ path, stage, step: 2, seq: 1, element: "stress_xx" });
  const nodal = host.readStep({ path, stage, step: 2, seq: 2 });
  assert.ok(full.ok && nodal.ok);
  assert.deepEqual(host.stepCounts(), { reads: 1, hits: 1 });
  assert.deepEqual(nodal.nodal, full.nodal);
  assert.deepEqual(nodal.gauss, {}, "not the element values it did not ask for");
  // The other way round is a read: a nodal answer does not hold the element values.
  host.readStep({ path, stage, step: 0, seq: 3 });
  host.readStep({ path, stage, step: 0, seq: 4, element: "stress_xx" });
  assert.equal(host.stepCounts().reads, 3);
  host.close();
});

test("spring and hinge requests are keyed apart from a nodal one", async () => {
  const springs = await opened("ladruno_springs3d.ladruno");
  const s0 = springs.meta.stages[0]!.id;
  const n = springs.host.readStep({ path: springs.path, stage: s0, step: 1, seq: 1 });
  const sp = springs.host.readStep({ path: springs.path, stage: s0, step: 1, seq: 2, springs: true });
  assert.ok(n.ok && sp.ok);
  assert.equal(n.springs, undefined);
  assert.ok(sp.springs, sp.springsError);
  assert.equal(springs.host.stepCounts().reads, 2);
  springs.host.close();
  const hinges = await opened("ladruno_hinges2d.ladruno");
  const h0 = hinges.meta.stages[0]!.id;
  hinges.host.readStep({ path: hinges.path, stage: h0, step: 1, seq: 1 });
  const hh = hinges.host.readStep({ path: hinges.path, stage: h0, step: 1, seq: 2, hinges: true });
  assert.ok(hh.ok && hh.hinges, hh.ok ? hh.hingesError : "");
  assert.equal(hinges.host.stepCounts().reads, 2);
  hinges.host.close();
});

test("a tool's step range reads through the viewport's read of the step", async () => {
  const { host, path, meta } = await opened("ladruno_hinges2d.ladruno");
  const st = meta.stages[0]!;
  const component = st.nodal[0]!;
  const viewport = host.readStep({ path, stage: st.id, step: 5, seq: 1 });
  assert.ok(viewport.ok);
  const r = host.stepRange({ path, stage: st.id, step: 5, component });
  assert.ok(r.ok, !r.ok ? r.error : "");
  assert.deepEqual(host.stepCounts(), { reads: 1, hits: 1 }, "the range was the viewport's read");
  const v = Array.from(viewport.nodal[component]!).filter(Number.isFinite);
  assert.deepEqual([r.min, r.max, r.nodes], [Math.min(...v), Math.max(...v), v.length]);
  const bad = host.stepRange({ path, stage: st.id, step: 5, component: "nope" });
  assert.equal(bad.ok, false);
  assert.match(!bad.ok ? bad.error : "", /no nodal component "nope"/);
  host.close();
});
