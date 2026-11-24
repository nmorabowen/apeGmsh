// ADR 0137 D10 (H2b): the data broker and the read behind it.
//
// - Only a well-formed request passes; a tool names no file.
// - Requests for the same data in flight share one read; different data reads apart; a request
//   after the read finished reads again (no cache in H2b: the shared step read is H3).
// - Newest wins per tool: an older request answered after a newer one is `superseded`; another
//   tool asking for the same data is not affected.
// - `ResultsHost.readNodeHistory` is the node's column of the step reads, on a real fixture.

import assert from "node:assert/strict";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { checkToolRequest, ToolBroker } from "../../src/main/toolBroker.ts";
import { ResultsHost, type NodeHistoryAnswer, type NodeHistoryRequest } from "../../src/main/results.ts";
import type { ResultsH5 } from "../../src/reader/results.ts";
import { defaultComponent } from "../../src/tools/step.ts";

const q = (node: number, component = "displacement_x") => ({ kind: "nodeHistory" as const, stage: "stage_0", node, component });

test("only a well-formed nodeHistory request passes", () => {
  assert.deepEqual(checkToolRequest(q(3)), { ok: true, q: q(3) });
  for (const bad of [null, 7, { kind: "readFile", path: "C:/x" }, { ...q(3), path: "C:/elsewhere.ladruno" }, { ...q(3), node: 1.5 }, { ...q(3), node: "3" }, { kind: "nodeHistory", stage: "s", node: 3 }]) {
    assert.equal(checkToolRequest(bad).ok, false, JSON.stringify(bad));
  }
});

/** A read the test finishes by hand, recording what it was asked. */
function manualRead() {
  const asked: NodeHistoryRequest[] = [];
  const finish: ((a: NodeHistoryAnswer) => void)[] = [];
  const read = (req: NodeHistoryRequest) =>
    new Promise<NodeHistoryAnswer>((r) => {
      asked.push(req);
      finish.push(r);
    });
  const ok = (req: NodeHistoryRequest): NodeHistoryAnswer => ({ ok: true, stage: req.stage, node: req.node, component: req.component, values: Float64Array.from([req.node, req.node * 2]) });
  return { read, asked, finish, ok };
}

test("the same data asked twice while it is read is read once; other data reads apart", async () => {
  const m = manualRead();
  const b = new ToolBroker(m.read);
  const toolA = {}, toolB = {};
  const p1 = b.request(toolA, "/r.ladruno", q(3), 1);
  const p2 = b.request(toolB, "/r.ladruno", q(3), 1);
  const p3 = b.request(toolB, "/r.ladruno", q(4), 2);
  assert.equal(b.reads, 2, "node 3 once, node 4 once");
  assert.deepEqual(m.asked.map((r) => [r.path, r.node]), [["/r.ladruno", 3], ["/r.ladruno", 4]]);
  m.finish[0]!(m.ok(m.asked[0]!));
  m.finish[1]!(m.ok(m.asked[1]!));
  const [a1, a2, a3] = await Promise.all([p1, p2, p3]);
  assert.deepEqual(a1.ok && Array.from(a1.values), [3, 6]);
  assert.equal(a2.ok, false, "tool B asked for node 4 after node 3: its node 3 answer is superseded");
  assert.ok(!a2.ok && a2.superseded);
  assert.deepEqual(a3.ok && Array.from(a3.values), [4, 8]);
  // Finished reads are not kept: asking again reads again.
  const p4 = b.request(toolA, "/r.ladruno", q(3), 2);
  assert.equal(b.reads, 3);
  m.finish[2]!(m.ok(m.asked[2]!));
  assert.equal((await p4).ok, true);
});

test("newest wins per tool: the older answer is superseded, the newer one arrives", async () => {
  const m = manualRead();
  const b = new ToolBroker(m.read);
  const tool = {};
  const older = b.request(tool, "/r.ladruno", q(1), 1);
  const newer = b.request(tool, "/r.ladruno", q(2), 2);
  // The newer read finishes first; then the older one.
  m.finish[1]!(m.ok(m.asked[1]!));
  m.finish[0]!(m.ok(m.asked[0]!));
  const [o, n] = await Promise.all([older, newer]);
  assert.deepEqual(o, { ok: false, superseded: true, error: "superseded by a newer request from this tool" });
  assert.equal(n.ok && n.node, 2);
});

test("a read's refusal reaches the tool with its reason", async () => {
  const b = new ToolBroker(async () => ({ ok: false, error: "stage stage_0 records no node 99" }));
  assert.deepEqual(await b.request({}, "/r.ladruno", q(99), 1), { ok: false, error: "stage stage_0 records no node 99" });
});

test("ResultsHost.readNodeHistory is the node's column of the step reads", async () => {
  const path = fileURLToPath(new URL("../../fixtures/contour_staged.results.h5", import.meta.url));
  const host = new ResultsHost(h5wasm as unknown as ResultsH5, h5wasm.ready);
  const o = await host.open(path);
  assert.ok(o.ok, !o.ok ? o.error : "");
  const stage = o.meta.stages.find((s) => s.time.length > 2)!;
  const ids = o.nodeIds[stage.id]!;
  const node = ids[Math.floor(ids.length / 2)]!;
  const j = Array.from(ids).indexOf(node);
  const a = host.readNodeHistory({ path, stage: stage.id, node, component: "displacement_x" });
  assert.ok(a.ok, !a.ok ? a.error : "");
  const fromSteps = stage.time.map((_, k) => {
    const s = host.readStep({ path, stage: stage.id, step: k, seq: k });
    assert.ok(s.ok);
    return s.nodal["displacement_x"]![j]!;
  });
  assert.deepEqual(Array.from(a.values), fromSteps);
  const missing = host.readNodeHistory({ path, stage: stage.id, node: 987654, component: "displacement_x" });
  assert.equal(missing.ok, false);
  assert.match(!missing.ok ? missing.error : "", /records no node 987654/);
  const other = host.readNodeHistory({ path: "/not/the/open/file.h5", stage: stage.id, node, component: "displacement_x" });
  assert.equal(other.ok, false, "only the file the host indexed");
  host.close();
});

test("the history starts on the stage's x displacement, else its first component", () => {
  assert.equal(defaultComponent(["displacement_y", "displacement_x", "displacement_z"]), "displacement_x");
  assert.equal(defaultComponent(["DISPLACEMENT/Uy", "DISPLACEMENT/Ux"]), "DISPLACEMENT/Ux");
  assert.equal(defaultComponent(["pore_pressure"]), "pore_pressure");
  assert.equal(defaultComponent([]), null);
});
