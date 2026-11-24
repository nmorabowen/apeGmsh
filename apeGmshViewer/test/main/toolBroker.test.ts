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
import { ResultsHost } from "../../src/main/results.ts";
import type { ToolAnswer, ToolRequest } from "../../src/tools/contract.ts";
import type { ResultsH5 } from "../../src/reader/results.ts";
import { defaultComponent } from "../../src/tools/step.ts";

const q = (node: number, component = "displacement_x") => ({ kind: "nodeHistory" as const, stage: "stage_0", node, component });

test("only a well-formed nodeHistory request passes", () => {
  assert.deepEqual(checkToolRequest(q(3)), { ok: true, q: q(3) });
  for (const bad of [null, 7, { kind: "readFile", path: "C:/x" }, { ...q(3), path: "C:/elsewhere.ladruno" }, { ...q(3), node: 1.5 }, { ...q(3), node: "3" }, { kind: "nodeHistory", stage: "s", node: 3 }]) {
    assert.equal(checkToolRequest(bad).ok, false, JSON.stringify(bad));
  }
});

type Asked = { path: string; node: number; stage: string; component: string };

/** A read the test finishes by hand, recording what it was asked. */
function manualRead() {
  const asked: Asked[] = [];
  const finish: ((a: ToolAnswer) => void)[] = [];
  const read = (path: string, q: ToolRequest) =>
    new Promise<ToolAnswer>((r) => {
      if (q.kind !== "nodeHistory") throw new Error("these tests ask for histories");
      asked.push({ path, node: q.node, stage: q.stage, component: q.component });
      finish.push(r);
    });
  const ok = (req: Asked): ToolAnswer => ({ ok: true, kind: "nodeHistory", stage: req.stage, node: req.node, component: req.component, values: Float64Array.from([req.node, req.node * 2]) });
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
  assert.deepEqual(a1.ok && a1.kind === "nodeHistory" && Array.from(a1.values), [3, 6]);
  assert.equal(a2.ok, false, "tool B asked for node 4 after node 3: its node 3 answer is superseded");
  assert.ok(!a2.ok && a2.superseded);
  assert.deepEqual(a3.ok && a3.kind === "nodeHistory" && Array.from(a3.values), [4, 8]);
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
  assert.equal(n.ok && n.kind === "nodeHistory" && n.node, 2);
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

test("H3: a well-formed stepRange request passes; a bad step or an extra field does not", () => {
  const r = { kind: "stepRange", stage: "stage_0", step: 4, component: "displacement_x" };
  assert.deepEqual(checkToolRequest(r), { ok: true, q: r });
  for (const bad of [{ ...r, step: -1 }, { ...r, step: 2.5 }, { ...r, node: 3 }, { kind: "stepRange", stage: "s", component: "c" }]) {
    assert.equal(checkToolRequest(bad).ok, false, JSON.stringify(bad));
  }
});

test("H3: a stepRange and a nodeHistory for the same stage and component are different reads", async () => {
  const seen: string[] = [];
  const b = new ToolBroker(async (_path, q) => {
    seen.push(q.kind);
    return q.kind === "stepRange" ? { ok: true, kind: "stepRange", stage: q.stage, step: q.step, component: q.component, min: 0, max: 1, nodes: 2 } : { ok: false, error: "x" };
  });
  const tool = {};
  await Promise.all([b.request(tool, "/r", { kind: "stepRange", stage: "s", step: 1, component: "c" }, 1), b.request({}, "/r", { kind: "nodeHistory", stage: "s", node: 1, component: "c" }, 1)]);
  assert.deepEqual(seen.sort(), ["nodeHistory", "stepRange"]);
});

test("H3: newest wins per kind: a tool's step range never supersedes its history, and the reverse", async () => {
  const finish: (() => void)[] = [];
  const b = new ToolBroker(
    (_path, q) =>
      new Promise<ToolAnswer>((r) =>
        finish.push(() => r(q.kind === "stepRange" ? { ok: true, kind: "stepRange", stage: q.stage, step: q.step, component: q.component, min: 0, max: 1, nodes: 2 } : { ok: true, kind: "nodeHistory", stage: q.stage, node: q.node, component: q.component, values: [1, 2] })),
      ),
  );
  const tool = {};
  // The Step readout's order: the history (seq 1), then the range of the step on screen (seq 2).
  const history = b.request(tool, "/r", { kind: "nodeHistory", stage: "s", node: 91, component: "c" }, 1);
  const range = b.request(tool, "/r", { kind: "stepRange", stage: "s", step: 1, component: "c" }, 2);
  finish.forEach((f) => f());
  const [h, r] = await Promise.all([history, range]);
  assert.equal(h.ok && h.kind, "nodeHistory", JSON.stringify(h));
  assert.equal(r.ok && r.kind, "stepRange", JSON.stringify(r));
});

test("H4: newest wins per tile: two Node history tiles of one window never supersede each other", async () => {
  const finish: (() => void)[] = [];
  const b = new ToolBroker(
    (_path, q) =>
      new Promise<ToolAnswer>((r) =>
        finish.push(() => r(q.kind === "nodeHistory" ? { ok: true, kind: "nodeHistory", stage: q.stage, node: q.node, component: q.component, values: [q.node] } : { ok: false, error: "x" })),
      ),
  );
  const window = {};
  const tileA = b.request(window, "/r", { kind: "nodeHistory", stage: "s", node: 1, component: "ux" }, 1, 1);
  const tileB = b.request(window, "/r", { kind: "nodeHistory", stage: "s", node: 1, component: "uy" }, 2, 2);
  // Tile A asks again: its first request is superseded, tile B's is not.
  const tileA2 = b.request(window, "/r", { kind: "nodeHistory", stage: "s", node: 1, component: "uz" }, 3, 1);
  finish.forEach((f) => f());
  const [a, bb, a2] = await Promise.all([tileA, tileB, tileA2]);
  assert.ok(!a.ok && a.superseded, "tile A's older request");
  assert.equal(bb.ok && bb.kind === "nodeHistory" && bb.component, "uy", "tile B is its own stream");
  assert.equal(a2.ok && a2.kind === "nodeHistory" && a2.component, "uz");
});
