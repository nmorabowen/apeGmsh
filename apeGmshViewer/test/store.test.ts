// The D6 store: the reducer is pure (no mutation, plain data only), the
// event union is exhaustive, and the chain the selector derives from the
// state equals the V1 resolver's chain on the fixture and on the synthetic
// fail-closed models (a cross-engine oracle).

import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import { resolveChain, walk, type Chain, type ChainNode, type ElementRef } from "../src/chain/resolve.ts";
import { buildMesh } from "../src/mesh/build.ts";
import type { ElementMeta, ModelFile, OpsObject, Param } from "../src/model/types.ts";
import { openModel } from "../src/reader/node.ts";
import { BlobStore } from "../src/state/blobs.ts";
import { cellPath, opsRowPath } from "../src/state/decls.ts";
import { loadModel } from "../src/state/load.ts";
import { chainOf, inspected, refusalsOf } from "../src/state/selectors.ts";
import { initialState, reduce, Store, type Event, type State } from "../src/state/store.ts";
import type { EventType, ModelLoad, Pick } from "../src/state/types.ts";
import { blobRefsOf, Effects, parseRefusal, type Bridge, type OpenSet } from "../src/effects.ts";

const FIXTURE = fileURLToPath(new URL("../fixtures/shoebuckle.h5", import.meta.url));

// ---- plain-data check -------------------------------------------------------

/** Fails on anything that is not a record, array, string, number, boolean or null. */
export function assertPlain(v: unknown, at = "state"): void {
  if (v === null || typeof v === "string" || typeof v === "number" || typeof v === "boolean") return;
  if (typeof v !== "object") throw new Error(`${at}: ${typeof v} is not plain data`);
  if (ArrayBuffer.isView(v) || v instanceof ArrayBuffer) throw new Error(`${at}: a ${v.constructor.name} is in the state`);
  if (v instanceof Map || v instanceof Set) throw new Error(`${at}: a ${v.constructor.name} is in the state`);
  if (Array.isArray(v)) {
    v.forEach((x, i) => assertPlain(x, `${at}[${i}]`));
    return;
  }
  const proto = Object.getPrototypeOf(v);
  if (proto !== Object.prototype && proto !== null) throw new Error(`${at}: a ${proto.constructor?.name} instance is in the state`);
  for (const [k, x] of Object.entries(v)) assertPlain(x, `${at}.${k}`);
}

function deepFreeze<T>(v: T): T {
  if (v && typeof v === "object" && !Object.isFrozen(v)) {
    Object.freeze(v);
    for (const x of Object.values(v as object)) deepFreeze(x);
  }
  return v;
}

test("the plain-data check itself catches a planted TypedArray, Map and class instance", () => {
  assert.throws(() => assertPlain({ a: new Float32Array(2) }), /Float32Array/);
  assert.throws(() => assertPlain({ a: { b: new Map() } }), /Map/);
  assert.throws(() => assertPlain({ a: new ArrayBuffer(8) }), /ArrayBuffer/);
  assert.throws(() => assertPlain({ a: new Set() }), /Set/);
  assert.throws(() => assertPlain({ a: new (class Foo {})() }), /Foo/);
  assertPlain({ a: [1, "x", null, { b: true }] });
});

test("a planted impure reducer is caught by the frozen input", () => {
  const impure = (s: State): State => {
    (s.selection as { decls: string[] }).decls.push("x");
    return s;
  };
  assert.throws(() => impure(deepFreeze(structuredClone(initialState))), TypeError);
});

// ---- the loaded fixture -----------------------------------------------------

const model = await openModel(FIXTURE);
const blobs = new BlobStore();
const load: ModelLoad = loadModel(model, blobs);
const loaded: State = reduce(initialState, { type: "fileLoaded", artifact: "model", load });
const archRef = (): { kind: "fem"; blockIndex: number; row: number } => {
  const blockIndex = model.blocks.findIndex((b) => b.alias === "line2");
  const arch = model.physicalGroups.find((g) => g.name === "Arch")!;
  const row = Array.from(model.blocks[blockIndex]!.ids).indexOf(arch.elementIds[10]!);
  return { kind: "fem", blockIndex, row };
};
const archPath = cellPath(model.blocks[archRef().blockIndex]!.ids[archRef().row]!);
const pick = (decl: string): Pick => ({ decl, at: null, nodes: null });
/** The node tags of a cell, as the viewport reads them from the connectivity blob. */
function nodesOf(m: ModelFile, r: ElementRef): number[] | null {
  if (r.kind !== "fem") return null;
  const b = m.blocks[r.blockIndex]!;
  return Array.from(b.connectivity.subarray(r.row * b.npe, (r.row + 1) * b.npe));
}

test("the loaded state is plain data, and every array of the file is a blob the store holds", () => {
  assertPlain(loaded);
  const refs = blobRefsOf(loaded);
  assert.ok(refs.length >= 7, "mesh blobs");
  for (const r of refs) assert.ok(blobs.has(r), r.id);
  assert.equal(loaded.mesh!.linePositions.shape[0], loaded.mesh!.elements.length);
  assert.equal(loaded.mesh!.elements.length, 2 * 22 + 79, "one segment per beam element");
});

test("decision 15: an element holds a range into the connectivity blob, never a node copy", () => {
  const d = loaded.decls[archPath]!;
  assert.ok(!("nodes" in d.element!), "no per-element node array");
  assert.deepEqual(d.element!.inlineNodes, []);
  const block = loaded.blocks[d.element!.cell!.block]!;
  assert.equal(block.alias, "line2");
  assert.ok(blobs.has(block.connectivity));
  const conn = blobs.get(block.connectivity);
  const r = archRef();
  assert.deepEqual(Array.from(conn.subarray(r.row * block.npe, (r.row + 1) * block.npe)), nodesOf(model, r));
  // The shared empties are one object each, not one per element.
  const other = loaded.decls[loaded.mesh!.elements[0]!]!;
  assert.equal(d.links, other.links);
  assert.equal(d.problems, other.problems);
  assert.equal(d.fields, other.fields);
});

/** One sample of every event type; the Record type makes a missing type a compile error. */
const EVERY: Record<EventType, Event> = {
  fileOpened: { type: "fileOpened", artifact: "geometry", path: "x.geometry.h5" },
  fileLoaded: { type: "fileLoaded", artifact: "model", load },
  setLoaded: { type: "setLoaded", model: load, geometry: null },
  fileFailed: { type: "fileFailed", artifact: "results", path: "x.results.h5", error: "no reader" },
  fileClosed: { type: "fileClosed", artifact: "results" },
  fileChanged: { type: "fileChanged", path: model.path },
  zoneRefused: { type: "zoneRefused", artifact: "model", zone: "neutral", version: "1.9.0", accepted: "2.10 and any later 2.x", newer: false, reason: "major 2 only" },
  select: { type: "select", pick: pick(archPath) },
  selectAdd: { type: "selectAdd", pick: pick(loaded.mesh!.elements[0]!) },
  clearSelection: { type: "clearSelection" },
  hover: { type: "hover", at: archPath },
  setHidden: { type: "setHidden", decls: ["mesh/physical_group/Arch"], hidden: true },
  isolate: { type: "isolate", decls: ["mesh/physical_group/Arch"] },
  showAll: { type: "showAll" },
  setEdges: { type: "setEdges", on: false },
  setOpacity: { type: "setOpacity", value: 0.5 },
  setPhase: { type: "setPhase", at: { kind: "mesh" } },
  setResultStep: { type: "setResultStep", step: 3 },
  inspectorPin: { type: "inspectorPin", decl: archPath },
  inspectorUnpin: { type: "inspectorUnpin", decl: archPath },
  openWindow: { type: "openWindow", window: "stages" },
  closeWindow: { type: "closeWindow", window: "stages" },
  requestSource: { type: "requestSource", decl: archPath },
  sourceResult: { type: "sourceResult", decl: archPath, ok: false, reason: "missing: x.py" },
};

test("reducer purity: every event leaves a frozen state untouched and returns plain data", () => {
  const frozen = deepFreeze(structuredClone(loaded));
  for (const e of Object.values(EVERY)) {
    let next: State;
    try {
      next = reduce(frozen, e);
    } catch (err) {
      // setResultStep refuses without a results axis: that is a thrown Error, not a mutation.
      assert.ok(err instanceof Error && !(err instanceof TypeError), `${e.type}: ${err}`);
      continue;
    }
    assertPlain(next, e.type);
  }
  assert.equal(Object.keys(EVERY).length, 24, "decision 17's 20 events plus fileClosed, requestSource, sourceResult and setLoaded (V2f)");
});

test("an event the union does not know raises at run time", () => {
  assert.throws(() => reduce(loaded, { type: "nope" } as unknown as Event), /unknown event/);
});

test("select, selectAdd, clearSelection, pins; an undeclared path is refused", () => {
  let s = reduce(loaded, EVERY.select);
  assert.deepEqual(s.selection.decls, [archPath]);
  s = reduce(s, EVERY.selectAdd);
  assert.equal(s.selection.decls.length, 2);
  assert.equal(reduce(s, EVERY.selectAdd), s, "a repeat add is a no-op");
  assert.throws(() => reduce(s, { type: "select", pick: pick("mesh/element/999999") }), /not a declaration/);
  s = reduce(s, EVERY.inspectorPin);
  assert.deepEqual(s.inspector.pinned, [archPath]);
  assert.equal(inspected(s).length, 2, "the pinned chain and the other selected one");
  s = reduce(s, EVERY.clearSelection);
  assert.deepEqual(s.selection.decls, []);
  assert.equal(inspected(s).length, 1, "the pin survives the clear");
  s = reduce(s, EVERY.inspectorUnpin);
  assert.deepEqual(s.inspector.pinned, []);
});

test("visibility: hide, isolate, showAll, edges, opacity clamped, NaN refused", () => {
  let s = reduce(loaded, EVERY.setHidden);
  assert.deepEqual(s.visibility.hidden, ["mesh/physical_group/Arch"]);
  s = reduce(s, { type: "setHidden", decls: ["mesh/physical_group/Arch"], hidden: false });
  assert.deepEqual(s.visibility.hidden, []);
  s = reduce(s, EVERY.isolate);
  assert.deepEqual(s.visibility.hidden.sort(), ["mesh/physical_group/Base", "mesh/physical_group/Frame", "mesh/physical_group/LeftColumn", "mesh/physical_group/RightColumn"]);
  s = reduce(s, EVERY.showAll);
  assert.deepEqual(s.visibility.hidden, []);
  assert.equal(reduce(s, EVERY.setEdges).visibility.edges, false);
  assert.equal(reduce(s, { type: "setOpacity", value: 7 }).visibility.opacity, 1);
  assert.throws(() => reduce(s, { type: "setOpacity", value: NaN }), RangeError);
});

test("phase: the mesh is on the axis after a load; a key off the axis is refused", () => {
  assert.deepEqual(loaded.phase, { axis: [{ kind: "mesh" }], at: { kind: "mesh" } });
  assert.throws(() => reduce(loaded, { type: "setPhase", at: { kind: "stage", index: 0 } }), /not on the phase axis/);
  assert.throws(() => reduce(loaded, EVERY.setResultStep), /no results/);
});

test("files: opened siblings, a stale model after fileChanged, a refused zone, a failure", () => {
  let s = reduce(loaded, EVERY.fileOpened);
  assert.equal(s.artifacts.geometry?.status, "opened");
  s = reduce(s, EVERY.fileChanged);
  assert.equal(s.artifacts.model?.stale, true);
  s = reduce(s, { type: "fileOpened", artifact: "model", path: model.path });
  assert.equal(s.artifacts.model?.stale, false, "re-opening the same path clears stale and keeps what was read");
  assert.equal(s.mesh, loaded.mesh);
  s = reduce(s, EVERY.zoneRefused);
  assert.deepEqual(refusalsOf(s), [
    "model file written by an older apeGmsh: its neutral zone is version 1.9.0, this app reads 2.10 and any later 2.x (major 2 only)",
  ]);
  const newer = reduce(s, { ...EVERY.zoneRefused, version: "3.0.0", newer: true } as Event);
  assert.match(refusalsOf(newer)[0]!, /^model file written by a newer apeGmsh: its neutral zone is version 3.0.0, this app reads 2.10 and any later 2.x/);
  assert.throws(() => reduce(initialState, EVERY.zoneRefused), /before fileOpened/);
  const failed = reduce(s, { type: "fileFailed", artifact: "model", path: model.path, error: "boom" });
  assert.equal(failed.mesh, null);
  assert.deepEqual(Object.keys(failed.decls), []);
  assert.equal(failed.artifacts.model?.status, "failed");
  assert.equal(failed.artifacts.geometry?.status, "opened", "the siblings survive the model's failure");
  assert.equal(reduce(loaded, EVERY.fileFailed).artifacts.results?.error, "no reader");
});

test("fileClosed drops a sibling, or the model with its derivations; a mid-read fileChanged survives fileLoaded", () => {
  let s = reduce(loaded, { type: "fileOpened", artifact: "results", path: "x.results.h5" });
  s = reduce(s, EVERY.fileClosed);
  assert.equal(s.artifacts.results, null);
  assert.equal(reduce(s, EVERY.fileClosed), s, "closing nothing is a no-op");
  const closed = reduce(s, { type: "fileClosed", artifact: "model" });
  assert.equal(closed.artifacts.model, null);
  assert.equal(closed.mesh, null);
  assert.deepEqual(closed.blocks, []);
  // fileOpened, then fileChanged while the read runs, then fileLoaded: still stale.
  let t = reduce(initialState, { type: "fileOpened", artifact: "model", path: model.path });
  t = reduce(t, EVERY.fileChanged);
  t = reduce(t, EVERY.fileLoaded);
  assert.equal(t.artifacts.model?.stale, true);
  assert.equal(reduce(initialState, EVERY.fileLoaded).artifacts.model?.stale, false);
});

test("a group or alias named like an Object.prototype member is just a name", () => {
  const m = structuredClone(model);
  m.physicalGroups.push({ name: "constructor", dim: 1, tag: 99, path: "/physical_groups/element_side/constructor", elementIds: new Float64Array([m.blocks[0]!.ids[0]!]) });
  m.labels.push({ name: "__proto__", dim: 1, tag: 98, path: "/labels/__proto__", elementIds: new Float64Array([]) });
  m.opensees!.names.push({ name: "toString", kind: "geomTransf", tag: 1 });
  const s = reduce(initialState, { type: "fileLoaded", artifact: "model", load: loadModel(m, new BlobStore()) });
  assert.deepEqual(s.names["constructor"], ["mesh/physical_group/constructor"]);
  assert.deepEqual(s.names["__proto__"], ["mesh/label/__proto__"]);
  assert.deepEqual(s.names["toString"], ["opensees/geomTransf/toString"]);
  assert.equal(s.names["hasOwnProperty"], undefined);
  assert.equal(s.decls["constructor"], undefined);
  assert.equal(Object.getPrototypeOf(s.names), null);
  assert.equal(Object.getPrototypeOf(s.decls), null);
});

// ---- effects sequencing ------------------------------------------------------

function deferred<T>() {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => (resolve = r));
  return { promise, resolve };
}

/** A bridge whose reads resolve when the test says; `window` is stubbed for attach(). */
function harness() {
  const reads: { path: string; d: ReturnType<typeof deferred<{ ok: true; model: ModelFile } | { ok: false; error: string }>> }[] = [];
  const bridge = {
    openModel: (path: string) => {
      const d = deferred<{ ok: true; model: ModelFile } | { ok: false; error: string }>();
      reads.push({ path, d });
      return d.promise;
    },
  } as unknown as Bridge;
  const store = new Store();
  const effects = new Effects(store, new BlobStore(), bridge);
  const modelAt = (path: string): ModelFile => ({ ...model, path });
  return { reads, store, effects, modelAt };
}

const tick = () => new Promise((r) => setTimeout(r, 0));

test("effects: open(A) then open(B); A resolving last does not replace B (latest wins)", async () => {
  const { reads, store, effects, modelAt } = harness();
  const pa = effects.open("A.h5");
  const pb = effects.open("B.h5");
  assert.equal(reads.length, 2);
  reads[1]!.d.resolve({ ok: true, model: modelAt("B.h5") });
  assert.equal(await pb, true);
  assert.equal(store.get().artifacts.model?.path, "B.h5");
  const meshB = store.get().mesh;
  reads[0]!.d.resolve({ ok: true, model: modelAt("A.h5") });
  assert.equal(await pa, false, "a stale read reports that it did not land");
  assert.equal(store.get().artifacts.model?.path, "B.h5");
  assert.equal(store.get().mesh, meshB);
});

test("effects: a fileChanged that arrives during a read is read once more afterwards", async () => {
  const { reads, store, effects, modelAt } = harness();
  const p = effects.open("A.h5");
  store.dispatch({ type: "fileChanged", path: "A.h5" });
  assert.equal(reads.length, 1, "the change is queued, not read while a read runs");
  reads[0]!.d.resolve({ ok: true, model: modelAt("A.h5") });
  assert.equal(await p, true);
  await tick();
  assert.equal(reads.length, 2, "one re-read after the first lands");
  assert.equal(store.get().artifacts.model?.status, "ready");
  reads[1]!.d.resolve({ ok: true, model: modelAt("A.h5") });
  await tick();
  await tick();
  assert.equal(reads.length, 2, "and only one");
  assert.equal(store.get().artifacts.model?.stale, false);
});

test("effects: a fileChanged during a read that fails is retried, at most MAX_RETRIES times in a row", async () => {
  const { reads, store, effects, modelAt } = harness();
  const fail = { ok: false as const, error: "/nodes is missing" };
  const p = effects.open("A.h5");
  store.dispatch({ type: "fileChanged", path: "A.h5" });
  reads[0]!.d.resolve(fail);
  assert.equal(await p, false);
  await tick();
  assert.equal(reads.length, 2, "the failed read is retried once, since the file changed under it");
  // A broken file that keeps changing: the retries stop after MAX_RETRIES.
  for (let k = 2; k <= Effects.MAX_RETRIES + 1; k++) {
    store.dispatch({ type: "fileChanged", path: "A.h5" });
    reads[k - 1]!.d.resolve(fail);
    await tick();
    await tick();
  }
  assert.equal(reads.length, Effects.MAX_RETRIES + 1, "MAX_RETRIES re-reads after the first, then no more");
  assert.equal(store.get().artifacts.model?.status, "failed");
  // A change after the retries ran out starts a fresh count; a success resets it.
  store.dispatch({ type: "fileChanged", path: "A.h5" });
  await tick();
  assert.equal(reads.length, Effects.MAX_RETRIES + 2);
  reads[reads.length - 1]!.d.resolve({ ok: true, model: modelAt("A.h5") });
  await tick();
  await tick();
  assert.equal(store.get().artifacts.model?.status, "ready");
  assert.equal(store.get().artifacts.model?.stale, false);
  assert.equal(reads.length, Effects.MAX_RETRIES + 2, "a landed read ends the sequence");
});

test("effects: an open set with results: null closes the results artifact", async () => {
  const { reads, store, effects, modelAt } = harness();
  store.dispatch({ type: "fileOpened", artifact: "results", path: "old.results.h5" });
  effects.openSet({ model: "A.h5", geometry: "A.geometry.h5", results: null } satisfies OpenSet);
  assert.equal(store.get().artifacts.results, null);
  assert.equal(store.get().artifacts.geometry?.path, "A.geometry.h5");
  reads[0]!.d.resolve({ ok: true, model: modelAt("A.h5") });
  await tick();
  assert.equal(store.get().artifacts.model?.status, "ready");
  effects.openSet({ model: null, geometry: null, results: null });
  assert.equal(store.get().artifacts.model, null);
  assert.equal(store.get().mesh, null);
});

test("effects: a refused read records the zone and the failure; the failure is never silent", async () => {
  const { reads, store, effects } = harness();
  const p = effects.open("old.h5");
  reads[0]!.d.resolve({ ok: false, error: "neutral_schema_version 3.0.0: this app reads major 2 only" });
  assert.equal(await p, false);
  assert.match(refusalsOf(store.get())[0]!, /a newer apeGmsh.*3\.0\.0.*2\.10 and any later 2\.x/);
  assert.equal(store.get().artifacts.model?.status, "failed");
});

test("windows open once and close", () => {
  let s = reduce(loaded, EVERY.openWindow);
  assert.equal(reduce(s, EVERY.openWindow), s);
  assert.deepEqual(s.windows.open, ["stages"]);
  s = reduce(s, EVERY.closeWindow);
  assert.deepEqual(s.windows.open, []);
});

test("the Store notifies on a change only, and unsubscribe stops it", () => {
  const store = new Store();
  const seen: string[] = [];
  const off = store.subscribe((s, prev) => seen.push(`${prev.mesh === null}->${s.mesh === null}`));
  store.dispatch({ type: "clearSelection" }); // no change: no notification
  store.dispatch({ type: "fileLoaded", artifact: "model", load });
  assert.deepEqual(seen, ["true->false"]);
  off();
  store.dispatch({ type: "clearSelection" });
  store.dispatch({ type: "select", pick: pick(archPath) });
  assert.equal(seen.length, 1);
});

test("the reader's refusal message parses into a zone, a version, the accepted range (floor through the major) and older/newer", () => {
  assert.deepEqual(parseRefusal("neutral_schema_version 3.0.0: this app reads major 2 only"), {
    zone: "neutral", version: "3.0.0", accepted: "2.10 and any later 2.x", newer: true, reason: "this app reads major 2 only",
  });
  assert.deepEqual(parseRefusal("neutral_schema_version 1.4.0: this app reads major 2 only"), {
    zone: "neutral", version: "1.4.0", accepted: "2.10 and any later 2.x", newer: false, reason: "this app reads major 2 only",
  });
  const old = parseRefusal("neutral_schema_version 2.9.0: layouts before 2.10 are not supported");
  assert.equal(old?.accepted, "2.10 and any later 2.x", "the floor through the major, never the 2.32-2.33 warn window");
  assert.doesNotMatch(old!.accepted, /2\.3[23]/);
  assert.equal(old?.newer, false);
  // The opensees floor is 2.11, the 0-based rank flip (ADR 0113 D1).
  assert.equal(parseRefusal("opensees_schema_version 3.1.0: this app reads major 2 only")?.accepted, "2.11 and any later 2.x");
  assert.equal(parseRefusal("/nodes is missing"), null);
});

test("the accepted range comes from the reader's one floor table", async () => {
  const { ZONE_FLOOR, NEUTRAL_TARGET } = await import("../src/reader/read.ts");
  const r = parseRefusal("neutral_schema_version 1.0.0: this app reads major 2 only")!;
  assert.equal(r.accepted, `${NEUTRAL_TARGET.major}.${ZONE_FLOOR.neutral} and any later ${NEUTRAL_TARGET.major}.x`);
});

// ---- declaration paths and names -------------------------------------------

test("declaration paths: tag-free keys; names index the registered aliases, labels and groups", () => {
  const paths = Object.keys(loaded.decls);
  assert.ok(paths.includes(archPath));
  assert.ok(paths.includes("mesh/physical_group/Arch"));
  const objects = paths.filter((p) => p.startsWith("opensees/") && !p.startsWith("opensees/element/"));
  // The fixture registers no names: every object is `opensees/<family>/#k`.
  for (const p of objects) assert.match(p, /^opensees\/(uniaxialMaterial|nDMaterial|section|geomTransf|beamIntegration)\/#\d+$/);
  for (const p of objects) assert.doesNotMatch(p, /_\d+$/, "no {type}_{tag} group name as a key");
  // The fixture carries "Arch" both as a label and as a physical group: one name, two declarations.
  assert.deepEqual(loaded.names["Arch"], ["mesh/physical_group/Arch", "mesh/label/Arch"]);
  assert.equal(loaded.decls["mesh/label/Arch"]!.kind, "label");
  assert.deepEqual(loaded.decls["mesh/physical_group/Arch"]!.group?.count, 79);
  assert.equal(Object.keys(loaded.overrides).length, 0);
});

test("a registered name keys the object and lands in the names index", () => {
  const m = structuredClone(model);
  m.opensees!.names.push({ name: "deck", kind: "section", tag: 1 });
  const s = reduce(initialState, { type: "fileLoaded", artifact: "model", load: loadModel(m, new BlobStore()) });
  assert.ok("opensees/section/deck" in s.decls);
  assert.deepEqual(s.names["deck"], ["opensees/section/deck"]);
  assert.equal(s.decls["opensees/section/deck"]!.nameSource, "/opensees/names");
  assert.equal(chainOf(s, archPath)!.root.children[1]!.children[0]!.name, "deck");
});

// ---- cross-engine oracle: selector chain == V1 resolver chain ----------------

function flat(c: Chain) {
  const node = (n: ChainNode) => ({
    role: n.role, name: n.name, nameSource: n.nameSource, type: n.type, path: n.path,
    via: n.via, fields: n.fields,
  });
  return { nodes: walk(c.root).map(({ node: n, depth }) => ({ depth, ...node(n) })), problems: [...c.problems].sort() };
}

function pathOf(m: ModelFile, r: ElementRef): string {
  return r.kind === "fem" ? cellPath(m.blocks[r.blockIndex]!.ids[r.row]!) : opsRowPath(m.opensees!.elementMeta[r.metaIndex]!.type, r.row);
}

test("oracle: on the fixture, every drawn element's chain from the state equals resolveChain's", () => {
  const mesh = buildMesh(model);
  const refs = [...mesh.lineRefs, ...mesh.triRefs];
  assert.ok(refs.length > 100);
  for (const r of refs) {
    const expected = flat(resolveChain(model, r));
    const got = flat(chainOf(loaded, pathOf(model, r), nodesOf(model, r))!);
    assert.deepEqual(got, expected, JSON.stringify(r));
  }
});

test("without a pick (a pinned element) the nodes row names the source and the count", () => {
  const c = chainOf(loaded, archPath)!;
  const nodes = c.root.fields.find((f) => f.label === "nodes")!;
  assert.equal(nodes.value, "2 node tags (select the element to list them)");
  assert.match(nodes.source, /^\/elements\/line2\/connectivity\[\d+\]$/);
  const s = reduce(loaded, { type: "select", pick: { decl: archPath, at: null, nodes: nodesOf(model, archRef()) } });
  assert.equal(inspected(s)[0]!.chain.root.fields.find((f) => f.label === "nodes")!.value, nodesOf(model, archRef())!.join(", "));
});

// Synthetic models (the failclosed and decoders test helpers), including the
// fail-closed cases: each problem text must match the resolver's.
function obj(family: OpsObject["family"], dir: string, type: string, tag: number, params: Param[] = [], tables: OpsObject["tables"] = {}): OpsObject {
  const groupName = `${type}_${tag}`;
  return { family, path: `${dir}/${groupName}`, groupName, type, tag, params, attrs: {}, tables };
}
function meta(type: string, args: Param[], femEid = 1): ElementMeta {
  return { type, path: `/opensees/element_meta/${type}`, ids: new Float64Array([7]), femEids: new Float64Array([femEid]), args: [args], inlineConnectivity: femEid < 0 ? [[1, 2]] : null };
}
function synthetic(objects: OpsObject[], elementMeta: ElementMeta[], opensees = true): ModelFile {
  return {
    path: "synthetic.h5", sizeBytes: 0, neutralVersion: "2.33.0", meta: {},
    nodeIds: new Float64Array([1, 2]), nodeCoords: new Float64Array([0, 0, 0, 1, 0, 0]),
    blocks: [{ alias: "line2", code: 1, dim: 1, npe: 2, ids: new Float64Array([1]), connectivity: new Float64Array([1, 2]) }],
    physicalGroups: [{ name: "G", dim: 1, tag: 1, path: "/physical_groups/element_side/G", elementIds: new Float64Array([1]) }],
    labels: [],
    opensees: opensees ? { version: "2.21.0", objects, elementMeta, names: [] } : null,
    warnings: [], readMs: 0,
  };
}
const T = "/opensees/transforms", BI = "/opensees/beam_integration", S = "/opensees/sections", U = "/opensees/materials/uniaxial", ND = "/opensees/materials/nd";
const fiber = (matPath: string) => obj("section", S, "Fiber", 1, [], { patches: { path: `${S}/Fiber_1/patches`, columns: ["kind", "material_ref"], rows: [["rect", matPath]] } });
const complete = () => [obj("geomTransf", T, "Linear", 1), obj("beamIntegration", BI, "Lobatto", 1, [1, 5]), fiber(`${U}/Steel02_1`), obj("uniaxialMaterial", U, "Steel02", 1, [420e6, 2e11, 0.01])];

const CASES: [string, ModelFile, ElementRef][] = [
  ["complete", synthetic(complete(), [meta("forceBeamColumn", [1, 1])]), { kind: "fem", blockIndex: 0, row: 0 }],
  ["unknown element type", synthetic(complete(), [meta("FooBeam", [1, 1])]), { kind: "fem", blockIndex: 0, row: 0 }],
  ["tag matches none", synthetic(complete(), [meta("forceBeamColumn", [9, 1])]), { kind: "fem", blockIndex: 0, row: 0 }],
  ["tag matches two", synthetic([...complete(), obj("geomTransf", T, "PDelta", 1)], [meta("forceBeamColumn", [1, 1])]), { kind: "fem", blockIndex: 0, row: 0 }],
  ["unknown integration", synthetic([complete()[0]!, obj("beamIntegration", BI, "UserHinge", 1, [1, 2, 3]), complete()[2]!, complete()[3]!], [meta("forceBeamColumn", [1, 1])]), { kind: "fem", blockIndex: 0, row: 0 }],
  ["dangling material_ref", synthetic([complete()[0]!, complete()[1]!, fiber(`${U}/Missing_9`), complete()[3]!], [meta("forceBeamColumn", [1, 1])]), { kind: "fem", blockIndex: 0, row: 0 }],
  ["no element_meta row", synthetic(complete(), []), { kind: "fem", blockIndex: 0, row: 0 }],
  ["no /opensees zone", synthetic([], [], false), { kind: "fem", blockIndex: 0, row: 0 }],
  ["old-style beam refused", synthetic(complete(), [meta("forceBeamColumn", [5, 1, 1])]), { kind: "fem", blockIndex: 0, row: 0 }],
  ["missing slot", synthetic(complete(), [meta("forceBeamColumn", [1, NaN])]), { kind: "fem", blockIndex: 0, row: 0 }],
  ["section without materials", synthetic([obj("geomTransf", T, "Linear", 4), obj("section", S, "ElasticMembranePlateSection", 9, [1, 2, 3, 4])], [meta("ASDShellQ4", [9, "-corotational"])]), { kind: "fem", blockIndex: 0, row: 0 }],
  ["section links not decoded", synthetic([obj("section", S, "Aggregator", 9)], [meta("ASDShellQ4", [9])]), { kind: "fem", blockIndex: 0, row: 0 }],
  ["solid matTag", synthetic([obj("nDMaterial", ND, "ElasticIsotropic", 3)], [meta("LadrunoBrick", [3, "-formulation", "bbar"])]), { kind: "fem", blockIndex: 0, row: 0 }],
  ["OpenSees-only truss row", synthetic([obj("uniaxialMaterial", U, "Elastic", 1)], [meta("CorotTruss", [2.8e-4, 1], -1)]), { kind: "ops", metaIndex: 0, row: 0 }],
];

for (const [name, m, ref] of CASES) {
  test(`oracle (synthetic): ${name}`, () => {
    const s = reduce(initialState, { type: "fileLoaded", artifact: "model", load: loadModel(m, new BlobStore()) });
    const expected = flat(resolveChain(m, ref));
    const got = flat(chainOf(s, pathOf(m, ref), nodesOf(m, ref))!);
    assert.deepEqual(got, expected);
  });
}

test("a group has no chain; an unknown path has none; a cycle is reported, never looped", () => {
  assert.equal(chainOf(loaded, "mesh/physical_group/Arch"), null);
  assert.equal(chainOf(loaded, "nowhere/x/y"), null);
  // Plant a cycle: a section whose material_ref names the section itself.
  const m = synthetic([obj("geomTransf", T, "Linear", 1), obj("beamIntegration", BI, "Lobatto", 1, [1, 5]), fiber(`${S}/Fiber_1`)], [meta("forceBeamColumn", [1, 1])]);
  const s = reduce(initialState, { type: "fileLoaded", artifact: "model", load: loadModel(m, new BlobStore()) });
  const c = chainOf(s, cellPath(1))!;
  assert.ok(c.problems.some((p) => /closes a cycle/.test(p)), c.problems.join(" | "));
});

test("the legend carries a generated colour per group and the two synthetic rows", () => {
  const legend = loaded.mesh!.legend;
  assert.deepEqual(legend.map((e) => [e.name, e.elements, e.decl]), [
    ["Arch", 79, "mesh/physical_group/Arch"],
    ["LeftColumn", 22, "mesh/physical_group/LeftColumn"],
    ["RightColumn", 22, "mesh/physical_group/RightColumn"],
  ]);
  const hues = new Set(legend.map((e) => e.color.join(",")));
  assert.equal(hues.size, 3, "three distinct colours");
  const groups = blobs.i32(loaded.mesh!.lineGroup);
  assert.equal(groups.length, loaded.mesh!.elements.length);
  for (const g of groups) assert.ok(g >= 0 && g < legend.length);
});
