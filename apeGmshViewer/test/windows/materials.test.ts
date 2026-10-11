// V3c4 (#1645): the materials and sections window.
//
// Oracles: fixtures/definitions.h5, written by hand by make_zone_fixtures.py
// (`definitions`). Its decl_params rows are literals of that script: the
// material ASDSteel1D {E: 2e11, fy: 3.75e8, fu: 4.8e8, eu: 0.2, fracture:
// true}; the section Fiber {GJ: null, patches: [RectPatch{material: $decl
// material, ny: 8, nz: 1}]}; the integration Legendre {section: $decl section,
// n: 2}; the element declaration dispBeamColumn owning 123 elements (its tag
// run) whose chain refs point at the integration. fixtures/shoebuckle.h5 is
// the same model at opensees 2.22.0, before /opensees/decls existed.
//
// The view is rendered against a minimal DOM (below): only what ui/dom.ts and
// the view call. A click on a row or a link must dispatch a store `select`.

import assert from "node:assert/strict";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import type { ModelFile } from "../../src/model/types.ts";
import { materialsOf, view, type MaterialRow, type Val } from "../../src/panels/windows/materials.ts";
import { WINDOWS } from "../../src/panels/windows/registry.ts";
import { openModel } from "../../src/reader/node.ts";
import { BlobStore } from "../../src/state/blobs.ts";
import { loadModel } from "../../src/state/load.ts";
import { initialState, reduce } from "../../src/state/reduce.ts";
import type { Event, State } from "../../src/state/types.ts";

const fixtures = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "fixtures");
const load = (m: ModelFile): State => reduce(initialState, { type: "fileLoaded", artifact: "model", load: loadModel(m, new BlobStore()) });
const model = await openModel(join(fixtures, "definitions.h5"));
const state = load(model);
const M = "opensees/uniaxialMaterial/#1", S = "opensees/section/#1", BI = "opensees/beamIntegration/#1";

// ---- a minimal DOM ------------------------------------------------------------

class FakeNode {
  children: FakeNode[] = [];
  className = "";
  title = "";
  dataset: Record<string, string> = {};
  style: Record<string, string> = { cssText: "" };
  private own: string | null = null;
  private handlers: ((e: { stopPropagation(): void }) => void)[] = [];
  readonly tagName: string;
  constructor(tagName: string) {
    this.tagName = tagName;
  }
  get textContent(): string {
    return this.own ?? this.children.map((c) => c.textContent).join("");
  }
  set textContent(t: string) {
    this.own = t;
    this.children = [];
  }
  append(...xs: (FakeNode | string)[]): void {
    if (this.own !== null) {
      const t = new FakeNode("#text");
      t.own = this.own;
      this.own = null;
      this.children.push(t);
    }
    for (const x of xs) {
      if (typeof x === "string") {
        const t = new FakeNode("#text");
        t.own = x;
        this.children.push(t);
      } else this.children.push(x);
    }
  }
  addEventListener(type: string, fn: (e: { stopPropagation(): void }) => void): void {
    if (type === "click") this.handlers.push(fn);
  }
  /** Fire this node's click listeners; true when one stopped propagation. */
  click(): boolean {
    let stopped = false;
    for (const h of this.handlers) h({ stopPropagation: () => (stopped = true) });
    return stopped;
  }
  all(pred: (n: FakeNode) => boolean): FakeNode[] {
    const out: FakeNode[] = [];
    const walk = (n: FakeNode) => {
      if (pred(n)) out.push(n);
      n.children.forEach(walk);
    };
    walk(this);
    return out;
  }
}
(globalThis as unknown as { document: unknown }).document = { createElement: (tag: string) => new FakeNode(tag.toUpperCase()) };

function render(s: State): { root: FakeNode; events: Event[] } {
  const events: Event[] = [];
  const root = view(s, (e) => events.push(e)) as unknown as FakeNode;
  return { root, events };
}
const rowsOf = (root: FakeNode) => root.all((n) => n.className.split(" ").includes("window-row"));
const linksIn = (n: FakeNode) => n.all((x) => x.className === "window-ref");

// ---- the data -----------------------------------------------------------------

const fields = (r: MaterialRow): [string, Val][] => r.owners[0]!.fields;
const text = (v: Val | undefined): string => (v && v.k === "text" ? v.text : `<${v?.k}>`);

test("the registry's materials entry is this view", () => {
  assert.equal(WINDOWS.materials.view, view);
});

test("every material, section and integration, grouped, by handle with the tag as evidence and the decls key", () => {
  const d = materialsOf(state);
  assert.equal(d.note, null, "a 2.26.0 file needs no window note");
  assert.deepEqual(d.groups.map((g) => [g.kind, g.rows.map((r) => [r.handle, r.type, r.tag, r.declKey])]), [
    ["uniaxialMaterial", [["#1", "ASDSteel1D", 1, M]]],
    ["section", [["#1", "Fiber", 1, S]]],
    ["beamIntegration", [["#1", "Legendre", 1, BI]]],
  ]);
});

test("parameters by field name from decl_params, in the writer's order, formatted from the archive's numbers", () => {
  const [mat] = materialsOf(state).groups[0]!.rows;
  assert.deepEqual(fields(mat!).map(([k, v]) => [k, text(v)]), [["E", "2e11"], ["fy", "3.75e8"], ["fu", "4.8e8"], ["eu", "0.2"], ["fracture", "true"]]);
  assert.equal(mat!.paramsNote, null);
});

test("$decl refs are links to the declaration a click selects: section -> patch material, integration -> section", () => {
  const groups = materialsOf(state).groups;
  const sec = groups[1]!.rows[0]!;
  const [gj, patches] = fields(sec);
  assert.deepEqual(gj, ["GJ", { k: "null" }]);
  assert.equal(patches![0], "patches");
  const list = patches![1];
  assert.ok(list.k === "list" && list.more === 0 && list.items.length === 1);
  const patch = list.items[0]!;
  assert.ok(patch.k === "struct" && patch.cls === "RectPatch");
  assert.deepEqual(patch.fields[0], ["material", { k: "ref", ref: { key: M, label: "uniaxialMaterial #1", target: M, why: null } }]);
  assert.deepEqual(patch.fields.slice(1).map(([k, v]) => [k, text(v)]), [["ny", "8"], ["nz", "1"]]);
  const bi = groups[2]!.rows[0]!;
  assert.deepEqual(fields(bi)[0], ["section", { k: "ref", ref: { key: S, label: "section #1", target: S, why: null } }]);
});

test("used by, from the definition chain's refs: material <- section <- integration <- 123 elements", () => {
  const [mat, sec, bi] = materialsOf(state).groups.map((g) => g.rows[0]!);
  assert.deepEqual(mat!.usedBy, { elements: 0, objects: [{ key: S, label: "section #1", target: S, why: null }] });
  assert.deepEqual(sec!.usedBy, { elements: 0, objects: [{ key: BI, label: "beamIntegration #1", target: BI, why: null }] });
  assert.deepEqual(bi!.usedBy, { elements: 123, objects: [] });
});

test("named in decl_params by: the declarations whose parameters reference it; an element declaration has no store target", () => {
  const [mat, sec, bi] = materialsOf(state).groups.map((g) => g.rows[0]!);
  assert.deepEqual(mat!.referencedBy.map((u) => [u.ref.key, u.type, u.elements]), [[S, "Fiber", 0]]);
  assert.deepEqual(sec!.referencedBy.map((u) => [u.ref.key, u.type, u.elements]), [[BI, "Legendre", 0]]);
  const [el] = bi!.referencedBy;
  assert.deepEqual([el!.ref.key, el!.ref.label, el!.ref.target, el!.type, el!.elements], ["opensees/element/#1", "element #1", null, "dispBeamColumn", 123]);
  assert.match(el!.ref.why!, /not a declaration the app lists/);
});

test("a Hinge integration's three sections are three links", () => {
  const m: ModelFile = structuredClone(model);
  const rec = m.opensees!.definitions.decls!.records.find((r) => r.key === BI)!;
  rec.params[0] = { ...rec.params[0]!, type: "HingeRadau", params: { secI: { $decl: S }, lpI: 0.5, secJ: { $decl: S }, lpJ: 0.5, secE: { $decl: S } } };
  const bi = materialsOf(load(m)).groups[2]!.rows[0]!;
  assert.deepEqual(fields(bi).filter(([, v]) => v.k === "ref").map(([k, v]) => [k, v.k === "ref" ? v.ref.target : null]), [["secI", S], ["secJ", S], ["secE", S]]);
  const { root } = render(load(m));
  const row = rowsOf(root).find((r) => r.dataset["decl"] === BI)!;
  assert.equal(linksIn(row).filter((l) => l.dataset["decl"] === S).length, 3, "one link per section field");
  // The section is named in decl_params by the integration, once (a set of keys per declaration).
  assert.deepEqual(materialsOf(load(m)).groups[1]!.rows[0]!.referencedBy.map((u) => [u.ref.key, u.type]), [[BI, "HingeRadau"]]);
});

test("a long list shows its first eight items and counts the rest", () => {
  const m: ModelFile = structuredClone(model);
  const rec = m.opensees!.definitions.decls!.records.find((r) => r.key === M)!;
  rec.params[0] = { ...rec.params[0]!, params: { strains: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] } };
  const v = fields(materialsOf(load(m)).groups[0]!.rows[0]!)[0]![1];
  assert.ok(v.k === "list");
  assert.deepEqual([v.items.map(text), v.more], [["1", "2", "3", "4", "5", "6", "7", "8"], 3]);
});

// ---- the explicit states --------------------------------------------------------

test("no model, and a model with no material or section, say so", () => {
  assert.deepEqual(materialsOf(initialState), { groups: [], note: "no model is open" });
  const m: ModelFile = structuredClone(model);
  m.opensees!.objects = [];
  assert.deepEqual(materialsOf(load(m)), { groups: [], note: "the model declares no material or section" });
});

test("a 2.22 file (no decls / decl_params) says parameters by field name are not in this file, and never re-reads argv as names", async () => {
  const s = load(await openModel(join(fixtures, "shoebuckle.h5")));
  const d = materialsOf(s);
  assert.match(d.note!, /^opensees 2\.22\.0: this file has no \/opensees\/decls or \/opensees\/decl_params \(written before 2\.25\.0\)/);
  assert.ok(d.groups.length > 0);
  for (const g of d.groups) for (const r of g.rows) {
    assert.deepEqual(r.owners, [], r.path);
    assert.equal(r.declKey, null);
  }
  const { root } = render(s);
  assert.equal(root.all((n) => n.className === "fields").length, 0, "no parameter table without decl_params");
});

test("a declaration with no decl_params row says so on its row", () => {
  const m: ModelFile = structuredClone(model);
  m.opensees!.definitions.decls!.records.find((r) => r.key === M)!.params = [];
  const mat = materialsOf(load(m)).groups[0]!.rows[0]!;
  assert.equal(mat.paramsNote, `/opensees/decl_params has no row for ${M}`);
});

test("a $decl key the decls lack is a broken reader invariant: the window throws, naming it", () => {
  const m: ModelFile = structuredClone(model);
  const rec = m.opensees!.definitions.decls!.records.find((r) => r.key === S)!;
  rec.params[0] = { ...rec.params[0]!, params: { GJ: { $decl: "opensees/uniaxialMaterial/ghost" } } };
  assert.throws(() => materialsOf(load(m)), /\$decl opensees\/uniaxialMaterial\/ghost is not an \/opensees\/decls key/);
});

// ---- the DOM and its events -----------------------------------------------------

test("the view draws a section heading per group, a row per declaration, a field table, the used-by line and the h5 path", () => {
  const { root } = render(state);
  assert.deepEqual(root.all((n) => n.className === "window-section").map((n) => n.textContent), [
    "uniaxial materials (1)",
    "sections (1)",
    "beam integrations (section placement) (1)",
  ]);
  const rows = rowsOf(root);
  assert.deepEqual(rows.map((r) => r.dataset["decl"]), [M, S, BI]);
  const labels = rows[0]!.all((n) => n.className === "f-label").map((n) => n.textContent);
  assert.deepEqual(labels, ["E", "fy", "fu", "eu", "fracture"]);
  assert.match(rows[0]!.textContent, /used by section #1/);
  assert.match(rows[2]!.textContent, /used by 123 elements/);
  assert.match(rows[2]!.textContent, /named in decl_params by element #1 dispBeamColumn \(123 elements\)/);
  assert.equal(rows[0]!.all((n) => n.className === "f-src")[0]!.textContent, "/opensees/materials/uniaxial/ASDSteel1D_1");
});

test("a row click selects its declaration; a link click selects the referenced one and does not bubble to the row", () => {
  const { root, events } = render(state);
  const [, sec] = rowsOf(root);
  sec!.click();
  const link = linksIn(sec!).find((l) => l.dataset["decl"] === M)!;
  assert.equal(link.click(), true, "the link stops propagation");
  assert.deepEqual(events, [
    { type: "select", pick: { decl: S, at: null, nodes: null } },
    { type: "select", pick: { decl: M, at: null, nodes: null } },
  ]);
  assert.deepEqual(reduce(state, events[1]!).selection.decls, [M], "the store selects the referenced declaration");
});

test("a link with no store target is drawn but dispatches nothing", () => {
  const { root, events } = render(state);
  const bi = rowsOf(root)[2]!;
  const el = linksIn(bi).find((l) => l.dataset["decl"] === "opensees/element/#1")!;
  assert.equal(el.click(), false);
  assert.deepEqual(events, []);
  assert.match(el.title, /not a declaration the app lists/);
});
