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
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import type { ModelFile } from "../../src/model/types.ts";
import { materialsOf, view, type MaterialRow, type Val } from "../../src/panels/windows/materials.ts";
import { markSelected, SELECTED_STYLE } from "../../src/panels/windows/frame.ts";
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
  assert.deepEqual(mat!.usedBy, { elements: 0, objects: [{ key: S, label: "section #1", target: S, why: null }], ties: 0 });
  assert.deepEqual(sec!.usedBy, { elements: 0, objects: [{ key: BI, label: "beamIntegration #1", target: BI, why: null }], ties: 0 });
  assert.deepEqual(bi!.usedBy, { elements: 123, objects: [], ties: 0 });
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

test("when the chain follows nothing to a declaration that decl_params names, the row says the two sources disagree (kr3pb's LadrunoQuad case)", () => {
  // Cut the chain's only link to the material (the section's material_ref
  // rows) while decl_params still names it from the section.
  const m: ModelFile = structuredClone(model);
  const sec = m.opensees!.objects.find((o) => o.family === "section")!;
  sec.tables = {};
  const s = load(m);
  const mat = materialsOf(s).groups[0]!.rows[0]!;
  assert.deepEqual(mat.usedBy, { elements: 0, objects: [], ties: 0 });
  assert.deepEqual(mat.referencedBy.map((u) => u.ref.key), [S]);
  const row = rowsOf(render(s).root)[0]!;
  const detail = row.all((n) => n.className === "window-row-detail")[0]!;
  assert.match(detail.textContent, /^the definition chain follows no element or object to it, but decl_params names it \(below\): the two sources disagree/);
  assert.match(detail.textContent, /decl_params names it from section #1 Fiber/);
  assert.doesNotMatch(detail.textContent, /used by nothing/);
  // With no decl_params user either, the plain sentence stays.
  // (a fresh clone: the window's reverse indexes are keyed by the loaded definitions object)
  const m2: ModelFile = structuredClone(m);
  m2.opensees!.definitions.decls!.records.find((r) => r.key === S)!.params[0]!.params = { GJ: null };
  const lone = rowsOf(render(load(m2)).root)[0]!.all((n) => n.className === "window-row-detail")[0]!;
  assert.equal(lone.textContent, "used by nothing in the definition chain");
});

test("a link with no store target is drawn but dispatches nothing", () => {
  const { root, events } = render(state);
  const bi = rowsOf(root)[2]!;
  const el = linksIn(bi).find((l) => l.dataset["decl"] === "opensees/element/#1")!;
  assert.equal(el.click(), false);
  assert.deepEqual(events, []);
  assert.match(el.title, /not a declaration the app lists/);
});

test("V4f: the frame highlights the row of the selected declaration (an unnamed #k one), and only that row", () => {
  const { root, events } = render(state);
  rowsOf(root).find((r) => r.dataset["decl"] === M)!.click();
  const selected = reduce(state, events[0]!);
  assert.deepEqual(selected.selection.decls, [M]);
  // The frame redraws from the store and marks the rows the selection names.
  const again = render(selected).root;
  assert.equal(markSelected(again as unknown as HTMLElement, selected.selection.decls), 1);
  const rows = rowsOf(again);
  const hit = rows.filter((r) => r.className.split(" ").includes("selected"));
  assert.deepEqual(hit.map((r) => r.dataset["decl"]), [M]);
  assert.equal(hit[0]!.style["background"], SELECTED_STYLE.background);
  assert.equal(hit[0]!.style["boxShadow"], SELECTED_STYLE.boxShadow);
  assert.ok(rows.filter((r) => r !== hit[0]).every((r) => !r.className.includes("selected") && r.style["background"] === undefined));
  // Nothing selected: nothing marked.
  assert.equal(markSelected(render(state).root as unknown as HTMLElement, []), 0);
});

// ---- V4m (#1846): embeddedRebar tie users -------------------------------------
//
// Oracle: fixtures/reinforced.h5, written by main's apeGmsh through
// scripts/make_viewer_results_fixtures.py --reinforced (opensees 2.31.0), and
// fixtures/reinforced.oracle.json beside it: Python's count of ties per bond
// tag from apeGmsh's own reader (h5_reader.embedded_rebars), not the app's
// parse. bond1 is named only by ties; bond2 by ties and by its own CorotTruss
// cells; the third bar is tied by perfect bond (no material).

const reinforced = await openModel(join(fixtures, "reinforced.h5"));
const oracle = JSON.parse(readFileSync(join(fixtures, "reinforced.oracle.json"), "utf-8")) as { ties: number; unbonded: number; bond: Record<string, number> };
const B1 = "opensees/uniaxialMaterial/bond1", B2 = "opensees/uniaxialMaterial/bond2";
const rowOf = (s: State, path: string): MaterialRow => materialsOf(s).groups.flatMap((g) => g.rows).find((r) => r.path === path)!;
const detailOf = (s: State, path: string): string =>
  rowsOf(render(s).root).find((r) => r.dataset["decl"] === path)!.all((n) => n.className === "window-row-detail").map((n) => n.textContent).join(" | ");
/** reinforced.h5 with its tie summary changed (a fresh clone: the window's indexes are keyed by the loaded definitions) */
const withTies = (edit: (t: NonNullable<ModelFile["opensees"]>["definitions"]["embeddedRebar"] & object) => void): State => {
  const m: ModelFile = structuredClone(reinforced);
  edit(m.opensees!.definitions.embeddedRebar!);
  return load(m);
};

test("V4m: per bond material, the app's tie count equals Python's count over the same group (reinforced.h5, 2.31)", () => {
  const s = load(reinforced);
  assert.equal(reinforced.opensees!.version, "2.31.0");
  assert.deepEqual(reinforced.warnings, []);
  const er = reinforced.opensees!.definitions.embeddedRebar!;
  assert.equal(er.count, oracle.ties);
  assert.equal(er.unbonded, oracle.unbonded);
  assert.deepEqual(er.unresolved, []);
  const uni = materialsOf(s).groups.find((g) => g.kind === "uniaxialMaterial")!.rows;
  // joined by the tag each row carries, never by position
  const app = Object.fromEntries(uni.filter((r) => r.usedBy.ties > 0).map((r) => [String(r.tag), r.usedBy.ties]));
  assert.deepEqual(app, oracle.bond);
  assert.ok(uni.every((r) => r.tiesUnknown === null));
});

test("V4m: a material used only by ties says 'used by N embeddedRebar ties', never 'used by nothing'", () => {
  const s = load(reinforced);
  const r = rowOf(s, B1);
  assert.deepEqual(r.usedBy, { elements: 0, objects: [], ties: oracle.bond[String(r.tag)] });
  const text = detailOf(s, B1);
  assert.equal(text, `used by ${r.usedBy.ties} embeddedRebar ties`);
  assert.doesNotMatch(text, /nothing/);
});

test("V4m: a material used by elements and ties shows both counts", () => {
  const s = load(reinforced);
  const r = rowOf(s, B2);
  assert.ok(r.usedBy.elements > 0);
  assert.equal(r.usedBy.ties, oracle.bond[String(r.tag)]);
  assert.equal(detailOf(s, B2), `used by ${r.usedBy.elements} element${r.usedBy.elements === 1 ? "" : "s"} · ${r.usedBy.ties} embeddedRebar ties`);
  // an nD material is never a bond law (the grammar's -bond reads a uniaxial tag)
  const nd = materialsOf(s).groups.find((g) => g.kind === "nDMaterial")!.rows[0]!;
  assert.equal(nd.usedBy.ties, 0);
  assert.equal(nd.tiesUnknown, null);
});

test("V4m: ties whose bond tag the reader could not read make every uniaxial row say 'users unknown', never 0 or 'used by nothing'", () => {
  const s = withTies((t) => {
    t.bond = [];
    t.unresolved = [{ why: "the value after -bond is not an integer tag", ties: 10 }];
  });
  const r = rowOf(s, B1);
  assert.deepEqual(r.usedBy, { elements: 0, objects: [], ties: 0 });
  assert.deepEqual(r.tiesUnknown, [{ why: "the value after -bond is not an integer tag", ties: 10 }]);
  const text = detailOf(s, B1);
  assert.match(text, /^users unknown: referenced by embeddedRebar ties10 ties: the value after -bond is not an integer tag$/);
  assert.doesNotMatch(text, /nothing|0/);
  // a row with known users keeps them and says the rest are unknown
  const both = detailOf(s, B2);
  assert.match(both, /^used by \d+ elementsusers unknown beyond these: referenced by embeddedRebar ties; 10 ties: /);
  // an unknown count is drawn as unknown, not as a number
  const u = withTies((t) => {
    t.bond = [];
    t.unresolved = [{ why: "/opensees/constraints/embeddedRebar has no arg_kinds", ties: null }];
  });
  assert.match(detailOf(u, B1), /users unknown: referenced by embeddedRebar tiesunknown number of ties: .*has no arg_kinds/);
});

test("V4m: a bond tag that names no uniaxial material of the file is 'users unknown', not dropped", () => {
  const s = withTies((t) => {
    t.bond = [{ tag: 99, ties: 4 }];
  });
  const r = rowOf(s, B1);
  assert.equal(r.usedBy.ties, 0);
  assert.deepEqual(r.tiesUnknown, [{ why: "bond tag 99 is no uniaxial material of this file", ties: 4 }]);
  assert.match(detailOf(s, B1), /^users unknown: referenced by embeddedRebar ties/);
});

test("V4m: ties with no -bond (perfect bond) name no material and leave the rows unchanged", () => {
  const s = withTies((t) => {
    t.bond = [];
    t.unbonded = 30;
  });
  assert.equal(rowOf(s, B1).tiesUnknown, null);
  assert.equal(detailOf(s, B1), "used by nothing in the definition chain");
});

test("V4m: a file without the group is unchanged: no tie count, no unknown", () => {
  assert.equal(model.opensees!.definitions.embeddedRebar, null);
  const mat = materialsOf(state).groups[0]!.rows[0]!;
  assert.equal(mat.usedBy.ties, 0);
  assert.equal(mat.tiesUnknown, null);
  assert.doesNotMatch(rowsOf(render(state).root)[0]!.textContent, /embeddedRebar|users unknown/);
});

test("V4m: reinforced_unknown.h5 (one tie's -bond tag stored as a float) reads that tie as unknown, and every uniaxial row says so", async () => {
  const m = await openModel(join(fixtures, "reinforced_unknown.h5"));
  const er = m.opensees!.definitions.embeddedRebar!;
  const bonded = Object.values(oracle.bond).reduce((a, b) => a + b, 0);
  assert.equal(er.bond.reduce((a, b) => a + b.ties, 0), bonded - 1);
  assert.deepEqual(er.unresolved, [{ why: "the value after -bond is not an integer tag", ties: 1 }]);
  const s = load(m);
  for (const p of [B1, B2]) {
    assert.deepEqual(rowOf(s, p).tiesUnknown, [{ why: "the value after -bond is not an integer tag", ties: 1 }]);
    assert.match(detailOf(s, p), /users unknown beyond these: referenced by embeddedRebar ties; 1 tie: the value after -bond is not an integer tag/);
  }
});

/** reinforced.h5 loaded, with one uniaxial declaration's tag replaced (a new decls object: the window's indexes are keyed by it) */
const withTag = (path: string, tag: number | null): State => {
  const s = load(reinforced);
  return { ...s, decls: { ...s.decls, [path]: { ...s.decls[path]!, tag } } };
};

test("V4m: a bond tag that two uniaxial materials carry is 'users unknown' and gives the ties to neither", () => {
  const t1 = rowOf(load(reinforced), B1).tag!, t2 = rowOf(load(reinforced), B2).tag!;
  const s = withTag(B2, t1);
  const n = oracle.bond[String(t1)]!;
  for (const p of [B1, B2]) {
    const r = rowOf(s, p);
    assert.equal(r.usedBy.ties, 0, p);
    // (bond2's own tag is now no uniaxial material's, so its ties are unknown too)
    assert.deepEqual(r.tiesUnknown, [
      { why: `bond tag ${t1} names 2 uniaxial materials`, ties: n },
      { why: `bond tag ${t2} is no uniaxial material of this file`, ties: oracle.bond[String(t2)] },
    ], p);
  }
  assert.match(detailOf(s, B1), /^users unknown: referenced by embeddedRebar ties/);
});

test("V4m: a uniaxial material with no tag, in a file with bonded ties, is 'users unknown', not 'used by nothing'", () => {
  const s = withTag(B1, null);
  const r = rowOf(s, B1);
  assert.equal(r.usedBy.ties, 0);
  const t1 = rowOf(load(reinforced), B1).tag!;
  assert.deepEqual(r.tiesUnknown, [
    { why: `bond tag ${t1} is no uniaxial material of this file`, ties: oracle.bond[String(t1)] },
    { why: "this material's tag is not in the file, so no tie's bond tag can be matched to it", ties: null },
  ]);
  assert.match(detailOf(s, B1), /^users unknown: referenced by embeddedRebar ties/);
  assert.doesNotMatch(detailOf(s, B1), /used by nothing/);
});
