// Spotlight (`Ctrl+K` / `Cmd+K`), the pure Search Engine (`state/search.ts`),
// the Search window (`panels/windows/search.ts`), and the exhaustive EventType
// ledger (`SPOTLIGHT_EVENT_LEDGER`).

import assert from "node:assert/strict";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { openModel } from "../src/reader/node.ts";
import { focusOf } from "../src/renderer/selection.ts";
import { elementMaskOf, visibleMaps } from "../src/renderer/visible.ts";
import { BlobStore } from "../src/state/blobs.ts";
import { loadModel } from "../src/state/load.ts";
import {
  clauseLabel,
  elementsOfDecl,
  hiddenElementsOf,
  indexOf,
  inspectedBatchOf,
  inspectedGroupsOf,
  parseSearchText,
  runSearchTerms,
  searchResultsOf,
  selectedElementsOf,
  SPOTLIGHT_EVENT_LEDGER,
  spotlightViewOf,
  windowContentOf,
} from "../src/state/selectors.ts";
import { initialState, reduce, Store, type State } from "../src/state/store.ts";
import type { EventType } from "../src/state/types.ts";

const fixtures = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures");
const shoeModel = await openModel(join(fixtures, "shoebuckle.h5"));
const shoeBlobs = new BlobStore();
const shoeLoad = loadModel(shoeModel, shoeBlobs);
const loaded: State = reduce(initialState, { type: "fileLoaded", artifact: "model", load: shoeLoad });

const defModel = await openModel(join(fixtures, "definitions.h5"));
const defBlobs = new BlobStore();
const defLoad = loadModel(defModel, defBlobs);
const defLoaded: State = reduce(initialState, { type: "fileLoaded", artifact: "model", load: defLoad });

// ---- 1. Exhaustive Event Ledger (`SPOTLIGHT_EVENT_LEDGER`) ------------------

test("SPOTLIGHT_EVENT_LEDGER classifies every EventType with a non-empty command/description or reason", () => {
  const entries = Object.entries(SPOTLIGHT_EVENT_LEDGER) as [EventType, (typeof SPOTLIGHT_EVENT_LEDGER)[EventType]][];
  assert.equal(entries.length, 75, "all 75 store events are classified in SPOTLIGHT_EVENT_LEDGER");
  const exposedCommands = new Set<string>();
  for (const [eventType, entry] of entries) {
    if (entry.kind === "expose") {
      assert.ok(entry.command.trim().length > 0, `${eventType}: expose entry must name a command`);
      assert.ok(entry.description.trim().length > 0, `${eventType}: expose entry must have a description`);
      exposedCommands.add(entry.command);
    } else {
      assert.ok(entry.reason.trim().length > 0, `${eventType}: ${entry.kind} entry must state a reason`);
    }
  }
  assert.ok(exposedCommands.size >= 30, "at least 30 distinct user commands are exposed");
});

// ---- 2. Pure Search Engine (`src/state/search.ts`) --------------------------

test("indexOf caches ModelIndex per `s.decls` via WeakMap and rebuilds only when `s.decls` changes", () => {
  const idx1 = indexOf(loaded);
  const idx2 = indexOf(loaded);
  assert.equal(idx1, idx2, "WeakMap returns the exact same ModelIndex instance for the same s.decls");
  const toggled = reduce(loaded, { type: "setEdges", on: false });
  assert.equal(indexOf(toggled), idx1, "non-decls state transitions reuse the cached ModelIndex");
  assert.ok(idx1.types.includes("dispBeamColumn"));
  assert.ok(idx1.types.includes("line2"));
  assert.ok(idx1.facets.groups.includes("Arch"));
  assert.ok(idx1.facets.groups.includes("LeftColumn"));
  assert.ok(idx1.facets.groups.includes("RightColumn"));
});

test("parseSearchText and runSearchTerms evaluate boolean AND / &, OR / |, NOT / ! and facets", () => {
  // 1. Physical group query: matches group + label + 79 elements
  const archRes = runSearchTerms(loaded, parseSearchText("group:Arch"));
  assert.equal(archRes.elementCount, 79, "Arch has 79 elements");
  assert.ok(archRes.matched.some((d) => d.path === "mesh/physical_group/Arch"));

  // 2. Boolean AND (`and` and `&`): beams in Arch
  const archBeamsAnd = runSearchTerms(loaded, parseSearchText("group:Arch and type:dispBeamColumn and kind:element"));
  const archBeamsAmp = runSearchTerms(loaded, parseSearchText("group:Arch & type:dispBeamColumn & kind:element"));
  assert.equal(archBeamsAnd.total, 79);
  assert.equal(archBeamsAmp.total, 79);

  // 3. Boolean OR (`or` and `|`): LeftColumn or RightColumn elements
  const colsOr = runSearchTerms(loaded, parseSearchText("kind:element and group:LeftColumn or kind:element and group:RightColumn"));
  assert.equal(colsOr.elementCount, 44, "22 LeftColumn + 22 RightColumn = 44 elements");
  const colsPipe = runSearchTerms(loaded, parseSearchText("group:LeftColumn | group:RightColumn"));
  assert.equal(colsPipe.elementCount, 44);

  // 4. Boolean NOT (`not`, `!`, `-`): elements not in Arch
  const notArch = runSearchTerms(loaded, parseSearchText("kind:element and not group:Arch"));
  assert.equal(notArch.total, 44, "123 total elements - 79 Arch elements = 44 column elements");
  const bangArch = runSearchTerms(loaded, parseSearchText("kind:element !group:Arch"));
  assert.equal(bangArch.total, 44);

  // 5. Transitive `uses:` (elements using the model's uniaxialMaterial, Fiber section, or beamIntegration)
  const matDecl = Object.values(loaded.decls).find((d) => d.kind === "uniaxialMaterial")!;
  const usesMat = runSearchTerms(loaded, parseSearchText(`kind:element and uses:${matDecl.type}`));
  assert.equal(usesMat.elementCount, 123, `every beam element transitively uses ${matDecl.type} through its Fiber section`);
  const intDecl = Object.values(loaded.decls).find((d) => d.kind === "beamIntegration")!;
  const usesInt = runSearchTerms(loaded, parseSearchText(`kind:element and uses:${intDecl.type}`));
  assert.equal(usesInt.elementCount, 123);

  // 6. ID ranges (`1..10`) and `@tag`
  const idRange = runSearchTerms(loaded, parseSearchText("kind:element and id:1..10"));
  assert.equal(idRange.total, 10);

  // 7. Parameter numeric comparisons (`E>1e10`, `E<1e5`) on definitions.h5
  const highE = runSearchTerms(defLoaded, parseSearchText("E>1e8"));
  assert.ok(highE.total >= 1, "matches material/elements with E > 1e8");
  const impossibleE = runSearchTerms(defLoaded, parseSearchText("E>1e30"));
  assert.equal(impossibleE.total, 0);

  // 8. Visibility facet (`visibility:hidden` vs `visibility:visible`)
  const withHidden = reduce(loaded, { type: "setHidden", decls: ["mesh/physical_group/Arch"], hidden: true });
  const hiddenOnly = runSearchTerms(withHidden, parseSearchText("kind:element and visibility:hidden"));
  assert.equal(hiddenOnly.total, 79);
  const visibleOnly = runSearchTerms(withHidden, parseSearchText("kind:element and visibility:visible"));
  assert.equal(visibleOnly.total, 44);
});

test("elementsOfDecl, selectedElementsOf, focusOf, and elementMaskOf work across groups, materials, and element batches", () => {
  // Selecting a physical group expands to its 79 elements for 3D highlight and camera framing
  const sGroup = reduce(loaded, { type: "selectDecls", decls: ["mesh/physical_group/Arch"] });
  assert.equal(selectedElementsOf(sGroup).length, 79);
  const focusGroup = focusOf(sGroup, shoeBlobs);
  assert.ok(focusGroup !== null && focusGroup.radius > 0, "focusOf computes bounding sphere for a selected physical group");

  // Selecting a uniaxialMaterial expands to all elements transitively using it
  const matPath = Object.keys(loaded.decls).find((p) => p.startsWith("opensees/uniaxialMaterial/"))!;
  assert.equal(elementsOfDecl(loaded, matPath).length, 123);

  // Isolating a physical group (Arch) when a superset physical group (Frame) exists
  const isolatedArchGroup = reduce(loaded, { type: "isolate", decls: ["mesh/physical_group/Arch"] });
  const hiddenByGroup = hiddenElementsOf(isolatedArchGroup);
  assert.equal(hiddenByGroup.size, 44, "isolating Arch hides only the 44 column elements, not Arch's 79 elements despite Frame being hidden");
  assert.equal(
    elementMaskOf(isolatedArchGroup, isolatedArchGroup.mesh!, shoeBlobs),
    null,
    "elementMaskOf is null when only physical_group declarations are hidden",
  );
  const maps = visibleMaps(
    isolatedArchGroup.mesh!.legend,
    isolatedArchGroup.visibility.hidden,
    shoeBlobs.i32(isolatedArchGroup.mesh!.lineGroup),
    shoeBlobs.i32(isolatedArchGroup.mesh!.triGroup),
    elementMaskOf(isolatedArchGroup, isolatedArchGroup.mesh!, shoeBlobs),
  );
  assert.equal(maps.lineMap.length, 79, "all 79 Arch line segments remain visible after isolating Arch");

  // Isolating a subset of elements (e.g. first 5 elements of Arch) hides all other elements in elementMaskOf
  const archElems = elementsOfDecl(loaded, "mesh/physical_group/Arch").slice(0, 5);
  const isolated = reduce(loaded, { type: "isolate", decls: [...archElems] });
  const hiddenSet = hiddenElementsOf(isolated);
  assert.equal(hiddenSet.size, 123 - 5, "all non-isolated elements are in hiddenElementsOf");
  const mask = elementMaskOf(isolated, isolated.mesh!, shoeBlobs);
  assert.ok(mask !== null, "elementMaskOf returns an active mask when individual elements are hidden");
  const visibleCount = mask.mask.reduce((acc, v) => acc + (v === 0 ? 1 : 0), 0);
  assert.equal(visibleCount, 5, "only the 5 isolated elements remain visible in elementMaskOf");
});

test("inspector selectors: inspectedGroupsOf and inspectedBatchOf summarize group and batch selections", () => {
  const sGroup = reduce(loaded, { type: "selectDecls", decls: ["mesh/physical_group/Arch"] });
  const groups = inspectedGroupsOf(sGroup);
  assert.equal(groups.length, 1);
  assert.equal(groups[0]!.name, "Arch");
  assert.equal(groups[0]!.count, 79);

  // Selecting 10 elements triggers inspectedBatchOf (> MAX_INSPECTED_CHAINS = 4)
  const tenElems = elementsOfDecl(loaded, "mesh/physical_group/Arch").slice(0, 10);
  const sBatch = reduce(loaded, { type: "selectDecls", decls: tenElems });
  const batch = inspectedBatchOf(sBatch);
  assert.ok(batch !== null);
  assert.equal(batch.count, 10);
  assert.equal(batch.elementCount, 10);
});

// ---- 3. Spotlight Selector & Drill-Down (`src/state/spotlight.ts`) ----------

test("spotlightViewOf: root commands, boolean queries, batch actions, drill-down modes, and Search window handoff", () => {
  // 1. Open Spotlight at root
  let s = reduce(loaded, { type: "openSpotlight" });
  assert.equal(s.spotlight.open, true);
  let view = spotlightViewOf(s);
  assert.equal(view.mode, "root");
  assert.ok(view.items.some((i) => i.id === "cmd:open-search-window"));
  assert.ok(view.items.some((i) => i.id === "cmd:set-opacity"));

  // 2. Query "set opacity" ranks the Set Surface Opacity drill-down command first
  s = reduce(s, { type: "setSpotlightQuery", query: "set opacity" });
  view = spotlightViewOf(s);
  assert.equal(view.items[0]!.id, "cmd:set-opacity");
  assert.deepEqual(view.items[0]!.events, [{ type: "setSpotlightMode", mode: "opacity" }]);

  // 3. Drill into "opacity" mode and type a custom percentage "35%"
  s = reduce(s, { type: "setSpotlightMode", mode: "opacity" });
  s = reduce(s, { type: "setSpotlightQuery", query: "35%" });
  view = spotlightViewOf(s);
  assert.equal(view.mode, "opacity");
  assert.equal(view.items[0]!.id, "val:opacity-custom");
  const afterOpacity = reduce(s, view.items[0]!.events[0]!);
  assert.ok(Math.abs(afterOpacity.visibility.opacity - 0.35) < 1e-6);

  // 4. Boolean query in Spotlight ("group:Arch and dispBeamColumn") surfaces batch select & batch actions
  s = reduce(loaded, { type: "openSpotlight", mode: "root", query: "group:Arch and dispBeamColumn" });
  view = spotlightViewOf(s);
  assert.equal(view.items[0]!.id, "batch:select-matched");
  assert.match(view.items[0]!.title, /Select 79 Matching Elements/);
  assert.deepEqual(view.items[0]!.shiftEvents, [
    { type: "addSpotlightClause", clause: { join: "and", negate: false, field: "group", op: ":", value: "Arch" } },
    { type: "addSpotlightClause", clause: { join: "and", negate: false, field: "any", op: ":", value: "dispBeamColumn" } },
    { type: "setSpotlightMode", mode: "selectionActions" },
  ]);

  // 5. Shift+Enter drills into `selectionActions` (Select & Frame, Add, Hide, Show, Isolate) and preserves query clauses
  for (const ev of view.items[0]!.shiftEvents!) s = reduce(s, ev);
  const actionView = spotlightViewOf(s);
  assert.deepEqual(
    actionView.items.map((i) => i.id),
    ["act:select", "act:select-add", "act:hide", "act:show", "act:isolate"],
  );
  // Typing "hide" or "turn off" in selectionActions filters to act:hide while keeping the 79 matched elements
  const hideView = spotlightViewOf(reduce(s, { type: "setSpotlightQuery", query: "hide" }));
  assert.equal(hideView.items[0]!.id, "act:hide");
  assert.match(hideView.items[0]!.title, /79 matched elements/);
  const turnOffView = spotlightViewOf(reduce(s, { type: "setSpotlightQuery", query: "turn off" }));
  assert.equal(turnOffView.items[0]!.id, "act:hide");
  const isolateEvent = actionView.items.find((i) => i.id === "act:isolate")!.events[0]!;
  const afterIsolate = reduce(s, isolateEvent);
  assert.equal(hiddenElementsOf(afterIsolate).size, 44, "isolating the 79 Arch beams hides the 44 column elements");

  // 6. Filter chips in Spotlight (`addSpotlightClause`, `removeSpotlightClause`) and handoff to Search window
  s = reduce(loaded, { type: "openSpotlight" });
  s = reduce(s, { type: "addSpotlightClause", clause: { join: "and", negate: false, field: "group", op: ":", value: "Arch" } });
  s = reduce(s, { type: "setSpotlightQuery", query: "dispBeamColumn" });
  view = spotlightViewOf(s);
  assert.equal(view.clauses.length, 1);
  assert.equal(clauseLabel(view.clauses[0]!), "group:Arch");
  const openSearchItem = view.items.find((i) => i.id === "batch:open-in-search")!;
  for (const ev of openSearchItem.events) s = reduce(s, ev);
  assert.ok(s.windows.open.includes("search"), "Search window is opened");
  assert.equal(s.search.text, "dispBeamColumn", "Spotlight query transferred to Search window");
  assert.equal(s.search.clauses.length, 1, "Spotlight filter chip transferred to Search window");
  assert.equal(searchResultsOf(s).elementCount, 79);
});

test("windowContentOf supports the 'search' window kind", () => {
  const s = reduce(loaded, { type: "setSearchText", text: "group:Arch" });
  const content = windowContentOf(s, "search");
  assert.ok(content.rows.length > 0);
  assert.match(content.note ?? "", /79 elements/);
});

// ---- 4. DOM Integration (`mountSpotlight`, `mountWindows`, `mountInspector`) -

class FakeEl {
  children: FakeEl[] = [];
  parent: FakeEl | null = null;
  className = "";
  id = "";
  title = "";
  hidden = false;
  disabled = false;
  type = "";
  value = "";
  placeholder = "";
  autocomplete = "";
  spellcheck = false;
  checked = false;
  selectionStart: number | null = null;
  selectionEnd: number | null = null;
  dataset: Record<string, string> = {};
  style: Record<string, string> = { cssText: "" };
  attrs: Record<string, string> = {};
  scrollTop = 0;
  scrollLeft = 0;
  private text = "";
  private listeners = new Map<string, ((e?: unknown) => void)[]>();
  readonly tagName: string;
  constructor(tagName: string) {
    this.tagName = tagName;
  }
  get textContent(): string {
    return this.text + this.children.map((c) => c.textContent).join("");
  }
  set textContent(t: string) {
    this.text = t;
    this.children = [];
  }
  classList = {
    toggle: (c: string, on: boolean) => {
      const set = new Set(this.className.split(" ").filter(Boolean));
      if (on) set.add(c);
      else set.delete(c);
      this.className = [...set].join(" ");
    },
  };
  setAttribute(k: string, v: string): void {
    this.attrs[k] = v;
  }
  append(...xs: (FakeEl | string)[]): void {
    for (const x of xs) {
      let n: FakeEl;
      if (typeof x === "string") {
        n = new FakeEl("#text");
        n.text = x;
      } else n = x;
      n.parent = this;
      this.children.push(n);
    }
  }
  replaceChildren(...xs: FakeEl[]): void {
    this.children = [];
    this.scrollTop = 0;
    this.append(...xs);
  }
  remove(): void {
    if (this.parent) this.parent.children = this.parent.children.filter((c) => c !== this);
  }
  addEventListener(type: string, fn: (e?: unknown) => void): void {
    const list = this.listeners.get(type) ?? [];
    list.push(fn);
    this.listeners.set(type, list);
  }
  removeEventListener(type: string, fn: (e?: unknown) => void): void {
    const list = this.listeners.get(type) ?? [];
    this.listeners.set(
      type,
      list.filter((f) => f !== fn),
    );
  }
  dispatch(type: string, ev: unknown = {}): void {
    for (const f of [...(this.listeners.get(type) ?? [])]) f(ev);
  }
  click(ev: unknown = {}): void {
    this.dispatch("click", ev);
  }
  focus(): void {
    fakeDoc.activeElement = this;
  }
  select(): void {}
  setSelectionRange(a: number, b: number): void {
    this.selectionStart = a;
    this.selectionEnd = b;
  }
  scrollIntoView(): void {}
  has(cls: string): boolean {
    return this.className.split(" ").includes(cls);
  }
  all(pred: (e: FakeEl) => boolean): FakeEl[] {
    const out: FakeEl[] = [];
    for (const c of this.children) {
      if (pred(c)) out.push(c);
      out.push(...c.all(pred));
    }
    return out;
  }
  closest(sel: string): FakeEl | null {
    if (sel === "[data-window]") {
      for (let e: FakeEl | null = this; e; e = e.parent) if (e.dataset["window"] !== undefined) return e;
      return null;
    }
    if (sel === "[data-focus-key]") {
      for (let e: FakeEl | null = this; e; e = e.parent) if (e.dataset["focusKey"] !== undefined) return e;
      return null;
    }
    return null;
  }
  querySelectorAll(sel: string): FakeEl[] {
    if (sel === "[data-window] .stage-timeline") return this.all((e) => e.has("stage-timeline") && e.parent?.closest("[data-window]") !== null);
    return [];
  }
  querySelector(sel: string): FakeEl | null {
    if (sel === ".stage-timeline") return this.all((e) => e.has("stage-timeline"))[0] ?? null;
    const m = /^\[data-focus-key="(.+)"\]$/.exec(sel);
    if (m) return this.all((e) => e.dataset["focusKey"] === m[1]!)[0] ?? null;
    return null;
  }
}

const fakeBody = new FakeEl("BODY");
const fakeInspector = new FakeEl("ASIDE");
fakeInspector.id = "inspector";
fakeBody.append(fakeInspector);
const fakeDoc = {
  body: fakeBody,
  activeElement: null as FakeEl | null,
  createElement: (t: string) => new FakeEl(t.toUpperCase()),
  getElementById: (id: string) => (fakeBody.id === id ? fakeBody : fakeBody.all((e) => e.id === id)[0] ?? null),
};
const fakeWin = new FakeEl("WINDOW");

test("DOM: mountSpotlight handles Ctrl+K, query input, Tab filter chips, Shift+Enter actions, and Enter execution", async () => {
  const prevDoc = (globalThis as { document?: unknown }).document;
  const prevWin = (globalThis as { window?: unknown }).window;
  (globalThis as { document?: unknown }).document = fakeDoc;
  (globalThis as { window?: unknown }).window = fakeWin;
  try {
    const { mountSpotlight } = await import("../src/panels/spotlight.ts");
    const { mountWindows } = await import("../src/panels/windows/frame.ts");
    const { mountInspector } = await import("../src/panels/inspector.ts");

    const store = new Store();
    store.dispatch({ type: "fileLoaded", artifact: "model", load: shoeLoad });

    const unSpot = mountSpotlight(store);
    const unWin = mountWindows(store);
    const unInsp = mountInspector(store);
    try {
      const spotRoot = fakeDoc.getElementById("spotlight")!;
      assert.equal(spotRoot.hidden, true, "Spotlight starts hidden");

      // 1. Press Ctrl+K on window to open Spotlight
      let prevented = false;
      fakeWin.dispatch("keydown", {
        key: "k",
        ctrlKey: true,
        metaKey: false,
        altKey: false,
        shiftKey: false,
        preventDefault: () => {
          prevented = true;
        },
      });
      assert.equal(prevented, true);
      assert.equal(store.get().spotlight.open, true);
      assert.equal(spotRoot.hidden, false);

      // 2. Type "group:Arch and dispBeamColumn" into the Spotlight input
      const input = spotRoot.all((e) => e.has("spotlight-input"))[0]!;
      input.value = "group:Arch and dispBeamColumn";
      input.dispatch("input");
      assert.equal(store.get().spotlight.query, "group:Arch and dispBeamColumn");
      const rows = spotRoot.all((e) => e.has("spotlight-row"));
      assert.ok(rows.length >= 2);
      assert.equal(rows[0]!.dataset["itemId"], "batch:select-matched");

      // 3. Press Enter on the first item (`batch:select-matched`) -> selects 79 elements and closes Spotlight
      fakeWin.dispatch("keydown", {
        key: "Enter",
        ctrlKey: false,
        metaKey: false,
        altKey: false,
        shiftKey: false,
        preventDefault: () => {},
      });
      assert.equal(store.get().spotlight.open, false);
      assert.equal(store.get().selection.decls.length, 79, "all 79 Arch beam elements are selected");
      assert.equal(fakeInspector.hidden, false, "Inspector shows the batch selection card");
      assert.match(fakeInspector.textContent, /Batch selection/);
      assert.match(fakeInspector.textContent, /79 selected/);

      // 4. Open Search window via mountWindows and verify query & batch buttons
      store.dispatch({ type: "openWindow", window: "search" });
      store.dispatch({ type: "setSearchText", text: "group:LeftColumn" });
      const winHost = fakeDoc.getElementById("windows")!;
      assert.match(winHost.textContent, /22 elements/, "Search window shows 22 elements for LeftColumn");
    } finally {
      unInsp();
      unWin();
      unSpot();
    }
  } finally {
    (globalThis as { document?: unknown }).document = prevDoc;
    (globalThis as { window?: unknown }).window = prevWin;
  }
});

test("spotlightViewOf exposes contour, unit, deform, and step drill-downs when paired results are on the phase axis", () => {
  const lineage = { modelHash: "a1b2c3d4e5f60000", femHash: "f6e5d4c3b2a10000", resultsHash: "1122334455667788" };
  let s = reduce(initialState, { type: "fileLoaded", artifact: "model", load: { ...shoeLoad, lineage } });
  s = reduce(s, { type: "fileOpened", artifact: "results", path: "shoe.results.h5" });
  s = reduce(s, {
    type: "resultsLoaded",
    info: {
      path: "shoe.results.h5",
      version: "1.1.0",
      sourceType: "domain_capture",
      lineage,
      partial: null,
      stages: [
        {
          id: "stage_000",
          name: "gravity",
          kind: "static",
          time: [0, 0.5, 1.0],
          nodal: ["displacement_x", "displacement_y", "displacement_z"],
          refusedNodal: [],
          gauss: [],
          unread: [],
        },
      ],
      unavailableZones: [],
      openseesVersion: null,
      programDecl: null,
    },
    sizeBytes: 1024,
    readMs: 2,
    warnings: [],
  });
  s = reduce(s, { type: "setPhase", at: { kind: "results", stage: 0, step: 0 } });
  s = reduce(s, { type: "openSpotlight" });

  // Root view with paired results includes contourUnits, contourField, step, and deformScale commands
  const rootView = spotlightViewOf(reduce(s, { type: "setSpotlightQuery", query: "contour units" }));
  assert.ok(rootView.items.some((i) => i.id === "cmd:contour-units"));

  // Drill into contourUnits and verify target/source unit items dispatch setContourUnits
  const unitsState = reduce(s, { type: "setSpotlightMode", mode: "contourUnits" });
  const unitsView = spotlightViewOf(unitsState);
  const tfItem = unitsView.items.find((i) => i.id === "val:unit-target-tf_m_s");
  assert.ok(tfItem, "contourUnits drill-down lists tf_m_s display unit option");
  for (const ev of tfItem.action.events) s = reduce(s, ev);
  assert.equal(s.contour.targetUnits, "tf_m_s");
});

test("English natural-language search, 'search' command ranking, 'set' options, and Tab / Shift+Enter workflows", async () => {
  // 1. Conversational English query: "select beams and elements that are controlled by displacement"
  const nlTerms = parseSearchText("select beams and elements that are controlled by displacement");
  assert.deepEqual(
    nlTerms.map((t) => ({ join: t.join, negate: t.negate, value: t.value })),
    [
      { join: "and", negate: false, value: "beams" },
      { join: "and", negate: false, value: "elements" },
      { join: "and", negate: false, value: "displacement" },
    ],
  );
  const nlRes = runSearchTerms(loaded, nlTerms);
  assert.equal(nlRes.total, 123, "matches all 123 dispBeamColumn beam elements in shoebuckle.h5");
  assert.equal(nlRes.elementCount, 123);

  // 2. English synonyms & field aliases ("select columns", "elements that are not in Arch", "section:Fiber")
  const colRes = runSearchTerms(loaded, parseSearchText("select columns"));
  assert.equal(colRes.elementCount, 44, "22 LeftColumn + 22 RightColumn = 44 column elements");
  const notArchEn = runSearchTerms(loaded, parseSearchText("elements that are not in Arch"));
  assert.equal(notArchEn.total, 44, "123 - 79 = 44 elements not in Arch");
  const secAlias = runSearchTerms(loaded, parseSearchText("section:Fiber"));
  assert.equal(secAlias.elementCount, 123);

  // 3. Typing "search" or "search Arch" ranks cmd:open-search-window at #1
  let s = reduce(loaded, { type: "openSpotlight", query: "search" });
  let view = spotlightViewOf(s);
  assert.equal(view.items[0]!.id, "cmd:open-search-window");

  s = reduce(loaded, { type: "openSpotlight", query: "search Arch" });
  view = spotlightViewOf(s);
  assert.equal(view.items[0]!.id, "cmd:open-search-window");
  for (const ev of view.items[0]!.events) s = reduce(s, ev);
  assert.ok(s.windows.open.includes("search"));
  assert.equal(s.search.text, "Arch");

  // 4. Typing "set" shows viewer property commands (opacity, colourBy, edges, planes, etc.)
  s = reduce(loaded, { type: "openSpotlight", query: "set" });
  view = spotlightViewOf(s);
  const cmdIds = view.items.map((i) => i.id);
  assert.ok(cmdIds.includes("cmd:set-opacity"));
  assert.ok(cmdIds.includes("cmd:set-colour-by"));
  assert.ok(cmdIds.includes("cmd:toggle-edges"));
  assert.ok(cmdIds.includes("cmd:planes"));

  // 5. DOM workflow: Tab at activeIndex=0 to pin chips with 'or', then Shift+Enter -> "hide" -> Enter to hide
  const prevDoc = (globalThis as { document?: unknown }).document;
  const prevWin = (globalThis as { window?: unknown }).window;
  (globalThis as { document?: unknown }).document = fakeDoc;
  (globalThis as { window?: unknown }).window = fakeWin;
  try {
    const { mountSpotlight } = await import("../src/panels/spotlight.ts");
    const store = new Store();
    store.dispatch({ type: "fileLoaded", artifact: "model", load: shoeLoad });
    const unSpot = mountSpotlight(store);
    try {
      const spotRoot = fakeDoc.getElementById("spotlight")!;
      store.dispatch({ type: "openSpotlight" });
      const input = spotRoot.all((e) => e.has("spotlight-input"))[0]!;

      // Type "LeftColumn" and press Tab right away at activeIndex = 0 (batch:select-matched)
      input.value = "LeftColumn";
      input.dispatch("input");
      fakeWin.dispatch("keydown", { key: "Tab", preventDefault: () => {} });
      assert.equal(store.get().spotlight.clauses.length, 1);
      assert.equal(store.get().spotlight.query, "");

      // Type "or RightColumn" and press Tab at activeIndex = 0 -> pins with join: "or"
      input.value = "or RightColumn";
      input.dispatch("input");
      fakeWin.dispatch("keydown", { key: "Tab", preventDefault: () => {} });
      assert.equal(store.get().spotlight.clauses.length, 2);
      assert.equal(store.get().spotlight.clauses[1]!.join, "or");
      assert.equal(store.get().spotlight.clauses[1]!.value, "RightColumn");

      // Press Shift+Enter to enter selectionActions on the 44 matched column elements
      fakeWin.dispatch("keydown", {
        key: "Enter",
        shiftKey: true,
        ctrlKey: false,
        metaKey: false,
        altKey: false,
        preventDefault: () => {},
      });
      assert.equal(store.get().spotlight.mode, "selectionActions");

      // Type "hide" and press Enter -> hides the 44 column elements and closes Spotlight
      input.value = "hide";
      input.dispatch("input");
      fakeWin.dispatch("keydown", {
        key: "Enter",
        shiftKey: false,
        ctrlKey: false,
        metaKey: false,
        altKey: false,
        preventDefault: () => {},
      });
      assert.equal(store.get().spotlight.open, false);
      assert.equal(store.get().spotlight.clauses.length, 0, "closeSpotlight resets clauses to []");
      assert.equal(hiddenElementsOf(store.get()).size, 44, "the 44 column elements are hidden");
    } finally {
      unSpot();
    }
  } finally {
    (globalThis as { document?: unknown }).document = prevDoc;
    (globalThis as { window?: unknown }).window = prevWin;
  }
});

test("edge cases: quoted field values, param operator preservation, pinned-clause 'search' handoff, and no scattered-subsequence command noise", () => {
  // 1. Quoted field values (`group:"LeftColumn"`, `name="Arch"`) unquote RHS cleanly
  const quotedGroup = runSearchTerms(loaded, parseSearchText('group:"LeftColumn"'));
  assert.equal(quotedGroup.elementCount, 22, 'group:"LeftColumn" unquotes RHS and matches 22 elements');
  const quotedEq = runSearchTerms(loaded, parseSearchText('group="RightColumn"'));
  assert.equal(quotedEq.elementCount, 22, 'group="RightColumn" unquotes RHS and matches 22 elements');

  // 2. Parameter operator inside value (`{ field: "param", op: ":", value: "E > 1e8" }` and negative comparison `E > -1e5`)
  const paramInlineOp = runSearchTerms(defLoaded, [{ join: "and", negate: false, field: "param", op: ":", value: "E > 1e8" }]);
  assert.ok(paramInlineOp.total >= 1, "param clause with inline '>' operator evaluates '>' instead of ':' equality");
  const paramImpossibleInline = runSearchTerms(defLoaded, [{ join: "and", negate: false, field: "param", op: ":", value: "E > 1e30" }]);
  assert.equal(paramImpossibleInline.total, 0);
  const paramNegative = runSearchTerms(defLoaded, parseSearchText("E>-1e5"));
  assert.ok(paramNegative.total >= 1, "negative numeric threshold in param comparison is parsed and evaluated");

  // 3. Batch title formats param clauses via clauseLabel (`E > 1e8`, never `param:1e8`)
  const sParam = reduce(defLoaded, { type: "openSpotlight", query: "E>1e8" });
  const vParam = spotlightViewOf(sParam);
  const batchSelect = vParam.items.find((i) => i.id === "batch:select-matched");
  assert.ok(batchSelect, "batch:select-matched is present for E>1e8");
  assert.match(batchSelect.title, /E > 1e8/, "batch title includes parameter key and operator");

  // 4. Typing "search" after pinning filter chips still ranks cmd:open-search-window at #1 and transfers chips
  let sPinned = reduce(loaded, { type: "openSpotlight" });
  sPinned = reduce(sPinned, {
    type: "addSpotlightClause",
    clause: { join: "and", negate: false, field: "group", op: ":", value: "LeftColumn" },
  });
  sPinned = reduce(sPinned, { type: "setSpotlightQuery", query: "search" });
  const vPinned = spotlightViewOf(sPinned);
  assert.equal(vPinned.items[0]!.id, "cmd:open-search-window", "typing 'search' with pinned clauses still ranks Open Search Window at #1");
  for (const ev of vPinned.items[0]!.events) sPinned = reduce(sPinned, ev);
  assert.ok(sPinned.windows.open.includes("search"));
  assert.equal(sPinned.search.clauses.length, 1);
  assert.equal(sPinned.search.clauses[0]!.value, "LeftColumn");

  // 5. Declaration queries like "LeftColumn" do not match unrelated commands via scattered character subsequences
  const vLeftCol = spotlightViewOf(reduce(loaded, { type: "openSpotlight", query: "LeftColumn" }));
  assert.deepEqual(
    vLeftCol.items.filter((i) => i.category === "command").map((i) => i.id),
    [],
    "no unrelated commands match 'LeftColumn'",
  );

  // 6. Natural union of disjoint groups when combined with 'and' ("select LeftColumn and RightColumn")
  const bothCols = runSearchTerms(loaded, parseSearchText("select LeftColumn and RightColumn"));
  assert.equal(bothCols.elementCount, 44, "'select LeftColumn and RightColumn' unions disjoint group selectors (22 + 22 = 44)");

  // 7. Sub-token matching across underscores ("Left_Column" matches "LeftColumn")
  const subTok = runSearchTerms(loaded, parseSearchText("Left_Column"));
  assert.equal(subTok.elementCount, 22, "'Left_Column' sub-token matches LeftColumn");

  // 8. Bare "select" lists physical groups with tabClause, and "?" / "help" opens the Help guide
  const vBareSelect = spotlightViewOf(reduce(loaded, { type: "openSpotlight", query: "select" }));
  assert.ok(vBareSelect.items.length > 0 && vBareSelect.items[0]!.tabClause !== undefined, "bare 'select' lists pinnable physical groups first");
  const vHelp = spotlightViewOf(reduce(loaded, { type: "openSpotlight", query: "?" }));
  assert.equal(vHelp.items[0]!.id, "cmd:help", "'?' ranks Help & Examples at #1");
  const sHelpMode = reduce(loaded, { type: "openSpotlight", mode: "help" });
  assert.ok(spotlightViewOf(sHelpMode).items.length >= 5, "help mode lists step-by-step interactive examples");
});
