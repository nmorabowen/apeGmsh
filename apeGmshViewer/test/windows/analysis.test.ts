// V3c1 (#1642): the analysis window.
//
// Oracles: fixtures/definitions.h5 (make_zone_fixtures.py `definitions`: two
// stage chains, gravity = Plain / RCM / UmfPack / NormDispIncr / Newton /
// LoadControl / Static with 10 steps, push = the same with Newmark /
// Transient, 5 steps, dt 0.01; no chain declaration in /opensees/decls) and
// fixtures/shoebuckle.h5 (opensees 2.22: a global /opensees/analysis, no
// decls). The chain declarations a 2.26 bridge writes (one per family per
// chain, in chain order: kr3pb's and staged_compose's shape) are added to the
// loaded state in memory, with literal parameters by field name; every
// parameter asserted is one of those literals, never a value read back from
// the app, and never the positional `*_args`. The view runs on a minimal
// DOM: a click dispatches `requestSource` through `act` and nothing else.

import assert from "node:assert/strict";
import { copyFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { analysisOf, CHAIN, view } from "../../src/panels/windows/analysis.ts";
import { WINDOWS } from "../../src/panels/windows/registry.ts";
import { openModel } from "../../src/reader/node.ts";
import { BlobStore } from "../../src/state/blobs.ts";
import { loadModel } from "../../src/state/load.ts";
import { initialState, reduce } from "../../src/state/reduce.ts";
import type { DeclRecord, DeclValue, Event, State } from "../../src/state/types.ts";

// ---- a minimal DOM ----

class FakeText {
  textContent: string;
  constructor(text: string) {
    this.textContent = text;
  }
}
class FakeEl {
  children: (FakeEl | FakeText)[] = [];
  attrs: Record<string, string> = {};
  dataset: Record<string, string> = {};
  className = "";
  title = "";
  tag: string;
  private listeners: ((e: { stopPropagation(): void }) => void)[] = [];
  constructor(tag: string) {
    this.tag = tag;
  }
  append(...xs: (FakeEl | FakeText | string)[]): void {
    for (const x of xs) this.children.push(typeof x === "string" ? new FakeText(x) : x);
  }
  get textContent(): string {
    return this.children.map((c) => c.textContent).join("");
  }
  set textContent(v: string) {
    this.children = [new FakeText(v)];
  }
  setAttribute(k: string, v: string): void {
    this.attrs[k] = v;
  }
  addEventListener(type: string, fn: (e: { stopPropagation(): void }) => void): void {
    assert.equal(type, "click", "the window listens for clicks only");
    this.listeners.push(fn);
  }
  get clickable(): boolean {
    return this.listeners.length > 0;
  }
  click(): void {
    for (const f of this.listeners) f({ stopPropagation() {} });
  }
  all(pred: (e: FakeEl) => boolean): FakeEl[] {
    const out: FakeEl[] = pred(this) ? [this] : [];
    for (const c of this.children) if (c instanceof FakeEl) out.push(...c.all(pred));
    return out;
  }
}
(globalThis as { document?: unknown }).document = { createElement: (tag: string) => new FakeEl(tag) };

// ---- the fixtures ----

await h5wasm.ready;
const fixtures = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "fixtures");
const loadPath = async (p: string): Promise<State> =>
  reduce(initialState, { type: "fileLoaded", artifact: "model", load: loadModel(await openModel(p), new BlobStore()) });
const defs = await loadPath(join(fixtures, "definitions.h5"));
const plain = await loadPath(join(fixtures, "shoebuckle.h5"));

/** The chain declarations of a 2.26 bridge: one per family per chain, in chain order, each with its decl_params row. */
const CHAIN_PARAMS: [string, string, Record<string, DeclValue>][][] = [
  [
    ["system", "UmfPack", {}], ["numberer", "RCM", {}], ["constraints", "Plain", {}],
    ["integrator", "LoadControl", { dlam: 0.1, num_iter: null, min_lam: null, max_lam: null }],
    ["algorithm", "Newton", { tangent: "tangent" }],
    ["test", "NormDispIncr", { tol: 1e-6, max_iter: 10, print_flag: 0, norm_type: 2 }],
    ["analysis", "Static", {}],
  ],
  [
    ["system", "UmfPack", {}], ["numberer", "RCM", {}], ["constraints", "Plain", {}],
    ["integrator", "Newmark", { gamma: 0.5, beta: 0.25 }],
    ["algorithm", "Newton", { tangent: "tangent" }],
    ["test", "NormDispIncr", { tol: 1e-6, max_iter: 10, print_flag: 0, norm_type: 2 }],
    ["analysis", "Transient", {}],
  ],
];

function withChainDecls(s: State, chains = CHAIN_PARAMS): State {
  const d = s.definitions!;
  const base = d.decls!.records;
  const added: DeclRecord[] = chains.flatMap((chain, k) =>
    chain.map(([family, type, params], j): DeclRecord => ({
      row: base.length + k * chain.length + j,
      key: `opensees/${family}/#${k + 1}`,
      family,
      name: "",
      synth: false,
      params: [{ row: base.length + k * chain.length + j, type, params, paramsNames: null, refs: { transf: [], integration: [], section: [] } }],
    })),
  );
  return { ...s, definitions: { ...d, decls: { ...d.decls!, records: [...base, ...added] } } };
}
const joined = withChainDecls(defs);

const render = (s: State): { root: FakeEl; events: Event[] } => {
  const events: Event[] = [];
  const root = view(s, (e) => events.push(e)) as unknown as FakeEl;
  return { root, events };
};
const partsText = (s: State, i: number) => analysisOf(s).chains[i]!.parts.map((p) => `${p.role}: ${p.token}{${p.params.map((l) => `${l.field}=${l.text}`).join(", ")}}`);

test("the registry's analysis entry is this window's view", () => {
  assert.equal(WINDOWS.analysis.view, view);
});

test("each stage's chain in the card's order, every member by its parameters' field names from decl_params", () => {
  const m = analysisOf(joined);
  assert.equal(m.note, null);
  assert.deepEqual(m.chains.map((c) => [c.title, c.h5]), [
    ["stage 0 · gravity", "/opensees/stages/stage_000/analysis"],
    ["stage 1 · push", "/opensees/stages/stage_001/analysis"],
  ]);
  assert.deepEqual(CHAIN.map((c) => c.role), ["system", "numberer", "constraints", "integrator", "algorithm", "test", "analysis"]);
  assert.deepEqual(partsText(joined, 0), [
    "system: UmfPack{}", "numberer: RCM{}", "constraints: Plain{}",
    "integrator: LoadControl{dlam=0.1, num_iter=null, min_lam=null, max_lam=null}",
    "algorithm: Newton{tangent=tangent}",
    "test: NormDispIncr{tol=0.000001, max_iter=10, print_flag=0, norm_type=2}",
    "analysis: Static{}",
  ]);
  assert.deepEqual(partsText(joined, 1)[3], "integrator: Newmark{gamma=0.5, beta=0.25}");
  // Unnamed declarations show their #k key; the k-th chain joins the k-th declaration of each family.
  assert.deepEqual(analysisOf(joined).chains.map((c) => c.parts[3]!.decl?.label), ["#1", "#2"]);
  assert.ok(analysisOf(joined).chains.every((c) => c.parts.every((p) => p.note === null)));
});

test("the parameters never come from the positional *_args", () => {
  // The fixture's integrator_args for gravity is [0.1] and test_args [1e-6, 10, 0, 2]: with no decl_params, no value is shown.
  const m = analysisOf(defs);
  for (const c of m.chains) for (const p of c.parts) assert.deepEqual(p.params, [], `${c.title} ${p.role}`);
});

test("steps and dt: from the stage, static says no dt; without /sequence the stage's own attributes are used", () => {
  assert.deepEqual(analysisOf(joined).chains.map((c) => [c.steps, c.dt]), [
    ["10 steps", "static (no dt)"],
    ["5 steps", "dt 0.01"],
  ]);
});

test("steps = -1 in /sequence (a deck that loops in user Tcl) shows no steps and says why", async () => {
  const dir = mkdtempSync(join(tmpdir(), "agv-analysis-"));
  const p = join(dir, "deck.seq.h5");
  copyFileSync(join(fixtures, "shoebuckle.h5"), p);
  const f = new h5wasm.File(p, "a");
  /* eslint-disable @typescript-eslint/no-explicit-any */
  (f.get("meta") as any).create_attribute("sequence_schema_version", "1.0.0");
  const g = f.create_group("sequence") as any;
  g.create_attribute("staged", 0);
  const st = g.create_group("stages");
  st.create_dataset({ name: "source", data: ["/opensees/analysis"] });
  st.create_dataset({ name: "steps", data: new Int32Array([-1]) });
  st.create_dataset({ name: "dt", data: new Float64Array([Number.NaN]) });
  st.create_dataset({ name: "analysis", data: ["Static"] });
  const ch = g.create_group("changes");
  ch.create_dataset({ name: "source", data: ["/opensees/patterns/Plain_1"] });
  f.close();
  const c = analysisOf(await loadPath(p)).chains[0]!;
  assert.equal(c.title, "implicit stage (unstaged deck)");
  assert.equal(c.steps, "no analyze_steps archived (steps = -1): the deck runs its own analysis loop, so no step count is known");
  assert.equal(c.dt, "static (no dt)");
});

test("the join is refused, loudly, when the counts differ or a declaration's type is not the chain's token", () => {
  // definitions.h5 has no chain declarations: 0 for 2 chains.
  const none = analysisOf(defs).chains[0]!.parts[3]!;
  assert.equal(none.token, "LoadControl");
  assert.equal(none.decl, null);
  assert.equal(none.note, "cannot join: /opensees/decls has 0 integrator declarations for 2 chains that set it (the archive records no chain↔declaration link, #1646)");
  // One chain's worth of declarations for two chains.
  const short = analysisOf(withChainDecls(defs, [CHAIN_PARAMS[0]!])).chains[1]!.parts[5]!;
  assert.match(short.note!, /has 1 test declaration for 2 chains/);
  // The second integrator declared as LoadControl while the push chain names Newmark.
  const swapped = CHAIN_PARAMS.map((c, k) => c.map((x) => (k === 1 && x[0] === "integrator" ? (["integrator", "LoadControl", { dlam: 1 }] as (typeof x)) : x)));
  const bad = analysisOf(withChainDecls(defs, swapped)).chains.map((c) => c.parts[3]!);
  assert.deepEqual(bad.map((p) => [p.decl, p.params]), [[null, []], [null, []]], "a refused join lists no parameter for any chain");
  assert.equal(bad[1]!.note, "cannot join: opensees/integrator/#2 is a LoadControl, the chain names Newmark (the archive records no chain↔declaration link, #1646)");
  // The other families still join.
  assert.equal(analysisOf(withChainDecls(defs, swapped)).chains[1]!.parts[5]!.note, null);
});

test("a 2.22 file (no /opensees/decls) shows its tokens and says the parameters are not in the file", () => {
  const m = analysisOf(plain);
  assert.equal(m.chains.length, 1);
  assert.equal(m.chains[0]!.title, "implicit stage (unstaged deck)");
  const p = m.chains[0]!.parts.find((x) => x.role === "integrator")!;
  assert.ok(p.token);
  assert.deepEqual(p.params, []);
  assert.equal(p.note, "parameters by field name: not in this file (no /opensees/decls, opensees < 2.25)");
});

test("empty states are explicit", () => {
  assert.equal(analysisOf(initialState).note, "no model is open");
  assert.equal(analysisOf({ ...defs, definitions: null }).note, "the model has no /opensees zone");
  const bare: State = { ...defs, definitions: { ...defs.definitions!, analysis: null, stages: [] } };
  assert.equal(analysisOf(bare).note, "the model declares no analysis");
  assert.equal(render(bare).root.textContent, "the model declares no analysis");
  // A member the chain does not set says so instead of guessing.
  const st = defs.definitions!.stages[0]!;
  const noTest: State = { ...joined, definitions: { ...joined.definitions!, stages: [{ ...st, analysis: { ...st.analysis!, attrs: { ...st.analysis!.attrs, test: undefined as never } } }] } };
  const t = analysisOf(noTest).chains[0]!.parts.find((x) => x.role === "test")!;
  assert.deepEqual([t.token, t.note], [null, "not set in /opensees/stages/stage_000/analysis"]);
});

test("the view: one section per stage, Token{field=value}, and a click on a declaration with a source dispatches requestSource", () => {
  const { root, events } = render(joined);
  assert.deepEqual(root.all((e) => e.className === "window-section").map((e) => e.textContent), ["stage 0 · gravity", "stage 1 · push"]);
  const integ = root.all((e) => e.dataset["role"] === "integrator");
  assert.match(integ[0]!.textContent, /^integrator LoadControl\{dlam=0\.1, num_iter=null, min_lam=null, max_lam=null\} #1/);
  assert.match(integ[1]!.textContent, /^integrator Newmark\{gamma=0\.5, beta=0\.25\} #2/);
  assert.ok(root.all((e) => e.className === "window-ref").every((r) => !r.clickable), "no source recorded: nothing is clickable");
  assert.deepEqual(events, []);

  const key = "opensees/integrator/#2";
  const s: State = {
    ...joined,
    provenance: [{ key, origin: "user", seq: 0, source: { file: "/work/model.py", line: 40, function: "<module>", sha256: "ab", recorded: true, script: null } }],
  };
  const r = render(s);
  const refs = r.root.all((e) => e.className === "window-ref" && e.clickable);
  assert.deepEqual(refs.map((x) => x.textContent), ["#2 ↗"]);
  refs[0]!.click();
  assert.deepEqual(r.events, [{ type: "requestSource", decl: key }]);
  assert.deepEqual(reduce(s, r.events[0]!).source.request, { decl: key, seq: 1 });
});

test("a {$decl} ref inside a member's parameters is a link to that declaration", () => {
  const chains = CHAIN_PARAMS.map((c) => c.map((x) => (x[0] === "algorithm" ? (["algorithm", "Newton", { tangent: "tangent", series: { $decl: "opensees/timeSeries/ramp" } }] as (typeof x)) : x)));
  const p = analysisOf(withChainDecls(defs, chains)).chains[0]!.parts.find((x) => x.role === "algorithm")!;
  assert.deepEqual(p.params.map((l) => [l.field, l.text, l.ref?.label ?? null]), [["tangent", "tangent", null], ["series", "opensees/timeSeries/ramp", "ramp"]]);
});
