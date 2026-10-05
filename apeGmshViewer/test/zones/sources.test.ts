// V2g (#1425): the sources listing. Every /provenance record of the model is
// listed, the objects apeGmsh synthesised inside a verb the user called
// included by default and marked; go-to-source on one opens the user's call
// of that verb.
//
// Oracles: fixtures/bridge_provenance.h5, written by main's own writer from
// the script in fixtures/README.md (provenance.test.ts checks the digest and
// that its synthesised records sit on the `s.support(` line, read from the
// README, not from the app); fixtures/zones.h5, a 1.0.0 file whose records all
// read as `user` (h5-schema.md, "/provenance").

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { Effects, type Bridge } from "../../src/effects.ts";
import type { ModelFile } from "../../src/model/types.ts";
import { openModel } from "../../src/reader/node.ts";
import { readProvenance, type ProvenanceZone } from "../../src/reader/provenance.ts";
import type { H5Module } from "../../src/reader/read.ts";
import { BlobStore } from "../../src/state/blobs.ts";
import { loadModel } from "../../src/state/load.ts";
import { initialState, reduce } from "../../src/state/reduce.ts";
import { pathSegments, sourceFor, sourcesHeaderOf, sourcesOf, warningsOf } from "../../src/state/selectors.ts";
import { Store } from "../../src/state/store.ts";

await h5wasm.ready;
const h5 = h5wasm as unknown as H5Module;
const fixtures = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "fixtures");
const BRIDGE = join(fixtures, "bridge_provenance.h5");
const ZONES = join(fixtures, "zones.h5");

const bridgeModel: ModelFile = await openModel(BRIDGE);
const bridgeZone: ProvenanceZone = readProvenance(h5, BRIDGE)!;
const zonesModel: ModelFile = await openModel(ZONES);
const zonesZone: ProvenanceZone = readProvenance(h5, ZONES)!;

// The `s.support(` line of the generator script, read from the README.
const readme = readFileSync(join(fixtures, "README.md"), "utf8").replace(/\r\n/g, "\n");
const script = readme.match(/## `bridge_provenance\.h5`[\s\S]*?```python\n([\s\S]*?)```/)![1]!.split("\n");
const SUPPORT_LINE = script.findIndex((l) => /^\s+s\.support\(/.test(l)) + 1;
const SCRIPT = `${bridgeZone.baseDir}/bridge_model.py`;
const HOLD = "opensees/timeSeries/support:gravity/hold";
const PATTERN = "opensees/pattern/support:gravity";

const loaded = (m: ModelFile, p: ProvenanceZone) =>
  reduce(initialState, { type: "fileLoaded", artifact: "model", load: loadModel(m, new BlobStore(), p) });

test("the listing shows the synthesised objects by default, marked, with their verb", () => {
  assert.ok(SUPPORT_LINE > 0);
  const rows = sourcesOf(loaded(bridgeModel, bridgeZone));
  assert.equal(rows.length, bridgeZone.records.path.length, "every record is listed; none is filtered out");
  const synth = rows.filter((r) => r.origin === "synthesised");
  assert.deepEqual(synth.map((r) => r.key), [HOLD, PATTERN]);
  for (const r of synth) {
    assert.equal(r.verb, "support");
    assert.equal(r.label, `bridge_model.py:${SUPPORT_LINE}`);
    assert.equal(r.off, null);
  }
  // The user's own declarations are listed as such, in capture order.
  const user = rows.filter((r) => r.origin === "user");
  assert.ok(user.some((r) => r.key === "opensees/nDMaterial/concrete" && r.verb === null));
  const seq = rows.map((r) => r.key).map((k) => bridgeZone.records.seq[bridgeZone.records.path.indexOf(k)]!);
  assert.deepEqual(seq, [...seq].sort((a, b) => a - b));
});

test("a /provenance 1.1.0 file raises no provenance banner", () => {
  const s = loaded(bridgeModel, bridgeZone);
  assert.deepEqual(s.artifacts.model!.zones["provenance"], { status: "ready", version: "1.1.0" });
  assert.deepEqual(warningsOf(s).filter((w) => /provenance/.test(w)), []);
});

test("a 1.0.0 file lists every record as the user's", () => {
  const rows = sourcesOf(loaded(zonesModel, zonesZone));
  assert.equal(rows.length, 7);
  assert.deepEqual([...new Set(rows.map((r) => r.origin))], ["user"]);
  assert.ok(rows.every((r) => r.verb === null));
});

test("sourceFor: a synthesised object goes to the user's verb call", () => {
  const s = loaded(bridgeModel, bridgeZone);
  for (const key of [HOLD, PATTERN]) {
    const w = sourceFor(s, key);
    assert.ok(w.ok, JSON.stringify(w));
    assert.equal(w.source.file, SCRIPT);
    assert.equal(w.source.line, SUPPORT_LINE);
    assert.equal(w.source.function, "<module>");
  }
});

test("effects: go-to-source on a synthesised object opens the s.support( line", async () => {
  const goto: [string, number][] = [];
  const bridge = {
    openModel: async () => ({ ok: true, model: bridgeModel, provenance: { ok: true, zone: bridgeZone } }),
    goToSource: async (file: string, line: number) => {
      goto.push([file, line]);
      return { ok: true };
    },
  } as unknown as Bridge;
  const store = new Store();
  const effects = new Effects(store, new BlobStore(), bridge);
  assert.equal(await effects.open(BRIDGE), true);
  store.dispatch({ type: "requestSource", decl: HOLD });
  await new Promise((r) => setTimeout(r, 10));
  assert.deepEqual(goto, [[SCRIPT, SUPPORT_LINE]]);
  assert.deepEqual(store.get().source.last, { decl: HOLD, ok: true, reason: null });
  // A key that is neither a declaration nor a record still fails loud.
  assert.throws(() => store.dispatch({ type: "requestSource", decl: "opensees/pattern/support:nope" }), /not a declaration or a \/provenance record/);
  effects.dispose();
});

// ---- the design brief's acceptance criteria (#1426) -------------------------
// AC1/AC2 (no overlap, the viewport keeps its width) and AC4's colours are
// layout facts: checked here on the stylesheet and the page, and in the still.

const renderer = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "src", "renderer");
const css = readFileSync(join(renderer, "style.css"), "utf8");
const html = readFileSync(join(renderer, "index.html"), "utf8");
/** The declarations of the first rule whose selector list is exactly `sel`. */
const rule = (sel: string): string => {
  const m = new RegExp(`(?:^|\\})\\s*${sel.replace(/[.#()[\]+*>]/g, "\\$&")}\\s*\\{([^}]*)\\}`, "m").exec(css);
  assert.ok(m, `style.css has no rule ${sel}`);
  return m[1]!;
};

test("AC1/AC2: the inspector and Sources share one rail, stacked, and the rail leaves the viewport >= 640 px of 1280", () => {
  assert.match(html, /<div id="rail">\s*<aside id="inspector" hidden><\/aside>\s*<aside id="sources" hidden><\/aside>\s*<\/div>/);
  const railCss = rule("#rail");
  assert.match(railCss, /position: fixed/);
  assert.match(railCss, /flex-direction: column/);
  const width = Number(/width: (\d+)px/.exec(railCss)![1]);
  const right = Number(/right: (\d+)px/.exec(railCss)![1]);
  assert.ok(1280 - width - right >= 640, `the rail covers ${width + right} px of 1280`);
  // Neither panel positions itself: the rail's flex column stacks them.
  assert.doesNotMatch(rule("#inspector"), /position:/);
  assert.doesNotMatch(rule("#sources"), /position:/);
  assert.match(rule("#inspector:not([hidden]) + #sources"), /max-height: 40%/);
});

test("AC3: rows in ascending seq; the header counts the records and the synthesised ones", () => {
  const s = loaded(bridgeModel, bridgeZone);
  const rows = sourcesOf(s);
  const seqs = rows.map((r) => s.provenance.find((p) => p.key === r.key)!.seq);
  assert.deepEqual(seqs, [...seqs].sort((a, b) => a - b));
  const header = sourcesHeaderOf(s)!;
  assert.equal(header.count, bridgeZone.records.path.length);
  assert.equal(header.synthesised, bridgeZone.records.origin.filter((o) => o === "synthesised").length);
  assert.equal(header.synthesised, 2);
  assert.equal(header.notice, null);
});

test("AC4: the synth marker is a dashed muted outline, never the warn or accent colour", () => {
  const synth = rule(".synth");
  assert.match(synth, /border: 1px dashed var\(--muted\)/);
  assert.match(synth, /color: var\(--muted\)/);
  assert.doesNotMatch(synth, /--interp|--accent/);
  assert.match(readFileSync(join(renderer, "..", "panels", "sources.ts"), "utf8"), /el\("span", "synth", "synth"\)/);
});

test("AC5: a #k collision row has no jump and states its reason", () => {
  const s0 = loaded(zonesModel, zonesZone);
  // An unnamed declaration of the app whose path is also a record key: the
  // two #k numberings differ, so the row must not jump.
  const clash = Object.keys(s0.decls).find((p) => /^opensees\/geomTransf\/#/.test(p))!;
  assert.ok(clash);
  const s = { ...s0, provenance: [...s0.provenance, { key: clash, origin: "user" as const, seq: 99, source: { file: "/x/m.py", line: 3, function: "f", sha256: "", script: null } }] };
  const row = sourcesOf(s).find((r) => r.key === clash)!;
  assert.equal(row.label, null);
  assert.match(row.off!, /unnamed declaration is not joined/);
  assert.equal(row.title, row.off);
  // The panel shows the reason on its own line and wires no click on a disabled row.
  const panel = readFileSync(join(renderer, "..", "panels", "sources.ts"), "utf8");
  assert.match(panel, /if \(r\.off === null\) \{[\s\S]*?listen\(go, "click"[\s\S]*?\} else go\.disabled = true;/);
  assert.match(panel, /el\("div", "f-src source-off", r\.off\)/);
});

test("AC6: a key wraps only after a /", () => {
  for (const key of [HOLD, PATTERN, "geometry/box/block", "opensees/element/#1", "a", "a/b:c/d"]) {
    const segs = pathSegments(key);
    assert.equal(segs.join(""), key);
    segs.slice(0, -1).forEach((g) => assert.match(g, /^[^/]*\/$/, `${key}: ${g}`));
    assert.doesNotMatch(segs.at(-1)!.slice(0, -1), /\//);
  }
  assert.deepEqual(pathSegments(HOLD), ["opensees/", "timeSeries/", "support:gravity/", "hold"]);
  const keyCss = rule(".source-key");
  assert.match(keyCss, /overflow-wrap: normal/);
  assert.match(keyCss, /word-break: keep-all/);
});

test("AC7: no /provenance hides the panel; a 1.0.x file shows the notice and no marker", () => {
  assert.equal(sourcesHeaderOf(initialState), null);
  const plain = reduce(initialState, { type: "fileLoaded", artifact: "model", load: loadModel(bridgeModel, new BlobStore(), null) });
  assert.equal(sourcesHeaderOf(plain), null);
  const old = loaded(zonesModel, zonesZone);
  const header = sourcesHeaderOf(old)!;
  assert.equal(header.notice, "provenance 1.0: synthesised records are not marked");
  assert.equal(header.synthesised, 0);
  assert.equal(sourcesOf(old).filter((r) => r.origin === "synthesised").length, 0);
});

test("a model re-read replaces the listing; a failed read clears it", () => {
  const s = loaded(bridgeModel, bridgeZone);
  assert.ok(s.provenance.length > 0);
  const failed = reduce(s, { type: "fileFailed", artifact: "model", path: bridgeModel.path, error: "gone" });
  assert.deepEqual(sourcesOf(failed), []);
  const plain = reduce(s, { type: "fileLoaded", artifact: "model", load: loadModel(bridgeModel, new BlobStore(), null) });
  assert.deepEqual(sourcesOf(plain), []);
});
