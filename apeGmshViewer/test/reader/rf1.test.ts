// reader/rf1.ts against the cross-language goldens of ADR 0130 A6.
//
// The oracle is tests/results/user/test_user_fingerprint.py: its goldens were
// computed by VU0's Python prototype and its TypeScript twin, independently of
// src/apeGmsh/results/user/_fingerprint.py, and that module equals them. This
// file repeats each golden and checks it is still the Python test's, so a
// change on either side fails here. The fixtures are the Python test's own
// (tests/fixtures/ladruno/ at the repository root, and the app's modes fixture).
//
// The element-only and domain-only goldens are of a file whose
// MODEL_STAGE[1]/RESULTS/ON_NODES Python deleted. h5wasm cannot delete a link,
// so the group is hidden here from `keys()`, which is all rf1 iterates: a
// deleted link and a hidden one list the same names.

import assert from "node:assert/strict";
import { copyFileSync, mkdtempSync, readdirSync, readFileSync, renameSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, type TestContext } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { byCodePoint, FingerprintRefused, pyFloat, resultsFingerprint, type FpH5 } from "../../src/reader/rf1.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as FpH5;
const repo = (rel: string) => fileURLToPath(new URL(`../../../${rel}`, import.meta.url));
const LADRUNO = (name: string) => repo(`tests/fixtures/ladruno/${name}`);
const MODES = repo("apeGmshViewer/fixtures/ladruno_modes.ladruno");
const PY_TEST = readFileSync(repo("tests/results/user/test_user_fingerprint.py"), "utf8");
const list = (dir: string) => readdirSync(dir);

const GOLDEN: Record<string, string> = {
  beam3d: "48ab1eaaa17047157458d3afb319f82940c10b1c8d377d988e5ff48d343748ac",
  bezier_tet10: "97b1afb18534d0462a7674b8c5133dae2830cd8bae0d7b9f99bab292bad6b3fe",
  bezier_tri6: "844792face2e8e8640b900127a288d446613be00ed8c0a0f4f5fb95409ee24de",
  energy: "b893d0b39fed406013e9b06fd1390f668c5adb50f0eca0f0b8071a14f2f976f3",
  fiberbeam: "a991d7666d23d8a9e1d7b8d954262058ccf019f3bf93ba1ea1bdb801d946a09a",
  node_envelope: "b897a1182c720f8bbfda96e77fdaa42d6602111173e8f25d2bca5cdf7376addb",
  quad2d: "2d3998a3fd7273399536406d9fcfcc5b88cc0af9249ba4b6070e5806ddd22bc6",
  truss2d: "76a404478cbb9ef919c2d79a676fbe1a0a300712819b344edd9a43e7289cfcc3",
};
const GOLDEN_PART_SET = "d62d58956af134bd46e7bd4e236f03808c777b99d33eff93ecec750d35621ce5";
const GOLDEN_MODES = "eecf2a003117de0df4521ba47d9b5e38142d76f1a597d10729057a86e65435fa";
const GOLDEN_ELEMENT_ONLY = "b3ef4e3f7e5d786e087af7346a786ab847f1445e9e412dbb04c4c3900067ca3c";
const GOLDEN_DOMAIN_ONLY = "af72edfe5cc9e388ee6e9f2f57901421d0b4ba6c4a051246df7cda3dc11e8dcf";

test("every golden here is the Python test's", () => {
  for (const hex of [...Object.values(GOLDEN), GOLDEN_PART_SET, GOLDEN_MODES, GOLDEN_ELEMENT_ONLY, GOLDEN_DOMAIN_ONLY]) {
    assert.ok(PY_TEST.includes(`"${hex}"`), `${hex} is not in tests/results/user/test_user_fingerprint.py: the goldens moved`);
  }
  for (const stem of Object.keys(GOLDEN)) assert.ok(PY_TEST.includes(`"${stem}": "${GOLDEN[stem]}"`), stem);
});

for (const [stem, hex] of Object.entries(GOLDEN)) {
  test(`rf1 of ${stem}.ladruno equals Python's`, () => {
    assert.equal(resultsFingerprint(H5, LADRUNO(`${stem}.ladruno`), list), `rf1:${hex}`);
  });
}

test("rf1 of a modes file hashes each LAMBDA and equals Python's", () => {
  assert.equal(resultsFingerprint(H5, MODES, list), `rf1:${GOLDEN_MODES}`);
});

for (const part of ["truss2d.part-0.ladruno", "truss2d.part-1.ladruno"]) {
  test(`rf1 of ${part} is the rf1 of its whole set`, () => {
    assert.equal(resultsFingerprint(H5, LADRUNO(part), list), `rf1:${GOLDEN_PART_SET}`);
  });
}

/** Hide `name` under the group at `parent` from every `keys()` call for the test's length. */
function hide(t: TestContext, parent: string, name: string): void {
  const proto = h5wasm.Group.prototype as unknown as { keys(this: { path: string }): string[] };
  const real = proto.keys;
  t.mock.method(proto, "keys", function (this: { path: string }) {
    const names = real.call(this);
    return this.path === parent ? names.filter((n) => n !== name) : names;
  });
}

test("a stage with no nodal result samples its first element bucket, as Python's (fallback 1)", (t) => {
  hide(t, "/MODEL_STAGE[1]/RESULTS", "ON_NODES");
  assert.equal(resultsFingerprint(H5, LADRUNO("truss2d.ladruno"), list), `rf1:${GOLDEN_ELEMENT_ONLY}`);
});

test("a stage with neither samples its first other result, as Python's (fallback 2)", (t) => {
  hide(t, "/MODEL_STAGE[1]/RESULTS", "ON_NODES");
  assert.equal(resultsFingerprint(H5, LADRUNO("energy.ladruno"), list), `rf1:${GOLDEN_DOMAIN_ONLY}`);
});

// ---- refusals: never a guessed hash ---------------------------------------

test("a missing file refuses", () => {
  assert.throws(() => resultsFingerprint(H5, join(tmpdir(), "agv-rf1-none.ladruno"), list), FingerprintRefused);
});

test("a part set with a gap refuses, naming the indices", () => {
  const dir = mkdtempSync(join(tmpdir(), "agv-rf1-"));
  copyFileSync(LADRUNO("truss2d.part-0.ladruno"), join(dir, "run.part-0.ladruno"));
  copyFileSync(LADRUNO("truss2d.part-1.ladruno"), join(dir, "run.part-1.ladruno"));
  renameSync(join(dir, "run.part-1.ladruno"), join(dir, "run.part-2.ladruno"));
  assert.throws(() => resultsFingerprint(H5, join(dir, "run.part-0.ladruno"), list), /not contiguous from 0: found indices 0, 2/);
});

test("a stage without KIND refuses", () => {
  const dir = mkdtempSync(join(tmpdir(), "agv-rf1-"));
  const p = join(dir, "run.ladruno");
  copyFileSync(LADRUNO("truss2d.ladruno"), p);
  const f = new h5wasm.File(p, "a");
  (f.get("MODEL_STAGE[1]") as h5wasm.Group).delete_attribute("KIND");
  f.close();
  assert.throws(() => resultsFingerprint(H5, p, list), /MODEL_STAGE\[1\] has no KIND/);
});

test("an HDF5 file with neither INFO nor a lineage results_hash refuses", () => {
  const dir = mkdtempSync(join(tmpdir(), "agv-rf1-"));
  const p = join(dir, "run.results.h5");
  const f = new h5wasm.File(p, "w");
  f.create_group("meta");
  f.close();
  assert.throws(() => resultsFingerprint(H5, p, list), /neither a recorder results file/);
});

test("a native results.h5 gives native:<its lineage results_hash>", () => {
  const dir = mkdtempSync(join(tmpdir(), "agv-rf1-"));
  const p = join(dir, "run.results.h5");
  const f = new h5wasm.File(p, "w");
  f.create_group("meta");
  (f.get("meta") as h5wasm.Group).create_group("lineage");
  (f.get("meta/lineage") as h5wasm.Group).create_attribute("results_hash", "abc123");
  f.close();
  assert.equal(resultsFingerprint(H5, p, list), "native:abc123");
});

// ---- the two Python behaviours this file imitates -------------------------

test("pyFloat prints what Python's repr of a float64 prints", () => {
  const cases: [number, string][] = [
    [1, "1.0"], [-2.5, "-2.5"], [0.1, "0.1"], [1e-5, "1e-05"], [0.0001, "0.0001"], [1e16, "1e+16"],
    [1e15, "1000000000000000.0"], [1.5e16, "1.5e+16"], [123.456, "123.456"], [-0, "-0.0"], [0, "0.0"],
    [NaN, "nan"], [Infinity, "inf"], [-Infinity, "-inf"], [2.5e-300, "2.5e-300"], [1.7976931348623157e308, "1.7976931348623157e+308"],
  ];
  for (const [x, want] of cases) assert.equal(pyFloat(x), want, String(x));
});

test("byCodePoint orders as Python's sorted() on str, not by UTF-16 unit", () => {
  // U+FF5E sorts before U+1F600 by code point, after it by UTF-16 unit.
  const names = ["\u{1F600}", "～", "B", "a"];
  assert.deepEqual([...names].sort(byCodePoint), ["B", "a", "～", "\u{1F600}"]);
});
