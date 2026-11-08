// The variants manifest reader (src/reader/variants.ts; ADR 0132 D8). The
// oracle is the schema's own corpus, architecture/schemas/examples/: the valid
// manifest opens with every row, each invalid one refuses the file or exactly
// the row it breaks, and the newer-minor one opens read-only with its unknown
// keys dropped. The disk is fake, so `missing` and `escapes` are named here.

import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { test } from "node:test";
import { isManifestPath, pathProblem, readVariants, rowOf, type VariantsFs } from "../../src/reader/variants.ts";

const EXAMPLES = join(import.meta.dirname, "..", "..", "..", "architecture", "schemas", "examples");
const study = resolve("/study");
const M = join(study, "s.variants.json");
const ROWS = ["slab_uncracked.h5", "slab_cracked.h5", join("bischoff", "slab_pass2.h5")].map((p) => join(study, p));

/** A fake disk holding the manifest text and the three slab models; `links` maps a path to what it resolves to. */
function disk(text: string, files: string[] = ROWS, links: Record<string, string> = {}): VariantsFs {
  return {
    readFile: (p) => {
      if (p !== M) throw new Error(`ENOENT ${p}`);
      return text;
    },
    isFile: (p) => files.includes(p),
    realpath: (p) => links[p] ?? p,
    fold: false,
  };
}

const read = (name: string, fs?: VariantsFs) => readVariants(M, fs ?? disk(readFileSync(join(EXAMPLES, name), "utf8")));

test("the valid example opens with every row, in order, each resolved under the manifest's folder", () => {
  const r = read("variants.valid.json");
  assert.ok(r.ok, JSON.stringify(r));
  assert.equal(r.newer, null);
  assert.deepEqual(r.refused, []);
  assert.deepEqual(r.rows.map((x) => [x.index, x.label, x.resolved]), [
    [0, "Uncracked", ROWS[0]],
    [1, "Cracked", ROWS[1]],
    [2, "Bischoff D+SD+0.25L", ROWS[2]],
  ]);
  assert.deepEqual(r.rows[2]!.params, { state: "cracked", bischoff: "D+SD+0.25L", pass: 2, converged: false });
});

/** Each invalid example: the file refusal code, or the one row refused and its code (the other rows open). */
const EXPECT: Record<string, { file: string } | { row: number; code: string }> = {
  "variants.param-null.json": { row: 1, code: "bad-param" },
  "variants.param-overflow.json": { row: 0, code: "bad-param" },
  "variants.path-absolute.json": { row: 0, code: "bad-path" },
  "variants.path-empty-stem-subfolder.json": { row: 2, code: "bad-path" },
  "variants.path-empty-stem.json": { row: 0, code: "bad-path" },
  "variants.path-folder-trailing-dot.json": { row: 2, code: "bad-path" },
  "variants.path-folder-trailing-space.json": { row: 2, code: "bad-path" },
  "variants.path-leaves-folder.json": { row: 2, code: "bad-path" },
  "variants.path-not-model.json": { row: 1, code: "bad-path" },
  "variants.path-user-sibling-uppercase.json": { row: 1, code: "bad-path" },
  "variants.path-workspace-sibling.json": { row: 1, code: "bad-path" },
  "variants.version-number.json": { file: "bad-version" },
  "variants.wrong-list-key.json": { file: "bad-envelope" },
};

test("every invalid example of the schema corpus is listed here, so a new one cannot pass unjudged", () => {
  assert.deepEqual(readdirSync(join(EXAMPLES, "invalid")).sort(), Object.keys(EXPECT).sort());
});

for (const [name, want] of Object.entries(EXPECT)) {
  test(`invalid example ${name} refuses ${"file" in want ? "the file" : `row ${want.row} alone`}`, () => {
    const r = read(join("invalid", name));
    if ("file" in want) {
      assert.equal(r.ok, false);
      assert.equal(!r.ok && r.code, want.file);
      return;
    }
    assert.ok(r.ok, JSON.stringify(r));
    assert.deepEqual(r.refused.map((x) => [x.index, x.code]), [[want.row, want.code]]);
    assert.deepEqual(r.rows.map((x) => x.index), [0, 1, 2].filter((i) => i !== want.row));
  });
}

test("a same-major newer minor opens with one banner, its unknown keys dropped at every depth", () => {
  const r = read("variants.newer-minor.json");
  assert.ok(r.ok, JSON.stringify(r));
  assert.match(r.newer!, /1\.1\.0, newer than this app's 1\.0\.0/);
  assert.equal(r.rows.length, 3);
  assert.deepEqual(r.refused, []);
  assert.ok(!("colour" in r.rows[0]!));
});

test("at 1.0 an unknown key refuses the file at the top and the row in a row", () => {
  const valid = JSON.parse(readFileSync(join(EXAMPLES, "variants.valid.json"), "utf8"));
  const top = readVariants(M, disk(JSON.stringify({ ...valid, shared_params: {} })));
  assert.equal(!top.ok && top.code, "bad-envelope");
  valid.variants[1].colour = "#fff";
  const row = readVariants(M, disk(JSON.stringify(valid)));
  assert.ok(row.ok);
  assert.deepEqual(row.refused.map((x) => [x.index, x.code]), [[1, "bad-param"]]);
});

test("another major, a version below the floor, another format and an empty list refuse the file", () => {
  const valid = JSON.parse(readFileSync(join(EXAMPLES, "variants.valid.json"), "utf8"));
  const code = (doc: unknown) => {
    const r = readVariants(M, disk(JSON.stringify(doc)));
    return r.ok ? "opened" : r.code;
  };
  assert.equal(code({ ...valid, version: "2.0.0" }), "major");
  assert.equal(code({ ...valid, version: "0.9.0" }), "major");
  assert.equal(code({ ...valid, version: "1.0" }), "bad-version");
  assert.equal(code({ ...valid, format: "apegmsh-variant" }), "not-a-manifest");
  assert.equal(code({ ...valid, variants: [] }), "bad-envelope");
  assert.equal(code([valid]), "not-a-manifest");
  assert.equal(readVariants(M, disk("{ not json")).ok, false);
});

test("a byte-order mark is stripped; NaN is not JSON", () => {
  const text = readFileSync(join(EXAMPLES, "variants.valid.json"), "utf8");
  assert.ok(readVariants(M, disk(`﻿${text}`)).ok);
  const nan = readVariants(M, disk(text.replace("1.0 }", "NaN }")));
  assert.equal(!nan.ok && nan.code, "bad-json");
});

test("missing, escapes (a link out of the folder), duplicate labels and duplicate paths refuse their rows", () => {
  const valid = JSON.parse(readFileSync(join(EXAMPLES, "variants.valid.json"), "utf8"));
  const gone = readVariants(M, disk(JSON.stringify(valid), ROWS.slice(0, 2)));
  assert.ok(gone.ok);
  assert.deepEqual(gone.refused.map((x) => [x.index, x.code]), [[2, "missing"]]);
  // bischoff/ is a junction to a folder outside the study: the joined path is inside, the resolved one is not.
  const out = readVariants(M, disk(JSON.stringify(valid), ROWS, { [ROWS[2]!]: resolve("/elsewhere/slab_pass2.h5") }));
  assert.ok(out.ok);
  assert.deepEqual(out.refused.map((x) => [x.index, x.code]), [[2, "escapes"]]);
  const dup = structuredClone(valid);
  dup.variants[2].label = "Uncracked";
  const d = readVariants(M, disk(JSON.stringify(dup)));
  assert.ok(d.ok);
  assert.deepEqual(d.refused.map((x) => [x.index, x.code]), [[0, "duplicate"], [2, "duplicate"]]);
  assert.deepEqual(d.rows.map((x) => x.index), [1]);
  // Two spellings of one file (a link) are one path.
  const twin = readVariants(M, disk(JSON.stringify(valid), ROWS, { [ROWS[2]!]: ROWS[0]! }));
  assert.ok(twin.ok);
  assert.deepEqual(twin.refused.map((x) => [x.index, x.code]), [[0, "duplicate-path"], [2, "duplicate-path"]]);
});

test("a label or a params key that breaks the name rule refuses its row as empty-name", () => {
  const valid = JSON.parse(readFileSync(join(EXAMPLES, "variants.valid.json"), "utf8"));
  const bad = structuredClone(valid);
  bad.variants[0].label = " Uncracked";
  bad.variants[1].params = { "": 1 };
  bad.variants[2].label = "Café"; // not NFC
  const r = readVariants(M, disk(JSON.stringify(bad)));
  assert.ok(r.ok);
  assert.deepEqual(r.refused.map((x) => [x.index, x.code]), [[0, "empty-name"], [1, "empty-name"], [2, "empty-name"]]);
});

test("the path form: the schema's rules, case-folded suffixes, .ladruno and part files refused", () => {
  for (const ok of ["a.h5", "sub/dir/a.H5", "a b.h5", "a.results.h5x.h5"]) assert.equal(pathProblem(ok), null, ok);
  for (const no of ["", "/a.h5", "a\\b.h5", "C:a.h5", "./a.h5", "a//b.h5", "a/../b.h5", "a./b.h5", "a /b.h5", ".h5", "x/.h5", "a.Geometry.h5", "a.RESULTS.h5", "a.user.H5", "a.workspace.h5", "a.ladruno", "a.part-0.ladruno", "a.json"]) {
    assert.notEqual(pathProblem(no), null, no);
  }
});

test("rowOf: the page names a row by index and label; a mismatch, a refused row or a path is refused", () => {
  const valid = JSON.parse(readFileSync(join(EXAMPLES, "variants.valid.json"), "utf8"));
  valid.variants[1].params = { x: null };
  const r = readVariants(M, disk(JSON.stringify(valid)));
  assert.ok(rowOf(r, 0, "Uncracked").ok);
  assert.match((rowOf(r, 0, "Cracked") as { reason: string }).reason, /is "Uncracked", not "Cracked"/);
  assert.match((rowOf(r, 1, "Cracked") as { reason: string }).reason, /refused \(bad-param/);
  assert.match((rowOf(r, 7, "x") as { reason: string }).reason, /no variant row 7/);
  assert.match((rowOf(r, ROWS[0], "Uncracked") as { reason: string }).reason, /named by its row index and label/);
  assert.match((rowOf(null, 0, "Uncracked") as { reason: string }).reason, /no variants manifest is open/);
});

test("isManifestPath: <study>.variants.json, any case, with a non-empty study", () => {
  assert.ok(isManifestPath(join(study, "s.variants.json")));
  assert.ok(isManifestPath(join(study, "S.VARIANTS.JSON")));
  assert.ok(!isManifestPath(join(study, ".variants.json")));
  assert.ok(!isManifestPath(join(study, "s.json")));
});
