// #2835: `has` lists a group once, not once per child, and stays a listing:
// a child that is listed but cannot be opened is refused by name (#424), which
// `group.get` alone would read as "absent".
//
// Oracles: fixtures/definitions.h5, and a copy with a dangling soft link added.

import assert from "node:assert/strict";
import { copyFileSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { openModel } from "../src/reader/node.ts";

await h5wasm.ready;
const FIXTURE = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "definitions.h5");

test("reading a model lists each group object a bounded number of times", async (t) => {
  const counts = new Map<object, number>();
  const keys = h5wasm.Group.prototype.keys;
  t.mock.method(h5wasm.Group.prototype, "keys", function (this: h5wasm.Group) {
    counts.set(this, (counts.get(this) ?? 0) + 1);
    return keys.call(this);
  });
  await openModel(FIXTURE);
  assert.ok(counts.size > 0);
  // Once for `has`, and at most once more where the reader walks the children itself.
  const most = Math.max(...counts.values());
  assert.ok(most <= 2, `a group was listed ${most} times`);
});

test("a listed child that cannot be opened is refused by name", async (t) => {
  const dir = mkdtempSync(join(tmpdir(), "viewer-group-keys-"));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const path = join(dir, "dangling.h5");
  copyFileSync(FIXTURE, path);
  const f = new h5wasm.File(path, "a");
  const pg = f.keys().includes("physical_groups") ? (f.get("physical_groups") as h5wasm.Group) : f.create_group("physical_groups");
  const side = pg.keys().includes("element_side") ? (pg.get("element_side") as h5wasm.Group) : pg.create_group("element_side");
  side.create_soft_link("/nowhere", "dangling");
  f.close();
  await assert.rejects(openModel(path), /physical_groups\/element_side\/dangling is listed but is not a readable group/);
});
